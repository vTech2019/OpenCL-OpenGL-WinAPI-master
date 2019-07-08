#include "Image_Stabilization.h"

Image_Stabilization::Image_Stabilization(cpuDevice* device, cl_uint width, cl_uint height, cl_uint block_x, cl_uint block_y, cl_uint radius)
{
	for (size_t i = 0; i < sizeof(*this); i++)
		((char*)this)[i] = 0;
	cpu_data = (C_code_data*)malloc(sizeof(C_code_data));
	float w = 4.0f * sqrt(2.0f * log(2.0f));
	*cpu_data = { device , NULL,  NULL, block_x, block_y, NULL, width, height, NULL, radius,  2.0f * (block_x / w) * (block_x / w), cl_float(block_x * block_y), ((block_x - 1) / 2),  ((block_y - 1) / 2) };

	cpu_data->offset_radius_aligned = radius + sizeof(__m256) / sizeof(float) - (radius % (sizeof(__m256) / sizeof(float)));
	cpu_data->pitch_width_image = width + sizeof(__m256) - (width % sizeof(__m256)) + cpu_data->offset_radius_aligned * 2;
	cpu_data->current_image = (cl_float4*)_mm_malloc((4 * radius + 2 * cpu_data->pitch_width_image * height) * sizeof(cl_float4), 4096);
	cpu_data->next_image = cpu_data->current_image + (2 * radius + cpu_data->pitch_width_image * height) * sizeof(cl_float4);

	ptr_stabilization_function = &Image_Stabilization::cpu_Stabilization_function;
}
Image_Stabilization::Image_Stabilization(AMP_device* device, cl_uint width, cl_uint height, cl_uint block_x, cl_uint block_y, cl_uint radius)
{
	for (size_t i = 0; i < sizeof(*this); i++)
		((char*)this)[i] = 0;
	amp_data = (AMP_data*)malloc(sizeof(AMP_data));
	amp_data->device = device;
	amp_data->block_x = block_x;
	amp_data->block_y = block_y;
	amp_data->width = width;
	amp_data->height = height;
	amp_data->radius = radius;
	ptr_stabilization_function = &Image_Stabilization::AMP_Stabilization_function;
}
//float Gaussian_filter_xy(float y, float x, float sigma) {
//	const float _y = y * y;
//	const float _x = x * x;
//	const float _sigma = 2.0f*sigma*sigma;
//	const float div = (_y + _x) / _sigma;
//	return exp(-div) / (_sigma * CL_M_PI);
//}
Image_Stabilization::Image_Stabilization(clDevice* device, cl_uint width, cl_uint height, cl_uint block_x, cl_uint block_y, cl_uint step_x, cl_uint step_y, cl_uint radius)
{
	for (size_t i = 0; i < sizeof(*this); i++)
		((char*)this)[i] = 0;
	gpu_data = (OpenCL_data*)malloc(sizeof(OpenCL_data));
	gpu_data->_device = device;
	gpu_data->width = width;
	gpu_data->height = height;
	gpu_data->kernel_image_stabilization_part1 = device->findKernel((const cl_char*)"stabilization_image_part1", sizeof("stabilization_image_part1"), gpu_data->index_program_stabilization);
	gpu_data->kernel_image_stabilization_part2 = device->findKernel((const cl_char*)"stabilization_image_part2", sizeof("stabilization_image_part2"), gpu_data->index_program_stabilization);
	gpu_data->kernel_image_stabilization_part3 = device->findKernel((const cl_char*)"stabilization_image_part3", sizeof("stabilization_image_part3"), gpu_data->index_program_stabilization);
	gpu_data->kernel_image_stabilization_part4 = device->findKernel((const cl_char*)"stabilization_image_part4", sizeof("stabilization_image_part4"), gpu_data->index_program_stabilization);
	size_t number_blocks_x = (width / step_x);
	size_t number_blocks_y = (height / step_y);
	size_t x = sqrt(float(gpu_data->_device->DeviceInfo.maxWorkGroupSize));
	size_t y = gpu_data->_device->DeviceInfo.maxWorkGroupSize / x;
	gpu_data->stabilization_indices = uint8{ width, height, step_x, step_y, block_x, block_y, radius, NULL };
	gpu_data->length_sync_buffer = (number_blocks_x * block_x) * number_blocks_y * block_y;
	gpu_data->stabilization_part2_localWork[0] = (x < block_x ? x : block_x) * (y < block_y ? y : block_y);
	gpu_data->stabilization_part2_localWork[1] = 1;
	gpu_data->stabilization_part2_localWork[2] = 1;
	gpu_data->stabilization_part2_globalWork[0] = gpu_data->stabilization_part2_localWork[0] * gpu_data->_device->DeviceInfo.maxComputeUnits;
	gpu_data->stabilization_part2_globalWork[1] = 1;
	gpu_data->stabilization_part2_globalWork[2] = 1;

	gpu_data->stabilization_globalWork[0] = number_blocks_x * block_x;
	gpu_data->stabilization_globalWork[1] = number_blocks_y * block_y;
	gpu_data->stabilization_globalWork[2] = 1;
	gpu_data->stabilization_localWork[0] = x;
	gpu_data->stabilization_localWork[1] = y;
	gpu_data->stabilization_localWork[2] = 1;
	int number_blocks_on_width = gpu_data->_device->DeviceInfo.maxComputeUnits / (width / gpu_data->stabilization_localWork[0]);
	gpu_data->stabilization_globalWork[0] = gpu_data->stabilization_localWork[0] * (gpu_data->_device->DeviceInfo.maxComputeUnits > (width / gpu_data->stabilization_localWork[0]) ? (width / gpu_data->stabilization_localWork[0]) : gpu_data->_device->DeviceInfo.maxComputeUnits);
	gpu_data->stabilization_globalWork[1] = gpu_data->stabilization_localWork[1] * (number_blocks_on_width > 0 ? number_blocks_on_width : 1);
	gpu_data->stabilization_globalWork[2] = 1;
	size_t length_row_pitch_data = width * sizeof(cl_uchar4);

	gpu_data->memory_buffer_0 = device->mallocBufferMemory(NULL, width * height * sizeof(cl_uchar4));
	gpu_data->memory_buffer_1 = device->mallocBufferMemory(NULL, width * height * sizeof(cl_uchar4));
	gpu_data->memory_buffer_2 = device->mallocBufferMemory(NULL, width * height * sizeof(cl_uchar4));
	gpu_data->sync_buffer = device->mallocBufferMemory(NULL, gpu_data->length_sync_buffer * sizeof(cl_int2));
	ptr_stabilization_function = &Image_Stabilization::gpu_Stabilization_function;

	size_t size_block = block_x * block_y;
	int number_steps_block = size_block / gpu_data->stabilization_part2_localWork[0] + (size_block % gpu_data->stabilization_part2_localWork[0] ? 1 : 0);
	int current_step = (pow((2), ceil((log2((gpu_data->stabilization_part2_localWork[0]))))));
	cl_int local_memory = -int(current_step * sizeof(cl_int2));
	gpu_data->stabilization_length_args = int8{ sizeof(cl_uint),sizeof(cl_uint),sizeof(cl_uint),sizeof(cl_uint),sizeof(cl_uint),sizeof(cl_uint), sizeof(cl_uint), local_memory };
	if (gpu_data->_device->DeviceInfo.supportImages &&
		gpu_data->_device->DeviceInfo.maxHeightImage2D > height &&
		gpu_data->_device->DeviceInfo.maxWidthImage2D > width) {
		gpu_data->norm_image_gpu_0 = device->mallocImage2DMemory(NULL, height, width, length_row_pitch_data, CL_RGBA, CL_UNORM_INT8);
		gpu_data->norm_image_gpu_1 = device->mallocImage2DMemory(NULL, height, width, length_row_pitch_data, CL_RGBA, CL_UNORM_INT8);
		std::string options = " -D LOCAL_STEPS=" + std::to_string(current_step) + " -D NUMBER_STEPS_BLOCK=" + std::to_string(number_steps_block)
			+ " -D RADIUS_BLOCK=" + std::to_string(radius) + " -D SIZE_BLOCK_X=" + std::to_string(block_x) + " -D SIZE_BLOCK_Y=" + std::to_string(block_y)
			+ " -D USE_TEXTURE";
		device->rebuildProgram(gpu_data->index_program_stabilization, (const cl_char*)options.c_str(), sizeof(options.size()));
	}
	else {
		std::string options = " -D LOCAL_STEPS=" + std::to_string(current_step) + " -D NUMBER_STEPS_BLOCK=" + std::to_string(number_steps_block)
			+ " -D RADIUS_BLOCK=" + std::to_string(radius) + " -D SIZE_BLOCK_X=" + std::to_string(block_x) + " -D SIZE_BLOCK_Y=" + std::to_string(block_y)
			;
		device->rebuildProgram(gpu_data->index_program_stabilization, (const cl_char*)options.c_str(), sizeof(options.size()));
	}

	//float w = 4.0f * sqrt(2.0f * log(2.0f));
	//float sigma = block_x / w;
	//for (int h = 0; h < block_y; h++) {
	//	for (int w = 0; w < block_x; w++) {
	//		float result_gauss = Gaussian_filter_xy(h - (int)block_y/2, w - (int)block_x/2, sigma);
	//		printf("%f ", result_gauss);
	//	}
	//	printf("\n");
	//}
}
void Image_Stabilization::gpu_Stabilization_function(void* data_next_image, void* result) {
	size_t buffers[] = { gpu_data->memory_buffer_0, gpu_data->memory_buffer_1, gpu_data->sync_buffer };
	size_t textures[] = { gpu_data->norm_image_gpu_0, gpu_data->norm_image_gpu_1 };
	gpu_data->_device->writeBuffer(result, gpu_data->memory_buffer_1, gpu_data->width * gpu_data->height * sizeof(uchar4));

	if (gpu_data->norm_image_gpu_0) {
		gpu_data->_device->write2DImage(data_next_image, gpu_data->norm_image_gpu_0, gpu_data->width, gpu_data->height);
		gpu_data->_device->write2DImage(result, gpu_data->norm_image_gpu_1, gpu_data->width, gpu_data->height);
		gpu_data->_device->callOpenclFunction(gpu_data->index_program_stabilization, gpu_data->kernel_image_stabilization_part1, &gpu_data->sync_buffer, textures, (cl_char*)& gpu_data->stabilization_indices, (cl_int*)& gpu_data->stabilization_length_args, 1, 2, 7, gpu_data->stabilization_globalWork, gpu_data->stabilization_localWork);
	}
	else {
		gpu_data->_device->writeBuffer(data_next_image, gpu_data->memory_buffer_0, gpu_data->width * gpu_data->height * sizeof(uchar4));
		gpu_data->_device->callOpenclFunction(gpu_data->index_program_stabilization, gpu_data->kernel_image_stabilization_part1, buffers, NULL, (cl_char*)& gpu_data->stabilization_indices, (cl_int*)& gpu_data->stabilization_length_args, 3, 0, 7, gpu_data->stabilization_globalWork, gpu_data->stabilization_localWork);
	}

	gpu_data->_device->callOpenclFunction(gpu_data->index_program_stabilization, gpu_data->kernel_image_stabilization_part2, buffers + 2, NULL, (cl_char*)& gpu_data->stabilization_indices, (cl_int*)& gpu_data->stabilization_length_args, 1, 0, 8, gpu_data->stabilization_part2_globalWork, gpu_data->stabilization_part2_localWork);
	gpu_data->_device->callOpenclFunction(gpu_data->index_program_stabilization, gpu_data->kernel_image_stabilization_part3, buffers + 1, NULL, (cl_char*)&gpu_data->stabilization_indices, (cl_int*)&gpu_data->stabilization_length_args, 2, 0, 7, gpu_data->stabilization_globalWork);
	//*buffers = gpu_data->memory_buffer_2;
	//gpu_data->_device->callOpenclFunction(gpu_data->index_program_stabilization, gpu_data->kernel_image_stabilization_part4, buffers, NULL, (cl_char*)& gpu_data->stabilization_indices, (cl_int*)& gpu_data->stabilization_length_args, 3, 0, 7, gpu_data->stabilization_globalWork);

	gpu_data->_device->readBuffer(result, gpu_data->memory_buffer_1, gpu_data->width * gpu_data->height * sizeof(uchar4));
}

void Image_Stabilization::AMP_Stabilization_function(void* data_next_image, void* result) {
}

void Image_Stabilization::cpu_Stabilization_function(void* data, void* result)
{
}
void Image_Stabilization::Calculate_Gauss_function(void* data, void* result) {
	(this->*ptr_gauss_function)(data, result);
}
void Image_Stabilization::Stabilization_function(void* data_next_image, void* result) {
	(this->*ptr_stabilization_function)(data_next_image, result);
}
Image_Stabilization::~Image_Stabilization()
{
	if (amp_data)
		free(amp_data);
	if (gpu_data) {
		gpu_data->_device->freeMemory(gpu_data->norm_image_gpu_0);
		gpu_data->_device->freeMemory(gpu_data->norm_image_gpu_1);
		gpu_data->_device->freeMemory(gpu_data->memory_buffer_0);
		gpu_data->_device->freeMemory(gpu_data->memory_buffer_1);
		gpu_data->_device->freeMemory(gpu_data->memory_buffer_2);
		free(gpu_data);
	}
	if (cpu_data) {
		if (cpu_data->current_image)
			_mm_free(cpu_data->current_image);
		free(cpu_data);
	}
}