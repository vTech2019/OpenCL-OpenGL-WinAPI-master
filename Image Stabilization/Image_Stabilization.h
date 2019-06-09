#pragma once
#include "../clDevice.h"
#include "../cpuDevice.h"
#include "../AMP_device.h"
#include <string>

struct AMP_data {
	AMP_device* device;
	size_t block_x;
	size_t block_y;
	size_t width;
	size_t height;
	size_t radius;
	size_t index_extent_gauss;
	size_t index_extent_MAD_block_matching;
};
struct OpenCL_data {
	size_t stabilization_globalWork[3];
	size_t stabilization_localWork[3];
	size_t stabilization_part2_globalWork[3];
	size_t stabilization_part2_localWork[3];
	uint8 stabilization_indices;
	int8 stabilization_length_args;
	cl_uint width;
	cl_uint height;
	cl_uint length_sync_buffer;
	cl_uint index_program_stabilization;
	cl_int kernel_image_stabilization_part1;
	cl_int kernel_image_stabilization_part2;
	cl_int kernel_image_stabilization_part3;
	clDevice* _device;
	size_t norm_image_gpu_0;
	size_t norm_image_gpu_1;
	size_t memory_buffer_0;
	size_t memory_buffer_1;
	size_t sync_buffer;
	size_t sync_index;
};
struct C_code_data {
	cpuDevice* ptrDevice;
	cl_float4* current_image;
	cl_float4* next_image;
	uint32_t block_x;
	uint32_t block_y;
	uint32_t pitch_width_image;
	uint32_t width_image;
	uint32_t height_image;
	uint32_t offset_radius_aligned;
	uint32_t radius;
	cl_float pow_sigma;
	cl_float normalize;
	uint32_t part_block_x;
	uint32_t part_block_y;
};
class Image_Stabilization
{
	OpenCL_data* gpu_data;
	C_code_data* cpu_data;
	AMP_data* amp_data;
	void(Image_Stabilization::*ptr_gauss_function)(void* data, void* result);
	void(Image_Stabilization::*ptr_stabilization_function)(void* data, void* result);
public:
	Image_Stabilization(clDevice * device, cl_uint width, cl_uint height, cl_uint block_x, cl_uint block_y, cl_uint step_x, cl_uint step_y, cl_uint radius);
	Image_Stabilization(cpuDevice * device, cl_uint width, cl_uint height, cl_uint block_x, cl_uint block_y, cl_uint radius);
	Image_Stabilization(AMP_device* device, cl_uint width, cl_uint height, cl_uint block_x, cl_uint block_y, cl_uint radius);

	void gpu_Stabilization_function(void * data_next_image, void * result);
	void AMP_Stabilization_function(void* data_next_image, void* result);
	void cpu_Stabilization_function(void* data, void* result);

	void Calculate_Gauss_function(void * data, void * result);
	void Stabilization_function(void * data_next_image, void * result);
	~Image_Stabilization();
};
