#include "clDevice.h"

void CL_CALLBACK pfnBuildProgram(cl_program program, void* userData)
{
	cl_device_id* deviceID = (cl_device_id*)userData;
	cl_build_status status;
	size_t length = 0;
	cl_int ret = clGetProgramBuildInfo(program, *deviceID, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, &length);
	if (status == CL_BUILD_IN_PROGRESS)
	{
		cl_int ret = clGetProgramBuildInfo(program, *deviceID, CL_PROGRAM_BUILD_LOG, NULL, NULL, &length);
		if (ret != CL_SUCCESS) {
			printf("clGetProgramBuildInfo - %d\n", ret);
		}
		cl_char* data = (cl_char*)malloc(length + 1);
		data[length] = 0;
		ret = clGetProgramBuildInfo(program, *deviceID, CL_PROGRAM_BUILD_LOG, length, data, NULL);
		if (ret != CL_SUCCESS)
			printf("clGetProgramBuildInfo - %d\n", ret);
		printf("CL_BUILD_IN_PROGRESS - %s\n", data);
		free(data);
	}
	else if (status == CL_BUILD_ERROR)
	{
		cl_int ret = clGetProgramBuildInfo(program, *deviceID, CL_PROGRAM_BUILD_LOG, NULL, NULL, &length);
		if (ret != CL_SUCCESS)
			printf("clGetProgramBuildInfo - %d\n", ret);
		cl_char* data = (cl_char*)malloc(length + 1);
		data[length] = 0;
		ret = clGetProgramBuildInfo(program, *deviceID, CL_PROGRAM_BUILD_LOG, length, data, NULL);
		if (ret != CL_SUCCESS)
			printf("clGetProgramBuildInfo - %d\n", ret);
		printf("CL_BUILD_ERROR - %s\n", data);
		free(data);
	}
	else if (status == CL_BUILD_NONE) {
		printf("CL_BUILD_NONE\n");
	}
	else if (status == CL_BUILD_SUCCESS) {
		printf("CL_BUILD_SUCCESS\n");
	}
}

const char* getInformationError(cl_int codeError)
{
	switch (codeError)
	{
	case CL_PLATFORM_NOT_FOUND_KHR:
		return "CL_PLATFORM_NOT_FOUND_KHR";
	case CL_SUCCESS:
		return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:
		return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:
		return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:
		return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:
		return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:
		return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:
		return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:
		return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:
		return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:
		return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case CL_COMPILE_PROGRAM_FAILURE:
		return "CL_COMPILE_PROGRAM_FAILURE";
	case CL_LINKER_NOT_AVAILABLE:
		return "CL_LINKER_NOT_AVAILABLE";
	case CL_LINK_PROGRAM_FAILURE:
		return "CL_LINK_PROGRAM_FAILURE";
	case CL_DEVICE_PARTITION_FAILED:
		return "CL_DEVICE_PARTITION_FAILED";
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
		return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
	case CL_INVALID_VALUE:
		return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:
		return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:
		return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:
		return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:
		return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:
		return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:
		return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:
		return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:
		return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:
		return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:
		return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:
		return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:
		return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:
		return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:
		return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:
		return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:
		return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:
		return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:
		return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:
		return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:
		return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:
		return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:
		return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:
		return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:
		return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:
		return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:
		return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:
		return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:
		return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:
		return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:
		return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:
		return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:
		return "CL_INVALID_GLOBAL_WORK_SIZE";
	case CL_INVALID_PROPERTY:
		return "CL_INVALID_PROPERTY";
	case CL_INVALID_IMAGE_DESCRIPTOR:
		return "CL_INVALID_IMAGE_DESCRIPTOR";
	case CL_INVALID_COMPILER_OPTIONS:
		return "CL_INVALID_COMPILER_OPTIONS";
	case CL_INVALID_LINKER_OPTIONS:
		return "CL_INVALID_LINKER_OPTIONS";
	case CL_INVALID_DEVICE_PARTITION_COUNT:
		return "CL_INVALID_DEVICE_PARTITION_COUNT";
	case -9999:
		return "Illegal read or write to a buffer (NVIDIA)";
	default:
		return "CL_INVALID_INDEX";
	}
}

void platformInfo(cl_platform_id platform, cl_platform_info information, const cl_char textInformation[], cl_char* strPlatformInfo[5]) {
	size_t length;
	size_t typeInfo = information - 0x900;
	CL_CHECK(clGetPlatformInfo(platform, information, NULL, NULL, &length), textInformation);
	strPlatformInfo[typeInfo] = (cl_char*)malloc(length + 1);
	strPlatformInfo[typeInfo][length] = 0;
	CL_CHECK(clGetPlatformInfo(platform, information, length, strPlatformInfo[typeInfo], NULL), textInformation);
	printf("%s - %s\n", textInformation, strPlatformInfo[typeInfo]);
}

void deviceInfo(cl_device_id device, cl_platform_info information, const cl_char textInformation[], void** data, cl_uint lengthData) {
	size_t length;
	CL_CHECK(clGetDeviceInfo(device, information, NULL, NULL, &length), textInformation);
	if (lengthData != length) {
		*data = (cl_char*)calloc((length + 1), sizeof(cl_char));
		CL_CHECK(clGetDeviceInfo(device, information, length, *data, NULL), textInformation);
		printf("%s - %s\n", textInformation, *data);
	}
	else {
		CL_CHECK(clGetDeviceInfo(device, information, length, &(*((cl_char*)data)), NULL), textInformation);
	}
}

clPlatform::clPlatform() {
	for (size_t i = 0; i < sizeof(*this); i++)
		((cl_char*)this)[i] = 0;
	cl_int errorCode;
	CL_CHECK(clGetPlatformIDs(NULL, NULL, &numberPlatforms), "clGetPlatformIDs");
	platforms = (cl_platform_id*)malloc(numberPlatforms * sizeof(cl_platform_id));
	context = (cl_context*)malloc(numberPlatforms * sizeof(cl_context));
	platformDevices = (cl_uint*)malloc(numberPlatforms * sizeof(cl_uint));
	CL_CHECK(clGetPlatformIDs(numberPlatforms, platforms, NULL), "clGetPlatformIDs");

	if (numberPlatforms == 0) return;
	for (cl_uint i = 0; i < numberPlatforms; i++) {
		cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[getNextPlatform], 0 };
		CL_CHECK(clGetDeviceIDs(platforms[getNextPlatform], CL_DEVICE_TYPE_ALL, NULL, NULL, &platformDevices[getNextPlatform]), "clGetDeviceIDs");
		devices = (cl_device_id*)realloc(devices, (platformDevices[getNextPlatform] + numberDevices) * sizeof(cl_device_id));
		queue = (cl_command_queue*)realloc(queue, (platformDevices[getNextPlatform] + numberDevices) * sizeof(cl_command_queue));
		CL_CHECK(clGetDeviceIDs(platforms[getNextPlatform], CL_DEVICE_TYPE_ALL, platformDevices[getNextPlatform], devices + numberDevices, NULL), "clGetDeviceIDs");
		cl_context ctx = clCreateContext(properties, platformDevices[getNextPlatform], devices + numberDevices, NULL, NULL, &errorCode);
		if (ctx) {
			context[getNextPlatform] = ctx;
			if (errorCode == CL_DEVICE_NOT_AVAILABLE) {
				cl_char* profileVersionNameVendorExtensions[5];
				platformInfo(platforms[getNextPlatform], CL_PLATFORM_VENDOR, (const cl_char*)"CL_PLATFORM_VENDOR", profileVersionNameVendorExtensions);
				platformInfo(platforms[getNextPlatform], CL_PLATFORM_NAME, (const cl_char*)"CL_PLATFORM_NAME", profileVersionNameVendorExtensions);
				platformInfo(platforms[getNextPlatform], CL_PLATFORM_VERSION, (const cl_char*)"CL_PLATFORM_VERSION", profileVersionNameVendorExtensions);
				platformInfo(platforms[getNextPlatform], CL_PLATFORM_PROFILE, (const cl_char*)"CL_PLATFORM_PROFILE", profileVersionNameVendorExtensions);
				printf("CL_DEVICE_NOT_AVAILABLE!\n");
				clReleaseDevice(devices[numberDevices]);
				devices = (cl_device_id*)realloc(devices, (numberDevices) * sizeof(cl_device_id));
				queue = (cl_command_queue*)realloc(queue, (numberDevices) * sizeof(cl_command_queue));
			}
			else {
				CL_CHECK(errorCode, "clCreateContext");
				for (cl_uint j = 0; j < platformDevices[getNextPlatform]; j++) {
					queue[numberDevices + j] = clCreateCommandQueue(context[getNextPlatform], devices[numberDevices + j], CL_QUEUE_PROFILING_ENABLE, &errorCode);
					CL_CHECK(errorCode, "clCreateCommandQueue");
				}
				numberDevices += platformDevices[getNextPlatform];
				getNextPlatform++;
			}
		}
		else {
			devices = (cl_device_id*)realloc(devices, (numberDevices) * sizeof(cl_device_id));
			queue = (cl_command_queue*)realloc(queue, (numberDevices) * sizeof(cl_command_queue));
		}
	}
}

cl_command_queue* clPlatform::getCommandQueueID(cl_uint index) {
	return index < numberDevices ? &queue[index] : NULL;
}
cl_device_id* clPlatform::getDeviceID(cl_uint index) {
	return index < numberDevices ? &devices[index] : NULL;
}
cl_platform_id* clPlatform::getPlatformID(cl_uint index) {
	if (index > numberDevices)
		return NULL;
	cl_uint sum = 0;
	for (size_t i = 0; i < numberPlatforms; i++) {
		sum += platformDevices[i];
		if (index < sum)
			return &platforms[i];
	}
	return NULL;
}
cl_context* clPlatform::getContextID(cl_uint index) {
	if (index > numberDevices)
		return NULL;
	cl_uint sum = 0;
	for (cl_uint i = 0; i < numberPlatforms; i++) {
		sum += platformDevices[i];
		if (index < sum)
			return &context[i];
	}
	return NULL;
}

clPlatform::~clPlatform() {
	for (cl_uint i = 0; i < numberDevices; i++) {
		clReleaseCommandQueue(queue[i]);
		clReleaseDevice(devices[i]);
	}
	for (cl_uint i = 0; i < numberPlatforms; i++)
		clReleaseContext(context[i]);
	if (context) free(context);
	if (queue) free(queue);
	if (devices) free(devices);
	if (platforms) free(platforms);
	if (platformDevices) free(platformDevices);
}
clDevice::clDevice(clPlatform* platformData, cl_uint indexDevice)
{
	memset(this, 0, sizeof(*this));
	if (platformData->getNumberDevices() < indexDevice) return;
	platform = platformData->getPlatformID(indexDevice);
	context = platformData->getContextID(indexDevice);
	device = platformData->getDeviceID(indexDevice);
	queue = platformData->getCommandQueueID(indexDevice);

	platformInfo(*platform, CL_PLATFORM_VENDOR, (const cl_char*)"CL_PLATFORM_VENDOR", profileVersionNameVendorExtensions);
	platformInfo(*platform, CL_PLATFORM_NAME, (const cl_char*)"CL_PLATFORM_NAME", profileVersionNameVendorExtensions);
	platformInfo(*platform, CL_PLATFORM_VERSION, (const cl_char*)"CL_PLATFORM_VERSION", profileVersionNameVendorExtensions);
	platformInfo(*platform, CL_PLATFORM_PROFILE, (const cl_char*)"CL_PLATFORM_PROFILE", profileVersionNameVendorExtensions);
	platformInfo(*platform, CL_PLATFORM_EXTENSIONS, (const cl_char*)"CL_PLATFORM_EXTENSIONS", profileVersionNameVendorExtensions);

	deviceInfo(*device, CL_DEVICE_EXTENSIONS, (const cl_char*)"CL_DEVICE_EXTENSIONS", (void**)& DeviceInfo.deviceExtensions, 0);
	deviceInfo(*device, CL_DEVICE_VENDOR, (const cl_char*)"CL_DEVICE_VENDOR_ID", (void**)& DeviceInfo.deviceVendor, 0);
	deviceInfo(*device, CL_DEVICE_TYPE, (const cl_char*)"CL_DEVICE_TYPE", (void**)& DeviceInfo.deviceType, sizeof(cl_device_type));
	deviceInfo(*device, CL_DEVICE_LOCAL_MEM_TYPE, (const cl_char*)"CL_DEVICE_LOCAL_MEM_TYPE", (void**)& DeviceInfo.localMemoryType, sizeof(cl_device_local_mem_type));
	deviceInfo(*device, CL_DEVICE_MAX_COMPUTE_UNITS, (const cl_char*)"CL_DEVICE_MAX_COMPUTE_UNITS", (void**)& DeviceInfo.maxComputeUnits, sizeof(cl_uint));
	deviceInfo(*device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, (const cl_char*)"CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS", (void**)& DeviceInfo.workItemDemension, sizeof(cl_uint));
	deviceInfo(*device, CL_DEVICE_MAX_WORK_ITEM_SIZES, (const cl_char*)"CL_DEVICE_MAX_WORK_ITEM_SIZES", (void**)& DeviceInfo.workItemSizes, 3 * sizeof(size_t));
	deviceInfo(*device, CL_DEVICE_LOCAL_MEM_SIZE, (const cl_char*)"CL_DEVICE_LOCAL_MEM_SIZE", (void**)& DeviceInfo.localMemorySize, sizeof(cl_ulong));
	deviceInfo(*device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, (const cl_char*)"CL_DEVICE_MAX_MEM_ALLOC_SIZE", (void**)& DeviceInfo.maxGlobalMemoryAllocate, sizeof(cl_ulong));
	deviceInfo(*device, CL_DEVICE_MAX_PARAMETER_SIZE, (const cl_char*)"CL_DEVICE_MAX_PARAMETER_SIZE", (void**)& DeviceInfo.maximumParametersInKernel, sizeof(size_t));
	deviceInfo(*device, CL_DEVICE_MAX_WORK_GROUP_SIZE, (const cl_char*)"CL_DEVICE_MAX_WORK_GROUP_SIZE", (void**)& DeviceInfo.maxWorkGroupSize, sizeof(size_t));
	deviceInfo(*device, CL_DEVICE_GLOBAL_MEM_SIZE, (const cl_char*)"CL_DEVICE_GLOBAL_MEM_SIZE", (void**)& DeviceInfo.globalMemSize, sizeof(cl_ulong));
	deviceInfo(*device, CL_DEVICE_IMAGE_SUPPORT, (const cl_char*)"CL_DEVICE_IMAGE_SUPPORT", (void**)& DeviceInfo.supportImages, sizeof(cl_bool));
	deviceInfo(*device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, (const cl_char*)"CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR", (void**)& DeviceInfo.preferVectorChar, sizeof(cl_uint));
	deviceInfo(*device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, (const cl_char*)"CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT", (void**)& DeviceInfo.preferVectorShort, sizeof(cl_uint));
	deviceInfo(*device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, (const cl_char*)"CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT", (void**)& DeviceInfo.preferVectorInt, sizeof(cl_uint));
	deviceInfo(*device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, (const cl_char*)"CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG", (void**)& DeviceInfo.preferVectorLong, sizeof(cl_uint));
	deviceInfo(*device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, (const cl_char*)"CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT", (void**)& DeviceInfo.preferVectorFloat, sizeof(cl_uint));
	deviceInfo(*device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, (const cl_char*)"CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE", (void**)& DeviceInfo.preferVectorDouble, sizeof(cl_uint));
	if (DeviceInfo.supportImages) {
		deviceInfo(*device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, (const cl_char*)"CL_DEVICE_IMAGE2D_MAX_HEIGHT", (void**)& DeviceInfo.maxHeightImage2D, sizeof(size_t));
		deviceInfo(*device, CL_DEVICE_IMAGE2D_MAX_WIDTH, (const cl_char*)"CL_DEVICE_IMAGE2D_MAX_WIDTH", (void**)& DeviceInfo.maxWidthImage2D, sizeof(size_t));
		deviceInfo(*device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, (const cl_char*)"CL_DEVICE_IMAGE3D_MAX_HEIGHT", (void**)& DeviceInfo.maxHeightImage3D, sizeof(size_t));
		deviceInfo(*device, CL_DEVICE_IMAGE3D_MAX_WIDTH, (const cl_char*)"CL_DEVICE_IMAGE3D_MAX_WIDTH", (void**)& DeviceInfo.maxWidthImage3D, sizeof(size_t));
		deviceInfo(*device, CL_DEVICE_IMAGE3D_MAX_DEPTH, (const cl_char*)"CL_DEVICE_IMAGE3D_MAX_DEPTH", (void**)& DeviceInfo.maxDepthImage3D, sizeof(size_t));
	}
}

bool clDevice::clPushProgram(cl_char* text, size_t lengthText, const cl_char* options)
{
	cl_uint number_kernels;
	cl_int errorCode;
	namesPrograms = (cl_char * *)realloc(namesPrograms, (numberPrograms + 1) * sizeof(cl_char*));
	namesPrograms[numberPrograms] = (cl_char*)malloc(lengthText * sizeof(cl_char));
	memcpy(namesPrograms[numberPrograms], text, lengthText);
	printf("-----------------Program---------------------\n");
	printf("%s \n", namesPrograms[numberPrograms]);
	printf("---------------------------------------------\n");
	programDevice = (programData*)realloc(programDevice, (numberPrograms + 1) * sizeof(programData));
	memset(&programDevice[numberPrograms], 0, sizeof(programData));
	programDevice[numberPrograms].programDevice = clCreateProgramWithSource(*context, 1, (const char**)& text, NULL, &errorCode);
	CL_CHECK(errorCode, "clCreateProgramWithSource");
	CL_CHECK(clBuildProgram(programDevice[numberPrograms].programDevice, 1, device, (const char*)options, pfnBuildProgram, (void*)device), "clBuildProgram");
	CL_CHECK(clCreateKernelsInProgram(programDevice[numberPrograms].programDevice, NULL, NULL, &number_kernels), "clCreateKernelsInProgram");
	if (number_kernels) {
		programDevice[numberPrograms].kernels = (cl_kernel*)realloc(programDevice[numberPrograms].kernels, (number_kernels) * sizeof(cl_kernel));
		CL_CHECK(clCreateKernelsInProgram(programDevice[numberPrograms].programDevice, number_kernels, programDevice[numberPrograms].kernels, NULL), "clCreateKernelsInProgram");
		programDevice[numberPrograms].namesKernels = (cl_char * *)realloc(programDevice[numberPrograms].namesKernels, (number_kernels) * sizeof(cl_char*));
		programDevice[numberPrograms].kernelInfo = (kernelInformation*)realloc(programDevice[numberPrograms].kernelInfo, (number_kernels) * sizeof(kernelInformation));
		printf("-----------------kernels---------------------\n");
		for (size_t i = 0; i < number_kernels; i++) {
			cl_char* kernel_name = NULL;
			size_t length_name_kernel;
			CL_CHECK(clGetKernelInfo(programDevice[numberPrograms].kernels[i], CL_KERNEL_FUNCTION_NAME, NULL, NULL, &length_name_kernel), "clGetKernelInfo");
			programDevice[numberPrograms].namesKernels[i] = (cl_char*)malloc((length_name_kernel + 1) * sizeof(cl_char));
			CL_CHECK(clGetKernelInfo(programDevice[numberPrograms].kernels[i], CL_KERNEL_FUNCTION_NAME, length_name_kernel, programDevice[numberPrograms].namesKernels[i], &length_name_kernel), "clGetKernelInfo");
			programDevice[numberPrograms].namesKernels[i][length_name_kernel] = 0;
			CL_CHECK(clGetKernelWorkGroupInfo(programDevice[numberPrograms].kernels[i], *device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(size_t[3]), &programDevice[numberPrograms].kernelInfo[i].local_work_size, NULL), "clGetKernelWorkGroupInfo");
			CL_CHECK(clGetKernelWorkGroupInfo(programDevice[numberPrograms].kernels[i], *device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &programDevice[numberPrograms].kernelInfo[i].max_work_group_size, NULL), "clGetKernelWorkGroupInfo");
			CL_CHECK(clGetKernelWorkGroupInfo(programDevice[numberPrograms].kernels[i], *device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &programDevice[numberPrograms].kernelInfo[i].local_size, NULL), "clGetKernelWorkGroupInfo");
			CL_CHECK(clGetKernelWorkGroupInfo(programDevice[numberPrograms].kernels[i], *device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &programDevice[numberPrograms].kernelInfo[i].prefer_work_group_size, NULL), "clGetKernelWorkGroupInfo");
			CL_CHECK(clGetKernelWorkGroupInfo(programDevice[numberPrograms].kernels[i], *device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &programDevice[numberPrograms].kernelInfo[i].private_memory, NULL), "clGetKernelWorkGroupInfo");
			printf("index kernel : %zi; name kernel : %s \n", i, programDevice[numberPrograms].namesKernels[i]);
		}
		printf("---------------------------------------------\n");
		programDevice[numberPrograms].numberKernels += number_kernels;
	}
	numberPrograms++;
	return true;
}
bool clDevice::rebuildProgram(size_t index_program, const cl_char* options, size_t length) {
	cl_uint number_kernels;
	if (index_program < numberPrograms) {
		for (size_t j = 0; j < programDevice[index_program].numberKernels; j++) {
			clReleaseKernel(programDevice[index_program].kernels[j]);
		}
		CL_CHECK(clBuildProgram(programDevice[index_program].programDevice, 1, device, (const char*)options, pfnBuildProgram, (void*)device), "clBuildProgram");
		CL_CHECK(clCreateKernelsInProgram(programDevice[index_program].programDevice, NULL, NULL, &number_kernels), "clCreateKernelsInProgram");
		if (number_kernels) {
			programDevice[index_program].kernels = (cl_kernel*)realloc(programDevice[index_program].kernels, (number_kernels) * sizeof(cl_kernel));
			CL_CHECK(clCreateKernelsInProgram(programDevice[index_program].programDevice, number_kernels, programDevice[index_program].kernels, NULL), "clCreateKernelsInProgram");
			programDevice[index_program].namesKernels = (cl_char * *)realloc(programDevice[index_program].namesKernels, (number_kernels) * sizeof(cl_char*));
			programDevice[index_program].kernelInfo = (kernelInformation*)realloc(programDevice[index_program].kernelInfo, (number_kernels) * sizeof(kernelInformation));
			printf("-----------------kernels---------------------\n");
			for (size_t i = 0; i < number_kernels; i++) {
				cl_char* kernel_name = NULL;
				size_t length_name_kernel;
				CL_CHECK(clGetKernelInfo(programDevice[index_program].kernels[i], CL_KERNEL_FUNCTION_NAME, NULL, NULL, &length_name_kernel), "clGetKernelInfo");
				programDevice[index_program].namesKernels[i] = (cl_char*)malloc((length_name_kernel + 1) * sizeof(cl_char));
				CL_CHECK(clGetKernelInfo(programDevice[index_program].kernels[i], CL_KERNEL_FUNCTION_NAME, length_name_kernel, programDevice[index_program].namesKernels[i], &length_name_kernel), "clGetKernelInfo");
				programDevice[index_program].namesKernels[i][length_name_kernel] = 0;
				CL_CHECK(clGetKernelWorkGroupInfo(programDevice[index_program].kernels[i], *device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(size_t[3]), &programDevice[index_program].kernelInfo[i].local_work_size, NULL), "clGetKernelWorkGroupInfo");
				CL_CHECK(clGetKernelWorkGroupInfo(programDevice[index_program].kernels[i], *device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &programDevice[index_program].kernelInfo[i].max_work_group_size, NULL), "clGetKernelWorkGroupInfo");
				CL_CHECK(clGetKernelWorkGroupInfo(programDevice[index_program].kernels[i], *device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &programDevice[index_program].kernelInfo[i].local_size, NULL), "clGetKernelWorkGroupInfo");
				CL_CHECK(clGetKernelWorkGroupInfo(programDevice[index_program].kernels[i], *device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &programDevice[index_program].kernelInfo[i].prefer_work_group_size, NULL), "clGetKernelWorkGroupInfo");
				CL_CHECK(clGetKernelWorkGroupInfo(programDevice[index_program].kernels[i], *device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &programDevice[index_program].kernelInfo[i].private_memory, NULL), "clGetKernelWorkGroupInfo");
				printf("index kernel : %zi; name kernel : %s \n", i, programDevice[index_program].namesKernels[i]);
			}
			printf("---------------------------------------------\n");
			programDevice[index_program].numberKernels = number_kernels;
		}
	}
	return true;
}

cl_int clDevice::findKernel(const cl_char* text, size_t length, cl_uint& indexProgram) {
	bool find = false;
	for (size_t index_program = 0; index_program < numberPrograms; index_program++)
		for (size_t i = 0; i < programDevice[index_program].numberKernels; i++) {
			for (size_t j = 0; j < length && programDevice[index_program].namesKernels[i][j] != 0 && programDevice[index_program].namesKernels[i][j] == text[j]; j++) {
				if (text[j + 1] == 0 && programDevice[index_program].namesKernels[i][j + 1] == 0)
					find = true;
			}
			if (find) {
				indexProgram = index_program;
				return i;
			}
		}
	return -1;
}

bool clDevice::clPushKernel(cl_char* text, size_t lengthText)
{
	cl_int errorCode;
	cl_kernel kernel = NULL;
	size_t i = 0;
	for (; i < numberPrograms; i++) {
		kernel = clCreateKernel(programDevice[i].programDevice, (const char*)text, &errorCode);
		if (CL_INVALID_KERNEL_NAME != errorCode)
			CL_CHECK(errorCode, "clCreateKernel");
		else
			break;
	}
	if (kernel == NULL)
		return false;
	programDevice[i].namesKernels = (cl_char * *)realloc(programDevice[i].namesKernels, (programDevice[i].numberKernels + 1) * sizeof(cl_char*));
	programDevice[i].namesKernels[programDevice[i].numberKernels] = (cl_char*)malloc((1 + lengthText) * sizeof(cl_char));
	memcpy(programDevice[i].namesKernels[programDevice[i].numberKernels], text, lengthText);
	programDevice[i].namesKernels[programDevice[i].numberKernels][lengthText] = 0;
	programDevice[i].kernels = (cl_kernel*)realloc(programDevice[i].kernels, (programDevice[i].numberKernels + 1) * sizeof(cl_kernel));
	programDevice[i].kernels[programDevice[i].numberKernels] = kernel;
	programDevice[i].kernelInfo = (kernelInformation*)realloc(programDevice[i].kernelInfo, (programDevice[i].numberKernels + 1) * sizeof(kernelInformation));
	clGetKernelWorkGroupInfo(kernel, *device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(size_t[3]), &programDevice[i].kernelInfo[programDevice[i].numberKernels].local_work_size, NULL);
	clGetKernelWorkGroupInfo(kernel, *device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &programDevice[i].kernelInfo[programDevice[i].numberKernels].max_work_group_size, NULL);
	clGetKernelWorkGroupInfo(kernel, *device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &programDevice[i].kernelInfo[programDevice[i].numberKernels].local_size, NULL);
	clGetKernelWorkGroupInfo(kernel, *device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &programDevice[i].kernelInfo[programDevice[i].numberKernels].prefer_work_group_size, NULL);
	clGetKernelWorkGroupInfo(kernel, *device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &programDevice[i].kernelInfo[programDevice[i].numberKernels].private_memory, NULL);
	programDevice[i].numberKernels++;
	return true;
}

size_t clDevice::mallocBufferMemory(const void* data, size_t lengthData) {
	cl_int errors;
	cl_memory_list* nextPtrBufferDevice = (cl_memory_list*)calloc(1, sizeof(cl_memory_list));
	if (!headPtrMemoryDevice) {
		headPtrMemoryDevice = nextPtrBufferDevice;
		ptrMemoryDevice = nextPtrBufferDevice;
	}
	else {
		nextPtrBufferDevice->previous = ptrMemoryDevice;
		ptrMemoryDevice->next = nextPtrBufferDevice;
		ptrMemoryDevice = nextPtrBufferDevice;
	}
	if ((void*)data)
		ptrMemoryDevice->memory_data = clCreateBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, lengthData, (void*)data, &errors);
	else
		ptrMemoryDevice->memory_data = clCreateBuffer(*context, CL_MEM_READ_WRITE, lengthData, NULL, &errors);
	CL_CHECK(errors, "clCreateBuffer");
	return (size_t)ptrMemoryDevice;
}

size_t clDevice::mallocImage2DMemory(const void* data, cl_uint height, cl_uint width, size_t rowPitch, int typeImage, int typeData) {
	cl_int errors;
	cl_image_format clImageFormat;
	clImageFormat.image_channel_order = typeImage;
	clImageFormat.image_channel_data_type = typeData;
	cl_memory_list* nextPtrImageDevice = (cl_memory_list*)calloc(1, sizeof(cl_memory_list));
	if (!headPtrMemoryDevice) {
		headPtrMemoryDevice = nextPtrImageDevice;
		ptrMemoryDevice = nextPtrImageDevice;
	}
	else {
		nextPtrImageDevice->previous = ptrMemoryDevice;
		ptrMemoryDevice->next = nextPtrImageDevice;
		ptrMemoryDevice = nextPtrImageDevice;
	}
	if ((void*)data)
		ptrMemoryDevice->memory_data = clCreateImage2D(*context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &clImageFormat, width, height, rowPitch, (void*)data, &errors);
	else
		ptrMemoryDevice->memory_data = clCreateImage2D(*context, CL_MEM_READ_WRITE, &clImageFormat, width, height, NULL, NULL, &errors);
	CL_CHECK(errors, "clCreateImage2D");
	return (size_t)ptrMemoryDevice;
}

bool clDevice::freeMemory(size_t address) {
	cl_memory_list* ptr_mem = (cl_memory_list*)address;
	CL_CHECK(clReleaseMemObject(ptr_mem->memory_data), "clReleaseMemObject");
	if (!ptr_mem->previous)
		if (ptr_mem->next)
			headPtrMemoryDevice = ptr_mem->next,
			headPtrMemoryDevice->previous = NULL;
		else
			headPtrMemoryDevice = NULL;
	ptr_mem->previous = ptr_mem->next;
	free(ptr_mem);
	return true;
}
void clDevice::callOpenclFunction(cl_uint index_program, size_t index_kernel, size_t* indices_buffers, size_t* indices_images, cl_char* indices_arguments, cl_int* size_indices_arguments, size_t number_buffers, size_t number_images, size_t number_arguments, size_t work_size[3]) {
	const size_t number_data = number_buffers + number_images + number_arguments;
	this->setArguments(index_program, index_kernel, indices_buffers, number_buffers, indices_images, number_images, indices_arguments, size_indices_arguments, number_arguments, 0);
	this->startCalculate(index_program, index_kernel, work_size);
}

void clDevice::callOpenclFunction(cl_uint index_program, size_t index_kernel, size_t* indices_buffers, size_t* indices_images, cl_char* indices_arguments, cl_int* size_indices_arguments, size_t number_buffers, size_t number_images, size_t number_arguments, size_t work_size[3], size_t local_work_size[3]) {
	const size_t number_data = number_buffers + number_images + number_arguments;
	this->setArguments(index_program, index_kernel, indices_buffers, number_buffers, indices_images, number_images, indices_arguments, size_indices_arguments, number_arguments, 0);
	this->startCalculate(index_program, index_kernel, work_size, local_work_size);
}
cl_bool clDevice::setArguments(cl_uint index_program, cl_uint index_kernel, size_t* addressMemoryBuffer, cl_uint numberIndicesMemoryBuffer, size_t* addressMemoryImage, cl_uint numberIndicesMemoryImage, cl_char* arguments, cl_int* typeArguments, cl_uint numberArguments, cl_uint index_kernel_arguments) {
	for (size_t i = 0; i < numberIndicesMemoryBuffer; i++) {
		cl_memory_list* memory = (cl_memory_list*)addressMemoryBuffer[i];
		CL_CHECK(clSetKernelArg(programDevice[index_program].kernels[index_kernel], index_kernel_arguments++, sizeof(cl_mem), &memory->memory_data), "clSetKernelArg");
	}
	for (size_t i = 0; i < numberIndicesMemoryImage; i++) {
		cl_memory_list* memory = (cl_memory_list*)addressMemoryImage[i];
		CL_CHECK(clSetKernelArg(programDevice[index_program].kernels[index_kernel], index_kernel_arguments++, sizeof(cl_mem), &memory->memory_data), "clSetKernelArg");
	}
	size_t offset = 0;
	for (size_t i = 0; i < numberArguments; i++) {
		if (typeArguments[i] < 0) {
			const int length_local_memory = -typeArguments[i];
			CL_CHECK(clSetKernelArg(programDevice[index_program].kernels[index_kernel], index_kernel_arguments++, length_local_memory, NULL), "clSetKernelArg");
			offset += sizeof(NULL);
		}
		else {
			CL_CHECK(clSetKernelArg(programDevice[index_program].kernels[index_kernel], index_kernel_arguments++, typeArguments[i], arguments + offset), "clSetKernelArg");
			offset += typeArguments[i];
		}
	}
	return true;
}
cl_bool clDevice::startCalculate(cl_uint index_program, cl_uint index_kernel, size_t globalWork[3], size_t localWork[3]) {
	cl_event kernelEvent;
	CL_CHECK(clEnqueueNDRangeKernel(*queue, programDevice[index_program].kernels[index_kernel], 2, NULL, globalWork, localWork, NULL, NULL, &kernelEvent), "clEnqueueNDRangeKernel");

	cl_ulong time_start, time_end;
	CL_CHECK(clWaitForEvents(1, &kernelEvent), "clWaitForEvents");
	CL_CHECK(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL), "clGetEventProfilingInfo");
	CL_CHECK(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL), "clGetEventProfilingInfo");
	const cl_ulong total_time = time_end - time_start;
	const cl_double time = cl_double(total_time) / 1000000.0;
	printf("Execution time:\t\t%0f ms\n", time);
	all_time += time;
	return true;
}
cl_bool clDevice::startCalculate(cl_uint index_program, cl_uint index_kernel, size_t globalWork[3]) {
	cl_event kernelEvent;
	//size_t x = sqrt(float(kernelInfo[index_kernel].max_work_group_size));
	//size_t y = kernelInfo[index_kernel].max_work_group_size / x;
	//size_t localWork[3] = { x, y, 1 };
	//if (globalWork[0] % localWork[0])
	//	globalWork[0] += localWork[0] - globalWork[0] % localWork[0];
	//if (globalWork[1] % localWork[1])
	//	globalWork[1] += localWork[1] - globalWork[1] % localWork[1];
	//if (globalWork[2] % localWork[2])
	//	globalWork[2] += localWork[2] - globalWork[2] % localWork[2];
	CL_CHECK(clEnqueueNDRangeKernel(*queue, programDevice[index_program].kernels[index_kernel], 2, NULL, globalWork, NULL, NULL, NULL, &kernelEvent), "clEnqueueNDRangeKernel");

	cl_ulong time_start, time_end;
	CL_CHECK(clWaitForEvents(1, &kernelEvent), "clWaitForEvents");
	CL_CHECK(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL), "clGetEventProfilingInfo");
	CL_CHECK(clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL), "clGetEventProfilingInfo");
	const cl_ulong total_time = time_end - time_start;
	const cl_double time = total_time / 1000000.0;
	printf("Execution time:\t\t%0f ms\n", time);
	all_time += time;
	return true;
}
cl_bool clDevice::copy2DImage(size_t image_src, size_t image_dst, size_t width, size_t height) {
	size_t region[] = { width, height , 1 };
	size_t  src_origin[] = { 0, 0, 0 };
	size_t dst_origin[] = { 0, 0, 0 };
	CL_CHECK(clEnqueueCopyImage(*queue, ((cl_memory_list*)image_src)->memory_data, ((cl_memory_list*)image_dst)->memory_data, src_origin, dst_origin, region, NULL, NULL, NULL), "clEnqueueCopyImage");
	return true;
}
cl_bool clDevice::copyBufferTo2DImage(size_t image, size_t buffer, size_t width, size_t height) {
	size_t region[] = { width, height , 1 };
	size_t dst_origin[] = { 0, 0, 0 };
	CL_CHECK(clEnqueueCopyBufferToImage(*queue, ((cl_memory_list*)buffer)->memory_data, ((cl_memory_list*)image)->memory_data, NULL, dst_origin, region, NULL, NULL, NULL), "clEnqueueCopyImage");

	return true;
}

cl_bool clDevice::copy2DImageToBuffer(size_t image, size_t buffer, size_t width, size_t height) {
	size_t region[] = { width, height , 1 };
	size_t  src_origin[] = { 0, 0, 0 };
	CL_CHECK(clEnqueueCopyImageToBuffer(*queue, ((cl_memory_list*)image)->memory_data, ((cl_memory_list*)buffer)->memory_data, src_origin, region, NULL, NULL, NULL, NULL), "clEnqueueCopyImage");

	return true;
}
cl_bool clDevice::writeBuffer(void* data, size_t buffer, size_t length) {
	CL_CHECK(clEnqueueWriteBuffer(*queue, ((cl_memory_list*)buffer)->memory_data, CL_FALSE, 0, length, data, NULL, NULL, NULL), "clEnqueueCopyImage");
	return true;
}
cl_bool clDevice::write2DImage(void* data, size_t image, size_t width, size_t height) {
	const size_t region[] = { width, height , 1 };
	const size_t  origin[] = { 0, 0, 0 };
	CL_CHECK(clEnqueueWriteImage(*queue, ((cl_memory_list*)image)->memory_data, CL_FALSE, origin, region, 0, 0, data, NULL, NULL, NULL), "clEnqueueCopyImage");
	return true;
}
cl_bool clDevice::readBuffer(void* returnedData, size_t memoryRead, cl_uint lengthWrite) {
	CL_CHECK(clEnqueueReadBuffer(*queue, ((cl_memory_list*)memoryRead)->memory_data, CL_TRUE, 0, lengthWrite, returnedData, NULL, NULL, NULL), "clEnqueueReadBuffer");
	return false;
}
cl_bool clDevice::readImage(void* returnedData, size_t memoryRead, size_t width, size_t height) {
	const size_t offsetImages[3] = { 0, 0, 0 };
	const size_t sizeRegion[3] = { width, height, 1 };
	CL_CHECK(clEnqueueReadImage(*queue, ((cl_memory_list*)memoryRead)->memory_data, CL_FALSE, offsetImages, sizeRegion, NULL, NULL, returnedData, NULL, NULL, NULL), "clEnqueueReadImage");
	return true;
}
cl_char* clDevice::getNameKernel(cl_uint index_program, cl_uint index_kernel) {
	return index_program < numberPrograms ? (programDevice[index_program].numberKernels > index_kernel ? programDevice[index_program].namesKernels[index_kernel] : NULL) : NULL;
}

cl_char* clDevice::getNameProgram(cl_uint index) {
	return numberPrograms > index ? namesPrograms[index] : NULL;
}
clDevice::~clDevice()
{
	free(profileVersionNameVendorExtensions[0]);
	free(profileVersionNameVendorExtensions[1]);
	free(profileVersionNameVendorExtensions[2]);
	free(profileVersionNameVendorExtensions[3]);
	free(profileVersionNameVendorExtensions[4]);
	if (headPtrMemoryDevice) {
		while (headPtrMemoryDevice->next != NULL) {
			cl_memory_list* PtrMemoryDevice = headPtrMemoryDevice;
			clReleaseMemObject(PtrMemoryDevice->memory_data);
			headPtrMemoryDevice = PtrMemoryDevice->next;
			free(PtrMemoryDevice);
		}
		if (headPtrMemoryDevice->memory_data)
			clReleaseMemObject(headPtrMemoryDevice->memory_data),
			free(headPtrMemoryDevice);
	}

	for (size_t i = 0; i < numberPrograms; i++) {
		for (size_t j = 0; j < programDevice[i].numberKernels; j++) {
			clReleaseKernel(programDevice[i].kernels[j]);
			if (programDevice[i].namesKernels[j]) free(programDevice[i].namesKernels[j]);
		}
		if (programDevice[i].kernelInfo) free(programDevice[i].kernelInfo);
		if (programDevice[i].kernels) free(programDevice[i].kernels);
		if (programDevice[i].namesKernels) free(programDevice[i].namesKernels);
	}
	for (size_t i = 0; i < numberPrograms; i++)
		clReleaseProgram(programDevice[i].programDevice),
		free(namesPrograms[i]);
	if (programDevice) free(programDevice);
	if (namesPrograms) free(namesPrograms);
}