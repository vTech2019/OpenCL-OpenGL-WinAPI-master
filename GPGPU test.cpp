// GPGPU test.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include "OpenCV.h"
#include "clDevice.h"
#include "glDevice.h"
#include "AMP_device.h"
#include "Load_Save_Images.h"
#include "WinAPI.h"
#include "Figures.h"
#include "InverseFilter.h"
#include "Wiener_filter.h"
#include "MakeNoise.h"
#include "fractalImageCompression.h"
#include "./Image Stabilization/Image_Stabilization.h"

const char convolution[] =
#include "convolution.cl"
;
const char fourier[] =
#include "fourier_transform.cl"
;
const char filters[] =
#include "filters.cl"
;
const char fractals[] =
#include "fractal.cl"
;

const char image_stabilization[] =
#include "image_stabilization.cl"
;
size_t WinAPI::numberLoopFunctions = 0;
void(**WinAPI::loopFunctions)(void* argument) = NULL;
void** WinAPI::ptrArgument = NULL;
int main()
{
	cpuDevice cpu_device;
	//WinAPI window_api;
	//window_api.InitWindow(L"Window");
	//window_api.InitOpenGLContext();
	//glDevice gl_device;
	//window_api.pushFunction(&Draw, NULL);
	clPlatform platform;
	clDevice** cl_devices = (clDevice**)_aligned_malloc(platform.getNumberDevices() * sizeof(clDevice*), alignof(clDevice*));
	for (size_t i = 0; i < platform.getNumberDevices(); i++) {
		cl_devices[i] = new clDevice(&platform, i);
		//cl_devices[i]->clPushProgram((cl_char*)convolution, sizeof(convolution), (cl_char*)"-cl-unsafe-math-optimizations -cl-fast-relaxed-math");
		//cl_devices[i]->clPushProgram((cl_char*)fourier, sizeof(fourier), (cl_char*)"-cl-unsafe-math-optimizations -cl-fast-relaxed-math");
		//cl_devices[i]->clPushProgram((cl_char*)filters, sizeof(filters), NULL);
		cl_devices[i]->clPushProgram((cl_char*)image_stabilization, sizeof(image_stabilization), NULL);
	}
	

	std::vector<std::wstring> dir_images = WinAPI::GetNameFolderFiles(L"Images");
	//dir_images.resize(30);
	std::vector<dataImage> images(dir_images.size());
	for (size_t i = 0; i < dir_images.size(); i++)
		images[i] = WIN_load_image(dir_images[i].c_str(), false);
	//AMP_device amp_device;
	size_t i = 1;
	//for (int i = 0; i< platform.getNumberDevices(); i++){
	if (i < platform.getNumberDevices()) {
		size_t width = images[0].width;
		size_t height = images[0].height;
		Image_Stabilization image(cl_devices[i], width, height, 7, 7, 10, 10, 1);
		for (size_t i = 0; i < images.size() -1; i++) {
			image.Stabilization_function(images[0].data, images[0].data);
		}
	}
	for (size_t i = 0; i < dir_images.size();i++) {
		size_t start = dir_images[i].size() - 1;
		std::wstring name;
		while (dir_images[i][start - 1] != L'//'&&start!=0)
			start--;
		while (start != dir_images[i].size() && dir_images[i][start] != '.')
			name.push_back(dir_images[i][start++]);
		name.push_back(0);
		WIN_save_image(images[i], L"Result//", name.c_str(), PNG);
	}
//	window_api.DispatchMessageWindow();
	for (size_t i = 0; i < platform.getNumberDevices(); i++) 
		delete cl_devices[i];
	_aligned_free(cl_devices);
}