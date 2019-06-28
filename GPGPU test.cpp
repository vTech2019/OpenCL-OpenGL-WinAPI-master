// GPGPU test.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include "Init_graphics_libraries.h"
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


size_t WinAPI::numberLoopFunctions = 0;
void(**WinAPI::loopFunctions)(void* argument) = NULL;
void** WinAPI::ptrArgument = NULL;
int main()
{
	cpuDevice cpu_device;
	Init_graphics_libraries graphics_data;
	WinAPI window_api;
	
	

	std::vector<std::wstring> dir_images = WinAPI::GetNameFolderFiles(L"Images");
	//dir_images.resize(30);
	std::vector<dataImage> images(dir_images.size());
	for (size_t i = 0; i < dir_images.size(); i++)
		images[i] = WIN_load_image(dir_images[i].c_str(), false);
	size_t i = 0;
	//for (int i = 0; i< platform.getNumberDevices(); i++){
		size_t width = images[0].width;
		size_t height = images[0].height;
		Image_Stabilization image(graphics_data.get_clDevice(i), width, height, 50, 50, 10, 10, 10);
		for (size_t i = 0; i < images.size() -1; i++) {
			image.Stabilization_function(images[0].data, images[i].data);
		}
 		printf("Execution time:\t\t%0f ms\n", graphics_data.get_clDevice(i)->all_time);
	//}
		
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
}