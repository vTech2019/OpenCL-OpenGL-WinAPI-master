#pragma once
#include "WinAPI.h"
#include "glDevice.h"
#include "clDevice.h"
#include "AMP_device.h"

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

class Init_graphics_libraries
{
	static void(*loopFunctions)(WCHAR* information, HWND hwnd);
	WinAPI main_window;
	glDevice* gl_device;
	clPlatform* cl_platform;
	clDevice** cl_devices;
	AMP_device* amp_device;
	bool init_clDevice();
	bool init_glDevice();
public:
	clDevice* get_clDevice(size_t i);
	size_t get_number_clDevice();
	Init_graphics_libraries();
	~Init_graphics_libraries();
};

