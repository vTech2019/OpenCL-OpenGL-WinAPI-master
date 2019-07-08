#include "Init_graphics_libraries.h"

void printf_information(TCHAR* information, HWND hwnd) {
	if (information) {
		HDC hDC = GetDC(hwnd);
		RECT rect;
		GetClientRect(hwnd, &rect);
		COLORREF colorText = RGB(128, 255, 255);
		SetTextColor(hDC, colorText);
		DrawText(hDC, information, -1, &rect, DT_SINGLELINE | DT_CENTER | DT_VCENTER);
	}
}

bool Init_graphics_libraries::init_clDevice()
{
	cl_platform = (clPlatform*)new clPlatform();
	cl_devices = (clDevice * *)malloc(sizeof(clDevice * *) * cl_platform->getNumberDevices());
	for (size_t i = 0; i < cl_platform->getNumberDevices(); i++) {
		cl_devices[i] = new clDevice(cl_platform, i);
		cl_devices[i]->clPushProgram((cl_char*)image_stabilization, sizeof(image_stabilization), (cl_char*)"-cl-unsafe-math-optimizations -cl-fast-relaxed-math");
	}
	return true;
}

bool Init_graphics_libraries::init_glDevice()
{
	gl_device = new glDevice();
	return true;
}

clDevice * Init_graphics_libraries::get_clDevice(size_t i)
{
	if (i < cl_platform->getNumberDevices()) {
		return cl_devices[i];
	}
	return nullptr;
}
size_t Init_graphics_libraries::get_number_clPlatform() {
	if (cl_platform)
		return cl_platform->getNumberPlatforms();
	else
		return NULL;
}
size_t Init_graphics_libraries::get_number_clDevice()
{
	if (cl_platform)
		return cl_platform->getNumberDevices();
	else
		return NULL;
}

Init_graphics_libraries::Init_graphics_libraries()
{
	size_t width, height;
	WinAPI information_window;
	information_window.getScreenSize(width, height);
	information_window.InitWindow(L"TOOLWINDOW", 400, 600, width / 2, height / 2, WS_VISIBLE | WS_POPUP | WS_EX_TOPMOST | WS_EX_TOOLWINDOW);
	information_window.updateWindow();
	for (size_t i = 0; i < sizeof * this; i++)
		((char*)this)[i] = 0;
	init_clDevice();
	init_glDevice();
}

Init_graphics_libraries::~Init_graphics_libraries()
{
	if (gl_device)
		delete gl_device;

	if (cl_devices) {
		for (size_t i = 0; i < cl_platform->getNumberDevices(); i++)
			delete cl_devices[i];
	}
	free(cl_devices);
	if (cl_platform)
		delete cl_platform;
	if (amp_device)
		delete amp_device;
}