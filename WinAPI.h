#pragma once
#define UNICODE
//#include "glDevice.h"
#include <windows.h>
#include <vector>
#include <string>
class WinAPI
{

	WNDCLASSW window_class;
	HWND main_window;
	MSG  msg;
	size_t numberLoopFunctions;
	void(**loopFunctions)(void* argument);
	void** ptrArgument;
public:
	void updateWindow();
	void getScreenSize(size_t& width, size_t& height);
	void getWindowSize(size_t& width, size_t& height);
	void hideWindow();
	void pushFunction(void(*ptrLoopFunctions)(void *argument), void * argument);
	void InitWindow(wchar_t * name_window, size_t width, size_t height, size_t position_x, size_t position_y, unsigned long parameters_window);
	void InitOpenGLContext(HGLRC& gl_context);
	void DispatchMessageWindow();
	WinAPI();
	~WinAPI(); 
	//void pushFunction(void(glDevice::*ptrLoopFunctions)(void *argument), void *argument);
	void PushFunction(void(*ptrLoopFunctions)(void *argument), void * argument);
	static std::vector<std::wstring> GetNameFolderFiles(std::wstring way);
	static LRESULT WndProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam);
};

