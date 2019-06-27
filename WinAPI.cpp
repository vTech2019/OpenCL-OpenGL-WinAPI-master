#include "WinAPI.h"

void WinAPI::getScreenSize(size_t& width, size_t& height)
{
	RECT rect;
	if (GetWindowRect(GetDesktopWindow(), &rect)) {
		width = rect.right - rect.left;
		height = rect.bottom - rect.top;
	}
}

void WinAPI::getWindowSize(size_t& width, size_t& height)
{
	RECT rect;
	if (GetWindowRect(main_window, &rect)) {
		width = rect.right - rect.left;
		height = rect.bottom - rect.top;
	}
}
void WinAPI::hideWindow()
{
	ShowWindow(main_window, SW_HIDE);
}

void WinAPI::pushFunction(void(*ptrLoopFunctions)(void* argument), void* argument)
{
	numberLoopFunctions++;
	void** memoryPtrFunctions = (void**)realloc(loopFunctions, numberLoopFunctions * sizeof(void(*)(void*)));
	loopFunctions = (void(**)(void* argument))memoryPtrFunctions;
	ptrArgument = (void**)realloc(ptrArgument, numberLoopFunctions * sizeof(void*));
	loopFunctions[numberLoopFunctions - 1] = ptrLoopFunctions;
	ptrArgument[numberLoopFunctions - 1] = argument;
}

std::vector<std::wstring> WinAPI::GetNameFolderFiles(std::wstring way) {
	WIN32_FIND_DATA FileData;
	HANDLE hFind;
	std::vector<std::wstring> fileName;
	way.append(L"//*");
	hFind = FindFirstFile(way.data(), &FileData);
	way.pop_back();
	do
	{
		if (!(FileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
		{
			fileName.push_back(way + FileData.cFileName);
		}
	} while (FindNextFile(hFind, &FileData));
	FindClose(hFind);
	return fileName;
}

void WinAPI::InitWindow(wchar_t* name_window, size_t width, size_t height, size_t position_x, size_t position_y, unsigned long parameters_window) {
	HINSTANCE hInstance = GetModuleHandle(NULL);
	if (!RegisterClass(&window_class)) {
		window_class.style = CS_HREDRAW | CS_VREDRAW;
		window_class.cbClsExtra = 0;
		window_class.cbWndExtra = 0;
		window_class.lpszClassName = name_window;
		window_class.hInstance = hInstance;
		window_class.hbrBackground = GetSysColorBrush(COLOR_3DFACE + 1);
		window_class.lpszMenuName = NULL;
		window_class.lpfnWndProc = WndProc;
		window_class.hCursor = LoadCursor(NULL, IDC_ARROW);
		window_class.hIcon = LoadIcon(NULL, IDI_APPLICATION);
		if (!RegisterClass(&window_class))
		{
			HWND hWnd = GetForegroundWindow();
			MessageBox(hWnd, L"Problem with WNDCLASS!", L"Problem with WNDCLASS!", MB_OK);
		}
		RegisterClassW(&window_class);
	}

	main_window = CreateWindowW(window_class.lpszClassName, name_window, parameters_window, position_x - width / 2, position_y - height / 2, width, height, NULL, NULL, hInstance, NULL);
}

void WinAPI::InitOpenGLContext(HGLRC& gl_context) {
	if (gl_context != NULL) {
		wglMakeCurrent(NULL, NULL);
		wglDeleteContext(gl_context);
	}
	HDC hdc = GetDC(main_window);
	PIXELFORMATDESCRIPTOR pfd;
	ZeroMemory(&pfd, sizeof(pfd));
	pfd.nSize = sizeof(pfd);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 24;
	pfd.cDepthBits = 24;
	pfd.iLayerType = PFD_MAIN_PLANE;
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);

	gl_context = wglCreateContext(hdc);
	wglMakeCurrent(hdc, gl_context);
}
void WinAPI::updateWindow() {
	UpdateWindow(main_window);
}
void WinAPI::DispatchMessageWindow() {
	if (!main_window)
		return;
	ShowWindow(main_window, SW_SHOW);
	UpdateWindow(main_window);
	while (GetMessage(&msg, NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}
void WinAPI::PushFunction(void(*ptrLoopFunctions)(void* argument), void* argument)
{
	numberLoopFunctions++;
	void** memoryPtrFunctions = (void**)realloc(loopFunctions, numberLoopFunctions * sizeof(void(*)(void*)));
	loopFunctions = (void(**)(void* argument))memoryPtrFunctions;
	ptrArgument = (void**)realloc(ptrArgument, numberLoopFunctions * sizeof(void*));
	loopFunctions[numberLoopFunctions - 1] = ptrLoopFunctions;
	ptrArgument[numberLoopFunctions - 1] = argument;
}

WinAPI::WinAPI()
{
	memset(this, 0, sizeof(*this));
}

WinAPI::~WinAPI()
{
}

LRESULT CALLBACK WinAPI::WndProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
	LONG_PTR winptr = GetWindowLongPtr(hwnd, GWLP_USERDATA);
	HDC hDC;
	PAINTSTRUCT ps;
	switch (message)
	{
	case WM_CREATE:
		break;
	case WM_PAINT:
		hDC = BeginPaint(hwnd, &ps);
		for (size_t i = 0; i < numberLoopFunctions; i++) {
			loopFunctions[i](ptrArgument[i]);
		}
		SwapBuffers(hDC);
		EndPaint(hwnd, &ps);
		break;
	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	}
	return DefWindowProc(hwnd, message, wparam, lparam);
}