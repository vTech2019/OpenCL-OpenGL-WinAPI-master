#pragma once

#include <stdio.h>
#include <Windows.h>
#include <mfreadwrite.h>
#include <mfapi.h>
#include <shlwapi.h>
#include <Mfidl.h>
#include <fileapi.h>
#pragma comment(lib,"Mfplat.lib")
#pragma comment(lib,"Mf.lib")
#pragma comment(lib,"Mfreadwrite.lib")
#pragma comment(lib,"mfuuid.lib")
#pragma comment(lib,"shlwapi.lib")

#ifdef ENABLE_DirectShow
#include <Dshow.h>
#include <qedit.h>
#pragma comment(lib, "Strmiids")
#pragma comment(lib, "Quartz")

struct VideoDevice {
	CHAR* friendlyName;
	WCHAR* filterName;
	IBaseFilter* filter;
};
struct DirectShowData {
	IFilterGraph2* pGraph;
	ICaptureGraphBuilder2* pBuild;
	IMediaControl* mediaControl;
	ICreateDevEnum* pDevEnum;
	IEnumMoniker* pEnum;
	IBaseFilter* pCap;
	VideoDevice* devices;
	size_t numberDevices;
	size_t indexCurrentDevices;
};
namespace DirectShow {
	DirectShowData InitCaptureGraph();
	DirectShowData SetSourceFilters(DirectShowData data);
	void DirectShow::SetSampleGrabber(DirectShowData data, size_t index_device);
}
#endif

namespace MMF {
	class Camera;
	class Media {
		IMFAttributes* pAttributes;
		IMFActivate** ppDevices;
		IMFMediaType*** ppType;
		IMFMediaSource** ppSource;
		UINT32  number_devices = NULL;
	public:
		Media();
		~Media();
		void CreateVideoDeviceSource();
		void GetCaptureFormats();
		HRESULT SetDeviceFormat(IMFMediaSource* pSource, DWORD dwFormatIndex);
		HRESULT SetMaxFrameRate(IMFMediaSource* pSource, DWORD dwTypeIndex);
		HRESULT CreateCaptureDevice(); 
		Camera* SetSourceReader(UINT index_device);
	};
	class Camera : public IMFSourceReaderCallback {
		friend Media;
		LPCRITICAL_SECTION criticalSection;
		IMFSourceReader* sourceReader;
		LONG referenceCount;
		UINT width;
		UINT height;
		UINT bytesPerPixel;
		STDMETHODIMP QueryInterface(REFIID riid, void** ppv);
		STDMETHODIMP_(ULONG) AddRef();
		STDMETHODIMP_(ULONG) Release();

		//  the class must implement the methods from IMFSourceReaderCallback
		STDMETHODIMP OnReadSample(HRESULT status, DWORD streamIndex, DWORD streamFlags, LONGLONG timeStamp, IMFSample* sample);
		STDMETHODIMP OnEvent(DWORD, IMFMediaEvent*);
		STDMETHODIMP OnFlush(DWORD);
	};
}