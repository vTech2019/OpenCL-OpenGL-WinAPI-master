#include "WebCamera.h"

#ifdef ENABLE_DirectShow
DirectShowData DirectShow::InitCaptureGraph()
{
	DirectShowData data = { NULL };
	if (SUCCEEDED(CoCreateInstance(CLSID_FilterGraph, NULL, CLSCTX_INPROC_SERVER, IID_IFilterGraph2, (void**)& data.pGraph))) {
		if (SUCCEEDED(CoCreateInstance(CLSID_CaptureGraphBuilder2, NULL, CLSCTX_INPROC_SERVER, IID_ICaptureGraphBuilder2, (void**)& data.pBuild))) {
			if (SUCCEEDED(data.pGraph->QueryInterface(IID_IMediaControl, (void**)& data.mediaControl))) {
				if (SUCCEEDED(data.pBuild->SetFiltergraph(data.pGraph))) {
					return data;
				}
				else
					data.pGraph->Release(),
					data.pBuild->Release(),
					data.mediaControl->Release();
			}
			else
				data.pGraph->Release(),
				data.pBuild->Release();
		}
		else
			data.pGraph->Release();
	}
	data = { NULL };
}
DirectShowData DirectShow::SetSourceFilters(DirectShowData data)
{
	VARIANT name;
	IMoniker* moniker;
	IPropertyBag* property;

	if (SUCCEEDED(CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER, IID_ICreateDevEnum, (void**)& data.pDevEnum))) {
		if (SUCCEEDED(data.pDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &data.pEnum, NULL))) {
			data.numberDevices = NULL;
			data.devices = NULL;
			while (S_OK == data.pEnum->Next(1, &moniker, NULL)) {
				if (SUCCEEDED(moniker->BindToStorage(0, 0, IID_IPropertyBag, (void**) & property))) {
					data.numberDevices++;
					data.devices = (VideoDevice*)realloc(data.devices, data.numberDevices * sizeof(VideoDevice));
					VariantInit(&name);
					if (FAILED(property->Read(L"Description", &name, 0)))
						if (FAILED(property->Read(L"FriendlyName", &name, 0))) {
							data.numberDevices--;
							data.devices = (VideoDevice*)realloc(data.devices, data.numberDevices * sizeof(VideoDevice));
							VariantClear(&name);
							return data;
						}
					printf("%S\n", name.bstrVal);
					BSTR ptr = name.bstrVal;
					for (UINT c = 0; *ptr; c++, ptr++)
					{
						data.devices->filterName[c] = *ptr;
						data.devices->friendlyName[c] = *ptr & 0xFF;
					}
					if (FAILED(data.pGraph->AddSourceFilterForMoniker(moniker, 0, data.devices->filterName, &data.devices->filter))) {
						data.numberDevices--;
						data.devices = (VideoDevice*)realloc(data.devices, data.numberDevices * sizeof(VideoDevice));
					}
					VariantClear(&name);
					property->Release();
				}
				moniker->Release();
			}
		}
	}
	return data;
}

void DirectShow::SetSampleGrabber(DirectShowData data, size_t index_device)
{
	HRESULT hr;

	if (SUCCEEDED(CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER, IID_IBaseFilter, (void**)& data.devices[index_device].filter))) {
		if (S_OK != data.pGraph->AddFilter(data.devices[index_device].filter, L"Sample Grabber")) {
			if (S_OK != data.devices[index_device].filter->QueryInterface(IID_ISampleGrabber, (void**)& data.devices[index_device].filter)) {
			}
		}
	}
}
#endif

using namespace MMF;
void ReleaseSource(IMFMediaSource* pSource, IMFAttributes* pAttributes, IMFActivate** ppDevices, UINT32  number_devices)
{
	for (UINT32 i = 0; i < number_devices; i++)
	{
		if (&ppDevices[i]) ppDevices[i]->Release();
	}
	CoTaskMemFree(ppDevices);
	if (pAttributes) pAttributes->Release();
	if (pSource) pSource->Release();
}

Media::Media()
{
	CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
	MFStartup(MF_VERSION);
}


void Media::CreateVideoDeviceSource()
{
	if (SUCCEEDED(MFCreateAttributes(&pAttributes, 1))) {
		if (SUCCEEDED(pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID))) {
			if (SUCCEEDED(MFEnumDeviceSources(pAttributes, &ppDevices, &number_devices))) {
				ppSource = (IMFMediaSource * *)calloc(number_devices, sizeof(IMFMediaSource*));
				for (size_t i = 0; i < number_devices; i++) {
					if (SUCCEEDED(ppDevices[i]->ActivateObject(IID_PPV_ARGS(&ppSource[i])))) {
						ppSource[i]->AddRef();
						WCHAR* nameString = NULL;
						UINT32 cchName;
						if (SUCCEEDED(ppDevices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &nameString, &cchName))) {
							wprintf(L"Index device: %zd. Name device: %ls \n", i, nameString);
							
						}
						CoTaskMemFree(nameString);
					}
				}
			}
		}
	}
}

void Media::GetCaptureFormats()
{
	IMFPresentationDescriptor* pPD = NULL;
	IMFStreamDescriptor* pSD = NULL;
	IMFMediaTypeHandler* pHandler = NULL;
	BOOL fSelected;
	DWORD cTypes = 0;
	UINT width, height;
	UINT numerator, denomirator;
	LONG stride;
	GUID subtype = GUID_NULL;
	for (DWORD i = 0; i < number_devices; i++) {
		if (SUCCEEDED(ppSource[i]->CreatePresentationDescriptor(&pPD))) {
			if (SUCCEEDED(pPD->GetStreamDescriptorByIndex(0, &fSelected, &pSD))) {
				if (SUCCEEDED(pSD->GetMediaTypeHandler(&pHandler))) {
					if (SUCCEEDED(pHandler->GetMediaTypeCount(&cTypes))) {
						ppType[i] = (IMFMediaType * *)calloc(cTypes, sizeof(IMFMediaType*));
						for (DWORD j = 0; j < cTypes; j++) {
							if (SUCCEEDED(pHandler->GetMediaTypeByIndex(j, &ppType[i][j]))) {
								if (SUCCEEDED(MFGetAttributeSize(ppType[i][j], MF_MT_FRAME_SIZE, &width, &height))) {
									if (SUCCEEDED(MFGetAttributeRatio(ppType[i][j], MF_MT_FRAME_RATE, &numerator, &denomirator))) {
										if (SUCCEEDED(ppType[i][j]->GetUINT32(MF_MT_DEFAULT_STRIDE, (UINT32*)& stride))) {
											if (SUCCEEDED(MFGetStrideForBitmapInfoHeader(subtype.Data1, width, &stride))) {
												ppType[i][j]->SetUINT32(MF_MT_DEFAULT_STRIDE, UINT32(stride));
												printf("Index device: %d: width = %d; height = %d; ratio = %d / %d; stride = %d \n", j, width, height, numerator, denomirator, stride);
											}
										}
									}
								}
							}
						}
					}
				}
				if (&pHandler)pHandler->Release();
			}
			if (&pSD)pSD->Release();
		}
		if (&pPD)pPD->Release();
	}
}

HRESULT Media::SetDeviceFormat(IMFMediaSource* pSource, DWORD dwFormatIndex)
{
	IMFPresentationDescriptor* pPD = NULL;
	IMFStreamDescriptor* pSD = NULL;
	IMFMediaTypeHandler* pHandler = NULL;
	IMFMediaType* pType = NULL;
	BOOL fSelected;
	HRESULT result = E_NOTIMPL;
	if (SUCCEEDED(pSource->CreatePresentationDescriptor(&pPD))) {
		if (SUCCEEDED(pPD->GetStreamDescriptorByIndex(0, &fSelected, &pSD))) {
			if (SUCCEEDED(pSD->GetMediaTypeHandler(&pHandler))) {
				if (SUCCEEDED(pHandler->GetMediaTypeByIndex(dwFormatIndex, &pType))) {
					result = pHandler->SetCurrentMediaType(pType);
				}
			}
		}
	}
	if (&pPD)pPD->Release();
	if (&pSD)pSD->Release();
	if (&pHandler)pHandler->Release();
	if (&pType)pType->Release();
	return result;
}

HRESULT Media::SetMaxFrameRate(IMFMediaSource* pSource, DWORD dwTypeIndex)
{
	IMFPresentationDescriptor* pPD = NULL;
	IMFStreamDescriptor* pSD = NULL;
	IMFMediaTypeHandler* pHandler = NULL;
	IMFMediaType* pType = NULL;
	BOOL fSelected;
	PROPVARIANT var;
	HRESULT result = E_NOTIMPL;
	if (SUCCEEDED(pSource->CreatePresentationDescriptor(&pPD))) {
		if (SUCCEEDED(pPD->GetStreamDescriptorByIndex(dwTypeIndex, &fSelected, &pSD))) {
			if (SUCCEEDED(pSD->GetMediaTypeHandler(&pHandler))) {
				if (SUCCEEDED(pHandler->GetCurrentMediaType(&pType))) {
					if (SUCCEEDED(pType->GetItem(MF_MT_FRAME_RATE_RANGE_MAX, &var))) {
						if (SUCCEEDED(pType->SetItem(MF_MT_FRAME_RATE, var))) {
							result = pHandler->SetCurrentMediaType(pType);
						}
						PropVariantClear(&var);
					}
				}
			}
		}
	}
	if (&pPD)pPD->Release();
	if (&pSD)pSD->Release();
	if (&pHandler)pHandler->Release();
	if (&pType)pType->Release();
	return result;
}

HRESULT Media::CreateCaptureDevice()
{
	HRESULT hr = S_OK;
	UINT32 count = 0;
	IMFAttributes* attributes = NULL;
	IMFActivate** devices = NULL;
	if (SUCCEEDED(CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE))) {
		if (SUCCEEDED(MFCreateAttributes(&attributes, 1))) {
		}
	}

	return E_NOTIMPL;
}

Camera* Media::SetSourceReader(UINT index_device)
{
	HRESULT hr = S_OK;
	WCHAR* wSymbolicLink;
	UINT32 cchSymbolicLink;
	IMFAttributes* videoConfig = NULL;
	IMFMediaType* mediaType = NULL;
	Camera* media = NULL;
	if (number_devices > index_device) {
		if (SUCCEEDED(ppDevices[index_device]->ActivateObject(__uuidof(IMFMediaSource), (void**)& ppSource[index_device]))) {
			if (SUCCEEDED(ppDevices[index_device]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &wSymbolicLink, &cchSymbolicLink))) {
				if (SUCCEEDED(MFCreateAttributes(&videoConfig, 2))) {
					if (SUCCEEDED(videoConfig->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE))) {
						media = (Camera*)malloc(sizeof(Camera));
						if (SUCCEEDED(videoConfig->SetUnknown(MF_SOURCE_READER_ASYNC_CALLBACK, (IUnknown*)media))) {
							IMFSourceReader* sourceReader;
							if (SUCCEEDED(MFCreateSourceReaderFromMediaSource(ppSource[index_device], videoConfig, &sourceReader))) {
								sourceReader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &mediaType);

								media->sourceReader = sourceReader;
							}
						}
					}
				}
			}
			CoTaskMemFree(wSymbolicLink);
		}
	}
	if (videoConfig) videoConfig->Release();
	if (mediaType) mediaType->Release();
	return media;
}
STDMETHODIMP_(HRESULT __stdcall) Camera::QueryInterface(REFIID riid, void** ppv)
{
	static const QITAB qit[] = { QITABENT(Media, IMFSourceReaderCallback),{ 0 }, };
	return QISearch(this, qit, riid, ppv);
}
STDMETHODIMP_(ULONG __stdcall) Camera::AddRef()
{
	return InterlockedIncrement(&referenceCount);
}
STDMETHODIMP_(ULONG __stdcall) Camera::Release()
{
	long cRef = InterlockedDecrement(&referenceCount);
	if (cRef == 0)
		delete this;
	return cRef;
}
STDMETHODIMP_(HRESULT __stdcall) Camera::OnReadSample(HRESULT status, DWORD streamIndex, DWORD streamFlags, LONGLONG timeStamp, IMFSample* sample)
{
	IMFMediaBuffer* mediaBuffer = NULL;
	IMFSample* videoSample = NULL;
	LONGLONG llVideoTimeStamp;
	EnterCriticalSection(criticalSection);

	sourceReader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, NULL, NULL, NULL, &videoSample);

	videoSample->SetSampleTime(llVideoTimeStamp);
	if (SUCCEEDED(status)) {
		if (SUCCEEDED(sample->GetBufferByIndex(0, &mediaBuffer))) {
			BYTE* data;
			mediaBuffer->Lock(&data, NULL, NULL);
			//ColorConversion(data);
			//Draw(data)
//CopyMemory(rawData, data, width * height * bytesPerPixel);
		}
	}

	if (mediaBuffer)	mediaBuffer->Release();
	if (videoSample) videoSample->Release();
	LeaveCriticalSection(criticalSection);
	return E_NOTIMPL;
}
STDMETHODIMP_(HRESULT __stdcall) Camera::OnEvent(DWORD, IMFMediaEvent*)
{
	return S_OK;
}
STDMETHODIMP_(HRESULT __stdcall) Camera::OnFlush(DWORD)
{
	return S_OK;
}