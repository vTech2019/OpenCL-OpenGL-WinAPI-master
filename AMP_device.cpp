#include "AMP_device.h"

AMP_device* AMP_device::operator[](size_t i)
{

	concurrency::accelerator::set_default(accelerators.at(i).device_path);
	return this;
}

AMP_device::AMP_device() {
	accelerators = concurrency::accelerator::get_all();


	for (size_t i = 0; i < accelerators.size(); i++)
	{
		std::wcout << "New accelerator: " << accelerators[i].description << std::endl;
		std::wcout << "is_debug = " << accelerators[i].is_debug << std::endl;
		std::wcout << "is_emulated = " << accelerators[i].is_emulated << std::endl;
		std::wcout << "dedicated_memory = " << accelerators[i].dedicated_memory << std::endl;
		std::wcout << "device_path = " << accelerators[i].device_path << std::endl;
		std::wcout << "has_display = " << accelerators[i].has_display << std::endl;
		std::wcout << "version = " << (accelerators[i].version >> 16) << '.' << (accelerators[i].version & 0xFFFF) << std::endl;
		std::wcout << "CPU = " << (std::wstring(accelerators[i].cpu_accelerator).compare(accelerators[i].device_path) ? L"false" : L"true") << std::endl;
	}

}
AMP_device::~AMP_device() {
}
