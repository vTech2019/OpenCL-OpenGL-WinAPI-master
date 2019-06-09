#include "AMP_device.h"
using namespace concurrency;
using namespace concurrency::graphics;

void AMP_device::gauss_function(uint32_t width, uint32_t height, uint32_t block_x, uint32_t block_y, uint32_t step_x, uint32_t step_y, uchar4* image, uchar4* result)
{
	extent<2> image_extent(height, width);
	texture<unorm_4, 2> texture_2D(image_extent, (unsigned int*)image, width * height * sizeof(uchar4), 8U);
	texture_view<const unorm_4, 2> texture_view_2D(texture_2D);
	array_view<uint32_t, 2> array_2D(image_extent, (unsigned int*)result);
	array_2D.discard_data();
	accelerator_view acc_view = accelerator(accelerator::direct3d_ref).default_view;
	try {
		parallel_for_each(acc_view, array_2D.extent,	[=](concurrency::index<2> idx) restrict(amp) {

				float_4 rgba_image = (255.0f * (float_4)texture_view_2D[idx]);
				uint32_t data = uint32_t(rgba_image.a) & 255;
				data <<= 8;
				data += (uint32_t(rgba_image.r) & 255);
				data <<= 8;
				data += (uint32_t(rgba_image.g) & 255);
				data <<= 8;
				data += (uint32_t(rgba_image.b) & 255);
				array_2D[idx] = data;
			}
		);
	}
	catch(concurrency::accelerator_view_removed& event){
		printf("TDR exception received: %s\n Error code: %d %d", event.what(), event.get_error_code(), event.get_view_removed_reason());
	}
	try {
		array_2D.synchronize();
	}
	catch (const Concurrency::accelerator_view_removed& event) {
		printf("%s\n", event.what());
	}
}