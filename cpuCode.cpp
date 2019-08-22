#pragma once
#include "cpuDevice.h"

__m128 _ExpSse(__m128 x)
{
	const __m128 a = _mm_set1_ps((1 << 22) / 0.693147180559945309417f);
	const __m128i b = _mm_set1_epi32(127 * (1 << 23));
	__m128i r = _mm_cvtps_epi32(_mm_mul_ps(a, x));
	__m128i s = _mm_add_epi32(b, r);
	__m128i t = _mm_sub_epi32(b, r);
	return _mm_div_ps(_mm_castsi128_ps(s), _mm_castsi128_ps(t));
}
void cpuDevice::cpu_sse2_Gauss_function(float4* data, uchar4* result, size_t width_image, size_t height_image, float part_block_x, float part_block_y, float pow_sigma, size_t pitch_width_image, size_t offset_radius_aligned, size_t block_x, size_t block_y)
{
	float4* ptr_memory_image = data;
	uchar4* ptr_image = (uchar4*)result;
	size_t pitch_width = pitch_width_image;
	size_t width = width_image;
	size_t height = height_image;
	size_t last_data = width % 4;
	float normalize = block_x * block_y;
	static const __m128 sigma_xmm = _mm_set1_ps(pow_sigma);
	static const __m128 sigma_pi_xmm = _mm_set1_ps(pow_sigma * 3.1415926535f);
	static const __m128 normalize_xmm = _mm_set1_ps(normalize);
	static const __m128 part_block_x_xmm = _mm_set1_ps(part_block_x);
	static const __m128 part_block_y_xmm = _mm_set1_ps(part_block_y);
	static const __m128 magic_float = _mm_set1_ps(8388608.0f);
	static const __m128i magic_int = _mm_set1_epi16(0x4B00);
	for (size_t h = 0; h < height; h++) {
		float* ptr_vector_result = (float*)(ptr_memory_image + h * pitch_width + offset_radius_aligned);
		__m128i* ptr_vector_image = (__m128i*)(ptr_image + h * width_image);
		const __m128 intex_y = _mm_set1_ps(float(h % block_y));
		for (size_t w = 0; w < width;) {
			__m128 _y = _mm_add_ps(part_block_y_xmm, intex_y);
			__m128 _x = _mm_add_ps(part_block_x_xmm, _mm_set_ps(w++ % block_x, w++ % block_x, w++ % block_x, w++ % block_x));
			__m128 _div = _mm_xor_ps(_mm_div_ps(_mm_add_ps(_mm_mul_ps(_x, _x), _mm_mul_ps(_y, _y)), sigma_xmm), _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));

			__m128 _gauss = _mm_mul_ps(_mm_div_ps(_ExpSse(_div), sigma_pi_xmm), normalize_xmm);
			__m128i xmm0 = _mm_loadu_si128(ptr_vector_image++);
			__m128i xmm1 = _mm_unpacklo_epi8(xmm0, _mm_setzero_si128());
			__m128i xmm2 = _mm_unpackhi_epi8(xmm0, _mm_setzero_si128());
			__m128i xmm3 = _mm_unpacklo_epi16(xmm1, magic_int);
			__m128i xmm4 = _mm_unpackhi_epi16(xmm1, magic_int);
			__m128i xmm5 = _mm_unpacklo_epi16(xmm2, magic_int);
			__m128i xmm6 = _mm_unpackhi_epi16(xmm2, magic_int);
			__m128 f_xmm3 = _mm_sub_ps(_mm_castsi128_ps(xmm3), magic_float);
			__m128 f_xmm4 = _mm_sub_ps(_mm_castsi128_ps(xmm4), magic_float);
			__m128 f_xmm5 = _mm_sub_ps(_mm_castsi128_ps(xmm5), magic_float);
			__m128 f_xmm6 = _mm_sub_ps(_mm_castsi128_ps(xmm6), magic_float);
			__m128 f_xmm7 = _mm_unpacklo_ps(_gauss, _gauss);
			__m128 f_xmm8 = _mm_unpacklo_ps(_gauss, _gauss);
			__m128 f_xmm9 = _mm_unpackhi_ps(_gauss, _gauss);
			__m128 f_xmm10 = _mm_unpackhi_ps(_gauss, _gauss);
			_mm_store_ps(ptr_vector_result, _mm_mul_ps(f_xmm3, _mm_unpacklo_ps(f_xmm7, f_xmm7)));
			_mm_store_ps(ptr_vector_result + 4, _mm_mul_ps(f_xmm4, _mm_unpackhi_ps(f_xmm8, f_xmm8)));
			_mm_store_ps(ptr_vector_result + 8, _mm_mul_ps(f_xmm5, _mm_unpacklo_ps(f_xmm9, f_xmm9)));
			_mm_store_ps(ptr_vector_result + 12, _mm_mul_ps(f_xmm6, _mm_unpackhi_ps(f_xmm10, f_xmm10)));
			ptr_vector_result += 16;
		}
	}
	//for (size_t i = 0; i < height; i++) {
	//	float* ptr_vector_result = (float*)(ptr_memory_image + i * pitch_width + _cpu_data->offset_radius_aligned);
	//	for (size_t j = 0; j < width; j++) {
	//		ptr_image[i*width + j].x = *(ptr_vector_result++);
	//		ptr_image[i*width + j].y = *(ptr_vector_result++);
	//		ptr_image[i*width + j].z = *(ptr_vector_result++);
	//		ptr_image[i*width + j].w = *(ptr_vector_result++);
	//	}
	//}
}

float gauss_function(float x, float y, float sigma) {
	const float _y = y * y;
	const float _x = x * x;
	const float _sigma = 2.0f*sigma*sigma;
	const float div = (_y + _x) / _sigma;
	return expf(-div) / (_sigma * M_PI);
}

void cpuDevice::cpu_SSE2_Stabilization_function(uchar4* current_image, 
	uchar4* next_image, int2* sync_data, size_t width, size_t height, 
	size_t block_x, size_t block_y, size_t step_x, size_t step_y, size_t radius ) {
	size_t number_block_x = width / block_x;
	size_t number_block_y = height / block_y;
	size_t end_block_x = number_block_x ;
	size_t end_block_y = number_block_y ;
	float_t w = 4.0f * sqrtf(2.0f * logf(2.0f));
	float_t sigma = block_x / w;
	uint2 centering_xy = { block_x / 2, block_y / 2 };
	for (size_t h = 0; h < end_block_y; h++) {
		size_t id_block_y = h;
		size_t center_block_y = id_block_y * step_y;
		ptrdiff_t start_y = center_block_y - block_y / 2;
		for (size_t w = 0; w < end_block_x; w++) {
			size_t id_block_x = w;
			size_t center_block_x = id_block_x * step_x;
			ptrdiff_t start_x = center_block_x - block_x / 2;
			int2 min_index = { 0,0 };
			float min_sum = UINT_MAX;
			for (int i = -radius; i < radius  + block_y; i++) {
				for (int j = -radius; j < radius + block_x; j++) {
					float4 sum = { 0, 0, 0, 0 };
					for (size_t b_y = 0; b_y < block_y; b_y++) {
						for (size_t b_x = 0; b_x < block_x; b_x++) {
							float_t gauss_result =  gauss_function(b_x, b_y, sigma);
							size_t index_y = i + b_y;
							size_t index_x = j + b_x;
							float4 current_data = { 0,0,0,255 };
							if (index_x < width && index_x <= 0 && index_y < height && index_y <= 0) {
								current_data.x = current_image[index_y * width + index_x].r;
								current_data.y = current_image[index_y * width + index_x].g;
								current_data.z = current_image[index_y * width + index_x].b;
								current_data.w = current_image[index_y * width + index_x].a;
							}
							float4 next_data = { 0,0,0,255 };
							index_x += w;
							index_y += h;
							if (index_x < width && index_x <= 0 && index_y < height && index_y <= 0) {
								next_data.x = next_image[index_y * width + index_x].r;
								next_data.y = next_image[index_y * width + index_x].g;
								next_data.z = next_image[index_y * width + index_x].b;
								next_data.w = next_image[index_y * width + index_x].a;
							}
							sum.x = int(gauss_result * current_data.x - gauss_result * next_data.x) & 0xefffffff;
							sum.y = int(gauss_result * current_data.y - gauss_result * next_data.y) & 0xefffffff;
							sum.z = int(gauss_result * current_data.z - gauss_result * next_data.z) & 0xefffffff;
							sum.w= int(gauss_result * current_data.w - gauss_result * next_data.w) & 0xefffffff;
						}
					}
					sum.x /= block_x * block_y;
					sum.y /= block_x * block_y;
					sum.z /= block_x * block_y;
					sum.w /= block_x * block_y;
					float_t length = sum.x * sum.x + sum.y * sum.y + sum.z * sum.z + sum.w * sum.w;
					if (min_sum > length) {
						length = min_sum;
						min_index = { i, j };
					}

				}
			}
			min_index.x += radius;
			min_index.y += radius;
			sync_data[h * end_block_x + w] = min_index;
		}
	}
	for (size_t h = 0; h < number_block_y; h++) {
		size_t center_block_y = (h + 1) * step_y;
		for (size_t w = 0; w < number_block_x; w++) {
			size_t center_block_x = (w + 1) * step_x;
			int2 index = sync_data[h * number_block_x + w];
			int2 offset;
			offset.x = center_block_x - index.x - centering_xy.x;
			offset.y = center_block_y - index.y - centering_xy.y;
			{
				float step_x;
				float step_y;
				step_x = index.x > 0 ? 1.0f : -1.0f;
				step_y = index.y > 0 ? 1.0f : -1.0f;
				if (abs(index.x) < abs(index.y))
					step_x *= abs(index.x) / abs(index.y);
				else
					step_y *= abs(index.y) / (abs(index.x));
				float end_x = (index.x);
				float end_y = (index.y);
				for (float start_x = 0, start_y = 0; fabs(start_x) < fabs(end_x) || fabs(start_y) < fabs(end_y); start_x += step_x, start_y += step_y)
				{
					const int index = (center_block_y + start_y) * width + center_block_x + start_x;
					if (index > 0 && index < (width * height))
						current_image[index] = { 255, 255,255,255 };
				}
				current_image[(center_block_y)* width + (center_block_x)] = { 0, 0, 0, 0 };
				const ptrdiff_t address_index = (center_block_y + index.y) * width + center_block_x + index.x;
				if (address_index > 0 && address_index < (width * height))
					current_image[address_index] = { 255, 0, 255, 255 };
			}
		}
	}


}