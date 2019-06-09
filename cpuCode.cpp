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
	float4* ptr_memory_image = data;// +radius * pitch_width_image;
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
void cpuDevice::cpu_MAD_SSE2_Stabilization_function(float4* data, uchar4* result, size_t width_image, size_t height_image, float part_block_x, float part_block_y, float pow_sigma, size_t pitch_width_image ) {

}