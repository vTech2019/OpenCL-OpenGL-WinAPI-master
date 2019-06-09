#pragma once
#include <immintrin.h>
#include <intrin.h>
#include <Windows.h>
#include <stdio.h>
#include "Structures.h"

extern "C" {
	void	_fastcall _cpu_sse2_Gauss_function(void* data, void* result, uint32_t width, uint32_t height, uint16_t part_block_x, uint16_t part_block_y, float pow_sigma, uint32_t pitch_width, uint32_t aligned_offset, uint32_t block_x, uint32_t block_y);
};

struct int4 {
	int x, y, z, w;
};
class cpuDevice
{
public:
	bool Intel_cpu;
	bool AMD_cpu;
	bool support_avx512F;
	bool support_avx512PF;
	bool support_avx512ER;
	bool support_avx512CD;
	bool support_avx2;
	bool support_avx;
	bool support_sse4a;
	bool support_sse42;
	bool support_sse41;
	bool support_ssse3;
	bool support_sse3;
	bool support_sse2;
	bool support_sse1;
	bool support_mmxext;
	bool support_mmx;
	cpuDevice();
	~cpuDevice();

	void cpu_sse2_Gauss_function(float4* data, uchar4* result, size_t width_image, size_t height_image, float part_block_x, float part_block_y, float pow_sigma, size_t pitch_width_image, size_t offset_radius_aligned, size_t block_x, size_t block_y);
	void cpu_MAD_SSE2_Stabilization_function(float4* data, uchar4* result, size_t width_image, size_t height_image, float part_block_x, float part_block_y, float pow_sigma, size_t pitch_width_image);
};
