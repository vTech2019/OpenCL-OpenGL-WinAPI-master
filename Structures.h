#pragma once
#include <stdint.h>
struct GLubyte4 {
	unsigned char r, g, b, a;
};
struct matrix_16 {
	float data[16];
};
struct uint8 {
	uint32_t s0, s1, s2, s3, s4, s5, s6, s7;
};
struct int2 {
	int32_t x, y;
};
struct int8 {
	int32_t s0, s1, s2, s3, s4, s5, s6, s7;
};
struct float3 {
	float x;
	float y;
	float z;
};
struct float4 {
	float x;
	float y;
	float z;
	float w;
};

struct uchar4 {
	uint8_t b;
	uint8_t g;
	uint8_t r;
	uint8_t a;
};

struct RGBA {
	uint8_t b;
	uint8_t g;
	uint8_t r;
	uint8_t a;
};
struct dataImage {
	size_t width;
	size_t height;
	size_t stride;
	int pixelFormat;
	bool rgba;
	uint8_t* data;
};