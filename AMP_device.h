#pragma once
#include <amp.h>
#include <amp_graphics.h>
#include <amp_short_vectors.h>
#include <vector>
#include <deque>
#include <iostream>
#include "Structures.h"
class AMP_device
{
	std::vector<concurrency::accelerator> accelerators;
	std::vector<concurrency::task<bool>> tasks; 
public:
	AMP_device* operator[](size_t i);
	AMP_device();
	~AMP_device();
	void gauss_function(uint32_t width, uint32_t height, uint32_t block_x, uint32_t block_y, uint32_t step_x, uint32_t step_y, uchar4* image, uchar4* result);
};
