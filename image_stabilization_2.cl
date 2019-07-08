
R"===(
#define PI 3.14159265359f
float Gaussian_filter_x(float x, float sigma, float mean){
	const float _x = x - mean;
	const float _sigma = 2*sigma;
	const float div = _x / _sigma;
	return native_exp(-div * div)/(sigma * sqrt(2.0f * PI));
}
float Gaussian_filter_y(float y, float sigma, float mean){
	const float _y = y - mean;
	const float _sigma = 2*sigma;
	const float div = _y / _sigma;
	return native_exp(-div * div)/(sigma * sqrt(2.0f * PI));
}
float Gaussian_filter_xy(float y, float x, float sigma){
	const float _y = y * y;
	const float _x = x * x;
	const float _sigma = 2.0f*sigma*sigma;
	const float div = (_y + _x) / _sigma;
	return native_exp(-div)/(_sigma * PI);
}
__attribute__((mangled_name(stabilization_image_part1_buffer)))
__kernel void stabilization_image_texture_part1(read_only image2d_t image_current, read_only image2d_t image_next, __global float2* sync_info, 
	int width, int height, int step_x, int step_y, int block_x, int block_y, int radius){
	if (step_x < 1 && step_y < 1)
		return;
	float w = 4.0f * sqrt(2.0f * log(2.0f));
	float sigma = block_x / w;
	int number_blocks_x = (width / step_x) ;
	int number_blocks_y = (height / step_y) ;
	int end_block_x = (number_blocks_x * block_x);
	int end_block_y = (number_blocks_y * block_y);
	float size_block = convert_float(block_x * block_y);
	int part_block_x = (radius - 1) / 2; 
	int part_block_y = (radius - 1) / 2; 
	for (int idy = get_global_id(1); idy < end_block_y; idy+= get_global_size(1)){
		for (int idx = get_global_id(0); idx < end_block_x; idx+= get_global_size(0)){ 
			
	#ifdef SIZE_BLOCK_X
			int id_block_x = idx / SIZE_BLOCK_X;
			int offset_block_x = idx % SIZE_BLOCK_X;
	#ifdef SIZE_BLOCK_Y
			int id_block_y = idy / SIZE_BLOCK_Y;
			int offset_block_y = idy % SIZE_BLOCK_Y;
	#else
			int id_block_y = idy / block_y;
			int offset_block_y = idy % block_y;
	#endif
	#else
			int id_block_x = idx / block_x;
			int id_block_y = idy / block_y;
			int offset_block_x = idx % block_x;
			int offset_block_y = idy % block_y;
	#endif
			int center_block_x = (id_block_x + 1) * step_x;
			int center_block_y = (id_block_y + 1) * step_y;
			int start_block_x = center_block_x - (block_x / 2) + offset_block_x;
			int start_block_y = center_block_y - (block_y / 2) + offset_block_y;
			int y = start_block_y;
			int x = start_block_x;
			int2 index = (int2)0;
			float4 sum = (float4)0;
			float min_sum = (int)INT_MAX;
	#ifdef RADIUS_BLOCK
			for (int i = -RADIUS_BLOCK + offset_block_y; i < RADIUS_BLOCK + block_y; i+=block_y){
				for (int j = -RADIUS_BLOCK + offset_block_x; j < RADIUS_BLOCK + block_x; j+=block_x){
	#else
			for (int i = -radius + offset_block_y; i < radius + block_y; i+=block_y){
				for (int j = -radius + offset_block_x; j < radius + block_x; j+=block_x){
	#endif
					sum = (float4)0;
					
	#ifdef SIZE_BLOCK_Y
					for (int h = 0; h < SIZE_BLOCK_Y; h++){
	#ifdef SIZE_BLOCK_X
						for (int w = 0; w < SIZE_BLOCK_X; w++){
	#else
						for (int w = 0; w < block_x; w++){
	#endif
	#else
					for (int h = 0; h < block_y; h++){
						for (int w = 0; w < block_x; w++){
	#endif
							float4 result_gauss = Gaussian_filter_xy(convert_float(h - part_block_y), convert_float(w - part_block_x), sigma);
							int index_x = (x + w);
							int index_y = (y + h);
							float4 current_image =  read_imagef(image_current, (int2)(index_x, index_y));

							index_x += j;
							index_y += i;
							float4 next_image =  read_imagef(image_next, (int2)(index_x, index_y));
							current_image *= result_gauss;
							next_image *= result_gauss;
							sum += fabs(current_image - next_image );
						}
					}
					sum = native_divide(sum, size_block);
					sum.x = hypot(sum.x, sum.y) + hypot(sum.z, sum.w);
					//sum.x = sum.x * sum.x + sum.y * sum.y + sum.z * sum.z + sum.w * sum.w;
					index = sum.x < min_sum ? (int2)(j, i) : index;
					min_sum = sum.x < min_sum ? sum.x : min_sum;
				}
			}	
	#ifdef RADIUS_BLOCK
			index += RADIUS_BLOCK;
	#else
			index += radius;
	#endif
			int index_sync = ((id_block_y) * number_blocks_x + id_block_x) * size_block + (offset_block_y * block_y + offset_block_x );
			sync_info[index_sync] = (float2)(min_sum, as_float(index.y * (block_x + radius * 2) + index.x) );
		}
	}
}
)==="
