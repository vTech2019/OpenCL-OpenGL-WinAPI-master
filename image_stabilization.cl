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
__kernel void gauss_function(write_only image2d_t result_image, read_only image2d_t image, int width, int height, int block_x, int block_y, int step_x, int step_y){

	const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE |  CLK_FILTER_NEAREST; 
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int stride_x = get_global_size(0);
    int stride_y = get_global_size(1);
	float w = 4.0f * sqrt(2.0f * log(2.0f));
	float sigma = block_x / w;
	float4 norm = block_x * block_y ; 
	int part_block_x = (block_x - 1) / 2; 
	int part_block_y = (block_y - 1) / 2; 
	int local_step_x = -step_x;
	int local_step_y = -step_y;
	int number_overlays_x = block_x / step_x + (block_x % step_x > 0 ? 1 : 0) ;
	int number_overlays_y = block_y / step_y + (block_y % step_y > 0 ? 1 : 0);
	for (int h = idy; h < height; h+= stride_y){ 
		for (int w = idx; w <  width; w+=stride_x){
				const float4 data_image = read_imagef(image, smp, (int2)(w, h));
				float4 sum = (float4)0;
				float4 sum_normalize = (float4)0;
				float number_gaussian = 0.0f;
				for (int i = 0; i < number_overlays_y; i++){
					const int new_position_y = h + local_step_y * i;
					const bool find_border_y = h - local_step_y * i < height ? new_position_y > -1 ? true : false : false;
					if (find_border_y)
					{ 
						int i =  -part_block_y + (new_position_y % block_y);
						for (int j = 0; j < number_overlays_x; j++){
							const int new_position_x = w + local_step_x * j;
							const bool find_border_x = w - local_step_x * j < width ? new_position_x > -1 ? true : false : false;
							if (find_border_x)
							{ 
								//const float4 data_image = read_imagef(image, (int2)(new_position_x, new_position_y));
								int j =  -part_block_x + (new_position_x % block_x);
								float4 result_gauss = Gaussian_filter_xy(convert_float(i), convert_float(j), sigma);
								sum += data_image * result_gauss;
								sum_normalize += result_gauss;
								number_gaussian += 1.0f;
							}
						}
					}
				}
				write_imagef(result_image,  (int2)(w, h), ((sum / (number_gaussian) ) * norm));
		}
	}
}

__kernel void MAD_block_matching(__global uchar4* result_buffer, __global float4* sync_buffer, read_only image2d_t image_current, read_only image2d_t image_next, int width, int height, int block_x, int block_y, int radius){

	const sampler_t smp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE |  CLK_FILTER_NEAREST; 

	//if (get_global_id(0) > width || get_global_id(1) > height)
	//	return;

    const float idx = convert_float(get_global_id(0))/convert_float(width);
    const float idy = convert_float(get_global_id(1))/convert_float(height);

    const float stride_x = convert_float(get_global_size(0))/convert_float(width);
    const float stride_y = convert_float(get_global_size(1))/convert_float(height);

	const float f_block_x = convert_float(block_x);
	const float f_block_y = convert_float(block_y);

    const float step_pixel_x = (1.0f)/convert_float(width);
    const float step_pixel_y = (1.0f)/convert_float(height);

	const float step_block_x = f_block_x/convert_float(width);
	const float step_block_y = f_block_y/convert_float(height);

	const float size_block = convert_float(block_x) * convert_float(block_y);

	const float search_radius_x = (convert_float(radius) * 1.0f / convert_float(width)) ;
	const float search_radius_y = (convert_float(radius) * 1.0f / convert_float(height)) ;

	float2 block_offset;

	int global_id_x = get_global_id(0);
	int global_id_y = get_global_id(1);
	int global_stride_x = get_global_size(0);
	int global_stride_y = get_global_size(1);


	for (float y = idy; y < 1.0f; y += stride_y, global_id_y += global_stride_y-1){
		float start_search_radius_y = y - search_radius_y;
		float end_search_radius_y = y + search_radius_y + step_block_y;
		for (float x = idx; x < 1.0f; x += stride_x, global_id_x += global_stride_x){	

			float start_search_radius_x = x - search_radius_x;
			float end_search_radius_x = x + search_radius_x + step_block_x;

			float2 index_current_block = (float2)(x, y);
			float4 current_image = 255.0f * read_imagef(image_current, smp, index_current_block);
			float4 min_sum = (float4)FLT_MAX;
			float4 sum = (float4)0.0f;;
			for (float i = start_search_radius_y; i <= end_search_radius_y; i+=step_block_y){

				float start_block_y = i;
				float end_block_y = step_block_y + i;

				for (float j = start_search_radius_x; j <= end_search_radius_x; j+=step_block_x){

					float start_block_x = j;
					float end_block_x = step_block_x + j;
					sum = (float4)0.0f;

					for (float index_block_y = start_block_y; index_block_y < end_block_y; index_block_y += step_pixel_y){
						for (float index_block_x = start_block_x; index_block_x < end_block_x; index_block_x += step_pixel_x){

							float2 index_next_block = (float2)(index_block_x, index_block_y);
							float4 next_image = 255.0f * read_imagef(image_next, smp, index_next_block);

							sum += fabs(current_image - next_image);
						}
					}
					min_sum = min_sum.x > sum.x ? sum : min_sum;
					block_offset = min_sum.x > sum.x ?(float2)(j, i) : block_offset;

				}
			}
			min_sum = min_sum / (size_block);

			{
				result_buffer[(global_id_y) * width + (global_id_x)] = convert_uchar4_sat_rte(255.0f * read_imagef(image_next,  smp, (float2)(x, y)));
			}
		}
	}

}

__attribute__((mangled_name(stabilization_image_part1_buffer)))
__kernel void stabilization_image_part1(const __global uchar4* restrict image_current,const __global uchar4* restrict image_next, __global int2* sync_info, 
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
							float4 current_image = (index_x < 0 || index_x >= width || index_y < 0 || index_y >= height) ? (float4)0 :
								convert_float4(image_current[index_y * width + index_x]);
							index_x += j;
							index_y += i;
							float4 next_image = (index_x < 0 || index_x >= width || index_y < 0 || index_y >= height) ? (float4)0 :
								convert_float4(image_next[index_y * width + index_x]);
							sum += fabs(current_image - next_image );
						}
					}
					sum = sum / (size_block);
					sum.x = sum.x * sum.x + sum.y * sum.y + sum.z * sum.z + sum.w * sum.w;
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
			sync_info[index_sync] = (int2)(convert_int(min_sum), index.y * (block_x + radius * 2) + index.x );
		}
	}
}
__kernel void stabilization_image_part2(__global int2* sync_info,
	int width, int height, int step_x, int step_y, int block_x, int block_y, int radius, __local int* local_data){
	int local_size = get_local_size(0);
	int number_blocks_x = (width / step_x) ;
	int number_blocks_y = (height / step_y) ;
	int size_block = block_x * block_y;
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int global_step = get_global_size(0);
	int global_work_end = number_blocks_x * number_blocks_y * local_size;

	#ifdef LOCAL_STEPS
		const int current_step = LOCAL_STEPS;
		#ifdef NUMBER_STEPS_BLOCK
		const int number_steps_block = NUMBER_STEPS_BLOCK;
		#else
		const int number_steps_block = size_block / local_size + (size_block % local_size ? 1 : 0);
		#endif
	#else
		const int number_steps_block = size_block / local_size + (size_block % local_size ? 1 : 0);
		const int current_step = convert_int(native_powr(convert_float(2), convert_float(convert_int_sat_rtp(log2(convert_float(local_size))))));
	#endif

	for (int i=get_local_id(0); i < 2 * current_step; i +=  get_local_size(0))
		local_data[i] = INT_MAX;
	barrier(CLK_LOCAL_MEM_FENCE);	

	 __local int* local_min = local_data + local_id;
	 __local int* local_index = local_data + current_step + local_id;

	for (int id = global_id; id < global_work_end; id+= global_step){
		int id_block = (id / local_size);
		int global_offset = id_block * size_block;
		int index_block = 0;
		int min_sum = INT_MAX;
		int min_index = INT_MAX;
		local_id = get_local_id(0);
		while (index_block < number_steps_block){ 
				const int2 info_sync = local_id < size_block ? sync_info[global_offset + local_id] : (int2)INT_MAX;
				local_min[0] = info_sync.x;
				local_index[0] = info_sync.y;
				barrier(CLK_LOCAL_MEM_FENCE);	
				for (int step_block = current_step  >> 1; step_block > 0; step_block>>=1 ){
					if ( get_local_id(0) < step_block){ 
						int current_value =  local_min[step_block];
						if (local_min[0] >current_value){ 
							local_min[0] = current_value;
							local_index[0] = local_index[step_block];
						}
					}
					barrier(CLK_LOCAL_MEM_FENCE);
				}
				local_id += local_size;
				index_block += 1;
				int current_value =  local_min[0];
				if (current_value  < min_sum){ 
					min_sum = current_value;
					min_index =  local_index[0];
				}
			} 
			if (get_local_id(0) == 0)
				sync_info[id_block] = (int2)(min_sum ,min_index);
		}
}
__kernel void stabilization_image_part3(__global uchar4* image, __global int2* sync_info,
	int width, int height, int step_x, int step_y, int block_x, int block_y, int radius){
	int number_blocks_x = (width / step_x) ;
	int number_blocks_y = (height / step_y) ;
	int part_block_x = (radius - 1) / 2; 
	int part_block_y = (radius - 1) / 2; 
	int size_block_fild_x = radius * 2 + block_x;
	int size_block_fild_y = radius * 2 + block_y;
	int2 centering_xy = (int2)(block_x / 2, block_y / 2);
	int number_blocks = number_blocks_y * number_blocks_x;
	for (int idy = get_global_id(1); idy < number_blocks_y; idy+= get_global_size(1)){
		int center_block_y = (idy + 1) * step_y;
		for (int idx = get_global_id(0); idx < number_blocks_x; idx+= get_global_size(0)){ 	
			int center_block_x = (idx + 1) * step_x;
			int2 index = sync_info[idy * number_blocks_x + idx];
			int2 index_xy = (int2)(index.y % size_block_fild_x, index.y / size_block_fild_x);
			index_xy -= (int2)radius;
			sync_info[(idy) * number_blocks_x + (idx)] = (int2)(center_block_x, center_block_y)  - index_xy - centering_xy;
			float step_x = index_xy.x < index_xy.y ?  convert_float(index_xy.x) / convert_float(index_xy.y):   1.0f;
			float step_y = index_xy.y < index_xy.x ?   convert_float(index_xy.y) / convert_float(index_xy.x) :   1.0f;
			float endy = convert_float_rtn(index_xy.y);
			float endx = convert_float_rtn(index_xy.x);
			if (index_xy.x == 0){
				for (int i = 0; i <endy;i+=step_y)
					image[(center_block_y +i  ) * width + center_block_x ] = (uchar4)255;
			}else if(index_xy.y == 0){
				for (int j = 0; j <endx;j+=step_x)
					image[(center_block_y  ) * width + center_block_x + j ] = (uchar4)255;
			}else
				for (float i = 0, j = 0; fabs(i) < fabs(endy) && fabs(j) < fabs(endx) ; i+= step_y, j+= step_x)
				{
					image[(center_block_y + convert_int_rtn(i)  ) * width + center_block_x + convert_int_rtn(j) ] = (uchar4)255;
				}

			image[(center_block_y  ) * width + (center_block_x  )]  = (uchar4)0;
			image[(center_block_y + index_xy.y) * width + center_block_x + index_xy.x]  = (uchar4)(255);
		}
	}
}

__kernel void stabilization_image_part4(__global uchar4* image_result,const __global uchar4* restrict image_next, __global int2* sync_info, 
	int width, int height, int step_x, int step_y, int block_x, int block_y, int radius){
	int number_blocks_x = (width / step_x) ;
	int number_blocks_y = (height / step_y) ;
	int global_work_end_x = number_blocks_x * block_x;
	int global_work_end_y = number_blocks_y * block_y;

	for (int idy = get_global_id(1); idy < global_work_end_y; idy+= get_global_size(1)){
		int index_block_y = idy / block_y;
		int offset_block_y = idy % block_y;
		int center_block_y = (index_block_y + 1) * step_y;
		int start_block_y = center_block_y - (block_y / 2) ;
		for (int idx = get_global_id(0); idx < global_work_end_x; idx+= get_global_size(0)){ 
			int index_block_x = idx / block_x;
			int offset_block_x = idx % block_x;
			int center_block_x = (index_block_x + 1) * step_x;
			int start_block_x = center_block_x - (block_x / 2);
			int2 index = sync_info[index_block_y * number_blocks_x + index_block_x];
			const int index_y = (index.y + offset_block_y);
			const int index_x = (index.x + offset_block_x);
			if (index_y > 0 && index_y < height && index_x > 0 && index_x < width)
				image_result[(start_block_y + offset_block_y) * width + (start_block_x + offset_block_x)  ] = image_next[(index.y + offset_block_y) * width + index.x  + offset_block_x] ;
		}
	}
}


)==="