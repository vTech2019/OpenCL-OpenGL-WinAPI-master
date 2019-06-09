

R"===(

#version 430 compatibility
#extension GL_ARB_compute_shader : enable

uniform vec3 number_block_x_y_z;
uniform vec3 start;
uniform vec3 near_xyz;
uniform vec3 far_xyz;
uniform vec3 format_block;

layout(std430) buffer Positions
{
	vec3 position[];
};

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main(void)
{
	float idx = float(gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x);
	float idy = float(gl_WorkGroupID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.y);
	float stride_y = float(gl_NumWorkGroups.y * gl_WorkGroupSize.y);
	float stride_x = float(gl_NumWorkGroups.x * gl_WorkGroupSize.x);
	float number_x = float(number_block_x_y_z.x);
	float number_y = float(number_block_x_y_z.y);
	float number_z = float(number_block_x_y_z.z);	
	vec3 step_xyz = (near_xyz - far_xyz);

	float step_x = number_x / step_xyz.x;
	float step_y = number_y / step_xyz.y;
	float step_z = number_z / step_xyz.z;
	for (float i = idy; i < number_y; i += stride_y) {
		for (float j = idx; j < number_x; j += stride_x) {

			vec2 start_x =   vec2(j * step_x, j * step_x);
			vec2 start_y =   vec2(start.y + i * step_y ,  start.y + i * step_y);
			vec2 start_z =   vec2(0.0f, 0.0f);
			position[int(i * number_x + j) * 2] = vec3(start_x.x, start_y.x, start_z.x);
			position[int(i * number_x + j) * 2 + 1] = vec3(start_x.y, start_y.y, start_z.y);
		}
	}
}
)==="