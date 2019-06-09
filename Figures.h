#pragma once
#include "glDevice.h"


static char* make_field_shader = 
	#include "area_view.glsl"
;

class Figures
{
public:
	Figures();
	void gl_make_field(glDevice* mainGL, float x_near, float x_far, float y_near, float y_far, float z_near, float z_far, float x_step, float y_step, float z_step);
	~Figures();
};

