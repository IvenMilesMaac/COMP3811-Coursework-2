#version 430

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;  
layout(location = 2) in vec2 aTexcoord; 

layout(location = 0) uniform mat4 uMVP;


void main()
{
	gl_Position = uMVP * vec4(aPosition, 1.0);
}
