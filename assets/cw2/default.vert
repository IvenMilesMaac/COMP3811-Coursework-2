#version 430

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;  
layout(location = 2) in vec2 aTexcoord; 

layout(location = 0) uniform mat4 uMVP;
layout(location = 1) uniform mat3 uNormalMatrix;

out vec3 v2fNormal;
out vec3 v2fColor;

void main()
{
	v2fNormal = normalize(uNormalMatrix * aNormal);
	v2fColor = vec3(0.8f, 0.8f, 0.8f);
	gl_Position = uMVP * vec4(aPosition, 1.0);
}
