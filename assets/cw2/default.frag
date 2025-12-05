#version 430

in vec3 v2fNormal;
in vec3 v2fColor;

layout(location = 2) uniform vec3 uLightDir;
layout(location = 3) uniform vec3 uLightDiffuse;
layout(location = 4) uniform vec3 uSceneAmbient;

out vec4 outColor;

void main()
{
	vec3 normal = normalize(v2fNormal);

	float nDotL = max(0.f, dot(normal, uLightDir));

	outColor = vec4((uSceneAmbient + nDotL * uLightDiffuse) * v2fColor, 1.f);	
}

