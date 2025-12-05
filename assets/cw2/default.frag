#version 430

in vec2 v2fTexcoord;
in vec3 v2fNormal;
in vec3 v2fColor;

layout(location = 2) uniform vec3 uLightDir;
layout(location = 3) uniform vec3 uLightDiffuse;
layout(location = 4) uniform vec3 uSceneAmbient;

uniform sampler2D uTexture;

out vec4 outColor;

void main()
{
	vec3 normal = normalize(v2fNormal);

	float nDotL = max(0.f, dot(normal, uLightDir));

	vec3 textureColor = texture(uTexture, v2fTexcoord).rgb;
	
	vec3 lighting = uSceneAmbient + nDotL * uLightDiffuse;
	outColor = vec4(lighting * textureColor, 1.f);
}

