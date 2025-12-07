#version 430

in vec2 v2fTexcoord;
in vec3 v2fNormal;

struct DirLight {
	vec3 direction;
	vec3 color;
	bool enabled;
};

layout(location = 4) uniform vec3 uSceneAmbient;
uniform DirLight uGlobalLight;

uniform sampler2D uTexture;
layout(location = 5) uniform bool uHasTexture;

out vec4 outColor;

void main()
{
	vec3 normal = normalize(v2fNormal);
	vec3 baseColor; // texture color or fallback color
	if (uHasTexture)
		baseColor = texture(uTexture, v2fTexcoord).rgb;
	else
		baseColor = vec3(0.5, 0.5, 0.5);

	vec3 lighting = uSceneAmbient * baseColor;

	if (uGlobalLight.enabled)
	{
		vec3 lightDir = normalize(uGlobalLight.direction);
		float nDotL = max(0.0, dot(normal, lightDir));
		vec3 diffuse = nDotL * uGlobalLight.color;

		lighting += diffuse * baseColor; 
	}

	outColor = vec4(lighting, 1.0);
}

