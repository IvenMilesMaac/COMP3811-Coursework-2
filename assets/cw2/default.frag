#version 430

in vec2 v2fTexcoord;
in vec3 v2fNormal;
in vec3 v2fworldPos;

struct DirLight {
	vec3 direction;
	vec3 color;
	bool enabled;
};

struct PointLight {
	vec3 position;
	vec3 color;
	bool enabled;
};

layout(location = 4) uniform vec3 uSceneAmbient;
uniform DirLight uGlobalLight;
uniform PointLight uPointLights[3];

uniform sampler2D uTexture;
layout(location = 5) uniform bool uHasTexture;

layout(location = 6) uniform vec3 uCameraPos;

out vec4 outColor;

void main()
{
	vec3 normal = normalize(v2fNormal);
	vec3 viewDir = normalize(uCameraPos - v2fworldPos);

	// Texture or gray for base color
	vec3 baseColor;
	if (uHasTexture)
		baseColor = texture(uTexture, v2fTexcoord).rgb;
	else
		baseColor = vec3(0.3, 0.3, 0.3);

	// Ambient term
	vec3 lighting = uSceneAmbient * baseColor;

	// Global Directional Lighting
	if (uGlobalLight.enabled)
	{
		vec3 lightDir = normalize(uGlobalLight.direction);
		float nDotL = max(0.0, dot(normal, lightDir));
		vec3 diffuse = nDotL * uGlobalLight.color;

		lighting += diffuse * baseColor; 
	}

	// 3 Local Point Lights
	float shininess = 40.0;

	for (int i = 0; i < 3; ++i)
	{
		if (uPointLights[i].enabled)
		{
			vec3 lightVec = uPointLights[i].position - v2fworldPos;
			float dist = length(lightVec);
			vec3 pLightDir = lightVec / dist;

			// Diffuse term
			float p_nDotL = max(0.0, dot(normal, pLightDir));
			vec3 pDiffuse = p_nDotL * uPointLights[i].color * baseColor;

			// Specular term
			vec3 halfwayDir = normalize(pLightDir + viewDir);;
			float nDotH = max(0.0, dot(normal, halfwayDir));
			float spec = pow(nDotH, shininess);
			vec3 specular = spec * uPointLights[i].color;

			// Attenuation
			float attenuation = 1.0 / (1.0 + 0.02 * dist * dist);

			lighting += (pDiffuse + specular) * attenuation;
		}
	}

	outColor = vec4(lighting, 1.0);
}

