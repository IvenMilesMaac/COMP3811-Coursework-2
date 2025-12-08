#version 430

flat in int v2fMaterialID;
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

uniform vec3 uMaterialDiffuse[16];
uniform float uMaterialShine[16];

layout(location = 6) uniform vec3 uCameraPos;

out vec4 outColor;

void main()
{
    vec3 normal = normalize(v2fNormal);
	vec3 viewDir = normalize(uCameraPos - v2fworldPos);
    vec3 materialColor = uMaterialDiffuse[v2fMaterialID]; // Color per material

    // Ambient term
    vec3 lighting = uSceneAmbient * materialColor;

	// Global Directional Lighting
    if (uGlobalLight.enabled)
	{
		vec3 lightDir = normalize(uGlobalLight.direction);
		float nDotL = max(0.0, dot(normal, lightDir));
		vec3 diffuse = nDotL * uGlobalLight.color;

		lighting += diffuse; 
	}

	// 3 Local Point Lights
	float shininess = uMaterialShine[v2fMaterialID]; // Shine per material

	for (int i = 0; i < 3; ++i)
	{
		if (uPointLights[i].enabled)
		{
			vec3 lightVec = uPointLights[i].position - v2fworldPos;
			float dist = length(lightVec);
			vec3 pLightDir = lightVec / dist;

			// Diffuse term
			float p_nDotL = max(0.0, dot(normal, pLightDir));
			vec3 pDiffuse = p_nDotL * uPointLights[i].color * materialColor;

			// Specular term
			vec3 halfwayDir = normalize(pLightDir + viewDir);;
			float nDotH = max(0.0, dot(normal, halfwayDir));
			float spec = pow(nDotH, shininess);
			vec3 specular = spec * uPointLights[i].color;

			// Attenuation
			float attenuation = 1.0/ (dist * dist);

			lighting += (pDiffuse + specular) * attenuation;
		}
	}

    outColor = vec4(lighting * 0.5, 1.0);
}