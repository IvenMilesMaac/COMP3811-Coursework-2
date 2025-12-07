#version 430

flat in int v2fMaterialID;
in vec3 v2fNormal;

struct DirLight {
	vec3 direction;
	vec3 color;
	bool enabled;
};

layout(location = 4) uniform vec3 uSceneAmbient;
uniform DirLight uGlobalLight;

uniform vec3 uMaterialDiffuse[16];
//uniform float uMaterialShine[16];

out vec4 outColor;

void main()
{
    vec3 normal = normalize(v2fNormal);
    vec3 materialColor = uMaterialDiffuse[v2fMaterialID];

    vec3 result = uSceneAmbient * materialColor;

    if (uGlobalLight.enabled)
    {
        vec3 lightDir = normalize(uGlobalLight.direction);
        float nDotL = max(0.0, dot(normal, lightDir));
        vec3 diffuse = nDotL * uGlobalLight.color * materialColor;

        result += diffuse;
    }

    outColor = vec4(result, 1.0);
}