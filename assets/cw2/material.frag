#version 430

flat in int v2fMaterialID;
in vec3 v2fNormal;

layout(location = 2) uniform vec3 uLightDir;
layout(location = 3) uniform vec3 uLightDiffuse;
layout(location = 4) uniform vec3 uSceneAmbient;

uniform vec3 uMaterialDiffuse[16];
uniform float uMaterialShine[16];

out vec4 outColor;

void main()
{
    vec3 normal = normalize(v2fNormal);
    float nDotL = max(0.0, dot(normal, uLightDir));
    
    vec3 materialColor = uMaterialDiffuse[v2fMaterialID];
    float shininess = uMaterialShine[v2fMaterialID];

    float diffuseFactor = pow(nDotL, shininess);
    vec3 lighting = uSceneAmbient + diffuseFactor * uLightDiffuse;

    outColor = vec4(lighting * materialColor, 1.0);
}