#version 430

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in float aMaterialID;

layout(location = 0) uniform mat4 uMVP;
layout(location = 1) uniform mat3 uNormalMatrix;
layout(location = 2) uniform mat4 world;

out vec3 v2fNormal;
flat out int v2fMaterialID;
out vec3 v2fworldPos;

void main()
{
    v2fNormal = normalize(uNormalMatrix * aNormal);
    v2fworldPos = (world * vec4(aPosition, 1.0)).xyz;
    v2fMaterialID = int(aMaterialID);
    gl_Position = uMVP * vec4(aPosition, 1.0);
}