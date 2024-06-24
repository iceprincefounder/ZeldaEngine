# version 450

#include "Common.glsl"

layout(set = 0, binding = 2)  uniform samplerCube cubemap;	// cubemap
layout(set = 0, binding = 3)  uniform sampler2D shadowmap;	// shadowmap
layout(set = 0, binding = 4)  uniform sampler2D sampler1;	// basecolor
layout(set = 0, binding = 5)  uniform sampler2D sampler2;	// metalic
layout(set = 0, binding = 6)  uniform sampler2D sampler3;	// roughness
layout(set = 0, binding = 7)  uniform sampler2D sampler4;	// normal
layout(set = 0, binding = 8)  uniform sampler2D sampler5;	// ambient occlution
layout(set = 0, binding = 9)  uniform sampler2D sampler6;	// emissive
layout(set = 0, binding = 10) uniform sampler2D sampler7;	// mask

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 0) out vec4 outSceneColor;
layout(location = 1) out vec4 outGBufferA;
layout(location = 2) out vec4 outGBufferB;
layout(location = 3) out vec4 outGBufferC;
layout(location = 4) out vec4 outGBufferD;

void main()
{
	vec3 VertexColor = fragColor;

	vec3 BaseColor = texture(sampler1, fragTexCoord).rgb;
	float Metallic = texture(sampler2, fragTexCoord).r;
	float Roughness = texture(sampler3, fragTexCoord).r;
	vec3 Normal = ComputeNormal(fragPosition, fragTexCoord, fragNormal, texture(sampler4, fragTexCoord).rgb);
	vec3 AmbientOcclution = texture(sampler5, fragTexCoord).rgb;
	vec3 Emissive = texture(sampler6, fragTexCoord).rgb;
	vec3 OpacityMask = texture(sampler7, fragTexCoord).rgb;

	Roughness = max(0.01, Roughness);
	float AO = AmbientOcclution.r;
	vec3 NormalPacked = (normalize(Normal) + 1.0) / 2.0;
	float Mask = OpacityMask.r;

	outSceneColor = vec4(Emissive, Mask); ;
	outGBufferA = vec4(vec3(NormalPacked), 1.0);
	outGBufferB = vec4(vec3(Metallic, 1.0, Roughness), 1.0);
	outGBufferC = vec4(vec3(BaseColor), AO);
	outGBufferD = vec4(vec3(fragPosition), 1.0);
}
