# version 450

#include "Common.glsl"

// [binding = 0] for vertex MVP buffer
layout(set = 0, binding = 1) uniform uniformbuffer
{
	mat4 viewProjSpace;
	mat4 shadowmapSpace;
	mat4 localToWorld;
	vec4 cameraInfo;
	vec4 viewportInfo;
	XkLight directionalLights [16];
	XkLight pointLights [512];
	XkLight spotLights [16];
	/* [0] for directionalLights, [1] for pointLights, [2] for spotLights, [3] for cubemap max mip num*/
	ivec4 lightsCount;
	float Time;
	float zNear;
	float zFar;
} view;
layout(set = 0, binding = 2)  uniform samplerCube cubemap;  // sky cubemap
layout(set = 0, binding = 3)  uniform sampler2D shadowmap;  // sky cubemap
layout(set = 0, binding = 4)  uniform sampler2D sampler1; // basecolor
layout(set = 0, binding = 5)  uniform sampler2D sampler2; // metalic
layout(set = 0, binding = 6)  uniform sampler2D sampler3; // roughness
layout(set = 0, binding = 7)  uniform sampler2D sampler4; // normalmap
layout(set = 0, binding = 8)  uniform sampler2D sampler5; // ambient occlution
layout(set = 0, binding = 9)  uniform sampler2D sampler6; // emissive
layout(set = 0, binding = 10) uniform sampler2D sampler7; // mask

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;


uint DIRECTIONAL_LIGHTS = view.lightsCount[0];
uint POINT_LIGHTS = view.lightsCount[1];
uint SPOT_LIGHTS = view.lightsCount[2];
uint SKY_MAXMIPS = view.lightsCount[3];


void main()
{
	vec3 VertexColor = fragColor;

	vec3 BaseColor = texture(sampler1, fragTexCoord).rgb;
	float Metallic = saturate(texture(sampler2, fragTexCoord).r);
	float Roughness = saturate(texture(sampler3, fragTexCoord).r);
	vec3 Normal = ComputeNormal(fragPosition, fragTexCoord, fragNormal, texture(sampler4, fragTexCoord).rgb);
	vec3 AmbientOcclution = texture(sampler5, fragTexCoord).rgb;

	Roughness = max(0.01, Roughness);
	float AO = AmbientOcclution.r;
	vec3 N = Normal;
	vec3 P = fragPosition;
	vec3 V = normalize(view.cameraInfo.xyz - P);
	float NdotV = saturate(dot(N, V));

	float ShadowFactor = 1.0;
	vec4 ShadowCoord = ComputeShadowCoord(view.shadowmapSpace, P);
	// we use PCF shadow filtering here
	// ShadowFactor = ShadowDepthProject(view.shadowmapSpace, ShadowCoord / ShadowCoord.w, vec2(0.0));
	ShadowFactor = ComputePCF(shadowmap, ShadowCoord / ShadowCoord.w, 2);

	// (1) Direct Lighting : DisneyDiffuse + SpecularGGX
	vec3 DirectLighting = vec3(0.0);
	vec3 DiffuseColor = BaseColor.rgb * (1.0 - Metallic);
	vec3 SpecularColor = vec3(1.0);
	for (uint i = 0u; i < DIRECTIONAL_LIGHTS; ++i)
	{
		vec3 L = GetDirectionalLightDirection(view.directionalLights[i]);
		vec3 H = normalize(V + L);

		float LdotH = saturate(dot(L, H));
		float NdotH = saturate(dot(N, H));
		float NdotL = saturate(dot(N, L));

		XkDirectLighting DirectionalLight = IntegrateBxDF(DiffuseColor, SpecularColor, Roughness, LdotH, NdotV, NdotL, NdotH);

		DirectLighting += ApplyDirectionalLight(view.directionalLights[i], N) * (DirectionalLight.Diffuse + DirectionalLight.Specular) * ShadowFactor;
	}
	for (uint i = 0u; i < POINT_LIGHTS; ++i)
	{
		vec3 L = GetPointLightDirection(view.pointLights[i], P);
		vec3 H = normalize(V + L);

		float LdotH = saturate(dot(L, H));
		float NdotH = saturate(dot(N, H));
		float NdotL = saturate(dot(N, L));

		XkDirectLighting PointLight = IntegrateBxDF(DiffuseColor, SpecularColor, Roughness, LdotH, NdotV, NdotL, NdotH);

		DirectLighting += ApplyPointLight(view.pointLights[i], P, N) * (PointLight.Diffuse + PointLight.Specular);
	}

	// (2) Indirect Lighting : Simple lambert diffuse as indirect lighting
	vec3 IndirectLighting = DiffuseColor / PI * AO * 0.3 * ShadowFactor;

	// (3) Reflection Specular : Image based lighting
	vec3 ReflectionSpec = ComputeF0(0.5, BaseColor, Metallic);
	vec3 ReflectionBRDF = EnvBRDFApprox(ReflectionSpec, Roughness, NdotV);
	float ratio = 1.00 / 1.52;
	vec3 I = V;
	vec3 R = refract(I, normalize(N), ratio);
	float MIPS = ComputeReflectionMipFromRoughness(Roughness, SKY_MAXMIPS);
	vec3 Reflection_L = textureLod(cubemap, R, MIPS).rgb * 10.0;
	float Reflection_V = GetSpecularOcclusion(NdotV, Roughness * Roughness, AO);
	vec3 ReflectionColor = Reflection_L * Reflection_V * ReflectionBRDF;

	vec3 FinalColor = DirectLighting + IndirectLighting + ReflectionColor;

	// Gamma correct
	FinalColor = pow(FinalColor, vec3(0.4545));

	switch (SPEC_CONSTANTS)
	{
		case 0:
			outColor = vec4(FinalColor * ShadowFactor, 1.0); break;
		case 1:
			outColor = vec4(vec3(BaseColor), 1.0); break;
		case 2:
			outColor = vec4(vec3(Metallic), 1.0); break;
		case 3:
			outColor = vec4(vec3(Roughness), 1.0); break;
		case 4:
			outColor = vec4(vec3(Normal), 1.0); break;
		case 5:
			outColor = vec4(vec3(AmbientOcclution), 1.0); break;
		case 6:
			outColor = vec4(vec3(VertexColor), 1.0); break;
		case 7:
			outColor = vec4(vec3(ReflectionColor), 1.0); break;
		case 8:
			outColor = vec4(vec3(ShadowFactor), 1.0); break;
		case 9:
			outColor = vec4(FinalColor * ShadowFactor, 1.0); break;
		default:
			outColor = vec4(FinalColor * ShadowFactor, 1.0); break;
	};
}
