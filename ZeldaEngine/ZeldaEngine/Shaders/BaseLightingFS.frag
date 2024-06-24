# version 450

#include "Common.glsl"

layout(set = 0, binding = 0) uniform uniformbuffer
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
layout(set = 0, binding = 1) uniform samplerCube CubeMapSampler;
layout(set = 0, binding = 2) uniform sampler2D ShadowMapSampler;
layout(set = 0, binding = 3) uniform sampler2D DepthStencilSampler;
layout(set = 0, binding = 4) uniform sampler2D SceneColorSampler;
layout(set = 0, binding = 5) uniform sampler2D GBufferASampler;
layout(set = 0, binding = 6) uniform sampler2D GBufferBSampler;
layout(set = 0, binding = 7) uniform sampler2D GBufferCSampler;
layout(set = 0, binding = 8) uniform sampler2D GBufferDSampler;


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;


uint DIRECTIONAL_LIGHTS = view.lightsCount[0];
uint POINT_LIGHTS = view.lightsCount[1];
uint SPOT_LIGHTS = view.lightsCount[2];
uint SKY_MAXMIPS = view.lightsCount[3];

vec3 GBufferVis(vec3 FinalColor)
{
	// empty space for imgui, ratio that imgui widget occupies
	vec2 EmptyRatio = view.viewportInfo.zw / view.viewportInfo.xy;
	vec2 UV = fragTexCoord * 3.0f / (1.0f - EmptyRatio);
	vec4 ShadowMap = texture(ShadowMapSampler, UV);
	vec4 DepthStencil = texture(DepthStencilSampler, UV);
	vec4 SceneColor = texture(SceneColorSampler, UV);
	vec4 GBufferA = texture(GBufferASampler, UV);
	vec4 GBufferB = texture(GBufferBSampler, UV);
	vec4 GBufferC = texture(GBufferCSampler, UV);
	vec4 GBufferD = texture(GBufferDSampler, UV);

	vec3 BaseColor = GBufferC.rgb;
	float Metallic = saturate(GBufferB.r);
	float Specular = saturate(GBufferB.g);
	float Roughness = saturate(GBufferB.b);
	vec3 Normal = GBufferA.rgb * 2.0 - 1.0;
	vec3 AmbientOcclution = vec3(GBufferC.a);
	vec3 EmissiveColor = SceneColor.rgb;
	float Mask = SceneColor.a;

	Roughness = max(0.01, Roughness);
	float AO = saturate(AmbientOcclution.r);
	vec3 N = normalize(Normal);
	vec3 P = GBufferD.xyz;
	vec3 V = normalize(view.cameraInfo.xyz - P);
	float NdotV = saturate(dot(N, V));

	const vec2 Step = (1.0f - EmptyRatio) / 3.0f;

	vec3 Result = FinalColor;
	if (fragTexCoord.x < Step.x && fragTexCoord.y < Step.y)
	{
		Result = pow(BaseColor, vec3(0.4545));
		if (fragTexCoord.x > Step.x * (1.0f - EmptyRatio.x) || fragTexCoord.y > Step.y * (1.0f - EmptyRatio.y))
		{
			Result = vec3(1.0f);
		}
	}
	else if (fragTexCoord.x < Step.x * 2.0f && fragTexCoord.y < Step.y)
	{
		Result = vec3(Metallic);
		if (fragTexCoord.x > Step.x * (2.0f - EmptyRatio.x) || fragTexCoord.y > Step.y * (1.0f - EmptyRatio.y))
		{
			Result = vec3(1.0f);
		}
	}
	else if (fragTexCoord.x < Step.x * 3.0f && fragTexCoord.y < Step.y)
	{
		Result = vec3(Roughness);
		if (fragTexCoord.x > Step.x * (3.0f - EmptyRatio.x) || fragTexCoord.y > Step.y * (1.0f - EmptyRatio.y))
		{
			Result = vec3(1.0f);
		}
	}
	else if (fragTexCoord.x < Step.x && fragTexCoord.y < Step.y * 2.0f)
	{
		Result = vec3(N);
		if (fragTexCoord.x > Step.x * (1.0f - EmptyRatio.x) || fragTexCoord.y > Step.y * (2.0f - EmptyRatio.y))
		{
			Result = vec3(1.0f);
		}
	}
	else if (fragTexCoord.x < 1.0f && fragTexCoord.y < Step.y * 2.0f && fragTexCoord.x > Step.x * 2.0f)
	{
		Result = vec3(AO);
		if ((fragTexCoord.x > Step.x * (3.0f - EmptyRatio.x)) || fragTexCoord.y > Step.y * (2.0f - EmptyRatio.y))
		{
			Result = vec3(1.0f);
		}
	}
	else if (fragTexCoord.x < Step.x && fragTexCoord.y < Step.x * 3.0f)
	{
		Result = vec3(0.0f);
		if (fragTexCoord.x > Step.x * (1.0f - EmptyRatio.x) || fragTexCoord.y > Step.y * (3.0f - EmptyRatio.y))
		{
			Result = vec3(1.0f);
		}
	}
	else if (fragTexCoord.x < Step.x * 2.0f && fragTexCoord.x > Step.x && fragTexCoord.y < Step.y * 3.0f && fragTexCoord.y > Step.y * 2.0f)
	{
		float ratio = 1.00 / 1.52;
		vec3 I = V;
		vec3 R = refract(I, normalize(N), ratio);
		vec3 Reflection_L = textureLod(CubeMapSampler, R, 0).rgb * 10.0;
		Result = vec3(Reflection_L);
		if ((fragTexCoord.x > Step.x * (2.0f - EmptyRatio.x)) || fragTexCoord.y > Step.y * (3.0f - EmptyRatio.y))
		{
			Result = vec3(1.0f);
		}
	}
	else if (fragTexCoord.x < Step.x * 3.0f && fragTexCoord.x > Step.x * 2.0f && fragTexCoord.y < Step.y * 3.0f && fragTexCoord.y > Step.y * 2.0f)
	{
		vec4 ShadowCoord = ComputeShadowCoord(view.shadowmapSpace, P);
		float ShadowFactor = ComputePCF(ShadowMapSampler, ShadowCoord / ShadowCoord.w, 2);
		Result = vec3(ShadowFactor);
		if ((fragTexCoord.x > Step.x * (3.0f - EmptyRatio.x)) || fragTexCoord.y > Step.y * (3.0f - EmptyRatio.y))
		{
			Result = vec3(1.0f);
		}
	}
	return Result;
}

void main()
{
	vec3 VertexColor = fragColor;

	vec4 ShadowMap = texture(ShadowMapSampler, fragTexCoord);
	vec4 DepthStencil = texture(DepthStencilSampler, fragTexCoord);
	vec4 SceneColor = texture(SceneColorSampler, fragTexCoord);
	vec4 GBufferA = texture(GBufferASampler, fragTexCoord);
	vec4 GBufferB = texture(GBufferBSampler, fragTexCoord);
	vec4 GBufferC = texture(GBufferCSampler, fragTexCoord);
	vec4 GBufferD = texture(GBufferDSampler, fragTexCoord);

	vec3 BaseColor = GBufferC.rgb;
	float Metallic = saturate(GBufferB.r);
	float Specular = saturate(GBufferB.g);
	float Roughness = saturate(GBufferB.b);
	vec3 Normal = GBufferA.rgb * 2.0 - 1.0;
	vec3 AmbientOcclution = vec3(GBufferC.a);
	vec3 EmissiveColor = SceneColor.rgb;
	float Mask = SceneColor.a;

	Roughness = max(0.01, Roughness);
	float AO = saturate(AmbientOcclution.r);
	vec3 N = normalize(Normal);
	vec3 P = GBufferD.xyz;
	vec3 V = normalize(view.cameraInfo.xyz - P);
	float NdotV = saturate(dot(N, V));

	vec4 ShadowCoord = ComputeShadowCoord(view.shadowmapSpace, P);
	float ShadowFactor = ComputePCF(ShadowMapSampler, ShadowCoord / ShadowCoord.w, 2);

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
	vec3 Reflection_L = textureLod(CubeMapSampler, R, MIPS).rgb * 10.0;
	float Reflection_V = GetSpecularOcclusion(NdotV, Roughness * Roughness, AO);
	vec3 ReflectionColor = Reflection_L * Reflection_V * ReflectionBRDF;

	vec3 FinalColor = DirectLighting + IndirectLighting + ReflectionColor;
	FinalColor *= Mask;

	// Gamma correct
	FinalColor = pow(FinalColor, vec3(0.4545));

	switch (SPEC_CONSTANTS)
	{
		case 0:
			outColor = vec4(FinalColor, 1.0); break;
		case 1:
			outColor = vec4(vec3(pow(BaseColor, vec3(0.4545))), 1.0); break;
		case 2:
			outColor = vec4(vec3(Metallic), 1.0); break;
		case 3:
			outColor = vec4(vec3(Roughness), 1.0); break;
		case 4:
			outColor = vec4(vec3(Normal), 1.0); break;
		case 5:
			outColor = vec4(vec3(AO), 1.0); break;
		case 6:
			outColor = vec4(vec3(VertexColor), 1.0); break;
		case 7:
			outColor = vec4(ReflectionColor, 1.0); break;
		case 8:
			outColor = vec4(vec3(ShadowFactor), 1.0); break;
		case 9:
			outColor = vec4(GBufferVis(FinalColor), 1.0); break;
		default:
			outColor = vec4(FinalColor * ShadowFactor, 1.0); break;
	};
}
