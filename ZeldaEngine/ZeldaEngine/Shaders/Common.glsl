const float PI = 3.14159265359;

struct XkLight
{
	/* position.w represents type of light */
	vec4 position;
	/* color.w represents light intensity */
	vec4 color;
	/* direction.w represents radius */
	vec4 direction;
	/* (only used for spot lights) info.x represents light inner cone angle, info.y represents light outer cone angle */
	vec4 info;
};

struct XkMaterialParameters
{
	float BasecolorOverride;
	float MetallicOverride;
	float SpecularOverride;
	float RoughnessOverride;
};

float saturate(float t)
{
	return clamp(t, 0.0, 1.0);
}

vec3 saturate(vec3 t)
{
	return clamp(t, 0.0, 1.0);
}

float lerp(float f1, float f2, float a)
{
	return ((1.0 - a) * f1 + a * f2);
}

vec3 lerp(vec3 v1, vec3 v2, float a)
{
	return ((1.0 - a) * v1 + a * v2);
}

float remap(float value, float inputMin, float inputMax, float outputMin, float outputMax)
{
	value = clamp(value, inputMin, inputMax);
	return (value - inputMin) / (inputMax - inputMin) * (outputMax - outputMin) + outputMin;
}

// https://www.ronja-tutorials.com/post/041-hsv-colorspace/
vec3 Hue2RGB(float hue) {
    hue = fract(hue); //only use fractional part of hue, making it loop
    float r = abs(hue * 6 - 3) - 1; //red
    float g = 2 - abs(hue * 6 - 2); //green
    float b = 2 - abs(hue * 6 - 4); //blue
    vec3 rgb = vec3(r,g,b); //combine components
    rgb = clamp(rgb,0.0,1.0); //clamp between 0 and 1
    return rgb;
}

mat4 MakeRotMatrix(vec3 R)
{
	mat4 mx, my, mz;
	// rotate around x
	float s = sin(R.x);
	float c = cos(R.x);
	mx[0] = vec4(c, 0.0, s, 0.0);
	mx[1] = vec4(0.0, 1.0, 0.0, 0.0);
	mx[2] = vec4(-s, 0.0, c, 0.0);
	mx[3] = vec4(0.0, 0.0, 0.0, 1.0);	
	// rotate around y
	s = sin(R.y);
	c = cos(R.y);
	my[0] = vec4(c, s, 0.0, 0.0);
	my[1] = vec4(-s, c, 0.0, 0.0);
	my[2] = vec4(0.0, 0.0, 1.0, 0.0);
	my[3] = vec4(0.0, 0.0, 0.0, 1.0);
	// rot around z
	s = sin(R.z);
	c = cos(R.z);
	mz[0] = vec4(1.0, 0.0, 0.0, 0.0);
	mz[1] = vec4(0.0, c, s, 0.0);
	mz[2] = vec4(0.0, -s, c, 0.0);
	mz[3] = vec4(0.0, 0.0, 0.0, 1.0);

	mat4 rotMat = mz * my * mx;
	return rotMat;
}

#ifdef GL_FRAGMENT_SHADER // only include the following code in the fragment shader

vec3 F0 = vec3(0.04);

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Normal
////////////////////////////////////////////////////////////////////////////////////////////////////////////

vec3 ComputeNormal(vec3 fragPosition, vec2 fragTexCoord, vec3 fragNormal)
{
	vec3 pos_dx = dFdx(fragPosition);
	vec3 pos_dy = dFdy(fragPosition);
	vec3 st1 = dFdx(vec3(fragTexCoord, 0.0));
	vec3 st2 = dFdy(vec3(fragTexCoord, 0.0));
	vec3 T = (st2.t * pos_dx - st1.t * pos_dy) / (st1.s * st2.t - st2.s * st1.t);
	vec3 N = normalize(fragNormal);
	T = normalize(T - N * dot(N, T));
	vec3 B = normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);

	return normalize(TBN[2].xyz);
}


vec3 ComputeNormal(vec3 fragPosition, vec2 fragTexCoord, vec3 fragNormal, vec3 texNormal)
{
	vec3 pos_dx = dFdx(fragPosition);
	vec3 pos_dy = dFdy(fragPosition);
	vec3 st1 = dFdx(vec3(fragTexCoord, 0.0));
	vec3 st2 = dFdy(vec3(fragTexCoord, 0.0));
	vec3 T = (st2.t * pos_dx - st1.t * pos_dy) / (st1.s * st2.t - st2.s * st1.t);
	vec3 N = normalize(fragNormal);
	T = normalize(T - N * dot(N, T));
	vec3 B = normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);

	vec3 n = normalize(texNormal);
	return normalize(TBN * normalize(2.0 * n - 1.0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PBR Functions: F term, D trem, V term, IBL, etc.
////////////////////////////////////////////////////////////////////////////////////////////////////////////

// [0] Frensel Schlick
vec3 F_Schlick(vec3 f0, float f90, float u)
{
	return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}


// [1] IBL Defuse Irradiance
vec3 F_Schlick_Roughness(vec3 F0, float cos_theta, float roughness)
{
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cos_theta, 5.0);
}


// [0] Diffuse Term
float Fr_DisneyDiffuse(float NdotV, float NdotL, float LdotH, float roughness)
{
	float E_bias = 0.0 * (1.0 - roughness) + 0.5 * roughness;
	float E_factor = 1.0 * (1.0 - roughness) + (1.0 / 1.51) * roughness;
	float fd90 = E_bias + 2.0 * LdotH * LdotH * roughness;
	vec3 f0 = vec3(1.0);
	float light_scatter = F_Schlick(f0, fd90, NdotL).r;
	float view_scatter = F_Schlick(f0, fd90, NdotV).r;
	return light_scatter * view_scatter * E_factor;
}


// [0] Specular Microfacet Model
float V_SmithGGXCorrelated(float NdotV, float NdotL, float roughness)
{
	float alphaRoughnessSq = roughness * roughness;

	float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
	float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);

	float GGX = GGXV + GGXL;
	if (GGX > 0.0)
	{
		return 0.5 / GGX;
	}
	return 0.0;
}


// [0] GGX Normal Distribution Function
float D_GGX(float NdotH, float roughness)
{
	float alphaRoughnessSq = roughness * roughness;
	float f = (NdotH * alphaRoughnessSq - NdotH) * NdotH + 1.0;
	return alphaRoughnessSq / (PI * f * f);
}


#define REFLECTION_CAPTURE_ROUGHEST_MIP 1
#define REFLECTION_CAPTURE_ROUGHNESS_MIP_SCALE 1.2
/** 
 * Compute absolute mip for a reflection capture cubemap given a roughness.
 */
float ComputeReflectionMipFromRoughness(float roughness, float cubemap_max_mip)
{
	// Heuristic that maps roughness to mip level
	// This is done in a way such that a certain mip level will always have the same roughness, regardless of how many mips are in the texture
	// Using more mips in the cubemap just allows sharper reflections to be supported
	float level_from_1x1 = REFLECTION_CAPTURE_ROUGHEST_MIP - REFLECTION_CAPTURE_ROUGHNESS_MIP_SCALE * log2(max(roughness, 0.001));
	return cubemap_max_mip - 1 - level_from_1x1;
}


vec2 EnvBRDFApproxLazarov(float Roughness, float NoV)
{
	// [ Lazarov 2013, "Getting More Physical in Call of Duty: Black Ops II" ]
	// Adaptation to fit our G term.
	const vec4 c0 = { -1, -0.0275, -0.572, 0.022 };
	const vec4 c1 = { 1, 0.0425, 1.04, -0.04 };
	vec4 r = Roughness * c0 + c1;
	float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
	vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;
	return AB;
}


vec3 EnvBRDFApprox(vec3 SpecularColor, float Roughness, float NoV)
{
	vec2 AB = EnvBRDFApproxLazarov(Roughness, NoV);

	// Anything less than 2% is physically impossible and is instead considered to be shadowing
	// Note: this is needed for the 'specular' show flag to work, since it uses a SpecularColor of 0
	float F90 = saturate(50.0 * SpecularColor.g);

	return SpecularColor * AB.x + F90 * AB.y;
}


float GetSpecularOcclusion(float NoV, float RoughnessSq, float AO)
{
	return saturate(pow(NoV + AO, RoughnessSq) - 1 + AO);
}


float DielectricSpecularToF0(float Specular)
{
	return F0.x * 2.0f * Specular;
}


vec3 ComputeF0(float Specular, vec3 BaseColor, float Metallic)
{
	// clamp pure black base color to get clear coat
	BaseColor = clamp(BaseColor, F0, vec3(1.0f));
	return lerp(DielectricSpecularToF0(Specular).xxx, BaseColor, Metallic.x);
}


struct XkDirectLighting
{
	vec3 Diffuse;
	vec3 Specular;
	vec3 Transmission;
};

vec3 Diffuse_Lambert(vec3 DiffuseColor)
{
	return DiffuseColor * (1 / PI);
}


XkDirectLighting DefaultLitBxDF(vec3 DiffuseColor, vec3 SpecularColor, float Roughness, float LoH, float NoV, float NoL, float NoH)
{
	XkDirectLighting Lighting;

	float F90 = saturate(50.0 * F0.r);
	vec3 F = F_Schlick(F0, F90, LoH);

	float Vis = V_SmithGGXCorrelated(NoV, NoL, Roughness);
	float D = D_GGX(NoH, Roughness);
	vec3 Fr = F * D * Vis;
	float Fd = Fr_DisneyDiffuse(NoV, NoL, LoH, Roughness);

	Lighting.Diffuse = DiffuseColor * (vec3(1.0) - F) * Fd;

	Lighting.Specular = Fr;

	// @TODO: Energy Conservation
	//FBxDFEnergyTermsRGB EnergyTerms = ComputeGGXSpecEnergyTermsRGB(GBuffer.Roughness, Context.NoV, GBuffer.SpecularColor);
	//Lighting.Diffuse *= ComputeEnergyPreservation(EnergyTerms);
	//Lighting.Specular *= ComputeEnergyConservation(EnergyTerms);

	Lighting.Transmission = vec3(0.0f);
	return Lighting;
}

XkDirectLighting IntegrateBxDF(vec3 DiffuseColor, vec3 SpecularColor, float Roughness, float LoH, float NoV, float NoL, float NoH)
{
	return DefaultLitBxDF(DiffuseColor, SpecularColor, Roughness, LoH, NoV, NoL, NoH);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Shadowmap Functions: PCF, PCSS(@TODO)
////////////////////////////////////////////////////////////////////////////////////////////////////////////

const mat4 BiasMat = mat4(
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0);


vec4 ComputeShadowCoord(mat4 ShadowmapSpace, vec3 Position)
{
	return BiasMat * ShadowmapSpace * vec4(Position, 1.0);
}


float ShadowDepthProject(sampler2D ShadowMapSampler, vec4 ShadowCoord, vec2 Offset)
{
	float ShadowFactor = 1.0;
	if (ShadowCoord.z > -1.0 && ShadowCoord.z < 1.0)
	{
		float Dist = texture(ShadowMapSampler, ShadowCoord.st + Offset).r;
		if (ShadowCoord.w > 0.0 && Dist < ShadowCoord.z)
		{
			ShadowFactor = 0.1;
		}
	}
	return ShadowFactor;
}


// Percentage Closer Filtering (PCF)
float ComputePCF(sampler2D sp /*shadowmap sampler*/, vec4 sc /*shadow croodinate*/, int r /*filtering range*/)
{
	ivec2 TexDim = textureSize(sp, 0);
	float Scale = 1.5;
	float dx = Scale * 1.0 / float(TexDim.x);
	float dy = Scale * 1.0 / float(TexDim.y);

	float ShadowFactor = 0.0;
	int Count = 0;

	for (int x = -r; x <= r; x++)
	{
		for (int y = -r; y <= r; y++)
		{
			ShadowFactor += ShadowDepthProject(sp, sc, vec2(dx * x, dy * y));
			Count++;
		}
	}
	return ShadowFactor / Count;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Light Attribute Help Functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////

vec3 GetDirectionalLightDirection(XkLight directionalLight)
{
	return normalize(directionalLight.direction.xyz);
}

vec3 GetDirectionalLightColor(XkLight directionalLight)
{
	return directionalLight.color.rgb;
}

float GetDirectionalLightIntensity(XkLight directionalLight)
{
	return directionalLight.color.w;
}

vec3 ApplyDirectionalLight(XkLight directionalLight, vec3 normal)
{
	vec3 l = GetDirectionalLightDirection(directionalLight);
	vec3 n = normalize(normal);
	float ndotl = clamp(dot(n, l), 0.0, 1.0);
	float density = GetDirectionalLightIntensity(directionalLight);
	vec3 color = GetDirectionalLightColor(directionalLight);
	return ndotl * density * color;
}

vec3 GetPointLightPosition(XkLight pointLight)
{
	return pointLight.position.xyz;
}

vec3 GetPointLightDirection(XkLight pointLight, vec3 position)
{
	return normalize(pointLight.position.xyz - position);
}

float GetPointLightFalloff(XkLight pointLight)
{
	return pointLight.direction.w;
}

vec3 GetPointLightColor(XkLight pointLight)
{
	return pointLight.color.rgb;
}

float GetPointLightIntensity(XkLight pointLight)
{
	return pointLight.color.w;
}

vec3 ApplyPointLight(XkLight pointLight, vec3 position, vec3 normal)
{
	vec3 l = GetPointLightDirection(pointLight, position);
	vec3 n = normalize(normal);
	vec3 light_pos = GetPointLightPosition(pointLight);
	float falloff = GetPointLightFalloff(pointLight);
	float density = GetPointLightIntensity(pointLight);
	vec3 color = GetPointLightColor(pointLight);

	float ndotl = clamp(dot(n, l), 0.0, 1.0);
	vec3 toLight = light_pos - position;
	float distanceSqr = dot(toLight, toLight);

	float dist = distance(light_pos, position);
	float attenuation = remap(dist, 0.0, falloff, 0.0, 1.0);
	attenuation = (1.0 - attenuation);
	return ndotl * density * color * attenuation;
}

#endif // GL_FRAGMENT_SHADER


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Specialization Constants
////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Use this constant to control the flow of the shader depending on the SPEC_CONSTANTS value 
// selected at pipeline creation time
layout(constant_id = 0) const int SPEC_CONSTANTS = 0;

// push constants block
layout(push_constant) uniform constants
{
	uint specConstants;
	uint specConstantsCount;
} global;