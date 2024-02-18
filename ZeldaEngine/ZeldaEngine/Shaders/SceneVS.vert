#version 450

// push constants block
layout( push_constant ) uniform constants
{
	float time;
	float roughness;
	float metallic;
	uint specConstants;
	uint specConstantsCount;
	uint index;
	uint indexCount;
} global;

layout(set = 0, binding = 0) uniform uniformbuffer
{
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

// Vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec3 outPositionWS;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outColor;
layout(location = 4) out vec2 outTexCoord;

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

void main()
{
	// Render object with MVP
	vec3 position = inPosition;
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
	outPosition = position;
	outPositionWS = (ubo.model * vec4(position, 1.0)).rgb;
	outNormal = (ubo.model * vec4(normalize(inNormal), 1.0)).rgb;
	outColor = Hue2RGB(float(global.index) * 1.71f);
	outTexCoord = inTexCoord;
}
