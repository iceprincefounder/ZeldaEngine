#version 450

vec2 positions[6] = vec2[](
	vec2(-1.0, -1.0),
	vec2(-1.0, 1.0),
	vec2(1.0, 1.0),
	vec2(-1.0, -1.0),
	vec2(1.0, 1.0),
	vec2(1.0, -1.0)
);

vec3 colors[6] = vec3[](
	vec3(1.0, 0.0, 0.0),
	vec3(0.0, 0.0, 1.0),
	vec3(0.0, 1.0, 0.0),
	vec3(1.0, 0.0, 0.0),
	vec3(0.0, 1.0, 0.0),
	vec3(0.0, 0.0, 1.0)
);

vec2 uvs[6] = vec2[](
	vec2(0.0, 0.0),
	vec2(0.0, 1.0),
	vec2(1.0, 1.0),
	vec2(0.0, 0.0),
	vec2(1.0, 1.0),
	vec2(1.0, 0.0)
);

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
	gl_Position = vec4(positions[gl_VertexIndex], 1.0, 1.0);
	fragColor = colors[gl_VertexIndex];
	fragTexCoord = uvs[gl_VertexIndex];
}