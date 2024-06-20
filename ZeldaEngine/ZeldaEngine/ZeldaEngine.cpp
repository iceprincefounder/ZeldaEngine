// Copyright ©XUKAI. All Rights Reserved.

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // Depth buffer range, OpenGL default -1.0 to 1.0, but Vulakn default as 0.0 to 1.0
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GLSL.std.450.h>
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/DirStackFileIncluder.h>
#include <glslang/Include/ShHandle.h>
#include <glslang/OSDependent/osinclude.h>

#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <regex>
#include <limits>
#include <optional>
#include <vector>
#include <set>
#include <array>
#include <chrono>
#include <unordered_map>
#include <random>
#include <thread>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"

#define MAX_FRAMES_IN_FLIGHT 2
#define VIEWPORT_WIDTH 1080
#define VIEWPORT_HEIGHT 720
#define PBR_SAMPLER_NUMBER 7 // BC + M + R + N + AO + Emissive + Mask
#define BG_SAMPLER_NUMBER 1
#define SKY_SAMPLER_NUMBER 1
#define GBUFFER_SAMPLER_NUMBER 6
#define POINT_LIGHTS_NUM 16
#define MAX_DIRECTIONAL_LIGHTS_NUM 16
#define MAX_POINT_LIGHTS_NUM 512
#define MAX_SPOT_LIGHTS_NUM 16
#define SHADOWMAP_DIM 1024
#define VERTEX_BUFFER_BIND_ID 0
#define INSTANCE_BUFFER_BIND_ID 1
#define ENABLE_WIREFRAME false
#define ENABLE_INDIRECT_DRAW false
#define ENABLE_DEFEERED_SHADING true
// @TODO: Implement Bindless Feature
// @see https://dev.to/gasim/implementing-bindless-design-in-vulkan-34no
#define ENABLE_BINDLESS false
#define ENABLE_IMGUI_EDITOR true
#define ENABLE_GENERATE_WORLD false

#define ASSETS(x) ProfabsAsset(x)

typedef std::string XkString;

/**
* Helper class to generate SPIRV code from GLSL source
* A very simple version of the glslValidator application
*/
class XkShaderCompiler
{
private:
	static glslang::EShTargetLanguage EnvTargetLanguage;
	static glslang::EShTargetLanguageVersion EnvTargetLanguageVersion;

public:
	/**
	* @brief Read shader source file into buffer
	* @param[out] outShaderSource Output shader source data buffer
	* @param[out] outShaderStage Output shader stage detect by file suffix
	* @param inFilename Shader file path
	*/
	static void ReadShaderFile(std::vector<uint8_t>& outShaderSource, VkShaderStageFlagBits& outShaderStage, const XkString& inFilename)
	{
		XkString::size_type const p(inFilename.find_last_of('.'));
		XkString ext = inFilename.substr(p + 1);
		if (ext == "vert")
			outShaderStage = VK_SHADER_STAGE_VERTEX_BIT;
		else if (ext == "frag")
			outShaderStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		else if (ext == "comp")
			outShaderStage = VK_SHADER_STAGE_COMPUTE_BIT;
		else if (ext == "geom")
			outShaderStage = VK_SHADER_STAGE_GEOMETRY_BIT;
		else if (ext == "tesc")
			outShaderStage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
		else if (ext == "tese")
			outShaderStage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
		else if (ext == "rgen")
			outShaderStage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
		else if (ext == "rahit")
			outShaderStage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
		else if (ext == "rchit")
			outShaderStage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
		else if (ext == "rmiss")
			outShaderStage = VK_SHADER_STAGE_MISS_BIT_KHR;
		else if (ext == "rint")
			outShaderStage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
		else if (ext == "rcall")
			outShaderStage = VK_SHADER_STAGE_CALLABLE_BIT_KHR;
#ifndef __APPLE__
		else if (ext == "mesh")
			outShaderStage = VK_SHADER_STAGE_MESH_BIT_NV /*VK_SHADER_STAGE_MESH_BIT_EXT*/;
		else if (ext == "task")
			outShaderStage = VK_SHADER_STAGE_TASK_BIT_NV /*VK_SHADER_STAGE_TASK_BIT_EXT*/;
#endif
		else
		{
			assert(true);
			throw std::runtime_error("[ReadShaderFile] File extension[" + ext + "]does not have a vulkan shader stage.");
		}

		std::ifstream file;

		file.open(inFilename, std::ios::in | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("[ReadShaderFile] Failed to open file: " + inFilename);
		}

		uint64_t read_count = 0;
		file.seekg(0, std::ios::end);
		read_count = static_cast<uint64_t>(file.tellg());
		file.seekg(0, std::ios::beg);

		outShaderSource.resize(static_cast<size_t>(read_count));
		file.read(reinterpret_cast<char*>(outShaderSource.data()), read_count);
		file.close();
	}

	/**
	* @brief Write shader buffer into new SPV file
	* @param inFilename SPV file path
	* @param inSpirvSource Spirv source buffer
	*/
	static void SaveShaderFile(const XkString& inFilename, const std::vector<unsigned int>& inSpirvSource)
	{
		std::ofstream out;
		out.open(inFilename.c_str(), std::ios::binary | std::ios::out);
		if (out.fail()){
			assert(true);
			throw std::runtime_error("[SaveShaderFile] ERROR: Failed to open file: " + inFilename);
		}
		for (int i = 0; i < (int)inSpirvSource.size(); ++i) {
			unsigned int word = inSpirvSource[i];
			out.write((const char*)&word, 4);
		}
		out.close();
	}

	/**
	* @brief Set the glslang target environment to translate to when generating code
	* @param target_language The language to translate to
	* @param target_language_version The version of the language to translate to
	*/
	static void SetTargetEnvironment(glslang::EShTargetLanguage inTargetLanguage,
		glslang::EShTargetLanguageVersion InTargetLanguageVersion) {
		EnvTargetLanguage = inTargetLanguage;
		EnvTargetLanguageVersion = InTargetLanguageVersion;
	};

	/**
	* @brief Reset the glslang target environment to the default values
	*/
	static void ResetTargetEnvironment() {
		EnvTargetLanguage = glslang::EShTargetLanguage::EShTargetNone;
		EnvTargetLanguageVersion = static_cast<glslang::EShTargetLanguageVersion>(0);
	};

	/**
	* @brief Compiles GLSL to SPIRV code, original function named "compile_to_spirv" in Vulkan-Samples
	* @param stage The Vulkan shader stage flag
	* @param glsl_source The GLSL source code to be compiled
	* @param entry_point The entrypoint function name of the shader stage
	* @param shader_variant_preamble The shader variant preamble
	* @param shader_variant_processes The shader variant processes
	* @param[out] spirv The generated SPIRV code
	* @param[out] info_log Stores any log messages during the compilation process
	*/
	bool CompileToSpirv(VkShaderStageFlagBits stage,
		const std::vector<uint8_t>& glsl_source,
		const XkString& entry_point,
		const XkString& shader_variant_preamble,
		const std::vector<XkString>& shader_variant_processes,
		std::vector<std::uint32_t>& spirv,
		XkString& info_log)
	{
		auto FindShaderLanguage = [](VkShaderStageFlagBits stage) -> EShLanguage
		{
			switch (stage)
			{
			case VK_SHADER_STAGE_VERTEX_BIT:
				return EShLangVertex;
			case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
				return EShLangTessControl;
			case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
				return EShLangTessEvaluation;
			case VK_SHADER_STAGE_GEOMETRY_BIT:
				return EShLangGeometry;
			case VK_SHADER_STAGE_FRAGMENT_BIT:
				return EShLangFragment;
			case VK_SHADER_STAGE_COMPUTE_BIT:
				return EShLangCompute;
			case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
				return EShLangRayGen;
			case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
				return EShLangAnyHit;
			case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
				return EShLangClosestHit;
			case VK_SHADER_STAGE_MISS_BIT_KHR:
				return EShLangMiss;
			case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
				return EShLangIntersect;
			case VK_SHADER_STAGE_CALLABLE_BIT_KHR:
				return EShLangCallable;
#ifndef __APPLE__ // not support macOS
			case VK_SHADER_STAGE_MESH_BIT_NV /*VK_SHADER_STAGE_MESH_BIT_EXT*/:
				return EShLangMesh;
			case VK_SHADER_STAGE_TASK_BIT_NV /*VK_SHADER_STAGE_TASK_BIT_EXT*/:
				return EShLangTask;
#endif // __APPLE__
			default:
				return EShLangVertex;
			}
		};

		// Initialize glslang library.
		glslang::InitializeProcess();

		EShMessages messages = static_cast<EShMessages>(EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules);

		EShLanguage language = FindShaderLanguage(stage);
		XkString source = XkString(glsl_source.begin(), glsl_source.end());

		const char* file_name_list[1] = { "" };
		const char* shader_source = reinterpret_cast<const char*>(source.data());

		glslang::TShader shader(language);
		shader.setStringsWithLengthsAndNames(&shader_source, nullptr, file_name_list, 1);
		shader.setEntryPoint(entry_point.c_str());
		shader.setSourceEntryPoint(entry_point.c_str());
		shader.setPreamble(shader_variant_preamble.c_str());
		shader.addProcesses(shader_variant_processes);
		if (EnvTargetLanguage != glslang::EShTargetLanguage::EShTargetNone)
		{
			shader.setEnvTarget(EnvTargetLanguage, EnvTargetLanguageVersion);
		}
		DirStackFileIncluder includeDir;
		includeDir.pushExternalLocalDirectory("shaders");

		if (!shader.parse(GetDefaultResources(), 100, false, messages, includeDir))
		{
			info_log = XkString(shader.getInfoLog()) + "\n" + XkString(shader.getInfoDebugLog());
			return false;
		}

		// Add shader to new program object.
		glslang::TProgram program;
		program.addShader(&shader);

		// Link program.
		if (!program.link(messages))
		{
			info_log = XkString(program.getInfoLog()) + "\n" + XkString(program.getInfoDebugLog());
			return false;
		}

		// Save any info log that was generated.
		if (shader.getInfoLog())
		{
			info_log += XkString(shader.getInfoLog()) + "\n" + XkString(shader.getInfoDebugLog()) + "\n";
		}

		if (program.getInfoLog())
		{
			info_log += XkString(program.getInfoLog()) + "\n" + XkString(program.getInfoDebugLog());
		}

		glslang::TIntermediate* intermediate = program.getIntermediate(language);

		// Translate to SPIRV.
		if (!intermediate)
		{
			info_log += "Failed to get shared intermediate code.\n";
			return false;
		}

		spv::SpvBuildLogger logger;

		glslang::GlslangToSpv(*intermediate, spirv, &logger);

		info_log += logger.getAllMessages() + "\n";

		// Shutdown glslang library.
		glslang::FinalizeProcess();

		return true;
	};
};

glslang::EShTargetLanguage XkShaderCompiler::EnvTargetLanguage = glslang::EShTargetLanguage::EShTargetNone;
glslang::EShTargetLanguageVersion XkShaderCompiler::EnvTargetLanguageVersion = static_cast<glslang::EShTargetLanguageVersion>(0);

enum class EXkRenderFlags : uint16_t
{
	None = 1 << 0, // None means Vertex only in Deferred Shading
	VertexIndexed = 1 << 1,
	Instanced = 1 << 2,
	ScreenRect = 1 << 3,
	TwoSided = 1 << 4,
	NoDepthTest = 1 << 5,
	Shadow = 1 << 6,
	Skydome = 1 << 7,
	Background = 1 << 8,
	ForwardShading = 1 << 9,
	DeferredScene = 1 << 10,
	DeferredLighting = 1 << 11,
};

inline EXkRenderFlags operator|(EXkRenderFlags a, EXkRenderFlags b)
{
	return static_cast<EXkRenderFlags>(static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
}

inline EXkRenderFlags operator&(EXkRenderFlags a, EXkRenderFlags b)
{
	return static_cast<EXkRenderFlags>(static_cast<uint16_t>(a) & static_cast<uint16_t>(b));
}

inline bool operator==(EXkRenderFlags a, EXkRenderFlags b)
{
	return ((static_cast<uint16_t>(a) & static_cast<uint16_t>(b))) == static_cast<uint16_t>(b);
}

/** Model MVP matrices */
struct XkUniformBufferObject {
	glm::mat4 Model;
	glm::mat4 View;
	glm::mat4 Proj;
};

/** Model transform matrix.*/
// @TODO: we use a global transform for all render object currently
// It's great to use model's own transform
struct XkTransfrom {
	glm::vec3 Location;
	glm::vec3 Rotation;
	glm::vec3 Scale3D;

	glm::quat Quaternion;
};

/** The instance of mesh data block*/
struct XkInstanceData {
	glm::vec3 InstancePosition;
	glm::vec3 InstanceRotation;
	glm::float32 InstancePScale;
	glm::uint8 InstanceTexIndex;
};

/** The vertex of mesh data block*/
struct XkVertex {
	glm::vec3 Position;
	glm::vec3 Normal;
	glm::vec3 Color;
	glm::vec2 TexCoord;

	// Vertex description
	static VkVertexInputBindingDescription GetBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = VERTEX_BUFFER_BIND_ID;
		bindingDescription.stride = sizeof(XkVertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 4> GetAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

		attributeDescriptions[0].binding = VERTEX_BUFFER_BIND_ID;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(XkVertex, Position);

		attributeDescriptions[1].binding = VERTEX_BUFFER_BIND_ID;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(XkVertex, Normal);

		attributeDescriptions[2].binding = VERTEX_BUFFER_BIND_ID;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(XkVertex, Color);

		attributeDescriptions[3].binding = VERTEX_BUFFER_BIND_ID;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(XkVertex, TexCoord);

		return attributeDescriptions;
	}

	// Vertex description with instance buffer
	static std::array<VkVertexInputBindingDescription, 2> GetBindingInstancedDescriptions() {
		VkVertexInputBindingDescription bindingDescription0{};
		bindingDescription0.binding = VERTEX_BUFFER_BIND_ID;
		bindingDescription0.stride = sizeof(XkVertex);
		bindingDescription0.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputBindingDescription bindingDescription1{};
		bindingDescription1.binding = INSTANCE_BUFFER_BIND_ID;
		bindingDescription1.stride = sizeof(XkInstanceData);
		bindingDescription1.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

		return { bindingDescription0, bindingDescription1 };
	}

	static std::array<VkVertexInputAttributeDescription, 8> GetAttributeInstancedDescriptions() {
		std::array<VkVertexInputAttributeDescription, 8> attributeDescriptions{};

		attributeDescriptions[0].binding = VERTEX_BUFFER_BIND_ID;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(XkVertex, Position);

		attributeDescriptions[1].binding = VERTEX_BUFFER_BIND_ID;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(XkVertex, Normal);

		attributeDescriptions[2].binding = VERTEX_BUFFER_BIND_ID;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(XkVertex, Color);

		attributeDescriptions[3].binding = VERTEX_BUFFER_BIND_ID;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(XkVertex, TexCoord);

		attributeDescriptions[4].binding = INSTANCE_BUFFER_BIND_ID;
		attributeDescriptions[4].location = 4;
		attributeDescriptions[4].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[4].offset = offsetof(XkInstanceData, InstancePosition);

		attributeDescriptions[5].binding = INSTANCE_BUFFER_BIND_ID;
		attributeDescriptions[5].location = 5;
		attributeDescriptions[5].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[5].offset = offsetof(XkInstanceData, InstanceRotation);

		attributeDescriptions[6].binding = INSTANCE_BUFFER_BIND_ID;
		attributeDescriptions[6].location = 6;
		attributeDescriptions[6].format = VK_FORMAT_R32_SFLOAT;
		attributeDescriptions[6].offset = offsetof(XkInstanceData, InstancePScale);

		attributeDescriptions[7].binding = INSTANCE_BUFFER_BIND_ID;
		attributeDescriptions[7].location = 7;
		attributeDescriptions[7].format = VK_FORMAT_R8_UINT;
		attributeDescriptions[7].offset = offsetof(XkInstanceData, InstanceTexIndex);

		return attributeDescriptions;
	}

	bool operator==(const XkVertex& other) const {
		return Position == other.Position && Normal == other.Normal && Color == other.Color && TexCoord == other.TexCoord;
	}
};

namespace std {
	template<> struct hash<XkVertex> {
		size_t operator()(XkVertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.Position) ^ (hash<glm::vec3>()(vertex.Color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.TexCoord) << 1);
		}
	};
}


/* Description of the rule that procedural generated object apply.*/
struct XkDesc
{
	XkDesc() : ObjectID(GenerateObjectID()) {}

	XkTransfrom Transfrom;
	uint32_t ObjectID;

	static uint32_t GenerateObjectID()
	{
		static uint32_t currentID = 0;
		return currentID++;
	}
};

/** Object data struct, to procedural generate object instance*/
struct XkObjectDesc : public XkDesc
{
	XkString ProfabName = "";
	EXkRenderFlags RenderFlags = EXkRenderFlags::None;

	//~ Begin procedural instance generation arguments
	uint32_t InstanceCount = 0;
	float MinRadius = 0.0f;
	float MaxRadius = 0.0f;
	float MinPScale = 0.0f;
	float MaxPScale = 0.0f;
	float MinRotYaw = 0.0f;
	float MaxRotYaw = 0.0f;
	float MinRotRoll = 0.0f;
	float MaxRotRoll = 0.0f;
	float MinRotPitch = 0.0f;
	float MaxRotPitch = 0.0f;
	//~ End procedural instance generation arguments

	static void GenerateInstance(std::vector<XkInstanceData>& OutData, const XkObjectDesc& Object)
	{
		OutData.resize(Object.InstanceCount);
		for (uint32_t i = 0; i < Object.InstanceCount; i++) {
			float radians = RandRange(0.0f, 360.0f, std::rand());
			float distance = RandRange(Object.MinRadius, Object.MaxRadius, std::rand());
			float X = sin(glm::radians(radians)) * distance;
			float Y = cos(glm::radians(radians)) * distance;
			float Z = 0.0;
			OutData[i].InstancePosition = glm::vec3(X, Y, Z);
			// Y(Pitch), Z(Yaw), X(Roll)
			float Yaw = float(M_PI) * RandRange(0.0f, 180.0f, std::rand());
			OutData[i].InstanceRotation = glm::vec3(0.0, Yaw, 0.0);
			OutData[i].InstancePScale = RandRange(Object.MinPScale, Object.MaxPScale, std::rand());
			OutData[i].InstanceTexIndex = RandRange(0, 255, std::rand());
		}
	}

	/* Random number engine */
	static int RandRange(int min, int max, uint32_t seed)
	{
		std::mt19937 gen(seed); // Initialize Mersenne Twister algorithm generator with seed value
		std::uniform_int_distribution<int> dis(min, max); // Define a uniform distribution from min to max
		return dis(gen); // Generate random number
	}
	static float RandRange(float min, float max, uint32_t seed)
	{
		std::mt19937 gen(seed); // Initialize Mersenne Twister algorithm generator with seed value
		std::uniform_real_distribution<float> dis(min, max); // Define a uniform distribution from min to max
		return dis(gen); // Generate random number
	}
};

/** Light data struct, to procedural generate light instance*/
struct XkLightDesc : public XkDesc
{
	glm::vec3 Position;
	uint8_t Type;
	glm::vec3 Color;
	float Intensity;
	glm::vec3 Direction;
	float Radius;
	glm::vec4 ExtraData;
};

/** Camera data struct.*/
struct XkCameraDesc : public XkDesc
{
	glm::vec3 Position;
	glm::vec3 Lookat;
	float Speed;
	float FOV;
	float zNear;
	float zFar;

	float ArmLength;
	float Yaw;
	float Pitch;
	float Roll;

	XkCameraDesc& operator= (const XkCameraDesc& rhs)
	{
		Transfrom = rhs.Transfrom;

		Position = rhs.Position;
		Lookat = rhs.Lookat;
		Speed = rhs.Speed;
		FOV = rhs.FOV;
		zNear = rhs.zNear;
		zFar = rhs.zFar;
		ArmLength = rhs.ArmLength;
		Yaw = rhs.Yaw;
		Pitch = rhs.Pitch;
		Roll = rhs.Roll;
		return *this;
	}
};

struct XkMesh 
{
	std::vector<XkVertex> Vertices;                       // Vertex
	std::vector<uint32_t> Indices;                       // Index
	VkBuffer VertexBuffer;                               // Vertex buffer
	VkDeviceMemory VertexBufferMemory;                   // Vertex buffer memory
	VkBuffer IndexBuffer;                                // Index buffer
	VkDeviceMemory IndexBufferMemory;                    // Index buffer memory

	// only init with instanced mesh
	VkBuffer InstancedBuffer;                            // Instanced buffer
	VkDeviceMemory InstancedBufferMemory;                // Instanced buffer memory
};

typedef XkMesh XkInstancedMesh;

/** Meshlet data block*/
struct XkMeshlet 
{
	uint32_t VertexOffset;
	uint32_t VertexCount;
	uint32_t TriangleOffset;
	uint32_t TriangleCount;
	float BoundsCenter[3];
	float BoundsRadius;
	float ConeApex[3];
	float ConeAxis[3];
	float ConeCutoff, Pad;
};

/** Meshlet group set data block, all meshlets make up a model mesh data.*/
struct XkMeshletSet 
{
	std::vector<XkMeshlet> Meshlets;
	std::vector<uint32_t> MeshletVertices;
	std::vector<uint8_t> MeshletTriangles;
};

struct XkMeshIndirect : public XkMesh
{
	XkMeshletSet MeshletSet;
};

typedef XkMeshIndirect XkInstancedMeshIndirect;

struct XkMaterial 
{
	std::vector<VkImage> TextureImages;
	std::vector<VkDeviceMemory> TextureImageMemorys;
	std::vector<VkImageView> TextureImageViews;
	std::vector<VkSampler> TextureImageSamplers;

	VkDescriptorPool DescriptorPool;
	std::vector<VkDescriptorSet> DescriptorSets;

	// DescriptorSetLayout is define by each render pass
	VkDescriptorSetLayout* DescriptorSetLayout; 
};

struct XkRenderBase
{
	XkMaterial Material;
	uint32_t InstanceCount;
	
	// @TODO: use model's own transform
	/* uniform buffers contain model transform */
	std::vector<VkBuffer> TransfromUniformBuffers;
	/* uniform buffers memory contain model transform */
	std::vector<VkDeviceMemory> TransfromUniformBuffersMemory;
};

/** RenderObject for a single draw call.*/
struct XkRenderObject : public XkRenderBase
{
	XkMesh Mesh;
};

struct XkRenderInstancedObject : public XkRenderBase
{
	XkInstancedMesh Mesh;
};

struct XkRenderObjectIndirectBase : public XkRenderBase
{
	VkBuffer IndirectCommandsBuffer;
	VkDeviceMemory IndirectCommandsBufferMemory;
	std::vector<VkDrawIndexedIndirectCommand> IndirectCommands;
};

struct XkRenderObjectIndirect : public XkRenderObjectIndirectBase
{
	XkMeshIndirect Mesh;
};

struct XkRenderInstancedObjectIndirect : public XkRenderObjectIndirectBase
{
	XkInstancedMeshIndirect Mesh;
};

typedef XkRenderObject FRenderDeferredObject;
typedef XkRenderInstancedObject FRenderDeferredInstancedObject;

/** Common light data struct.*/
struct XkLight
{
	XkLight() : Position(glm::vec4(0.0f)), Color(glm::vec4(1.0f)), Direction(glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)), LightInfo(glm::vec4(0.0f)) {};

	XkLight(const XkLightDesc& Desc)
	{
		Position = glm::vec4(Desc.Position, (float)Desc.Type);
		Color = glm::vec4(Desc.Color, Desc.Intensity);
		Direction = glm::vec4(Desc.Direction, Desc.Radius);
		LightInfo = Desc.ExtraData;
	}

	glm::vec4 Position;
	glm::vec4 Color; // rgb for Color, a for intensity
	glm::vec4 Direction;
	glm::vec4 LightInfo;

	XkLight& operator=(const XkLight& rhs)
	{
		Position = rhs.Position;
		Color = rhs.Color;
		Direction = rhs.Direction;
		LightInfo = rhs.LightInfo;
		return *this;
	}
};

const std::vector<const char*> ValidationLayers = { "VK_LAYER_KHRONOS_validation" };
#if ENABLE_BINDLESS
const std::vector<const char*> DeviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME };
#else
const std::vector<const char*> DeviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
#endif

#ifdef NDEBUG
const bool bEnableValidationLayers = false;  // Build Configuration: Release
#else
const bool bEnableValidationLayers = true;   // Build Configuration: Debug
#endif


static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}


static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		func(instance, debugMessenger, pAllocator);
	}
}


/** All hardware device data struct*/
struct XkQueueFamilyIndices
{
	std::optional<uint32_t> GraphicsFamily;
	std::optional<uint32_t> PresentFamily;

	bool IsComplete()
	{
		return GraphicsFamily.has_value() && PresentFamily.has_value();
	}
};


/** All support hardware device details data struct*/
struct XkSwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR Capabilities;
	std::vector<VkSurfaceFormatKHR> Formats;
	std::vector<VkPresentModeKHR> PresentModes;
};


/**
 * ZeldaEngine: A tiny real time vulkan based 3D engine with modern graphics
 * All implementations in this class.
 */
class XkZeldaEngineApp
{
	struct XkGlobalInput {
		bool bGameMode;

		float CurrentTime;
		float DeltaTime;
		bool bPlayStageRoll;
		float RollStage;
		bool bPlayLightRoll;
		float RollLight;

		double LastMouseX, LastMouseY;
		bool bInitMouse;
		
		XkCameraDesc* MainCamera;
		bool bCameraFocus;
		bool bCameraMoving;

		void InitCamera(XkCameraDesc* Camera) { MainCamera = Camera; }

		void ResetToFocus()
		{
			bGameMode = false;
			MainCamera->Position = glm::vec3(2.0, 2.0, 2.0);
			MainCamera->Lookat = glm::vec3(0.0, 0.0, 0.5);
			MainCamera->Speed = 2.5;
			MainCamera->FOV = 45.0;
			MainCamera->zNear = 0.1f;
			MainCamera->zFar = 45.0f;

			glm::mat4 transform = glm::lookAt(MainCamera->Position, MainCamera->Lookat, glm::vec3(0.0f, 0.0f, 1.0f));
			glm::quat rotation(transform);
			glm::vec3 Direction = glm::normalize(MainCamera->Lookat - MainCamera->Position);
			MainCamera->Yaw = 30.0; //glm::degrees(glm::atan(Direction.x, Direction.y));
			MainCamera->Pitch = 60.0; //glm::degrees(glm::asin(Direction.z));
			MainCamera->ArmLength = 10.0;
			bCameraMoving = false;
			bCameraFocus = true;

			MainCamera->Position.x = cos(glm::radians(MainCamera->Yaw)) * cos(glm::radians(MainCamera->Pitch)) * MainCamera->ArmLength;
			MainCamera->Position.y = sin(glm::radians(MainCamera->Yaw)) * cos(glm::radians(MainCamera->Pitch)) * MainCamera->ArmLength;
			MainCamera->Position.z = sin(glm::radians(MainCamera->Pitch)) * MainCamera->ArmLength;
		}

		void ResetAnimation()
		{
			bPlayStageRoll = false;
			RollStage = 0.0;
			bPlayLightRoll = false;
			RollLight = 0.0;
		}
	} GlobalInput;

	/** Global constants*/
	struct XkGlobalConstants {
		float BaseColorOverride;
		float MetallicOverride;
		float SpecularOverride;
		float RoughnessOverride;
		uint32_t SpecConstants;
		uint32_t SpecConstantsCount;
		void ResetConstants()
		{
			BaseColorOverride = 1.0f;
			MetallicOverride = 0.0;
			SpecularOverride = 1.0;
			RoughnessOverride = 1.0f;
			SpecConstants = 0;
			SpecConstantsCount = 10;
		}
	} GlobalConstants;

	/** Scene viewport data struct.*/
	struct XkView {
		glm::mat4 ViewProjSpace;
		glm::mat4 ShadowmapSpace;
		glm::mat4 LocalToWorld;
		glm::vec4 CameraInfo;
		glm::vec4 ViewportInfo;
		XkLight DirectionalLights[MAX_DIRECTIONAL_LIGHTS_NUM];
		XkLight PointLights[MAX_POINT_LIGHTS_NUM];
		XkLight SpotLights[MAX_SPOT_LIGHTS_NUM];
		// LightsCount: 
		// [0] for number of DirectionalLights
		// [1] for number of PointLights
		// [2] for number of SpotLights
		// [3] for number of cube map max miplevels
		glm::ivec4 LightsCount;
		glm::float32 Time;
		glm::float32 zNear;
		glm::float32 zFar;

		XkView& operator=(const XkView& rhs)
		{
			ViewProjSpace = rhs.ViewProjSpace;
			ShadowmapSpace = rhs.ShadowmapSpace;
			LocalToWorld = rhs.LocalToWorld;
			CameraInfo = rhs.CameraInfo;
			ViewportInfo = rhs.ViewportInfo;
			for (size_t i = 0; i < MAX_DIRECTIONAL_LIGHTS_NUM; i++)
			{
				DirectionalLights[i] = rhs.DirectionalLights[i];
			}
			for (size_t i = 0; i < MAX_POINT_LIGHTS_NUM; i++)
			{
				PointLights[i] = rhs.PointLights[i];
			}
			for (size_t i = 0; i < MAX_SPOT_LIGHTS_NUM; i++)
			{
				SpotLights[i] = rhs.SpotLights[i];
			}
			LightsCount = rhs.LightsCount;
			zNear = rhs.zNear;
			zFar = rhs.zFar;
			return *this;
		}
	} View;

	struct XkSocketListener
	{
        std::string receivedData;
#ifdef _WIN32
		WSADATA wsaData;
		char recvbuf[65720];
		int recvbuflen = 65720;

		SOCKET ListenSocket = INVALID_SOCKET;
		SOCKET ClientSocket = INVALID_SOCKET;
#else
#endif
		void Release()
		{
#ifdef _WIN32
			closesocket(ClientSocket);
			closesocket(ListenSocket);
			WSACleanup();
#else
#endif
		}
	} SocketListener;

	/** Hold all the render object proxy data.*/
	struct XkScene {
		std::vector<XkRenderObject> RenderObjects;
		std::vector<XkRenderInstancedObject> RenderInstancedObjects;
		std::vector<XkRenderObjectIndirect> RenderIndirectObjects;
		std::vector<XkRenderInstancedObjectIndirect> RenderIndirectInstancedObjects;
		std::vector<FRenderDeferredObject> RenderDeferredObjects;
		std::vector<FRenderDeferredInstancedObject> RenderDeferredInstancedObjects;

		VkDescriptorSetLayout* DescriptorSetLayout = nullptr;
		VkDescriptorSetLayout* IndirectDescriptorSetLayout = nullptr;
		VkDescriptorSetLayout* DeferredSceneDescriptorSetLayout = nullptr;
		VkDescriptorSetLayout* DeferredLightingDescriptorSetLayout = nullptr;

		bool bReload = false;

		void Reset()
		{
			RenderObjects.clear();
			RenderInstancedObjects.clear();
			RenderIndirectObjects.clear();
			RenderIndirectInstancedObjects.clear();
			RenderDeferredObjects.clear();
			RenderDeferredInstancedObjects.clear();

			DescriptorSetLayout = nullptr;
			IndirectDescriptorSetLayout = nullptr;
			DeferredSceneDescriptorSetLayout = nullptr;
			DeferredLightingDescriptorSetLayout = nullptr;
		}
	} Scene;

	struct XkWorld
	{
		XkString FilePath = "Content/World.json";

		bool EnableSkydome = true;
		bool OverrideSkydome = false;
		XkString SkydomeFileName;

		bool OverrideCubemap = false;
		std::array<XkString, 6> CubemapFileNames;

		bool EnableBackground = false;
		bool OverrideBackground = false;
		XkString BackgroundFileName;

		XkCameraDesc MainCamera;

		std::vector<XkLightDesc> DirectionalLights;
		std::vector<XkLightDesc> PointLights;
		std::vector<XkLightDesc> SpotLights;
		std::vector<XkLightDesc> QuadLights;

		std::vector<XkObjectDesc> ObjectDescs;

		void Load(const std::string& RawData = std::string())
		{
			// reset data before load it
			Reset();

			rapidjson::Document JsonDocument;
			if (RawData.empty())
			{
				FILE* fp = fopen(FilePath.c_str(), "rb");
				if (fp == nullptr) {
					throw std::runtime_error("[WORLD] Failed to load file: " + FilePath);
					return;
				}
				std::vector<char> readBuffer(65720);
				rapidjson::FileReadStream is(fp, readBuffer.data(), sizeof(readBuffer));
				JsonDocument.ParseStream(is);
				fclose(fp);
			}
			else
			{
				if (JsonDocument.Parse(RawData.c_str()).HasParseError()) {
					throw std::runtime_error("[WORLD] JSON parse error");
					return;
				}
			}

			const auto& MainCameraObject = JsonDocument["MainCamera"];
			MainCamera.Position = glm::vec3(MainCameraObject["Position"][0].GetFloat(), MainCameraObject["Position"][1].GetFloat(), MainCameraObject["Position"][2].GetFloat());
			MainCamera.Lookat = glm::vec3(MainCameraObject["Lookat"][0].GetFloat(), MainCameraObject["Lookat"][1].GetFloat(), MainCameraObject["Lookat"][2].GetFloat());
			MainCamera.FOV = MainCameraObject["FOV"].GetFloat();
			MainCamera.Speed = MainCameraObject["Speed"].GetFloat();
			MainCamera.zNear = MainCameraObject["zNear"].GetFloat();
			MainCamera.zFar = MainCameraObject["zFar"].GetFloat();

			const auto& SkydomeObject = JsonDocument["Skydome"];
			EnableSkydome = SkydomeObject["EnableSkydome"].GetBool();
			OverrideSkydome = SkydomeObject["OverrideSkydome"].GetBool();
			SkydomeFileName = SkydomeObject["SkydomeFileName"].GetString();
			OverrideCubemap = SkydomeObject["OverrideCubemap"].GetBool();
			const auto& CubemapFileNamesArray = SkydomeObject["CubemapFileNames"].GetArray();
			for (uint32_t i = 0; i < CubemapFileNamesArray.Size(); i++) {
				CubemapFileNames[i] = CubemapFileNamesArray[i].GetString();
			}

			const auto& BackgroundObject = JsonDocument["Background"];
			EnableBackground = BackgroundObject["EnableBackground"].GetBool();
			OverrideBackground = BackgroundObject["OverrideBackground"].GetBool();
			BackgroundFileName = BackgroundObject["BackgroundFileName"].GetString();

			auto PushLightsFromJson = [](std::vector<XkLightDesc>& OutLights, const auto& JsonArray)
				{
					for (uint32_t i = 0; i < JsonArray.Size(); i++) {
						const auto& JsonObject = JsonArray[i];
						XkLightDesc LocalLight;
						const auto& PositionArray = JsonObject["Position"].GetArray();
						LocalLight.Position = glm::vec3(PositionArray[0].GetFloat(), PositionArray[1].GetFloat(), PositionArray[2].GetFloat());
						LocalLight.Type = JsonObject["Type"].GetUint();
						const auto& ColorArray = JsonObject["Color"].GetArray();
						LocalLight.Color = glm::vec3(ColorArray[0].GetFloat(), ColorArray[1].GetFloat(), ColorArray[2].GetFloat());
						LocalLight.Intensity = JsonObject["Intensity"].GetFloat();
						const auto& DirectionArray = JsonObject["Direction"].GetArray();
						LocalLight.Direction = glm::vec3(DirectionArray[0].GetFloat(), DirectionArray[1].GetFloat(), DirectionArray[2].GetFloat());
						LocalLight.Radius = JsonObject["Radius"].GetFloat();
						const auto& ExtraDataArray = JsonObject["ExtraData"].GetArray();
						LocalLight.ExtraData = glm::vec4(ExtraDataArray[0].GetFloat(), ExtraDataArray[1].GetFloat(), ExtraDataArray[2].GetFloat(), ExtraDataArray[3].GetFloat());
						OutLights.push_back(LocalLight);
					}
				};
			const auto& DirectionalLightsArray = JsonDocument["DirectionalLights"].GetArray();
			PushLightsFromJson(DirectionalLights, DirectionalLightsArray);
			
			const auto& PointLightsArray = JsonDocument["PointLights"].GetArray();
			PushLightsFromJson(PointLights, PointLightsArray);

			const auto& SpotLightsArray = JsonDocument["SpotLights"].GetArray();
			PushLightsFromJson(SpotLights, SpotLightsArray);

			const auto& ObjectsArray = JsonDocument["Objects"];
			for (uint32_t i = 0; i < ObjectsArray.Size(); i++) {
				const auto& JsonObject = ObjectsArray[i];
				XkObjectDesc LocalObject;
				LocalObject.RenderFlags = (EXkRenderFlags)JsonObject["RenderFlags"].GetUint();
				LocalObject.ProfabName = JsonObject["ProfabName"].GetString();
				LocalObject.InstanceCount = JsonObject["InstanceCount"].GetUint();
				LocalObject.MinRadius = JsonObject["MinRadius"].GetFloat();
				LocalObject.MaxRadius = JsonObject["MaxRadius"].GetFloat();
				LocalObject.MinRotYaw = JsonObject["MinRotYaw"].GetFloat();
				LocalObject.MaxRotYaw = JsonObject["MaxRotYaw"].GetFloat();
				LocalObject.MinRotRoll = JsonObject["MinRotRoll"].GetFloat();
				LocalObject.MaxRotRoll = JsonObject["MaxRotRoll"].GetFloat();
				LocalObject.MinRotPitch = JsonObject["MinRotPitch"].GetFloat();
				LocalObject.MaxRotPitch = JsonObject["MaxRotPitch"].GetFloat();
				LocalObject.MinPScale = JsonObject["MinPScale"].GetFloat();
				LocalObject.MaxPScale = JsonObject["MaxPScale"].GetFloat();
				ObjectDescs.push_back(LocalObject);
			}
		}

		void Save()
		{
			rapidjson::Document JsonDocument;
			JsonDocument.SetObject();
			rapidjson::Document::AllocatorType& JsonAllocator = JsonDocument.GetAllocator();

			XkCameraDesc& Camera = MainCamera;
			rapidjson::Value CameraObject(rapidjson::kObjectType);
			CameraObject.AddMember("Position", rapidjson::Value(rapidjson::kArrayType), JsonAllocator);
			CameraObject["Position"].PushBack(Camera.Position.x, JsonAllocator);
			CameraObject["Position"].PushBack(Camera.Position.y, JsonAllocator);
			CameraObject["Position"].PushBack(Camera.Position.z, JsonAllocator);
			CameraObject.AddMember("Lookat", rapidjson::Value(rapidjson::kArrayType), JsonAllocator);
			CameraObject["Lookat"].PushBack(Camera.Lookat.x, JsonAllocator);
			CameraObject["Lookat"].PushBack(Camera.Lookat.y, JsonAllocator);
			CameraObject["Lookat"].PushBack(Camera.Lookat.z, JsonAllocator);
			CameraObject.AddMember("Speed", rapidjson::Value(Camera.Speed), JsonAllocator);
			CameraObject.AddMember("FOV", rapidjson::Value(Camera.FOV), JsonAllocator);
			CameraObject.AddMember("zNear", rapidjson::Value(Camera.zNear), JsonAllocator);
			CameraObject.AddMember("zFar", rapidjson::Value(Camera.zFar), JsonAllocator);
			JsonDocument.AddMember("MainCamera", CameraObject, JsonAllocator);

			rapidjson::Value SkydomeObject(rapidjson::kObjectType);
			SkydomeObject.AddMember("EnableSkydome", rapidjson::Value(EnableSkydome), JsonAllocator);
			SkydomeObject.AddMember("OverrideSkydome", rapidjson::Value(OverrideSkydome), JsonAllocator);
			SkydomeObject.AddMember("SkydomeFileName", rapidjson::Value(SkydomeFileName.c_str(), JsonAllocator), JsonAllocator);
			SkydomeObject.AddMember("OverrideCubemap", rapidjson::Value(EnableSkydome), JsonAllocator);
			rapidjson::Value CubemapFileNamesArray(rapidjson::kArrayType);
			for (const auto& fileName : CubemapFileNames) {
				rapidjson::Value strVal(fileName.c_str(), JsonAllocator);
				CubemapFileNamesArray.PushBack(strVal, JsonAllocator);
			}
			SkydomeObject.AddMember("CubemapFileNames", CubemapFileNamesArray, JsonAllocator);
			JsonDocument.AddMember("Skydome", SkydomeObject, JsonAllocator);

			rapidjson::Value BackgroundObject(rapidjson::kObjectType);
			BackgroundObject.AddMember("EnableBackground", rapidjson::Value(EnableBackground), JsonAllocator);
			BackgroundObject.AddMember("OverrideBackground", rapidjson::Value(OverrideBackground), JsonAllocator);
			BackgroundObject.AddMember("BackgroundFileName", rapidjson::Value(BackgroundFileName.c_str(), JsonAllocator), JsonAllocator);
			JsonDocument.AddMember("Background", BackgroundObject, JsonAllocator);

			auto PushLightsToJsonArray = [&JsonDocument, &JsonAllocator](rapidjson::Value& OutArray, const std::vector<XkLightDesc>& Lights) {
				for (const auto& light : Lights) {
					rapidjson::Value lightObj(rapidjson::kObjectType);
					lightObj.AddMember("Position", rapidjson::Value(rapidjson::kArrayType), JsonAllocator);
					lightObj["Position"].PushBack(light.Position.x, JsonAllocator);
					lightObj["Position"].PushBack(light.Position.y, JsonAllocator);
					lightObj["Position"].PushBack(light.Position.z, JsonAllocator);
					lightObj.AddMember("Type", rapidjson::Value(light.Type), JsonAllocator);

					lightObj.AddMember("Color", rapidjson::Value(rapidjson::kArrayType), JsonAllocator);
					lightObj["Color"].PushBack(light.Color.x, JsonAllocator);
					lightObj["Color"].PushBack(light.Color.y, JsonAllocator);
					lightObj["Color"].PushBack(light.Color.z, JsonAllocator);
					lightObj.AddMember("Intensity", rapidjson::Value(light.Intensity), JsonAllocator);

					lightObj.AddMember("Direction", rapidjson::Value(rapidjson::kArrayType), JsonAllocator);
					lightObj["Direction"].PushBack(light.Direction.x, JsonAllocator);
					lightObj["Direction"].PushBack(light.Direction.y, JsonAllocator);
					lightObj["Direction"].PushBack(light.Direction.z, JsonAllocator);
					lightObj.AddMember("Radius", rapidjson::Value(light.Radius), JsonAllocator);

					lightObj.AddMember("ExtraData", rapidjson::Value(rapidjson::kArrayType), JsonAllocator);
					lightObj["ExtraData"].PushBack(light.ExtraData.x, JsonAllocator);
					lightObj["ExtraData"].PushBack(light.ExtraData.y, JsonAllocator);
					lightObj["ExtraData"].PushBack(light.ExtraData.z, JsonAllocator);
					lightObj["ExtraData"].PushBack(light.ExtraData.w, JsonAllocator);

					OutArray.PushBack(lightObj, JsonAllocator);
				}
			};
			rapidjson::Value DirectionalLightsArray(rapidjson::kArrayType);
			PushLightsToJsonArray(DirectionalLightsArray, DirectionalLights);
			JsonDocument.AddMember("DirectionalLights", DirectionalLightsArray, JsonAllocator);
			rapidjson::Value PointLightsArray(rapidjson::kArrayType);
			PushLightsToJsonArray(PointLightsArray, PointLights);
			JsonDocument.AddMember("PointLights", PointLightsArray, JsonAllocator);
			rapidjson::Value SpotLightsArray(rapidjson::kArrayType);
			PushLightsToJsonArray(SpotLightsArray, SpotLights);
			JsonDocument.AddMember("SpotLights", SpotLightsArray, JsonAllocator);

			rapidjson::Value ObjectsArray(rapidjson::kArrayType);
			for (const auto& objDesc : ObjectDescs) {
				rapidjson::Value JsonObject(rapidjson::kObjectType);
				JsonObject.AddMember("RenderFlags", rapidjson::Value((uint16_t)objDesc.RenderFlags), JsonAllocator);
				JsonObject.AddMember("ProfabName", rapidjson::Value(objDesc.ProfabName.c_str(), JsonAllocator), JsonAllocator);
				JsonObject.AddMember("InstanceCount", rapidjson::Value(objDesc.InstanceCount), JsonAllocator);
				JsonObject.AddMember("MinRadius", rapidjson::Value(objDesc.MinRadius), JsonAllocator);
				JsonObject.AddMember("MaxRadius", rapidjson::Value(objDesc.MaxRadius), JsonAllocator);
				JsonObject.AddMember("MinRotYaw", rapidjson::Value(objDesc.MinRotYaw), JsonAllocator);
				JsonObject.AddMember("MaxRotYaw", rapidjson::Value(objDesc.MaxRotYaw), JsonAllocator);
				JsonObject.AddMember("MinRotRoll", rapidjson::Value(objDesc.MinRotRoll), JsonAllocator);
				JsonObject.AddMember("MaxRotRoll", rapidjson::Value(objDesc.MaxRotRoll), JsonAllocator);
				JsonObject.AddMember("MinRotPitch", rapidjson::Value(objDesc.MinRotPitch), JsonAllocator);
				JsonObject.AddMember("MaxRotPitch", rapidjson::Value(objDesc.MaxRotPitch), JsonAllocator);
				JsonObject.AddMember("MinPScale", rapidjson::Value(objDesc.MinPScale), JsonAllocator);
				JsonObject.AddMember("MaxPScale", rapidjson::Value(objDesc.MaxPScale), JsonAllocator);
				ObjectsArray.PushBack(JsonObject, JsonAllocator);
			}
			JsonDocument.AddMember("Objects", ObjectsArray, JsonAllocator);
			
			std::vector<XkObjectDesc> Objects;
			rapidjson::StringBuffer buffer;
			rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
			JsonDocument.Accept(writer);

			FILE* fp = fopen(FilePath.c_str(), "w");
			if (!fp) {
				throw std::runtime_error("[WORLD] Failed to save file: " + FilePath);
				return;
			}

			fputs(buffer.GetString(), fp);
			fclose(fp);
		}

		void Reset()
		{
			EnableSkydome = true;
			OverrideSkydome = true;
			SkydomeFileName = "Content/Textures/skydome.png";

			OverrideCubemap = true;
			CubemapFileNames = {
			"Content/Textures/cubemap_X0.png",
			"Content/Textures/cubemap_X1.png",
			"Content/Textures/cubemap_Y2.png",
			"Content/Textures/cubemap_Y3.png",
			"Content/Textures/cubemap_Z4.png",
			"Content/Textures/cubemap_Z5.png"};
			
			EnableBackground = true;
			OverrideBackground = true;
			BackgroundFileName = "Content/Textures/background.png";

			DirectionalLights.clear();
			PointLights.clear();
			SpotLights.clear();
			//QuadLights.clear();

			ObjectDescs.clear();
		}
	} World;

	/** GBuffer for Deferred shading*/
	struct XkGBuffer {
		// Depth Stencil RGBAFloat
		VkFormat DepthStencilFormat;
		VkImage DepthStencilImage;
		VkDeviceMemory DepthStencilMemory;
		VkImageView DepthStencilImageView;
		VkSampler DepthStencilSampler;
		// SceneColorDeferred RGBAHalf
		VkFormat SceneColorFormat;
		VkImage SceneColorImage;
		VkDeviceMemory SceneColorMemory;
		VkImageView SceneColorImageView;
		VkSampler SceneColorSampler;
		// Normal+CastShadow R10G10B10A2
		VkFormat GBufferAFormat;
		VkImage GBufferAImage;
		VkDeviceMemory GBufferAMemory;
		VkImageView GBufferAImageView;
		VkSampler GBufferASampler;
		// M+S+R+(ShadingModelID+SelectiveOutputMask) RGBA8888
		VkFormat GBufferBFormat;
		VkImage GBufferBImage;
		VkDeviceMemory GBufferBMemory;
		VkImageView GBufferBImageView;
		VkSampler GBufferBSampler;
		// BaseColor+AO
		VkFormat GBufferCFormat;
		VkImage GBufferCImage;
		VkDeviceMemory GBufferCMemory;
		VkImageView GBufferCImageView;
		VkSampler GBufferCSampler;
		// Position+ID
		VkFormat GBufferDFormat;
		VkImage GBufferDImage;
		VkDeviceMemory GBufferDMemory;
		VkImageView GBufferDImageView;
		VkSampler GBufferDSampler;
		// MotionVector+Velocity(Currently not implemented)
		//VkImage GBufferVelocityImage;
		//VkDeviceMemory GBufferVelocityMemory;
		//VkImageView GBufferVelocityImageView;

		std::vector<VkImageView> ImageViews() const
		{
			return std::vector<VkImageView>{
				DepthStencilImageView,
				SceneColorImageView,
				GBufferAImageView,
				GBufferBImageView,
				GBufferCImageView,
				GBufferDImageView
			};
		};
		std::vector<VkSampler> Samplers() const
		{
			return std::vector<VkSampler>{
				DepthStencilSampler,
				SceneColorSampler,
				GBufferASampler,
				GBufferBSampler,
				GBufferCSampler,
				GBufferDSampler
			};
		}
		std::vector<VkFormat> Formats() const
		{
			return std::vector<VkFormat>{
				DepthStencilFormat,
				SceneColorFormat,
				GBufferAFormat,
				GBufferBFormat,
				GBufferCFormat,
				GBufferDFormat
			};
		}
	} GBuffer;

	/** ShadowmapPass vulkan resources*/
	struct XkShadowmapPass {
		std::vector<XkRenderObject*> RenderObjects;
		std::vector<XkRenderInstancedObject*> RenderInstancedObjects;
		std::vector<XkRenderObjectIndirect*> RenderIndirectObjects;
		std::vector<XkRenderInstancedObjectIndirect*> RenderIndirectInstancedObjects;
		float zNear, zFar;
		int32_t Width, Height;
		VkFormat Format;
		VkFramebuffer FrameBuffer;
		VkRenderPass RenderPass;
		VkImage Image;
		VkDeviceMemory Memory;
		VkImageView ImageView;
		VkSampler Sampler;
		VkDescriptorSetLayout DescriptorSetLayout;
		VkDescriptorPool DescriptorPool;
		std::vector<VkDescriptorSet> DescriptorSets;
		VkPipelineLayout PipelineLayout;
		std::vector<VkPipeline> Pipelines;
		std::vector<VkPipeline> PipelinesInstanced;
		std::vector<VkBuffer> UniformBuffers;
		std::vector<VkDeviceMemory> UniformBuffersMemory;
	} ShadowmapPass;

	/** BackgroundPass vulkan resources*/
	struct XkBackgroundPass {
		bool EnableBackground;
		std::vector<VkImage> Images;
		std::vector<VkDeviceMemory> ImageMemorys;
		std::vector<VkImageView> ImageViews;
		std::vector<VkSampler> ImageSamplers;
		VkRenderPass RenderPass;
		VkDescriptorSetLayout DescriptorSetLayout;
		VkDescriptorPool DescriptorPool;
		std::vector<VkDescriptorSet> DescriptorSets;
		VkPipelineLayout PipelineLayout;
		std::vector<VkPipeline> Pipelines;
	} BackgroundPass;

	/** SkydomePass vulkan resources*/
	struct XkSkydomePass {
		bool EnableSkydome;
		XkMesh SkydomeMesh;
		std::vector<VkImage> Images;
		std::vector<VkDeviceMemory> ImageMemorys;
		std::vector<VkImageView> ImageViews;
		std::vector<VkSampler> ImageSamplers;
		VkRenderPass RenderPass;
		VkDescriptorSetLayout DescriptorSetLayout;
		VkDescriptorPool DescriptorPool;
		std::vector<VkDescriptorSet> DescriptorSets;
		VkPipelineLayout PipelineLayout;
		std::vector<VkPipeline> Pipelines;
	} SkydomePass;

	/** 构建 BasePass 需要的 Vulkan 资源*/
	struct XkBasePass {
		std::vector<XkRenderObject*> RenderObjects;
		std::vector<XkRenderInstancedObject*> RenderInstancedObjects;
		VkDescriptorSetLayout DescriptorSetLayout;
		VkPipelineLayout PipelineLayout;
		std::vector<VkPipeline> Pipelines;
		std::vector<VkPipeline> PipelinesInstanced;

#if ENABLE_INDIRECT_DRAW
		std::vector<XkRenderObjectIndirect*> RenderIndirectObjects;
		std::vector<XkRenderInstancedObjectIndirect*> RenderIndirectInstancedObjects;
		VkDescriptorSetLayout IndirectDescriptorSetLayout;
		VkPipelineLayout IndirectPipelineLayout;
		std::vector<VkPipeline> IndirectPipelines;
		std::vector<VkPipeline> IndirectPipelinesInstanced;
#endif

#if ENABLE_DEFEERED_SHADING
		std::vector<FRenderDeferredObject*> RenderDeferredObjects;
		std::vector<FRenderDeferredInstancedObject*> RenderDeferredInstancedObjects;
		VkDescriptorSetLayout DeferredSceneDescriptorSetLayout;
		VkPipelineLayout DeferredScenePipelineLayout;
		std::vector<VkPipeline> DeferredScenePipelines;
		std::vector<VkPipeline> DeferredScenePipelinesInstanced;
		VkFramebuffer DeferredSceneFrameBuffer;
		VkRenderPass DeferredSceneRenderPass;
		VkDescriptorSetLayout DeferredLightingDescriptorSetLayout;
		VkDescriptorPool DeferredLightingDescriptorPool;
		std::vector<VkDescriptorSet> DeferredLightingDescriptorSets;
		VkPipelineLayout DeferredLightingPipelineLayout;
		std::vector<VkPipeline> DeferredLightingPipelines;
#endif
	} BasePass;

	struct XkImGuiPass {
		VkRenderPass RenderPass;
		VkDescriptorPool DescriptorPool;

		float RightBarSpace;
		float BottomBarSpace;

		ImFont* RobotoSlabBold;
		ImFont* RobotoSlabMedium;
		ImFont* RobotoSlabRegular;
		ImFont* RobotoSlabSemiBold;
		ImFont* SourceCodeProMedium;
	} ImGuiPass;

	/* GLFW Window */
	GLFWwindow* Window;
	/* Vulkan Instance link to Window*/
	VkInstance Instance;
	/* Vulkan Pipeline Cache*/
	VkPipelineCache PipelineCache = VK_NULL_HANDLE;
	/* Vulkan Debug Messenger*/
	VkAllocationCallbacks* Allocator = VK_NULL_HANDLE;
	/* Vulkan Debug Messenger*/
	VkDebugUtilsMessengerEXT DebugMessenger;
	/* Surface link to Vulkan Instance*/
	VkSurfaceKHR Surface;

	/* Logical Device Queue Family Indices*/
	XkQueueFamilyIndices QueueFamilyIndices;
	/* Physical Device of GPU hardware*/
	VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
	/* Logic hardware, refer to physical device*/
	VkDevice Device;

	/* Graphic queue*/
	VkQueue GraphicsQueue;
	/* Display present queue*/
	VkQueue PresentQueue;

	/* Swap chain image button to sync to display*/
	VkSwapchainKHR SwapChain;
	/* Render image queue*/
	std::vector<VkImage> SwapChainImages;
	/* Render image format*/
	VkFormat SwapChainImageFormat;
	/* Render extent*/
	VkExtent2D SwapChainExtent;
	/* Render image view extent*/
	std::vector<VkImageView> SwapChainImageViews;
	/* Swap chain frame buffers*/
	std::vector<VkFramebuffer> SwapChainFramebuffers;
	/* Main render pass*/
	VkRenderPass MainRenderPass;

	/* Depth image attach to main render pass*/
	VkImage DepthImage;
	/* Depth image memory */
	VkDeviceMemory DepthImageMemory;
	/* Depth image view for shadow depth sampling */
	VkImageView DepthImageView;

	/* Cubemap max mips */
	uint32_t CubemapMaxMips;
	/* Cubemap image */
	VkImage CubemapImage;
	/* Cubemap image memory*/
	VkDeviceMemory CubemapImageMemory;
	/* Cubemap image view */
	VkImageView CubemapImageView;
	/* Cubemap image sampler */
	VkSampler CubemapSampler;

	/* Uniform buffers for Base */
	std::vector<VkBuffer> BaseUniformBuffers;
	/* Uniform buffers memory for Base */
	std::vector<VkDeviceMemory> BaseUniformBuffersMemory;

	/* Uniform buffers for View */
	std::vector<VkBuffer> ViewUniformBuffers;
	/* Uniform buffers memory for View */
	std::vector<VkDeviceMemory> ViewUniformBuffersMemory;

	/* Command pool */
	VkCommandPool CommandPool;
	/* Command buffers */
	std::vector<VkCommandBuffer> CommandBuffers;
	/* Semaphores check image is available */
	std::vector<VkSemaphore> ImageAvailableSemaphores;
	/* Semaphores check renders is finished */
	std::vector<VkSemaphore> RenderFinishedSemaphores;
	/* Fence, to wait for last frame finish rendering */
	std::vector<VkFence> InFlightFences;
	/* Current frame number */
	uint32_t CurrentFrame = 0;

	/* Restore frame buffer resized state */
	bool bFramebufferResized = false;
public:
	/** Main Function */
	void Run()
	{
		InitWindow(); // Init GLFW window
		InitVulkan(); // Init Vulkan rendering pipeline
		MainTick(); // Main tick loop
		DestroyVulkan(); // Destroy Vulkan rendering pipeline
		DestroyWindow(); // Destroy GLFW window
	}

public:
	/** Init GUI window */
	void InitWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		uint32_t viewportWidth = VIEWPORT_WIDTH;
		uint32_t viewportHeight = VIEWPORT_HEIGHT;
		Window = glfwCreateWindow(viewportWidth, viewportHeight, "Zelda Engine ©XUKAI", nullptr /* glfwGetPrimaryMonitor() full screen mode*/, nullptr);
		glfwSetWindowUserPointer(Window, this);
		glfwSetFramebufferSizeCallback(Window, FramebufferResizeCallback);
		GLFWimage iconImages[2];
		iconImages[0].pixels = stbi_load("Content/AppData/vulkan_renderer.png", &iconImages[0].width, &iconImages[0].height, 0, STBI_rgb_alpha);
		iconImages[1].pixels = stbi_load("Content/AppData/vulkan_renderer_small.png", &iconImages[1].width, &iconImages[1].height, 0, STBI_rgb_alpha);
		glfwSetWindowIcon(Window, 2, iconImages);
		stbi_image_free(iconImages[0].pixels);
		stbi_image_free(iconImages[1].pixels);

		GlobalInput.InitCamera(&World.MainCamera);
		GlobalInput.ResetToFocus();
		GlobalInput.ResetAnimation();
		GlobalConstants.ResetConstants();

		glfwSetKeyCallback(Window, KeyboardCallback);
		glfwSetMouseButtonCallback(Window, MouseButtonCallback);
		glfwSetCursorPosCallback(Window, MousePositionCallback);
		glfwSetScrollCallback(Window, MouseScrollCallback);

		// we raise a thread to listen to the python IDE
		std::thread listenThread([this]
			{
#ifdef _WIN32
				struct addrinfo* result = NULL;
				struct addrinfo hints;
				int iResult, iSendResult;
				iResult = WSAStartup(MAKEWORD(2, 2), &SocketListener.wsaData);
				if (iResult != 0) {
					std::cerr << "[Socket] WSAStartup failed: " << iResult << std::endl;
					return;
				}

				ZeroMemory(&hints, sizeof(hints));
				hints.ai_family = AF_INET; // IPv4
				hints.ai_socktype = SOCK_STREAM; // Stream socket
				hints.ai_protocol = IPPROTO_TCP; // TCP
				hints.ai_flags = AI_PASSIVE; // For wild card IP address

				// Resolve the local address and port to be used by the server
				iResult = getaddrinfo(NULL, "8080", &hints, &result);
				if (iResult != 0) {
					std::cerr << "[Socket] getaddrinfo failed: " << iResult << std::endl;
					WSACleanup();
					return;
				}

				SocketListener.ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
				if (SocketListener.ListenSocket == INVALID_SOCKET) {
					std::cerr << "[Socket] socket failed: " << WSAGetLastError() << std::endl;
					freeaddrinfo(result);
					WSACleanup();
					return;
				}

				iResult = bind(SocketListener.ListenSocket, result->ai_addr, (int)result->ai_addrlen);
				if (iResult == SOCKET_ERROR) {
					std::cerr << "[Socket] bind failed: " << WSAGetLastError() << std::endl;
					freeaddrinfo(result);
					closesocket(SocketListener.ListenSocket);
					WSACleanup();
					return;
				}

				freeaddrinfo(result);

				iResult = listen(SocketListener.ListenSocket, SOMAXCONN);
				if (iResult == SOCKET_ERROR) {
					std::cerr << "[Socket] listen failed: " << WSAGetLastError() << std::endl;
					closesocket(SocketListener.ListenSocket);
					WSACleanup();
					return;
				}

				std::cout << "[Socket] Listening on port 8080..." << std::endl;

				// Accept a client socket
				while (true)
				{
					SocketListener.ClientSocket = accept(SocketListener.ListenSocket, NULL, NULL);
					if (SocketListener.ClientSocket == INVALID_SOCKET)
					{
						std::cerr << "[Socket] accept failed: " << WSAGetLastError() << std::endl;
						continue; // 继续监听下一个连接，而不是退出程序
					}

					// accept data
					iResult = recv(SocketListener.ClientSocket, SocketListener.recvbuf, SocketListener.recvbuflen, 0);
					if (iResult > 0) {
						std::cout << "[Socket] Received: " << std::string(SocketListener.recvbuf, 0, iResult) << std::endl;
						SocketListener.receivedData = std::string(SocketListener.recvbuf, iResult);
						RecreateWorldAndScene();
					}
					else if (iResult == 0)
						std::cout << "[Socket] Connection closing..." << std::endl;
					else {
						std::cerr << "[Socket] recv failed: " << WSAGetLastError() << std::endl;
						closesocket(SocketListener.ClientSocket);
						WSACleanup();
						return;
					}

					// Shutdown the connection since we're done
					iResult = shutdown(SocketListener.ClientSocket, SD_SEND);
					if (iResult == SOCKET_ERROR) {
						std::cerr << "[Socket] shutdown failed: " << WSAGetLastError() << std::endl;
					}
				}

				SocketListener.Release();
#else
				// @TODO: Implement for macOS and Linux
#endif				
			});
		listenThread.detach(); // Detach the thread to run independently
	}

	/** Init Vulkan render pipeline */
	void InitVulkan()
	{
		CreateInstance(); // Create Vulkan instance, driven by GPU driver
		CreateDebugMessenger(); // Create debug output massager
		CreateWindowsSurface(); // Link GLFW window to vulkan rendering instance
		CreateLogicalDevice(); // Create logic device from physical device we found
		CreateSwapChain(); // // Create SwapChain for render image and display present image
		CreateSwapChainImageViews(); // Create SwapChain image view
		CreateRenderPass(); // Create main SwapChain render pass
		CreateFramebuffers(); // Create main SwapChain frame buffers
		CreateCommandPool(); // Create command pool to keep all commands inside
		// CreateShaderSPIRVs(); // Create and compile shader into SPIRV hardware-agnostic intermediate bytecode
		CreateUniformBuffers(); // Create uniform buffers
		CreateShadowmapPass(); // Create shadow depths render pass
		CreateSkydomePass(); // Create sky dome sphere render pass
		CreateBasePass(); // Create base scene render pass
		CreateBackgroundPass(); // Create background rect render pass
		CreateCommandBuffer(); // Create command buffer from command before submit
		CreateSyncObjects(); // Create sync fence to ensure next frame render after the last frame finished
#if ENABLE_IMGUI_EDITOR
		CreateImGuiForVulkan(); // Create ImGui for Vulkan
#endif
		CreateEngineWorld(); // Create main world
		CreateEngineScene(); // Create main rendering scene
	}

	/** Main tick to submit render command */
	void MainTick()
	{
		while (!glfwWindowShouldClose(Window))
		{
			glfwSwapBuffers(Window);
			glfwPollEvents();

			DrawFrame(); // Draw a frame
		}

		vkDeviceWaitIdle(Device);
	}

	/** Destroy GLFW window */
	void DestroyWindow()
	{
		glfwDestroyWindow(Window);
		glfwTerminate();

		// Release the socket listener
		SocketListener.Release();
	}

	static void FramebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<XkZeldaEngineApp*>(glfwGetWindowUserPointer(window));
		app->bFramebufferResized = true;
	}

	static void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
#if ENABLE_IMGUI_EDITOR
		if (ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)
		{
			return;
		}
#endif
		XkZeldaEngineApp* app = reinterpret_cast<XkZeldaEngineApp*>(glfwGetWindowUserPointer(window));
		XkGlobalInput* input = &app->GlobalInput;
		XkGlobalConstants* constants = &app->GlobalConstants;

		if (action == GLFW_PRESS && key == GLFW_KEY_F)
		{
			input->ResetToFocus();
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_R)
		{
			input->ResetAnimation();
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_G)
		{
			input->bGameMode = !input->bGameMode;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_M)
		{
			input->bPlayStageRoll = !input->bPlayStageRoll;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_L)
		{
			input->bPlayLightRoll = !input->bPlayLightRoll;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_0)
		{
			constants->SpecConstants = 0;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_1)
		{
			constants->SpecConstants = 1;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_2)
		{
			constants->SpecConstants = 2;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_3)
		{
			constants->SpecConstants = 3;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_4)
		{
			constants->SpecConstants = 4;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_5)
		{
			constants->SpecConstants = 5;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_6)
		{
			constants->SpecConstants = 6;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_7)
		{
			constants->SpecConstants = 7;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_8)
		{
			constants->SpecConstants = 8;
		}
		if (action == GLFW_PRESS && key == GLFW_KEY_9)
		{
			constants->SpecConstants = 9;
		}
	}

	static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
	{
#if ENABLE_IMGUI_EDITOR
		if (ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)
		{
			return;
		}
#endif
		XkZeldaEngineApp* app = reinterpret_cast<XkZeldaEngineApp*>(glfwGetWindowUserPointer(window));
		XkGlobalInput* input = &app->GlobalInput;

		if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT)
		{
			input->bCameraMoving = true;
			input->bInitMouse = true;
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
		if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT)
		{
			input->bCameraMoving = false;
			input->bInitMouse = true;
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	static void MousePositionCallback(GLFWwindow* window, double xpos, double ypos)
	{
#if ENABLE_IMGUI_EDITOR
		if (ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)
		{
			return;
		}
#endif
		XkZeldaEngineApp* app = reinterpret_cast<XkZeldaEngineApp*>(glfwGetWindowUserPointer(window));
		XkGlobalInput* input = &app->GlobalInput;

		if (!input->bCameraMoving)
		{
			return;
		}

		if (input->bInitMouse)
		{
			input->LastMouseX = xpos;
			input->LastMouseY = ypos;
			input->bInitMouse = false;
		}

		float Yaw = input->MainCamera->Yaw;
		float Pitch = input->MainCamera->Pitch;

		float xoffset = (float)(xpos - input->LastMouseX);
		float yoffset = (float)(ypos - input->LastMouseY);
		input->LastMouseX = xpos;
		input->LastMouseY = ypos;

		float sensitivityX = 1.0;
		float sensitivityY = 0.5;
		xoffset *= sensitivityX;
		yoffset *= sensitivityY;

		Yaw -= xoffset;
		Pitch += yoffset;
		if (Pitch > 89.0)
			Pitch = 89.0;
		if (Pitch < -89.0)
			Pitch = -89.0;

		if (input->bCameraFocus)
		{
			glm::vec3 Position = input->MainCamera->Position;
			float ArmLength = input->MainCamera->ArmLength;
			Position.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch)) * ArmLength;
			Position.y = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch)) * ArmLength;
			Position.z = sin(glm::radians(Pitch)) * ArmLength;
			input->MainCamera->Position = Position;
			input->MainCamera->Yaw = Yaw;
			input->MainCamera->Pitch = Pitch;
		}
		else
		{
			// @TODO: 鼠标右键控制相机的旋转
		}
	}

	static void MouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		if (ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)
		{
			return;
		}

		XkZeldaEngineApp* app = reinterpret_cast<XkZeldaEngineApp*>(glfwGetWindowUserPointer(window));
		XkGlobalInput* input = &app->GlobalInput;

		if (input->bCameraFocus)
		{
			glm::vec3 Position = input->MainCamera->Position;
			glm::vec3 Lookat = input->MainCamera->Lookat;
			glm::vec3 lookatToPos = Lookat - Position;
			glm::vec3 Direction = glm::normalize(lookatToPos);
			float CameraDeltaMove = (float)yoffset * 0.5f;
			float camerArm = input->MainCamera->ArmLength;
			camerArm += CameraDeltaMove;
			camerArm = glm::max(camerArm, 1.0f);
			Position = Lookat - camerArm * Direction;
			input->MainCamera->Position = Position;
			input->MainCamera->ArmLength = camerArm;
		}
	}

	/** Draw a frame */
	void DrawFrame()
	{
		// if scene reload, wait for all frame finish rendering
		if (Scene.bReload)
		{
			for (size_t i = 0; i < InFlightFences.size(); i++)
			{
				vkWaitForFences(Device, 1, &InFlightFences[i], VK_TRUE, UINT64_MAX);
			}
			CreateEngineScene();
			Scene.bReload = false;
		}

		if(bFramebufferResized)
		{
			for (size_t i = 0; i < InFlightFences.size(); i++)
			{
				vkWaitForFences(Device, 1, &InFlightFences[i], VK_TRUE, UINT64_MAX);
			}

			bFramebufferResized = false;

			// @TODO: recreate swap chain on macOS
			RecreateSwapChain();
			return;
		}
		// Wait for the previous frame to finish rendering
		vkWaitForFences(Device, 1, &InFlightFences[CurrentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(Device, SwapChain, UINT64_MAX, ImageAvailableSemaphores[CurrentFrame], VK_NULL_HANDLE, &imageIndex);

		// If the window is out of date (window size changed or window minimized and then restored), recreate the SwapChain and stop rendering for this frame
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			RecreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("[DrawFrane] Failed to acquire swap chain image!");
		}

		// update os inputs
		UpdateWorld();
#if ENABLE_IMGUI_EDITOR
		// update imgui	widgets
		UpdateImGuiWidgets();
#endif
		// update uniform buffer（UBO）
		UpdateUniformBuffer(CurrentFrame);

		vkResetFences(Device, 1, &InFlightFences[CurrentFrame]);

		// Clear the render command buffer
		vkResetCommandBuffer(CommandBuffers[CurrentFrame], /*VkCommandBufferResetFlagBits*/ 0);
		// Record new render command buffers
		RecordCommandBuffer(CommandBuffers[CurrentFrame], imageIndex);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { ImageAvailableSemaphores[CurrentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &CommandBuffers[CurrentFrame];

		VkSemaphore signalSemaphores[] = { RenderFinishedSemaphores[CurrentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		// Submit render command
		if (vkQueueSubmit(GraphicsQueue, 1, &submitInfo, InFlightFences[CurrentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("[DrawFrame] Failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { SwapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;

		presentInfo.pImageIndices = &imageIndex;

		vkQueuePresentKHR(PresentQueue, &presentInfo);

		CurrentFrame = (CurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

protected:
	/** Establishes the connection between the program and Vulkan, involving specific details between the program and the graphics card driver. */
	void CreateInstance()
	{
		if (bEnableValidationLayers && !CheckValidationLayerSupport())
		{
			throw std::runtime_error("[CreateInstance] Validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Zelda Engine Renderer";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "Zelda Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
#ifdef __APPLE__
		createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
		// 获取需要的glfw拓展名
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (bEnableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}
#ifdef __APPLE__
		// Fix issue on Mac(m2) "vkCreateInstance: Found no drivers!"
		extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		// Issue on Mac(m2), it's not a error, but a warning, ignore it by this time~
		// vkCreateDevice():  VK_KHR_portability_subset must be enabled because physical device VkPhysicalDevice 0x600003764f40[] supports it. The Vulkan spec states: If the VK_KHR_portability_subset extension is included in pProperties of vkEnumerateDeviceExtensionProperties, ppEnabledExtensionNames must include "VK_KHR_portability_subset"
		extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (bEnableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(ValidationLayers.size());
			createInfo.ppEnabledLayerNames = ValidationLayers.data();

			PopulateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, Allocator, &Instance) != VK_SUCCESS)
		{
			throw std::runtime_error("[CreateInstance] Failed to Create Instance!");
		}
	}

	/** Validation Layers
	 *    - Check parameter specifications and usage
	 *    - Track object creation and destruction to detect resource leaks
	 *    - Ensure thread safety by tracing the original thread calls
	 *    - Print output for each call
	 *    - Optimize and trace Vulkan calls for debugging purposes
	*/
	void CreateDebugMessenger()
	{
		if (!bEnableValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		PopulateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(Instance, &createInfo, nullptr, &DebugMessenger) != VK_SUCCESS)
		{
			throw std::runtime_error("[CreateDebugMessenger] Failed to set up debug messenger!");
		}
	}

	/** WSI (Window System Integration) links Vulkan and the Window system, rendering Vulkan to the desktop */
	void CreateWindowsSurface()
	{
		if (glfwCreateWindowSurface(Instance, Window, nullptr, &Surface) != VK_SUCCESS) {
			throw std::runtime_error("[CreateWindowsSurface] Failed to Create Window Surface!");
		}
	}

	/** Select the physical device that supports Vulkan */
	void SelectPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(Instance, &deviceCount, nullptr);

		if (deviceCount == 0)
		{
			throw std::runtime_error("[SelectPhysicalDevice] Failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(Instance, &deviceCount, devices.data());

		for (const auto& Device : devices)
		{
			if (IsDeviceSuitable(Device))
			{
				PhysicalDevice = Device;
				break;
			}
		}

		if (PhysicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("[SelectPhysicalDevice] Failed to find a suitable GPU!");
		}
	}

	/** Create logical device to interface with the physical device, multiple logical devices can correspond to the same physical device */
	void CreateLogicalDevice()
	{
		// Find the physical graphics card hardware of this computer
		SelectPhysicalDevice();

		QueueFamilyIndices = FindQueueFamilies(PhysicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { QueueFamilyIndices.GraphicsFamily.value(), QueueFamilyIndices.PresentFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.multiDrawIndirect = VK_TRUE;
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.fillModeNonSolid = ENABLE_WIREFRAME ? VK_TRUE : VK_FALSE;
#if ENABLE_BINDLESS
		VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeatures{};
		descriptorIndexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
		descriptorIndexingFeatures.pNext = nullptr;

		VkPhysicalDeviceFeatures2 deviceFeatures2{};
		deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		deviceFeatures2.pNext = &descriptorIndexingFeatures;

		// Fetch all features from physical device
		vkGetPhysicalDeviceFeatures2(PhysicalDevice, &deviceFeatures2);

		// Non-uniform indexing and update after bind
		// binding flags for textures, uniforms, and buffers
		// are required for our extension
		assert(descriptorIndexingFeatures.shaderSampledImageArrayNonUniformIndexing);
		assert(descriptorIndexingFeatures.descriptorBindingSampledImageUpdateAfterBind);
		assert(descriptorIndexingFeatures.shaderUniformBufferArrayNonUniformIndexing);
		assert(descriptorIndexingFeatures.descriptorBindingUniformBufferUpdateAfterBind);
		assert(descriptorIndexingFeatures.shaderStorageBufferArrayNonUniformIndexing);
		assert(descriptorIndexingFeatures.descriptorBindingStorageBufferUpdateAfterBind);
#endif
		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.pEnabledFeatures = &deviceFeatures;
#if ENABLE_BINDLESS
		createInfo.pNext = &deviceFeatures2;
#endif
		createInfo.enabledExtensionCount = static_cast<uint32_t>(DeviceExtensions.size());
		createInfo.ppEnabledExtensionNames = DeviceExtensions.data();
		createInfo.enabledLayerCount = 0;
		if (bEnableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(ValidationLayers.size());
			createInfo.ppEnabledLayerNames = ValidationLayers.data();
		}

		if (vkCreateDevice(PhysicalDevice, &createInfo, nullptr, &Device) != VK_SUCCESS)
		{
			throw std::runtime_error("[CreateLogicalDevice] Failed to Create logical Device!");
		}

		vkGetDeviceQueue(Device, QueueFamilyIndices.GraphicsFamily.value(), 0, &GraphicsQueue);
		vkGetDeviceQueue(Device, QueueFamilyIndices.PresentFamily.value(), 0, &PresentQueue);
	}

	/** Swap Chain
	 * Vulkan structure that holds frame buffers
	 * Swap Chain holds a queue of images to be displayed on the window
	 * Typically, Vulkan acquires an image, renders onto it, and then pushes the image into the Swap Chain's queue of images
	 * The Swap Chain displays the images, usually synchronized with the screen refresh rate
	*/
	void CreateSwapChain()
	{
		XkSwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(PhysicalDevice);

		VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.Formats);
		VkPresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.PresentModes);
		VkExtent2D extent = ChooseSwapExtent(swapChainSupport.Capabilities);

		uint32_t imageCount = swapChainSupport.Capabilities.minImageCount + 1;
		if (swapChainSupport.Capabilities.maxImageCount > 0 && imageCount > swapChainSupport.Capabilities.maxImageCount)
		{
			imageCount = swapChainSupport.Capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = Surface;

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		XkQueueFamilyIndices queue_family_indices = FindQueueFamilies(PhysicalDevice);
		uint32_t queueFamilyIndices[] = { queue_family_indices.GraphicsFamily.value(), queue_family_indices.PresentFamily.value() };

		if (queue_family_indices.GraphicsFamily != queue_family_indices.PresentFamily)
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = swapChainSupport.Capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;

		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(Device, &createInfo, nullptr, &SwapChain) != VK_SUCCESS)
		{
			throw std::runtime_error("[CreateSwapChain] Failed to Create swap chain!");
		}

		vkGetSwapchainImagesKHR(Device, SwapChain, &imageCount, nullptr);
		SwapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(Device, SwapChain, &imageCount, SwapChainImages.data());

		SwapChainImageFormat = surfaceFormat.format;
		SwapChainExtent = extent;
	}

	/** Recreate SwapChain*/
	void RecreateSwapChain()
	{
		// When both the width and height of the window are zero, it means the window has been minimized, so we need to wait
		int Width = 0, Height = 0;
		glfwGetFramebufferSize(Window, &Width, &Height);
		while (Width == 0 || Height == 0) {
			glfwGetFramebufferSize(Window, &Width, &Height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(Device);

		CleanupSwapChain();
		CleanupBasePass();

		CreateSwapChain();
		CreateSwapChainImageViews();
		CreateFramebuffers();
		CreateBasePass();
	}

	/** Image View
	 * Displays the view as an image
	 * ImageView defines what the images in the SwapChain look like
	 * For example, an RGB image with depth information
	*/
	void CreateSwapChainImageViews()
	{
		SwapChainImageViews.resize(SwapChainImages.size());

		for (size_t i = 0; i < SwapChainImages.size(); i++)
		{
			CreateImageView(SwapChainImageViews[i], SwapChainImages[i], SwapChainImageFormat);
		}
	}

	/** RenderPass
	 * Before creating a render pipeline, we need to create a render pass to specify the frame buffers used during rendering.
	 * We need to specify the number of color and depth buffers used in the render pass, as well as their sampling information.
	*/
	void CreateRenderPass()
	{
		VkAttachmentDescription ColorAttachment{};
		ColorAttachment.format = SwapChainImageFormat;
		ColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		ColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		ColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		ColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		ColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		ColorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		ColorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentDescription DepthAttachment{};
		DepthAttachment.format = FindDepthFormat();
		DepthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
#if !ENABLE_DEFEERED_SHADING
		DepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		DepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
#else
		DepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		DepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
#endif
		DepthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		DepthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
#if !ENABLE_DEFEERED_SHADING
		DepthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
#else
		DepthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
#endif
		DepthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference ColorAttachmentRef{};
		ColorAttachmentRef.attachment = 0;
		ColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference DepthAttachmentRef{};
		DepthAttachmentRef.attachment = 1;
		DepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// Subpass for rendering
		// A subpass is a subordinate task of a RenderPass, sharing rendering resources such as Framebuffer
		// Some rendering operations, such as post-processing like Blooming, depend on the previous rendering result but with unchanged rendering resources. Subpass can optimize performance in such cases.
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &ColorAttachmentRef;
		subpass.pDepthStencilAttachment = &DepthAttachmentRef;

		// Simplify the rendering of the triangle into a single SubPass submission here
		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		std::array<VkAttachmentDescription, 2> attachments = { ColorAttachment, DepthAttachment };
		VkRenderPassCreateInfo renderPassCI{};
		renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassCI.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassCI.pAttachments = attachments.data();
		renderPassCI.subpassCount = 1;
		renderPassCI.pSubpasses = &subpass;
		renderPassCI.dependencyCount = 1;
		renderPassCI.pDependencies = &dependency;
		if (vkCreateRenderPass(Device, &renderPassCI, nullptr, &MainRenderPass) != VK_SUCCESS) {
			throw std::runtime_error("[CreateRenderPass] Failed to Create render pass!");
		}
	}

	/** Create framebuffers, which hold the rendering data for each frame */
	void CreateFramebuffers()
	{
		// Create depth texture resources
		VkFormat depthFormat = FindDepthFormat();
		CreateImage(DepthImage, DepthImageMemory, SwapChainExtent.width, SwapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		CreateImageView(DepthImageView, DepthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

		SwapChainFramebuffers.resize(SwapChainImageViews.size());

		for (size_t i = 0; i < SwapChainImageViews.size(); i++)
		{
			std::array<VkImageView, 2> attachments =
			{
				SwapChainImageViews[i],
				DepthImageView
			};
			VkFramebufferCreateInfo frameBufferCI{};
			frameBufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frameBufferCI.renderPass = MainRenderPass;
			frameBufferCI.attachmentCount = static_cast<uint32_t>(attachments.size());
			frameBufferCI.pAttachments = attachments.data();
			frameBufferCI.width = SwapChainExtent.width;
			frameBufferCI.height = SwapChainExtent.height;
			frameBufferCI.layers = 1;

			if (vkCreateFramebuffer(Device, &frameBufferCI, nullptr, &SwapChainFramebuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("[CreateFramebuffers] Failed to Create framebuffer!");
			}
		}
	}

	/** Create command pool to manage all commands, such as DrawCall or memory transfers */
	void CreateCommandPool()
	{
		XkQueueFamilyIndices queueFamilyIndices = FindQueueFamilies(PhysicalDevice);

		VkCommandPoolCreateInfo poolCI{};
		poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolCI.queueFamilyIndex = queueFamilyIndices.GraphicsFamily.value();

		if (vkCreateCommandPool(Device, &poolCI, nullptr, &CommandPool) != VK_SUCCESS)
		{
			throw std::runtime_error("[CreateCommandPool] Failed to Create command pool!");
		}
	}

	void CreateShaderSPIRVs()
	{
		XkString file_path = __FILE__;
		XkString dir_path = file_path.substr(0, file_path.rfind("\\"));
		// @TODO: compile shaders with glslang
		std::vector<uint8_t> Source; VkShaderStageFlagBits ShaderStage;
		XkShaderCompiler::ReadShaderFile(Source, ShaderStage, "Content/meshshader.mesh");
		std::vector<uint32_t> spirv;
		XkString info_log;
		XkShaderCompiler ShaderCompiler;
		ShaderCompiler.CompileToSpirv(ShaderStage, Source, "main", "", {}, spirv, info_log);
		XkShaderCompiler::SaveShaderFile("Content/Meshshader.spv", spirv);
	}

	/** Create uniform buffers (UBO) */
	void CreateUniformBuffers()
	{
		VkDeviceSize baseBufferSize = sizeof(XkUniformBufferObject);
		BaseUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		BaseUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			CreateBuffer(baseBufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				BaseUniformBuffers[i],
				BaseUniformBuffersMemory[i]);

			// 这里会导致 memory stack overflow ，不应该在这里 vkMapMemory
			//vkMapMemory(Device, BaseUniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}

		VkDeviceSize bufferSizeOfView = sizeof(XkView);
		ViewUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		ViewUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			CreateBuffer(
				bufferSizeOfView,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				ViewUniformBuffers[i],
				ViewUniformBuffersMemory[i]);
		}
	}


	/**
	 * Create shadow map resources
	*/
	void CreateShadowmapPass()
	{
		ShadowmapPass.Width = SHADOWMAP_DIM;
		ShadowmapPass.Height = SHADOWMAP_DIM;
		ShadowmapPass.Format = FindDepthFormat();
		CreateImage(
			ShadowmapPass.Image,
			ShadowmapPass.Memory,
			ShadowmapPass.Width, ShadowmapPass.Height, ShadowmapPass.Format,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		CreateImageView(ShadowmapPass.ImageView, ShadowmapPass.Image, ShadowmapPass.Format, VK_IMAGE_ASPECT_DEPTH_BIT);
		CreateSampler(ShadowmapPass.Sampler,
			VK_FILTER_LINEAR,
			VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE); // @TODO: 是否应该将 sample filter 改成 VK_FILTER_NEAREST

		VkAttachmentDescription attachmentDescription{};
		attachmentDescription.format = ShadowmapPass.Format;
		attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
		// Clear depth at beginning of the render pass
		attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		// We will read from depth, so it's important to store the depth attachment results
		attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		// We don't care about initial layout of the attachment
		attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		// Attachment will be transitioned to shader read at render pass end
		attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 0;
		// Attachment will be used as depth/stencil during render pass
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		// No color attachments
		subpass.colorAttachmentCount = 0;
		// Reference to our depth attachment
		subpass.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassCreateInfo{};
		renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassCreateInfo.attachmentCount = 1;
		renderPassCreateInfo.pAttachments = &attachmentDescription;
		renderPassCreateInfo.subpassCount = 1;
		renderPassCreateInfo.pSubpasses = &subpass;
		renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());;
		renderPassCreateInfo.pDependencies = dependencies.data();

		if (vkCreateRenderPass(Device, &renderPassCreateInfo, nullptr, &ShadowmapPass.RenderPass)) {
			throw std::runtime_error("[CreateShadowmapPass] Failed to Create shadow map render pass!");
		}

		// Create frame buffer
		VkFramebufferCreateInfo frameBufferCreateInfo{};
		frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		frameBufferCreateInfo.renderPass = ShadowmapPass.RenderPass;
		frameBufferCreateInfo.attachmentCount = 1;
		frameBufferCreateInfo.pAttachments = &ShadowmapPass.ImageView;
		frameBufferCreateInfo.width = ShadowmapPass.Width;
		frameBufferCreateInfo.height = ShadowmapPass.Height;
		frameBufferCreateInfo.layers = 1;

		if (vkCreateFramebuffer(Device, &frameBufferCreateInfo, nullptr, &ShadowmapPass.FrameBuffer)) {
			throw std::runtime_error("[CreateShadowmapPass] Failed to Create shadow map frame buffer!");
		}


		//////////////////////////////////////////////////////////
		// Create UniformBuffers and UniformBuffersMemory
		VkDeviceSize bufferSize = sizeof(XkUniformBufferObject);
		ShadowmapPass.UniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		ShadowmapPass.UniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			CreateBuffer(bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				ShadowmapPass.UniformBuffers[i],
				ShadowmapPass.UniformBuffersMemory[i]);
		}

		//////////////////////////////////////////////////////////
		// Create DescriptorSetLayout
		CreateDescriptorSetLayout(ShadowmapPass.DescriptorSetLayout, EXkRenderFlags::Shadow);

		//////////////////////////////////////////////////////////
		// Bind DescriptorSet
		CreateDescriptorSet(ShadowmapPass.DescriptorSets, ShadowmapPass.DescriptorPool, ShadowmapPass.DescriptorSetLayout, 
			std::vector<VkImageView>(), std::vector<VkSampler>{}, EXkRenderFlags::Shadow);

		CreatePipelineLayout(ShadowmapPass.PipelineLayout, ShadowmapPass.DescriptorSetLayout);

		ShadowmapPass.Pipelines.resize(1);
		ShadowmapPass.PipelinesInstanced.resize(1);

		CreateGraphicsPipelines(
			ShadowmapPass.Pipelines,
			ShadowmapPass.PipelineLayout,
			ShadowmapPass.RenderPass,
			"Shaders/Shadowmap_VS.spv",
			"Shaders/Shadowmap_FS.spv", 
			EXkRenderFlags::Shadow);

		CreateGraphicsPipelines(
			ShadowmapPass.PipelinesInstanced,
			ShadowmapPass.PipelineLayout,
			ShadowmapPass.RenderPass,
			"Shaders/ShadowmapInstanced_VS.spv",
			"Shaders/Shadowmap_FS.spv",
		EXkRenderFlags::Instanced | EXkRenderFlags::Shadow);
	}

	void CreateBackgroundPass()
	{
		uint32_t ImageNum = BG_SAMPLER_NUMBER;
		BackgroundPass.Images.resize(ImageNum);
		BackgroundPass.ImageMemorys.resize(ImageNum);
		BackgroundPass.ImageViews.resize(ImageNum);
		BackgroundPass.ImageSamplers.resize(ImageNum);
		CreateImageContext(
			BackgroundPass.Images[0],
			BackgroundPass.ImageMemorys[0],
			BackgroundPass.ImageViews[0],
			BackgroundPass.ImageSamplers[0],
			"Content/Textures/background.png");
		CreateDescriptorSetLayout(BackgroundPass.DescriptorSetLayout, EXkRenderFlags::Background);
		CreateDescriptorSet(
			BackgroundPass.DescriptorSets,
			BackgroundPass.DescriptorPool,
			BackgroundPass.DescriptorSetLayout,
			BackgroundPass.ImageViews, BackgroundPass.ImageSamplers, EXkRenderFlags::Background);

		uint32_t PipelineNum = 1;
		BackgroundPass.Pipelines.resize(PipelineNum);
		CreatePipelineLayout(BackgroundPass.PipelineLayout, BackgroundPass.DescriptorSetLayout, EXkRenderFlags::Background);
		CreateGraphicsPipelines(
			BackgroundPass.Pipelines,
			BackgroundPass.PipelineLayout,
			MainRenderPass,
			"Shaders/Background_VS.spv",
			"Shaders/Background_FS.spv",
			EXkRenderFlags::Background | EXkRenderFlags::ScreenRect);
	}

	void CreateSkydomePass()
	{
		/** Cube map Faces Rules:
		 *       Y3
		 *       ||
		 * X1 == Z4 == X0 == Z5
		 *       ||
		 *       Y4
		 * | https://matheowis.github.io/HDRI-to-CubeMap/    |
		 * | X0(p-x) | X1(n-x) | Y2(p-z) | Y3(n-z) | Z4(n-y) | Z5(p-y)|
		 * |    90   |   -90   |    0    |   180   |    0    |   180  |
		 */
		CreateImageCubeContext(CubemapImage, CubemapImageMemory, CubemapImageView, CubemapSampler, CubemapMaxMips, {
			"Content/Textures/cubemap_X0.png",
			"Content/Textures/cubemap_X1.png",
			"Content/Textures/cubemap_Y2.png",
			"Content/Textures/cubemap_Y3.png",
			"Content/Textures/cubemap_Z4.png",
			"Content/Textures/cubemap_Z5.png" });

		uint32_t ImageNum = SKY_SAMPLER_NUMBER;
		SkydomePass.Images.resize(ImageNum);
		SkydomePass.ImageMemorys.resize(ImageNum);
		SkydomePass.ImageSamplers.resize(ImageNum);
		SkydomePass.ImageViews.resize(ImageNum);
		SkydomePass.ImageSamplers.resize(ImageNum);

		CreateImageContext(
			SkydomePass.Images[0],
			SkydomePass.ImageMemorys[0],
			SkydomePass.ImageViews[0],
			SkydomePass.ImageSamplers[0],
			"Content/Textures/skydome.png");
		CreateDescriptorSetLayout(SkydomePass.DescriptorSetLayout, EXkRenderFlags::Skydome);
		CreateDescriptorSet(
			SkydomePass.DescriptorSets,
			SkydomePass.DescriptorPool,
			SkydomePass.DescriptorSetLayout,
			SkydomePass.ImageViews, SkydomePass.ImageSamplers,
			EXkRenderFlags::Skydome);

		uint32_t PipelineNum = 1;
		SkydomePass.Pipelines.resize(PipelineNum);
		CreatePipelineLayout(SkydomePass.PipelineLayout, SkydomePass.DescriptorSetLayout, EXkRenderFlags::Skydome);
		CreateGraphicsPipelines(
			SkydomePass.Pipelines,
			SkydomePass.PipelineLayout,
			MainRenderPass,
			"Shaders/Skydome_VS.spv",
			"Shaders/Skydome_FS.spv",
			EXkRenderFlags::VertexIndexed | EXkRenderFlags::Skydome);
		XkString skydome_obj = "Content/Meshes/skydome.obj";
		CreateMesh(SkydomePass.SkydomeMesh.Vertices, SkydomePass.SkydomeMesh.Indices, skydome_obj);
		CreateVertexBuffer(
			SkydomePass.SkydomeMesh.VertexBuffer,
			SkydomePass.SkydomeMesh.VertexBufferMemory,
			SkydomePass.SkydomeMesh.Vertices);
		CreateIndexBuffer(
			SkydomePass.SkydomeMesh.IndexBuffer,
			SkydomePass.SkydomeMesh.IndexBufferMemory,
			SkydomePass.SkydomeMesh.Indices);
	}

	void CreateBasePass()
	{
		//~ Begin create scene rendering pipeline and shaders
		{
			uint32_t SpecConstantsCount = GlobalConstants.SpecConstantsCount;
			CreateDescriptorSetLayout(BasePass.DescriptorSetLayout);
			BasePass.Pipelines.resize(SpecConstantsCount);
			BasePass.PipelinesInstanced.resize(SpecConstantsCount);
			CreatePipelineLayout(BasePass.PipelineLayout, BasePass.DescriptorSetLayout);
			CreateGraphicsPipelines(
				BasePass.Pipelines,
				BasePass.PipelineLayout,
				MainRenderPass,
				"Shaders/Base_VS.spv",
				"Shaders/Base_FS.spv",
				EXkRenderFlags::VertexIndexed);
			CreateGraphicsPipelines(
				BasePass.PipelinesInstanced,
				BasePass.PipelineLayout,
				MainRenderPass,
				"Shaders/BaseInstanced_VS.spv",
				"Shaders/Base_FS.spv",
				EXkRenderFlags::Instanced);
		}
		//~ End of creating scene, including VBO, UBO, textures, etc.

#if ENABLE_INDIRECT_DRAW
		//~ Begin create indirect BasePass
		{
			// Create scene rendering pipeline and shaders
			CreateDescriptorSetLayout(BasePass.DescriptorSetLayout);
			uint32_t SpecConstantsCount = GlobalConstants.SpecConstantsCount;
			BasePass.Pipelines.resize(SpecConstantsCount);
			BasePass.PipelinesInstanced.resize(SpecConstantsCount);
			CreatePipelineLayout(BasePass.PipelineLayout, BasePass.DescriptorSetLayout);
			CreateGraphicsPipelines(
				BasePass.Pipelines,
				BasePass.PipelineLayout,
				MainRenderPass,
				"Shaders/Base_VS.spv",
				"Shaders/Base_FS.spv",
				EXkRenderFlags::VertexIndexed);
			CreateGraphicsPipelines(
				BasePass.PipelinesInstanced,
				BasePass.PipelineLayout,
				MainRenderPass,
				"Shaders/BaseInstanced_VS.spv",
				"Shaders/Base_FS.spv",
				EXkRenderFlags::Instanced);
		}
		//~ End create indirect BasePass
#endif

#if ENABLE_DEFEERED_SHADING
		//~ Begin create deferred BasePass
		{
			// Depth Stencil (Currently depth-only)
			GBuffer.DepthStencilFormat = VK_FORMAT_D32_SFLOAT;
			CreateImage(GBuffer.DepthStencilImage, GBuffer.DepthStencilMemory, SwapChainExtent.width, SwapChainExtent.height, GBuffer.DepthStencilFormat,
				VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			CreateImageView(GBuffer.DepthStencilImageView, GBuffer.DepthStencilImage, GBuffer.DepthStencilFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
			CreateSampler(GBuffer.DepthStencilSampler);

			// Scene Color
			GBuffer.SceneColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
			CreateImage(GBuffer.SceneColorImage, GBuffer.SceneColorMemory, SwapChainExtent.width, SwapChainExtent.height, GBuffer.SceneColorFormat,
				VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			CreateImageView(GBuffer.SceneColorImageView, GBuffer.SceneColorImage, GBuffer.SceneColorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
			CreateSampler(GBuffer.SceneColorSampler);

			// GBufferA Normal+(CastShadow+Masked)
			GBuffer.GBufferAFormat = VK_FORMAT_A2R10G10B10_UNORM_PACK32;
			CreateImage(GBuffer.GBufferAImage, GBuffer.GBufferAMemory, SwapChainExtent.width, SwapChainExtent.height, GBuffer.GBufferAFormat,
				VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			CreateImageView(GBuffer.GBufferAImageView, GBuffer.GBufferAImage, GBuffer.GBufferAFormat, VK_IMAGE_ASPECT_COLOR_BIT);
			CreateSampler(GBuffer.GBufferASampler);

			// GBufferB M+S+R+(ShadingModelID+SelectiveOutputMask) for unreal
			// GBufferB M+S+R+(OpacityMask) for me
			GBuffer.GBufferBFormat = VK_FORMAT_R8G8B8A8_UNORM;
			CreateImage(GBuffer.GBufferBImage, GBuffer.GBufferBMemory, SwapChainExtent.width, SwapChainExtent.height, GBuffer.GBufferBFormat,
				VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			CreateImageView(GBuffer.GBufferBImageView, GBuffer.GBufferBImage, GBuffer.GBufferBFormat, VK_IMAGE_ASPECT_COLOR_BIT);
			CreateSampler(GBuffer.GBufferBSampler);

			// GBufferC BaseColor + AO
			GBuffer.GBufferCFormat = VK_FORMAT_R8G8B8A8_UNORM;
			CreateImage(GBuffer.GBufferCImage, GBuffer.GBufferCMemory, SwapChainExtent.width, SwapChainExtent.height, GBuffer.GBufferCFormat,
				VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			CreateImageView(GBuffer.GBufferCImageView, GBuffer.GBufferCImage, GBuffer.GBufferCFormat, VK_IMAGE_ASPECT_COLOR_BIT);
			CreateSampler(GBuffer.GBufferCSampler);

			// GBufferD Position + ID
			GBuffer.GBufferDFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
			CreateImage(GBuffer.GBufferDImage, GBuffer.GBufferDMemory, SwapChainExtent.width, SwapChainExtent.height, GBuffer.GBufferDFormat,
				VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			CreateImageView(GBuffer.GBufferDImageView, GBuffer.GBufferDImage, GBuffer.GBufferDFormat, VK_IMAGE_ASPECT_COLOR_BIT);
			CreateSampler(GBuffer.GBufferDSampler);

			VkAttachmentDescription DepthAttachment{};
			DepthAttachment.format = FindDepthFormat();
			DepthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
			DepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			DepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			DepthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			DepthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
			DepthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			DepthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

			VkAttachmentDescription ColorAttachment{};
			ColorAttachment.format = SwapChainImageFormat;
			ColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
			ColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			ColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			ColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			ColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			ColorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			ColorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			std::array<VkAttachmentDescription, GBUFFER_SAMPLER_NUMBER> AttachmentDescriptions = {};
			for (size_t i = 0; i < GBUFFER_SAMPLER_NUMBER; i++)
			{
				AttachmentDescriptions[i] = ColorAttachment;
				if (i == 0)
				{
					AttachmentDescriptions[i] = DepthAttachment;
				}
				AttachmentDescriptions[i].format = GBuffer.Formats()[i];
			}

			VkAttachmentReference DepthAttachmentRef = {};
			std::vector<VkAttachmentReference> ColorAttachmentRefs;
			for (size_t i = 0; i < GBUFFER_SAMPLER_NUMBER; i++)
			{
				if (i == 0)
				{
					DepthAttachmentRef.attachment = static_cast<uint32_t>(i);
					DepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
				}
				else
				{
					VkAttachmentReference AttachmentReference;
					AttachmentReference.attachment = static_cast<uint32_t>(i);
					AttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
					ColorAttachmentRefs.push_back(AttachmentReference);
				}
			}

			VkSubpassDescription subpass{};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = static_cast<uint32_t>(ColorAttachmentRefs.size());
			subpass.pColorAttachments = ColorAttachmentRefs.data();
			subpass.pDepthStencilAttachment = &DepthAttachmentRef;

			std::array<VkSubpassDependency, 2> dependencies;
			dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
			dependencies[0].dstSubpass = 0;
			dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
			dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

			dependencies[1].srcSubpass = 0;
			dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
			dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
			dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

			VkRenderPassCreateInfo renderPassCI = {};
			renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassCI.attachmentCount = static_cast<uint32_t>(AttachmentDescriptions.size());
			renderPassCI.pAttachments = AttachmentDescriptions.data();
			renderPassCI.subpassCount = 1;
			renderPassCI.pSubpasses = &subpass;
			renderPassCI.dependencyCount = static_cast<uint32_t>(dependencies.size());
			renderPassCI.pDependencies = dependencies.data();

			if (vkCreateRenderPass(Device, &renderPassCI, nullptr, &BasePass.DeferredSceneRenderPass) != VK_SUCCESS) {
				throw std::runtime_error("[CreateBaseDeferredPass] Failed to Create render pass!");
			}

			std::array<VkImageView, GBUFFER_SAMPLER_NUMBER> attachments;
			for (size_t i = 0; i < GBUFFER_SAMPLER_NUMBER; i++)
			{
				attachments[i] = GBuffer.ImageViews()[i];
			}

			VkFramebufferCreateInfo frameBufferCI{};
			frameBufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frameBufferCI.renderPass = BasePass.DeferredSceneRenderPass;
			frameBufferCI.attachmentCount = static_cast<uint32_t>(attachments.size());
			frameBufferCI.pAttachments = attachments.data();
			frameBufferCI.width = SwapChainExtent.width;
			frameBufferCI.height = SwapChainExtent.height;
			frameBufferCI.layers = 1;

			if (vkCreateFramebuffer(Device, &frameBufferCI, nullptr, &BasePass.DeferredSceneFrameBuffer) != VK_SUCCESS)
			{
				throw std::runtime_error("[CreateBaseDeferredPass] Failed to Create framebuffer!");
			}

			CreateDescriptorSetLayout(BasePass.DeferredSceneDescriptorSetLayout, EXkRenderFlags::DeferredScene);
			BasePass.DeferredScenePipelines.resize(GlobalConstants.SpecConstantsCount);
			BasePass.DeferredScenePipelinesInstanced.resize(GlobalConstants.SpecConstantsCount);
			CreatePipelineLayout(BasePass.DeferredScenePipelineLayout, BasePass.DeferredSceneDescriptorSetLayout);
			CreateGraphicsPipelines(
				BasePass.DeferredScenePipelines,
				BasePass.DeferredScenePipelineLayout,
				BasePass.DeferredSceneRenderPass,
				"Shaders/Base_VS.spv",
				"Shaders/BaseScene_FS.spv",
				EXkRenderFlags::VertexIndexed | EXkRenderFlags::DeferredScene);
			CreateGraphicsPipelines(
				BasePass.DeferredScenePipelinesInstanced,
				BasePass.DeferredScenePipelineLayout,
				BasePass.DeferredSceneRenderPass,
				"Shaders/BaseInstanced_VS.spv",
				"Shaders/BaseScene_FS.spv",
				EXkRenderFlags::Instanced | EXkRenderFlags::DeferredScene);

			/** Create DescriptorSetLayout for Lighting*/
			CreateDescriptorSetLayout(BasePass.DeferredLightingDescriptorSetLayout, EXkRenderFlags::DeferredLighting);

			/** Create DescriptorPool and DescriptorSets for Lighting*/
			CreateDescriptorSet(
				BasePass.DeferredLightingDescriptorSets,
				BasePass.DeferredLightingDescriptorPool,
				BasePass.DeferredLightingDescriptorSetLayout,
				GBuffer.ImageViews(),
				GBuffer.Samplers(),
				EXkRenderFlags::DeferredLighting);

			CreatePipelineLayout(BasePass.DeferredLightingPipelineLayout, BasePass.DeferredLightingDescriptorSetLayout);
			BasePass.DeferredLightingPipelines.resize(GlobalConstants.SpecConstantsCount);
			CreateGraphicsPipelines(
				BasePass.DeferredLightingPipelines,
				BasePass.DeferredLightingPipelineLayout,
				MainRenderPass,
				"Shaders/Background_VS.spv",
				"Shaders/BaseLighting_FS.spv",
				EXkRenderFlags::ScreenRect | EXkRenderFlags::NoDepthTest | EXkRenderFlags::DeferredLighting);
		}
		//~ End create deferred BasePass
#endif
	}

	/** Create command buffer, multiple CPU Cores can send commands to the CommandBuffer in parallel, making full use of the multi-core performance of the CPU */
	void CreateCommandBuffer()
	{
		CommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = CommandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)CommandBuffers.size();

		if (vkAllocateCommandBuffers(Device, &allocInfo, CommandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("[CreateCommandBuffer] Failed to allocate command buffers!");
		}
	}

	/** Create synchronization objects for synchronizing the current rendering */
	void CreateSyncObjects()
	{
		ImageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		RenderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		InFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(Device, &semaphoreInfo, nullptr, &ImageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(Device, &semaphoreInfo, nullptr, &RenderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(Device, &fenceInfo, nullptr, &InFlightFences[i]) != VK_SUCCESS) {

				throw std::runtime_error("[CreateSyncObjects] Failed to Create synchronization objects for a frame!");
			}
		}
	}

#if ENABLE_IMGUI_EDITOR
	/** Create ImGui Vulkan args*/
	void CreateImGuiForVulkan()
	{
		VkAttachmentDescription ColorAttachment{};
		ColorAttachment.format = SwapChainImageFormat;
		ColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		ColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		ColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		ColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		ColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		ColorAttachment.initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		ColorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentDescription DepthAttachment{};
		DepthAttachment.format = FindDepthFormat();
		DepthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		DepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		DepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		DepthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		DepthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		DepthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		DepthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference ColorAttachmentRef{};
		ColorAttachmentRef.attachment = 0;
		ColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference DepthAttachmentRef{};
		DepthAttachmentRef.attachment = 1;
		DepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &ColorAttachmentRef;
		subpass.pDepthStencilAttachment = &DepthAttachmentRef;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		std::array<VkAttachmentDescription, 2> attachments = { ColorAttachment, DepthAttachment };
		VkRenderPassCreateInfo renderPassCI{};
		renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassCI.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassCI.pAttachments = attachments.data();
		renderPassCI.subpassCount = 1;
		renderPassCI.pSubpasses = &subpass;
		renderPassCI.dependencyCount = 1;
		renderPassCI.pDependencies = &dependency;
		if (vkCreateRenderPass(Device, &renderPassCI, nullptr, &ImGuiPass.RenderPass) != VK_SUCCESS) {
			throw std::runtime_error("[CreateImGuiForVulkan] Failed to Create render pass!");
		}

		VkDescriptorPoolSize pool_sizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
		};

		VkDescriptorPoolCreateInfo pool_info{};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		pool_info.maxSets = IM_ARRAYSIZE(pool_sizes);
		pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
		pool_info.pPoolSizes = pool_sizes;

		if (vkCreateDescriptorPool(Device, &pool_info, Allocator, &ImGuiPass.DescriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("[CreateImGuiForVulkan] Failed to create descriptor pool!");
		}

		// Init ImGui
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		ImGuiPass.RobotoSlabBold = io.Fonts->AddFontFromFileTTF("Content/AppData/RobotoSlab-Bold.ttf", 20.0f);
		ImGuiPass.RobotoSlabMedium = io.Fonts->AddFontFromFileTTF("Content/AppData/RobotoSlab-Medium.ttf", 20.0f);
		ImGuiPass.RobotoSlabRegular = io.Fonts->AddFontFromFileTTF("Content/AppData/RobotoSlab-Regular.ttf", 20.0f);
		ImGuiPass.RobotoSlabSemiBold = io.Fonts->AddFontFromFileTTF("Content/AppData/RobotoSlab-SemiBold.ttf", 20.0f);
		ImGuiPass.SourceCodeProMedium = io.Fonts->AddFontFromFileTTF("Content/AppData/SourceCodePro-Medium.ttf", 20.0f);
		if (ImGuiPass.RobotoSlabSemiBold)
		{
			ImFontConfig fontConfig;
			fontConfig.SizePixels = 18.0f;
			io.Fonts->AddFontDefault(&fontConfig);
		}
		ImGui::StyleColorsDark();

		ImGui_ImplGlfw_InitForVulkan(Window, true);
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = Instance;
		init_info.PhysicalDevice = PhysicalDevice;
		init_info.Device = Device;
		init_info.QueueFamily = QueueFamilyIndices.GraphicsFamily.value();
		init_info.Queue = GraphicsQueue;
		init_info.PipelineCache = PipelineCache;
		init_info.DescriptorPool = ImGuiPass.DescriptorPool;
		init_info.Allocator = Allocator;
		init_info.MinImageCount = 2;
		init_info.ImageCount = 2;
		init_info.RenderPass = ImGuiPass.RenderPass;
		init_info.CheckVkResultFn = [](VkResult err)
			{
				if (err != 0)
				{
					std::cerr << "VkResult " << err << " is not VK_SUCCESS" << std::endl;
					throw std::runtime_error("[CreateImGuiForVulkan] Failed to create ImGui_ImplVulkan_InitInfo!");
				}
			};
		ImGui_ImplVulkan_Init(&init_info);
	}
#endif
	
	/** Write the commands to be executed into the command buffer, corresponding to each image of the SwapChain */
	void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
	{
		auto BeginTransitionImageLayoutRT = [commandBuffer](VkImage& image, const VkImageAspectFlagBits aspectMask, const VkImageLayout oldLayout, const VkImageLayout newLayout)
		{
			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = oldLayout;
			barrier.newLayout = newLayout;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = image;
			barrier.subresourceRange.aspectMask = aspectMask;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;

			VkPipelineStageFlags sourceStage;
			VkPipelineStageFlags destinationStage;

			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

			vkCmdPipelineBarrier(
				commandBuffer,
				sourceStage, destinationStage,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);
		};

		auto EndTransitionImageLayoutRT = [commandBuffer](VkImage& image, const VkImageAspectFlagBits aspectMask, const VkImageLayout oldLayout, const VkImageLayout newLayout)
		{
			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = oldLayout;
			barrier.newLayout = newLayout;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = image;
			barrier.subresourceRange.aspectMask = aspectMask;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;

			VkPipelineStageFlags sourceStage;
			VkPipelineStageFlags destinationStage;

			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

			vkCmdPipelineBarrier(
				commandBuffer,
				sourceStage, destinationStage,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);
		};

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		// 开始记录指令
		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
		{
			throw std::runtime_error("[RecordCommandBuffer] Failed to begin recording command buffer!");
		}

		// 【阴影】渲染阴影
		{
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = ShadowmapPass.RenderPass;
			renderPassInfo.framebuffer = ShadowmapPass.FrameBuffer;
			renderPassInfo.renderArea.extent.width = ShadowmapPass.Width;
			renderPassInfo.renderArea.extent.height = ShadowmapPass.Height;
			std::array<VkClearValue, 1> clearValues{};
			clearValues[0].depthStencil = { 1.0f, 0 };
			renderPassInfo.clearValueCount = 1;
			renderPassInfo.pClearValues = clearValues.data();

			// 【阴影】开始 RenderPass
			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			// 【阴影】视口信息
			VkViewport viewport{};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = (float)ShadowmapPass.Width;
			viewport.height = (float)ShadowmapPass.Height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			// 【阴影】视口剪切信息
			VkRect2D scissor{};
			scissor.offset = { 0, 0 };
			scissor.extent.width = ShadowmapPass.Width;
			scissor.extent.height = ShadowmapPass.Height;

			// 【阴影】设置渲染视口
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

			// 【阴影】设置视口剪切，是否可以通过这个函数来实现 Tiled-Based Rendering ？
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			// Set depth bias (aka "Polygon offset")
			// Required to avoid shadow mapping artifacts
			// Depth bias (and slope) are used to avoid shadowing artifacts
			// Constant depth bias factor (always applied)
			float depthBiasConstant = 1.25f;
			// Slope depth bias factor, applied depending on polygon's slope
			float depthBiasSlope = 7.5; // change from 1.75f to fix PCF artifact
			vkCmdSetDepthBias(
				commandBuffer,
				depthBiasConstant,
				0.0f,
				depthBiasSlope);

			// 【阴影】渲染场景
			for (size_t i = 0; i < ShadowmapPass.RenderObjects.size(); i++)
			{
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ShadowmapPass.Pipelines[0]);
				XkRenderObject* renderObject = ShadowmapPass.RenderObjects[i];
				VkBuffer objectVertexBuffers[] = { renderObject->Mesh.VertexBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, renderObject->Mesh.IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
					ShadowmapPass.PipelineLayout, 0, 1, &ShadowmapPass.DescriptorSets[CurrentFrame], 0, nullptr);
				vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(renderObject->Mesh.Indices.size()), 1, 0, 0, 0);
			}
			// 【阴影】渲染 Instanced 场景
			for (size_t i = 0; i < ShadowmapPass.RenderInstancedObjects.size(); i++)
			{
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ShadowmapPass.PipelinesInstanced[0]);
				XkRenderInstancedObject* renderInstancedObject = ShadowmapPass.RenderInstancedObjects[i];
				VkBuffer objectVertexBuffers[] = { renderInstancedObject->Mesh.VertexBuffer };
				VkBuffer objectInstanceBuffers[] = { renderInstancedObject->Mesh.InstancedBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, VERTEX_BUFFER_BIND_ID, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindVertexBuffers(commandBuffer, INSTANCE_BUFFER_BIND_ID, 1, objectInstanceBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, renderInstancedObject->Mesh.IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
					ShadowmapPass.PipelineLayout, 0, 1, &ShadowmapPass.DescriptorSets[CurrentFrame], 0, nullptr);
				vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(renderInstancedObject->Mesh.Indices.size()), renderInstancedObject->InstanceCount, 0, 0, 0);
			}
			// 【阴影】渲染Indirect场景
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ShadowmapPass.Pipelines[0]);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				ShadowmapPass.PipelineLayout, 0, 1, &ShadowmapPass.DescriptorSets[CurrentFrame], 0, nullptr);
			for (size_t i = 0; i < ShadowmapPass.RenderIndirectObjects.size(); i++)
			{
				XkRenderObjectIndirect* RenderIndirectObject = ShadowmapPass.RenderIndirectObjects[i];
				VkBuffer objectVertexBuffers[] = { RenderIndirectObject->Mesh.VertexBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, RenderIndirectObject->Mesh.IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
				uint32_t indirectDrawCount = static_cast<uint32_t>(RenderIndirectObject->IndirectCommands.size());
				if (IsSupportMultiDrawIndirect(PhysicalDevice))
				{
					vkCmdDrawIndexedIndirect(
						commandBuffer, /*commandBuffer*/
						RenderIndirectObject->IndirectCommandsBuffer, /*buffer*/
						0, /*offset*/
						indirectDrawCount, /*drawCount*/
						sizeof(VkDrawIndexedIndirectCommand) /*stride*/);
				}
				else
				{
					// If multi draw is not available, we must issue separate draw commands
					for (auto j = 0; j < RenderIndirectObject->IndirectCommands.size(); j++)
					{
						vkCmdDrawIndexedIndirect(
							commandBuffer, /*commandBuffer*/
							RenderIndirectObject->IndirectCommandsBuffer, /*buffer*/
							j * sizeof(VkDrawIndexedIndirectCommand), /*offset*/
							1, /*drawCount*/
							sizeof(VkDrawIndexedIndirectCommand) /*stride*/);
					}
				}
			}
			// 【阴影】渲染Indirect Instanced 场景
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ShadowmapPass.PipelinesInstanced[0]);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				ShadowmapPass.PipelineLayout, 0, 1, &ShadowmapPass.DescriptorSets[CurrentFrame], 0, nullptr);
			for (size_t i = 0; i < ShadowmapPass.RenderIndirectInstancedObjects.size(); i++)
			{
				XkRenderInstancedObjectIndirect* RenderIndirectInstancedObject = ShadowmapPass.RenderIndirectInstancedObjects[i];
				VkBuffer objectVertexBuffers[] = { RenderIndirectInstancedObject->Mesh.VertexBuffer };
				VkBuffer objectInstanceBuffers[] = { RenderIndirectInstancedObject->Mesh.InstancedBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, VERTEX_BUFFER_BIND_ID, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindVertexBuffers(commandBuffer, INSTANCE_BUFFER_BIND_ID, 1, objectInstanceBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, RenderIndirectInstancedObject->Mesh.IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
				uint32_t indirectDrawCount = static_cast<uint32_t>(RenderIndirectInstancedObject->IndirectCommands.size());
				if (IsSupportMultiDrawIndirect(PhysicalDevice))
				{
					vkCmdDrawIndexedIndirect(
						commandBuffer, /*commandBuffer*/
						RenderIndirectInstancedObject->IndirectCommandsBuffer, /*buffer*/
						0, /*offset*/
						indirectDrawCount, /*drawCount*/
						sizeof(VkDrawIndexedIndirectCommand) /*stride*/);
				}
				else
				{
					// If multi draw is not available, we must issue separate draw commands
					for (auto j = 0; j < RenderIndirectInstancedObject->IndirectCommands.size(); j++)
					{
						vkCmdDrawIndexedIndirect(
							commandBuffer, /*commandBuffer*/
							RenderIndirectInstancedObject->IndirectCommandsBuffer, /*buffer*/
							j * sizeof(VkDrawIndexedIndirectCommand), /*offset*/
							1, /*drawCount*/
							sizeof(VkDrawIndexedIndirectCommand) /*stride*/);
					}
				}
			}

			// 【阴影】结束 RenderPass
			vkCmdEndRenderPass(commandBuffer);
		}

		// 渲染视口信息
		VkViewport mainViewport{};
		mainViewport.x = 0.0f;
		mainViewport.y = 0.0f;
		mainViewport.width = (float)SwapChainExtent.width - ImGuiPass.RightBarSpace;
		mainViewport.height = (float)SwapChainExtent.height - ImGuiPass.BottomBarSpace;
		mainViewport.minDepth = 0.0f;
		mainViewport.maxDepth = 1.0f;

		VkViewport mainWindow{};
		mainWindow.x = 0.0f;
		mainWindow.y = 0.0f;
		mainWindow.width = (float)SwapChainExtent.width;
		mainWindow.height = (float)SwapChainExtent.height;
		mainWindow.minDepth = 0.0f;
		mainWindow.maxDepth = 1.0f;

		// 视口剪切信息
		VkRect2D mainScissor{};
		mainScissor.offset = { 0, 0 };
		mainScissor.extent = SwapChainExtent;

#if ENABLE_DEFEERED_SHADING
		// 【延迟渲染】渲染场景
		{
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = BasePass.DeferredSceneRenderPass;
			renderPassInfo.framebuffer = BasePass.DeferredSceneFrameBuffer;
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = SwapChainExtent;

			std::array<VkClearValue, 6> clearValues{};
			clearValues[0].depthStencil = { 1.0f, 0 };
			clearValues[1].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
			clearValues[2].color = { {0.0f, 0.0f, 0.0f, 0.0f} };
			clearValues[3].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
			clearValues[4].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
			clearValues[5].color = { {0.0f, 0.0f, 0.0f, 1.0f} };

			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			// 【延迟渲染】开始 RenderPass
			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdSetViewport(commandBuffer, 0, 1, &mainViewport);
			vkCmdSetScissor(commandBuffer, 0, 1, &mainScissor);

			// 【延迟渲染】渲染场景
			for (size_t i = 0; i < BasePass.RenderDeferredObjects.size(); i++)
			{
				uint32_t SpecConstants = GlobalConstants.SpecConstants;
				VkPipeline baseScenePassPipeline = BasePass.DeferredScenePipelines[SpecConstants];
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, baseScenePassPipeline);
				FRenderDeferredObject* renderObject = BasePass.RenderDeferredObjects[i];
				VkBuffer objectVertexBuffers[] = { renderObject->Mesh.VertexBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, renderObject->Mesh.IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdBindDescriptorSets(
					commandBuffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					BasePass.DeferredScenePipelineLayout, 0, 1,
					&renderObject->Material.DescriptorSets[CurrentFrame], 0, nullptr);
				vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(renderObject->Mesh.Indices.size()), 1, 0, 0, 0);
			}
			// 【延迟渲染】渲染 Instanced 场景
			for (size_t i = 0; i < BasePass.RenderDeferredInstancedObjects.size(); i++)
			{
				uint32_t SpecConstants = GlobalConstants.SpecConstants;
				VkPipeline BaseScenePassPipelineInstanced = BasePass.DeferredScenePipelinesInstanced[SpecConstants];
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, BaseScenePassPipelineInstanced);
				FRenderDeferredInstancedObject* renderInstancedObject = BasePass.RenderDeferredInstancedObjects[i];
				VkBuffer objectVertexBuffers[] = { renderInstancedObject->Mesh.VertexBuffer };
				VkBuffer objectInstanceBuffers[] = { renderInstancedObject->Mesh.InstancedBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, VERTEX_BUFFER_BIND_ID, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindVertexBuffers(commandBuffer, INSTANCE_BUFFER_BIND_ID, 1, objectInstanceBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, renderInstancedObject->Mesh.IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdBindDescriptorSets(
					commandBuffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					BasePass.DeferredScenePipelineLayout, 0, 1,
					&renderInstancedObject->Material.DescriptorSets[CurrentFrame], 0, nullptr);
				vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(renderInstancedObject->Mesh.Indices.size()), renderInstancedObject->InstanceCount, 0, 0, 0);
			}

			// 【延迟渲染】结束RenderPass
			vkCmdEndRenderPass(commandBuffer);
		}

		VkImageCopy copyRegion = {};
		copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		copyRegion.srcSubresource.baseArrayLayer = 0;
		copyRegion.srcSubresource.mipLevel = 0;
		copyRegion.srcSubresource.layerCount = 1;
		copyRegion.srcOffset = { 0, 0, 0 };
		copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		copyRegion.dstSubresource.baseArrayLayer = 0;
		copyRegion.dstSubresource.mipLevel = 0;
		copyRegion.dstSubresource.layerCount = 1;
		copyRegion.dstOffset = { 0, 0, 0 };
		copyRegion.extent.width = static_cast<uint32_t>(SwapChainExtent.width);
		copyRegion.extent.height = static_cast<uint32_t>(SwapChainExtent.height);
		copyRegion.extent.depth = 1;

		BeginTransitionImageLayoutRT(GBuffer.DepthStencilImage,
			VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		BeginTransitionImageLayoutRT(DepthImage,
			VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		vkCmdCopyImage(commandBuffer, GBuffer.DepthStencilImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			DepthImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
		EndTransitionImageLayoutRT(GBuffer.DepthStencilImage,
			VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
		EndTransitionImageLayoutRT(DepthImage,
			VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
#endif

		// 【主场景】渲染场景
		{
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = MainRenderPass;
			renderPassInfo.framebuffer = SwapChainFramebuffers[imageIndex];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = SwapChainExtent;

			std::array<VkClearValue, 2> clearValues{};
			clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
			clearValues[1].depthStencil = { 1.0f, 0 };

			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			// 【主场景】开始 RenderPass
			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			// 【主场景】设置视口剪切
			vkCmdSetScissor(commandBuffer, 0, 1, &mainScissor);

#if ENABLE_DEFEERED_SHADING
			// 【主场景】设置延迟渲染视口
			vkCmdSetViewport(commandBuffer, 0, 1, &mainWindow);

			// 【主场景】渲染延迟渲染灯光
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, BasePass.DeferredLightingPipelines[GlobalConstants.SpecConstants]);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, BasePass.DeferredLightingPipelineLayout, 0, 1, &BasePass.DeferredLightingDescriptorSets[CurrentFrame], 0, nullptr);
			vkCmdPushConstants(commandBuffer, BasePass.DeferredLightingPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(XkGlobalConstants), &GlobalConstants);
			vkCmdDraw(commandBuffer, 6, 1, 0, 0);
#endif
			// 【主场景】设置前向渲染视口
			vkCmdSetViewport(commandBuffer, 0, 1, &mainViewport);

			// 【主场景】渲染场景
			for (size_t i = 0; i < BasePass.RenderObjects.size(); i++)
			{
				VkPipeline baseScenePassPipeline = BasePass.Pipelines[GlobalConstants.SpecConstants];
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, baseScenePassPipeline);
				XkRenderObject* renderObject = BasePass.RenderObjects[i];
				VkBuffer objectVertexBuffers[] = { renderObject->Mesh.VertexBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, renderObject->Mesh.IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdBindDescriptorSets(
					commandBuffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					BasePass.PipelineLayout, 0, 1,
					&renderObject->Material.DescriptorSets[CurrentFrame], 0, nullptr);
				vkCmdPushConstants(commandBuffer, BasePass.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(XkGlobalConstants), &GlobalConstants);
				vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(renderObject->Mesh.Indices.size()), 1, 0, 0, 0);
			}
			// 【主场景】渲染 Instanced 场景
			for (size_t i = 0; i < BasePass.RenderInstancedObjects.size(); i++)
			{
				VkPipeline BaseScenePassPipelineInstanced = BasePass.PipelinesInstanced[GlobalConstants.SpecConstants];
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, BaseScenePassPipelineInstanced);
				XkRenderInstancedObject* renderInstancedObject = BasePass.RenderInstancedObjects[i];
				VkBuffer objectVertexBuffers[] = { renderInstancedObject->Mesh.VertexBuffer };
				VkBuffer objectInstanceBuffers[] = { renderInstancedObject->Mesh.InstancedBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, VERTEX_BUFFER_BIND_ID, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindVertexBuffers(commandBuffer, INSTANCE_BUFFER_BIND_ID, 1, objectInstanceBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, renderInstancedObject->Mesh.IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdBindDescriptorSets(
					commandBuffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					BasePass.PipelineLayout, 0, 1,
					&renderInstancedObject->Material.DescriptorSets[CurrentFrame], 0, nullptr);
				vkCmdPushConstants(commandBuffer, BasePass.PipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(XkGlobalConstants), &GlobalConstants);
				vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(renderInstancedObject->Mesh.Indices.size()), renderInstancedObject->InstanceCount, 0, 0, 0);
			}
#if ENABLE_INDIRECT_DRAW
			// 【主场景】渲染 Indirect 场景
			for (size_t i = 0; i < BasePass.RenderIndirectObjects.size(); i++)
			{
				VkPipeline indirectScenePassPipeline = BasePass.IndirectPipelines[GlobalConstants.SpecConstants];
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, indirectScenePassPipeline);
				XkRenderObjectIndirect* RenderIndirectObject = BasePass.RenderIndirectObjects[i];
				VkBuffer objectVertexBuffers[] = { RenderIndirectObject->Mesh.VertexBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, RenderIndirectObject->Mesh.IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdBindDescriptorSets(
					commandBuffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					BasePass.IndirectPipelineLayout, 0, 1,
					&RenderIndirectObject->Material.DescriptorSets[CurrentFrame], 0, nullptr);
				vkCmdPushConstants(commandBuffer, BasePass.IndirectPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(XkGlobalConstants), &GlobalConstants);
				uint32_t indirectDrawCount = static_cast<uint32_t>(RenderIndirectObject->IndirectCommands.size());
				if (IsSupportMultiDrawIndirect(PhysicalDevice))
				{
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					//01 void FakeDrawIndexedIndirect(VkCommandBuffer commandBuffer, void* buffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
					//02 {
					//03 	char* memory = (char*)buffer + offset;
					//04 
					//05 	for (uint32_t i = 0; i < drawCount; i++)
					//06 	{
					//07 		VkDrawIndexedIndirectCommand* command = (VkDrawIndexedIndirectCommand*)(memory + (i * stride));
					//08 
					//09 		vkCmdDrawIndexed(commandBuffer,
					//10 			command->indexCount,
					//11 			command->instanceCount,
					//12 			command->firstIndex,
					//13 			command->vertexOffset,
					//14 			command->firstInstance);
					//15 	}
					//16 }
					vkCmdDrawIndexedIndirect(
						commandBuffer, /*commandBuffer*/
						RenderIndirectObject->IndirectCommandsBuffer, /*buffer*/
						0, /*offset*/
						indirectDrawCount, /*drawCount*/
						sizeof(VkDrawIndexedIndirectCommand) /*stride*/);
				}
				else
				{
					// If multi draw is not available, we must issue separate draw commands
					for (auto j = 0; j < RenderIndirectObject->IndirectCommands.size(); j++)
					{
						vkCmdDrawIndexedIndirect(
							commandBuffer, /*commandBuffer*/
							RenderIndirectObject->IndirectCommandsBuffer, /*buffer*/
							j * sizeof(VkDrawIndexedIndirectCommand), /*offset*/
							1, /*drawCount*/
							sizeof(VkDrawIndexedIndirectCommand) /*stride*/);
					}
				}
			}
			// 【主场景】渲染 Indirect Instanced 场景
			for (size_t i = 0; i < BasePass.RenderIndirectInstancedObjects.size(); i++)
			{
				VkPipeline indirectScenePassPipelineInstanced = BasePass.IndirectPipelinesInstanced[GlobalConstants.SpecConstants];
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, indirectScenePassPipelineInstanced);
				XkRenderInstancedObjectIndirect* RenderIndirectInstancedObject = BasePass.RenderIndirectInstancedObjects[i];
				VkBuffer objectVertexBuffers[] = { RenderIndirectInstancedObject->Mesh.VertexBuffer };
				VkBuffer objectInstanceBuffers[] = { RenderIndirectInstancedObject->Mesh.InstancedBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				// Binding point 0 : Mesh vertex buffer
				vkCmdBindVertexBuffers(commandBuffer, VERTEX_BUFFER_BIND_ID, 1, objectVertexBuffers, objectOffsets);
				// Binding point 1 : Instance data buffer
				vkCmdBindVertexBuffers(commandBuffer, INSTANCE_BUFFER_BIND_ID, 1, objectInstanceBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, RenderIndirectInstancedObject->Mesh.IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdBindDescriptorSets(
					commandBuffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					BasePass.IndirectPipelineLayout, 0, 1,
					&RenderIndirectInstancedObject->Material.DescriptorSets[CurrentFrame], 0, nullptr);
				vkCmdPushConstants(commandBuffer, BasePass.IndirectPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(XkGlobalConstants), &GlobalConstants);
				uint32_t indirectDrawCount = static_cast<uint32_t>(RenderIndirectInstancedObject->IndirectCommands.size());
				if (IsSupportMultiDrawIndirect(PhysicalDevice))
				{
					vkCmdDrawIndexedIndirect(
						commandBuffer, /*commandBuffer*/
						RenderIndirectInstancedObject->IndirectCommandsBuffer, /*buffer*/
						0, /*offset*/
						indirectDrawCount, /*drawCount*/
						sizeof(VkDrawIndexedIndirectCommand) /*stride*/);
				}
				else
				{
					// If multi draw is not available, we must issue separate draw commands
					for (auto j = 0; j < RenderIndirectInstancedObject->IndirectCommands.size(); j++)
					{
						vkCmdDrawIndexedIndirect(
							commandBuffer, /*commandBuffer*/
							RenderIndirectInstancedObject->IndirectCommandsBuffer, /*buffer*/
							j * sizeof(VkDrawIndexedIndirectCommand), /*offset*/
							1, /*drawCount*/
							sizeof(VkDrawIndexedIndirectCommand) /*stride*/);
					}
				}
			}
#endif
			// 【主场景】渲染天空球
			if (SkydomePass.EnableSkydome && GlobalConstants.SpecConstants == 0 /* Don't render if on debug mode*/)
			{
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, SkydomePass.Pipelines[0]);
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, SkydomePass.PipelineLayout, 0, 1, &SkydomePass.DescriptorSets[CurrentFrame], 0, nullptr);
				VkBuffer objectVertexBuffers[] = { SkydomePass.SkydomeMesh.VertexBuffer };
				VkDeviceSize objectOffsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, objectVertexBuffers, objectOffsets);
				vkCmdBindIndexBuffer(commandBuffer, SkydomePass.SkydomeMesh.IndexBuffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(SkydomePass.SkydomeMesh.Indices.size()), 1, 0, 0, 0);
			}

			if (BackgroundPass.EnableBackground && GlobalConstants.SpecConstants == 0 /* Don't render if on debug mode*/)
			{
				// 【主场景】渲染背景面片
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, BackgroundPass.Pipelines[0]);
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, BackgroundPass.PipelineLayout, 0, 1, &BackgroundPass.DescriptorSets[CurrentFrame], 0, nullptr);
				vkCmdDraw(commandBuffer, 6, 1, 0, 0);
			}

			// 【主场景】结束RenderPass
			vkCmdEndRenderPass(commandBuffer);
		}

#if ENABLE_IMGUI_EDITOR
		// 【主界面】渲染主界面
		{
			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = ImGuiPass.RenderPass;
			renderPassInfo.framebuffer = SwapChainFramebuffers[imageIndex];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = SwapChainExtent;

			VkViewport mainWindow{};
			mainWindow.x = 0.0f;
			mainWindow.y = 0.0f;
			mainWindow.width = (float)SwapChainExtent.width;
			mainWindow.height = (float)SwapChainExtent.height;
			mainWindow.minDepth = 0.0f;
			mainWindow.maxDepth = 1.0f;

			// 【主界面】开始 RenderPass
			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			// 【主界面】设置渲染视口
			vkCmdSetViewport(commandBuffer, 0, 1, &mainWindow);

			// 【【主界面】设置视口剪切
			vkCmdSetScissor(commandBuffer, 0, 1, &mainScissor);

			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

			// 【主界面】结束RenderPass
			vkCmdEndRenderPass(commandBuffer);
		}
#endif

		// 结束记录指令
		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("[RecordCommandBuffer] Failed to record command buffer!");
		}
	}

	/** Delete elements created in the InitVulkan function */
	void DestroyVulkan()
	{
		// Clean up resources related to FrameBuffer
		CleanupSwapChain();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(Device, RenderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(Device, ImageAvailableSemaphores[i], nullptr);
			vkDestroyFence(Device, InFlightFences[i], nullptr);
		}

		vkDestroyRenderPass(Device, MainRenderPass, nullptr);

#if ENABLE_IMGUI_EDITOR
		// clear up imgui pass
		CleanupImgui();
#endif

		// Clean up UniformBuffers
		CleanupUniformBuffers();

		CleanupCubeMaps();

		// Clean up ShadowmapPass
		CleanupShadowPass();

		// Clean up SkydomePass
		CleanupSkydomePass();

		// Clean up BackgroundPass
		CleanupBackgroundPass();

		// Cleanup BasePass
		CleanupBasePass();

		vkFreeCommandBuffers(Device, CommandPool, static_cast<uint32_t>(CommandBuffers.size()), CommandBuffers.data());
		vkDestroyCommandPool(Device, CommandPool, nullptr);

		vkDestroyDevice(Device, nullptr);

		if (bEnableValidationLayers)
		{
			DestroyDebugUtilsMessengerEXT(Instance, DebugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(Instance, Surface, nullptr);
		vkDestroyInstance(Instance, nullptr);
	}

	/** Clean up old SwapChain */
	void CleanupSwapChain() {
		vkDestroyImageView(Device, DepthImageView, nullptr);
		vkDestroyImage(Device, DepthImage, nullptr);
		vkFreeMemory(Device, DepthImageMemory, nullptr);

		for (auto framebuffer : SwapChainFramebuffers) {
			vkDestroyFramebuffer(Device, framebuffer, nullptr);
		}

		for (auto imageView : SwapChainImageViews) {
			vkDestroyImageView(Device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(Device, SwapChain, nullptr);
	}

	void CleanupUniformBuffers()
	{
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroyBuffer(Device, BaseUniformBuffers[i], nullptr);
			vkFreeMemory(Device, BaseUniformBuffersMemory[i], nullptr);
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroyBuffer(Device, ViewUniformBuffers[i], nullptr);
			vkFreeMemory(Device, ViewUniformBuffersMemory[i], nullptr);
		}
	}

	void CleanupShadowPass()
	{
		vkDestroyRenderPass(Device, ShadowmapPass.RenderPass, nullptr);
		vkDestroyFramebuffer(Device, ShadowmapPass.FrameBuffer, nullptr);
		vkDestroyDescriptorPool(Device, ShadowmapPass.DescriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(Device, ShadowmapPass.DescriptorSetLayout, nullptr);
		vkDestroyPipelineLayout(Device, ShadowmapPass.PipelineLayout, nullptr);
		for (size_t i = 0; i < ShadowmapPass.Pipelines.size(); i++)
		{
			vkDestroyPipeline(Device, ShadowmapPass.Pipelines[i], nullptr);
		}
		for (size_t i = 0; i < ShadowmapPass.PipelinesInstanced.size(); i++)
		{
			vkDestroyPipeline(Device, ShadowmapPass.PipelinesInstanced[i], nullptr);
		}
		vkDestroyImageView(Device, ShadowmapPass.ImageView, nullptr);
		vkDestroySampler(Device, ShadowmapPass.Sampler, nullptr);
		vkDestroyImage(Device, ShadowmapPass.Image, nullptr);
		vkFreeMemory(Device, ShadowmapPass.Memory, nullptr);
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroyBuffer(Device, ShadowmapPass.UniformBuffers[i], nullptr);
			vkFreeMemory(Device, ShadowmapPass.UniformBuffersMemory[i], nullptr);
		}
	}

	void CleanupCubeMaps()
	{
		vkDestroyImageView(Device, CubemapImageView, nullptr);
		vkDestroySampler(Device, CubemapSampler, nullptr);
		vkDestroyImage(Device, CubemapImage, nullptr);
		vkFreeMemory(Device, CubemapImageMemory, nullptr);
	}

	void CleanupSkydomeResource()
	{
		for (size_t i = 0; i < SkydomePass.ImageViews.size(); i++)
		{
			vkDestroyImageView(Device, SkydomePass.ImageViews[i], nullptr);
			vkDestroySampler(Device, SkydomePass.ImageSamplers[i], nullptr);
			vkDestroyImage(Device, SkydomePass.Images[i], nullptr);
			vkFreeMemory(Device, SkydomePass.ImageMemorys[i], nullptr);
		}
	}

	void CleanupSkydomePass()
	{
		vkDestroyDescriptorSetLayout(Device, SkydomePass.DescriptorSetLayout, nullptr);
		vkDestroyPipelineLayout(Device, SkydomePass.PipelineLayout, nullptr);
		for (size_t i = 0; i < SkydomePass.Pipelines.size(); i++)
		{
			vkDestroyPipeline(Device, SkydomePass.Pipelines[i], nullptr);
		}
		vkDestroyDescriptorPool(Device, SkydomePass.DescriptorPool, nullptr);
		vkDestroyBuffer(Device, SkydomePass.SkydomeMesh.VertexBuffer, nullptr);
		vkFreeMemory(Device, SkydomePass.SkydomeMesh.VertexBufferMemory, nullptr);
		vkDestroyBuffer(Device, SkydomePass.SkydomeMesh.IndexBuffer, nullptr);
		vkFreeMemory(Device, SkydomePass.SkydomeMesh.IndexBufferMemory, nullptr);
		CleanupSkydomeResource();
	}

	void CleanupBackgroundResource()
	{
		for (size_t i = 0; i < BackgroundPass.ImageViews.size(); i++)
		{
			vkDestroyImageView(Device, BackgroundPass.ImageViews[i], nullptr);
			vkDestroySampler(Device, BackgroundPass.ImageSamplers[i], nullptr);
			vkDestroyImage(Device, BackgroundPass.Images[i], nullptr);
			vkFreeMemory(Device, BackgroundPass.ImageMemorys[i], nullptr);
		}
	}

	void CleanupBackgroundPass()
	{
		vkDestroyDescriptorSetLayout(Device, BackgroundPass.DescriptorSetLayout, nullptr);
		vkDestroyPipelineLayout(Device, BackgroundPass.PipelineLayout, nullptr);
		for (size_t i = 0; i < BackgroundPass.Pipelines.size(); i++)
		{
			vkDestroyPipeline(Device, BackgroundPass.Pipelines[i], nullptr);
		}
		vkDestroyDescriptorPool(Device, BackgroundPass.DescriptorPool, nullptr);
		CleanupBackgroundResource();
	}

	void CleanupBasePassResources()
	{
		// Clean up Scene
		for (size_t i = 0; i < Scene.RenderObjects.size(); i++)
		{
			XkRenderObject& renderObject = Scene.RenderObjects[i];
			DestroyRenderObject(renderObject);
		}
		for (size_t i = 0; i < Scene.RenderInstancedObjects.size(); i++)
		{
			XkRenderInstancedObject& renderInstancedObject = Scene.RenderInstancedObjects[i];

			DestroyRenderObject(renderInstancedObject);

			vkDestroyBuffer(Device, renderInstancedObject.Mesh.InstancedBuffer, nullptr);
			vkFreeMemory(Device, renderInstancedObject.Mesh.InstancedBufferMemory, nullptr);
		}
		for (size_t i = 0; i < Scene.RenderIndirectObjects.size(); i++)
		{
			XkRenderObjectIndirect& RenderIndirectObject = Scene.RenderIndirectObjects[i];

			DestroyRenderObject(RenderIndirectObject);

			vkDestroyBuffer(Device, RenderIndirectObject.IndirectCommandsBuffer, nullptr);
			vkFreeMemory(Device, RenderIndirectObject.IndirectCommandsBufferMemory, nullptr);
		}
		for (size_t i = 0; i < Scene.RenderIndirectInstancedObjects.size(); i++)
		{
			XkRenderInstancedObjectIndirect& RenderIndirectInstancedObject = Scene.RenderIndirectInstancedObjects[i];

			DestroyRenderObject(RenderIndirectInstancedObject);

			vkDestroyBuffer(Device, RenderIndirectInstancedObject.Mesh.InstancedBuffer, nullptr);
			vkFreeMemory(Device, RenderIndirectInstancedObject.Mesh.InstancedBufferMemory, nullptr);

			vkDestroyBuffer(Device, RenderIndirectInstancedObject.IndirectCommandsBuffer, nullptr);
			vkFreeMemory(Device, RenderIndirectInstancedObject.IndirectCommandsBufferMemory, nullptr);
		}
#if ENABLE_DEFEERED_SHADING
		for (size_t i = 0; i < Scene.RenderDeferredObjects.size(); i++)
		{
			FRenderDeferredObject& renderObject = Scene.RenderDeferredObjects[i];

			DestroyRenderObject(renderObject);
		}
		for (size_t i = 0; i < Scene.RenderDeferredInstancedObjects.size(); i++)
		{
			FRenderDeferredInstancedObject& renderInstancedObject = Scene.RenderDeferredInstancedObjects[i];

			DestroyRenderObject(renderInstancedObject);

			vkDestroyBuffer(Device, renderInstancedObject.Mesh.InstancedBuffer, nullptr);
			vkFreeMemory(Device, renderInstancedObject.Mesh.InstancedBufferMemory, nullptr);
		}
#endif
		Scene.Reset();
	}

	void CleanupBasePass()
	{
		// Clean up BasePass
		vkDestroyDescriptorSetLayout(Device, BasePass.DescriptorSetLayout, nullptr);
		vkDestroyPipelineLayout(Device, BasePass.PipelineLayout, nullptr);
		for (size_t i = 0; i < BasePass.Pipelines.size(); i++)
		{
			assert(BasePass.Pipelines.size() == BasePass.PipelinesInstanced.size());
			vkDestroyPipeline(Device, BasePass.Pipelines[i], nullptr);
			vkDestroyPipeline(Device, BasePass.PipelinesInstanced[i], nullptr);
		}

#if ENABLE_INDIRECT_DRAW
		// Clean up BasePass
		vkDestroyDescriptorSetLayout(Device, BasePass.DescriptorSetLayout, nullptr);
		vkDestroyPipelineLayout(Device, BasePass.PipelineLayout, nullptr);
		for (size_t i = 0; i < BasePass.Pipelines.size(); i++)
		{
			assert(BasePass.Pipelines.size() == BasePass.PipelinesInstanced.size());
			vkDestroyPipeline(Device, BasePass.Pipelines[i], nullptr);
			vkDestroyPipeline(Device, BasePass.PipelinesInstanced[i], nullptr);
		}
#endif

#if ENABLE_DEFEERED_SHADING
		// Clean up BasePass
		vkDestroyDescriptorSetLayout(Device, BasePass.DeferredLightingDescriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(Device, BasePass.DeferredLightingDescriptorPool, nullptr);
		vkDestroyPipelineLayout(Device, BasePass.DeferredLightingPipelineLayout, nullptr);
		for (size_t i = 0; i < BasePass.DeferredLightingPipelines.size(); i++)
		{
			vkDestroyPipeline(Device, BasePass.DeferredLightingPipelines[i], nullptr);
		}
		vkDestroyRenderPass(Device, BasePass.DeferredSceneRenderPass, nullptr);
		vkDestroyFramebuffer(Device, BasePass.DeferredSceneFrameBuffer, nullptr);
		vkDestroyDescriptorSetLayout(Device, BasePass.DeferredSceneDescriptorSetLayout, nullptr);
		vkDestroyPipelineLayout(Device, BasePass.DeferredScenePipelineLayout, nullptr);
		for (size_t i = 0; i < BasePass.DeferredScenePipelines.size(); i++)
		{
			assert(BasePass.DeferredScenePipelines.size() == BasePass.DeferredScenePipelinesInstanced.size());
			vkDestroyPipeline(Device, BasePass.DeferredScenePipelines[i], nullptr);
			vkDestroyPipeline(Device, BasePass.DeferredScenePipelinesInstanced[i], nullptr);
		}

		vkDestroyImageView(Device, GBuffer.DepthStencilImageView, nullptr);
		vkDestroySampler(Device, GBuffer.DepthStencilSampler, nullptr);
		vkDestroyImage(Device, GBuffer.DepthStencilImage, nullptr);
		vkFreeMemory(Device, GBuffer.DepthStencilMemory, nullptr);
		vkDestroyImageView(Device, GBuffer.SceneColorImageView, nullptr);
		vkDestroySampler(Device, GBuffer.SceneColorSampler, nullptr);
		vkDestroyImage(Device, GBuffer.SceneColorImage, nullptr);
		vkFreeMemory(Device, GBuffer.SceneColorMemory, nullptr);
		vkDestroyImageView(Device, GBuffer.GBufferAImageView, nullptr);
		vkDestroySampler(Device, GBuffer.GBufferASampler, nullptr);
		vkDestroyImage(Device, GBuffer.GBufferAImage, nullptr);
		vkFreeMemory(Device, GBuffer.GBufferAMemory, nullptr);
		vkDestroyImageView(Device, GBuffer.GBufferBImageView, nullptr);
		vkDestroySampler(Device, GBuffer.GBufferBSampler, nullptr);
		vkDestroyImage(Device, GBuffer.GBufferBImage, nullptr);
		vkFreeMemory(Device, GBuffer.GBufferBMemory, nullptr);
		vkDestroyImageView(Device, GBuffer.GBufferCImageView, nullptr);
		vkDestroySampler(Device, GBuffer.GBufferCSampler, nullptr);
		vkDestroyImage(Device, GBuffer.GBufferCImage, nullptr);
		vkFreeMemory(Device, GBuffer.GBufferCMemory, nullptr);
		vkDestroyImageView(Device, GBuffer.GBufferDImageView, nullptr);
		vkDestroySampler(Device, GBuffer.GBufferDSampler, nullptr);
		vkDestroyImage(Device, GBuffer.GBufferDImage, nullptr);
		vkFreeMemory(Device, GBuffer.GBufferDMemory, nullptr);
#endif

		CleanupBasePassResources();
	}

#if ENABLE_IMGUI_EDITOR
	void CleanupImgui()
	{
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		// cleanup ImGuiPass
		vkDestroyDescriptorPool(Device, ImGuiPass.DescriptorPool, nullptr);
		vkDestroyRenderPass(Device, ImGuiPass.RenderPass, nullptr);
	}
#endif
	
public:
	void CreateEngineWorld()
	{
		GlobalInput.MainCamera = &World.MainCamera;
#if ENABLE_GENERATE_WORLD
		World.EnableSkydome = true;
		World.OverrideSkydome = true;
		World.SkydomeFileName = "grassland_night.png";

		World.OverrideCubemap = true;
		World.CubemapFileNames = {
			"grassland_night_X0.png",
			"grassland_night_X1.png",
			"grassland_night_Y2.png",
			"grassland_night_Y3.png",
			"grassland_night_Z4.png",
			"grassland_night_Z5.png"
		};

		World.EnableBackground = true;
		World.OverrideBackground = true;
		World.BackgroundFileName = "background.png";

		XkObjectDesc terrain;
		terrain.RenderFlags = EXkRenderFlags::None;
		terrain.ProfabName = "terrain";
		terrain.InstanceCount = 1;
		World.ObjectDescs.push_back(terrain);

		XkObjectDesc rock_01;
		rock_01.RenderFlags = EXkRenderFlags::DeferredScene;
		rock_01.ProfabName = "rock_01";
		rock_01.InstanceCount = 1;
		World.ObjectDescs.push_back(rock_01);

		XkObjectDesc rock_02;
		rock_02.RenderFlags = EXkRenderFlags::DeferredScene;
		rock_02.ProfabName = "rock_02";
		rock_02.InstanceCount = 64;
		rock_02.MinRadius = 1.0f;
		rock_02.MaxRadius = 5.0f;
		rock_02.MinPScale = 0.2f;
		rock_02.MaxPScale = 0.5f;
		World.ObjectDescs.push_back(rock_02);

		XkObjectDesc grass_01;
		grass_01.RenderFlags = EXkRenderFlags::DeferredScene;
		grass_01.ProfabName = "grass_01";
		grass_01.InstanceCount = 10000;
		grass_01.MinRadius = 2.0f;
		grass_01.MaxRadius = 8.0f;
		grass_01.MinPScale = 0.1f;
		grass_01.MaxPScale = 0.5f;
		World.ObjectDescs.push_back(grass_01);

		XkObjectDesc grass_02;
		grass_02.RenderFlags = EXkRenderFlags::DeferredScene;
		grass_02.ProfabName = "grass_02";
		grass_02.InstanceCount = 10000;
		grass_02.MinRadius = 1.0f;
		grass_02.MaxRadius = 9.0f;
		grass_02.MinPScale = 0.1f;
		grass_02.MaxPScale = 0.5f;
		World.ObjectDescs.push_back(grass_02);

		XkLightDesc Moonlight;
		Moonlight.Position = glm::vec3(20.0f, 0.0f, 20.0f);
		Moonlight.Type = 0;
		Moonlight.Color = glm::vec3(0.0, 0.1, 0.6);
		Moonlight.Intensity = 15.0;
		Moonlight.Direction = glm::vec3(glm::normalize(glm::vec3(Moonlight.Position.x, Moonlight.Position.y, Moonlight.Position.z)));
		Moonlight.Radius = 0.0;
		Moonlight.ExtraData = glm::vec4(0.0, 0.0, 0.0, 0.0);
		World.DirectionalLights.push_back(Moonlight);

		uint32_t PointLightNum = 16;
		for (uint32_t i = 0; i < PointLightNum; i++)
		{
			XkLightDesc PointLight;
			float radians = XkObjectDesc::RandRange(0.0f, 360.0f, i);
			float distance = XkObjectDesc::RandRange(0.1f, 0.6f, i);
			float X = sin(glm::radians(radians)) * distance;
			float Y = cos(glm::radians(radians)) * distance;
			float Z = 1.0;
			PointLight.Position = glm::vec3(X, Y, Z);
			PointLight.Type = 1;
			float R = (((float)XkObjectDesc::RandRange(50, 75, i) / 100.0f));
			float G = (((float)XkObjectDesc::RandRange(25, 50, i) / 100.0f));
			float B = 0.0;
			PointLight.Color = glm::vec3(R, G, B);
			PointLight.Intensity = 10.0;
			PointLight.Direction = glm::vec3(0.0, 0.0, 1.0);
			PointLight.Radius = 1.5;
			PointLight.ExtraData = glm::vec4(0.0, 0.0, 0.0, 0.0);
			World.PointLights.push_back(PointLight);
		}
#endif
	}
	void CreateEngineScene()
	{
		if (Scene.bReload)
		{
			CleanupBasePass();
		}

		SkydomePass.EnableSkydome = World.EnableSkydome;
		/* (1) Override skydome and background */
		if (World.OverrideCubemap)
		{
			CleanupCubeMaps();
			CreateImageCubeContext(CubemapImage, CubemapImageMemory, CubemapImageView, CubemapSampler, CubemapMaxMips, {
			ASSETS(World.CubemapFileNames[0]),
			ASSETS(World.CubemapFileNames[1]),
			ASSETS(World.CubemapFileNames[2]),
			ASSETS(World.CubemapFileNames[3]),
			ASSETS(World.CubemapFileNames[4]),
			ASSETS(World.CubemapFileNames[5]) });
		}

		if (World.OverrideSkydome)
		{
			CleanupSkydomeResource();
			CreateImageContext(
				SkydomePass.Images[0],
				SkydomePass.ImageMemorys[0],
				SkydomePass.ImageViews[0],
				SkydomePass.ImageSamplers[0],
				ASSETS(World.SkydomeFileName));
			UpdateDescriptorSet(SkydomePass.DescriptorSets, SkydomePass.ImageViews, SkydomePass.ImageSamplers, EXkRenderFlags::Skydome);
		}

		BackgroundPass.EnableBackground = World.EnableBackground;
		if (World.OverrideBackground)
		{
			CleanupBackgroundResource();
			CreateImageContext(
				BackgroundPass.Images[0],
				BackgroundPass.ImageMemorys[0],
				BackgroundPass.ImageViews[0],
				BackgroundPass.ImageSamplers[0],
				ASSETS(World.BackgroundFileName));
			UpdateDescriptorSet(BackgroundPass.DescriptorSets, BackgroundPass.ImageViews, BackgroundPass.ImageSamplers, EXkRenderFlags::Background);
		}

		if (Scene.bReload)
		{
			CreateBasePass();
		}
		Scene.DescriptorSetLayout = &BasePass.DescriptorSetLayout;
#if ENABLE_INDIRECT_DRAW
		Scene.IndirectDescriptorSetLayout = &BasePass.IndirectDescriptorSetLayout;
#endif
#if ENABLE_DEFEERED_SHADING
		Scene.DeferredSceneDescriptorSetLayout = &BasePass.DeferredSceneDescriptorSetLayout;
		Scene.DeferredLightingDescriptorSetLayout = &BasePass.DeferredLightingDescriptorSetLayout;
#endif
		/* (2) Create base scene */
		if (ENABLE_INDIRECT_DRAW)
		{
			XkRenderObjectIndirect object;
			XkString object_obj = "Content/Meshes/dragon.meshlet";
			std::vector<XkString> object_imgs = {
				"Content/Textures/default_grey.png",	// BaseColor
				"Content/Textures/default_black.png",	// Metallic
				"Content/Textures/default_white.png",	// Roughness
				"Content/Textures/default_normal.png",	// Normal
				"Content/Textures/default_white.png",	// AmbientOcclution
				"Content/Textures/default_black.png",	// Emissive
				"Content/Textures/default_white.png" };	// Mask

			CreateRenderIndirectObject<XkRenderObjectIndirect>(object, object_obj, object_imgs);

			object.IndirectCommands.clear();

			for (uint32_t i = 0; i < object.Mesh.MeshletSet.Meshlets.size(); ++i)
			{
				const XkMeshlet meshletSet = object.Mesh.MeshletSet.Meshlets[i];
				// Member of XkMeshlet:
				//uint32_t VertexOffset;
				//uint32_t VertexCount;
				//uint32_t TriangleOffset;
				//uint32_t TriangleCount;
				//float BoundsCenter[3];
				//float BoundsRadius;
				//float ConeApex[3];
				//float ConeAxis[3];
				//float ConeCutoff, Pad;
				VkDrawIndexedIndirectCommand indirectCmd{};
				// @Note: indexCount = TriangleCount * 3, this is easy to get confused
				indirectCmd.indexCount = (uint32_t)(meshletSet.TriangleCount * 3); /*indexCount*/
				indirectCmd.instanceCount = 1; /*instanceCount*/
				indirectCmd.firstIndex = meshletSet.TriangleOffset; /*firstIndex*/
				indirectCmd.vertexOffset = meshletSet.VertexOffset; /*vertexOffset*/
				indirectCmd.firstInstance = 0; /*firstInstance*/
				object.IndirectCommands.push_back(indirectCmd);
			}
			//VkDrawIndexedIndirectCommand indirectCmd{};
			//indirectCmd.indexCount = (uint32_t)object.Mesh.Indices.size(); /*indexCount*/
			//indirectCmd.instanceCount = 1; /*instanceCount*/
			//indirectCmd.firstIndex = 0; /*firstIndex*/
			//indirectCmd.vertexOffset = 0; /*vertexOffset*/
			//indirectCmd.firstInstance = 0; /*firstInstance*/
			//object.IndirectCommands.push_back(indirectCmd);

			CreateRenderIndirectBuffer<XkRenderObjectIndirect>(object);
			Scene.RenderIndirectObjects.push_back(object);
		}

		for (const XkObjectDesc& ObjectDesc : World.ObjectDescs)
		{
			if (ObjectDesc.InstanceCount > 1)
			{
				std::vector<XkInstanceData> Data;
				XkObjectDesc::GenerateInstance(Data, ObjectDesc);
#if ENABLE_DEFEERED_SHADING
				CreateRenderObjectsFromProfabs(
					Scene.RenderDeferredInstancedObjects,
					*Scene.DeferredSceneDescriptorSetLayout, ObjectDesc.ProfabName, Data);
			}
			else
			{
				CreateRenderObjectsFromProfabs(Scene.RenderDeferredObjects,
					*Scene.DeferredSceneDescriptorSetLayout, ObjectDesc.ProfabName);
			}

			UpdateDescriptorSet(BasePass.DeferredLightingDescriptorSets, GBuffer.ImageViews(), GBuffer.Samplers(), EXkRenderFlags::DeferredLighting);
#else
				CreateRenderObjectsFromProfabs(
					Scene.RenderInstancedObjects,
					*Scene.DescriptorSetLayout, Object.ProfabName, Data);
		}
			else
			{
				CreateRenderObjectsFromProfabs(Scene.RenderObjects,
					*Scene.DescriptorSetLayout, Object.ProfabName);
			}
#endif
		}

		/* 
		* (3) Create light here
		*/
	}

	void RecreateWorldAndScene()
	{
		// we write a json file to a path and tell the engine to load it
		World.Load(SocketListener.receivedData);
		Scene.bReload = true;
	}

	/** Update player inputs*/
	void UpdateWorld()
	{
		for (uint32_t i = 0; i < World.DirectionalLights.size(); i++)
		{
			View.DirectionalLights[i] = World.DirectionalLights[i];
		}
		for (uint32_t i = 0; i < World.PointLights.size(); i++)
		{
			View.PointLights[i] = World.PointLights[i];
		}
		for (uint32_t i = 0; i < World.SpotLights.size(); i++)
		{
			View.SpotLights[i] = World.SpotLights[i];
		}
		View.LightsCount = glm::ivec4(World.DirectionalLights.size(), World.PointLights.size(), World.SpotLights.size(), CubemapMaxMips);

		float DeltaTime = (float)GlobalInput.DeltaTime;    // Time between current frame and last frame
		float LastFrame = (float)GlobalInput.CurrentTime;  // Time of last frame

		float CurrentFrame = (float)glfwGetTime();
		DeltaTime = CurrentFrame - LastFrame;

		GlobalInput.CurrentTime = CurrentFrame;
		GlobalInput.DeltaTime = DeltaTime;

		bool bCameraFocus = GlobalInput.bCameraFocus;
		glm::vec3 Position = GlobalInput.MainCamera->Position;
		glm::vec3 Lookat = GlobalInput.MainCamera->Lookat;
		glm::vec3 CameraForward = glm::vec3(-1.0, 0.0, 0.0);
		glm::vec3 CameraUp = glm::vec3(0.0, 0.0, 1.0);
		float Speed = GlobalInput.MainCamera->Speed;
		float Yaw = GlobalInput.MainCamera->Yaw;
		float Pitch = GlobalInput.MainCamera->Pitch;
		glm::vec3 CameraDirection = glm::normalize(Lookat - Position);
		const float CameraDeltaMove = 2.5f * DeltaTime; // adjust accordingly

		// @TODO: WASD control Camera
	}

#if ENABLE_IMGUI_EDITOR
	/** Update ImGui widgets*/
	void UpdateImGuiWidgets()
	{
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();

		ImGui::NewFrame();
		if (!GlobalInput.bGameMode)
		{
			// get window size
			int windowWidth, windowHeight;
			glfwGetWindowSize(Window, &windowWidth, &windowHeight);
			int frameBufferWidth, frameBufferHeight;
			glfwGetFramebufferSize(Window, &frameBufferWidth, &frameBufferHeight);

			const float rightBarWidth = windowWidth * 0.2f;
			const float bottomBarWidth = windowHeight * 0.2f;

			// windows size is point but framebuffer size is pixel
			// Retina got mutiple pixel size in save window area
			ImGuiPass.RightBarSpace = frameBufferWidth * 0.2f;
			ImGuiPass.BottomBarSpace = frameBufferHeight * 0.2f;

			ImGui::PushFont(ImGuiPass.RobotoSlabSemiBold);
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));

			// create menu bar
			ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(0.05f, 0.05f, 0.05f, 1.0f));
			if (ImGui::BeginMainMenuBar())
			{
				if (ImGui::BeginMenu("File"))
				{
					if (ImGui::MenuItem("New")) 
					{
						World.Reset();
						Scene.bReload = true;
					}
					if (ImGui::MenuItem("Save"))
					{
						World.Save();
					}
					if (ImGui::MenuItem("Reload")) 
					{
						World.Load();
						Scene.bReload = true;
					}

					if (ImGui::MenuItem("Exit")) { glfwSetWindowShouldClose(Window, true); }

					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Edit"))
				{
					if (ImGui::MenuItem("Redo")) { }
					if (ImGui::MenuItem("Undo")) { }

					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Run"))
				{
					if (ImGui::MenuItem("Compile Scripts [Ctrl + B]"))
					{
					}
					if (ImGui::MenuItem("Simulate Physics [Ctrl + T]"))
					{
					}
					if (ImGui::MenuItem("Run the Game [Ctrl + R]"))
					{
					}

					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Window"))
				{
					if (ImGui::MenuItem("Focus to Select [F]"))
					{
						GlobalInput.ResetToFocus();
					}
					if (ImGui::MenuItem("Light Rolling [L]"))
					{
						GlobalInput.bPlayLightRoll = !GlobalInput.bPlayLightRoll;
					}
					if (ImGui::MenuItem("Stage Rolling [M]"))
					{
						GlobalInput.bPlayStageRoll = !GlobalInput.bPlayStageRoll;
					}
					if (ImGui::MenuItem("Game Mode [G]"))
					{
						GlobalInput.bGameMode = true;
					}

					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Help"))
				{
					if (ImGui::MenuItem("Document")) { }
					if (ImGui::MenuItem("About")) { }

					ImGui::EndMenu();
				}
				ImGui::EndMainMenuBar();
			}
			ImGui::PopStyleColor();

			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.05f, 1.0f));
			// create tree outliner
			float menuHeight = ImGui::GetFrameHeight();
			ImGui::SetNextWindowPos(ImVec2(windowWidth - rightBarWidth, menuHeight));
			ImGui::SetNextWindowSize(ImVec2(rightBarWidth, windowHeight / 2.0f));

			// create outliner window
			ImGui::Begin("Outliner", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_HorizontalScrollbar);
			if (ImGui::TreeNode("Cameras"))
			{
				if (ImGui::TreeNodeEx("Main Camera", ImGuiTreeNodeFlags_Leaf))
				{
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Lights"))
			{
				for (size_t i = 0; i < World.DirectionalLights.size(); i++)
				{
					if (ImGui::TreeNodeEx("Directional Light", ImGuiTreeNodeFlags_Leaf))
					{
						ImGui::TreePop();
					}
				}
				for (size_t i = 0; i < World.PointLights.size(); i++)
				{
					if (ImGui::TreeNodeEx("Point Light", ImGuiTreeNodeFlags_Leaf))
					{
						ImGui::TreePop();
					}
				}
				for (size_t i = 0; i < World.SpotLights.size(); i++)
				{
					if (ImGui::TreeNodeEx("Spot Light", ImGuiTreeNodeFlags_Leaf))
					{
						ImGui::TreePop();
					}
				}
				for (size_t i = 0; i < World.QuadLights.size(); i++)
				{
					if (ImGui::TreeNodeEx("Quad Light", ImGuiTreeNodeFlags_Leaf))
					{
						ImGui::TreePop();
					}
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("ShadowPass"))
			{
				if (ImGui::TreeNodeEx("PCF Shadow Map", ImGuiTreeNodeFlags_Leaf))
				{
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("SkydomePass"))
			{
				if (ImGui::TreeNodeEx("Skydome", ImGuiTreeNodeFlags_Leaf))
				{
					ImGui::TreePop();
				}
				if (ImGui::TreeNodeEx("Atmosphere", ImGuiTreeNodeFlags_Leaf))
				{
					ImGui::TreePop();
				}
				if (ImGui::TreeNodeEx("Volumetric", ImGuiTreeNodeFlags_Leaf))
				{
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("BasePass"))
			{
				for (size_t i = 0; i < World.ObjectDescs.size(); i++)
				{
					const auto& ObjectDesc = World.ObjectDescs[i];
					XkString ObjectName = ObjectDesc.ProfabName;
					ObjectName[0] = std::toupper(ObjectName[0]); // upper the first string
					if (ImGui::TreeNodeEx(ObjectName.c_str(), ImGuiTreeNodeFlags_Leaf))
					{
						ImGui::TreePop();
					}
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("BackgroundPass"))
			{
				if (ImGui::TreeNodeEx("BackgroundRect", ImGuiTreeNodeFlags_Leaf))
				{
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("PostProcess"))
			{
				if (ImGui::TreeNodeEx("GlobalPostProcessVolume", ImGuiTreeNodeFlags_Leaf))
				{
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}

			ImGui::End();

			// create details window
			ImGui::SetNextWindowPos(ImVec2(windowWidth - rightBarWidth, windowHeight / 2.0f + menuHeight));
			ImGui::SetNextWindowSize(ImVec2(rightBarWidth, windowHeight / 2.0f - menuHeight));
			ImGui::Begin("Details", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_HorizontalScrollbar);
			ImGui::Text("Select an item to view its details.");
			ImGui::End();
			// create a python IDE window
			ImGui::SetNextWindowPos(ImVec2(0, windowHeight - bottomBarWidth));
			ImGui::SetNextWindowSize(ImVec2(windowWidth - rightBarWidth, bottomBarWidth));
			ImGui::Begin("Python IDE", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
			static char code[1024 * 16] =
				"# This is a Python code example\n"
				"print('Hello, world!')\n";
			ImGui::Spacing();
			ImGui::PushFont(ImGuiPass.SourceCodeProMedium);
			ImGui::InputTextMultiline("Code", code, IM_ARRAYSIZE(code), ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 16), ImGuiInputTextFlags_AllowTabInput);
			ImGui::PopFont();
			static char filePath[1024 * 16] = "Content/World/Untitled.json";
			std::strncpy(filePath, World.FilePath.c_str(), sizeof(filePath) - 1);
			if (ImGui::InputText("##FilePath", filePath, IM_ARRAYSIZE(filePath), ImGuiInputTextFlags_AllowTabInput))
			{
				World.FilePath = filePath;
			}
			ImGui::SameLine();
			// create run button
			if (ImGui::Button("Run"))
			{
				// @TODO: Run Python code
				// ...
			}
			ImGui::End();
			ImGui::PopStyleColor();

			ImGui::PopStyleColor();
			ImGui::PopStyleColor();
			ImGui::PopFont();
		}
		else
		{
			ImGuiPass.RightBarSpace = 0.0f;
			ImGuiPass.BottomBarSpace = 0.0f;
		}
		ImGui::Render();
	}
#endif

	/** Update Uniform Buffer Object (UBO) */
	void UpdateUniformBuffer(const uint32_t currentImageIdx)
	{
		glm::vec3 Position = GlobalInput.MainCamera->Position;
		glm::vec3 Lookat = GlobalInput.MainCamera->Lookat;
		glm::vec3 CameraUp = glm::vec3(0.0f, 0.0f, 1.0f);
		float FOV = GlobalInput.MainCamera->FOV;
		float zNear = GlobalInput.MainCamera->zNear;
		float zFar = GlobalInput.MainCamera->zFar;

		static auto StartTime = std::chrono::high_resolution_clock::now();
		auto CurrentTime = std::chrono::high_resolution_clock::now();
		float Time = std::chrono::duration<float, std::chrono::seconds::period>(CurrentTime - StartTime).count();

		ShadowmapPass.zNear = zNear;
		ShadowmapPass.zFar = zFar;

		float RollLight = GlobalInput.bPlayLightRoll ?
			(GlobalInput.RollLight + GlobalInput.DeltaTime) : GlobalInput.RollLight;
		GlobalInput.RollLight = RollLight;

		glm::vec3 center = glm::vec3(0.0f);

		XkLight* MainLight = &View.DirectionalLights[0];
		glm::vec3 lightPos = glm::vec3(MainLight->Position.x, MainLight->Position.y, MainLight->Position.z);
		float RollStage = GlobalInput.bPlayStageRoll ?
			(GlobalInput.RollStage + GlobalInput.DeltaTime * glm::radians(15.0f)) : GlobalInput.RollStage;
		GlobalInput.RollStage = RollStage;
		glm::mat4 localToWorld = glm::rotate(glm::mat4(1.0f), RollStage, glm::vec3(0.0f, 0.0f, 1.0f));
		glm::mat4 shadowView = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		glm::mat4 shadowProjection = glm::perspective(glm::radians(FOV), 1.0f, ShadowmapPass.zNear, ShadowmapPass.zFar);
		shadowProjection[1][1] *= -1;

		glm::mat4 cameraView = glm::lookAt(Position, Lookat, CameraUp);
		glm::mat4 cameraProj = glm::perspective(glm::radians(FOV), SwapChainExtent.width / (float)SwapChainExtent.height, zNear, zFar);

		XkUniformBufferObject BaseData{};
		BaseData.Model = localToWorld;
		BaseData.View = cameraView;
		BaseData.Proj = cameraProj;
		BaseData.Proj[1][1] *= -1;

		void* data_base_ubo;
		vkMapMemory(Device, BaseUniformBuffersMemory[currentImageIdx], 0, sizeof(BaseData), 0, &data_base_ubo);
		memcpy(data_base_ubo, &BaseData, sizeof(BaseData));
		vkUnmapMemory(Device, BaseUniformBuffersMemory[currentImageIdx]);

		// ShadowmapSpace 的 MVP 矩阵中，M矩阵在FS中计算，所以传入 localToWorld 进入FS
		View.ViewProjSpace = cameraProj * cameraView;
		View.ShadowmapSpace = shadowProjection * shadowView;
		View.LocalToWorld = localToWorld;
		View.CameraInfo = glm::vec4(Position, FOV);
		View.ViewportInfo = glm::vec4(SwapChainExtent.width, SwapChainExtent.height, ImGuiPass.RightBarSpace, ImGuiPass.BottomBarSpace);
		uint32_t PointLightNum = View.LightsCount[1];
		for (uint32_t i = 0; i < PointLightNum; i++)
		{
			float radians = ((float)i / (float)PointLightNum) * 360.0f - RollLight * 100.0f;
			float distance = ((float)i / (float)PointLightNum) * 5.0f + 2.5f;
			float X = sin(glm::radians(radians)) * distance;
			float Y = cos(glm::radians(radians)) * distance;
			float Z = 1.5;
			View.PointLights[i].Position = glm::vec4(X, Y, Z, 1.0);
		}
		View.Time = Time;
		View.zNear = ShadowmapPass.zNear;
		View.zFar = ShadowmapPass.zFar;

		void* data_view;
		vkMapMemory(Device, ViewUniformBuffersMemory[currentImageIdx], 0, sizeof(View), 0, &data_view);
		memcpy(data_view, &View, sizeof(View));
		vkUnmapMemory(Device, ViewUniformBuffersMemory[currentImageIdx]);

		XkUniformBufferObject UBOShadowData{};
		UBOShadowData.Model = localToWorld;
		UBOShadowData.View = shadowView;
		UBOShadowData.Proj = shadowProjection;

		void* data_shadow_ubo;
		vkMapMemory(Device, ShadowmapPass.UniformBuffersMemory[currentImageIdx], 0, sizeof(UBOShadowData), 0, &data_shadow_ubo);
		memcpy(data_shadow_ubo, &UBOShadowData, sizeof(UBOShadowData));
		vkUnmapMemory(Device, ShadowmapPass.UniformBuffersMemory[currentImageIdx]);

		ShadowmapPass.RenderObjects.clear();
		ShadowmapPass.RenderInstancedObjects.clear();
		ShadowmapPass.RenderIndirectObjects.clear();
		ShadowmapPass.RenderIndirectInstancedObjects.clear();
		BasePass.RenderObjects.clear();
		BasePass.RenderInstancedObjects.clear();
#if ENABLE_INDIRECT_DRAW
		BasePass.RenderIndirectObjects.clear();
		BasePass.RenderIndirectInstancedObjects.clear();
#endif
#if ENABLE_DEFEERED_SHADING
		BasePass.RenderDeferredObjects.clear();
		BasePass.RenderDeferredInstancedObjects.clear();
#endif
		for (size_t i = 0; i < Scene.RenderObjects.size(); i++)
		{
			XkRenderObject* RenderObject = &Scene.RenderObjects[i];
			BasePass.RenderObjects.push_back(RenderObject);
			ShadowmapPass.RenderObjects.push_back(RenderObject);
		}
		for (size_t i = 0; i < Scene.RenderInstancedObjects.size(); i++)
		{
			XkRenderInstancedObject* RenderInstancedObject = &Scene.RenderInstancedObjects[i];
			BasePass.RenderInstancedObjects.push_back(RenderInstancedObject);
			ShadowmapPass.RenderInstancedObjects.push_back(RenderInstancedObject);
		}
#if ENABLE_INDIRECT_DRAW
		for (size_t i = 0; i < Scene.RenderIndirectObjects.size(); i++)
		{
			XkRenderObjectIndirect* RenderIndirectObject = &Scene.RenderIndirectObjects[i];
			BasePass.RenderIndirectObjects.push_back(RenderIndirectObject);
			ShadowmapPass.RenderIndirectObjects.push_back(RenderIndirectObject);
		}
		for (size_t i = 0; i < Scene.RenderIndirectInstancedObjects.size(); i++)
		{
			XkRenderInstancedObjectIndirect* RenderIndirectInstancedObject = &Scene.RenderIndirectInstancedObjects[i];
			BasePass.RenderIndirectInstancedObjects.push_back(RenderIndirectInstancedObject);
			ShadowmapPass.RenderIndirectInstancedObjects.push_back(RenderIndirectInstancedObject);
		}
#endif
#if ENABLE_DEFEERED_SHADING
		for (size_t i = 0; i < Scene.RenderDeferredObjects.size(); i++)
		{
			FRenderDeferredObject* RenderDeferredObject = &Scene.RenderDeferredObjects[i];
			BasePass.RenderDeferredObjects.push_back(RenderDeferredObject);
			ShadowmapPass.RenderObjects.push_back(RenderDeferredObject);
		}
		for (size_t i = 0; i < Scene.RenderDeferredInstancedObjects.size(); i++)
		{
			FRenderDeferredInstancedObject* RenderDeferredInstancedObject = &Scene.RenderDeferredInstancedObjects[i];
			BasePass.RenderDeferredInstancedObjects.push_back(RenderDeferredInstancedObject);
			ShadowmapPass.RenderInstancedObjects.push_back(RenderDeferredInstancedObject);
		}
#endif
	}

	/** Create graphics pipeline */
	void CreatePipelineLayout(
		VkPipelineLayout& outPipelineLayout, 
		const VkDescriptorSetLayout& inDescriptorSetLayout,
		const EXkRenderFlags rFlags = EXkRenderFlags::None)
	{
		// 设置 push constants
		VkPushConstantRange pushConstant;
		// 这个PushConstant的范围从头开始
		pushConstant.offset = 0;
		pushConstant.size = sizeof(XkGlobalConstants);
		// 这是个全局PushConstant，所以希望各个着色器都能访问到
		pushConstant.stageFlags = VK_SHADER_STAGE_ALL;

		// 在渲染管线创建时，指定DescriptorSetLayout，用来传UniformBuffer
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &inDescriptorSetLayout;
		pipelineLayoutInfo.pPushConstantRanges = &pushConstant;
		pipelineLayoutInfo.pushConstantRangeCount = 1;
		// PipelineLayout可以用来创建和绑定VertexBuffer和UniformBuffer，这样可以往着色器中传递参数
		if (vkCreatePipelineLayout(Device, &pipelineLayoutInfo, nullptr, &outPipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("[CreatePipelineLayout] Failed to Create pipeline layout!");
		}
	}

	/** Create graphics pipeline */
	void CreateGraphicsPipelines(
		std::vector<VkPipeline>& outPipelines,
		const VkPipelineLayout& inPipelineLayout,
		const VkRenderPass& inRenderPass,
		const XkString& inVertFilename,
		const XkString& inFragFilename,
		const EXkRenderFlags rFlags = EXkRenderFlags::None)
	{
		/* create shaders */
		auto vertShaderCode = LoadShaderSource(inVertFilename);
		auto fragShaderCode = LoadShaderSource(inFragFilename);
		VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);
		VkPipelineShaderStageCreateInfo vertShaderStageCI{};
		vertShaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageCI.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageCI.module = vertShaderModule;
		vertShaderStageCI.pName = "main";
		VkPipelineShaderStageCreateInfo fragShaderStageCI{};
		fragShaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageCI.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageCI.module = fragShaderModule;
		fragShaderStageCI.pName = "main";
		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageCI, fragShaderStageCI };

		/* vertex data description */
		auto bindingDescription = XkVertex::GetBindingDescription();
		auto attributeDescriptions = XkVertex::GetAttributeDescriptions();
		auto bindingInstancedDescriptions = XkVertex::GetBindingInstancedDescriptions();
		auto attributeInstancedDescriptions = XkVertex::GetAttributeInstancedDescriptions();

		/* vertex buffer input */
		VkPipelineVertexInputStateCreateInfo vertexInputCI{};
		if (rFlags == EXkRenderFlags::Instanced)
		{
			vertexInputCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			vertexInputCI.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingInstancedDescriptions.size());
			vertexInputCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeInstancedDescriptions.size());
			vertexInputCI.pVertexBindingDescriptions = bindingInstancedDescriptions.data();
			vertexInputCI.pVertexAttributeDescriptions = attributeInstancedDescriptions.data();
		}
		else if (rFlags == EXkRenderFlags::ScreenRect)
		{
			vertexInputCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			vertexInputCI.vertexBindingDescriptionCount = 0;
			vertexInputCI.vertexAttributeDescriptionCount = 0;
		}
		else
		{
			vertexInputCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			vertexInputCI.vertexBindingDescriptionCount = 1;
			vertexInputCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
			vertexInputCI.pVertexBindingDescriptions = &bindingDescription;
			vertexInputCI.pVertexAttributeDescriptions = attributeDescriptions.data();
		}

		/* geometry primitive type */
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyCI{};
		inputAssemblyCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssemblyCI.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportStateCI{};
		viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportStateCI.viewportCount = 1;
		viewportStateCI.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizerCI{};
		rasterizerCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizerCI.depthClampEnable = VK_FALSE;
		rasterizerCI.rasterizerDiscardEnable = VK_FALSE;
		rasterizerCI.polygonMode = VK_POLYGON_MODE_FILL;
#if ENABLE_WIREFRAME
		rasterizerCI.polygonMode = VK_POLYGON_MODE_LINE;
#endif
		rasterizerCI.lineWidth = 1.0f;
		rasterizerCI.cullMode = VK_CULL_MODE_BACK_BIT;
		if (rFlags == EXkRenderFlags::TwoSided || rFlags == EXkRenderFlags::Shadow)
		{
			rasterizerCI.cullMode = VK_CULL_MODE_NONE;
		}
		rasterizerCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizerCI.depthBiasEnable = VK_FALSE;
		if (rFlags == EXkRenderFlags::Shadow)
		{
			rasterizerCI.depthBiasEnable = VK_TRUE;
		}

		/* MSAA */
		VkPipelineMultisampleStateCreateInfo multiSamplingCI{};
		multiSamplingCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multiSamplingCI.sampleShadingEnable = VK_FALSE;
		multiSamplingCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		/* depth test */
		VkPipelineDepthStencilStateCreateInfo depthStencilCI{};
		depthStencilCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencilCI.depthTestEnable = VK_TRUE;
		depthStencilCI.depthWriteEnable = VK_TRUE;
		if (rFlags == EXkRenderFlags::NoDepthTest)
		{
			depthStencilCI.depthTestEnable = VK_FALSE;
			depthStencilCI.depthWriteEnable = VK_FALSE;
		}
		depthStencilCI.depthCompareOp = VK_COMPARE_OP_LESS;
		if (rFlags == EXkRenderFlags::Background || rFlags == EXkRenderFlags::Shadow)
		{
			depthStencilCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		}
		depthStencilCI.depthBoundsTestEnable = VK_FALSE;
		depthStencilCI.minDepthBounds = 0.0f; // Optional
		depthStencilCI.maxDepthBounds = 1.0f; // Optional
		depthStencilCI.stencilTestEnable = VK_FALSE; // 没有写轮廓信息，所以跳过轮廓测试
		depthStencilCI.front = {}; // Optional
		depthStencilCI.back = {}; // Optional

		/* color blending */
		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlendingCI{};
		colorBlendingCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlendingCI.logicOpEnable = VK_FALSE;
		colorBlendingCI.logicOp = VK_LOGIC_OP_COPY;
		colorBlendingCI.blendConstants[0] = 0.0f;
		colorBlendingCI.blendConstants[1] = 0.0f;
		colorBlendingCI.blendConstants[2] = 0.0f;
		colorBlendingCI.blendConstants[3] = 0.0f;
		colorBlendingCI.attachmentCount = 1;
		colorBlendingCI.pAttachments = &colorBlendAttachment;
		if (rFlags == EXkRenderFlags::DeferredScene)
		{
			std::array<VkPipelineColorBlendAttachmentState, 5> colorBlendAttachments;
			colorBlendAttachments[0] = colorBlendAttachment;
			colorBlendAttachments[1] = colorBlendAttachment;
			colorBlendAttachments[2] = colorBlendAttachment;
			colorBlendAttachments[3] = colorBlendAttachment;
			colorBlendAttachments[4] = colorBlendAttachment;
			colorBlendingCI.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size());
			colorBlendingCI.pAttachments = colorBlendAttachments.data();
		}
		else if (rFlags == EXkRenderFlags::DeferredLighting)
		{
			colorBlendingCI.attachmentCount = 1;
			colorBlendingCI.pAttachments = &colorBlendAttachment;
		}
		else if (rFlags == EXkRenderFlags::Shadow)
		{
			// No blend attachment states (no color attachments used)
			colorBlendingCI.attachmentCount = 0;
			colorBlendAttachment.colorWriteMask = 0xf;
			colorBlendingCI.pAttachments = &colorBlendAttachment;
		}

		std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		if (rFlags == EXkRenderFlags::Shadow)
		{
			// Add depth bias to dynamic state, so we can change it at runtime
			dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
		}
		VkPipelineDynamicStateCreateInfo dynamicStateCI{};
		dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicStateCI.pDynamicStates = dynamicStates.data();

		VkGraphicsPipelineCreateInfo pipelineCI{};
		pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineCI.stageCount = 2;
		if (rFlags == EXkRenderFlags::Shadow)
		{
			// Off screen pipeline (vertex shader only)
			pipelineCI.stageCount = 1;
		}
		pipelineCI.pStages = shaderStages;
		pipelineCI.pVertexInputState = &vertexInputCI;
		pipelineCI.pInputAssemblyState = &inputAssemblyCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pRasterizationState = &rasterizerCI;
		pipelineCI.pMultisampleState = &multiSamplingCI;
		pipelineCI.pDepthStencilState = &depthStencilCI;
		pipelineCI.pColorBlendState = &colorBlendingCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.layout = inPipelineLayout;
		pipelineCI.renderPass = inRenderPass;
		pipelineCI.subpass = 0;
		pipelineCI.basePipelineHandle = VK_NULL_HANDLE;

		// Use specialization constants optimize shader variants
		const uint32_t SpecConstantsCount = static_cast<uint32_t>(outPipelines.size());
		for (uint32_t i = 0; i < SpecConstantsCount; i++)
		{
			uint32_t SpecConstants = i;
			VkSpecializationMapEntry specializationMapEntry = VkSpecializationMapEntry{};
			specializationMapEntry.constantID = 0;
			specializationMapEntry.offset = 0;
			specializationMapEntry.size = sizeof(uint32_t);
			VkSpecializationInfo specializationCI = VkSpecializationInfo();
			specializationCI.mapEntryCount = 1;
			specializationCI.pMapEntries = &specializationMapEntry;
			specializationCI.dataSize = sizeof(uint32_t);
			specializationCI.pData = &SpecConstants;
			shaderStages[1].pSpecializationInfo = &specializationCI;

			if (vkCreateGraphicsPipelines(Device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &outPipelines[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("[CreateGraphicsPipelines] Failed to Create graphics pipeline!");
			}
		}

		vkDestroyShaderModule(Device, fragShaderModule, nullptr);
		vkDestroyShaderModule(Device, vertShaderModule, nullptr);
	}

	/** Generic function for creating DescriptorSetLayout */
	void CreateDescriptorSetLayout(VkDescriptorSetLayout& outDescriptorSetLayout, const EXkRenderFlags rFlags = EXkRenderFlags::None)
	{
		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		uint32_t samplerNumber = PBR_SAMPLER_NUMBER;
		uint32_t bindingOffset = 0;
		if (rFlags == EXkRenderFlags::Shadow)
		{
			VkDescriptorSetLayoutBinding uboLayoutBinding{};
			uboLayoutBinding.binding = 0;
			uboLayoutBinding.descriptorCount = 1;
			uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			uboLayoutBinding.pImmutableSamplers = nullptr;
			uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

			layoutBindings.resize(1);
			layoutBindings[0] = uboLayoutBinding;
		}
		else if (rFlags == EXkRenderFlags::Skydome)
		{
			samplerNumber = SKY_SAMPLER_NUMBER;
			bindingOffset = 2;
			layoutBindings.resize(samplerNumber + bindingOffset);

			VkDescriptorSetLayoutBinding baseUBOLayoutBinding{};
			baseUBOLayoutBinding.binding = 0;
			baseUBOLayoutBinding.descriptorCount = 1;
			baseUBOLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			baseUBOLayoutBinding.pImmutableSamplers = nullptr;
			baseUBOLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
			layoutBindings[0] = baseUBOLayoutBinding;

			VkDescriptorSetLayoutBinding cubemapLayoutBinding{};
			cubemapLayoutBinding.binding = 1;
			cubemapLayoutBinding.descriptorCount = 1;
			cubemapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			cubemapLayoutBinding.pImmutableSamplers = nullptr;
			cubemapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			layoutBindings[1] = cubemapLayoutBinding;

			for (size_t i = 0; i < samplerNumber; i++)
			{
				VkDescriptorSetLayoutBinding samplerLayoutBinding{};
				samplerLayoutBinding.binding = static_cast<uint32_t>(i + bindingOffset);
				samplerLayoutBinding.descriptorCount = 1;
				samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				samplerLayoutBinding.pImmutableSamplers = nullptr;
				samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

				layoutBindings[i + bindingOffset] = samplerLayoutBinding;
			}
		}
		else if (rFlags == EXkRenderFlags::Background)
		{
			samplerNumber = BG_SAMPLER_NUMBER;
			bindingOffset = 1;
			layoutBindings.resize(samplerNumber + bindingOffset);

			VkDescriptorSetLayoutBinding cubemapLayoutBinding{};
			cubemapLayoutBinding.binding = 0;
			cubemapLayoutBinding.descriptorCount = 1;
			cubemapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			cubemapLayoutBinding.pImmutableSamplers = nullptr;
			cubemapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			layoutBindings[0] = cubemapLayoutBinding;

			for (size_t i = 0; i < samplerNumber; i++)
			{
				VkDescriptorSetLayoutBinding samplerLayoutBinding{};
				samplerLayoutBinding.binding = static_cast<uint32_t>(i + bindingOffset);
				samplerLayoutBinding.descriptorCount = 1;
				samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				samplerLayoutBinding.pImmutableSamplers = nullptr;
				samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

				layoutBindings[i + bindingOffset] = samplerLayoutBinding;
			}
		}
		else if (rFlags == EXkRenderFlags::DeferredLighting)
		{
			samplerNumber = GBUFFER_SAMPLER_NUMBER;
			VkDescriptorSetLayoutBinding viewLayoutBinding{};
			viewLayoutBinding.binding = 0;
			viewLayoutBinding.descriptorCount = 1;
			viewLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			viewLayoutBinding.pImmutableSamplers = nullptr;
			viewLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			// cube map view
			VkDescriptorSetLayoutBinding cubemapLayoutBinding{};
			cubemapLayoutBinding.binding = 1;
			cubemapLayoutBinding.descriptorCount = 1;
			cubemapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			cubemapLayoutBinding.pImmutableSamplers = nullptr;
			cubemapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			// shadow map view
			VkDescriptorSetLayoutBinding shadowmapLayoutBinding{};
			shadowmapLayoutBinding.binding = 2;
			shadowmapLayoutBinding.descriptorCount = 1;
			shadowmapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			shadowmapLayoutBinding.pImmutableSamplers = nullptr;
			shadowmapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			layoutBindings.resize(samplerNumber + 3);
			layoutBindings[0] = viewLayoutBinding;
			layoutBindings[1] = cubemapLayoutBinding;
			layoutBindings[2] = shadowmapLayoutBinding;
			for (size_t i = 0; i < samplerNumber; i++)
			{
				VkDescriptorSetLayoutBinding samplerLayoutBinding{};
				samplerLayoutBinding.binding = static_cast<uint32_t>(i + 3);
				samplerLayoutBinding.descriptorCount = 1;
				samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				samplerLayoutBinding.pImmutableSamplers = nullptr;
				samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

				layoutBindings[i + 3] = samplerLayoutBinding;
			}
		}
		else
		{
			// Uniform Buffer Object (UBO) binding
			VkDescriptorSetLayoutBinding baseUBOLayoutBinding{};
			baseUBOLayoutBinding.binding = 0;
			baseUBOLayoutBinding.descriptorCount = 1;
			baseUBOLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			baseUBOLayoutBinding.pImmutableSamplers = nullptr;
			baseUBOLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

			// Uniform Buffer Object (UBO) binding
			VkDescriptorSetLayoutBinding viewUBOLayoutBinding{};
			viewUBOLayoutBinding.binding = 1;
			viewUBOLayoutBinding.descriptorCount = 1;
			viewUBOLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			viewUBOLayoutBinding.pImmutableSamplers = nullptr;
			viewUBOLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			// Bind environment reflection Cubemap texture
			VkDescriptorSetLayoutBinding cubemapLayoutBinding{};
			cubemapLayoutBinding.binding = 2;
			cubemapLayoutBinding.descriptorCount = 1;
			cubemapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			cubemapLayoutBinding.pImmutableSamplers = nullptr;
			cubemapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			VkDescriptorSetLayoutBinding shadowmapLayoutBinding{};
			shadowmapLayoutBinding.binding = 3;
			shadowmapLayoutBinding.descriptorCount = 1;
			shadowmapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			shadowmapLayoutBinding.pImmutableSamplers = nullptr;
			shadowmapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			// Bind Uniform Buffer Objects and texture samplers to DescriptorSetLayout
			layoutBindings.resize(samplerNumber + 4);
			layoutBindings[0] = baseUBOLayoutBinding;
			layoutBindings[1] = viewUBOLayoutBinding;
			layoutBindings[2] = cubemapLayoutBinding;
			layoutBindings[3] = shadowmapLayoutBinding;
			for (size_t i = 0; i < samplerNumber; i++)
			{
				VkDescriptorSetLayoutBinding samplerLayoutBinding{};
				samplerLayoutBinding.binding = static_cast<uint32_t>(i + 4);
				samplerLayoutBinding.descriptorCount = 1;
				samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				samplerLayoutBinding.pImmutableSamplers = nullptr;
				samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

				layoutBindings[i + 4] = samplerLayoutBinding;
			}
#if ENABLE_BINDLESS
			layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
#endif
		}
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
		layoutInfo.pBindings = layoutBindings.data();
		if (vkCreateDescriptorSetLayout(Device, &layoutInfo, nullptr, &outDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("[CreateDescriptorSetLayout] Failed to Create descriptor set layout!");
		}
	}

	/** Generic function for creating DescriptorSets */
	void CreateDescriptorSet(std::vector<VkDescriptorSet>& outDescriptorSets, VkDescriptorPool& outDescriptorPool,
		const VkDescriptorSetLayout& inDescriptorSetLayout, const std::vector<VkImageView>& inImageViews, const std::vector<VkSampler>& inSamplers,
		const EXkRenderFlags rFlags = EXkRenderFlags::None)
	{
		std::vector<VkDescriptorPoolSize> poolSizes;
		VkDescriptorPoolCreateInfo poolCI{};
		uint32_t samplerNumber = static_cast<uint32_t>(inSamplers.size());
		uint32_t bindingOffset = 0;
		if (rFlags == EXkRenderFlags::Shadow)
		{
			samplerNumber = 0;
			bindingOffset = 1;
			poolSizes.resize(samplerNumber + bindingOffset);
			poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		}
		else if (rFlags == EXkRenderFlags::Skydome)
		{
			bindingOffset = 2;

			poolSizes.resize(samplerNumber + bindingOffset);
			poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			for (size_t i = 0; i < samplerNumber; i++)
			{
				poolSizes[i + bindingOffset].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				poolSizes[i + bindingOffset].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			}
		}
		else if (rFlags == EXkRenderFlags::Background)
		{
			bindingOffset = 1;

			poolSizes.resize(samplerNumber + bindingOffset);
			poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			for (size_t i = 0; i < samplerNumber; i++)
			{
				poolSizes[i + bindingOffset].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				poolSizes[i + bindingOffset].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			}
		}
		else if (rFlags == EXkRenderFlags::DeferredLighting)
		{
			bindingOffset = 3;

			poolSizes.resize(samplerNumber + bindingOffset);
			poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			for (size_t i = 0; i < samplerNumber; i++)
			{
				poolSizes[i + bindingOffset].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				poolSizes[i + bindingOffset].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			}
		}
		else
		{
			bindingOffset = 4;

			poolSizes.resize(samplerNumber + bindingOffset);
			poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			poolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			poolSizes[3].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			for (size_t i = 0; i < samplerNumber; i++)
			{
				poolSizes[i + bindingOffset].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				poolSizes[i + bindingOffset].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			}

#if ENABLE_BINDLESS
			//poolCI.flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
#endif
		}

		/* create DescriptorPool */
		poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolCI.pPoolSizes = poolSizes.data();
		poolCI.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		if (vkCreateDescriptorPool(Device, &poolCI, nullptr, &outDescriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("[CreateDescriptorSet] Failed to Create descriptor pool!");
		}

		/* create DescriptorSet */
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, inDescriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = outDescriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();
		outDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(Device, &allocInfo, outDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("[CreateDescriptorSet] Failed to allocate descriptor sets!");
		}

		/* update DescriptorSet */
		UpdateDescriptorSet(outDescriptorSets, inImageViews, inSamplers, rFlags);
	}

	void UpdateDescriptorSet(
		std::vector<VkDescriptorSet>& outDescriptorSets,
		const std::vector<VkImageView>& inImageViews,
		const std::vector<VkSampler>& inSamplers,
		const EXkRenderFlags rFlags = EXkRenderFlags::None)
	{
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			uint32_t samplerNumber = static_cast<uint32_t>(inSamplers.size());
			uint32_t bindingOffset = 0;
			if (rFlags == EXkRenderFlags::Shadow)
			{
				bindingOffset = 1;
				std::vector<VkWriteDescriptorSet> descriptorWrites{};
				descriptorWrites.resize(bindingOffset);
				VkDescriptorBufferInfo bufferInfo{};
				bufferInfo.buffer = ShadowmapPass.UniformBuffers[i];
				bufferInfo.offset = 0;
				bufferInfo.range = sizeof(XkUniformBufferObject);
				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = outDescriptorSets[i];
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].pBufferInfo = &bufferInfo;

				vkUpdateDescriptorSets(Device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
			else if (rFlags == EXkRenderFlags::Skydome)
			{
				bindingOffset = 2;
				std::vector<VkWriteDescriptorSet> descriptorWrites{};
				descriptorWrites.resize(samplerNumber + bindingOffset);

				// MVP uniform buffer
				VkDescriptorBufferInfo baseBufferInfo{};
				baseBufferInfo.buffer = BaseUniformBuffers[i];
				baseBufferInfo.offset = 0;
				baseBufferInfo.range = sizeof(XkUniformBufferObject);

				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = outDescriptorSets[i];
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].pBufferInfo = &baseBufferInfo;

				// cube map view
				VkDescriptorImageInfo imageInfo{};
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo.imageView = CubemapImageView;
				imageInfo.sampler = CubemapSampler;

				descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[1].dstSet = outDescriptorSets[i];
				descriptorWrites[1].dstBinding = 1;
				descriptorWrites[1].dstArrayElement = 0;
				descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[1].descriptorCount = 1;
				descriptorWrites[1].pImageInfo = &imageInfo;

				// descriptorWrites will reference each created VkDescriptorImageInfo, so we need to store them in an array
				std::vector<VkDescriptorImageInfo> imageInfos;
				imageInfos.resize(inImageViews.size());
				for (size_t j = 0; j < samplerNumber; j++)
				{
					VkDescriptorImageInfo imageInfo{};
					imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					imageInfo.imageView = inImageViews[j];
					imageInfo.sampler = inSamplers[j];
					imageInfos[j] = imageInfo;

					descriptorWrites[j + bindingOffset].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					descriptorWrites[j + bindingOffset].dstSet = outDescriptorSets[i];
					descriptorWrites[j + bindingOffset].dstBinding = static_cast<uint32_t>(j + bindingOffset);
					descriptorWrites[j + bindingOffset].dstArrayElement = 0;
					descriptorWrites[j + bindingOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					descriptorWrites[j + bindingOffset].descriptorCount = 1;
					descriptorWrites[j + bindingOffset].pImageInfo = &imageInfos[j];
				}

				vkUpdateDescriptorSets(Device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
			else if (rFlags == EXkRenderFlags::Background)
			{
				bindingOffset = 1;
				std::vector<VkWriteDescriptorSet> descriptorWrites{};
				descriptorWrites.resize(samplerNumber + bindingOffset);

				// cube map view
				VkDescriptorImageInfo imageInfo{};
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo.imageView = CubemapImageView;
				imageInfo.sampler = CubemapSampler;

				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = outDescriptorSets[i];
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].pImageInfo = &imageInfo;

				// descriptorWrites will reference each created VkDescriptorImageInfo, so we need to store them in an array
				std::vector<VkDescriptorImageInfo> imageInfos;
				imageInfos.resize(inImageViews.size());
				for (size_t j = 0; j < samplerNumber; j++)
				{
					VkDescriptorImageInfo imageInfo{};
					imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					imageInfo.imageView = inImageViews[j];
					imageInfo.sampler = inSamplers[j];
					imageInfos[j] = imageInfo;

					descriptorWrites[j + bindingOffset].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					descriptorWrites[j + bindingOffset].dstSet = outDescriptorSets[i];
					descriptorWrites[j + bindingOffset].dstBinding = static_cast<uint32_t>(j + bindingOffset);
					descriptorWrites[j + bindingOffset].dstArrayElement = 0;
					descriptorWrites[j + bindingOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					descriptorWrites[j + bindingOffset].descriptorCount = 1;
					descriptorWrites[j + bindingOffset].pImageInfo = &imageInfos[j];
				}

				vkUpdateDescriptorSets(Device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
			else if (rFlags == EXkRenderFlags::DeferredLighting)
			{
				bindingOffset = 3;
				std::vector<VkWriteDescriptorSet> descriptorWrites{};
				descriptorWrites.resize(samplerNumber + bindingOffset);

				VkDescriptorBufferInfo viewBufferInfo{};
				viewBufferInfo.buffer = ViewUniformBuffers[i];
				viewBufferInfo.offset = 0;
				viewBufferInfo.range = sizeof(XkView);

				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = outDescriptorSets[i];
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].pBufferInfo = &viewBufferInfo;

				VkDescriptorImageInfo cubemapImageInfo{};
				cubemapImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				cubemapImageInfo.imageView = CubemapImageView;
				cubemapImageInfo.sampler = CubemapSampler;

				descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[1].dstSet = outDescriptorSets[i];
				descriptorWrites[1].dstBinding = 1;
				descriptorWrites[1].dstArrayElement = 0;
				descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[1].descriptorCount = 1;
				descriptorWrites[1].pImageInfo = &cubemapImageInfo;

				VkDescriptorImageInfo shadowmapImageInfo{};
				shadowmapImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
				shadowmapImageInfo.imageView = ShadowmapPass.ImageView;
				shadowmapImageInfo.sampler = ShadowmapPass.Sampler;

				descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[2].dstSet = outDescriptorSets[i];
				descriptorWrites[2].dstBinding = 2;
				descriptorWrites[2].dstArrayElement = 0;
				descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[2].descriptorCount = 1;
				descriptorWrites[2].pImageInfo = &shadowmapImageInfo;

				std::vector<VkDescriptorImageInfo> imageInfos;
				imageInfos.resize(samplerNumber);
				for (size_t j = 0; j < samplerNumber; j++)
				{
					VkDescriptorImageInfo imageInfo{};
					imageInfo.imageLayout = (j == 0) ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					imageInfo.imageView = inImageViews[j];
					imageInfo.sampler = inSamplers[j];
					imageInfos[j] = imageInfo;

					descriptorWrites[j + bindingOffset].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					descriptorWrites[j + bindingOffset].dstSet = outDescriptorSets[i];
					descriptorWrites[j + bindingOffset].dstBinding = static_cast<uint32_t>(j + bindingOffset);
					descriptorWrites[j + bindingOffset].dstArrayElement = 0;
					descriptorWrites[j + bindingOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					descriptorWrites[j + bindingOffset].descriptorCount = 1;
					descriptorWrites[j + bindingOffset].pImageInfo = &imageInfos[j];
				}

				vkUpdateDescriptorSets(Device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
			else
			{
				bindingOffset = 4;
				std::vector<VkWriteDescriptorSet> descriptorWrites{};
				descriptorWrites.resize(samplerNumber + bindingOffset);

				// binding transform uniform buffer
				VkDescriptorBufferInfo baseBufferInfo{};
				baseBufferInfo.buffer = BaseUniformBuffers[i];
				baseBufferInfo.offset = 0;
				baseBufferInfo.range = sizeof(XkUniformBufferObject);

				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = outDescriptorSets[i];
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].pBufferInfo = &baseBufferInfo;

				// binding view uniform buffer
				VkDescriptorBufferInfo viewBufferInfo{};
				viewBufferInfo.buffer = ViewUniformBuffers[i];
				viewBufferInfo.offset = 0;
				viewBufferInfo.range = sizeof(XkView);

				descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[1].dstSet = outDescriptorSets[i];
				descriptorWrites[1].dstBinding = 1;
				descriptorWrites[1].dstArrayElement = 0;
				descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrites[1].descriptorCount = 1;
				descriptorWrites[1].pBufferInfo = &viewBufferInfo;

				// binding cube map sampler
				VkDescriptorImageInfo imageInfo{};
				imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				imageInfo.imageView = CubemapImageView;
				imageInfo.sampler = CubemapSampler;

				descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[2].dstSet = outDescriptorSets[i];
				descriptorWrites[2].dstBinding = 2;
				descriptorWrites[2].dstArrayElement = 0;
				descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[2].descriptorCount = 1;
				descriptorWrites[2].pImageInfo = &imageInfo;

				// binding shadow map sampler
				VkDescriptorImageInfo shadowmapImageInfo{};
				shadowmapImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
				shadowmapImageInfo.imageView = ShadowmapPass.ImageView;
				shadowmapImageInfo.sampler = ShadowmapPass.Sampler;

				descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[3].dstSet = outDescriptorSets[i];
				descriptorWrites[3].dstBinding = 3;
				descriptorWrites[3].dstArrayElement = 0;
				descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[3].descriptorCount = 1;
				descriptorWrites[3].pImageInfo = &shadowmapImageInfo;

				// descriptorWrites会引用每一个创建的VkDescriptorImageInfo，所以需要用一个数组把它们存储起来
				std::vector<VkDescriptorImageInfo> imageInfos;
				imageInfos.resize(samplerNumber);
				for (size_t j = 0; j < samplerNumber; j++)
				{
					VkDescriptorImageInfo imageInfo{};
					imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					imageInfo.imageView = inImageViews[j];
					imageInfo.sampler = inSamplers[j];
					imageInfos[j] = imageInfo;

					descriptorWrites[j + bindingOffset].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					descriptorWrites[j + bindingOffset].dstSet = outDescriptorSets[i];
					descriptorWrites[j + bindingOffset].dstBinding = static_cast<uint32_t>(j + bindingOffset);
					descriptorWrites[j + bindingOffset].dstArrayElement = 0;
					descriptorWrites[j + bindingOffset].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					descriptorWrites[j + bindingOffset].descriptorCount = 1;
					// @Note: 这里是引用了VkDescriptorImageInfo，所有需要创建imageInfos这个数组，存储所有的imageInfo而不是使用局部变量imageInfo
					descriptorWrites[j + bindingOffset].pImageInfo = &imageInfos[j];
				}

				vkUpdateDescriptorSets(Device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
		}
	}

	/** Read a path to a texture and create image, image view, and sampler resources */
	void CreateImageContext(
		VkImage& outImage,
		VkDeviceMemory& outMemory,
		VkImageView& outImageView,
		VkSampler& outSampler,
		const XkString& filename, bool sRGB = true)
	{
		int texWidth, texHeight, texChannels, mipLevels;
		std::vector<uint8_t> pixels;

		LoadTextureAsset(filename, pixels, texWidth, texHeight, texChannels, mipLevels);

		VkDeviceSize imageSize = texWidth * texHeight * 4;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(Device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels.data(), static_cast<size_t>(imageSize));
		vkUnmapMemory(Device, stagingBufferMemory);

		VkFormat format = sRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
		// VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT 告诉Vulkan这张贴图即要被读也要被写
		CreateImage(
			outImage,
			outMemory,
			texWidth, texHeight, format,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mipLevels);

		TransitionImageLayout(outImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
		CopyBufferToImage(stagingBuffer, outImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		//TransitionImageLayout(outImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(Device, stagingBuffer, nullptr);
		vkFreeMemory(Device, stagingBufferMemory, nullptr);

		GenerateMipmaps(outImage, format, texWidth, texHeight, mipLevels);

		CreateImageView(outImageView, outImage, format, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		CreateSampler(outSampler,
			VK_FILTER_LINEAR,
			VK_SAMPLER_ADDRESS_MODE_REPEAT,
			VK_SAMPLER_ADDRESS_MODE_REPEAT,
			VK_SAMPLER_ADDRESS_MODE_REPEAT,
			VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			mipLevels);
	}

	/** Read a path to an HDR texture and create a CUBEMAP image resource */
	void CreateImageCubeContext(
		VkImage& outImage,
		VkDeviceMemory& outMemory,
		VkImageView& outImageView,
		VkSampler& outSampler,
		uint32_t& outMaxMipLevels,
		const std::vector<XkString>& filenames)
	{
		// https://matheowis.github.io/HDRI-to-CubeMap/
		int texWidth, texHeight, texChannels, mipLevels;
		std::vector<std::vector<uint8_t>> pixels_array;
		pixels_array.resize(6);
		for (int i = 0; i < 6; ++i) {
			LoadTextureAsset(filenames[i], pixels_array[i], texWidth, texHeight, texChannels, mipLevels);
		}
		VkDeviceSize imageSize = texWidth * texHeight * 4;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(imageSize * 6, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		for (int i = 0; i < 6; ++i) {
			void* writeLocation;
			vkMapMemory(Device, stagingBufferMemory, imageSize * i, imageSize, 0, &writeLocation);
			memcpy(writeLocation, pixels_array[i].data(), imageSize);
			vkUnmapMemory(Device, stagingBufferMemory);
		}

		// CreateImage
		{
			VkImageCreateInfo imageInfo{};
			imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageInfo.imageType = VK_IMAGE_TYPE_2D;
			imageInfo.extent.width = texWidth;
			imageInfo.extent.height = texHeight;
			imageInfo.extent.depth = 1;
			imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
			imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
			imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			imageInfo.mipLevels = mipLevels;
			imageInfo.arrayLayers = 6;
			imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

			if (vkCreateImage(Device, &imageInfo, nullptr, &outImage) != VK_SUCCESS) {
				throw std::runtime_error("[CreateImageCubeContext] Failed to Create image!");
			}

			VkMemoryRequirements memRequirements;
			vkGetImageMemoryRequirements(Device, outImage, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

			if (vkAllocateMemory(Device, &allocInfo, nullptr, &outMemory) != VK_SUCCESS) {
				throw std::runtime_error("[CreateImageCubeContext] Failed to allocate image memory!");
			}

			vkBindImageMemory(Device, outImage, outMemory, 0);
		}
		// TransitionImageLayout
		{
			VkCommandBuffer CommandBuffer = BeginSingleTimeCommands();

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = outImage;
			VkImageSubresourceRange subresourceRange;
			subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = mipLevels;
			subresourceRange.baseArrayLayer = 0;
			subresourceRange.layerCount = 6;
			barrier.subresourceRange = subresourceRange;

			VkPipelineStageFlags sourceStage;
			VkPipelineStageFlags destinationStage;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			vkCmdPipelineBarrier(
				CommandBuffer,
				sourceStage, destinationStage,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);
			EndSingleTimeCommands(CommandBuffer);
		}
		// CopyBufferToImage
		{
			VkCommandBuffer CommandBuffer = BeginSingleTimeCommands();

			VkBufferImageCopy region{};
			region.bufferOffset = 0;
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;
			VkImageSubresourceLayers imageSubresource;
			imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageSubresource.mipLevel = 0;
			imageSubresource.baseArrayLayer = 0;
			imageSubresource.layerCount = 6;
			region.imageSubresource = imageSubresource;
			region.imageOffset = { 0, 0, 0 };
			region.imageExtent = {
				static_cast<uint32_t>(texWidth),
				static_cast<uint32_t>(texHeight),
				1
			};
			vkCmdCopyBufferToImage(CommandBuffer, stagingBuffer, outImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
			EndSingleTimeCommands(CommandBuffer);
		}
		// GenerateMipmaps
		{
			VkFormatProperties formatProperties;
			vkGetPhysicalDeviceFormatProperties(PhysicalDevice, VK_FORMAT_R8G8B8A8_SRGB, &formatProperties);

			if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
				throw std::runtime_error("texture image format does not support linear blitting!");
			}

			VkCommandBuffer CommandBuffer = BeginSingleTimeCommands();

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = outImage;
			VkImageSubresourceRange subresourceRange;
			subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = 1; // 注意，这里传入的CUBEMAP只有一级Mips，不是mipLevels
			subresourceRange.baseArrayLayer = 0;
			subresourceRange.layerCount = 6;
			barrier.subresourceRange = subresourceRange;

			// 当layerCount = 6时，vkCmd执行时会 loop每个faces，所以不用在外部写循环去逐个执行Cubemap的faces ： for(uint32_t face = 0; face < 6; face++) {...}
			int32_t mipWidth = texWidth;
			int32_t mipHeight = texHeight;
			for (uint32_t i = 1; i < static_cast<uint32_t>(mipLevels); i++)
			{
				barrier.subresourceRange.baseMipLevel = i - 1;
				barrier.subresourceRange.levelCount = 1; // 注意，这里传入的CUBEMAP只有一级Mips，不是mipLevels
				barrier.subresourceRange.baseArrayLayer = 0;
				barrier.subresourceRange.layerCount = 6;
				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

				vkCmdPipelineBarrier(CommandBuffer,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
					0, nullptr,
					0, nullptr,
					1, &barrier);

				VkImageBlit blit{};
				blit.srcOffsets[0] = { 0, 0, 0 };
				blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
				blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit.srcSubresource.mipLevel = i - 1;
				blit.srcSubresource.baseArrayLayer = 0;
				blit.srcSubresource.layerCount = 6;
				blit.dstOffsets[0] = { 0, 0, 0 };
				blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
				blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				blit.dstSubresource.mipLevel = i;
				blit.dstSubresource.baseArrayLayer = 0;
				blit.dstSubresource.layerCount = 6;

				vkCmdBlitImage(CommandBuffer,
					outImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					outImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					1, &blit,
					VK_FILTER_LINEAR);

				barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

				vkCmdPipelineBarrier(CommandBuffer,
					VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
					0, nullptr,
					0, nullptr,
					1, &barrier);

				if (mipWidth > 1) mipWidth /= 2;
				if (mipHeight > 1) mipHeight /= 2;
			}

			barrier.subresourceRange.baseMipLevel = mipLevels - 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 6;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(CommandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			EndSingleTimeCommands(CommandBuffer);
		}
		vkDestroyBuffer(Device, stagingBuffer, nullptr);
		vkFreeMemory(Device, stagingBufferMemory, nullptr);

		// CreateImageView
		{
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = outImage;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
			viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = mipLevels;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 6;
			if (vkCreateImageView(Device, &viewInfo, nullptr, &outImageView) != VK_SUCCESS) {
				throw std::runtime_error("[CreateImageCubeContext] Failed to Create texture image View!");
			}
		}
		CreateSampler(outSampler,
			VK_FILTER_LINEAR,
			VK_SAMPLER_ADDRESS_MODE_REPEAT,
			VK_SAMPLER_ADDRESS_MODE_REPEAT,
			VK_SAMPLER_ADDRESS_MODE_REPEAT,
			VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			mipLevels);
		outMaxMipLevels = mipLevels;
	}

	template <typename T>
	void CreateRenderObject(T& outObject, const XkString& objfile, const std::vector<XkString>& pngfiles, const VkDescriptorSetLayout& inDescriptorSetLayout)
	{
		CreateMesh(outObject.Mesh.Vertices, outObject.Mesh.Indices, objfile);
		outObject.Material.TextureImages.resize(pngfiles.size());
		outObject.Material.TextureImageMemorys.resize(pngfiles.size());
		outObject.Material.TextureImageViews.resize(pngfiles.size());
		outObject.Material.TextureImageSamplers.resize(pngfiles.size());
		for (size_t i = 0; i < pngfiles.size(); i++)
		{
			bool sRGB = (i == 0);
			CreateImageContext(
				outObject.Material.TextureImages[i],
				outObject.Material.TextureImageMemorys[i],
				outObject.Material.TextureImageViews[i],
				outObject.Material.TextureImageSamplers[i],
				pngfiles[i], sRGB);
		}

		CreateVertexBuffer(
			outObject.Mesh.VertexBuffer,
			outObject.Mesh.VertexBufferMemory,
			outObject.Mesh.Vertices);
		CreateIndexBuffer(
			outObject.Mesh.IndexBuffer,
			outObject.Mesh.IndexBufferMemory,
			outObject.Mesh.Indices);
		CreateDescriptorSet(
			outObject.Material.DescriptorSets,
			outObject.Material.DescriptorPool,
			inDescriptorSetLayout,
			outObject.Material.TextureImageViews,
			outObject.Material.TextureImageSamplers);

		VkDeviceSize BufferSize = sizeof(XkTransfrom);
		outObject.TransfromUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		outObject.TransfromUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			CreateBuffer(BufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				outObject.TransfromUniformBuffers[i],
				outObject.TransfromUniformBuffersMemory[i]);
		}

		XkTransfrom Transfrom{};
		Transfrom.Location = glm::vec3(0.0f);
		Transfrom.Rotation = glm::vec3(0.0f);
		Transfrom.Scale3D = glm::vec3(1.0f);

		void* data_ubo;
		vkMapMemory(Device, outObject.TransfromUniformBuffersMemory[CurrentFrame], 0, sizeof(Transfrom), 0, &data_ubo);
		memcpy(data_ubo, &Transfrom, sizeof(Transfrom));
		vkUnmapMemory(Device, outObject.TransfromUniformBuffersMemory[CurrentFrame]);
	};

	template <typename T>
	void DestroyRenderObject(T& outObject)
	{
		vkDestroyDescriptorPool(Device, outObject.Material.DescriptorPool, nullptr);

		for (size_t j = 0; j < outObject.Material.TextureImages.size(); j++)
		{
			vkDestroyImageView(Device, outObject.Material.TextureImageViews[j], nullptr);
			vkDestroySampler(Device, outObject.Material.TextureImageSamplers[j], nullptr);
			vkDestroyImage(Device, outObject.Material.TextureImages[j], nullptr);
			vkFreeMemory(Device, outObject.Material.TextureImageMemorys[j], nullptr);
		}

		vkDestroyBuffer(Device, outObject.Mesh.VertexBuffer, nullptr);
		vkFreeMemory(Device, outObject.Mesh.VertexBufferMemory, nullptr);
		vkDestroyBuffer(Device, outObject.Mesh.IndexBuffer, nullptr);
		vkFreeMemory(Device, outObject.Mesh.IndexBufferMemory, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroyBuffer(Device, outObject.TransfromUniformBuffers[i], nullptr);
			vkFreeMemory(Device, outObject.TransfromUniformBuffersMemory[i], nullptr);
		}
	};

	template <typename T>
	void CreateRenderIndirectObject(T& outObject, const XkString& objfile, const std::vector<XkString>& pngfiles)
	{
		CreateMeshlet(outObject.Mesh.Vertices, outObject.Mesh.Indices,
			outObject.Mesh.MeshletSet.Meshlets, outObject.Mesh.MeshletSet.MeshletVertices, outObject.Mesh.MeshletSet.MeshletTriangles, objfile);

		outObject.Material.TextureImages.resize(pngfiles.size());
		outObject.Material.TextureImageMemorys.resize(pngfiles.size());
		outObject.Material.TextureImageViews.resize(pngfiles.size());
		outObject.Material.TextureImageSamplers.resize(pngfiles.size());
		for (size_t i = 0; i < pngfiles.size(); i++)
		{
			bool sRGB = (i == 0);
			CreateImageContext(
				outObject.Material.TextureImages[i],
				outObject.Material.TextureImageMemorys[i],
				outObject.Material.TextureImageViews[i],
				outObject.Material.TextureImageSamplers[i],
				pngfiles[i], sRGB);
		}

		std::vector<XkVertex> tmpVertices = outObject.Mesh.Vertices;
		outObject.Mesh.Vertices.resize(outObject.Mesh.MeshletSet.MeshletVertices.size());
		for (uint32_t i = 0; i < outObject.Mesh.MeshletSet.MeshletVertices.size(); i++)
		{
			outObject.Mesh.Vertices[i] = tmpVertices[outObject.Mesh.MeshletSet.MeshletVertices[i]];
		}
		outObject.Mesh.Indices.resize(outObject.Mesh.MeshletSet.MeshletTriangles.size());
		uint32_t triangleCount = 0;
		uint32_t triangleOffset = 0;
		for (uint32_t i = 0; i < outObject.Mesh.MeshletSet.Meshlets.size(); i++)
		{
			XkMeshlet meshlet = outObject.Mesh.MeshletSet.Meshlets[i];
			triangleCount += meshlet.TriangleCount;
			triangleOffset += meshlet.TriangleOffset;
		}
		for (uint32_t i = 0; i < outObject.Mesh.MeshletSet.MeshletTriangles.size(); i++)
		{
			outObject.Mesh.Indices[i] = outObject.Mesh.MeshletSet.MeshletTriangles[i];
		}
		CreateVertexBuffer(
			outObject.Mesh.VertexBuffer,
			outObject.Mesh.VertexBufferMemory,
			outObject.Mesh.Vertices);
		CreateIndexBuffer(
			outObject.Mesh.IndexBuffer,
			outObject.Mesh.IndexBufferMemory,
			outObject.Mesh.Indices);
		CreateDescriptorSet(
			outObject.Material.DescriptorSets,
			outObject.Material.DescriptorPool,
			BasePass.DescriptorSetLayout,
			outObject.Material.TextureImageViews,
			outObject.Material.TextureImageSamplers);
	};

	template <typename T>
	void CreateInstancedBuffer(T& outObject, const std::vector<XkInstanceData>& inInstanceData)
	{
		outObject.InstanceCount = static_cast<uint32_t>(inInstanceData.size());
		VkDeviceSize bufferSize = inInstanceData.size() * sizeof(XkInstanceData);
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer,
			stagingBufferMemory);

		void* data;
		vkMapMemory(Device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, inInstanceData.data(), (size_t)bufferSize);
		vkUnmapMemory(Device, stagingBufferMemory);

		CreateBuffer(
			bufferSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			outObject.Mesh.InstancedBuffer,
			outObject.Mesh.InstancedBufferMemory);
		CopyBuffer(stagingBuffer, outObject.Mesh.InstancedBuffer, bufferSize);

		vkDestroyBuffer(Device, stagingBuffer, nullptr);
		vkFreeMemory(Device, stagingBufferMemory, nullptr);
	};

	template <typename T>
	void CreateRenderIndirectBuffer(T& outObject)
	{
		VkDeviceSize bufferSize = outObject.IndirectCommands.size() * sizeof(VkDrawIndexedIndirectCommand);
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer,
			stagingBufferMemory);

		void* data;
		vkMapMemory(Device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, outObject.IndirectCommands.data(), (size_t)bufferSize);
		vkUnmapMemory(Device, stagingBufferMemory);

		CreateBuffer(
			bufferSize,
			VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			outObject.IndirectCommandsBuffer,
			outObject.IndirectCommandsBufferMemory);
		CopyBuffer(stagingBuffer, outObject.IndirectCommandsBuffer, bufferSize);

		vkDestroyBuffer(Device, stagingBuffer, nullptr);
		vkFreeMemory(Device, stagingBufferMemory, nullptr);
	};
protected:
	/** Choose the format of the image to render to the SwapChain */
	VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		// Find the suitable image format
		// VK_FORMAT_R8G8B8A8_SNORM stores the image as BGRA in unsigned normalized format, using SRGB non-linear encoding, color space is non-linear space, no need for gamma correction for the final result
		// VK_FORMAT_R8G8B8A8_UNORM stores the image as BGRA in unsigned normalized format, color space is linear space, the final output color of the pixel needs gamma correction
		for (const auto& availableFormat : availableFormats)
		{
			// Set the FrameBuffer Image to linear space for PBR workflow and color correction
			//if (availableFormat.format == VK_FORMAT_R8G8B8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			if (availableFormat.format == VK_FORMAT_R8G8B8A8_UNORM)
			{
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	/** Choose the display mode of the SwapChain
	 * VK_PRESENT_MODE_IMMEDIATE_KHR Graphics are immediately displayed on the screen, causing tearing
	 * VK_PRESENT_MODE_FIFO_KHR Images are pushed into a queue and displayed on the screen in a first-in-first-out manner. If the queue is full, the program will wait, similar to vertical synchronization
	 * VK_PRESENT_MODE_FIFO_RELAXED_KHR Based on the second mode, when the queue is full, the program will not wait and will render directly to the screen, causing tearing
	 * VK_PRESENT_MODE_MAILBOX_KHR Based on the second mode, when the queue is full, the program will not wait and will replace the image in the queue
	*/
	VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
	{
		for (const auto& availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& Capabilities)
	{
		if (Capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return Capabilities.currentExtent;
		}
		else
		{
			int Width, Height;
			glfwGetFramebufferSize(Window, &Width, &Height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(Width),
				static_cast<uint32_t>(Height)
			};

			actualExtent.width = std::clamp(actualExtent.width, Capabilities.minImageExtent.width, Capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, Capabilities.minImageExtent.height, Capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	XkSwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice Device)
	{
		XkSwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(Device, Surface, &details.Capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(Device, Surface, &formatCount, nullptr);

		if (formatCount != 0)
		{
			details.Formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(Device, Surface, &formatCount, details.Formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(Device, Surface, &presentModeCount, nullptr);

		if (presentModeCount != 0)
		{
			details.PresentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(Device, Surface, &presentModeCount, details.PresentModes.data());
		}

		return details;
	}

	/** Check if the hardware supports multiDrawIndirect */
	bool IsSupportMultiDrawIndirect(VkPhysicalDevice Device)
	{
		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(Device, &supportedFeatures);
		return supportedFeatures.multiDrawIndirect;
	}

	/** Check if the hardware is suitable */
	bool IsDeviceSuitable(VkPhysicalDevice Device)
	{
		XkQueueFamilyIndices queue_family_indices = FindQueueFamilies(Device);

		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(Device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(Device, nullptr, &extensionCount, availableExtensions.data());

		std::set<XkString> requiredExtensions(DeviceExtensions.begin(), DeviceExtensions.end());

		for (const auto& extension : availableExtensions)
		{
			requiredExtensions.erase(extension.extensionName);
		}

		bool extensionsSupported = requiredExtensions.empty();

		bool swapChainAdequate = false;
		if (extensionsSupported)
		{
			XkSwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(Device);
			swapChainAdequate = !swapChainSupport.Formats.empty() && !swapChainSupport.PresentModes.empty();
		}

		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(Device, &supportedFeatures);

		return queue_family_indices.IsComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
	}

	/** Queue Family
	 * Find all supported Vulkan hardware devices
	*/
	XkQueueFamilyIndices FindQueueFamilies(VkPhysicalDevice Device)
	{
		XkQueueFamilyIndices queue_family_indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(Device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(Device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies)
		{
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				queue_family_indices.GraphicsFamily = i;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(Device, i, Surface, &presentSupport);

			if (presentSupport)
			{
				queue_family_indices.PresentFamily = i;
			}

			if (queue_family_indices.IsComplete())
			{
				break;
			}

			i++;
		}

		return queue_family_indices;
	}

	/** Find the image formats supported by the physical hardware */
	VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(PhysicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("[FindSupportedFormat] Failed to find supported format!");
	}

	/** Find the format that supports depth textures */
	VkFormat FindDepthFormat() {
		return FindSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	/** Find memory type */
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(PhysicalDevice, &memProperties);

		// Automatically find suitable memory type
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("[FindMemoryType] Failed to find suitable memory type!");
	}

	VkCommandBuffer BeginSingleTimeCommands()
	{
		// Copy the buffer using commandBuffer, just like rendering
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = CommandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(Device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void EndSingleTimeCommands(VkCommandBuffer commandBuffer)
	{
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(GraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(GraphicsQueue);

		vkFreeCommandBuffers(Device, CommandPool, 1, &commandBuffer);
	}

	/** Utility function for creating Buffer */
	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(Device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("[CreateBuffer] Failed to Create buffer!");
		}

		// Allocate memory for VertexBuffer and bind it
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(Device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		// Automatically find the suitable memory type
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);
		// Bind the allocated memory address
		if (vkAllocateMemory(Device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("[CreateBuffer] Failed to allocate buffer memory!");
		}
		// Bind the VertexBuffer and its memory address
		vkBindBufferMemory(Device, buffer, bufferMemory, 0);
	}

	/** Utility function for copying buffers */
	void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer CommandBuffer = BeginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		vkCmdCopyBuffer(CommandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		EndSingleTimeCommands(CommandBuffer);
	}

	/** Read vertices and indices from a file */
	void CreateMesh(std::vector<XkVertex>& outVertices, std::vector<uint32_t>& outIndices, const XkString& filename)
	{
		LoadMeshAsset(filename, outVertices, outIndices);
	}

	void CreateMeshlet(std::vector<XkVertex>& outVertices, std::vector<uint32_t>& outIndices,
		std::vector<XkMeshlet>& outMeshlets, std::vector<uint32_t>& outMeshletVertices, std::vector<uint8_t>& outMeshletTriangles, const XkString& filename)
	{
		LoadMeshletAsset(filename, outVertices, outIndices, outMeshlets, outMeshletVertices, outMeshletTriangles);
	}

	/** Create vertex buffer VBO */
	void CreateVertexBuffer(VkBuffer& outBuffer, VkDeviceMemory& outMemory, const std::vector<XkVertex>& inVertices)
	{
		// Create VertexBuffer based on the size of Vertices
		VkDeviceSize bufferSize = sizeof(inVertices[0]) * inVertices.size();

		// Why do we need a stagingBuffer? Because creating the VertexBuffer directly allows CPU-side access to the GPU memory used by the VertexBufferMemory, which is dangerous.
		// So we first create a temporary Buffer to write the data to, and then copy this Buffer to the final VertexBuffer.
		// The VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT flag ensures that the final VertexBuffer is located in hardware local memory, such as the graphics card's VRAM.
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		// A generic function for creating VertexBuffer, which makes it convenient to create StagingBuffer and the actual VertexBuffer
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		// Copy the data to the vertex buffer
		void* data;
		vkMapMemory(Device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, inVertices.data(), (size_t)bufferSize);
		vkUnmapMemory(Device, stagingBufferMemory);

		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuffer, outMemory);

		CopyBuffer(stagingBuffer, outBuffer, bufferSize);

		vkDestroyBuffer(Device, stagingBuffer, nullptr);
		vkFreeMemory(Device, stagingBufferMemory, nullptr);
	}

	/** Create index buffer IBO */
	template <typename T>
	void CreateIndexBuffer(VkBuffer& outBuffer, VkDeviceMemory& outMemory, const std::vector<T>& inIndices)
	{
		VkDeviceSize bufferSize = sizeof(inIndices[0]) * inIndices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(Device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, inIndices.data(), (size_t)bufferSize);
		vkUnmapMemory(Device, stagingBufferMemory);

		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuffer, outMemory);

		CopyBuffer(stagingBuffer, outBuffer, bufferSize);

		vkDestroyBuffer(Device, stagingBuffer, nullptr);
		vkFreeMemory(Device, stagingBufferMemory, nullptr);
	}

	/** Create Shader module */
	VkShaderModule CreateShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(Device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
		{
			throw std::runtime_error("[CreateShaderModule] Failed to Create shader module!");
		}

		return shaderModule;
	}

	/** Use ImageMemoryBarrier to synchronize access to texture resources, avoiding a texture being read while it is being written */
	void TransitionImageLayout(VkImage& image, const VkImageLayout oldLayout, const VkImageLayout newLayout, const uint32_t miplevels = 1)
	{
		VkCommandBuffer CommandBuffer = BeginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = miplevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			CommandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		EndSingleTimeCommands(CommandBuffer);
	}

	void GenerateMipmaps(VkImage& outImage, const VkFormat& imageFormat, const int32_t texWidth, const int32_t texHeight, const uint32_t mipLevels)
	{
		// Check if the image format supports linear blitting
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(PhysicalDevice, imageFormat, &formatProperties);

		if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw std::runtime_error("[GenerateMipmaps] Texture image format does not support linear blitting!");
		}

		VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = outImage;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; i++) {
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0, 0, 0 };
			blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;

			vkCmdBlitImage(commandBuffer,
				outImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				outImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				VK_FILTER_LINEAR);
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		EndSingleTimeCommands(commandBuffer);
	}

	/** Copy the buffer to the image object */
	void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer CommandBuffer = BeginSingleTimeCommands();

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = { width, height, 1 };

		vkCmdCopyBufferToImage(CommandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		EndSingleTimeCommands(CommandBuffer);
	}

	/** Create image resources */
	void CreateImage(
		VkImage& outImage,
		VkDeviceMemory& outImageMemory,
		const uint32_t inWidth, const uint32_t inHeight, const VkFormat format,
		const VkImageTiling tiling, const VkImageUsageFlags usage,
		const VkMemoryPropertyFlags properties, const uint32_t miplevels = 1)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = inWidth;
		imageInfo.extent.height = inHeight;
		imageInfo.extent.depth = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.arrayLayers = 1;
		imageInfo.mipLevels = miplevels;

		if (vkCreateImage(Device, &imageInfo, nullptr, &outImage) != VK_SUCCESS) {
			throw std::runtime_error("[CreateImage] Failed to Create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(Device, outImage, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(Device, &allocInfo, nullptr, &outImageMemory) != VK_SUCCESS) {
			throw std::runtime_error("[CreateImage] Failed to allocate image memory!");
		}

		vkBindImageMemory(Device, outImage, outImageMemory, 0);
	}

	/** Create image viewport */
	void CreateImageView(
		VkImageView& outImageView,
		const VkImage& inImage,
		const VkFormat inFormat,
		const VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
		const uint32_t levelCount = 1)
	{
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = inImage;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = inFormat;
		viewInfo.subresourceRange.aspectMask = aspectFlags; // VK_IMAGE_ASPECT_COLOR_BIT 颜色 VK_IMAGE_ASPECT_DEPTH_BIT 深度
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = levelCount;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(Device, &viewInfo, nullptr, &outImageView) != VK_SUCCESS) {
			throw std::runtime_error("[CreateImageView] Failed to Create texture image View!");
		}
	}

	/** Create sampler */
	void CreateSampler(
		VkSampler& outSampler,
		const VkFilter filter = VK_FILTER_LINEAR,
		const VkSamplerAddressMode addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		const VkSamplerAddressMode addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		const VkSamplerAddressMode addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		const VkBorderColor borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		const uint32_t miplevels = 1)
	{
		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(PhysicalDevice, &properties);

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = filter;
		samplerInfo.minFilter = filter;
		samplerInfo.addressModeU = addressModeU;
		samplerInfo.addressModeV = addressModeV;
		samplerInfo.addressModeW = addressModeW;
		// Disable various anisotropy here, some hardware may not support it
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = borderColor;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = static_cast<float>(miplevels);
		samplerInfo.mipLodBias = 0.0f;

		if (vkCreateSampler(Device, &samplerInfo, nullptr, &outSampler) != VK_SUCCESS) {
			throw std::runtime_error("[CreateSampler] Failed to Create texture sampler!");
		}
	}

	template <typename T>
	void CreateRenderObjectsFromProfabs(std::vector<T>& outRenderObjects, const VkDescriptorSetLayout& inLayout, const XkString& inAssetName, const std::vector<XkInstanceData>& inInstanceData = {})
	{
		XkString asset_set_dir = "Profabs";
		for (const auto& folder : std::filesystem::directory_iterator(asset_set_dir))
		{
			XkString asset_name = folder.path().filename().generic_string();
			XkString asset_set = folder.path().generic_string();
			if (inAssetName != asset_name)
			{
				continue;
			}
			XkString models_dir = asset_set + XkString("/models/");
			XkString textures_dir = asset_set + XkString("/textures/");
			if (!std::filesystem::is_directory(models_dir) ||
				!std::filesystem::is_directory(textures_dir))
			{
				continue;
			}
			for (const auto& Model : std::filesystem::directory_iterator(models_dir))
			{
				XkString model_file = Model.path().generic_string();
				XkString model_file_name = model_file.substr(model_file.find_last_of("/\\") + 1);
				XkString::size_type const p(model_file_name.find_last_of('.'));
				XkString model_name = model_file_name.substr(0, p);
				XkString model_suffix = model_file_name.substr(p + 1);
				if (model_suffix != "obj") {
					continue;
				}
				XkString texture_bc = textures_dir + model_name + XkString("_bc.png");
				if (!std::filesystem::exists(texture_bc)) {
					texture_bc = XkString("Content/Textures/default_grey.png");
				}
				XkString texture_m = textures_dir + model_name + XkString("_m.png");
				if (!std::filesystem::exists(texture_m)) {
					texture_m = XkString("Content/Textures/default_black.png");
				}
				XkString texture_r = textures_dir + model_name + XkString("_r.png");
				if (!std::filesystem::exists(texture_r)) {
					texture_r = XkString("Content/Textures/default_white.png");
				}
				XkString texture_n = textures_dir + model_name + XkString("_n.png");
				if (!std::filesystem::exists(texture_n)) {
					texture_n = XkString("Content/Textures/default_normal.png");
				}
				XkString texture_ao = textures_dir + model_name + XkString("_ao.png");
				if (!std::filesystem::exists(texture_ao)) {
					texture_ao = XkString("Content/Textures/default_white.png");
				}
				XkString texture_ev = textures_dir + model_name + XkString("_ev.png");
				if (!std::filesystem::exists(texture_ev)) {
					texture_ev = XkString("Content/Textures/default_black.png");
				}
				XkString texture_ms = textures_dir + model_name + XkString("_ms.png");
				if (!std::filesystem::exists(texture_ms)) {
					texture_ms = XkString("Content/Textures/default_white.png");
				}

				T asset;
				XkString asset_obj = model_file;
				std::vector<XkString> asset_imgs = {
					texture_bc,
					texture_m,
					texture_r,
					texture_n,
					texture_ao,
					texture_ev,
					texture_ms };

				CreateRenderObject<T>(asset, asset_obj, asset_imgs, inLayout);
				if (inInstanceData.size() > 0)
				{
					CreateInstancedBuffer<T>(asset, inInstanceData);
				}
				outRenderObjects.push_back(asset);
			}
		}
	}

private:
	/** Load the compiled shader binary SPV file into a memory buffer */
	static std::vector<char> LoadShaderSource(const XkString& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			assert(true);
			throw std::runtime_error("[LoadShaderSource] Failed to open shader file:" + filename);
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}

	/** Load pixel information from an image file */
	static void LoadTextureAsset(const XkString& filename, std::vector<uint8_t>& outPixels, int& outWidth, int& outHeight, int& outChannels, int& outMipLevels)
	{
		stbi_hdr_to_ldr_scale(2.2f);
		stbi_uc* pixels = stbi_load(filename.c_str(), &outWidth, &outHeight, &outChannels, STBI_rgb_alpha);
		stbi_hdr_to_ldr_scale(1.0f);
		outMipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(outWidth, outHeight)))) + 1;
		if (!pixels) {
			assert(0);
			throw std::runtime_error("[LoadTextureAsset] Failed to load texture image:" + filename);
		}
		outPixels.resize(outWidth * outHeight * 4);
		std::memcpy(outPixels.data(), pixels, static_cast<size_t>(outWidth * outHeight * 4));
		// clear pixels data.
		stbi_image_free(pixels);
	}

	/** Load vertex information from a model file */
	static void LoadMeshAsset(const XkString& filename, std::vector<XkVertex>& outVertices, std::vector<uint32_t>& outIndices)
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		XkString warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str())) {
			assert(0);
			throw std::runtime_error("[LoadMeshAsset] Fail to load obj:" + warn + err);
		}

		std::unordered_map<XkVertex, uint32_t> uniqueVertices{};

		for (const auto& shape : shapes) {
			for (const auto& index : shape.mesh.indices) {
				XkVertex vertex{};

				vertex.Position = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				vertex.Normal = {
					attrib.normals[3 * index.vertex_index + 0],
					attrib.normals[3 * index.vertex_index + 1],
					attrib.normals[3 * index.vertex_index + 2]
				};

				vertex.Color = { 1.0f, 1.0f, 1.0f };

				vertex.TexCoord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};

				if (uniqueVertices.count(vertex) == 0) {
					uniqueVertices[vertex] = static_cast<uint32_t>(outVertices.size());
					outVertices.push_back(vertex);
				}

				outIndices.push_back(uniqueVertices[vertex]);
			}
		}
	}

	/** Load meshlet struct data file*/
	static void LoadMeshletAsset(const XkString& filename, std::vector<XkVertex>& outVertices, std::vector<uint32_t>& outIndices,
		std::vector<XkMeshlet>& outMeshlets, std::vector<uint32_t>& outMeshletVertices, std::vector<uint8_t>& outMeshletTriangles)
	{
		typedef XkMeshlet Meshlet;
		// Begin Copy Data Struct from ZeldaMeshlet~
		struct Vertex {
			float x, y, z;
			float nx, ny, nz;
			float u, v;
			
			bool operator==(const Vertex& other) const {
				return x == other.x && y == other.y && z == other.z &&
				nx == other.nx && ny == other.ny && nz == other.nz &&
				u == other.u && v == other.v;
			}
		};

		struct Actor {
			void save(std::ofstream& output) const
			{
				size_t size;
				
				size = meshlets.size();
				output.write((char*)&size,sizeof(size));
				output.write((char*)&meshlets[0], size * sizeof(Meshlet));

				size = meshletVertices.size();
				output.write((char*)&size,sizeof(size));
				output.write((char*)&meshletVertices[0], size * sizeof(uint32_t));

				size = meshletTriangles.size();
				output.write((char*)&size,sizeof(size));
				output.write((char*)&meshletTriangles[0], size * sizeof(uint8_t));

				size = vertices.size();
				output.write((char*)&size,sizeof(size));
				output.write((char*)&vertices[0], size * sizeof(Vertex));

				size = indices.size();
				output.write((char*)&size,sizeof(size));
				output.write((char*)&indices[0], size * sizeof(uint32_t));
			}
			
			void load(std::ifstream& input)
			{
				size_t size;
				
				input.read((char*)&size,sizeof(size));
				meshlets.resize(size);
				for (unsigned i=0; i<size; ++i)
				{
					input.read((char*)&meshlets[i],sizeof(Meshlet));
				}

				input.read((char*)&size,sizeof(size));
				meshletVertices.resize(size);
				for (unsigned i=0; i<size; ++i)
				{
					input.read((char*)&meshletVertices[i],sizeof(uint32_t));
				}

				input.read((char*)&size,sizeof(size));
				meshletTriangles.resize(size);
				for (unsigned i=0; i<size; ++i)
				{
					input.read((char*)&meshletTriangles[i],sizeof(uint8_t));
				}

				input.read((char*)&size,sizeof(size));
				vertices.resize(size);
				for (unsigned i=0; i<size; ++i)
				{
					input.read((char*)&vertices[i],sizeof(Vertex));
				}

				input.read((char*)&size,sizeof(size));
				indices.resize(size);
				for (unsigned i=0; i<size; ++i)
				{
					input.read((char*)&indices[i],sizeof(uint32_t));
				}
			}
			
			std::vector<XkMeshlet> meshlets;
			std::vector<uint32_t> meshletVertices;
			std::vector<uint8_t> meshletTriangles;
			std::vector<Vertex> vertices;
			std::vector<uint32_t> indices;
		} cache;
		// End Copy Data Struct from ZeldaMeshlet~

		std::ifstream iout(filename, std::ios::in | std::ios::binary);
		cache.load(iout);
		iout.close();

		outMeshlets = cache.meshlets;
		outMeshletVertices = cache.meshletVertices;
		outMeshletTriangles = cache.meshletTriangles;

		std::vector<Vertex> tempVertices = cache.vertices;
		for (uint32_t i = 0; i < tempVertices.size(); ++i) {
			XkVertex vertex;
			vertex.Position = {
				tempVertices[i].x,
				tempVertices[i].y,
				tempVertices[i].z
			};

			vertex.Normal = {
				tempVertices[i].nx,
				tempVertices[i].ny,
				tempVertices[i].nz
			};

			vertex.Color = { 1.0f, 1.0f, 1.0f };

			vertex.TexCoord = {
				tempVertices[i].u,
				tempVertices[i].v
			};
			outVertices.push_back(vertex);
		}
		outIndices = cache.indices;
	}
	
	// Find file in Profabs folder.
	XkString ProfabsAsset(const XkString& inFilename)
	{
		if (std::filesystem::exists(inFilename))
		{
			return inFilename;
		}
		XkString file_name_with_suffix = inFilename.substr(inFilename.find_last_of("/\\") + 1);
		XkString::size_type const p(file_name_with_suffix.find_last_of('.'));
		XkString file_name = file_name_with_suffix.substr(0, p);

		XkString asset_set_dir = "Profabs";
		for (const auto& folder : std::filesystem::directory_iterator(asset_set_dir))
		{
			XkString asset_name = folder.path().filename().generic_string();
			XkString asset_set = folder.path().generic_string();
			XkString models_dir = asset_set + XkString("/models/");
			XkString textures_dir = asset_set + XkString("/textures/");
			if (std::filesystem::is_directory(models_dir))
			{
				for (const auto& model : std::filesystem::directory_iterator(models_dir))
				{
					XkString model_file = model.path().generic_string();
					XkString model_file_name_with_suffix = model_file.substr(model_file.find_last_of("/\\") + 1);
					if (model_file_name_with_suffix == file_name_with_suffix)
					{
						return model_file;
					}
				}
			}
			if (std::filesystem::is_directory(textures_dir))
			{
				for (const auto& texture : std::filesystem::directory_iterator(textures_dir))
				{
					XkString texture_file = texture.path().generic_string();
					XkString texture_file_name_with_suffix = texture_file.substr(texture_file.find_last_of("/\\") + 1);
					if (texture_file_name_with_suffix == file_name_with_suffix)
					{
						return texture_file;
					}

				}
			}
		}
		return inFilename;
	}

	/** Select the content to print for Debug information */
	void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
	{
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = DebugCallback;
	}

	/** Check if validation layer support is available */
	bool CheckValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : ValidationLayers)
		{
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers)
			{
				if (strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
			{
				return false;
			}
		}

		return true;
	}

	/** Callback function for printing debug information, can be used to handle debug messages */
	static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
	{
		XkString output = XkString("[LOG]: ") + XkString(pCallbackData->pMessage);
		std::cerr << output.c_str() << std::endl;
#ifdef NDEBUG
		OutputDebugString(output.c_str());
#endif
		return VK_FALSE;
	}

};


int main()
{
	XkZeldaEngineApp EngineApp;

	try {
		EngineApp.Run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
