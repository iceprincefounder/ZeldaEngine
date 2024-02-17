// Copyright @xukai. All Rights Reserved.

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>

#include <meshoptimizer.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

static const size_t kMeshletMaxVertices = 255;  // constants copied from meshoptimizer/clusterizer.cpp
static const size_t kMeshletMaxTriangles = 512;

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

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<float>()(vertex.x) ^ (hash<float>()(vertex.y) << 1)) >> 1) ^ (hash<float>()(vertex.z) << 1);
        }
    };
}

struct Meshlet {
    uint32_t vertexOffset;
    uint32_t vertexCount;
    uint32_t triangleOffset;
    uint32_t triangleCount;
    float boundsCenter[3];
    float boundsRadius;
    float coneApex[3];
    float coneAxis[3];
    float coneCutoff, pad;
};

struct FActor {
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
    
    std::vector<Meshlet> meshlets;
    std::vector<uint32_t> meshletVertices;
    std::vector<uint8_t> meshletTriangles;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

static void printUsage(void) {
    printf("Usage: meshletgen [-v <max_meshlet_vertex_count>] [-t <max_meshlet_primitive_count>] -i <input_path> -o <output_path>\n");
}

static void printUnknownOption(const char* flag) {
    printf("Unknown flag or option: %s\n", flag);
}

static bool BuildMeshlets(std::vector<Vertex> const& vertices, std::vector<uint32_t> const& indices,
                          size_t maxMeshletVertexCount, size_t maxMeshletTriangleCount,
                          std::vector<Meshlet> /* out */ &outMeshlets,
                          std::vector<uint32_t> /* out */ &outMeshletVertices,
                          std::vector<uint8_t> /* out */ &outMeshletTriangles)
{
    const float coneWeight = 0.2f;
    size_t maxMeshletCount = meshopt_buildMeshletsBound(indices.size(), maxMeshletVertexCount, maxMeshletTriangleCount);
    std::vector<meshopt_Meshlet> meshletsInternal(maxMeshletCount);
    outMeshletVertices = std::vector<uint32_t>(maxMeshletCount * maxMeshletVertexCount);
    outMeshletTriangles = std::vector<uint8_t>(maxMeshletCount * maxMeshletTriangleCount * 3);

    size_t meshletCount = meshopt_buildMeshlets(meshletsInternal.data(), outMeshletVertices.data(), outMeshletTriangles.data(),
                                                indices.data(), indices.size(), &vertices[0].x, vertices.size(),
                                                sizeof(Vertex), maxMeshletVertexCount, maxMeshletTriangleCount, coneWeight);

    outMeshlets.reserve(meshletCount);
    for (int i = 0; i < meshletCount; ++i) {
        auto const& meshlet = meshletsInternal[i];
        meshopt_Bounds bounds = meshopt_computeMeshletBounds(outMeshletVertices.data() + meshlet.vertex_offset,
                                                             outMeshletTriangles.data() + meshlet.triangle_offset,
                                                             meshlet.triangle_count,
                                                             &vertices.data()[0].x,
                                                             vertices.size(), sizeof(Vertex));

        outMeshlets.push_back(Meshlet {
            meshlet.vertex_offset, meshlet.vertex_count,
            meshlet.triangle_offset, meshlet.triangle_count,
            { bounds.center[0], bounds.center[1], bounds.center[2] }, bounds.radius,
            { bounds.cone_apex[0], bounds.cone_apex[1], bounds.cone_apex[2] },
            { bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2] },
            bounds.cone_cutoff,
            0.0f // pad
        });
    }

    // Trim zeros from overestimated meshlet count
    outMeshletTriangles.resize(outMeshlets.back().triangleOffset + outMeshlets.back().triangleCount * 3);

    return (meshletCount > 0);
}

static bool buildMeshletsFromAsset(const std::string& inputURL, const std::string& outputURL, size_t maxMeshletVertexCount, size_t maxMeshletTriangleCount)
{
    std::vector<Vertex> cachedVertices;
    std::vector<uint32_t> cachedIndices;
    
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputURL.c_str())) {
        assert(true);
        throw std::runtime_error(warn + err);
    }

    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    for (const auto& shape : shapes) 
    {
        for (const auto& index : shape.mesh.indices) 
        {
            Vertex vertex{};

            vertex.x = attrib.vertices[3 * index.vertex_index + 0];
            vertex.y = attrib.vertices[3 * index.vertex_index + 1];
            vertex.z = attrib.vertices[3 * index.vertex_index + 2];

            vertex.nx = attrib.normals[3 * index.normal_index + 0];
            vertex.ny = attrib.normals[3 * index.normal_index + 1];
            vertex.nz = attrib.normals[3 * index.normal_index + 2];

            //vertex.Color = { 1.0f, 1.0f, 1.0f };

            vertex.u = attrib.texcoords[2 * index.texcoord_index + 0];
            vertex.v = 1.0f - attrib.texcoords[2 * index.texcoord_index + 1];

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(cachedVertices.size());
                cachedVertices.push_back(vertex);
            }
            cachedIndices.push_back(uniqueVertices[vertex]);
        }
    }
    std::vector<Meshlet> cachedMeshlets;
    std::vector<uint32_t> cachedMeshletVertices;
    std::vector<uint8_t> cachedMeshletTriangles;
    BuildMeshlets(cachedVertices, cachedIndices, maxMeshletVertexCount, maxMeshletTriangleCount, cachedMeshlets, cachedMeshletVertices, cachedMeshletTriangles);
    FActor actor;
    actor.meshlets = cachedMeshlets;
    actor.meshletVertices = cachedMeshletVertices;
    actor.meshletTriangles = cachedMeshletTriangles;
    actor.vertices = cachedVertices;
    actor.indices = cachedIndices;

    std::ofstream fout(outputURL, std::ios::out | std::ios::binary);
    actor.save(fout);
    fout.close();

    return true;
}

int main(int argc, const char * argv[]) {
    std::string inputPath;
    std::string outputPath;
    int maxVertCount = 64, maxTriCount = 124;

    inputPath = "Resources/Contents/Meshes/sphere.obj";
    outputPath = "Resources/Contents/Meshes/sphere.meshlet";

    buildMeshletsFromAsset(inputPath, outputPath, maxVertCount, maxTriCount);

    return 1;

    if (argc < 5) {
        printUsage();
        return 0;
    }

    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-v") == 0) {
            assert(argc > i);
            maxVertCount = atoi(argv[i + 1]);
            continue;
        }
        if (strcmp(argv[i], "-t") == 0) {
            assert(argc > i);
            maxTriCount = atoi(argv[i + 1]);
            continue;
        }
        if (strcmp(argv[i], "-i") == 0) {
            assert(argc > i);
            inputPath = std::string(argv[i + 1]);
            continue;
        }
        if (strcmp(argv[i], "-o") == 0) {
            assert(argc > i);
            outputPath = std::string(argv[i + 1]);
            continue;
        }
        printUnknownOption(argv[i]);
        return 1;
    }

    if (std::filesystem::exists(inputPath))
    {
        printf("Could not find input file %s\n", inputPath.c_str());
        return 1;
    }

    // Sanity-check output limits
    if (maxVertCount < 3) { maxVertCount = 3; }
    if (maxTriCount < 1) { maxTriCount = 1; }

    // Constrain output limits to meshoptimizer implementation limits
    if (maxVertCount > kMeshletMaxVertices) { maxVertCount = kMeshletMaxVertices; }
    if (maxTriCount > kMeshletMaxTriangles) { maxTriCount = kMeshletMaxTriangles; }

    buildMeshletsFromAsset(inputPath, outputPath, maxVertCount, maxTriCount);

    return 1;
}
