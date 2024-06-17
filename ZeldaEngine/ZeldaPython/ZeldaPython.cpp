#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>

static void printUsage(void) {
    printf("Usage: zeldaPython [<python scripts>] [-o <output json file>]\n");
}

int main(int argc, const char * argv[]) {
    std::string inputPath;
    std::string outputPath;
    int maxVertCount = 64, maxTriCount = 124;

    inputPath = "Resources/Profabs/dragon/models/dragon.obj";
    outputPath = "Resources/Profabs/dragon/models/dragon.meshlet";

    if (argc != 2) {
        printUsage();
        return 0;
    }

    return 1;
}
