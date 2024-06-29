// Copyright ©XUKAI. All Rights Reserved.

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <cstdlib>

static void printUsage(void) {
    printf("Usage: zeldaPython [<python scripts>] [-o <output json file>]\n");
}

int main(int argc, const char * argv[]) {
    std::string inputPath;
    std::string outputPath;
    int maxVertCount = 64, maxTriCount = 124;

#ifdef _WIN32
	if (_putenv_s("PYTHONPATH", "=%PYTHONPATH%;C:/path/to/your/module") != 0) {
		std::cerr << "Failed to set environment variable" << std::endl;
		return 1;
	}
#elif __APPLE__
	if (setenv("PYTHONPATH", "=%PYTHONPATH%;C:/path/to/your/module", 1) != 0) {
		std::cerr << "Failed to set environment variable" << std::endl;
		return 1;
	}
#endif

    if (argc != 2) {
        printUsage();
        return 0;
    }

    std::string cmd = "python" + std::string(argv[0]) + std::string(argv[1]);
    system(cmd.c_str());
    return 1;
}
