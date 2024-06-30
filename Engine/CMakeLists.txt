include(CMakeParseArguments)

# Function for building application icon
function(BuildAppIcon AppSources)
	set(Options)
	set(OneValueArgs OUTFILE_BASENAME)
	set(MultiValueArgs ICONS)
	cmake_parse_arguments(ARG "${Options}" "${OneValueArgs}" "${MultiValueArgs}" ${ARGN})
	if (NOT ARG_ICONS)
		message(FATAL_ERROR "No ICONS argument given to BuildAppIcon")
	endif()
	if (ARG_UNPARSED_ARGUMENTS)
		message(FATAL_ERROR "Unexpected arguments to ecm_add_app_icon: ${ARG_UNPARSED_ARGUMENTS}")
	endif()
	foreach (icon ${ARG_ICONS})
		get_filename_component(IconFull ${icon} ABSOLUTE)
		get_filename_component(IconType ${IconFull} EXT)
		get_filename_component(IconName ${IconFull} NAME_WE) 
		if (APPLE)
			if (${IconType} STREQUAL ".icns")
				set(IconFullOutput ${CMAKE_CURRENT_BINARY_DIR}/${IconName}.icns)
				configure_file(${IconFull} ${IconFullOutput} COPYONLY)
				set(MACOSX_BUNDLE_ICON_FILE ${IconName}.icns PARENT_SCOPE)
				set(${AppSources} "${${AppSources}};${IconFullOutput}" PARENT_SCOPE)
				set_source_files_properties(${IconFullOutput} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
				return()
			endif()            
		endif()
		if (MSVC)        
			if (${IconType} STREQUAL ".ico")
				set(IconFullOutput ${CMAKE_CURRENT_BINARY_DIR}/${IconName}.ico)
				configure_file(${IconFull} ${IconFullOutput} COPYONLY)
				file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${IconName}.rc.in" "IDI_ICON1 ICON DISCARDABLE\"${IconName}.ico\"\n")
				add_custom_command(
						OUTPUT "${IconName}.rc"
						COMMAND ${CMAKE_COMMAND}
						ARGS -E copy "${IconName}.rc.in" "${IconName}.rc"
						DEPENDS "${IconName}.ico"
						WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
				set(${AppSources} "${${AppSources}};${IconName}.rc" PARENT_SCOPE)
				return()
			endif()
		endif()
	endforeach()
	return()
endfunction()

# Function for building zelda engine source
function(BuildEngine ENGINE_NAME)
	SET(ENGINE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${ENGINE_NAME})
	message(STATUS "Generating project file for example in ${ENGINE_FOLDER}")
	# Main
	# file(GLOB SOURCE *.cpp ${BASE_HEADERS} ${ENGINE_FOLDER}/*.cpp)
	SET (MAIN_CPP ${ENGINE_FOLDER}/${ENGINE_NAME}.cpp)
	set (CONTENT_DIR ${ENGINE_FOLDER}/Content)
	set (SHADERS_DIR ${ENGINE_FOLDER}/Shaders)
	set (CONTENT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Content)
	set (SHADERS_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shaders)
	set (PROFABS_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Profabs)

	if(EXISTS ${ENGINE_FOLDER}/main.cpp)
		SET(MAIN_CPP ${ENGINE_FOLDER}/main.cpp)
	endif()
	if(EXISTS ${ENGINE_FOLDER}/${ENGINE_NAME}.h)
		SET(MAIN_HEADER ${ENGINE_FOLDER}/${ENGINE_NAME}.h)
	endif()
	# Set application icon
	BuildAppIcon(MAIN_CPP ICONS ${CONTENT_DIR}/AppData/vulkan_renderer.ico ${CONTENT_DIR}/AppData/vulkan_renderer.icns)

	# Compile shader and copy resources
	file(GLOB SHADER_SOURCES "${SHADERS_DIR}/*")
	add_custom_command(
		# Compile multiple shaders but just test Base_VS.spv output
		OUTPUT always_rebuild
		COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_PATH}
		COMMAND glslc ARGS -g ${SHADERS_DIR}/Base.frag -o ${SHADERS_PATH}/Base.frag.spv
		COMMAND glslc ARGS -g ${SHADERS_DIR}/Base.vert -o ${SHADERS_PATH}/Base.vert.spv
		COMMAND glslc ARGS -g ${SHADERS_DIR}/BaseInstanced.vert -o ${SHADERS_PATH}/BaseInstanced.vert.spv
		COMMAND glslc ARGS -g ${SHADERS_DIR}/BaseScene.frag -o ${SHADERS_PATH}/BaseScene.frag.spv
		COMMAND glslc ARGS -g ${SHADERS_DIR}/BaseLighting.frag -o ${SHADERS_PATH}/BaseLighting.frag.spv
		COMMAND glslc ARGS -g ${SHADERS_DIR}/Shadowmap.frag -o ${SHADERS_PATH}/Shadowmap.frag.spv
		COMMAND glslc ARGS -g ${SHADERS_DIR}/Shadowmap.vert -o ${SHADERS_PATH}/Shadowmap.vert.spv
		COMMAND glslc ARGS -g ${SHADERS_DIR}/ShadowmapInstanced.vert -o ${SHADERS_PATH}/ShadowmapInstanced.vert.spv
		COMMAND glslc ARGS -g ${SHADERS_DIR}/Background.frag -o ${SHADERS_PATH}/Background.frag.spv
		COMMAND glslc ARGS -g ${SHADERS_DIR}/Background.vert -o ${SHADERS_PATH}/Background.vert.spv
		COMMAND glslc ARGS -g ${SHADERS_DIR}/Skydome.frag -o ${SHADERS_PATH}/Skydome.frag.spv
		COMMAND glslc ARGS -g ${SHADERS_DIR}/Skydome.vert -o ${SHADERS_PATH}/Skydome.vert.spv
		WORKING_DIRECTORY ${SHADERS_DIR}
		DEPENDS ${SHADER_SOURCES}
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CONTENT_DIR}/ ${CONTENT_PATH}/
		VERBATIM
		COMMAND ${CMAKE_COMMAND} -E make_directory ${PROFABS_PATH}
		VERBATIM
		COMMENT "Update Zelda Engine Resource Success!"
	)

	set(COMPILE_SHADER_TARGET ${ENGINE_NAME}Shaders)
	add_custom_target(${COMPILE_SHADER_TARGET} ALL DEPENDS always_rebuild SOURCES ${SHADER_SOURCES})

	# add executable
	if(WIN32)
		add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
		add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
		add_executable(${ENGINE_NAME} WIN32 ${MAIN_CPP} ${IMGUI_SOURCES} ${OPENFBX_SOURCES})
		target_link_libraries(${ENGINE_NAME} ${Vulkan_LIBRARY} ${WINLIBS})
	else(WIN32)
		# Note: add_subdirectory(ZeldaEngine) and add_executable(ZeldaEngine) have the same name which will cause makefile link error
		add_executable(${ENGINE_NAME} ${MAIN_CPP} ${IMGUI_SOURCES} ${OPENFBX_SOURCES})
		target_link_libraries(${ENGINE_NAME} ${Vulkan_LIBRARY})
	endif(WIN32)

	# add shader compile dependency
	add_dependencies (${ENGINE_NAME} ${COMPILE_SHADER_TARGET})

	set_target_properties(${ENGINE_NAME} PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	set_target_properties(${ENGINE_NAME} PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
	if(WIN32 AND ${CMAKE_GENERATOR} MATCHES "Visual Studio")
		#set(SUBSYSTEM_LINKER_OPTIONS "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
		set(SUBSYSTEM_LINKER_OPTIONS "/SUBSYSTEM:CONSOLE") # with debug log window
		set_target_properties(${ENGINE_NAME} PROPERTIES LINK_FLAGS ${SUBSYSTEM_LINKER_OPTIONS} VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	endif(WIN32 AND ${CMAKE_GENERATOR} MATCHES "Visual Studio")
	if(APPLE AND ${CMAKE_GENERATOR} STREQUAL "Xcode")
		set_target_properties(${ENGINE_NAME} PROPERTIES XCODE_GENERATE_SCHEME TRUE XCODE_SCHEME_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	endif(APPLE AND ${CMAKE_GENERATOR} STREQUAL "Xcode")
	# Note: fix glslang libraries link issue
	# https://stackoverflow.com/questions/38234986/how-to-use-glslang
	target_link_libraries(${ENGINE_NAME} glfw glm meshoptimizer glslang SPIRV glslang-default-resource-limits)
	# window socket library
	if(WIN32)
		target_link_libraries(${ENGINE_NAME} Ws2_32)
	endif(WIN32)
endfunction(BuildEngine)

# Function for building engine tools
function(BuildToolkit TOOLKIT_NAME)
	SET(TOOL_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${TOOLKIT_NAME})
	message(STATUS "Generating tool file in ${TOOL_FOLDER}")
	SET (MAIN_CPP ${TOOL_FOLDER}/${TOOLKIT_NAME}.cpp)

	if(EXISTS ${TOOL_FOLDER}/main.cpp)
		SET(MAIN_CPP ${TOOL_FOLDER}/main.cpp)
	endif()
	if(EXISTS ${TOOL_FOLDER}/${TOOLKIT_NAME}.h)
		SET(MAIN_HEADER ${TOOL_FOLDER}/${TOOLKIT_NAME}.h)
	endif()
	if(WIN32)
		add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
		add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
		add_executable(${TOOLKIT_NAME} WIN32 ${MAIN_CPP})
	else(WIN32)
		add_executable(${TOOLKIT_NAME} ${MAIN_CPP})
	endif(WIN32)
	
	set_target_properties(${TOOLKIT_NAME} PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	set_target_properties(${TOOLKIT_NAME} PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
	target_link_libraries(${TOOLKIT_NAME} glfw glm meshoptimizer)
endfunction(BuildToolkit)

# Function for building shared library
function(BuildSharedLib SHAREDLIB_NAME)
	# a python lib source name
	SET(SHAREDLIB_PY_NAME ${SHAREDLIB_NAME}Lib)
	SET(LIB_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${SHAREDLIB_NAME})
	message(STATUS "Generating tool file in ${LIB_FOLDER}")
	SET (MAIN_CPP ${LIB_FOLDER}/${SHAREDLIB_PY_NAME}.cpp)

	if(EXISTS ${LIB_FOLDER}/main.cpp)
		SET(MAIN_CPP ${LIB_FOLDER}/main.cpp)
	endif()
	if(EXISTS ${LIB_FOLDER}/${SHAREDLIB_PY_NAME}.h)
		SET(MAIN_HEADER ${LIB_FOLDER}/${SHAREDLIB_PY_NAME}.h)
	endif()

	add_library(${SHAREDLIB_PY_NAME} STATIC SHARED ${MAIN_CPP})
	target_sources(${SHAREDLIB_PY_NAME} PUBLIC ${MAIN_HEADER})
	if(WIN32)
		add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
		add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
		target_link_libraries(${SHAREDLIB_PY_NAME} PRIVATE Ws2_32)
	endif(WIN32)
	target_link_libraries(${SHAREDLIB_PY_NAME} PRIVATE glfw glm meshoptimizer)
	# option name for shared library
	set_target_properties(${SHAREDLIB_PY_NAME} PROPERTIES OUTPUT_NAME ${SHAREDLIB_PY_NAME})
endfunction(BuildSharedLib)

BuildEngine(ZeldaEngine)
BuildToolkit(ZeldaMeshlet)
BuildToolkit(ZeldaPython)
BuildSharedLib(ZeldaPython)