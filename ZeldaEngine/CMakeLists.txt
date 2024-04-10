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
	set (SHADERS_DIR ${ENGINE_FOLDER}/Shaders)
	set (RESOURCES_DIR ${ENGINE_FOLDER}/Resources)
    set (RESOURCES_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Resources)
    set (SHADERS_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Resources/Shaders)
    set (PROFABS_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Resources/Profabs)

	if(EXISTS ${ENGINE_FOLDER}/main.cpp)
		SET(MAIN_CPP ${ENGINE_FOLDER}/main.cpp)
	endif()
	if(EXISTS ${ENGINE_FOLDER}/${ENGINE_NAME}.h)
		SET(MAIN_HEADER ${ENGINE_FOLDER}/${ENGINE_NAME}.h)
	endif()
    # Set application icon
    BuildAppIcon(MAIN_CPP ICONS ${RESOURCES_DIR}/Appicons/vulkan_renderer.ico ${RESOURCES_DIR}/Appicons/vulkan_renderer.icns)
	if(WIN32)
        add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
        add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
		add_executable(${ENGINE_NAME} WIN32 ${MAIN_CPP})
		target_link_libraries(${ENGINE_NAME} ${Vulkan_LIBRARY} ${WINLIBS})
	else(WIN32)
		add_executable(${ENGINE_NAME} ${MAIN_CPP})
		target_link_libraries(${ENGINE_NAME})
	endif(WIN32)

	# Compile shader and copy texures
	add_custom_command(
    OUTPUT SHADER_COMPILE
	COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_PATH}
    COMMAND glslc ARGS -g ${SHADERS_DIR}/SceneFS.frag -o ${SHADERS_PATH}/Scene_FS.spv
    COMMAND glslc ARGS -g ${SHADERS_DIR}/SceneVS.vert -o ${SHADERS_PATH}/Scene_VS.spv
    COMMAND glslc ARGS -g ${SHADERS_DIR}/SceneBaseFS.frag -o ${SHADERS_PATH}/SceneBase_FS.spv
    COMMAND glslc ARGS -g ${SHADERS_DIR}/SceneBaseLightingFS.frag -o ${SHADERS_PATH}/SceneBaseLighting_FS.spv
    COMMAND glslc ARGS -g ${SHADERS_DIR}/SceneInstancedVS.vert -o ${SHADERS_PATH}/SceneInstanced_VS.spv
    COMMAND glslc ARGS -g ${SHADERS_DIR}/ShadowmapFS.frag -o ${SHADERS_PATH}/Shadowmap_FS.spv
    COMMAND glslc ARGS -g ${SHADERS_DIR}/ShadowmapVS.vert -o ${SHADERS_PATH}/Shadowmap_VS.spv
    COMMAND glslc ARGS -g ${SHADERS_DIR}/ShadowmapInstancedVS.vert -o ${SHADERS_PATH}/ShadowmapInstanced_VS.spv
    COMMAND glslc ARGS -g ${SHADERS_DIR}/BackgroundFS.frag -o ${SHADERS_PATH}/Background_FS.spv
    COMMAND glslc ARGS -g ${SHADERS_DIR}/BackgroundVS.vert -o ${SHADERS_PATH}/Background_VS.spv
    COMMAND glslc ARGS -g ${SHADERS_DIR}/SkydomeFS.frag -o ${SHADERS_PATH}/Skydome_FS.spv
    COMMAND glslc ARGS -g ${SHADERS_DIR}/SkydomeVS.vert -o ${SHADERS_PATH}/Skydome_VS.spv
	WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Shaders Success!"
	VERBATIM
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCES_DIR}/ ${RESOURCES_PATH}/
    COMMENT "Copying Resources Success!"
    VERBATIM
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PROFABS_PATH}
    COMMENT "Create Profabs Dir Success!"
    )

	set(COMPILE_SHADER_TARGET ${ENGINE_NAME}Shaders)
	set(SHADER_SOURCES 
        ${SHADERS_DIR}/SceneFS.frag ${SHADERS_DIR}/SceneVS.vert ${SHADERS_DIR}/SceneInstancedVS.vert
        ${SHADERS_DIR}/SceneBaseFS.frag ${SHADERS_DIR}/SceneBaseLightingFS.frag
        ${SHADERS_DIR}/ShadowmapFS.frag ${SHADERS_DIR}/ShadowmapVS.vert ${SHADERS_DIR}/ShadowmapInstancedVS.vert
        ${SHADERS_DIR}/BackgroundFS.frag ${SHADERS_DIR}/BackgroundVS.vert
        ${SHADERS_DIR}/SkydomeFS.frag ${SHADERS_DIR}/SkydomeVS.vert
        )
	add_custom_target(${COMPILE_SHADER_TARGET} ALL DEPENDS SHADER_COMPILE SOURCES ${SHADER_SOURCES})
	add_dependencies (${ENGINE_NAME} ${COMPILE_SHADER_TARGET})
	
	set_target_properties(${ENGINE_NAME} PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set_target_properties(${ENGINE_NAME} PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
    if(WIN32)
        #set(SUBSYSTEM_LINKER_OPTIONS "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
        set(SUBSYSTEM_LINKER_OPTIONS "/SUBSYSTEM:CONSOLE") # with debug log window
        set_target_properties(${ENGINE_NAME} PROPERTIES LINK_FLAGS ${SUBSYSTEM_LINKER_OPTIONS} VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif(WIN32)
    if(APPLE)
        set_target_properties(${ENGINE_NAME} PROPERTIES XCODE_GENERATE_SCHEME TRUE XCODE_SCHEME_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif(APPLE)
    # @Note: fix glslang libraries link issue
    # https://stackoverflow.com/questions/38234986/how-to-use-glslang
    target_link_libraries(${ENGINE_NAME} glfw glm glslang SPIRV glslang-default-resource-limits)

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
		target_link_libraries(${TOOLKIT_NAME} ${Vulkan_LIBRARY} ${WINLIBS})
	else(WIN32)
		add_executable(${TOOLKIT_NAME} ${MAIN_CPP})
		target_link_libraries(${TOOLKIT_NAME})
	endif(WIN32)
	
	set_target_properties(${TOOLKIT_NAME} PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set_target_properties(${TOOLKIT_NAME} PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
    if(WIN32)
        set_target_properties(${TOOLKIT_NAME} PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif(WIN32)
    if(APPLE)
        set_target_properties(${TOOLKIT_NAME} PROPERTIES XCODE_GENERATE_SCHEME TRUE XCODE_SCHEME_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif(APPLE)
	target_link_libraries(${TOOLKIT_NAME} glfw glm meshoptimizer tinyobjloader)
endfunction(BuildToolkit)

BuildEngine(ZeldaEngine)
BuildToolkit(ZeldaMeshlet)
