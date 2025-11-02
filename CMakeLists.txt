cmake_minimum_required(VERSION 3.20)
project(shepherd VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Generate version header from template
configure_file(version.h.in version.h @ONLY)

# Options
option(ENABLE_TENSORRT "Enable TensorRT-LLM backend" OFF)
option(ENABLE_LLAMACPP "Enable llama.cpp backend" ON)
option(ENABLE_API_BACKENDS "Enable API backends (OpenAI, Anthropic, etc)" ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(OpenSSL REQUIRED)
pkg_check_modules(SQLITE3 REQUIRED sqlite3)
pkg_check_modules(CURL REQUIRED libcurl)

# Core source files (from current Makefile)
set(SHEPHERD_SOURCES
    # Core sources
    main.cpp
    cli.cpp
    session.cpp
    config.cpp
    logger.cpp
    http_client.cpp
    rag.cpp

    # Backend sources (always included)
    backends/backend.cpp
    backends/factory.cpp
    backends/models.cpp

    # Tool sources
    tools/command_tools.cpp
    tools/core_tools.cpp
    tools/filesystem_tools.cpp
    tools/http_tools.cpp
    tools/json_tools.cpp
    tools/mcp_resource_tools.cpp
    tools/memory_tools.cpp
    tools/tool.cpp
    tools/tool_parser.cpp
    tools/utf8_sanitizer.cpp
    tools/web_search.cpp

    # MCP sources
    mcp/mcp_client.cpp
    mcp/mcp_config.cpp
    mcp/mcp.cpp
    mcp/mcp_server.cpp
    mcp/mcp_tool.cpp

    # Server sources
    server/server.cpp
)

# Add backend sources conditionally
if(ENABLE_API_BACKENDS)
    list(APPEND SHEPHERD_SOURCES
        backends/api.cpp
        backends/openai.cpp
        backends/ollama.cpp
        backends/anthropic.cpp
        backends/gemini.cpp
    )
endif()

if(ENABLE_TENSORRT)
    list(APPEND SHEPHERD_SOURCES backends/tensorrt.cpp)
endif()

if(ENABLE_LLAMACPP)
    list(APPEND SHEPHERD_SOURCES backends/llamacpp.cpp)
endif()

# Main executable
add_executable(shepherd ${SHEPHERD_SOURCES})

# Include directories
target_include_directories(shepherd PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated version.h
    ${SQLITE3_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

# TensorRT-LLM setup - find Python first
if(ENABLE_TENSORRT)
    # Check for virtual environment first, then fall back to system Python
    if(DEFINED ENV{VIRTUAL_ENV})
        set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
        message(STATUS "Using Python from virtual environment: ${Python3_EXECUTABLE}")
    else()
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
    endif()
endif()

# TensorRT-LLM headers (bundled with Shepherd + from package)
if(ENABLE_TENSORRT)
    target_include_directories(shepherd PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    # Add CUDA headers from tensorrt_llm package
    execute_process(
        COMMAND bash -c "${Python3_EXECUTABLE} -c 'import tensorrt_llm; import os; print(os.path.join(os.path.dirname(tensorrt_llm.__file__), \"include\"))' 2>/dev/null | tail -1"
        OUTPUT_VARIABLE TENSORRT_LLM_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(TENSORRT_LLM_INCLUDE_DIR)
        target_include_directories(shepherd PRIVATE ${TENSORRT_LLM_INCLUDE_DIR})
        message(STATUS "TensorRT-LLM include directory: ${TENSORRT_LLM_INCLUDE_DIR}")
    endif()

    # Add CUDA runtime headers
    execute_process(
        COMMAND bash -c "${Python3_EXECUTABLE} -c 'from nvidia import cuda_runtime; import os; print(os.path.join(os.path.dirname(cuda_runtime.__file__), \"include\"))' 2>/dev/null | tail -1"
        OUTPUT_VARIABLE CUDA_RUNTIME_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CUDA_RUNTIME_INCLUDE_DIR)
        target_include_directories(shepherd PRIVATE ${CUDA_RUNTIME_INCLUDE_DIR})
        message(STATUS "CUDA runtime include directory: ${CUDA_RUNTIME_INCLUDE_DIR}")
    endif()

    # Add triton CUDA headers (contains crt/host_defines.h and other CUDA headers)
    execute_process(
        COMMAND bash -c "${Python3_EXECUTABLE} -c 'import triton.backends.nvidia; import os; print(os.path.join(os.path.dirname(triton.backends.nvidia.__file__), \"include\"))' 2>/dev/null | tail -1"
        OUTPUT_VARIABLE TRITON_CUDA_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(TRITON_CUDA_INCLUDE_DIR)
        target_include_directories(shepherd PRIVATE ${TRITON_CUDA_INCLUDE_DIR})
        message(STATUS "Triton CUDA include directory: ${TRITON_CUDA_INCLUDE_DIR}")
    endif()
endif()

# Link standard libraries
target_link_libraries(shepherd PRIVATE
    ${SQLITE3_LIBRARIES}
    ${CURL_LIBRARIES}
    OpenSSL::Crypto
    pthread
    dl
)

# TensorRT-LLM library from pip installation
if(ENABLE_TENSORRT)
    # Find MPI
    find_package(MPI REQUIRED)
    target_include_directories(shepherd PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(shepherd PRIVATE ${MPI_CXX_LIBRARIES})

    # Try to find TensorRT-LLM from Python, fallback to hardcoded venv path
    execute_process(
        COMMAND bash -c "source ~/venv/bin/activate && python -c 'import tensorrt_llm; import os; print(os.path.join(os.path.dirname(tensorrt_llm.__file__), \"libs\"))' 2>/dev/null | tail -1"
        OUTPUT_VARIABLE TENSORRT_LLM_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TENSORRT_LLM_FIND_RESULT
    )

    # If not found, try hardcoded venv path
    if(NOT TENSORRT_LLM_LIB_DIR OR NOT EXISTS "${TENSORRT_LLM_LIB_DIR}")
        set(TENSORRT_LLM_LIB_DIR "$ENV{HOME}/venv/lib/python3.12/site-packages/tensorrt_llm/libs")
        if(NOT EXISTS "${TENSORRT_LLM_LIB_DIR}")
            message(FATAL_ERROR "Could not find tensorrt_llm library directory at ${TENSORRT_LLM_LIB_DIR}")
        endif()
    endif()

    message(STATUS "TensorRT-LLM library directory: ${TENSORRT_LLM_LIB_DIR}")

    # Find additional required libraries (TensorRT, NCCL)
    execute_process(
        COMMAND bash -c "source ~/venv/bin/activate && python -c 'import tensorrt_libs; import os; print(os.path.dirname(tensorrt_libs.__file__))' 2>/dev/null | tail -1"
        OUTPUT_VARIABLE TENSORRT_LIBS_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND bash -c "source ~/venv/bin/activate && python -c 'from nvidia import nccl; import os; print(os.path.join(os.path.dirname(nccl.__file__), \"lib\"))' 2>/dev/null | tail -1"
        OUTPUT_VARIABLE NCCL_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Fallback to hardcoded paths
    if(NOT TENSORRT_LIBS_DIR OR NOT EXISTS "${TENSORRT_LIBS_DIR}")
        set(TENSORRT_LIBS_DIR "$ENV{HOME}/venv/lib/python3.12/site-packages/tensorrt_libs")
    endif()
    if(NOT NCCL_LIB_DIR OR NOT EXISTS "${NCCL_LIB_DIR}")
        set(NCCL_LIB_DIR "$ENV{HOME}/venv/lib/python3.12/site-packages/nvidia/nccl/lib")
    endif()

    if(TENSORRT_LIBS_DIR AND EXISTS "${TENSORRT_LIBS_DIR}")
        message(STATUS "TensorRT libs directory: ${TENSORRT_LIBS_DIR}")
        target_link_directories(shepherd PRIVATE ${TENSORRT_LIBS_DIR})
    endif()
    if(NCCL_LIB_DIR AND EXISTS "${NCCL_LIB_DIR}")
        message(STATUS "NCCL library directory: ${NCCL_LIB_DIR}")
        target_link_directories(shepherd PRIVATE ${NCCL_LIB_DIR})
    endif()

    target_link_directories(shepherd PRIVATE ${TENSORRT_LLM_LIB_DIR})
    target_link_libraries(shepherd PRIVATE
        -Wl,--no-as-needed
        ${TENSORRT_LLM_LIB_DIR}/libtensorrt_llm.so
        ${TENSORRT_LLM_LIB_DIR}/libnvinfer_plugin_tensorrt_llm.so
        ${TENSORRT_LIBS_DIR}/libnvinfer.so.10
        ${NCCL_LIB_DIR}/libnccl.so.2
        ${CMAKE_SOURCE_DIR}/lib/libtokenizers_cpp.a
        ${CMAKE_SOURCE_DIR}/lib/libtokenizers_c.a
        dl
        pthread
    )

    # Add RPATH so the executable can find the libraries at runtime
    # Using --disable-new-dtags to force RPATH instead of RUNPATH (for transitive deps)
    # Using --export-dynamic to make all symbols globally available (like RTLD_GLOBAL)
    set_target_properties(shepherd PROPERTIES
        BUILD_RPATH "${TENSORRT_LLM_LIB_DIR}:${TENSORRT_LIBS_DIR}:${NCCL_LIB_DIR}"
        INSTALL_RPATH "${TENSORRT_LLM_LIB_DIR}:${TENSORRT_LIBS_DIR}:${NCCL_LIB_DIR}"
        LINK_FLAGS "-Wl,--disable-new-dtags -Wl,--export-dynamic"
        ENABLE_EXPORTS ON
    )
endif()

# Compile definitions
if(ENABLE_API_BACKENDS)
    target_compile_definitions(shepherd PRIVATE ENABLE_API_BACKENDS)
endif()

if(ENABLE_TENSORRT)
    target_compile_definitions(shepherd PRIVATE ENABLE_TENSORRT)
endif()

if(ENABLE_LLAMACPP)
    target_compile_definitions(shepherd PRIVATE ENABLE_LLAMACPP)
    # Link llama.cpp library
    set(LLAMACPP_DIR "${CMAKE_SOURCE_DIR}/llama.cpp")
    target_include_directories(shepherd PRIVATE
        ${LLAMACPP_DIR}/include
        ${LLAMACPP_DIR}/ggml/include
        ${LLAMACPP_DIR}/common
    )

    # Determine shared library extension based on platform
    if(APPLE)
        set(SHARED_LIB_EXT "dylib")
    else()
        set(SHARED_LIB_EXT "so")
    endif()

    # Link llama.cpp libraries
    target_link_directories(shepherd PRIVATE
        ${LLAMACPP_DIR}/build/bin
        ${LLAMACPP_DIR}/build/common
    )

    # Base libraries (all platforms)
    target_link_libraries(shepherd PRIVATE
        ${LLAMACPP_DIR}/build/common/libcommon.a
        llama
        ggml
        ggml-base
        ggml-cpu
    )

    # Platform-specific GPU libraries
    if(APPLE)
        # macOS: Use Metal backend
        target_link_libraries(shepherd PRIVATE ggml-metal)
    else()
        # Linux: Use CUDA backend
        target_link_libraries(shepherd PRIVATE ggml-cuda)
    endif()

    # Set RPATH so the executable can find llama.cpp libraries
    set_target_properties(shepherd PROPERTIES
        BUILD_RPATH "${LLAMACPP_DIR}/build/bin"
        INSTALL_RPATH "${LLAMACPP_DIR}/build/bin"
    )
endif()

# Readline support
find_library(READLINE_LIB readline)
if(READLINE_LIB)
    target_link_libraries(shepherd PRIVATE ${READLINE_LIB})
    target_compile_definitions(shepherd PRIVATE USE_READLINE)
    message(STATUS "Using readline for line editing")
endif()

# Compiler options (matching Makefile debug settings)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(shepherd PRIVATE
        -Wall -Wextra -g -O0 -D_DEBUG
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-but-set-variable
        -Wno-ignored-attributes
        -Wno-maybe-uninitialized
        -Wno-unused-function
    )
endif()

message(STATUS "Shepherd configuration:")
message(STATUS "  TensorRT backend: ${ENABLE_TENSORRT}")
message(STATUS "  llama.cpp backend: ${ENABLE_LLAMACPP}")
message(STATUS "  API backends: ${ENABLE_API_BACKENDS}")

# Install targets
install(TARGETS shepherd RUNTIME DESTINATION bin)

# Install server directory for HTTP API mode
install(DIRECTORY server/
        DESTINATION share/shepherd/server
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.txt"
        PATTERN "*.md")
