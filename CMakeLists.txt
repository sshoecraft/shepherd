cmake_minimum_required(VERSION 3.20)
project(shepherd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    session.cpp
    config.cpp
    provider.cpp
    generation_thread.cpp
    http_client.cpp
    rag.cpp
    sse_parser.cpp
    scheduler.cpp
    frontend.cpp
    backend.cpp
    server.cpp
    auth.cpp

    # Frontend sources
    frontends/cli.cpp
    frontends/tui.cpp
    frontends/api_server.cpp
    frontends/cli_server.cpp
    frontends/client_output.cpp

    # Backend sources (always included)
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
    tools/tools.cpp
    tools/tool_parser.cpp
    tools/utf8_sanitizer.cpp
    tools/web_search.cpp

    # MCP sources
    mcp/mcp_client.cpp
    mcp/mcp_config.cpp
    mcp/mcp.cpp
    mcp/mcp_server.cpp
    mcp/mcp_tool.cpp

    # API tools sources
    tools/api_tools.cpp
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

# CLI client backend (always available - simple HTTP)
list(APPEND SHEPHERD_SOURCES backends/cli_client.cpp)

# GPU backend base class and shared code (needed by both llamacpp and tensorrt)
if(ENABLE_LLAMACPP OR ENABLE_TENSORRT)
    list(APPEND SHEPHERD_SOURCES
        backends/gpu.cpp
        backends/chat_template.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/frontends
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

    # Find Python library for linking
    find_package(Python3 COMPONENTS Development REQUIRED)
endif()

# TensorRT-LLM headers (bundled with Shepherd + from package)
if(ENABLE_TENSORRT)
    # Detect installed TensorRT-LLM version
    # Read version.py directly to avoid import side effects
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import site; exec(open(site.getsitepackages()[0]+'/tensorrt_llm/version.py').read()); print(__version__)"
        OUTPUT_VARIABLE TRT_LLM_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT TRT_LLM_VERSION)
        message(FATAL_ERROR "Could not detect TensorRT-LLM version. Is tensorrt_llm installed?")
    endif()
    message(STATUS "Detected TensorRT-LLM version: ${TRT_LLM_VERSION}")

    # Check if we have headers for this version
    set(TRT_LLM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/tensorrt-llm/${TRT_LLM_VERSION}")
    if(NOT EXISTS "${TRT_LLM_INCLUDE_DIR}")
        message(FATAL_ERROR "No headers found for TensorRT-LLM ${TRT_LLM_VERSION}. Expected: ${TRT_LLM_INCLUDE_DIR}")
    endif()
    message(STATUS "Using TensorRT-LLM headers from: ${TRT_LLM_INCLUDE_DIR}")

    target_include_directories(shepherd PRIVATE
        ${TRT_LLM_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/include
    )
    # Add CUDA headers from tensorrt_llm package
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import site, os; print(os.path.join(site.getsitepackages()[0], 'tensorrt_llm', 'include'))"
        OUTPUT_VARIABLE TENSORRT_LLM_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(TENSORRT_LLM_INCLUDE_DIR AND EXISTS "${TENSORRT_LLM_INCLUDE_DIR}")
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
    ncursesw
)

# TensorRT-LLM library from pip installation
if(ENABLE_TENSORRT)
    # Find MPI
    find_package(MPI REQUIRED)
    target_include_directories(shepherd PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(shepherd PRIVATE ${MPI_CXX_LIBRARIES})

    # Find TensorRT-LLM from current Python installation
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import tensorrt_llm, os; print(os.path.join(os.path.dirname(tensorrt_llm.__file__), 'libs'))"
        OUTPUT_VARIABLE TENSORRT_LLM_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # If not found, try to locate via site-packages
    if(NOT TENSORRT_LLM_LIB_DIR OR NOT EXISTS "${TENSORRT_LLM_LIB_DIR}")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE VENV_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(TENSORRT_LLM_LIB_DIR "${VENV_SITE_PACKAGES}/tensorrt_llm/libs")
        if(NOT EXISTS "${TENSORRT_LLM_LIB_DIR}")
            message(FATAL_ERROR "Could not find tensorrt_llm library directory at ${TENSORRT_LLM_LIB_DIR}")
        endif()
    endif()

    message(STATUS "TensorRT-LLM library directory: ${TENSORRT_LLM_LIB_DIR}")

    # Find additional required libraries (TensorRT, NCCL)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import tensorrt_libs; import os; print(os.path.dirname(tensorrt_libs.__file__))"
        OUTPUT_VARIABLE TENSORRT_LIBS_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "from nvidia import nccl; import os; print(os.path.join(os.path.dirname(nccl.__file__), 'lib'))"
        OUTPUT_VARIABLE NCCL_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Fallback to /usr/local/tensorrt, then user site-packages
    if(NOT TENSORRT_LIBS_DIR OR NOT EXISTS "${TENSORRT_LIBS_DIR}")
        set(TENSORRT_LIBS_DIR "/usr/local/tensorrt/lib")
    endif()
    if(NOT EXISTS "${TENSORRT_LIBS_DIR}")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE VENV_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(TENSORRT_LIBS_DIR "${VENV_SITE_PACKAGES}/tensorrt_libs")
    endif()
    if(NOT NCCL_LIB_DIR OR NOT EXISTS "${NCCL_LIB_DIR}")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE VENV_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(NCCL_LIB_DIR "${VENV_SITE_PACKAGES}/nvidia/nccl/lib")
    endif()

    if(TENSORRT_LIBS_DIR AND EXISTS "${TENSORRT_LIBS_DIR}")
        message(STATUS "TensorRT libs directory: ${TENSORRT_LIBS_DIR}")
        target_link_directories(shepherd PRIVATE ${TENSORRT_LIBS_DIR})
    endif()
    if(NCCL_LIB_DIR AND EXISTS "${NCCL_LIB_DIR}")
        message(STATUS "NCCL library directory: ${NCCL_LIB_DIR}")
        target_link_directories(shepherd PRIVATE ${NCCL_LIB_DIR})
    endif()

    # Find PyTorch libraries
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
        OUTPUT_VARIABLE TORCH_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    target_link_directories(shepherd PRIVATE ${TENSORRT_LLM_LIB_DIR} ${TENSORRT_LIBS_DIR} ${NCCL_LIB_DIR} ${TORCH_LIB_DIR})
    target_link_libraries(shepherd PRIVATE
        -Wl,--no-as-needed
        -ltensorrt_llm
        -lnvinfer_plugin_tensorrt_llm
        -lnvinfer
        -lnccl
        -ltorch_python
        -ltorch
        -ltorch_cpu
        Python3::Python
        -ltorch_cuda
        -lc10
        -lc10_cuda
        ${CMAKE_SOURCE_DIR}/tokenizers/rust/target/release/libtokenizers_c.a
        dl
        pthread
    )

    # Add RPATH so the executable can find the libraries at runtime
    # Using --disable-new-dtags to force RPATH instead of RUNPATH (for transitive deps)
    # Using --export-dynamic to make all symbols globally available (like RTLD_GLOBAL)
    set_target_properties(shepherd PROPERTIES
        BUILD_RPATH "${TENSORRT_LLM_LIB_DIR}:${TENSORRT_LIBS_DIR}:${NCCL_LIB_DIR}:${TORCH_LIB_DIR}"
        INSTALL_RPATH "${TENSORRT_LLM_LIB_DIR}:${TENSORRT_LIBS_DIR}:${NCCL_LIB_DIR}:${TORCH_LIB_DIR}"
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
        # Linux: Use CUDA backend if available
        if(EXISTS "${LLAMACPP_DIR}/build/bin/libggml-cuda.${SHARED_LIB_EXT}")
            target_link_libraries(shepherd PRIVATE ggml-cuda)
        endif()
    endif()

    # Set RPATH so the executable can find llama.cpp libraries
    set_target_properties(shepherd PROPERTIES
        BUILD_RPATH "${LLAMACPP_DIR}/build/bin"
        INSTALL_RPATH "${LLAMACPP_DIR}/build/bin"
    )
endif()

# replxx for line editing (used in non-TUI mode)
add_subdirectory(vendor/replxx)
target_link_libraries(shepherd PRIVATE replxx)
target_include_directories(shepherd PRIVATE vendor/replxx/include)
message(STATUS "Using replxx for line editing")

# Streaming parsers for output processing
# GenericParser: For non-harmony models (tool calls, thinking tags)
# HarmonyParser: O(n) parser for GPT-OSS channel format
# Legacy harmony.cpp: Still included for backward compatibility (will be removed)
if(ENABLE_LLAMACPP OR ENABLE_TENSORRT)
    target_sources(shepherd PRIVATE
        backends/generic_parser.cpp
        backends/harmony_parser.cpp
        backends/harmony.cpp
    )
    message(STATUS "Using streaming parsers (GenericParser, HarmonyParser)")
endif()

# Compiler options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Common warning flags
    target_compile_options(shepherd PRIVATE
        -Wall -Wextra
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-but-set-variable
        -Wno-ignored-attributes
        -Wno-unused-function
    )

    # GCC-specific warning flags
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(shepherd PRIVATE
            -Wno-maybe-uninitialized
        )
    endif()

    # Default to Debug build if not specified (matching Makefile behavior)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
        message(STATUS "No build type specified, defaulting to Debug")
    endif()

    # Apply build-type specific flags
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(shepherd PRIVATE -g -O0)
        target_compile_definitions(shepherd PRIVATE _DEBUG)
        message(STATUS "Debug build: -g -O0 -D_DEBUG")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(shepherd PRIVATE -O3 -march=native -pipe -flto)
        target_link_options(shepherd PRIVATE -flto)
        target_compile_definitions(shepherd PRIVATE NDEBUG)
        message(STATUS "Release build: -O3 -march=native -pipe -flto -DNDEBUG")
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        target_compile_options(shepherd PRIVATE -g -O2)
        target_compile_definitions(shepherd PRIVATE NDEBUG)
        message(STATUS "RelWithDebInfo build: -g -O2 -DNDEBUG")
    endif()
endif()

message(STATUS "Shepherd configuration:")
message(STATUS "  TensorRT backend: ${ENABLE_TENSORRT}")
message(STATUS "  llama.cpp backend: ${ENABLE_LLAMACPP}")
message(STATUS "  API backends: ${ENABLE_API_BACKENDS}")

# Install targets
install(TARGETS shepherd RUNTIME DESTINATION bin)

# Test suite (optional, excluded from default build)
# Build with: cmake -DBUILD_TESTS=ON ..
add_subdirectory(tests EXCLUDE_FROM_ALL)
