# Shepherd Test Suite CMake Configuration
cmake_minimum_required(VERSION 3.20)

option(BUILD_TESTS "Build test suite" OFF)

if(NOT BUILD_TESTS)
    return()
endif()

# Find Google Test
find_package(GTest REQUIRED)
include(GoogleTest)

# Find required packages (same as main project)
find_package(PkgConfig REQUIRED)
find_package(OpenSSL REQUIRED)
pkg_check_modules(SQLITE3 REQUIRED sqlite3)
pkg_check_modules(CURL REQUIRED libcurl)

# Source files needed for testing (subset of main project sources)
# We compile these directly to avoid circular dependencies
set(SHEPHERD_TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/config.cpp
    ${CMAKE_SOURCE_DIR}/session.cpp
    ${CMAKE_SOURCE_DIR}/scheduler.cpp
    ${CMAKE_SOURCE_DIR}/provider.cpp
    ${CMAKE_SOURCE_DIR}/sse_parser.cpp
    ${CMAKE_SOURCE_DIR}/backend.cpp
    ${CMAKE_SOURCE_DIR}/auth.cpp
    ${CMAKE_SOURCE_DIR}/rag.cpp
    ${CMAKE_SOURCE_DIR}/backends/models.cpp
    ${CMAKE_SOURCE_DIR}/backends/factory.cpp
    ${CMAKE_SOURCE_DIR}/backends/cli_client.cpp
    ${CMAKE_SOURCE_DIR}/backends/harmony.cpp
    ${CMAKE_SOURCE_DIR}/tools/tool.cpp
    ${CMAKE_SOURCE_DIR}/tools/tools.cpp
    ${CMAKE_SOURCE_DIR}/tools/tool_parser.cpp
    ${CMAKE_SOURCE_DIR}/tools/utf8_sanitizer.cpp
    ${CMAKE_SOURCE_DIR}/tools/api_tools.cpp
)

# Add API backend sources if enabled
if(ENABLE_API_BACKENDS)
    list(APPEND SHEPHERD_TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/http_client.cpp
        ${CMAKE_SOURCE_DIR}/backends/api.cpp
        ${CMAKE_SOURCE_DIR}/backends/openai.cpp
        ${CMAKE_SOURCE_DIR}/backends/anthropic.cpp
        ${CMAKE_SOURCE_DIR}/backends/gemini.cpp
        ${CMAKE_SOURCE_DIR}/backends/ollama.cpp
    )
endif()

# Add GPU backend base class if either GPU backend is enabled
if(ENABLE_LLAMACPP OR ENABLE_TENSORRT)
    list(APPEND SHEPHERD_TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/backends/gpu.cpp
        ${CMAKE_SOURCE_DIR}/backends/chat_template.cpp
    )
endif()

# Add llama.cpp sources if enabled
if(ENABLE_LLAMACPP)
    list(APPEND SHEPHERD_TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/backends/llamacpp.cpp
    )
endif()

# Create static library from shepherd sources for test linking
add_library(shepherd_testlib STATIC ${SHEPHERD_TEST_SOURCES})

target_include_directories(shepherd_testlib PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/backends
    ${CMAKE_SOURCE_DIR}/tools
    ${CMAKE_SOURCE_DIR}/frontends
    ${SQLITE3_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

target_link_libraries(shepherd_testlib PUBLIC
    ${SQLITE3_LIBRARIES}
    ${CURL_LIBRARIES}
    OpenSSL::Crypto
    pthread
)

# Compile definitions (match main project)
if(ENABLE_API_BACKENDS)
    target_compile_definitions(shepherd_testlib PUBLIC ENABLE_API_BACKENDS)
endif()

if(ENABLE_LLAMACPP)
    set(LLAMACPP_DIR ${CMAKE_SOURCE_DIR}/llama.cpp)
    target_compile_definitions(shepherd_testlib PUBLIC ENABLE_LLAMACPP)
    target_include_directories(shepherd_testlib PUBLIC
        ${CMAKE_SOURCE_DIR}/llama.cpp/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/common
    )
    target_link_libraries(shepherd_testlib PUBLIC
        ${LLAMACPP_DIR}/build/common/libcommon.a
        llama
        ggml
        ggml-base
        ggml-cpu
    )
    # Set RPATH for llama.cpp libraries
    set_target_properties(shepherd_testlib PROPERTIES
        BUILD_RPATH "${LLAMACPP_DIR}/build/bin"
    )
endif()

# Suppress warnings in test library (inherited from main project)
target_compile_options(shepherd_testlib PRIVATE
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-unused-function
)

# ============================================================================
# Unit Tests (includes provider tests - providers are core functionality)
# ============================================================================

add_executable(test_unit
    test_main.cpp
    test_stubs.cpp
    unit/test_config.cpp
    unit/test_session.cpp
    unit/test_scheduler.cpp
    unit/test_sse_parser.cpp
    unit/test_message.cpp
    unit/test_providers.cpp
    unit/test_auth.cpp
    unit/test_models.cpp
    unit/test_chat_template.cpp
    unit/test_harmony.cpp
)

target_include_directories(test_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
)

target_link_libraries(test_unit PRIVATE
    shepherd_testlib
    GTest::gtest
    GTest::gtest_main
)

# ============================================================================
# Tool Tests
# ============================================================================

# Tool tests require additional sources
# Note: core_tools.cpp and web_search.cpp have complex dependencies on global config
# They are excluded from unit tests - integration tests will cover them
set(TOOL_TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/tools/filesystem_tools.cpp
    ${CMAKE_SOURCE_DIR}/tools/command_tools.cpp
    ${CMAKE_SOURCE_DIR}/tools/json_tools.cpp
    ${CMAKE_SOURCE_DIR}/tools/http_tools.cpp
)

add_executable(test_tools
    test_main.cpp
    tools/test_tools_manager.cpp
    ${TOOL_TEST_SOURCES}
)

target_include_directories(test_tools PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
)

target_link_libraries(test_tools PRIVATE
    shepherd_testlib
    GTest::gtest
    GTest::gtest_main
)

# ============================================================================
# CTest Integration
# ============================================================================

enable_testing()
gtest_discover_tests(test_unit)
gtest_discover_tests(test_tools)
