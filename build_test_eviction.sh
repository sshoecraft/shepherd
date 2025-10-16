#!/bin/bash
# Build script for eviction test

set -e  # Exit on error

echo "================================"
echo "Building Eviction Test"
echo "================================"

# Source files
TEST_SRC="test_eviction.cpp"
TEST_BIN="test_eviction"

# Dependencies
DEPS="context_manager.cpp backends/api_backend.cpp logger.cpp rag.cpp backend_manager.cpp tools/tool.cpp config.cpp rag_system.cpp"

# Compiler flags
CXX=${CXX:-g++}
CXXFLAGS="-std=c++17 -Wall -Wextra -g -O0"
INCLUDES="-I. -I./llama.cpp/include"

# Libraries
LIBS="-lpthread -lsqlite3 -lcurl"

# Build command
echo "Compiling test..."
$CXX $CXXFLAGS $INCLUDES -o $TEST_BIN $TEST_SRC $DEPS $LIBS

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Run the test with: ./$TEST_BIN"
    echo ""

    # Optionally run the test
    if [ "$1" == "run" ]; then
        echo "================================"
        echo "Running Test"
        echo "================================"
        ./$TEST_BIN
    fi
else
    echo "✗ Build failed!"
    exit 1
fi
