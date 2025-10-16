#!/bin/bash
# Script to apply shepherd patches to llama.cpp

PATCHES_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$PATCHES_DIR/../llama.cpp"

cd "$LLAMA_DIR"

if [ -f ".shepherd-patches-applied" ]; then
    echo "Patches already applied (marker file exists)"
    echo "To reapply: cd llama.cpp && git checkout . && git clean -fd && rm .shepherd-patches-applied"
    exit 0
fi

echo "Applying Shepherd patches to llama.cpp..."
echo ""

echo "[1/1] Applying llama.cpp patches (KV cache callbacks)..."
patch -p1 < "$PATCHES_DIR/llama.patch"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to apply llama.patch"
    exit 1
fi
echo "✓ llama.cpp patches applied (includes KV space & tokens callbacks)"
echo ""

touch .shepherd-patches-applied
echo "============================================"
echo "All Shepherd patches applied successfully!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Rebuild llama.cpp: cd llama.cpp/build && cmake .. && make -j8"
echo "  2. Rebuild shepherd: cd build && cmake .. && make"
echo ""
