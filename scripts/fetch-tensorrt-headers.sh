#!/bin/bash
#
# Download TensorRT-LLM C++ headers from GitHub
# Usage: ./scripts/fetch-tensorrt-headers.sh <version>
# Example: ./scripts/fetch-tensorrt-headers.sh 1.1.0
#

set -e

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.1.0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEST_DIR="$PROJECT_DIR/include/tensorrt-llm/$VERSION"

if [ -d "$DEST_DIR/tensorrt_llm/executor" ]; then
    echo "Headers already exist at $DEST_DIR"
    exit 0
fi

echo "Downloading TensorRT-LLM $VERSION headers..."

# Create temp directory for sparse checkout
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

cd "$TEMP_DIR"

# Initialize sparse checkout
git init -q
git remote add origin https://github.com/NVIDIA/TensorRT-LLM.git
git config core.sparseCheckout true

# Only checkout the C++ include directory
echo "cpp/include/tensorrt_llm/" > .git/info/sparse-checkout

# Fetch only the tag we need with minimal depth
echo "Fetching headers (sparse checkout)..."
git fetch -q --depth=1 origin "refs/tags/v$VERSION"
git checkout -q FETCH_HEAD

# Copy headers to destination
mkdir -p "$DEST_DIR"
cp -r cpp/include/tensorrt_llm "$DEST_DIR/"

echo "Headers installed to $DEST_DIR"
echo "Done."
