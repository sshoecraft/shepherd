#!/bin/bash
#
# Shepherd Integration Test Runner
#
# Usage:
#   ./run_integration_tests.sh [options]
#
# Options:
#   --provider NAME    Provider to use for tests (required for server tests)
#   --model NAME       Model name (optional)
#   --port PORT        Port for API server (default: 18080)
#   --skip-server      Skip server-based tests (API, CLI server)
#   --only-unit        Run only unit tests
#   --only-cli         Run only CLI command tests
#   --help             Show this help
#
# Environment variables:
#   PROVIDER           Same as --provider
#   MODEL              Same as --model
#   SHEPHERD_BINARY    Path to shepherd binary (default: ./build/shepherd)
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
TESTS_DIR="$PROJECT_ROOT/tests"

# Defaults
PORT="${PORT:-18080}"
SHEPHERD_BINARY="${SHEPHERD_BINARY:-$BUILD_DIR/shepherd}"
PROVIDER="${PROVIDER:-}"
MODEL="${MODEL:-}"
SKIP_SERVER=false
ONLY_UNIT=false
ONLY_CLI=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --skip-server)
            SKIP_SERVER=true
            shift
            ;;
        --only-unit)
            ONLY_UNIT=true
            shift
            ;;
        --only-cli)
            ONLY_CLI=true
            shift
            ;;
        --help)
            head -30 "$0" | tail -28
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Export environment
export PROVIDER
export MODEL
export PORT
export SHEPHERD_BINARY

echo "=============================================="
echo "Shepherd Integration Test Suite"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Binary: $SHEPHERD_BINARY"
echo "Provider: ${PROVIDER:-<not set>}"
echo "Model: ${MODEL:-<default>}"
echo "Port: $PORT"
echo ""

# Check binary exists
if [[ ! -x "$SHEPHERD_BINARY" ]]; then
    echo -e "${RED}ERROR: Shepherd binary not found or not executable: $SHEPHERD_BINARY${NC}"
    echo "Run 'make' to build the project first."
    exit 1
fi

# Track results
TOTAL_PASSED=0
TOTAL_FAILED=0
declare -a FAILED_TESTS

run_test() {
    local name="$1"
    local cmd="$2"

    echo ""
    echo -e "${YELLOW}Running: $name${NC}"
    echo "Command: $cmd"
    echo "---"

    if eval "$cmd"; then
        echo -e "${GREEN}PASSED: $name${NC}"
        ((TOTAL_PASSED++))
    else
        echo -e "${RED}FAILED: $name${NC}"
        ((TOTAL_FAILED++))
        FAILED_TESTS+=("$name")
    fi
}

# ============================================================
# Unit Tests (C++)
# ============================================================
if [[ "$ONLY_CLI" != "true" ]]; then
    echo ""
    echo "=============================================="
    echo "Unit Tests (C++)"
    echo "=============================================="

    if [[ -x "$BUILD_DIR/tests/test_unit" ]]; then
        run_test "Unit Tests (test_unit)" "$BUILD_DIR/tests/test_unit"
    else
        echo -e "${YELLOW}Skipping: test_unit not found${NC}"
    fi

    if [[ -x "$BUILD_DIR/tests/test_tools" ]]; then
        run_test "Tool Tests (test_tools)" "$BUILD_DIR/tests/test_tools"
    else
        echo -e "${YELLOW}Skipping: test_tools not found${NC}"
    fi
fi

if [[ "$ONLY_UNIT" == "true" ]]; then
    echo ""
    echo "=============================================="
    echo "Skipping integration tests (--only-unit)"
    echo "=============================================="
else
    # ============================================================
    # CLI Command Tests (no server needed)
    # ============================================================
    echo ""
    echo "=============================================="
    echo "CLI Command Tests"
    echo "=============================================="

    if [[ -f "$TESTS_DIR/integration/test_cli_commands.py" ]]; then
        run_test "CLI Commands" "python3 $TESTS_DIR/integration/test_cli_commands.py"
    else
        echo -e "${YELLOW}Skipping: test_cli_commands.py not found${NC}"
    fi

    # ============================================================
    # Server-based Tests
    # ============================================================
    if [[ "$SKIP_SERVER" == "true" ]]; then
        echo ""
        echo "=============================================="
        echo "Skipping server tests (--skip-server)"
        echo "=============================================="
    elif [[ -z "$PROVIDER" ]]; then
        echo ""
        echo -e "${YELLOW}=============================================="
        echo "Skipping server tests (PROVIDER not set)"
        echo "Set PROVIDER=<name> to run server tests"
        echo "===============================================${NC}"
    else
        echo ""
        echo "=============================================="
        echo "API Server Tests"
        echo "=============================================="

        if [[ -f "$TESTS_DIR/integration/test_api_server.py" ]]; then
            run_test "API Server" "python3 $TESTS_DIR/integration/test_api_server.py"
        else
            echo -e "${YELLOW}Skipping: test_api_server.py not found${NC}"
        fi

        echo ""
        echo "=============================================="
        echo "CLI Server Tests"
        echo "=============================================="

        if [[ -f "$TESTS_DIR/integration/test_cli_server.py" ]]; then
            run_test "CLI Server" "python3 $TESTS_DIR/integration/test_cli_server.py"
        else
            echo -e "${YELLOW}Skipping: test_cli_server.py not found${NC}"
        fi
    fi
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "=============================================="
echo "Test Summary"
echo "=============================================="
echo -e "Passed: ${GREEN}$TOTAL_PASSED${NC}"
echo -e "Failed: ${RED}$TOTAL_FAILED${NC}"

if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
fi

echo ""
if [[ $TOTAL_FAILED -eq 0 ]]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
