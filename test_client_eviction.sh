#!/bin/bash
#
# Shepherd CLIENT Eviction Test Script
#
# Tests the actual ./shepherd binary's client-side eviction and RAG behavior
# when connecting to a Shepherd server.
#

set -e

# Configuration
SHEPHERD="./shepherd"
SERVER="http://192.168.1.166:8080/v1"
MODEL="gpt-4"
TEST_DIR="/tmp/shepherd_eviction_test_$$"
LOG_FILE="${TEST_DIR}/shepherd.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# ============================================================================
# Helper Functions
# ============================================================================

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

test_header() {
    echo ""
    echo "======================================================================"
    echo "TEST $1: $2"
    echo "======================================================================"
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

test_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

test_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

# ============================================================================
# Setup
# ============================================================================

setup() {
    info "Setting up test environment"

    # Create test directory
    mkdir -p "$TEST_DIR"

    # Check if shepherd binary exists
    if [ ! -x "$SHEPHERD" ]; then
        error "Shepherd binary not found or not executable: $SHEPHERD"
        exit 1
    fi

    info "Test directory: $TEST_DIR"
}

cleanup() {
    info "Cleaning up test environment"
    rm -rf "$TEST_DIR"
}

# ============================================================================
# RAG Database Inspection
# ============================================================================

count_rag_conversations() {
    local db="$1"
    if [ ! -f "$db" ]; then
        echo "0"
        return
    fi
    sqlite3 "$db" "SELECT COUNT(*) FROM conversations;" 2>/dev/null || echo "0"
}

search_rag() {
    local db="$1"
    local query="$2"
    sqlite3 "$db" "SELECT user_message FROM conversations WHERE user_message LIKE '%${query}%' LIMIT 5;" 2>/dev/null || true
}

# ============================================================================
# Test Functions
# ============================================================================

test_1_basic_eviction() {
    test_header "1.1" "Basic Eviction Trigger"

    local context_size=2048
    local rag_db="${TEST_DIR}/test_1_1.db"
    local log="${TEST_DIR}/test_1_1.log"

    info "Context size: $context_size tokens"
    info "RAG database: $rag_db"

    # Create input file with messages that will fill context
    local input_file="${TEST_DIR}/test_1_1_input.txt"

    # Generate enough messages to fill context (assume ~100 tokens per message)
    for i in {1..30}; do
        echo "This is test message number $i. $(head -c 400 /dev/zero | tr '\0' 'X')"
    done > "$input_file"

    # Run shepherd with input
    info "Running shepherd..."
    timeout 60s "$SHEPHERD" \
        --backend openai \
        --api-base "$SERVER" \
        --model "$MODEL" \
        --context-size "$context_size" \
        --rag-db "$rag_db" \
        < "$input_file" > "$log" 2>&1 || true

    # Check logs for eviction
    local eviction_count=$(grep -c "Evicting messages" "$log" 2>/dev/null || echo "0")

    # Check RAG database
    local rag_count=$(count_rag_conversations "$rag_db")

    info "Evictions in logs: $eviction_count"
    info "Conversations in RAG: $rag_count"

    # Verify
    if [ "$eviction_count" -gt 0 ] || [ "$rag_count" -gt 0 ]; then
        test_pass "Eviction detected (logs: $eviction_count, RAG: $rag_count)"
    else
        test_fail "No eviction occurred when expected"
        warn "Log excerpt:"
        tail -20 "$log"
    fi
}

test_2_rag_archival() {
    test_header "2.1" "RAG Archival Verification"

    local context_size=2048
    local rag_db="${TEST_DIR}/test_2_1.db"
    local log="${TEST_DIR}/test_2_1.log"

    # Create input with distinctive messages
    local input_file="${TEST_DIR}/test_2_1_input.txt"

    # Send distinctive messages
    {
    echo "DISTINCTIVE_MESSAGE_1: The capital of France is Paris"
    echo "DISTINCTIVE_MESSAGE_2: Python is a programming language"
    echo "DISTINCTIVE_MESSAGE_3: Machine learning uses neural networks"

    # Fill with generic messages to force eviction
    for i in {1..40}; do
        echo "Generic filler message number $i $(head -c 400 /dev/zero | tr '\0' 'X')"
    done

    } > "$input_file"

    # Run shepherd
    info "Running shepherd..."
    timeout 90s "$SHEPHERD" \
        --backend openai \
        --api-base "$SERVER" \
        --model "$MODEL" \
        --context-size "$context_size" \
        --rag-db "$rag_db" \
        < "$input_file" > "$log" 2>&1 || true

    # Check if distinctive messages made it to RAG
    local found_count=0

    if search_rag "$rag_db" "DISTINCTIVE_MESSAGE_1" | grep -q "France"; then
        info "Found DISTINCTIVE_MESSAGE_1 in RAG"
        found_count=$((found_count + 1))
    fi

    if search_rag "$rag_db" "DISTINCTIVE_MESSAGE_2" | grep -q "Python"; then
        info "Found DISTINCTIVE_MESSAGE_2 in RAG"
        found_count=$((found_count + 1))
    fi

    if search_rag "$rag_db" "DISTINCTIVE_MESSAGE_3" | grep -q "neural"; then
        info "Found DISTINCTIVE_MESSAGE_3 in RAG"
        found_count=$((found_count + 1))
    fi

    local total_rag=$(count_rag_conversations "$rag_db")
    info "Total conversations in RAG: $total_rag"
    info "Distinctive messages found: $found_count/3"

    # Verify
    if [ "$found_count" -gt 0 ]; then
        test_pass "Evicted messages archived to RAG ($found_count/3 found)"
    else
        test_fail "Distinctive messages not found in RAG"
        if [ "$total_rag" -gt 0 ]; then
            warn "RAG has $total_rag conversations but distinctive messages missing"
        fi
    fi
}

test_3_small_context() {
    test_header "3.1" "Small Context (512 tokens) - Frequent Eviction"

    local context_size=512
    local rag_db="${TEST_DIR}/test_3_1.db"
    local log="${TEST_DIR}/test_3_1.log"

    info "Context size: $context_size tokens (very small)"

    # Create input
    local input_file="${TEST_DIR}/test_3_1_input.txt"

    # Send many small messages
    for i in {1..50}; do
        echo "Message $i $(head -c 200 /dev/zero | tr '\0' 'X')"
    done > "$input_file"

    # Run shepherd
    info "Running shepherd..."
    timeout 90s "$SHEPHERD" \
        --backend openai \
        --api-base "$SERVER" \
        --model "$MODEL" \
        --context-size "$context_size" \
        --rag-db "$rag_db" \
        < "$input_file" > "$log" 2>&1 || true

    # Check results
    local eviction_count=$(grep -c "Evicting messages" "$log" 2>/dev/null || echo "0")
    local rag_count=$(count_rag_conversations "$rag_db")

    info "Evictions in logs: $eviction_count"
    info "Conversations in RAG: $rag_count"

    # With 512 token context, we should see MANY evictions
    if [ "$eviction_count" -gt 10 ] || [ "$rag_count" -gt 10 ]; then
        test_pass "Frequent eviction with small context (logs: $eviction_count, RAG: $rag_count)"
    else
        test_fail "Expected frequent evictions with 512-token context"
    fi
}

test_4_server_never_sees_overflow() {
    test_header "4.1" "Server Never Sees Overflow (Client Manages Context)"

    local context_size=2048
    local rag_db="${TEST_DIR}/test_4_1.db"
    local log="${TEST_DIR}/test_4_1.log"

    info "Client context: $context_size tokens"
    info "Server context: ~98K tokens"
    info "Client should evict before server sees overflow"

    # Create input
    local input_file="${TEST_DIR}/test_4_1_input.txt"

    # Send many messages
    for i in {1..50}; do
        echo "Overflow test message $i $(head -c 400 /dev/zero | tr '\0' 'X')"
    done > "$input_file"

    # Run shepherd
    info "Running shepherd..."
    timeout 90s "$SHEPHERD" \
        --backend openai \
        --api-base "$SERVER" \
        --model "$MODEL" \
        --context-size "$context_size" \
        --rag-db "$rag_db" \
        < "$input_file" > "$log" 2>&1 || true

    # Check for server errors
    local server_errors=$(grep -c "400\|Context limit exceeded in server mode" "$log" 2>/dev/null || echo "0")
    local client_evictions=$(grep -c "Evicting messages" "$log" 2>/dev/null || echo "0")

    info "Server errors: $server_errors"
    info "Client evictions: $client_evictions"

    # Verify: server should NOT see errors, client should evict
    if [ "$server_errors" -eq 0 ] && [ "$client_evictions" -gt 0 ]; then
        test_pass "Client managed context, server saw no errors"
    elif [ "$server_errors" -gt 0 ]; then
        test_fail "Server saw $server_errors errors - client failed to manage context"
    else
        test_fail "No client evictions detected"
    fi
}

# ============================================================================
# Main Test Runner
# ============================================================================

main() {
    echo "======================================================================"
    echo "Shepherd CLIENT Eviction Test Suite (Bash)"
    echo "======================================================================"
    echo "Server: $SERVER"
    echo "Model: $MODEL"
    echo "======================================================================"
    echo ""

    setup

    # Run tests
    test_1_basic_eviction
    test_2_rag_archival
    test_3_small_context
    test_4_server_never_sees_overflow

    # Summary
    echo ""
    echo "======================================================================"
    echo "SUMMARY"
    echo "======================================================================"
    echo "Total Tests: $TESTS_TOTAL"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    echo "======================================================================"
    echo ""

    cleanup

    if [ "$TESTS_FAILED" -gt 0 ]; then
        exit 1
    fi
    exit 0
}

# Run tests
main "$@"
