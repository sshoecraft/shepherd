#!/bin/bash
# Eviction Test Script - Tests context manager eviction with small context window

echo "========================================="
echo "Context Manager Eviction Test"
echo "========================================="
echo ""

# Test 1: Multiple conversation turns that trigger eviction
test_multiple_turns() {
    echo "TEST 1: Multiple Conversation Turns (should trigger eviction)"
    echo "-------------------------------------------------------------"

    ./build/shepherd --context-size 2048 --debug << 'EOF'
Hello, can you tell me what 2 + 2 equals?
What is 5 + 5?
What is 10 + 10?
What is 20 + 20?
What is 50 + 50?
What is 100 + 100?
What is 200 + 200?
What is 500 + 500?
status
exit
EOF

    echo ""
    echo "Test 1 complete."
    echo ""
}

# Test 2: Tool calls that create mini-turns
test_tool_calls() {
    echo "TEST 2: Tool Calls Creating Mini-Turns (should trigger eviction)"
    echo "----------------------------------------------------------------"

    ./build/shepherd --context-size 2048 --debug << 'EOF'
Use the list directory tool to see what files are in the current directory.
Now use read file to read the context_manager.h file.
Can you search for the word "evict" in the context_manager.cpp file?
What is the purpose of the add_message function?
status
exit
EOF

    echo ""
    echo "Test 2 complete."
    echo ""
}

# Test 3: Long messages that fill context quickly
test_long_messages() {
    echo "TEST 3: Long Messages (should trigger aggressive eviction)"
    echo "-----------------------------------------------------------"

    ./build/shepherd --context-size 1536 --debug << 'EOF'
Please explain in detail what a context manager does in a language model system. Include information about token counting, eviction strategies, and how it manages conversation history. Be thorough and detailed in your explanation.
Now explain what the purpose of a RAG system is and how it integrates with context management. Describe the archival process and retrieval mechanisms.
Can you provide a comprehensive overview of how tool calling works in language models, including the request-response cycle, parameter passing, and result handling?
status
exit
EOF

    echo ""
    echo "Test 3 complete."
    echo ""
}

# Test 4: Edge case - very small context
test_tiny_context() {
    echo "TEST 4: Very Small Context (1024 tokens - edge case)"
    echo "-----------------------------------------------------"

    ./build/shepherd --context-size 1024 --debug << 'EOF'
Hello!
How are you?
What is your name?
What can you do?
Tell me about yourself in detail.
status
exit
EOF

    echo ""
    echo "Test 4 complete."
    echo ""
}

# Test 5: Gradual fill - watch utilization increase
test_gradual_fill() {
    echo "TEST 5: Gradual Context Fill (watch utilization grow)"
    echo "------------------------------------------------------"

    ./build/shepherd --context-size 2048 --debug << 'EOF'
Hi there!
status
Can you count to 10?
status
Tell me about programming languages.
status
What is Python used for?
status
Explain object-oriented programming.
status
What are design patterns?
status
exit
EOF

    echo ""
    echo "Test 5 complete."
    echo ""
}

# Test 6: Many tool calls in single turn - forces Pass 2 eviction
test_many_tools_single_turn() {
    echo "TEST 6: Many Tool Calls in Single Turn (forces Pass 2 mini-turn eviction)"
    echo "--------------------------------------------------------------------------"

    ./build/shepherd --context-size 1536 --debug << 'EOF'
Read every file in this directory and tell me what this project does in a single paragraph.
exit
EOF

    echo ""
    echo "Test 6 complete."
    echo ""
}

# Run tests based on argument, or all if no argument
case "${1:-all}" in
    1|turns)
        test_multiple_turns
        ;;
    2|tools)
        test_tool_calls
        ;;
    3|long)
        test_long_messages
        ;;
    4|tiny)
        test_tiny_context
        ;;
    5|gradual)
        test_gradual_fill
        ;;
    6|single)
        test_many_tools_single_turn
        ;;
    all)
        test_multiple_turns
        echo "========================================="
        echo ""
        test_tool_calls
        echo "========================================="
        echo ""
        test_long_messages
        echo "========================================="
        echo ""
        test_tiny_context
        echo "========================================="
        echo ""
        test_gradual_fill
        echo "========================================="
        echo ""
        test_many_tools_single_turn
        ;;
    *)
        echo "Usage: $0 [test_number|all]"
        echo ""
        echo "Available tests:"
        echo "  1 | turns    - Multiple conversation turns"
        echo "  2 | tools    - Tool calls creating mini-turns"
        echo "  3 | long     - Long messages requiring eviction"
        echo "  4 | tiny     - Very small context (1024 tokens)"
        echo "  5 | gradual  - Gradual fill with status checks"
        echo "  6 | single   - Many tool calls in single turn (Pass 2)"
        echo "  all          - Run all tests (default)"
        echo ""
        echo "Examples:"
        echo "  $0           # Run all tests"
        echo "  $0 1         # Run test 1 only"
        echo "  $0 tools     # Run test 2 only"
        echo "  $0 6         # Run test 6 (reproduces Pass 2 failure)"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Eviction Testing Complete!"
echo "========================================="
echo ""
echo "Look for these log messages in the output:"
echo "  [DEBUG] Context full, evicting old messages..."
echo "  [INFO] Pass 1 freed X tokens (needed Y)"
echo "  [INFO] Evicting messages [X, Y]"
echo "  [INFO] Successfully evicted N messages"
echo ""
echo "Check context utilization with 'status' commands."
echo "Expected: Context should stay under 100% utilization."
