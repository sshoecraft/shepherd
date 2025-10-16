#!/bin/bash
# Simple eviction test - guaranteed to trigger eviction

echo "=== Context Manager Eviction Test (Aggressive) ==="
echo "Context size: 1024 tokens (very small)"
echo "Sending long messages to force eviction..."
echo ""

./build/shepherd --backend llamacpp \
    --model ~/models/qwen2.5-3b-instruct-q5_k_m.gguf \
    --context-size 1024 \
    --debug << 'EOF'
Please write a detailed explanation of what a programming language is. Include information about syntax, semantics, compilers, interpreters, and provide examples of different types of programming languages like compiled languages, interpreted languages, and scripting languages. Make your response comprehensive and educational.
status
Now explain in detail what object-oriented programming is. Discuss classes, objects, inheritance, polymorphism, encapsulation, and abstraction. Provide code examples and explain how OOP differs from procedural programming. Be thorough and include best practices.
status
Describe the concept of algorithms and data structures. Explain what Big-O notation is, discuss common data structures like arrays, linked lists, trees, and graphs. Explain sorting algorithms like bubble sort, merge sort, and quick sort. Include complexity analysis.
status
exit
EOF

echo ""
echo "=== Test Complete ==="
echo "Check the debug output above for eviction messages:"
echo "  - Look for '[DEBUG] Context full, evicting old messages...'"
echo "  - Look for '[INFO] Pass 1 freed X tokens (needed Y)'"
echo "  - Look for '[INFO] Evicting messages [X, Y]'"
echo "  - Check 'status' output for context utilization"
