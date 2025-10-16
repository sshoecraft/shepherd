#!/usr/bin/env python3
"""
Helper script for tokenizing text using HuggingFace transformers.
Used by TensorRT backend for tokenization.
"""
import sys
import json
from transformers import AutoTokenizer

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: tokenize_helper.py <model_path> <encode|decode> [text|tokens]"}))
        sys.exit(1)

    model_path = sys.argv[1]
    operation = sys.argv[2]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

        if operation == "encode":
            text = sys.stdin.read() if len(sys.argv) == 3 else " ".join(sys.argv[3:])
            tokens = tokenizer.encode(text, add_special_tokens=True)
            print(json.dumps({"tokens": tokens}))

        elif operation == "decode":
            tokens_json = sys.stdin.read() if len(sys.argv) == 3 else sys.argv[3]
            tokens = json.loads(tokens_json)
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            print(json.dumps({"text": text}))

        else:
            print(json.dumps({"error": f"Unknown operation: {operation}"}))
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
