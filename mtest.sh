#!/bin/bash
one() {
#./build/shepherd --backend llamacpp --model /home/steve/models/WizardCoder-15B-V1.0.Q4_K_M.gguf --template /home/steve/models/WizardCoder-15B-V1.0.Q4_K_M.jinja --debug << EOF
#./build/shepherd --backend llamacpp --model /home/steve/models/llama-3.1-70b-instruct-Q4_K_M.gguf --context-size 16384 --debug << EOF
./build/shepherd --backend llamacpp --model /home/steve/models/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf --context-size 32768 --debug << EOF
use the list directory tool to get a list of all the files in this directory, in the current directory, and then use read file to read each of the .cpp files that you got from this directory and analyze what they do and give me a summary of what each one does and what the overall project does
exit
EOF
exit 0
}
one

./build/shepherd --backend llamacpp --model ~/models/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --template ~/models/Qwen3-Coder-30B-A3B-Instruct-config/chat_template.jinja --context-size 16384 --debug << EOF
read all of the files in the current directory and do a full analysis on what this project does and how it could be improved
status
status
exit
EOF
