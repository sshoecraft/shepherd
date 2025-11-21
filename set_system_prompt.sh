#!/bin/bash
# Generate system_prompt.h from system_prompt.txt
{
    echo '#define SYSTEM_PROMPT R"('
    cat system_prompt.txt
    echo ')"'
} > system_prompt.h
