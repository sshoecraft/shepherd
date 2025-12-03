#!/bin/bash
# Generate system_prompt.h from system_prompt.txt
{
    echo '#ifndef SYSTEM_PROMPT_H'
    echo '#define SYSTEM_PROMPT_H'
    echo ''
    echo 'constexpr const char* SYSTEM_PROMPT = R"DELIM('
    cat system_prompt.txt
    echo ')DELIM";'
    echo ''
    echo '#endif'
} > system_prompt.h
