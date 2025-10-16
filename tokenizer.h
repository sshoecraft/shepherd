#pragma once

#include <string>
#include <vector>
#include <stdexcept>

/// @brief Abstract base class for tokenization
/// Each backend implements this with their specific tokenizer
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    /// @brief Count tokens in text
    /// @param text Text to tokenize
    /// @return Number of tokens
    virtual int count_tokens(const std::string& text) = 0;

    /// @brief Encode text to token IDs (optional, for backends that need it)
    /// @param text Text to encode
    /// @return Vector of token IDs
    virtual std::vector<int> encode(const std::string& text) = 0;

    /// @brief Decode token IDs back to text (optional, for backends that need it)
    /// @param tokens Vector of token IDs
    /// @return Decoded text
    virtual std::string decode(const std::vector<int>& tokens) = 0;

    /// @brief Get the tokenizer name/type
    /// @return Tokenizer identifier
    virtual std::string get_tokenizer_name() const = 0;
};

/// @brief Exception thrown by tokenizers
class TokenizerError : public std::runtime_error {
public:
    explicit TokenizerError(const std::string& message)
        : std::runtime_error("Tokenizer: " + message) {}
};