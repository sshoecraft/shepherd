#pragma once

#include <string>
#include <vector>
#include <memory>
#include "memory_manager.h"

struct Document {
    std::string id;
    std::string content;
    std::vector<float> embedding;
    double score;
};

class RAGSystem {
public:
    explicit RAGSystem(MemoryManager* memory_manager);
    ~RAGSystem();

    // Main RAG functionality
    std::string enhance_prompt(const std::string& query);
    bool add_document(const std::string& id, const std::string& content);
    std::vector<Document> retrieve_relevant(const std::string& query, int top_k = 5);

    // Configuration
    void set_embedding_model(const std::string& model_path);
    void set_similarity_threshold(double threshold);

private:
    // Core methods
    std::vector<float> generate_embedding(const std::string& text);
    double calculate_similarity(const std::vector<float>& a, const std::vector<float>& b);
    std::string format_context(const std::vector<Document>& documents);

    // Members
    MemoryManager* memory_manager_;
    std::vector<Document> document_store_;
    std::string embedding_model_path_;
    double similarity_threshold_;
    bool initialized_;
};