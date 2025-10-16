#include "rag_system.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>

RAGSystem::RAGSystem(MemoryManager* memory_manager)
    : memory_manager_(memory_manager)
    , similarity_threshold_(0.7)
    , initialized_(false) {

    std::cout << "Initializing RAG system..." << std::endl;
    initialized_ = true;
}

RAGSystem::~RAGSystem() {
    // Cleanup if needed
}

std::string RAGSystem::enhance_prompt(const std::string& query) {
    if (!initialized_) {
        return query; // Return original query if not initialized
    }

    // Retrieve relevant documents
    auto relevant_docs = retrieve_relevant(query);

    if (relevant_docs.empty()) {
        std::cout << "No relevant documents found for query" << std::endl;
        return query;
    }

    // Format enhanced prompt
    std::ostringstream enhanced;
    enhanced << "Context information:\n";
    enhanced << format_context(relevant_docs);
    enhanced << "\n\nBased on the above context, please answer the following question:\n";
    enhanced << query;

    std::cout << "Enhanced prompt with " << relevant_docs.size()
              << " relevant documents" << std::endl;

    return enhanced.str();
}

bool RAGSystem::add_document(const std::string& id, const std::string& content) {
    if (!initialized_) {
        std::cerr << "RAG system not initialized" << std::endl;
        return false;
    }

    try {
        Document doc;
        doc.id = id;
        doc.content = content;
        doc.embedding = generate_embedding(content);
        doc.score = 0.0;

        document_store_.push_back(std::move(doc));

        std::cout << "Added document: " << id << " (length: "
                  << content.length() << " chars)" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to add document " << id << ": " << e.what() << std::endl;
        return false;
    }
}

std::vector<Document> RAGSystem::retrieve_relevant(const std::string& query, int top_k) {
    if (document_store_.empty()) {
        return {};
    }

    // Generate query embedding
    auto query_embedding = generate_embedding(query);

    // Calculate similarities
    std::vector<Document> scored_docs;
    for (auto& doc : document_store_) {
        Document scored_doc = doc; // Copy
        scored_doc.score = calculate_similarity(query_embedding, doc.embedding);

        if (scored_doc.score >= similarity_threshold_) {
            scored_docs.push_back(scored_doc);
        }
    }

    // Sort by similarity score (descending)
    std::sort(scored_docs.begin(), scored_docs.end(),
              [](const Document& a, const Document& b) {
                  return a.score > b.score;
              });

    // Return top_k results
    if (scored_docs.size() > static_cast<size_t>(top_k)) {
        scored_docs.resize(top_k);
    }

    return scored_docs;
}

void RAGSystem::set_embedding_model(const std::string& model_path) {
    embedding_model_path_ = model_path;
    std::cout << "Set embedding model: " << model_path << std::endl;
}

void RAGSystem::set_similarity_threshold(double threshold) {
    similarity_threshold_ = threshold;
    std::cout << "Set similarity threshold: " << threshold << std::endl;
}

std::vector<float> RAGSystem::generate_embedding(const std::string& text) {
    // TODO: Implement actual embedding generation
    // For now, create a simple hash-based embedding for demonstration

    const size_t embedding_dim = 384; // Common embedding dimension
    std::vector<float> embedding(embedding_dim, 0.0f);

    // Simple hash-based embedding (replace with actual model)
    std::hash<std::string> hasher;
    size_t hash = hasher(text);

    for (size_t i = 0; i < embedding_dim; ++i) {
        embedding[i] = static_cast<float>((hash + i) % 1000) / 1000.0f - 0.5f;
    }

    // Normalize
    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 0.0f) {
        for (float& val : embedding) {
            val /= norm;
        }
    }

    return embedding;
}

double RAGSystem::calculate_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return 0.0;
    }

    // Cosine similarity
    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    if (norm_a == 0.0 || norm_b == 0.0) {
        return 0.0;
    }

    return dot_product / (norm_a * norm_b);
}

std::string RAGSystem::format_context(const std::vector<Document>& documents) {
    std::ostringstream context;

    for (size_t i = 0; i < documents.size(); ++i) {
        context << "Document " << (i + 1) << " (relevance: "
                << std::fixed << std::setprecision(3) << documents[i].score << "):\n";
        context << documents[i].content << "\n\n";
    }

    return context.str();
}