#include "web_search.h"
#include "http_client.h"
#include "nlohmann/json.hpp"
#include "logger.h"
#include "shepherd.h"
#include <sstream>

using json = nlohmann::json;

// BraveSearchProvider implementation
BraveSearchProvider::BraveSearchProvider(const std::string& api_key)
    : api_key_(api_key) {
    LOG_INFO("Initialized Brave Search provider");
}

std::vector<WebSearchResult> BraveSearchProvider::search(const std::string& query) {
    std::vector<WebSearchResult> results;

    try {
        HttpClient client;

        // URL encode the query
        std::string encoded_query;
        for (char c : query) {
            if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                encoded_query += c;
            } else if (c == ' ') {
                encoded_query += '+';
            } else {
                char hex[4];
                snprintf(hex, sizeof(hex), "%%%02X", (unsigned char)c);
                encoded_query += hex;
            }
        }

        std::string url = "https://api.search.brave.com/res/v1/web/search?q=" + encoded_query;

        std::map<std::string, std::string> headers;
        headers["Accept"] = "application/json";
        headers["X-Subscription-Token"] = api_key_;

        HttpResponse response = client.get(url, headers);

        if (!response.is_success()) {
            LOG_ERROR("Brave Search API request failed: " + response.error_message);
            return results;
        }

        auto json_response = json::parse(response.body);

        if (json_response.contains("web") && json_response["web"].contains("results")) {
            for (const auto& item : json_response["web"]["results"]) {
                WebSearchResult result;
                result.title = item.value("title", "");
                result.url = item.value("url", "");
                result.description = item.value("description", "");
                results.push_back(result);

                if (results.size() >= 10) break; // Limit to 10 results
            }
        }

    } catch (const std::exception& e) {
        LOG_ERROR("Brave Search error: " + std::string(e.what()));
    }

    return results;
}

// DuckDuckGoProvider implementation
std::vector<WebSearchResult> DuckDuckGoProvider::search(const std::string& query) {
    std::vector<WebSearchResult> results;

    try {
        HttpClient client;

        // URL encode the query
        std::string encoded_query;
        for (char c : query) {
            if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                encoded_query += c;
            } else if (c == ' ') {
                encoded_query += '+';
            } else {
                char hex[4];
                snprintf(hex, sizeof(hex), "%%%02X", (unsigned char)c);
                encoded_query += hex;
            }
        }

        // Use DuckDuckGo's Instant Answer API (limited but free)
        std::string url = "https://api.duckduckgo.com/?q=" + encoded_query + "&format=json";

        std::map<std::string, std::string> headers;
        headers["Accept"] = "application/json";

        HttpResponse response = client.get(url, headers);

        if (!response.is_success()) {
            LOG_ERROR("DuckDuckGo API request failed: " + response.error_message);
            return results;
        }

        auto json_response = json::parse(response.body);

        // DuckDuckGo instant answer format
        if (json_response.contains("RelatedTopics")) {
            for (const auto& topic : json_response["RelatedTopics"]) {
                if (topic.is_object() && topic.contains("Text") && topic.contains("FirstURL")) {
                    WebSearchResult result;
                    result.description = topic.value("Text", "");
                    result.url = topic.value("FirstURL", "");
                    // Extract title from text (usually first sentence)
                    std::string text = result.description;
                    size_t dash_pos = text.find(" - ");
                    if (dash_pos != std::string::npos) {
                        result.title = text.substr(0, dash_pos);
                    } else {
                        result.title = text.substr(0, std::min(size_t(80), text.length()));
                    }
                    results.push_back(result);

                    if (results.size() >= 10) break;
                }
            }
        }

    } catch (const std::exception& e) {
        LOG_ERROR("DuckDuckGo error: " + std::string(e.what()));
    }

    return results;
}

// SearXNGProvider implementation
SearXNGProvider::SearXNGProvider(const std::string& instance_url)
    : instance_url_(instance_url) {
    LOG_INFO("Initialized SearXNG provider with instance: " + instance_url);
}

std::vector<WebSearchResult> SearXNGProvider::search(const std::string& query) {
    std::vector<WebSearchResult> results;

    try {
        HttpClient client;

        // URL encode the query
        std::string encoded_query;
        for (char c : query) {
            if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                encoded_query += c;
            } else if (c == ' ') {
                encoded_query += '+';
            } else {
                char hex[4];
                snprintf(hex, sizeof(hex), "%%%02X", (unsigned char)c);
                encoded_query += hex;
            }
        }

        std::string url = instance_url_ + "/search?q=" + encoded_query + "&format=json";

        std::map<std::string, std::string> headers;
        headers["Accept"] = "application/json";

        HttpResponse response = client.get(url, headers);

        if (!response.is_success()) {
            LOG_ERROR("SearXNG request failed: " + response.error_message);
            return results;
        }

        auto json_response = json::parse(response.body);

        if (json_response.contains("results")) {
            for (const auto& item : json_response["results"]) {
                WebSearchResult result;
                result.title = item.value("title", "");
                result.url = item.value("url", "");
                result.description = item.value("content", "");
                results.push_back(result);

                if (results.size() >= 10) break;
            }
        }

    } catch (const std::exception& e) {
        LOG_ERROR("SearXNG error: " + std::string(e.what()));
    }

    return results;
}

// WebSearch implementation
WebSearch& WebSearch::instance() {
    static WebSearch manager;

    // Auto-initialize from config on first use
    if (!manager.is_available() && config && !config->web_search_provider.empty()) {
        manager.initialize(config->web_search_provider,
                          config->web_search_api_key,
                          config->web_search_instance_url);
    }

    return manager;
}

void WebSearch::initialize(const std::string& provider, const std::string& api_key, const std::string& instance_url) {
    if (provider == "brave") {
        if (api_key.empty()) {
            LOG_ERROR("Brave Search requires an API key");
            return;
        }
        provider_ = std::make_unique<BraveSearchProvider>(api_key);
    } else if (provider == "duckduckgo") {
        provider_ = std::make_unique<DuckDuckGoProvider>();
    } else if (provider == "searxng") {
        if (instance_url.empty()) {
            LOG_ERROR("SearXNG requires an instance URL");
            return;
        }
        provider_ = std::make_unique<SearXNGProvider>(instance_url);
    } else {
        LOG_ERROR("Unknown search provider: " + provider);
        return;
    }

    LOG_INFO("Web search initialized with provider: " + provider);
}

std::vector<WebSearchResult> WebSearch::search(const std::string& query) {
    if (!provider_) {
        LOG_WARN("Web search not initialized");
        return {};
    }

    return provider_->search(query);
}

std::string WebSearch::get_provider_name() const {
    if (!provider_) {
        return "none";
    }
    return provider_->get_name();
}
