#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>

/// Search result from a web search provider
struct WebSearchResult {
    std::string title;
    std::string url;
    std::string description;
};

/// Abstract interface for web search providers
class SearchProvider {
public:
    virtual ~SearchProvider() = default;

    /// Perform a web search
    /// @param query Search query
    /// @return Vector of search results
    virtual std::vector<WebSearchResult> search(const std::string& query) = 0;

    /// Get the name of this search provider
    virtual std::string get_name() const = 0;
};

/// Brave Search API provider
class BraveSearchProvider : public SearchProvider {
public:
    explicit BraveSearchProvider(const std::string& api_key);

    std::vector<WebSearchResult> search(const std::string& query) override;
    std::string get_name() const override { return "brave"; }

private:
    std::string api_key_;
};

/// DuckDuckGo provider (uses HTML scraping, no API key needed)
class DuckDuckGoProvider : public SearchProvider {
public:
    DuckDuckGoProvider() = default;

    std::vector<WebSearchResult> search(const std::string& query) override;
    std::string get_name() const override { return "duckduckgo"; }
};

/// SearXNG provider (self-hosted)
class SearXNGProvider : public SearchProvider {
public:
    explicit SearXNGProvider(const std::string& instance_url);

    std::vector<WebSearchResult> search(const std::string& query) override;
    std::string get_name() const override { return "searxng"; }

private:
    std::string instance_url_;
};

/// Web search manager - singleton that manages search providers
class WebSearchManager {
public:
    static WebSearchManager& instance();

    /// Initialize with provider from config
    /// @param provider Provider name ("brave", "duckduckgo", "searxng")
    /// @param api_key API key (if required by provider)
    /// @param instance_url Instance URL (for SearXNG)
    void initialize(const std::string& provider, const std::string& api_key = "", const std::string& instance_url = "");

    /// Perform a web search using the configured provider
    /// @param query Search query
    /// @return Vector of search results
    std::vector<WebSearchResult> search(const std::string& query);

    /// Check if search is available
    bool is_available() const { return provider_ != nullptr; }

    /// Get current provider name
    std::string get_provider_name() const;

private:
    WebSearchManager() = default;
    std::unique_ptr<SearchProvider> provider_;
};
