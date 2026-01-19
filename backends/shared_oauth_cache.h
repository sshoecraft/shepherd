#pragma once

#include <string>
#include <map>
#include <mutex>
#include <memory>
#include <ctime>
#include "http_client.h"

/// @brief Shared OAuth token cache for per-request backends
/// Caches OAuth tokens by token_url to avoid re-authentication on every request
class SharedOAuthCache {
public:
    struct OAuthToken {
        std::string access_token;
        std::string token_type = "Bearer";
        time_t expires_at = 0;  // Unix timestamp when token expires
        bool is_valid() const { return !access_token.empty() && time(nullptr) < expires_at; }
    };

    SharedOAuthCache();
    ~SharedOAuthCache();

    /// @brief Get a valid OAuth token, refreshing if expired
    /// Thread-safe - multiple backends can call this concurrently
    /// @param client_id OAuth client ID
    /// @param client_secret OAuth client secret
    /// @param token_url OAuth token endpoint URL
    /// @param scope OAuth scope (optional)
    /// @return Valid OAuth token, or empty token on failure
    OAuthToken get_token(const std::string& client_id,
                         const std::string& client_secret,
                         const std::string& token_url,
                         const std::string& scope = "");

    /// @brief Check if OAuth is configured (has credentials)
    static bool is_configured(const std::string& client_id,
                              const std::string& client_secret,
                              const std::string& token_url);

private:
    /// @brief Acquire a new OAuth token from the token endpoint
    OAuthToken acquire_token(const std::string& client_id,
                             const std::string& client_secret,
                             const std::string& token_url,
                             const std::string& scope);

    /// @brief Generate cache key from token URL and client ID
    std::string make_cache_key(const std::string& token_url, const std::string& client_id);

    std::map<std::string, OAuthToken> tokens_;
    std::mutex mutex_;
    std::unique_ptr<HttpClient> http_client_;
};
