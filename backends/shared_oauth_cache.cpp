#include "shared_oauth_cache.h"
#include "nlohmann/json.hpp"
#include "shepherd.h"
#include <iostream>

SharedOAuthCache::SharedOAuthCache() {
    http_client_ = std::make_unique<HttpClient>();
}

SharedOAuthCache::~SharedOAuthCache() = default;

std::string SharedOAuthCache::make_cache_key(const std::string& token_url, const std::string& client_id) {
    return token_url + "|" + client_id;
}

bool SharedOAuthCache::is_configured(const std::string& client_id,
                                      const std::string& client_secret,
                                      const std::string& token_url) {
    return !client_id.empty() && !client_secret.empty() && !token_url.empty();
}

SharedOAuthCache::OAuthToken SharedOAuthCache::get_token(const std::string& client_id,
                                                          const std::string& client_secret,
                                                          const std::string& token_url,
                                                          const std::string& scope) {
    if (!is_configured(client_id, client_secret, token_url)) {
        return OAuthToken{};  // Return empty token if not configured
    }

    std::lock_guard<std::mutex> lock(mutex_);

    std::string key = make_cache_key(token_url, client_id);

    // Check if we have a valid cached token
    auto it = tokens_.find(key);
    if (it != tokens_.end() && it->second.is_valid()) {
        dout(1) << "SharedOAuthCache: Using cached token for " + token_url << std::endl;
        return it->second;
    }

    // Token expired or missing, acquire new one
    dout(1) << "SharedOAuthCache: Acquiring new token for " + token_url << std::endl;
    OAuthToken token = acquire_token(client_id, client_secret, token_url, scope);

    if (!token.access_token.empty()) {
        tokens_[key] = token;
        dout(1) << "SharedOAuthCache: Token cached successfully" << std::endl;
    }

    return token;
}

SharedOAuthCache::OAuthToken SharedOAuthCache::acquire_token(const std::string& client_id,
                                                              const std::string& client_secret,
                                                              const std::string& token_url,
                                                              const std::string& scope) {
    OAuthToken token;

    try {
        // Build form-urlencoded request body
        std::string body = "grant_type=client_credentials&client_id=" + client_id +
                          "&client_secret=" + client_secret;
        if (!scope.empty()) {
            body += "&scope=" + scope;
        }

        // Set headers for OAuth token request
        std::map<std::string, std::string> headers = {
            {"Content-Type", "application/x-www-form-urlencoded"},
            {"Accept", "application/json"}
        };

        // Make POST request
        HttpResponse response = http_client_->post(token_url, body, headers);

        if (!response.is_success()) {
            std::cerr << "SharedOAuthCache: Token request failed with status "
                      << response.status_code << std::endl;
            std::cerr << "Response body: " << response.body << std::endl;
            return token;  // Return empty token
        }

        // Parse JSON response
        nlohmann::json response_json = nlohmann::json::parse(response.body);

        token.access_token = response_json.value("access_token", "");
        token.token_type = response_json.value("token_type", "Bearer");
        int expires_in = response_json.value("expires_in", 3600);

        // Set expiry time (current time + expires_in, minus 60 seconds buffer)
        token.expires_at = time(nullptr) + expires_in - 60;

        dout(1) << "SharedOAuthCache: Token acquired (expires in "
                << expires_in << " seconds)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "SharedOAuthCache: Failed to acquire token: " << e.what() << std::endl;
    }

    return token;
}
