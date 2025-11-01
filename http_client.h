#pragma once

#include <string>
#include <map>
#include <functional>
#include <memory>

#ifdef ENABLE_API_BACKENDS
#include <curl/curl.h>
#endif

/// @brief HTTP response structure
struct HttpResponse {
    long status_code = 0;
    std::string body;
    std::map<std::string, std::string> headers;
    std::string error_message;

    bool is_success() const {
        return status_code >= 200 && status_code < 300;
    }

    bool is_error() const {
        return status_code >= 400 || status_code == 0;
    }
};

/// @brief Callback for streaming responses
/// @param chunk The chunk of data received
/// @param user_data User-provided data pointer
/// @return true to continue streaming, false to abort
using StreamCallback = std::function<bool(const std::string& chunk, void* user_data)>;

/// @brief Shared HTTP client for API backends
/// Provides GET/POST methods with header management and streaming support
class HttpClient {
public:
    HttpClient();
    ~HttpClient();

    // Disable copy
    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    /// @brief Perform HTTP GET request
    /// @param url Full URL to request
    /// @param headers Optional custom headers
    /// @return Response object
    HttpResponse get(const std::string& url,
                     const std::map<std::string, std::string>& headers = {});

    /// @brief Perform HTTP POST request
    /// @param url Full URL to request
    /// @param body Request body (typically JSON)
    /// @param headers Optional custom headers
    /// @return Response object
    HttpResponse post(const std::string& url,
                      const std::string& body,
                      const std::map<std::string, std::string>& headers = {});

    /// @brief Perform HTTP POST request with streaming response
    /// @param url Full URL to request
    /// @param body Request body (typically JSON)
    /// @param headers Optional custom headers
    /// @param callback Callback function for each chunk
    /// @param user_data User data passed to callback
    /// @return Response object (body will be empty as it's streamed)
    HttpResponse post_stream(const std::string& url,
                            const std::string& body,
                            const std::map<std::string, std::string>& headers,
                            StreamCallback callback,
                            void* user_data = nullptr);

    /// @brief Set request timeout in seconds
    /// @param timeout_seconds Timeout in seconds (0 = no timeout)
    void set_timeout(long timeout_seconds);

    /// @brief Set whether to verify SSL certificates
    /// @param verify true to verify (default), false to skip verification
    void set_ssl_verify(bool verify);

    /// @brief Set custom CA bundle path for SSL verification
    /// @param ca_bundle_path Path to CA bundle file
    void set_ca_bundle(const std::string& ca_bundle_path);

    /// @brief Enable/disable verbose debug output
    /// @param verbose true to enable curl verbose output
    void set_verbose(bool verbose);

private:
#ifdef ENABLE_API_BACKENDS
    CURL* curl_ = nullptr;
    long timeout_seconds_ = 0; // No timeout by default
    bool ssl_verify_ = true;
    bool verbose_ = false;
    std::string ca_bundle_path_;

    /// @brief Configure curl handle with common options
    void configure_curl();

    /// @brief Set headers on curl handle
    /// @param headers Headers to set
    /// @return curl_slist that must be freed by caller
    struct curl_slist* set_headers(const std::map<std::string, std::string>& headers);

    /// @brief Static callback for curl write function
    static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata);

    /// @brief Static callback for curl header function
    static size_t header_callback(char* ptr, size_t size, size_t nmemb, void* userdata);

    /// @brief Static callback for curl progress function (used for cancellation)
    static int progress_callback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow);

    /// @brief Static callback for streaming write function
    static size_t stream_callback(char* ptr, size_t size, size_t nmemb, void* userdata);


    /// @brief Structure for streaming callback data
    struct StreamCallbackData {
        StreamCallback callback;
        void* user_data;
        bool continue_streaming = true;
    };
#endif
};
