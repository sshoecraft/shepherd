#include "http_client.h"
#include "logger.h"
#include "terminal_io.h"
#include <sstream>
#include <cstring>
#include <fstream>

HttpClient::HttpClient() {
    curl_ = curl_easy_init();
    if (!curl_) {
        LOG_ERROR("Failed to initialize CURL for HttpClient");
    }
    multi_handle_ = curl_multi_init();
    if (!multi_handle_) {
        LOG_ERROR("Failed to initialize CURL multi handle for HttpClient");
    }
    LOG_DEBUG("HttpClient initialized");
}

HttpClient::~HttpClient() {
    if (multi_handle_) {
        curl_multi_cleanup(multi_handle_);
        multi_handle_ = nullptr;
    }
    if (curl_) {
        curl_easy_cleanup(curl_);
        curl_ = nullptr;
    }
    LOG_DEBUG("HttpClient destroyed");
}

void HttpClient::set_timeout(long timeout_seconds) {
    timeout_seconds_ = timeout_seconds;
    LOG_DEBUG("HttpClient timeout set to " + std::to_string(timeout_seconds) + " seconds");
}

void HttpClient::set_ssl_verify(bool verify) {
    ssl_verify_ = verify;
    LOG_DEBUG("HttpClient SSL verify: " + std::string(verify ? "enabled" : "disabled"));
}

void HttpClient::set_ca_bundle(const std::string& ca_bundle_path) {
    ca_bundle_path_ = ca_bundle_path;
    LOG_DEBUG("HttpClient CA bundle: " + ca_bundle_path);
}

void HttpClient::set_verbose(bool verbose) {
    verbose_ = verbose;
    LOG_DEBUG("HttpClient verbose: " + std::string(verbose ? "enabled" : "disabled"));
}

#ifdef ENABLE_API_BACKENDS
void HttpClient::configure_curl() {
    if (!curl_) return;

    // Reset to clean state
    curl_easy_reset(curl_);

    // Set timeout
    if (timeout_seconds_ > 0) {
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT, timeout_seconds_);
        // Also set connect timeout to avoid hanging on connection
        curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT, 30L);
    }

    // SSL options
    curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYPEER, ssl_verify_ ? 1L : 0L);
    curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYHOST, ssl_verify_ ? 2L : 0L);

    if (!ca_bundle_path_.empty()) {
        curl_easy_setopt(curl_, CURLOPT_CAINFO, ca_bundle_path_.c_str());
    }

    // Verbose output for debugging
    if (verbose_) {
        curl_easy_setopt(curl_, CURLOPT_VERBOSE, 1L);
    }

    // Follow redirects
    curl_easy_setopt(curl_, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl_, CURLOPT_MAXREDIRS, 5L);
}

struct curl_slist* HttpClient::set_headers(const std::map<std::string, std::string>& headers) {
    struct curl_slist* header_list = nullptr;

    for (const auto& [key, value] : headers) {
        std::string header = key + ": " + value;
        header_list = curl_slist_append(header_list, header.c_str());
    }

    return header_list;
}

size_t HttpClient::write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total_size = size * nmemb;
    auto* response = static_cast<HttpResponse*>(userdata);
    response->body.append(ptr, total_size);
    return total_size;
}

size_t HttpClient::header_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total_size = size * nmemb;
    auto* response = static_cast<HttpResponse*>(userdata);

    std::string header_line(ptr, total_size);

    // Parse header line (format: "Key: Value\r\n")
    size_t colon_pos = header_line.find(':');
    if (colon_pos != std::string::npos) {
        std::string key = header_line.substr(0, colon_pos);
        std::string value = header_line.substr(colon_pos + 1);

        // Trim whitespace
        size_t start = value.find_first_not_of(" \t\r\n");
        size_t end = value.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos) {
            value = value.substr(start, end - start + 1);
        }

        response->headers[key] = value;
    }

    return total_size;
}

size_t HttpClient::stream_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total_size = size * nmemb;
    auto* data = static_cast<StreamCallbackData*>(userdata);

    if (!data->continue_streaming) {
        return 0; // Abort transfer
    }

    std::string chunk(ptr, total_size);

    // Call user callback
    if (data->callback) {
        data->continue_streaming = data->callback(chunk, data->user_data);
    }

    return data->continue_streaming ? total_size : 0;
}
#endif

HttpResponse HttpClient::get(const std::string& url,
                              const std::map<std::string, std::string>& headers) {
    HttpResponse response;

#ifdef ENABLE_API_BACKENDS
    if (!curl_) {
        response.error_message = "CURL not initialized";
        return response;
    }

    LOG_DEBUG("HTTP GET: " + url);

    configure_curl();

    // Set URL
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());

    // Set callbacks
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl_, CURLOPT_HEADERDATA, &response);

    // Set headers
    struct curl_slist* header_list = set_headers(headers);
    if (header_list) {
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
    }

    // Perform request
    CURLcode res = curl_easy_perform(curl_);

    // Get status code
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response.status_code);

    // Clean up
    if (header_list) {
        curl_slist_free_all(header_list);
    }

    if (res != CURLE_OK) {
        response.error_message = curl_easy_strerror(res);
        LOG_ERROR("HTTP GET failed: " + response.error_message);
    } else {
        LOG_DEBUG("HTTP GET completed with status: " + std::to_string(response.status_code));
    }
#else
    response.error_message = "API backends not compiled in";
    LOG_ERROR("HttpClient::get() called but API backends not compiled in");
#endif

    return response;
}

HttpResponse HttpClient::get_stream(const std::string& url,
                                     const std::map<std::string, std::string>& headers,
                                     StreamCallback callback,
                                     void* user_data) {
    HttpResponse response;

#ifdef ENABLE_API_BACKENDS
    if (!curl_) {
        response.error_message = "CURL not initialized";
        return response;
    }

    LOG_DEBUG("HTTP GET (streaming): " + url);

    configure_curl();

    // Set URL
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());

    // Set streaming callback
    StreamCallbackData callback_data;
    callback_data.callback = callback;
    callback_data.user_data = user_data;

    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, stream_callback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &callback_data);
    curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl_, CURLOPT_HEADERDATA, &response);

    // Set headers
    struct curl_slist* header_list = set_headers(headers);
    if (header_list) {
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
    }

    // No timeout for SSE connections
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 0L);

    // Perform request
    CURLcode res = curl_easy_perform(curl_);

    // Get status code
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response.status_code);

    // Clean up
    if (header_list) {
        curl_slist_free_all(header_list);
    }

    if (res != CURLE_OK) {
        response.error_message = curl_easy_strerror(res);
        // CURLE_WRITE_ERROR (23) happens when callback returns false - intentional cancellation
        if (res == CURLE_WRITE_ERROR) {
            LOG_DEBUG("HTTP GET (streaming) stopped by callback");
        } else {
            LOG_ERROR("HTTP GET (streaming) failed: " + response.error_message);
        }
    } else {
        LOG_DEBUG("HTTP GET (streaming) completed with status: " + std::to_string(response.status_code));
    }
#else
    response.error_message = "API backends not compiled in";
    LOG_ERROR("HttpClient::get_stream() called but API backends not compiled in");
#endif

    return response;
}

HttpResponse HttpClient::post(const std::string& url,
                               const std::string& body,
                               const std::map<std::string, std::string>& headers) {
    HttpResponse response;

#ifdef ENABLE_API_BACKENDS
    if (!curl_) {
        response.error_message = "CURL not initialized";
        return response;
    }

    LOG_DEBUG("HTTP POST: " + url);
    LOG_DEBUG("POST body length: " + std::to_string(body.length()));

    // Dump full request body at high debug level
    extern int g_debug_level;
    if (g_debug_level >= 5 && body.length() < 50000) {
        LOG_DEBUG("POST body:\n" + body);
#if 0
        // Also write to file for easy inspection
        std::ofstream dump_file("/tmp/shepherd_request.json");
        if (dump_file.is_open()) {
            dump_file << body;
            dump_file.close();
            LOG_DEBUG("Request saved to /tmp/shepherd_request.json");
        }
#endif
    }

    configure_curl();

    // Set URL
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());

    // Set POST
    curl_easy_setopt(curl_, CURLOPT_POST, 1L);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, body.length());

    // Set callbacks
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl_, CURLOPT_HEADERDATA, &response);

    // Set headers
    struct curl_slist* header_list = set_headers(headers);
    if (header_list) {
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
    }

    // Perform request
    CURLcode res = curl_easy_perform(curl_);

    // Get status code
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response.status_code);

    // Clean up
    if (header_list) {
        curl_slist_free_all(header_list);
    }

    if (res != CURLE_OK) {
        response.error_message = curl_easy_strerror(res);
        LOG_ERROR("HTTP POST failed: " + response.error_message);
    } else {
        LOG_DEBUG("HTTP POST completed with status: " + std::to_string(response.status_code));
        if (response.body.length() > 100) {
            LOG_DEBUG("Response body (first 100 chars): " + response.body.substr(0, 100));
            if (g_debug_level >= 5) {
                LOG_DEBUG("Full response body:\n" + response.body);
            }
        } else {
            LOG_DEBUG("Response body: " + response.body);
        }
    }
#else
    response.error_message = "API backends not compiled in";
    LOG_ERROR("HttpClient::post() called but API backends not compiled in");
#endif

    return response;
}

HttpResponse HttpClient::post_stream(const std::string& url,
                                      const std::string& body,
                                      const std::map<std::string, std::string>& headers,
                                      StreamCallback callback,
                                      void* user_data) {
    HttpResponse response;

#ifdef ENABLE_API_BACKENDS
    if (!curl_) {
        response.error_message = "CURL not initialized";
        return response;
    }

    LOG_DEBUG("HTTP POST (streaming): " + url);
    LOG_DEBUG("POST body length: " + std::to_string(body.length()));

    // Dump full request body at high debug level
    extern int g_debug_level;
    if (g_debug_level >= 5 && body.length() < 50000) {
        LOG_DEBUG("POST body:\n" + body);
    }

    configure_curl();

    // Set URL
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());

    // Set POST
    curl_easy_setopt(curl_, CURLOPT_POST, 1L);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, body.length());

    // Set streaming callback
    StreamCallbackData callback_data;
    callback_data.callback = callback;
    callback_data.user_data = user_data;

    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, stream_callback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &callback_data);
    curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl_, CURLOPT_HEADERDATA, &response);

    // Set headers
    struct curl_slist* header_list = set_headers(headers);
    if (header_list) {
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
    }

    // Perform request
    CURLcode res = curl_easy_perform(curl_);

    // Get status code
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response.status_code);

    // Clean up
    if (header_list) {
        curl_slist_free_all(header_list);
    }

    if (res != CURLE_OK) {
        response.error_message = curl_easy_strerror(res);
        LOG_ERROR("HTTP POST (streaming) failed: " + response.error_message);
    } else {
        LOG_DEBUG("HTTP POST (streaming) completed with status: " + std::to_string(response.status_code));
    }
#else
    response.error_message = "API backends not compiled in";
    LOG_ERROR("HttpClient::post_stream() called but API backends not compiled in");
#endif

    return response;
}

HttpResponse HttpClient::post_stream_cancellable(const std::string& url,
                                                  const std::string& body,
                                                  const std::map<std::string, std::string>& headers,
                                                  StreamCallback callback,
                                                  void* user_data) {
    HttpResponse response;

#ifdef ENABLE_API_BACKENDS
    if (!curl_ || !multi_handle_) {
        response.error_message = "CURL not initialized";
        return response;
    }

    LOG_DEBUG("HTTP POST (streaming cancellable): " + url);
    LOG_DEBUG("POST body length: " + std::to_string(body.length()));

    // Dump full request body at high debug level
    extern int g_debug_level;
    if (g_debug_level >= 5 && body.length() < 50000) {
        LOG_DEBUG("POST body:\n" + body);
    }

    configure_curl();

    // Set URL
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());

    // Set POST
    curl_easy_setopt(curl_, CURLOPT_POST, 1L);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, body.length());

    // Set streaming callback
    StreamCallbackData callback_data;
    callback_data.callback = callback;
    callback_data.user_data = user_data;

    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, stream_callback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &callback_data);
    curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl_, CURLOPT_HEADERDATA, &response);

    // Set headers
    struct curl_slist* header_list = set_headers(headers);
    if (header_list) {
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
    }

    // Enable raw mode for escape key detection
    tio.set_raw_mode();

    // Add handle to multi interface
    CURLMcode mres = curl_multi_add_handle(multi_handle_, curl_);
    if (mres != CURLM_OK) {
        response.error_message = "Failed to add handle to multi interface";
        LOG_ERROR("curl_multi_add_handle failed: " + std::string(curl_multi_strerror(mres)));
        tio.restore_terminal();
        if (header_list) {
            curl_slist_free_all(header_list);
        }
        return response;
    }

    // Event loop with escape key checking
    int still_running = 0;
    bool cancelled = false;
    CURLcode curl_result = CURLE_OK;

    do {
        // Perform transfers
        mres = curl_multi_perform(multi_handle_, &still_running);
        if (mres != CURLM_OK) {
            LOG_ERROR("curl_multi_perform failed: " + std::string(curl_multi_strerror(mres)));
            break;
        }

        // Check for escape key
        if (tio.check_escape_pressed()) {
            LOG_INFO("Escape key pressed, cancelling request");
            cancelled = true;
            break;
        }

        // Wait for activity with short timeout (100ms for responsive cancellation)
        if (still_running) {
            mres = curl_multi_poll(multi_handle_, nullptr, 0, 100, nullptr);
            if (mres != CURLM_OK) {
                LOG_ERROR("curl_multi_poll failed: " + std::string(curl_multi_strerror(mres)));
                break;
            }
        }
    } while (still_running);

    // Check for completion messages
    int msgs_left = 0;
    CURLMsg* msg = nullptr;
    while ((msg = curl_multi_info_read(multi_handle_, &msgs_left))) {
        if (msg->msg == CURLMSG_DONE) {
            curl_result = msg->data.result;
            break;
        }
    }

    // Get status code
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response.status_code);

    // Remove handle from multi interface
    curl_multi_remove_handle(multi_handle_, curl_);

    // Restore terminal
    tio.restore_terminal();

    // Clean up headers
    if (header_list) {
        curl_slist_free_all(header_list);
    }

    // Set error message if cancelled or failed
    if (cancelled) {
        response.error_message = "Request cancelled by user";
        response.status_code = 0;
        LOG_INFO("HTTP POST (streaming cancellable) cancelled");
    } else if (curl_result != CURLE_OK) {
        response.error_message = curl_easy_strerror(curl_result);
        LOG_ERROR("HTTP POST (streaming cancellable) failed: " + response.error_message);
    } else {
        LOG_DEBUG("HTTP POST (streaming cancellable) completed with status: " + std::to_string(response.status_code));
    }
#else
    response.error_message = "API backends not compiled in";
    LOG_ERROR("HttpClient::post_stream_cancellable() called but API backends not compiled in");
#endif

    return response;
}
