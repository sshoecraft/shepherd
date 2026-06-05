#include "shepherd.h"
#include "openapi_server.h"
#include "../version.h"
#include "../tools/tool.h"
#include "../tools/json_schema.h"
#include "../tools/utf8_sanitizer.h"
#include "../config.h"
#include <iostream>
#include <sstream>
#include <set>
#include <map>

using json = nlohmann::json;

extern std::unique_ptr<Config> config;

OpenAPIServer::OpenAPIServer(const std::string& host, int port,
                             const std::string& ssl_cert, const std::string& ssl_key,
                             const std::string& base_url)
    : Server(host, port, "openapi", ssl_cert, ssl_key), base_url(base_url) {
}

OpenAPIServer::~OpenAPIServer() {
}

void OpenAPIServer::init(const FrontendFlags& flags) {
    // Capture --openapi-base-url from flags (main.cpp populates it).
    if (base_url.empty() && !flags.openapi_base_url.empty()) {
        base_url = flags.openapi_base_url;
    }
    // Always initialize tools locally — we ARE the tool server.
    init_tools(flags, true);
}

std::string OpenAPIServer::extract_bearer_token(const httplib::Request& req) const {
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) return "";
    const std::string& auth = it->second;
    const std::string prefix = "Bearer ";
    if (auth.size() > prefix.size() && auth.compare(0, prefix.size(), prefix) == 0) {
        return auth.substr(prefix.size());
    }
    return "";
}

int OpenAPIServer::run(Provider* /*cmdline_provider*/) {
    // No provider, no backend — we never call into an LLM. Set start time and
    // jump straight to endpoint registration + listen.
    start_time = std::chrono::steady_clock::now();

    // Register /health and /status from base.
    // (private in Server, but register_endpoints() is invoked from Server::run
    // normally — here we replicate the relevant subset.)
    //
    // /health and /status: cpp-httplib lets us define them inline; the base
    // class normally adds them via register_common_endpoints(). Since that's
    // private, we just add our own /health here and skip /status (it depends
    // on backend stats we don't have).
    tcp_server->Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json body = {{"status", "ok"}, {"service", "shepherd-openapi"}};
        res.set_content(body.dump(), "application/json");
    });

    register_endpoints();

    // Auth middleware: same pattern as Server::run, but allow /health and
    // /openapi.json unauthenticated (clients need the spec before they have
    // credentials to use, in some workflows).
    if (key_store->is_enabled()) {
        tcp_server->set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
            if (req.path == "/health" || req.path == "/openapi.json") {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            if (!check_auth(req, res)) {
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        std::cout << "API key authentication enabled" << std::endl;
    } else {
        // Decision #1 from plan: open by default, but warn loudly.
        int tool_count = 0;
        for (const auto& name : tools.list()) {
            if (tools.is_enabled(name)) tool_count++;
        }
        std::cout << "WARNING: openapi-server running without authentication ("
                  << tool_count << " tools exposed). "
                  << "Use --apikey-store to require Bearer tokens." << std::endl;
    }

    // HTTP access logging.
    tcp_server->set_logger([](const httplib::Request& req, const httplib::Response& res) {
        std::cout << "[http] " << req.remote_addr << " - \""
                  << req.method << " " << req.path << " HTTP/1.1\" "
                  << res.status << std::endl;
    });

    std::cout << "openapi server listening on "
              << (tls_enabled ? "https://" : "http://") << host << ":" << port << std::endl;

    bool success = tcp_server->listen(host.c_str(), port);
    running = false;

    if (!success) {
        std::cerr << "Failed to start openapi server on " << host << ":" << port << std::endl;
        return 1;
    }

    dout(1) << "openapi server stopped" << std::endl;
    return 0;
}

void OpenAPIServer::register_endpoints() {
    // GET /openapi.json — the OpenAPI 3.1 document.
    tcp_server->Get("/openapi.json", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            json doc = build_openapi_document(req);
            res.set_content(doc.dump(2), "application/json");
        } catch (const std::exception& e) {
            std::cerr << "Exception in /openapi.json: " << e.what() << std::endl;
            res.status = 500;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        }
    });

    // GET / — friendly hint pointing humans at the spec.
    tcp_server->Get("/", [](const httplib::Request&, httplib::Response& res) {
        json body = {
            {"service", "shepherd-openapi"},
            {"openapi", "/openapi.json"},
            {"health", "/health"}
        };
        res.set_content(body.dump(2), "application/json");
    });

    // One POST /tools/<name> per enabled tool. Routes are snapshotted at
    // startup; tools added later require a restart (matches /v1/tools today).
    int registered = 0;
    for (const auto& name : tools.list()) {
        Tool* tool = tools.get(name);
        if (!tool || !tools.is_enabled(name)) continue;

        std::string route = "/tools/" + tool->name();
        std::string tool_name = tool->name();
        tcp_server->Post(route, [this, tool_name](const httplib::Request& req, httplib::Response& res) {
            handle_tool_call(tool_name, req, res);
        });
        registered++;
    }

    dout(1) << "OpenAPI endpoints registered: /openapi.json, / , " + std::to_string(registered) + " tool routes" << std::endl;
}

void OpenAPIServer::handle_tool_call(const std::string& tool_name,
                                     const httplib::Request& req,
                                     httplib::Response& res) {
    try {
        std::string api_key = extract_bearer_token(req);

        Tool* tool = tools.get(tool_name);
        if (!tool || !tools.is_enabled(tool_name)) {
            res.status = 404;
            json err = {{"error", "Tool not found: " + tool_name}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Body is the argument object directly (per OpenAPI conventions).
        // Empty body is treated as no arguments.
        std::string arguments_json = req.body.empty() ? "{}" : req.body;
        // Validate JSON parses (don't reject yet — let the tool decide).
        try {
            (void)json::parse(arguments_json);
        } catch (const std::exception&) {
            res.status = 400;
            json err = {{"error", "Request body must be valid JSON"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Resolve user_id from API key entry (multi-tenant tools use this).
        std::string tool_user_id = user_id;  // Frontend's local user_id is a sensible fallback.
        if (key_store && !api_key.empty()) {
            const ApiKeyEntry* entry = key_store->get_entry(api_key);
            if (entry && !entry->name.empty()) {
                tool_user_id = entry->name;
            }
        }

        // Log the call (truncate long args).
        std::string params_log = arguments_json;
        if (params_log.length() > 100) params_log = params_log.substr(0, 100) + "...";
        std::cout << "[tool] " << tool_name << ": calling with " << params_log << std::endl;

        // Execute directly via tools.execute — bypasses Frontend::execute_tool
        // because that helper requires a backend (for truncation math) which
        // we don't have. OWUI / its LLM handle token budgets, not us.
        ToolResult result = tools.execute(tool_name, arguments_json, tool_user_id);

        // Build response — wrapped per Decision #2.
        if (result.success) {
            std::string content = utf8_sanitizer::sanitize_utf8(result.content);
            std::string log_summary = result.summary.empty() ? content : result.summary;
            if (log_summary.length() > 150) log_summary = log_summary.substr(0, 150) + "...";
            std::replace(log_summary.begin(), log_summary.end(), '\n', ' ');
            std::cout << "[tool] " << tool_name << ": OK - " << log_summary << std::endl;

            json body;
            body["content"] = content;
            if (!result.summary.empty()) body["summary"] = result.summary;
            res.set_content(body.dump(), "application/json");
        } else {
            std::cout << "[tool] " << tool_name << ": FAILED - " << result.error << std::endl;
            res.status = 500;
            json err = {{"error", result.error.empty() ? "Tool execution failed" : result.error}};
            res.set_content(err.dump(), "application/json");
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in POST /tools/" << tool_name << ": " << e.what() << std::endl;
        res.status = 500;
        json err = {{"error", e.what()}};
        res.set_content(err.dump(), "application/json");
    }
}

nlohmann::json OpenAPIServer::build_openapi_document(const httplib::Request& req) {
    json doc;
    doc["openapi"] = "3.1.0";
    doc["info"] = {
        {"title", "Shepherd Tool Server"},
        {"version", SHEPHERD_VERSION},
        {"description", "MCP/SMCP tools aggregated by shepherd, exposed as OpenAPI 3.1."}
    };

    // servers[0].url — explicit --openapi-base-url wins; otherwise derive from
    // the request's Host header (best effort) or fall back to --host/--port.
    std::string server_url;
    if (!base_url.empty()) {
        server_url = base_url;
    } else {
        auto host_hdr = req.headers.find("Host");
        std::string scheme = tls_enabled ? "https" : "http";
        if (host_hdr != req.headers.end() && !host_hdr->second.empty()) {
            server_url = scheme + "://" + host_hdr->second;
        } else {
            server_url = scheme + "://" + host + ":" + std::to_string(port);
        }
    }
    doc["servers"] = json::array({json{{"url", server_url}}});

    // Security scheme — only declared if a key store is active.
    bool auth_required = key_store && key_store->is_enabled();
    if (auth_required) {
        doc["components"] = {
            {"securitySchemes", {
                {"bearerAuth", {{"type", "http"}, {"scheme", "bearer"}}}
            }}
        };
        doc["security"] = json::array({json{{"bearerAuth", json::array()}}});
    }

    // One path per enabled tool.
    json paths = json::object();
    // Preserve a stable order: iterate the registry's list (already sorted by name).
    for (const auto& name : tools.list()) {
        Tool* tool = tools.get(name);
        if (!tool || !tools.is_enabled(name)) continue;

        std::string desc = tool->description();
        // summary = first line of description, capped at 120 chars.
        std::string summary = desc;
        size_t nl = summary.find('\n');
        if (nl != std::string::npos) summary = summary.substr(0, nl);
        if (summary.length() > 120) summary = summary.substr(0, 117) + "...";

        json op;
        op["operationId"] = tool->name();
        op["summary"] = summary;
        if (!desc.empty()) op["description"] = desc;
        op["tags"] = json::array({tool->source()});

        op["requestBody"] = {
            {"required", true},
            {"content", {
                {"application/json", {
                    {"schema", tool_schema::params_to_object_schema(tool->get_parameters_schema())}
                }}
            }}
        };

        op["responses"] = {
            {"200", {
                {"description", "Tool result"},
                {"content", {
                    {"application/json", {
                        {"schema", {
                            {"type", "object"},
                            {"properties", {
                                {"content", {{"type", "string"}}},
                                {"summary", {{"type", "string"}}}
                            }}
                        }}
                    }}
                }}
            }},
            {"400", {{"description", "Bad request"}}},
            {"401", {{"description", "Unauthorized"}}},
            {"404", {{"description", "Tool not found"}}},
            {"500", {{"description", "Tool error"}}}
        };

        paths["/tools/" + tool->name()] = {{"post", op}};
    }
    doc["paths"] = paths;

    // Tag descriptions: one per unique source, so OWUI/Swagger UI can group.
    std::set<std::string> sources;
    for (const auto& name : tools.list()) {
        Tool* tool = tools.get(name);
        if (tool && tools.is_enabled(name)) {
            sources.insert(tool->source());
        }
    }
    json tags_arr = json::array();
    for (const auto& s : sources) {
        tags_arr.push_back(json{{"name", s}});
    }
    doc["tags"] = tags_arr;

    return doc;
}
