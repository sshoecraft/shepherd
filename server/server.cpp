#include "shepherd.h"
#include "server/server.h"
#include "server/api_server.h"
#include "tools/tool_parser.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// Server base class implementation
Server::Server(const std::string& host, int port)
    : Frontend(), host(host), port(port) {
}

Server::~Server() {
}

