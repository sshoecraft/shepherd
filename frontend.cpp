#include "frontend.h"
#include "cli.h"
#include "server/api_server.h"
#include "server/cli_server.h"

Frontend::Frontend() {
}

Frontend::~Frontend() {
}

std::unique_ptr<Frontend> Frontend::create( const std::string& mode, const std::string& host, int port) {
    if (mode == "cli") {
        return std::make_unique<CLI>();
    }
    if (mode == "api-server") {
        return std::make_unique<APIServer>(host, port);
    }
    if (mode == "cli-server") {
        return std::make_unique<CLIServer>(host, port);
    }

    throw std::runtime_error("Invalid frontend mode: " + mode);
}
