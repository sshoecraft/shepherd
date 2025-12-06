#pragma once

#include "../frontend.h"
#include <memory>
#include <string>

/// @brief Base class for all server frontends (API Server, CLI Server)
class Server : public Frontend {
public:
    Server(const std::string& host, int port);
    virtual ~Server();

protected:
    std::string host;
    int port;
};

