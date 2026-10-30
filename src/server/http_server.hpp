// src/server/http_server.hpp - the HTTP side of lamina-server (serve/server.py's Handler): the OpenAI, Anthropic and
// Responses APIs, the web app and its endpoints, the Monitor's, and who may use them (the Host check, API keys, web
// pages' origins, CORS).
#pragma once

#include "service.hpp"

#include <memory>
#include <string>
#include <thread>

namespace httplib {
class Server;
}

namespace lamina {

namespace prometheus {
class Latencies;
}

class HttpServer {
public:
    explicit HttpServer(Service& svc);
    ~HttpServer();
    /// listens on host:port (port 0: one the OS picks); std::runtime_error with the reason when it cannot
    void bind(const std::string& host, int port);
    int port() const { return port_; }
    /// serves on a thread of its own
    void start();
    void stop();
    std::shared_ptr<prometheus::Latencies> latencies;

private:
    Service& svc_;
    std::unique_ptr<httplib::Server> http_;
    std::thread thread_;
    int port_ = 0;
};

/// the most a request body may hold: 256 MiB, or LAMINA_MAX_BODY_MIB
size_t body_limit();

}  // namespace lamina
