// src/server/http_client.hpp - outgoing HTTP(S) requests: image URLs and MCP servers reached by address.
// Windows: WinHTTP (the system's TLS, certificates and proxy settings); elsewhere cpp-httplib (TLS with OpenSSL).
#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lamina {

using HttpHeaders = std::vector<std::pair<std::string, std::string>>;

struct HttpReply {
    int status = 0;
    std::string reason;    ///< the status line's words ("Not Found")
    HttpHeaders headers;   ///< names in lower case, in the order received
    std::string body;      ///< empty when an on_data callback took it
    /// a header's value ("" when absent); `name` in lower case
    std::string header(std::string_view name) const;
};

/// the request could not be made or read (no connection, a timeout, a TLS failure, too large a body)
struct HttpError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// One request; redirects are followed. `timeout_s` bounds the connection and each read. `max_bytes` > 0: an
/// HttpError once the body is larger. `on_data` takes the body as it arrives (returning false stops reading) instead
/// of collecting it. A reply of any status is returned (an HTTP error status is not an exception here).
HttpReply http_request(const std::string& method, const std::string& url, const HttpHeaders& headers,
                       const std::string& body, double timeout_s, size_t max_bytes = 0,
                       const std::function<bool(std::string_view)>& on_data = nullptr);

}  // namespace lamina
