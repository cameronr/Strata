// src/server/net.hpp - who may reach the server: the Host check (DNS rebinding), web pages' origins, API keys, CORS
// origins, this PC's network addresses, and why a port cannot be listened on.
#pragma once

#include "json.hpp"

#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace lamina {

/// the names that are this PC itself
extern const std::vector<std::string> kLoopbackNames;

/// this PC's IPv4 addresses on its networks (what another device types in), loopback and link-local left out
std::vector<std::string> lan_addresses();
/// the name in a Host header or an origin's host[:port] ("Example.com:8080" -> "example.com", "[::1]:8095" -> "::1");
/// "" when it is malformed
std::string host_name(std::string_view value);
bool is_ip(const std::string& name);
/// `name` is one of `names`, or below an entry that starts with a dot (".example.com")
bool name_in(const std::string& name, const std::set<std::string>& names);
/// the config's allowed_hosts (a name or a list) plus $STRATA_ALLOWED_HOSTS; std::invalid_argument for a wrong entry
std::vector<std::string> allowed_hosts_of(const json& value, const std::string& env);
/// every name this server answers to besides an IP address
std::set<std::string> host_names_for(const std::string& bind_host, const std::vector<std::string>& allowed_hosts,
                                     const std::vector<std::string>& trusted_origins);
/// DNS rebinding protection: only the names this server answers to (or an IP, *.localhost, no Host at all) pass
bool host_allowed(const std::string& host, const std::set<std::string>& names, bool any_host);
/// the keys a server accepts: "k1,k2" (llama.cpp's form) or a config list
std::vector<std::string> key_list(const json& value);
/// the key as a client can send it (spaces and line ends around it dropped); std::invalid_argument when it is empty
json api_key_of(const json& value);
/// the Authorization / x-api-key value against the keys, in constant time, as UTF-8 and as Latin-1
bool key_matches(std::string_view given, const json& key);
/// a web page's Origin that may use the model without an API key
bool origin_allowed(std::string_view origin, const std::string& host, const std::set<std::string>& names,
                    const std::vector<std::string>& origins);
/// the config's origin list (cors_origins, trusted_origins); std::invalid_argument for a wrong entry
std::vector<std::string> origins_of(const json& value, const std::string& key, bool wildcard);
/// #769: why the server cannot listen on host:port, from the OS's error code
std::string listen_problem(const std::string& host, int port, int code);
/// binds and closes a socket on host:port: 0 when it is free, else the OS's error code
int probe_listen(const std::string& host, int port);

}  // namespace lamina
