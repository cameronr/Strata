// src/server/net.cpp - who may reach the server (net.hpp).
#include "net.hpp"

#include "frontend.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace lamina {
namespace {

#ifdef _WIN32
struct WsaInit {
    WsaInit() {
        WSADATA d;
        WSAStartup(MAKEWORD(2, 2), &d);
    }
};
const WsaInit wsa_init;
using sock_t = SOCKET;
const sock_t kNoSock = INVALID_SOCKET;
void close_sock(sock_t s) { closesocket(s); }
int last_error() { return WSAGetLastError(); }
#else
using sock_t = int;
const sock_t kNoSock = -1;
void close_sock(sock_t s) { ::close(s); }
int last_error() { return errno; }
#endif

std::string gethost() {
    char buf[256] = {0};
    if (gethostname(buf, sizeof buf - 1) != 0) return "";
    return buf;
}

bool all_digits(std::string_view s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return c >= '0' && c <= '9'; });
}

}  // namespace

const std::vector<std::string> kLoopbackNames = {"localhost", "127.0.0.1", "::1"};

std::vector<std::string> lan_addresses() {
    std::string first;
    // the address of the default route; sends nothing (UDP)
    const sock_t s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s != kNoSock) {
        sockaddr_in to{};
        to.sin_family = AF_INET;
        to.sin_port = htons(1);
        inet_pton(AF_INET, "10.255.255.255", &to.sin_addr);
        if (connect(s, (sockaddr*) &to, sizeof to) == 0) {
            sockaddr_in me{};
            socklen_t len = sizeof me;
            if (getsockname(s, (sockaddr*) &me, &len) == 0) {
                char ip[INET_ADDRSTRLEN] = {0};
                inet_ntop(AF_INET, &me.sin_addr, ip, sizeof ip);
                first = ip;
            }
        }
        close_sock(s);
    }
    std::set<std::string> ips;
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    const std::string host = gethost();
    if (!host.empty() && getaddrinfo(host.c_str(), nullptr, &hints, &res) == 0) {
        for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
            char ip[INET_ADDRSTRLEN] = {0};
            inet_ntop(AF_INET, &((sockaddr_in*) p->ai_addr)->sin_addr, ip, sizeof ip);
            ips.insert(ip);
        }
        freeaddrinfo(res);
    }
    auto ok = [](const std::string& ip) {
        return !ip.empty() && !starts_with(ip, "127.") && !starts_with(ip, "169.254.") && !starts_with(ip, "0.");
    };
    std::vector<std::string> out;
    if (ok(first)) out.push_back(first);
    for (const auto& ip : ips)
        if (ok(ip) && ip != first) out.push_back(ip);
    return out;
}

std::string host_name(std::string_view value) {
    std::string v = lower(strip(value));
    if (starts_with(v, "[")) {
        const size_t close = v.find(']');
        if (close == std::string::npos) return "";
        const std::string name = v.substr(1, close - 1), rest = v.substr(close + 1);
        return rest.empty() || (rest[0] == ':' && all_digits(std::string_view(rest).substr(1))) ? name : "";
    }
    const size_t colons = (size_t) std::count(v.begin(), v.end(), ':');
    if (colons == 1) {
        const size_t c = v.find(':');
        if (!all_digits(std::string_view(v).substr(c + 1))) return "";
        v = v.substr(0, c);
    } else if (colons > 1) {
        return v;   // a bare IPv6 address (a config entry)
    }
    while (!v.empty() && v.back() == '.') v.pop_back();
    if (v.empty()) return "";
    for (const unsigned char c : v)
        if (!(std::isalnum(c) || c == '-' || c == '.' || c == '_' || c >= 0x80)) return "";
    return v;
}

bool is_ip(const std::string& name) {
    in_addr a4;
    in6_addr a6;
    return inet_pton(AF_INET, name.c_str(), &a4) == 1 || inet_pton(AF_INET6, name.c_str(), &a6) == 1;
}

bool name_in(const std::string& name, const std::set<std::string>& names) {
    if (names.count(name)) return true;
    for (const auto& n : names)
        if (starts_with(n, ".") && (ends_with(name, n) || name == n.substr(1))) return true;
    return false;
}

std::vector<std::string> allowed_hosts_of(const json& value, const std::string& env) {
    std::vector<std::string> items;
    if (value.is_string()) {
        if (!value.get_ref<const std::string&>().empty()) items.push_back(value.get<std::string>());
    } else if (value.is_array()) {
        for (const auto& x : value) {
            if (!x.is_string()) throw std::invalid_argument("allowed_hosts: expected a host name or a list of them");
            items.push_back(x.get<std::string>());
        }
    } else if (!value.is_null()) {
        throw std::invalid_argument("allowed_hosts: expected a host name or a list of them");
    }
    for (const auto& x : split(env, ","))
        if (!strip(x).empty()) items.push_back(x);
    std::vector<std::string> out;
    for (const auto& raw : items) {
        std::string x = lower(strip(raw));
        if (x != "*") {
            const size_t scheme = x.find("://");
            if (scheme != std::string::npos) x = x.substr(scheme + 3);
            x = x.substr(0, x.find('/'));
            const bool dot = starts_with(x, ".");
            const std::string name = host_name(dot ? std::string_view(x).substr(1) : std::string_view(x));
            if (name.empty())
                throw std::invalid_argument("allowed_hosts: " + py_repr_value(json(raw)) +
                                            " is not a host name like lamina.example.com");
            x = dot ? "." + name : name;
        }
        if (std::find(out.begin(), out.end(), x) == out.end()) out.push_back(x);
    }
    return out;
}

std::set<std::string> host_names_for(const std::string& bind_host, const std::vector<std::string>& allowed_hosts,
                                     const std::vector<std::string>& trusted_origins) {
    std::set<std::string> names(kLoopbackNames.begin(), kLoopbackNames.end());
    const std::string bind = host_name(bind_host);
    if (!bind.empty() && bind != "0.0.0.0" && bind != "::") names.insert(bind);
    if (std::find(kLoopbackNames.begin(), kLoopbackNames.end(), bind) == kLoopbackNames.end()) {
        const std::string pc = lower(gethost());
        if (!pc.empty()) {
            names.insert(pc);
            names.insert(pc + ".local");
        }
        names.insert("host.docker.internal");
        for (const auto& ip : lan_addresses()) names.insert(ip);
    }
    for (const auto& x : allowed_hosts)
        if (x != "*") names.insert(x);
    for (const auto& o : trusted_origins) {
        const size_t s = o.find("://");
        const std::string n = host_name(s == std::string::npos ? o : o.substr(s + 3));
        if (!n.empty()) names.insert(n);
    }
    return names;
}

bool host_allowed(const std::string& host, const std::set<std::string>& names, bool any_host) {
    if (any_host || strip(host).empty()) return true;
    const std::string name = host_name(host);
    return !name.empty() && (is_ip(name) || name == "localhost" || ends_with(name, ".localhost") || name_in(name, names));
}

std::vector<std::string> key_list(const json& value) {
    std::vector<std::string> out;
    if (value.is_null() || (value.is_string() && value.get_ref<const std::string&>().empty())) return out;
    if (value.is_array()) {
        for (const auto& k : value) {
            const std::string s = k.is_string() ? k.get<std::string>() : py_dumps(k);
            if (!s.empty()) out.push_back(s);
        }
        return out;
    }
    const std::string s = value.is_string() ? value.get<std::string>() : py_dumps(value);
    for (const auto& k : split(s, ","))
        if (!strip(k).empty()) out.emplace_back(strip(k));
    return out;
}

json api_key_of(const json& value) {
    if (value.is_array()) {
        json keys = json::array();
        for (const auto& k : value) keys.push_back(std::string(strip(k.is_string() ? k.get<std::string>() : py_dumps(k))));
        if (keys.empty() || std::any_of(keys.begin(), keys.end(), [](const json& k) { return k.get<std::string>().empty(); }))
            throw std::invalid_argument("an API key was given but it is empty");
        return keys;
    }
    const std::string key = value.is_null() ? "" : value.is_string() ? value.get<std::string>() : py_dumps(value);
    if (!key.empty() && key_list(json(key)).empty()) throw std::invalid_argument("an API key was given but it is empty");
    return std::string(strip(key));
}

bool key_matches(std::string_view given, const json& key) {
    auto same = [](std::string_view a, std::string_view b) {   // constant time in b's length
        unsigned diff = a.size() == b.size() ? 0u : 1u;
        for (size_t i = 0; i < b.size(); ++i) diff |= (unsigned) (unsigned char) b[i] ^ (i < a.size() ? (unsigned char) a[i] : 0u);
        return diff == 0;
    };
    bool ok = false;
    for (const auto& k : key_list(key)) {
        ok |= same(given, k);
        std::string latin1;   // the key in Latin-1, when it has one
        bool fits = true;
        for (size_t i = 0; i < k.size();) {
            size_t n = 1;
            const uint32_t cp = utf8_at(k, i, &n);
            if (cp > 0xff) {
                fits = false;
                break;
            }
            latin1.push_back((char) cp);
            i += n;
        }
        if (fits) ok |= same(given, latin1);
    }
    return ok;
}

bool origin_allowed(std::string_view origin_in, const std::string& host, const std::set<std::string>& names,
                    const std::vector<std::string>& origins) {
    std::string origin(rstrip_chars(strip(origin_in), "/"));
    if (std::find(origins.begin(), origins.end(), origin) != origins.end() ||
        std::find(origins.begin(), origins.end(), "*") != origins.end())
        return true;   // cors_origins ["*"]: the config lets every page in
    const std::string low = lower(origin);
    const size_t sep = low.find("://");
    if (sep == std::string::npos || sep == 0) return false;
    const std::string scheme = low.substr(0, sep), rest = low.substr(sep + 3);
    if (scheme != "http" && scheme != "https") return true;   // an extension's or an app's page: no web site sends one
    if (!host.empty() && rest == lower(strip(host))) return true;
    const std::string name = host_name(rest);
    return !name.empty() && (std::find(kLoopbackNames.begin(), kLoopbackNames.end(), name) != kLoopbackNames.end() ||
                             ends_with(name, ".localhost") || name_in(name, names));
}

std::vector<std::string> origins_of(const json& value, const std::string& key, bool wildcard) {
    std::vector<std::string> out;
    if (value.is_null() || (value.is_string() && value.get_ref<const std::string&>().empty()) ||
        (value.is_array() && value.empty()))
        return out;
    std::vector<std::string> items;
    if (value.is_string()) {
        items.push_back(value.get<std::string>());
    } else if (value.is_array() && std::all_of(value.begin(), value.end(), [](const json& x) { return x.is_string(); })) {
        for (const auto& x : value) items.push_back(x.get<std::string>());
    } else {
        throw std::invalid_argument(key + ": expected an origin or a list of origins");
    }
    for (const auto& raw : items) {
        const std::string x(rstrip_chars(strip(raw), "/"));
        if (x == "*" && wildcard) {
            out.push_back(x);
            continue;
        }
        const size_t sep = x.find("://");
        const std::string scheme = sep == std::string::npos ? x : x.substr(0, sep);
        const std::string rest = sep == std::string::npos ? "" : x.substr(sep + 3);
        if ((scheme != "http" && scheme != "https") || sep == std::string::npos || rest.empty() ||
            rest.find('/') != std::string::npos || rest.find('*') != std::string::npos)
            throw std::invalid_argument(key + ": " + py_repr_value(json(x)) + " is not an origin like https://chat.example.com" +
                                        (wildcard ? "" : " (no wildcards here)"));
        out.push_back(x);
    }
    return out;
}

std::string listen_problem(const std::string& host, int port, int code) {
#ifdef _WIN32
    const bool in_use = code == WSAEADDRINUSE, denied = code == WSAEACCES, no_addr = code == WSAEADDRNOTAVAIL;
    char* buf = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
                   (DWORD) code, 0, (LPSTR) &buf, 0, nullptr);
    std::string what = buf != nullptr ? std::string(rstrip(buf)) : "error";
    if (buf != nullptr) LocalFree(buf);
#else
    const bool in_use = code == EADDRINUSE, denied = code == EACCES, no_addr = code == EADDRNOTAVAIL;
    std::string what = std::strerror(code);
#endif
    if (in_use)
        return "port " + std::to_string(port) + " is already in use - is Lamina (or another server) already running? "
               "Close it, or start this one with a different --port";
    std::string hint;
    if (denied)
        hint = "Windows may keep this port for itself (see `netsh int ipv4 show excludedportrange protocol=tcp`) or a "
               "firewall rule forbids it: start this one with a different --port";
    else if (no_addr)
        hint = host + " is not an address of this PC: check \"host\" in the model's lamina-<model>.json or --host";
    else
        hint = "check --host and --port";
    return "cannot listen on " + host + ":" + std::to_string(port) + ": " + what + " [" + std::to_string(code) + "]. " + hint;
}

int probe_listen(const std::string& host, int port) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    const std::string service = std::to_string(port);
    if (getaddrinfo(host.empty() ? nullptr : host.c_str(), service.c_str(), &hints, &res) != 0 || res == nullptr)
#ifdef _WIN32
        return WSAEADDRNOTAVAIL;
#else
        return EADDRNOTAVAIL;
#endif
    const sock_t s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    int code = 0;
    if (s == kNoSock) {
        code = last_error();
    } else {
#ifdef _WIN32
        // without SO_REUSEADDR a second server cannot bind a port that is already serving (on Windows that option lets
        // requests land on either one); exclusive use makes the second start fail loudly
        BOOL on = TRUE;
        setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char*) &on, sizeof on);
#else
        int on = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
#endif
        if (bind(s, res->ai_addr, (int) res->ai_addrlen) != 0 || listen(s, 1) != 0) code = last_error();
        close_sock(s);
    }
    freeaddrinfo(res);
    return code;
}

}  // namespace lamina
