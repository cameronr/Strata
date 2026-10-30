// src/server/http_client.cpp - outgoing HTTP(S) requests (http_client.hpp).
#include "http_client.hpp"

#include "json.hpp"

#include <algorithm>
#include <memory>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>
#else
#include <httplib.h>
#endif

namespace lamina {

std::string HttpReply::header(std::string_view name) const {
    for (const auto& [k, v] : headers)
        if (k == name) return v;
    return "";
}

#ifdef _WIN32
namespace {

std::wstring widen(std::string_view s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}

std::string narrow(std::wstring_view w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

std::string win_error(const char* what) {
    const DWORD code = GetLastError();
    std::string text;
    if (code == ERROR_WINHTTP_TIMEOUT) text = "timed out";
    else if (code == ERROR_WINHTTP_NAME_NOT_RESOLVED) text = "the server's name could not be resolved";
    else if (code == ERROR_WINHTTP_CANNOT_CONNECT) text = "the connection was refused";
    else if (code == ERROR_WINHTTP_CONNECTION_ERROR) text = "the connection was reset";
    else if (code == ERROR_WINHTTP_SECURE_FAILURE) text = "the TLS connection could not be made";
    else if (code == ERROR_WINHTTP_UNRECOGNIZED_SCHEME || code == ERROR_WINHTTP_INVALID_URL) text = "not a valid URL";
    else text = "error " + std::to_string(code);
    return std::string(what) + ": " + text;
}

struct Handle {
    HINTERNET h = nullptr;
    explicit Handle(HINTERNET x) : h(x) {}
    ~Handle() {
        if (h != nullptr) WinHttpCloseHandle(h);
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};

}  // namespace

HttpReply http_request(const std::string& method, const std::string& url, const HttpHeaders& headers,
                       const std::string& body, double timeout_s, size_t max_bytes,
                       const std::function<bool(std::string_view)>& on_data) {
    const std::wstring wurl = widen(url);
    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof uc;
    uc.dwSchemeLength = uc.dwHostNameLength = uc.dwUrlPathLength = uc.dwExtraInfoLength = (DWORD) -1;
    if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc) || (uc.nScheme != INTERNET_SCHEME_HTTP && uc.nScheme != INTERNET_SCHEME_HTTPS))
        throw HttpError("not an http(s) URL: " + url);
    const std::wstring host(uc.lpszHostName, uc.dwHostNameLength);
    const std::wstring path = std::wstring(uc.lpszUrlPath, uc.dwUrlPathLength) +
                              std::wstring(uc.lpszExtraInfo != nullptr ? uc.lpszExtraInfo : L"", uc.dwExtraInfoLength);
    Handle session(WinHttpOpen(L"lamina", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                               WINHTTP_NO_PROXY_BYPASS, 0));
    if (session.h == nullptr) throw HttpError(win_error("the request could not be made"));
    const int ms = (int) std::clamp(timeout_s * 1000.0, 1.0, 2.0e9);
    WinHttpSetTimeouts(session.h, ms, ms, ms, ms);
    Handle connection(WinHttpConnect(session.h, host.c_str(), uc.nPort, 0));
    if (connection.h == nullptr) throw HttpError(win_error("the connection failed"));
    Handle request(WinHttpOpenRequest(connection.h, widen(method).c_str(), path.empty() ? L"/" : path.c_str(), nullptr,
                                      WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                      uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0));
    if (request.h == nullptr) throw HttpError(win_error("the request could not be made"));
    std::wstring head;
    for (const auto& [k, v] : headers) head += widen(k + ": " + v + "\r\n");
    if (!WinHttpSendRequest(request.h, head.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : head.c_str(), (DWORD) -1L,
                            body.empty() ? WINHTTP_NO_REQUEST_DATA : (LPVOID) body.data(), (DWORD) body.size(),
                            (DWORD) body.size(), 0) ||
        !WinHttpReceiveResponse(request.h, nullptr))
        throw HttpError(win_error("the request failed"));
    HttpReply reply;
    DWORD status = 0, size = sizeof status;
    WinHttpQueryHeaders(request.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                        &status, &size, WINHTTP_NO_HEADER_INDEX);
    reply.status = (int) status;
    size = 0;
    WinHttpQueryHeaders(request.h, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &size,
                        WINHTTP_NO_HEADER_INDEX);
    if (size > 0) {
        std::wstring raw(size / sizeof(wchar_t), L'\0');
        if (WinHttpQueryHeaders(request.h, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, raw.data(),
                                &size, WINHTTP_NO_HEADER_INDEX)) {
            const std::vector<std::string> lines = split(narrow(raw.c_str()), "\r\n");
            for (size_t i = 0; i < lines.size(); ++i) {
                if (i == 0) {   // the status line: "HTTP/1.1 404 Not Found"
                    const size_t a = lines[0].find(' '), b = a == std::string::npos ? a : lines[0].find(' ', a + 1);
                    if (b != std::string::npos) reply.reason = lines[0].substr(b + 1);
                    continue;
                }
                const size_t colon = lines[i].find(':');
                if (colon == std::string::npos) continue;
                reply.headers.emplace_back(lower(lines[i].substr(0, colon)), std::string(strip(lines[i].substr(colon + 1))));
            }
        }
    }
    size_t total = 0;
    std::string buf;
    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(request.h, &avail)) throw HttpError(win_error("the reply could not be read"));
        if (avail == 0) break;
        buf.resize(avail);
        DWORD got = 0;
        if (!WinHttpReadData(request.h, buf.data(), avail, &got)) throw HttpError(win_error("the reply could not be read"));
        if (got == 0) break;
        total += got;
        if (max_bytes > 0 && total > max_bytes) throw HttpError("the reply is larger than " + std::to_string(max_bytes) + " bytes");
        if (on_data) {
            if (!on_data(std::string_view(buf.data(), got))) break;
        } else {
            reply.body.append(buf.data(), got);
        }
    }
    return reply;
}

#else

HttpReply http_request(const std::string& method, const std::string& url, const HttpHeaders& headers,
                       const std::string& body, double timeout_s, size_t max_bytes,
                       const std::function<bool(std::string_view)>& on_data) {
    const size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) throw HttpError("not an http(s) URL: " + url);
    const size_t path_at = url.find('/', scheme_end + 3);
    const std::string base = url.substr(0, path_at), path = path_at == std::string::npos ? "/" : url.substr(path_at);
    httplib::Client cli(base);
    if (!cli.is_valid()) throw HttpError("not an http(s) URL (or no TLS support in this build): " + url);
    const time_t sec = (time_t) timeout_s;
    const time_t usec = (time_t) ((timeout_s - (double) sec) * 1e6);
    cli.set_connection_timeout(sec, usec);
    cli.set_read_timeout(sec, usec);
    cli.set_write_timeout(sec, usec);
    cli.set_follow_location(true);
    httplib::Headers h;
    for (const auto& [k, v] : headers) h.emplace(k, v);
    HttpReply reply;
    size_t total = 0;
    bool too_big = false, stopped = false;
    httplib::Request req;
    req.method = method;
    req.path = path;
    req.headers = h;
    req.body = body;
    req.response_handler = [&](const httplib::Response& r) {   // the status and headers, before the body
        reply.status = r.status;
        reply.reason = r.reason;
        for (const auto& [k, v] : r.headers) reply.headers.emplace_back(lower(k), v);
        return true;
    };
    req.content_receiver = [&](const char* data, size_t n, uint64_t, uint64_t) {
        total += n;
        if (max_bytes > 0 && total > max_bytes) {
            too_big = true;
            return false;
        }
        if (on_data) {
            stopped = !on_data(std::string_view(data, n));
            return !stopped;
        }
        reply.body.append(data, n);
        return true;
    };
    httplib::Result res = cli.send(req);
    if (too_big) throw HttpError("the reply is larger than " + std::to_string(max_bytes) + " bytes");
    if (!res && !stopped) throw HttpError("the request failed: " + httplib::to_string(res.error()));
    return reply;
}

#endif

}  // namespace lamina
