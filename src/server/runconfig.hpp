// src/server/runconfig.hpp - #564: the web page's Settings view of the run config (lamina-<model>.json;
// serve/runconfig.py's port).
//
// A short list of documented keys can be read and changed from Lamina's own page (GET / POST /config). Everything
// else in the file - the keys setup writes, the network and security keys (host, api_key, cors_origins,
// trusted_origins, allowed_hosts), the MCP servers and before_load (which run programs), sampling keys not listed
// here - is kept as it is: a change touches only the keys it names (as setup run again keeps the user's keys, #629).
// The earlier file is kept as lamina-<model>.json.bak. The server reads the config when it starts, so a change is
// used from the next start on.
//
// A value or a file that is not what it must be throws RequestError (frontend.hpp, Python's ValueError: a 400); a
// file that cannot be read or written throws ConfigFileError (Python's OSError).
#pragma once

#include "frontend.hpp"
#include "json.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lamina::runconfig {

/// the run config cannot be read or written (Python's OSError), in Python's words: "[Errno 2] ...: 'path'"
struct ConfigFileError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// a key the Settings view lists
struct Editable {
    std::string_view key;
    std::string_view kind;   ///< "bool", "int>=0", "num>=0", "enum", "names", "sampling" or "arg"
    /// "sampling": the server's own rule for the key ("num>=0", "0<x<=1", "0<=x<=1", "1..64"); "arg": its flag
    std::string_view rule;
    std::vector<std::string_view> choices;   ///< "enum"
    std::string_view help;
};
/// the keys of the "sampling" block, the top-level keys, and an engine option kept in "args"
extern const std::vector<Editable> kEditable;

/// a listed key's value in `cfg` (null: not set, the default applies); std::out_of_range for a key not listed
json value_of(const json& cfg, std::string_view key);
/// GET /config: the editable keys with their values
json view(const json& cfg, const std::string& path);
/// the value to store for `key`, or a RequestError naming what is expected; null removes the key (its default)
json check(const std::string& key, const json& v, const json& cfg);
/// the config with `changes` ({key: value or null}) made, and the keys that changed. Every value is checked first,
/// so a request with one bad value changes nothing. Call it as runconfig::apply: unqualified, with json arguments,
/// argument-dependent lookup prefers std::apply
std::pair<json, std::vector<std::string>> apply(const json& cfg, const json& changes);
/// the config written whole (a temporary file moved over the old one), the earlier one kept as <name>.bak: its path
std::string save(const std::string& path, const json& cfg);
json load(const std::string& path);

}  // namespace lamina::runconfig
