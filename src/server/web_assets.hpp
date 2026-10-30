// src/server/web_assets.hpp - the web app's files, built into lamina-server (cmake/embed_web.cmake).
#pragma once

#include <string_view>

namespace lamina {

struct WebAsset {
    std::string_view name;   ///< its path below src/server/web ("index.html", "fonts/outfit-latin-wght.woff2")
    std::string_view data;
};

/// the file, or nullptr
const WebAsset* web_asset(std::string_view name);

}  // namespace lamina
