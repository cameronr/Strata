// src/server/images.hpp - image generation (Qwen-Image-2.1-Turbo) through stable-diffusion.cpp's sd-cli
// (serve/images.py's port).
//
// POST /v1/images/generations takes OpenAI's request shape (prompt, size, n, response_format b64_json|url) plus
// Lamina's own fields seed and steps. Each request runs one sd-cli process with the arguments the model's card gives
// for stable-diffusion.cpp (8 steps, CFG 1, euler, Turbo's sigmas, --diffusion-fa) and returns the PNG files that
// process wrote, unchanged: an image from the API is byte for byte the image plain sd-cli writes with the same
// arguments.
//
// The GPU is shared with the text engine one job at a time: an image job takes the service's turn (the FIFO every chat
// request takes), so no chat turn runs on the GPU meanwhile, and the config's "gpu_policy" says what happens to the
// text engine during the job (the service's GPU turn):
//   keep    - it stays loaded and idle (an APU's shared memory, a card with room for both)
//   unload  - the engine is stopped; the next request (or, with "reload": "eager", the job's end) starts it again
//   auto    - (default) keep when there is room for the image job next to it (need_mib), else unload
// The process per request keeps nothing on the GPU between images: when the job ends, all of its memory is free again.
//
// The config's "images" section (written by setup --images, tools/image/install.py):
//   {"exe": "engine-image/sd-cli", "backend": "vulkan", "diffusion_model": "...Q8_0.gguf", "vae": "...",
//    "llm": "...Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf", "quant": "Q8_0", "args": ["--diffusion-fa"],
//    "gpu_policy": "auto", "reload": "lazy"|"eager", "max_n": 4, "max_side": 2048, "timeout_s": 3600,
//    "lib_dirs": [...], "uma": false, "need_mib": null, "out_dir": null, "keep_files": 64,
//    "native": false}   (native: Lamina's own DiT runner, tools/image/sdcpp_native; CUDA builds)
#pragma once

#include "json.hpp"
#include "process.hpp"
#include "sync.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lamina::images {

/// Qwen-Image-2.1-Turbo's own schedule (model_index.json "sample_sigmas") plus the terminal 0, as Unsloth's card passes
/// it to sd-cli; an 8-step schedule, so another step count leaves the schedule to sd.cpp
inline constexpr std::string_view kTurboSigmas = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0";
inline constexpr int64_t kTurboSteps = 8;
inline constexpr std::string_view kModelName = "qwen-image-2.1-turbo";
inline constexpr std::array<std::string_view, 3> kPolicies = {"auto", "keep", "unload"};
inline constexpr int64_t kSizeStep = 32;   ///< sd.cpp's Qwen-Image-2.1 docs: dimensions divisible by 32
inline constexpr int64_t kMinSide = 256;

/// a request the image endpoint refuses (400)
struct ImageRequestError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
/// sd-cli ended without the images (500); the message carries the end of its log
struct ImageJobFailed : std::runtime_error {
    explicit ImageJobFailed(const std::string& m) : std::runtime_error(m), message(m) {}
    std::string message;   ///< the whole message: what() ends at a NUL byte, which sd-cli's log can have
};
/// a config value of a type the Python failed on with a TypeError, AttributeError or KeyError (`kind`), which its server
/// answered as a malformed request: 400, "the request is malformed ({kind}: {what()})"
struct ConfigTypeError : std::invalid_argument {
    ConfigTypeError(std::string kind_, const std::string& message) : std::invalid_argument(message), kind(std::move(kind_)) {}
    std::string kind;
};

/// one job: parse_request's fields
struct Job {
    std::string prompt;
    int64_t width = 1024, height = 1024, n = 1, steps = kTurboSteps, seed = 0;
    std::string response_format = "b64_json";
};

/// OpenAI's size, "1024x1024" (or "auto", or none: 1024x1024) -> (width, height); each side a multiple of 32 from 256
/// to max_side
std::pair<int64_t, int64_t> parse_size(const json& value, int64_t max_side);
/// the fields of one /v1/images/generations request, checked (ImageRequestError). Unknown OpenAI fields (quality,
/// style, user, background, moderation) are accepted and ignored; an output format other than PNG is refused. No seed:
/// a random one below 2**31
Job parse_request(const json& req, const json& cfg);
/// the sd-cli command for one job: the model card's stable-diffusion.cpp command (8 steps, CFG 1, euler, Turbo's
/// sigmas, --diffusion-fa from the config's args), the size, seed and batch count, and the config's other args
std::vector<std::string> sd_cli_args(const json& cfg, const Job& job, const std::string& out_pattern);
/// sd-cli's environment (`base`: this process's when null): the libraries its backend needs (ROCm's, setup's
/// lib_dirs) first on the search path, the GPU to use when the config names one, and the config's own "env"
Environment image_env(const json& cfg, const Environment* base = nullptr);
/// the GPU memory (MiB) an image job is planned with: the three model files plus the working buffers (about 2 GiB plus
/// 4 GiB per megapixel at Turbo's settings; the 780M baseline peaked 7.4 GB over the weights at 1024x1024 on HIP); the
/// config's "need_mib" overrides it
int64_t need_mib(const json& cfg, int64_t width, int64_t height);

/// one sd-cli process per request, one request at a time (the service's turn)
class ImageGen {
public:
    /// the config's "images" section, checked (std::invalid_argument); `log`: the server's log file, which say()
    /// appends to ("": none)
    explicit ImageGen(const json& cfg, std::string log = "");

    /// exe, diffusion_model, vae and llm as the config names them, those that are not there
    std::vector<std::string> missing() const;
    /// a stored image (response_format url) by its name, or nullopt
    std::optional<std::filesystem::path> file(const std::string& name) const;
    /// runs sd-cli for `job` -> (the PNG files' bytes in order, the figures); the caller holds the GPU turn.
    /// ImageJobFailed when sd-cli fails, times out or is cancelled. `policy_used`: the policy the GPU turn used, the
    /// figures' "gpu_policy" (also in status()'s "last"; "": none)
    std::pair<std::vector<std::string>, json> run(const Job& job, Cancel* cancel = nullptr, const std::string& policy_used = "");
    /// keeps one PNG for response_format url -> its file name (32 random hex digits: the name is the access); the
    /// newest keep_files stay
    std::string store(const std::string& data);
    /// the /v1/images/generations answer: each image as base64, or (url) stored and named by its URL under base_url
    json response(const std::vector<std::string>& images, const Job& job, const json& figures, const std::string& base_url);
    /// /v1/status' "image_generation": the totals, the running job's step, the last job's figures
    json status() const;
    /// a line on the console ("[lamina] text") and in the log ("[images] text")
    void say(const std::string& text) const;

    json cfg;                    ///< the config's section
    json model;                  ///< the name it is listed under: "model_name", else kModelName
    std::string policy;          ///< gpu_policy: auto, keep or unload
    std::string reload;          ///< lazy or eager: when an unloaded engine starts again
    std::filesystem::path dir;   ///< the stored images and the jobs' folders: out_dir, else a new temporary folder
    int64_t keep_files = 64;
    std::string log;

private:
    void prune();

    mutable std::mutex state_mu_;   // totals_, last_, progress_
    std::mutex run_mu_;             // one sd-cli at a time even when the service's turn is not taken
    std::mutex files_mu_;           // store and prune
    json totals_ = {{"images", 0}, {"jobs", 0}, {"seconds", 0.0}, {"failed", 0}};
    json last_;                                              // the last job's figures (null: none yet)
    std::optional<std::pair<int64_t, int64_t>> progress_;    // (step, steps) of the running job
};

}  // namespace lamina::images
