// src/server/prometheus.hpp - GET /metrics in the Prometheus text format, with vLLM's metric names
// (serve/prometheus.py's port), so the dashboards and alerts written for a vLLM server read a Lamina server unchanged.
//
// Everything comes from the server's own record (Service::metrics(), the object the JSON /metrics returns) plus three
// latency histograms observed when a request finishes. What vLLM has no name for is the JSON's own facts under
// `lamina:`, named after the JSON's keys (live.tok_s -> lamina:live_tok_s, totals.decode_ms ->
// lamina:totals_decode_seconds_total, hardware.gpu_util -> lamina:gpu_util), so the two formats say the same thing
// and a panel can be moved from the Monitor tab to Grafana by its key.
#pragma once

#include "json.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace lamina::prometheus {

/// vLLM's buckets for the request latencies (seconds)
inline constexpr std::array<double, 22> kBuckets = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75,
                                                    1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0};
extern const char* const kContentType;

/// Prometheus asks with Accept: text/plain or application/openmetrics-text; a person can add ?format=prometheus.
/// Anything else (the Monitor tab, curl without a header) keeps the JSON.
bool wants_prometheus(std::string_view accept, std::string_view query);

class Histogram {
public:
    std::array<int64_t, kBuckets.size()> counts{};   ///< cumulative: observations <= each bucket
    double sum = 0.0;
    int64_t n = 0;

    /// `times` observations of v (none: nothing)
    void observe(std::optional<double> v, int64_t times = 1);
};

/// Time to first token, the time between tokens and the whole request, per finished request (thread-safe).
class Latencies {
public:
    void observe(std::optional<double> ttft_s, int64_t output_tokens, double decode_s, double total_s);
    /// {"ttft" | "itl" | "e2e": [counts, sum, n]}
    json snapshot() const;

private:
    mutable std::mutex mu_;
    Histogram ttft_, itl_, e2e_;
};

/// the text for one scrape: `m` is Service::metrics(), `lat` Latencies::snapshot()
std::string render(const json& m, const json& lat);

}  // namespace lamina::prometheus
