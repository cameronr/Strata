// src/server/prometheus.cpp - GET /metrics in the Prometheus text format (serve/prometheus.py's port): vLLM's names
// read from the server's record (Service::metrics()) and the three latency histograms, then under `lamina:` the JSON's
// facts vLLM has no name for. Every value prints as the Python prints it: ints as ints, floats as repr, round() as
// Python rounds (the decimal nearest to the exact value, ties to even).
#include "prometheus.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>

namespace lamina::prometheus {

const char* const kContentType = "text/plain; version=0.0.4; charset=utf-8";

namespace {

const json kNull;

template <typename... T>
void put(std::string& out, const T&... parts) {
    (out.append(parts), ...);
}

bool is_str(const json& v, std::string_view s) { return v.is_string() && v.get_ref<const std::string&>() == s; }

/// f"{v}": str() of a value. A list or an object (which Service::metrics() never puts where a label or a value is
/// read) prints as JSON, not as Python's repr
std::string py_str(const json& v) {
    switch (v.type()) {
        case json::value_t::null: return "None";
        case json::value_t::boolean: return v.get<bool>() ? "True" : "False";
        case json::value_t::number_integer: return std::to_string(v.get<int64_t>());
        case json::value_t::number_unsigned: return std::to_string(v.get<uint64_t>());
        case json::value_t::number_float: return py_repr(v.get<double>());
        case json::value_t::string: return v.get<std::string>();
        default: return py_dumps(v);
    }
}

/// round(v, nd) on a JSON number: an int stays itself
json round_json(const json& v, int nd) { return v.is_number_float() ? json(py_round(v.get<double>(), nd)) : v; }

/// _num: the number, or `dflt` for anything else (a bool too)
json num(const json& v, const json& dflt = 0) { return is_num(v) ? v : dflt; }

bool as_i64(const json& v, int64_t& out) {
    if (v.is_number_unsigned()) {
        const uint64_t u = v.get<uint64_t>();
        out = (int64_t) u;
        return u <= (uint64_t) std::numeric_limits<int64_t>::max();
    }
    out = v.is_number_integer() ? v.get<int64_t>() : 0;
    return v.is_number_integer();
}

/// a + b on two numbers: an int for two ints, else a float (beyond 64 bits too, where Python's ints go on)
json add(const json& a, const json& b) {
    int64_t x, y;
    constexpr int64_t hi = std::numeric_limits<int64_t>::max(), lo = std::numeric_limits<int64_t>::min();
    if (as_i64(a, x) && as_i64(b, y) && !(y > 0 && x > hi - y) && !(y < 0 && x < lo - y)) return x + y;
    return a.get<double>() + b.get<double>();
}

/// `(ms or 0) / 1000` rounded to the millisecond; no sample for a value that is no number (Python: a TypeError)
json seconds(const json& ms) {
    if (!truthy(ms)) return py_round(0.0, 3);
    if (ms.is_boolean()) return py_round(1.0 / 1000, 3);
    return is_num(ms) ? json(py_round(ms.get<double>() / 1000, 3)) : kNull;
}

/// the JSON's facts vLLM has no name for, under `lamina:` and the JSON's key
void lamina_metrics(const json& m, std::string& out, const std::string& lab) {
    const json& live = jget(m, "live");
    const json& totals = jget(m, "totals");
    const json& hw = jget(m, "hardware");
    const json& eng = jget(m, "engine");
    const json& reqs = jget(m, "requests");
    std::set<std::string_view> typed;
    auto metric = [&](std::string_view name, std::string_view kind, std::string_view help, const json& value,
                      const std::string& labels = "") {
        if (!is_num(value)) return;          // unknown here (no GPU telemetry, an older engine): no sample
        if (typed.insert(name).second)       // one HELP / TYPE per family, before its first sample
            put(out, "# HELP lamina:", name, " ", help, "\n# TYPE lamina:", name, " ", kind, "\n");
        put(out, "lamina:", name, "{", lab, labels, "} ", py_str(value), "\n");
    };
    auto or0 = [](const json& v) { return truthy(v) ? v : json(0); };
    auto state_label = [](std::string_view s) { return ",state=\"" + std::string(s) + "\""; };

    for (const std::string_view state : {"unloaded", "idle", "reading", "generating"})
        metric("live_state", "gauge", "1 for what the engine is doing (live.state).",
               (int) is_str(jget(live, "state"), state), state_label(state));
    metric("live_tok_s", "gauge", "Decode rate over the last seconds (live.tok_s).", or0(jget(live, "tok_s")));
    metric("live_prefill_tok_s_mean", "gauge", "Prompt reading rate of the running request (live.prefill_tok_s_mean).",
           or0(jget(live, "prefill_tok_s_mean")));
    metric("live_prompt_read", "gauge", "Prompt tokens read so far by the running request (live.prompt_read).",
           or0(jget(live, "prompt_read")));
    const json& slots = jget(live, "slots");
    for (const std::string_view state : {"idle", "reading", "generating"}) {
        if (!slots.is_array() || slots.empty()) continue;
        int64_t k = 0;
        for (const auto& x : slots)
            if (x.is_object() && is_str(jget(x, "state"), state)) ++k;
        metric("live_slots", "gauge", "The batch slots in each state (live.slots).", k, state_label(state));
    }
    metric("engine_max_context", "gauge", "The engine's context (engine.max_context).", jget(eng, "max_context"));
    metric("totals_prompt_seconds_total", "counter", "Time spent reading prompts (totals.prompt_ms).",
           seconds(jget(totals, "prompt_ms")));
    metric("totals_decode_seconds_total", "counter", "Time spent generating (totals.decode_ms).",
           seconds(jget(totals, "decode_ms")));
    const json& last = reqs.is_array() && !reqs.empty() ? reqs[0] : kNull;
    metric("last_hit_rate", "gauge", "The last request's expert cache hit rate (requests[0].hit_rate).",
           jget(last, "hit_rate"));
    metric("last_decode_tok_s", "gauge", "The last request's decode rate (requests[0].decode_tok_s).",
           jget(last, "decode_tok_s"));
    json gpus = jget(hw, "gpus");
    if (!truthy(gpus)) {   // an older telemetry: one card's figures at the top level
        gpus = json::array();
        if (!jget(hw, "gpu_util").is_null())
            gpus.push_back(json{{"index", 0}, {"util", jget(hw, "gpu_util")}, {"mem_used", jget(hw, "gpu_mem_used")},
                                {"mem_total", jget(hw, "gpu_mem_total")}, {"temp", jget(hw, "gpu_temp")},
                                {"power", jget(hw, "gpu_power")}});
    }
    static constexpr std::array<std::array<std::string_view, 3>, 5> families = {{
        {"gpu_util", "util", "GPU busy, percent (hardware.gpu_util)."},
        {"gpu_mem_used_bytes", "mem_used", "GPU memory in use (hardware.gpu_mem_used)."},
        {"gpu_mem_total_bytes", "mem_total", "GPU memory (hardware.gpu_mem_total)."},
        {"gpu_temp_celsius", "temp", "GPU temperature (hardware.gpu_temp)."},
        {"gpu_power_watts", "power", "GPU power draw (hardware.gpu_power)."},
    }};
    if (gpus.is_array())
        for (const auto& [name, key, help] : families)
            for (const auto& g : gpus)   // a family's samples together, one per card
                metric(name, "gauge", help, jget(g, key), ",gpu=\"" + py_str(jget(g, "index")) + "\"");
    metric("cpu", "gauge", "CPU busy, percent (hardware.cpu).", jget(hw, "cpu"));
    metric("ram_used_bytes", "gauge", "RAM in use (hardware.ram_used).", jget(hw, "ram_used"));
    metric("ram_total_bytes", "gauge", "RAM (hardware.ram_total).", jget(hw, "ram_total"));
}

}  // namespace

bool wants_prometheus(std::string_view accept, std::string_view query) {
    return query.find("format=prometheus") != std::string_view::npos ||
           accept.find("openmetrics") != std::string_view::npos || accept.find("text/plain") != std::string_view::npos;
}

void Histogram::observe(std::optional<double> v, int64_t times) {
    if (!v || times <= 0) return;
    for (size_t i = 0; i < kBuckets.size(); ++i)
        if (*v <= kBuckets[i]) counts[i] += times;
    sum += *v * (double) times;
    n += times;
}

void Latencies::observe(std::optional<double> ttft_s, int64_t output_tokens, double decode_s, double total_s) {
    std::lock_guard<std::mutex> lk(mu_);
    ttft_.observe(ttft_s);
    if (output_tokens > 1 && decode_s != 0.0)
        // the mean gap, once per gap: the engine reports the decode time, not each token's
        itl_.observe(decode_s / (double) (output_tokens - 1), output_tokens - 1);
    e2e_.observe(total_s);
}

json Latencies::snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    const std::pair<const char*, const Histogram*> hs[] = {{"ttft", &ttft_}, {"itl", &itl_}, {"e2e", &e2e_}};
    json out = json::object();
    for (const auto& [key, h] : hs) out[key] = json::array({json(h->counts), h->sum, h->n});
    return out;
}

std::string render(const json& m, const json& lat) {
    const json& eng = jget(m, "engine");
    const json& live = jget(m, "live");
    const json& totals = jget(m, "totals");
    const json& model = jget(eng, "model");
    const std::string model_name = truthy(model) ? py_str(model) : "lamina";
    const std::string lab = "model_name=\"" + replace_all(replace_all(model_name, "\\", "\\\\"), "\"", "\\\"") + "\"";
    std::string out;
    auto metric = [&](std::string_view name, std::string_view kind, std::string_view help, const json& value) {
        put(out, "# HELP ", name, " ", help, "\n# TYPE ", name, " ", kind, "\n", name, "{", lab, "} ", py_str(value),
            "\n");
    };

    const json& state = jget(live, "state");
    const bool busy = is_str(state, "reading") || is_str(state, "generating");
    const json ctx = num(jget(eng, "max_context"), 0);
    const json used = busy ? add(num(jget(live, "prompt_tokens")), num(jget(live, "generated"))) : json(0);
    // "parallel": live.running / live.waiting count the requests together (waiting: for a slot or the control lines)
    metric("vllm:num_requests_running", "gauge", "Requests reading their prompt or generating.",
           num(jget(live, "running"), (int) busy));
    metric("vllm:num_requests_waiting", "gauge", "Requests waiting for their turn.",
           add(num(jget(live, "queued")), num(jget(live, "waiting"))));
    const double share = truthy(ctx) ? used.get<double>() / ctx.get<double>() : 0.0;
    metric("vllm:kv_cache_usage_perc", "gauge", "The running request's share of the context (1 = full).",
           truthy(ctx) ? json(py_round(share < 1.0 ? share : 1.0, 4)) : json(0));
    metric("vllm:prompt_tokens_total", "counter", "Prompt tokens of the finished requests.",
           num(jget(totals, "prompt_tokens")));
    metric("vllm:generation_tokens_total", "counter", "Generated tokens.", num(jget(totals, "output_tokens")));
    metric("vllm:request_success_total", "counter", "Finished requests.", num(jget(totals, "requests")));
    metric("vllm:prefix_cache_queries_total", "counter", "Prompt tokens looked up in the prompt cache.",
           num(jget(totals, "prompt_tokens")));
    metric("vllm:prefix_cache_hits_total", "counter", "Prompt tokens the prompt cache already held.",
           num(jget(totals, "reused")));
    metric("vllm:spec_decode_num_draft_tokens_total", "counter", "MTP draft tokens offered.",
           num(jget(totals, "drafts_offered")));
    metric("vllm:spec_decode_num_accepted_tokens_total", "counter", "MTP draft tokens accepted.",
           num(jget(totals, "drafts_accepted")));
    metric("vllm:num_preemptions_total", "counter", "Preempted requests (Lamina does not preempt).", 0);
    static constexpr std::array<std::array<std::string_view, 3>, 3> histograms = {{
        {"vllm:time_to_first_token_seconds", "ttft", "Time to the first token."},
        {"vllm:inter_token_latency_seconds", "itl", "Time between two tokens."},
        {"vllm:e2e_request_latency_seconds", "e2e", "A request from start to end."},
    }};
    for (const auto& [name, key, help] : histograms) {
        const json& h = lat.at(std::string(key));   // [counts, sum, n]
        const json& counts = h.at(0);
        const std::string n = py_str(h.at(2));
        put(out, "# HELP ", name, " ", help, "\n# TYPE ", name, " histogram\n");
        for (size_t i = 0; i < kBuckets.size() && i < counts.size(); ++i)
            put(out, name, "_bucket{", lab, ",le=\"", py_repr(kBuckets[i]), "\"} ", py_str(counts[i]), "\n");
        put(out, name, "_bucket{", lab, ",le=\"+Inf\"} ", n, "\n");
        put(out, name, "_sum{", lab, "} ", py_str(round_json(h.at(1), 6)), "\n", name, "_count{", lab, "} ", n, "\n");
    }
    lamina_metrics(m, out, lab);
    return out;
}

}  // namespace lamina::prometheus
