// src/core/gpu_select.cpp - see include/strata/core/gpu_select.hpp.
#include "strata/core/gpu_select.hpp"

#include <cuda_runtime.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <thread>

namespace strata::core {
namespace {

std::string lower(std::string s) {
    for (char& c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

std::string trimmed(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\"'"), b = s.find_last_not_of(" \t\"'");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

GpuInfo info_of(int ordinal) {
    GpuInfo g;
    g.ordinal = ordinal;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        (void) cudaGetLastError();
        return g;
    }
    g.name = p.name;
    g.sms = p.multiProcessorCount;
    char buf[64];
    std::snprintf(buf, sizeof buf, "%04x:%02x:%02x.0", (unsigned) p.pciDomainID, (unsigned) p.pciBusID,
                  (unsigned) p.pciDeviceID);
    g.bus_id = buf;
    const auto* u = (const unsigned char*) p.uuid.bytes;
    std::snprintf(buf, sizeof buf, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    g.uuid = buf;
    return g;
}

// NVML from the driver (nvml.dll, libnvidia-ml.so.1), loaded at the first use: whether a display is active on a GPU and
// how many processes hold graphics contexts there, and its PCIe link's generation now and at most (the lower of the
// card's and its root port's).  Without NVML every handle is null.
struct Nvml {
    using Init = int (*)();
    using ByBus = int (*)(const char*, void**);
    using Get = int (*)(void*, int*);
    using GetU = int (*)(void*, unsigned*);
    using Procs = int (*)(void*, unsigned*, void*);
    ByBus by_bus = nullptr;
    Get display = nullptr;
    GetU gen = nullptr, max_gen = nullptr;
    Procs graphics = nullptr;
    bool ok = false;
    Nvml() {
#if defined(_WIN32)
        HMODULE h = LoadLibraryA("nvml.dll");
        auto sym = [&](const char* n) { return h ? (void*) GetProcAddress(h, n) : nullptr; };
#else
        void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        auto sym = [&](const char* n) { return h ? dlsym(h, n) : nullptr; };
#endif
        const auto init = (Init) sym("nvmlInit_v2");
        by_bus = (ByBus) sym("nvmlDeviceGetHandleByPciBusId_v2");
        display = (Get) sym("nvmlDeviceGetDisplayActive");
        gen = (GetU) sym("nvmlDeviceGetCurrPcieLinkGeneration");
        max_gen = (GetU) sym("nvmlDeviceGetMaxPcieLinkGeneration");
        graphics = (Procs) sym("nvmlDeviceGetGraphicsRunningProcesses_v3");
        ok = init != nullptr && by_bus != nullptr && display != nullptr && gen != nullptr && max_gen != nullptr &&
             graphics != nullptr && init() == 0;
    }
    static Nvml& get() {
        static Nvml n;
        return n;
    }
    void* handle(const std::string& bus_id) const {
        void* d = nullptr;
        return ok && !bus_id.empty() && by_bus(bus_id.c_str(), &d) == 0 ? d : nullptr;
    }
};

// a link at its full generation: one the copies woke (idle, PCIe 1.0 or 2.0)
bool awake(void* h) {
    unsigned now = 0, most = 0;
    return h != nullptr && Nvml::get().gen(h, &now) == 0 && Nvml::get().max_gen(h, &most) == 0 && now >= most;
}

// The desktop's GPU, or another's in graphics use: a display active there or a process's graphics context (dwm.exe and
// the desktop's apps on Windows).  The display flag alone was off for a few minutes while the PC sat unattended.  False
// with `known` false when NVML cannot say.
bool desktop(void* h, bool& known) {
    int active = 0;
    unsigned procs = 0;
    const int r = h != nullptr ? Nvml::get().graphics(h, &procs, nullptr) : -1;
    // 7: NVML_ERROR_INSUFFICIENT_SIZE, processes to list
    known = h != nullptr && Nvml::get().display(h, &active) == 0 && (r == 0 || r == 7);
    return known && (active != 0 || r == 7 || procs > 0);
}

}  // namespace

std::vector<GpuInfo> list_gpus() {
    std::vector<GpuInfo> out;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        (void) cudaGetLastError();
        return out;
    }
    for (int i = 0; i < n; ++i) out.push_back(info_of(i));
    return out;
}

int find_gpu(const std::vector<GpuInfo>& gpus, const std::string& spec_in, std::string& why) {
    const std::string spec = trimmed(spec_in), ls = lower(spec);
    if (spec.empty()) {
        why = "no GPU named";
        return -1;
    }
    std::vector<const GpuInfo*> hits;
    const bool number =
        std::all_of(spec.begin(), spec.end(), [](char c) { return std::isdigit((unsigned char) c) != 0; });
    if (number) {   // an ordinal, else a part of a name ("3090")
        const int k = std::atoi(spec.c_str());
        for (const GpuInfo& g : gpus)
            if (g.ordinal == k) return k;
    }
    static const std::regex bus(R"(^(?:([0-9a-fA-F]+):)?([0-9a-fA-F]+):([0-9a-fA-F]+)(?:\.[0-9a-fA-F]+)?$)");
    std::smatch m;
    if (ls.rfind("gpu-", 0) == 0) {   // a UUID, or its start
        for (const GpuInfo& g : gpus)
            if (!g.uuid.empty() && lower(g.uuid).rfind(ls, 0) == 0) hits.push_back(&g);
    } else if (std::regex_match(spec, m, bus)) {   // [domain:]bus:device[.function], hexadecimal
        const unsigned long dom = m[1].matched ? std::strtoul(m[1].str().c_str(), nullptr, 16) : 0;
        const unsigned long b = std::strtoul(m[2].str().c_str(), nullptr, 16);
        const unsigned long d = std::strtoul(m[3].str().c_str(), nullptr, 16);
        for (const GpuInfo& g : gpus) {
            unsigned gd = 0, gb = 0, gv = 0;
            if (std::sscanf(g.bus_id.c_str(), "%x:%x:%x", &gd, &gb, &gv) == 3 && gd == dom && gb == b && gv == d)
                hits.push_back(&g);
        }
    } else {   // a name: a whole one first, then a part of one (a number only as a word of it: "3090", not "7")
        auto part = [&](const std::string& name) {
            for (size_t at = name.find(ls); at != std::string::npos; at = name.find(ls, at + 1)) {
                const size_t end = at + ls.size();
                if (!number || ((at == 0 || !std::isalnum((unsigned char) name[at - 1])) &&
                                (end == name.size() || !std::isalnum((unsigned char) name[end]))))
                    return true;
            }
            return false;
        };
        for (const GpuInfo& g : gpus)
            if (lower(g.name) == ls) hits.push_back(&g);
        if (hits.empty())
            for (const GpuInfo& g : gpus)
                if (part(lower(g.name))) hits.push_back(&g);
    }
    if (hits.size() == 1) return hits[0]->ordinal;
    if (hits.empty() && number) {
        why = "no visible CUDA device " + spec + " (" + std::to_string(gpus.size()) + " visible) or GPU named so";
    } else if (hits.empty()) {
        why = "no visible GPU is " + spec;
    } else {
        why = "several GPUs match " + spec + " (name one by its UUID):";
        for (const GpuInfo* g : hits) why += " " + std::to_string(g->ordinal) + " " + g->name + " " + g->uuid + ";";
        why.pop_back();
    }
    return -1;
}

std::vector<GpuProbe> probe_gpus(const std::vector<int>& devices, int main_device) {
    struct Dev {
        void* buf = nullptr;
        cudaStream_t s = nullptr;
        cudaEvent_t a = nullptr, b = nullptr;   // timing
        cudaEvent_t done[2] = {};               // the copies in flight, in turn
        long queued = 0;
        void* nvml = nullptr;
        bool ok = false, up = false;
    };
    const size_t n = devices.size();
    constexpr size_t kBytes = 64ull << 20;
    std::vector<GpuProbe> out(n);
    std::vector<Dev> d(n);
    const Nvml& nvml = Nvml::get();
    void* host = nullptr;
    const bool pinned = cudaHostAlloc(&host, kBytes, cudaHostAllocPortable) == cudaSuccess;
    bool all_known = true;   // every GPU's P-state readable
    for (size_t i = 0; i < n; ++i) {
        Dev& x = d[i];
        x.nvml = nvml.handle(info_of(devices[i]).bus_id);
        out[i].display = desktop(x.nvml, out[i].display_known) || !out[i].display_known;
        x.ok = pinned && cudaSetDevice(devices[i]) == cudaSuccess && cudaMalloc(&x.buf, kBytes) == cudaSuccess &&
               cudaStreamCreateWithFlags(&x.s, cudaStreamNonBlocking) == cudaSuccess &&
               cudaEventCreate(&x.a) == cudaSuccess && cudaEventCreate(&x.b) == cudaSuccess &&
               cudaEventCreateWithFlags(&x.done[0], cudaEventDisableTiming) == cudaSuccess &&
               cudaEventCreateWithFlags(&x.done[1], cudaEventDisableTiming) == cudaSuccess;
        all_known = all_known && (!x.ok || x.nvml != nullptr);
    }
    // The links woken, all at once: an idle one runs at PCIe 1.0 or 2.0 rates until 0.2-0.5 s of copies raise its
    // P-state (P8, or a display's P5), and its link with it.  Two copies in flight on each, a new one as soon as one is
    // done: with the link idle between pairs of copies, for no more than the 2 ms between the polls, the 5070 Ti stayed
    // in P5.  Until each link is at its full generation, at most 1 s; without NVML 0.6 s.  (Waiting for a P-state of P2
    // or faster ran into the 1 s limit on cold starts, the 5070 Ti's link at PCIe 5.0 by then.)
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        bool up = true;
        for (size_t i = 0; i < n; ++i) {
            Dev& x = d[i];
            if (!x.ok) continue;
            cudaSetDevice(devices[i]);
            while (x.queued < 2 || cudaEventQuery(x.done[x.queued % 2]) == cudaSuccess) {
                cudaMemcpyAsync(x.buf, host, kBytes, cudaMemcpyHostToDevice, x.s);
                cudaEventRecord(x.done[x.queued % 2], x.s);
                ++x.queued;
            }
            x.up = x.up || awake(x.nvml);
            up = up && x.up;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if ((all_known && up) || ms >= (all_known ? 1000.0 : 600.0)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    for (Dev& x : d)
        if (x.ok) cudaStreamSynchronize(x.s);
    // then each link alone: the best of five copies
    for (size_t i = 0; i < n; ++i) {
        Dev& x = d[i];
        for (int r = 0; x.ok && r < 5; ++r) {
            float ms = 0;
            cudaSetDevice(devices[i]);
            cudaEventRecord(x.a, x.s);
            cudaMemcpyAsync(x.buf, host, kBytes, cudaMemcpyHostToDevice, x.s);
            cudaEventRecord(x.b, x.s);
            if (cudaEventSynchronize(x.b) == cudaSuccess && cudaEventElapsedTime(&ms, x.a, x.b) == cudaSuccess &&
                ms > 0)
                out[i].h2d_gbps = std::max(out[i].h2d_gbps, (double) kBytes / (double) ms / 1e6);
        }
        cudaSetDevice(devices[i]);
        for (cudaEvent_t e : {x.a, x.b, x.done[0], x.done[1]})
            if (e) cudaEventDestroy(e);
        if (x.s) cudaStreamDestroy(x.s);
        if (x.buf) cudaFree(x.buf);
    }
    if (pinned) cudaFreeHost(host);
    (void) cudaGetLastError();
    cudaSetDevice(main_device);
    return out;
}

}  // namespace strata::core
