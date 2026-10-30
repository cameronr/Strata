// src/server/telemetry.hpp - hardware readings for the web app's Monitor tab (serve/telemetry.py; idea from PR #22 by
// code-martin).
//
// A background thread samples once a second and keeps the last 60 readings of each series for the sparklines: the
// GPU through NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver, loaded at run
// time): load, VRAM, temperature, power, PCIe link and throughput; CPU load, RAM and the disks' read / write rates
// from the OS, as psutil reads them. Anything that cannot be read is null; nothing here can stop the server.
#pragma once

#include "json.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace lamina {

class Telemetry {
public:
    /// `extra()`: more series to record each second (the server's tok/s). `gpu_index`: the card the engine runs on,
    /// numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when the model is split
    /// across several (#112) - the gpu_* readings are then their total (memory, power, PCIe traffic), mean (load) or
    /// hottest (temperature), and "gpus" has each card's own.
    explicit Telemetry(std::function<json()> extra = nullptr, int gpu_index = 0, std::vector<int> gpu_indices = {});
    ~Telemetry();
    Telemetry(const Telemetry&) = delete;
    Telemetry& operator=(const Telemetry&) = delete;

    /// {"now": the last sample, "history": the last 60 readings of each series, "static": the machine's facts}
    json snapshot();
    /// ends the sampler thread (a server that stops) and waits for it, so extra() is not called after this returns
    /// (not to be called holding a lock extra() takes); the destructor closes too
    void close();
    /// free VRAM of a card in MiB, -1 when it cannot be read
    static int64_t free_vram_mib(int index = 0);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;   // shared with the sampler thread, which may outlive a close() made from extra()
};

}  // namespace lamina
