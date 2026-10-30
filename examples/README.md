# Example configs

The configs Lamina is developed with, on a PC with an RTX 3090, an RTX 5070 Ti, an RTX 5060 Ti and 160 GB of RAM
(their speeds: [COMPARISON.md](../docs/COMPARISON.md)). Setup writes each PC's own configs from them, with its GPUs
and data folder ([SETUP.md](../docs/SETUP.md)):

- `iq3_s.json`: Qwen3.8-Flash-Next, ISTA-DASLab's GSQ-RCO IQ3_S (~53 GB of RAM taken).
- `ud-q4_k_xl.json`: Qwen3.8-Flash-Next, unsloth's UD-Q4_K_XL (~81 GB of RAM taken).
- `qwen36-ud-q4_k_xl.json`: Qwen3.6-35B-A3B, unsloth's UD-Q4_K_XL, on the engine's Qwen3.6 path.

`run-<model>.bat` starts each from the clone's folder, for models in setup's data folder (`../Lamina-data`). Their
engine options: [SETUP.md](../docs/SETUP.md#engine-options); the server's keys: [USING.md](../docs/USING.md#the-config).

## The development PC's own options

The development PC's Flash-Next configs name their tiers (`--second-gpu "RTX 5070 Ti,RTX 5060 Ti"`, where the examples
take every other GPU with `auto`) and add three options measured for its hardware, which the examples leave out. Each
pays only where a PC has the same trait: measure it through the server before keeping it.

- **`--arena-place`** (both): for RAM whose regions read at different rates, as unpaired DIMMs make (here two 32 GB and
  two 48 GB modules leave 32 GB on one channel), the engine times its RAM arena at the start and keeps the experts the
  CPU reads out of the slow region, again between requests. Elsewhere it finds nothing to move and costs ~2 s at the
  start.
- **`--no-file-cache`** (UD-Q4_K_XL): the engine reads the experts from the GGUF past the OS file cache. Where the RAM
  holds the 72 GiB arena and the cached 72 GB file together (this PC's 160 GB), the file settles in the cache on pages
  the later starts' arenas would have had: here generation fell 7-8% over a session's starts, and with the option it
  stays at a fresh boot's speed. Each start then reads the experts from the drive (here 4.3-4.4 GiB/s against 6 from
  the cache, ~5 s more). With less RAM the cache cannot keep the file beside the arena, and the option costs only the
  start time.
- **`--prompt-split 26`** (both): a prompt of more than one chunk runs in two stages, the first tier taking each
  chunk's layers 26-47 whole while the main GPU runs the next chunk. It pays where the tier alone reads a chunk about as
  fast as both GPUs together: this PC's 5070 Ti runs the dense steps 1.44x as fast as its 3090, and prompts of
  32K-128K tokens read 1.13-1.55x as fast. It takes 0.7 GiB more pinned RAM and lends 6-8 GiB of the tier's expert
  cache while such a prompt runs. Its layer needs measuring too, through the server: here generate mode put IQ3_S's
  best layer at 23 (6-11% faster than 26), while through the API 26 was 2-4% faster than 23.
