# This fork's configs

The configs this fork runs with on a PC with an RTX 3090, an RTX 5070 Ti and 160 GB of RAM, for `serve/server.py`
(their speeds: [COMPARISON.md](../docs/COMPARISON.md)):

- `iq3_s.json`: ISTA-DASLab's GSQ-RCO IQ3_S (~53 GB of RAM taken).
- `ud-q4_k_xl.json`: unsloth's UD-Q4_K_XL (~81 GB of RAM taken).

This branch's engine leaves out some of upstream's options, which [DETAILS.md](../docs/DETAILS.md) describes and setup
may write into its own configs: the layer split and the helper GPUs' caches (`--peer-device`; the second GPU here is
`--second-gpu`), KV streaming (`--kv-resident`: the KV cache stays in VRAM), the VRAM resize (`--vram-elastic`), the
low-RAM modes, the file tier and the RAM budget (`--resident-budget-gib`), the experimental speed projection (control
vectors), upstream's conversation cache (`--conversation-cache-mib`, `--conversation-cache-park-gain`; the prompt cache
here keeps replaced conversations in RAM) and its checkpoint spread (`--prompt-cache-spread`), session files, parallel
requests (`--batch`), token biases, prompt lookup chained after the drafts (`--lookup-chain`), upstream's routing
approximations (the route tail skip, on by default in upstream's CUDA server), the adaptive tier's overlapped copies
(`STRATA_ADAPT_OVERLAP`), the CPU pool's `--host-core` and `--pool-affinity`, the engine's own watchdog and stall
report, the PLE read probe, 0.1.44's NVFP4 experts, Qwen3.6-35B-A3B engine and batched commit graph, 0.1.45's CUDA
opt-in switches but `STRATA_DEQUANT_WIDE`, the engine of 0.2.0-beta (0.1.45.3): the switches on by default per
architecture (`STRATA_SM86_PERF`, `STRATA_SM120_PERF`), agentionai's Gyro files (trellis expert types, Hadamard-folded
experts), the CPU pool's planned worker count, the pinned-memory reserve, the prompt planner and the Linux file tier's
read-ahead, and the AMD, Intel, Metal (Mac), Turing, Volta and Pascal builds. A
server config with `parallel` or `vram_elastic` passes the engine a flag it refuses; with `slot_save_path`, a save
request ends the engine.

The configs keep what each model's run learned (`expert_profile_save`, upstream's #477): the engine writes a profile
(`<model>-learned.bin` in the Strata folder) at exit, on Ctrl+C in the server's window and between requests at most once
a minute, and the next start fills both GPUs' expert caches from it instead of the shipped profile. It ranks the
experts by this run's routing and the profile it started from at equal weight. Against the shipped profile, after a
session of other code and reasoning requests, a round of fixed text took 9-14% less time on code, -3.8 to +0.9% on
thinking and edits, 0 to +2.0% after a 16K-token prompt. Remove the two keys to start from the shipped profile every
time.

A request that sends no sampling fields is sampled with the configs' `sampling` block: temperature 0.7, top_p 0.95,
top_k 20, min_p 0, presence_penalty 0, with thinking on or off (the model card's thinking settings, with its temperature
of 1.0 lowered to 0.7). The server fills what a config leaves out from the model card's set for the request's thinking
state, so the block's presence_penalty 0 keeps the card's 1.5 off the answers without thinking. A request's own fields
win, and `"temperature": 0` decodes greedy. Sampled generation's speed against greedy:
[COMPARISON.md](../docs/COMPARISON.md).

Their paths are relative to the Strata folder, where the server starts: the engine in `build/`; the model files, the
packs and the MTP draft layer in setup's data folder, `../Strata-data` (unless setup was given another: `--data-dir`).
A pack is what `tools/iq_pack.py` makes from a model's GGUF files in seconds (~1.5 GB): the tokenizer with the chat
template, the model's float tensors in the forms the engine takes, and an index of the rest; the engine reads the
quantized tensors, the experts and most projections, from the GGUF files at start.

1. Set up IQ3_S once: `START-HERE.bat --setup --model IQ3_S --no-start` downloads the model into
   `../Strata-data/models/IQ3_S`, makes its pack in `../Strata-data/packs/iq3_s`, and makes the MTP draft layer in
   `../Strata-data/mtp/rt` from the ~5 GB of MTP tensors of Qwen's original checkpoint. The configs' draft layer is
   `../Strata-data/mtp/rt-tuned`: a copy of `mtp/rt` with its `dense.bin` replaced by the self-distilled one published
   with jesdga95's basalt fork ([jesdga/Qwen3.8-Flash-Next-Basalt](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt),
   `convert/mtp/rt/dense.bin`, sha256 `dfe18647e0c95d05302907be88f021590119645a1567fcd770c8d309df852d75`, under Qwen's
   license). It drafts better (0.2 more tokens per verify pass on thinking text and after long prompts: 4-6% faster
   there), and the output stays the same: every token is verified.
2. Build the engine from this branch with code for both cards (the ready-made engine setup puts in `engine/` is
   upstream's, without this fork's options), with MSVC, CUDA 13, CMake and Ninja (setup puts the last two in
   `.venv\Scripts`):

       cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF ^
         "-DCMAKE_CUDA_ARCHITECTURES=86;120"
       cmake --build build --target strata

   `86;120` is an RTX 30 card and an RTX 50 card. The engine needs the CUDA runtime DLLs (`cudart64_13.dll`,
   `cublas64_13.dll`, `cublasLt64_13.dll`) beside it, on the PATH, or their folder in the config's `lib_dirs`.
3. For UD-Q4_K_XL, download unsloth's four `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf` files into
   `../Strata-data/models/UD-Q4_K_XL` and make its pack (`--compat-bf16`: the file stores the hyper-connection
   projections and the PLE value in Q8_0, which the pack holds rounded to BF16, the form the engine reads):

       .venv\Scripts\python.exe tools\iq_pack.py --compat-bf16 --out ../Strata-data/packs/ud-q4_k_xl ^
         --gguf ../Strata-data/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

4. Images: the configs' `vision` section starts the image encoder (`strata-vision`, llama.cpp's mtmd library with the
   model's projector) on the CPU beside the engine, so both GPUs' expert caches keep all of their VRAM. Build it
   without CUDA, for any AVX2 CPU (the default):

       cmake -G Ninja -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release
       cmake --build build-vision --target strata-vision

   (a native build on an AVX-512 CPU, `-DSTRATA_PORTABLE=OFF`, runs without flash attention, about twice as slow on
   this PC's Ryzen 7 9700X). Its projector, the same for both models: ISTA-DASLab's
   `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (0.9 GB; setup's `--vision` downloads it into IQ3_S's folder) in Q8_0, which
   encodes ~20% faster on the CPU, made by llama.cpp's `llama-quantize`:

       llama-quantize mmproj-Qwen3.8-Flash-Next-BF16.gguf ../Strata-data/packs/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf Q8_0

   A picture of ~1,000 tokens (`max_tokens`: up to 1,024) takes ~9 s to encode on the 9700X (`threads`: its 16 logical
   cores), once per picture, and the encoder ~1 GB of RAM. Without images, remove the section and `--vision`.
5. Name the GPUs. `--main-gpu` runs the model (`RTX 3090` here), and nothing else may use that card: a display or
   another program on it slows generation to a few tokens per second. Name a card by its name or a part only its name
   has (`RTX 3090`), its UUID (`nvidia-smi -L` lists them), its PCI bus id, or its number as `nvidia-smi` numbers the
   cards; a number moves when a card is added. `--second-gpu auto` makes every other card a further expert tier, up
   to three, the fastest link first (a copy test at startup; the log shows each card's rate): each holds more experts
   and may drive the display. A card the desktop uses keeps 2048 MiB free, a headless one 512
   (`--second-gpu-reserve-mib` sets them). To fix the tiers and their order, name them:
   `--second-gpu "RTX 5070 Ti,RTX 5060 Ti"`.
6. Start a config with its launcher, `run-iq3_s.bat` or `run-ud-q4_k_xl.bat`, or from the Strata folder:

       .venv\Scripts\python.exe serve\server.py --engine strata --config examples\iq3_s.json

   The server listens on the config's `port` (8081 and 8082; `--port` overrides it).

   (`START-HERE.bat` starts setup's own config.)

Both of this PC's configs also have `--arena-place`, left out here: for RAM whose regions read at different rates,
as unpaired DIMMs make (two 32 GB and two 48 GB modules), the engine times its RAM arena at the start and keeps the
experts the CPU reads out of the slow region, again between requests ([DETAILS.md](../docs/DETAILS.md)). On such a
PC add it to a config's arguments; elsewhere it finds nothing to move and costs ~2 s at the start.

This PC's UD-Q4_K_XL config also has `--no-file-cache`, left out here: the engine reads the experts from the GGUF past
the OS file cache. Where the RAM holds the 72 GiB arena and the cached 72 GB file together (this PC's 160 GB), the
file settles in the cache on pages the later starts' arenas would have had: here generation fell 7-8% over a session's
starts, and with the flag it stays at a fresh boot's speed. Each start then reads the experts from the drive (here
4.3-4.4 GiB/s against 6 from the cache, ~5 s more). With less RAM the cache cannot keep the file beside the arena, and
the flag costs only the start time ([DETAILS.md](../docs/DETAILS.md)).

Both of this PC's configs also have `--prompt-split 26`, left out here: a prompt of more than one chunk runs in two
stages, the second GPU taking each chunk's layers 26-47 whole while the first runs the next chunk. It pays where the
second GPU alone reads a chunk about as fast as both together: this PC's 5070 Ti runs the dense steps 1.44x as fast as
its 3090, and prompts of 32K-128K tokens read 1.13-1.55x as fast. It costs 0.7 GiB more pinned RAM and lends 6-8 GiB
of the second GPU's cache during such a prompt ([DETAILS.md](../docs/DETAILS.md)). Measure it on your pair before
adding it, and its layer too, through the server: here generate mode put IQ3_S's best layer at 23 (6-11% faster
than 26), while through the API 26 was 2-4% faster than 23.

On Linux (built and run on CachyOS with GCC 16.2.1 and CUDA 13.4, on two RTX 3090 and an RTX 5060 Ti) the steps are
the same, with these differences:

- Build with `.venv/bin/cmake` and `.venv/bin/ninja` (`\` for the line breaks, `"-DCMAKE_CUDA_ARCHITECTURES=86;120"`
  quoted for the shell). The engine is then `./build/strata` and the image encoder `./build-vision/bin/strata-vision`:
  the configs' two `exe` paths. Python is `.venv/bin/python`.
- The CUDA libraries come from the toolkit through the loader's path, or from the config's `lib_dirs`, which the
  server puts on `LD_LIBRARY_PATH`.
- Start a config from the Strata folder with `.venv/bin/python serve/server.py --engine strata --config
  examples/iq3_s.json` (without `--engine strata` the server starts its mock engine); the `.bat` launchers
  have no Linux counterpart.
