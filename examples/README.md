# This fork's configs

The configs this fork runs with on a PC with an RTX 3090, an RTX 5070 Ti, an RTX 5060 Ti and 160 GB of RAM, for
`serve/server.py`
(their speeds: [COMPARISON.md](../docs/COMPARISON.md)). Setup writes each PC's own configs from them
([below](#setup)):

- `iq3_s.json`: ISTA-DASLab's GSQ-RCO IQ3_S (~53 GB of RAM taken).
- `ud-q4_k_xl.json`: unsloth's UD-Q4_K_XL (~81 GB of RAM taken).
- `qwen36-ud-q4_k_xl.json`: Qwen3.6-35B-A3B, unsloth's UD-Q4_K_XL, on the engine's Qwen3.6 path
  ([below](#qwen36-35b-a3b)).

This branch's engine leaves out some of upstream's options, which [DETAILS.md](../docs/DETAILS.md) describes: the
layer split and the helper GPUs' caches (`--peer-device`; the second GPU here is
`--second-gpu`), KV streaming (`--kv-resident`: the KV cache stays in VRAM), the VRAM resize (`--vram-elastic`), the
low-RAM modes, the file tier and the RAM budget (`--resident-budget-gib`), the experimental speed projection (control
vectors), upstream's conversation cache (`--conversation-cache-mib`, `--conversation-cache-park-gain`; the prompt cache
here keeps replaced conversations in RAM) and its checkpoint spread (`--prompt-cache-spread`), session files, parallel
requests (`--batch`), token biases, prompt lookup chained after the drafts (`--lookup-chain`), upstream's routing
approximations (the route tail skip, on by default in upstream's CUDA server), the adaptive tier's overlapped copies
(`STRATA_ADAPT_OVERLAP`), the CPU pool's `--host-core` and `--pool-affinity`, the engine's own watchdog and stall
report, the PLE read probe, 0.1.44's NVFP4 experts and batched commit graph, 0.1.45's CUDA
opt-in switches but `STRATA_DEQUANT_WIDE`, the engine of 0.2.0-beta (0.1.45.3): the switches on by default per
architecture (`STRATA_SM86_PERF`, `STRATA_SM120_PERF`), agentionai's Gyro files (trellis expert types, Hadamard-folded
experts), the CPU pool's planned worker count, the pinned-memory reserve, the prompt planner and the Linux file tier's
read-ahead, and the AMD, Intel, Metal (Mac), Turing, Volta and Pascal builds. A
Flash-Next config with `parallel` or `vram_elastic` passes the engine a flag it refuses; with `slot_save_path`, a save
request ends the engine.

The configs keep what each model's run learned (`expert_profile_save`, upstream's #477): the engine writes a profile
(`<model>-learned.bin`: in the Lamina folder for these configs, in the data folder for setup's) at exit, on Ctrl+C in
the server's window and between requests at most once a minute, and the next start fills both GPUs' expert caches from
it instead of the shipped profile. It ranks the
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

## Setup

`setup.bat --model iq3_s` (Windows) or `./setup.sh --model iq3_s` (Linux) prepares what a config needs, from this
checkout, and writes `lamina-iq3_s.json` and its launcher `run-iq3_s.bat` (`run-iq3_s.sh`) beside it, from
`iq3_s.json` with this PC's GPUs and data folder. `--model ud-q4_k_xl` and `--model qwen36-ud-q4_k_xl` set up the
others, `--check` says what the PC has and what a model needs without changing anything, and `--help` lists the rest
(`--data-dir`, `--vision off`, `--main-gpu`, `--context`, `--port`, `--update`). Setup asks nothing, installs no system
software (it names what is missing) and moves or deletes no file. It needs NVIDIA's driver, the CUDA toolkit 13.0 or
newer, Visual Studio 2022's C++ tools (Windows) or GCC (Linux), and Python 3.10 or newer: `setup.bat` and `setup.sh`
make `.venv` with it and install the server's packages and CMake and Ninja there. A step whose result is there is
skipped:

1. llama.cpp at the commit the engine is built against (`3cf03257`), into `third_party/llama.cpp`: ggml for the
   engine, mtmd for the image encoder, gguf-py for the packs.
2. The engine, `build/lamina` (`.exe`, with the CUDA runtime DLLs beside it), for the CUDA architectures of this PC's
   GPUs, and the image encoder, `build-vision/bin/lamina-vision`: llama.cpp's mtmd library with the model's projector,
   run on the CPU beside the engine, so the GPUs' expert caches keep all of their VRAM. It is built for any AVX2 CPU: a
   native build on an AVX-512 CPU runs without flash attention, about twice as slow on this PC's Ryzen 7 9700X.
3. The model's files from Hugging Face at pinned revisions, checked by size and SHA-256, into the data folder,
   `../Lamina-data` unless `--data-dir` names another.
4. Their preparation. For Flash-Next: the pack, in `packs/<model>`; the MTP draft layer, `mtp/rt-distilled`
   ([eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP), under
   Qwen's license: Qwen's MTP layer with its dense part trained on this engine's output, its card has how), whose
   drafts carry more tokens a verify pass than the layer as Qwen published it (the output stays the same: every token
   is verified); and with images, ISTA-DASLab's `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (0.9 GB) in Q8_0
   (`tools/mmproj_q8.py`, byte for byte llama-quantize's Q8_0), which encodes ~20% faster on the CPU. For Qwen3.6: its
   tokenizer, and with images its file's `mmproj-F16.gguf`.
5. The config and the launcher. Setup names the GPU with the most VRAM that drives no display as `--main-gpu`, keeps
   `--second-gpu auto`, and gives the image encoder the CPU's logical cores (`threads`). A config written before keeps
   the keys the example lacks (an `api_key`, the `images` of `tools/image/install.py`).

A pack is what `tools/iq_pack.py` makes from a model's GGUF files in seconds (~1.5 GB): the tokenizer with the chat
template, the model's float tensors in the forms the engine takes, and an index of the rest; the engine reads the
quantized tensors, the experts and most projections, from the GGUF files at start. UD-Q4_K_XL's is made with
`--compat-bf16`: the file stores the hyper-connection projections and the PLE value in Q8_0, which the pack holds rounded
to BF16, the form the engine reads.

Start a config with its launcher, or from the Lamina folder:

    .venv\Scripts\python.exe serve\server.py --engine strata --config lamina-iq3_s.json

The server listens on the config's `port` (8081, 8082 and 8083; `--port` overrides it). The configs' paths are relative
to the Lamina folder, where the server starts. On Linux the CUDA libraries come from the toolkit through the loader's
path, or from the config's `lib_dirs` (setup writes the toolkit's), which the server puts on `LD_LIBRARY_PATH`.

`--main-gpu` runs the model (`RTX 3090` here), and nothing else may use that card: a display or another program on it
slows generation to a few tokens per second. Name a card by its name or a part only its name has (`RTX 3090`), its UUID
(`nvidia-smi -L` lists them), its PCI bus id, or its number as `nvidia-smi` numbers the cards; a number moves when a
card is added. `--second-gpu auto` makes every other card a further expert tier, up to three, the fastest link first (a
copy test at startup; the log shows each card's rate): each holds more experts and may drive the display. A card the
desktop uses keeps 2048 MiB free, a headless one 512 (`--second-gpu-reserve-mib` sets them). To fix the tiers and their
order, name them: `--second-gpu "RTX 5070 Ti,RTX 5060 Ti"`.

A picture of ~1,000 tokens (`max_tokens`: up to 1,024) takes ~9 s to encode on the 9700X (`threads`: its 16 logical
cores), once per picture, and the encoder ~1 GB of RAM. Without images, set up with `--vision off`.

Both of this PC's Flash-Next configs also have `--arena-place`, left out here: for RAM whose regions read at different
rates, as unpaired DIMMs make (two 32 GB and two 48 GB modules), the engine times its RAM arena at the start and keeps
the experts the CPU reads out of the slow region, again between requests ([DETAILS.md](../docs/DETAILS.md)). On such a
PC add it to a config's arguments; elsewhere it finds nothing to move and costs ~2 s at the start.

This PC's Flash-Next UD-Q4_K_XL config also has `--no-file-cache`, left out here: the engine reads the experts from the
GGUF past the OS file cache. Where the RAM holds the 72 GiB arena and the cached 72 GB file together (this PC's 160 GB),
the file settles in the cache on pages the later starts' arenas would have had: here generation fell 7-8% over a
session's starts, and with the flag it stays at a fresh boot's speed. Each start then reads the experts from the drive
(here 4.3-4.4 GiB/s against 6 from the cache, ~5 s more). With less RAM the cache cannot keep the file beside the arena,
and the flag costs only the start time ([DETAILS.md](../docs/DETAILS.md)).

Both of this PC's Flash-Next configs also have `--prompt-split 26`, left out here: a prompt of more than one chunk runs
in two stages, the second GPU taking each chunk's layers 26-47 whole while the first runs the next chunk. It pays where
the second GPU alone reads a chunk about as fast as both together: this PC's 5070 Ti runs the dense steps 1.44x as fast
as its 3090, and prompts of 32K-128K tokens read 1.13-1.55x as fast. It costs 0.7 GiB more pinned RAM and lends 6-8 GiB
of the second GPU's cache during such a prompt ([DETAILS.md](../docs/DETAILS.md)). Measure it on your pair before adding
it, and its layer too, through the server: here generate mode put IQ3_S's best layer at 23 (6-11% faster than 26), while
through the API 26 was 2-4% faster than 23.

## Qwen3.6-35B-A3B

The engine runs Qwen3.6-35B-A3B (GGUF architecture `qwen35moe`) on a path of its own (`src/qwen35`, from upstream):
`lamina` switches to it from the model file's header, so the config names the file with `--native` like the others. It
reads the GGUF as it is (no pack, no expert profile; the drafts come from the file's own MTP block, `--mtp N` of them),
keeps every expert that fits in the VRAM of one GPU (`--main-gpu`) and computes the rest on the CPU. Setup downloads
unsloth's `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` and `mmproj-F16.gguf` from
[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d)
and writes the tokenizer (`tools/strata_tokenizer.py`: the same vocabulary as Flash-Next's, its own chat template).

The path keeps K and V in fp32 (~45 KB a cell), so the context decides how many experts fit: on a 24 GB card every
expert at 8,192 cells, 192 of 256 a layer at the config's 65,536 (setup's `--context` sets another). Its speeds against
llama.cpp on the same file: [COMPARISON.md](../docs/COMPARISON.md). Not on this path yet: the prompt cache's
checkpoints (a follow-up turn whose history differs anywhere from what the engine holds is read again from the start),
`/v1/choose`, and the GPU tiers.

## Docker

The `Dockerfile` builds the engine and the image encoder with setup for RTX 30, 40 and 50 GPUs (`CUDA_ARCHITECTURES`
narrows them), and its container prepares a model in `/data` with `setup.py --no-build` at the first start (later
starts check the files), then serves it on port 8080 to other machines with an API key:

    docker build -t lamina .
    docker run --rm --gpus all --ulimit memlock=-1 -p 127.0.0.1:8081:8080 -v lamina-data:/data \
               -e MODEL=iq3_s -e API_KEY=<a long random key> lamina

`VISION` (on, off), `CONTEXT` and `MAIN_GPU` pass the setup flags of those names.
