# Setup

What a PC needs: [the README](../README.md#what-you-need). This page covers what setup does, the config it writes, the
GPUs, the engine options worth knowing, Qwen3.6-35B-A3B and Docker.

## Setup

From a clone of this repository:

```
setup.bat --model iq3_s          (Windows)
./setup.sh --model iq3_s         (Linux)
```

`setup.bat` and `setup.sh` make a Python environment, `.venv`, install the server's packages and CMake and Ninja into
it, and run `setup.py` with the same arguments. The models: `iq3_s` (Qwen3.8-Flash-Next, ISTA-DASLab's GSQ-RCO IQ3_S),
`ud-q4_k_xl` (Qwen3.8-Flash-Next, unsloth's UD-Q4_K_XL) and `qwen36-ud-q4_k_xl` (Qwen3.6-35B-A3B, unsloth's UD-Q4_K_XL).
Without `--model`, setup builds the engine only.

| Option | |
| --- | --- |
| `--check` | say what this PC has (GPUs, RAM, disk, the tools) and what the model needs; change nothing |
| `--data-dir DIR` | where the model files go (default `../Lamina-data`, beside the clone) |
| `--vision on\|off` | pictures (default on; the image encoder runs on the CPU) |
| `--main-gpu GPU` | the GPU the model runs on (default: the one with the most VRAM that drives no display) |
| `--context N` | the context in tokens (default: the example config's) |
| `--port N` | the server's port (default 8081, 8082, 8083 by model) |
| `--update` | `git pull` this clone first |
| `--no-build` | use the binaries already built |
| `--cuda-archs A` | the CUDA architectures to build for, e.g. `86;120` (default: this PC's GPUs) |
| `--portable` | build the engine for other PCs (any AVX2 CPU), as the Docker image does |

Setup asks nothing, installs no system software (it names what is missing) and moves or deletes no file. A step whose
result is already there is skipped, so setup runs again after an interruption or for another model:

1. llama.cpp at the commit the engine is built against, into `third_party/llama.cpp`: ggml for the engine, mtmd for
   the image encoder, gguf-py for the preparation tools.
2. The server with the engine, `build/lamina-server` (`.exe`, with the CUDA runtime DLLs beside it; `build/lamina` is
   the engine alone), for this PC's GPUs, and the image encoder, `build-vision/bin/lamina-vision`: llama.cpp's mtmd
   library with the model's projector, run on the CPU beside the engine so the GPUs keep their VRAM for experts. The encoder is built for any AVX2 CPU: a native build on an AVX-512
   CPU loses flash attention and encodes about half as fast.
3. The model's files from Hugging Face at pinned revisions, into the data folder, resumed when interrupted and checked
   once by size and SHA-256 (`<file>.sha256` beside each records the check):
   [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
   [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) and
   [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF). `HF_ENDPOINT` names a
   mirror.
4. Their preparation, for Flash-Next only: the MTP draft layer, `mtp/rt-distilled`, from
   [eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP); with
   pictures, ISTA-DASLab's image projector converted to Q8_0 (`tools/mmproj_q8.py`), which the CPU encodes ~20% faster.
   Everything else comes from the GGUF files at start: the server reads the tokenizer and the chat template, the engine
   the experts, the projections and the float tensors (put in the forms it takes in memory, in milliseconds).
   Qwen3.6 with pictures uses the file's own projector.
5. The config, `lamina-<model>.json`, and its launcher, `run-<model>.bat` (`run-<model>.sh` on Linux), from
   `examples/<model>.json` with this PC's GPUs and data folder. A config written before keeps the keys the example
   lacks (an `api_key`, an `images` section).

## The config

`lamina-<model>.json` holds the engine's command line and the server's settings. Its keys from setup:

| Key | |
| --- | --- |
| `args` | the engine's options (below and `build/lamina --help`) |
| `cwd`, `lib_dirs` | the folder the engine runs in; on Linux, folders for the CUDA libraries (setup writes the toolkit's) |
| `model_name` | the name the API lists |
| `port`, `log` | the server's port; the engine's log file |
| `sampling` | the defaults for requests that send no sampling fields ([USING.md](USING.md#sampling)) |
| `vision` | the image encoder: its binary, the projector, the model, `threads` (the CPU's logical cores) |
| `expert_profile_save`, `expert_profile_save_every` | what the run learned about the experts (below) |

The server's other keys (the network, API keys, idle unload, MCP servers, image generation):
[USING.md](USING.md#the-config). Relative paths are the clone's: the launchers start the server there.

**The learned expert profile.** The GPUs' expert caches start from a ranking of the experts by how often the model
routes to them (`data/expert-profile.bin`). With `expert_profile_save`, the engine also counts this PC's routing and
writes a profile (`<model>-learned.bin` in the data folder) at exit, on Ctrl+C and between requests at most once a
minute (`expert_profile_save_every`); the next start fills the caches from it. It ranks the experts by this PC's routing
and the profile it started from at equal weight. On the development PC, after a session of code and reasoning requests,
a round of fixed text took 9-14% less time on code, -3.8 to +0.9% on thinking and edits. Remove both keys to start from
the shipped profile every time.

## The GPUs

**`--main-gpu`** names the GPU the model runs on. Nothing else may use it: the engine synchronizes the GPU with its host
loop through spin-waits, and a display or another program on the same GPU makes the driver time-slice it, which slows
generation to a few tokens per second. Name it by its name or a part only its name has (`"RTX 3090"`), its UUID or the
start of it (`nvidia-smi -L` lists them), its PCI bus id, or its number as `nvidia-smi` numbers the GPUs (a number
moves when a card is added). The engine leaves `--vram-reserve-mib` (default 700) of it free and fills the rest with
experts (`--expert-cache auto`).

**`--second-gpu auto`** (the examples' setting) makes every other GPU a further expert tier, up to three, the fastest
link first: a short copy test at the start measures each GPU's rate from the host, and the log shows it. Each tier
holds the next experts of the profile and of the conversation and computes them beside the CPU; the first one also
computes the experts of a prompt that the main GPU lacks and part of the output head. A tier may drive the display: one
that does keeps 2048 MiB free, a headless one 512 (`--second-gpu-reserve-mib`). To fix the tiers and their order, name
them: `--second-gpu "RTX 5070 Ti,RTX 5060 Ti"`; `--second-gpu -1` uses none. Only the first tier takes part in
prompts. The others help generation, where an expert crosses the link once and stays, so a GPU on a slow link (an x1
or x4 slot) still pays as a later tier.

## Engine options

The example configs' options, and the ones that pay on some PCs. `build/lamina --help` lists every one.

| Option | |
| --- | --- |
| `--max-context N` | the context the K/V is sized for (the examples: 262,144, the trained context) |
| `--kv-grow` | the K/V takes VRAM only for the cells the requests reach; the expert cache holds the rest and gives it back as a conversation grows |
| `--kv fp16\|int8\|k8v4\|q4_0` | the K/V's format: 8-bit (`int8`), 8-bit keys and 4-bit values (`k8v4`), 4-bit both; smaller formats leave more VRAM to experts, at a cost in precision |
| `--spec T`, `--spec-min-p P`, `--spec-lookup N` | verify windows of up to T tokens (the last one and T-1 drafts), a draft kept while its probability under the draft layer stays above P, and drafts from earlier text that repeats (16 tokens of it) |
| `--prefill N` | the prompt chunk (the examples: 16,384 tokens) |
| `--feed-max N` | prompt parts of up to N tokens go through verify windows instead of the prompt path |
| `--hc-q8` | the hyper-connection projections in 8 bits only: half their VRAM, read ~30% faster |
| `--prompt-shift F` | on a second GPU, a share of the experts the main GPU holds is computed beside it in long prompts |
| `--pcie-mode kernel --pcie-frac 0` | the experts no GPU holds are all computed by the CPU, none read by the main GPU over its link (the examples' setting) |
| `--prompt-split L` | off by default: a prompt of more than one chunk runs in two stages, the second GPU taking each chunk's layers L and up while the main GPU runs the next chunk |
| `--arena-place` | off by default: for RAM with slower regions (unpaired DIMMs), the experts the CPU reads are kept out of them |
| `--no-file-cache` | off by default: the experts are read past the OS file cache, which then holds no second copy of the model |

The last three are measured on the development PC in [examples/](../examples/README.md); measure them on yours before
keeping them.

## Qwen3.6-35B-A3B

The engine runs Qwen3.6-35B-A3B (GGUF architecture `qwen35moe`) on a path of its own (`src/qwen35`): `lamina` switches
to it from the model file's header, so its config names the file with `--native` like the others. It reads the GGUF as
it is (no expert profile; the drafts come from the file's own MTP block, `--mtp N` of them), keeps every
expert that fits in the main GPU's VRAM and computes the rest on the CPU. It uses one GPU.

Its K and V are kept in fp32, ~45 KB a cell, so the context decides how many experts fit: on a 24 GB GPU every expert at
8,192 cells, 192 of 256 a layer at the example's 65,536 (`--context` sets another). Not on this path yet: the prompt
cache's checkpoints (a follow-up turn whose history differs from what the engine holds is read again from the start),
`/v1/choose` and the GPU tiers.

## Docker

The `Dockerfile` builds the engine and the image encoder with setup for RTX 30, 40 and 50 GPUs (`CUDA_ARCHITECTURES`
narrows them). The container prepares a model in `/data` at its first start (later starts check the files) and serves
it on port 8080 with an API key:

```
docker build -t lamina .
docker run --rm --gpus all --ulimit memlock=-1 -p 127.0.0.1:8081:8080 -v lamina-data:/data \
           -e MODEL=iq3_s -e API_KEY=<a long random key> lamina
```

`VISION` (on, off), `CONTEXT` and `MAIN_GPU` pass the setup options of those names. The engine pins the experts in
RAM, hence `--ulimit memlock=-1`.

## Updating

`setup --update --model <model>` pulls this clone, rebuilds what changed and keeps the config's own keys. A model's
files stay where they are.
