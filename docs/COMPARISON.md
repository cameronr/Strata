# This fork, upstream Strata and llama.cpp on one PC

Qwen3.8-Flash-Next in two quantizations, IQ3_S and UD-Q4_K_XL, through three engines on an RTX 3090 + RTX 5070 Ti PC:
llama.cpp's mainline build 11606 and upstream Strata a0116b27 (engine 0.1.43.1) measured on 2026-10-13/14, this fork
(speeds through the API, on fixed text and sampled, start and memory) on 2026-10-18/19 at dbecd594, after a restart
with the file cache's standby list emptied first, with its configs' draft layer trained on both models' own output, its
prompts in two stages and UD-Q4_K_XL's experts read past the file cache; its agreement with llama.cpp on 2026-10-18
at 08877797, whose arithmetic is the same.

**In short**, with both GPUs:

- **IQ3_S:** this fork reads prompts 3.5-7.2x as fast as llama.cpp and writes answers 5.5-10.2x as fast. Against
  upstream Strata's layer split: prompts 1.35x / 1.54x / 1.70x / 1.71x as fast at 16K / 32K / 64K / 128K tokens,
  answers 1.3-2.1x (the most on answers that quote the prompt). On the 3090 alone, against upstream there: prompts
  1.8-1.9x, answers 1.4-1.9x.
- **UD-Q4_K_XL:** prompts 13.1-23.1x and answers 5.0-7.4x as fast as llama.cpp; against upstream's layer split
  prompts 3.1-4.1x and answers 2.0-2.4x, on the 3090 alone prompts 2.0-2.7x and answers 1.7-2.1x.
- Much of the lead in long prompts comes from settings measured for this PC, the two-stage prompts above all
  (`--prompt-split`: 1.24-1.55x at 32K-128K here): on another pair of GPUs they need measuring again (below).
- Sampled with the configs' settings (temperature 0.7), this fork thinks 9-10% slower than greedy (fewer drafts
  pass), writes code 5-9% slower and edits as fast; llama.cpp's speed does not change, so this fork thinks 4.3-4.7x as
  fast as llama.cpp (greedy 4.8-5.2x).
- A follow-up turn after a 16K-token prompt starts in 0.11-0.16 s (llama.cpp 0.61-0.72 s; upstream 0.26-0.42 s on
  the 3090, 0.45-1.06 s on both GPUs).
- The next-token distributions match llama.cpp's about as closely as llama.cpp's own CPU and CUDA backends match each
  other: UD-Q4_K_XL 2.2-3.4x as far as those backends are apart (KL), from llama.cpp's 8-bit rounding of the file's
  Q8_0 hyper-connection weights; IQ3_S with the 4-bit V cache its config runs (k8v4) 1.5-4.5x (with 8-bit KV
  0.8-1.4x).

## The PC and the engines

- **PC:** Ryzen 7 9700X (8 cores), 160 GB DDR5-5600 (2 x 32 + 2 x 48 GB), Windows 11. An RTX 3090 (24 GB, PCIe 4.0
  x4, nothing else on it) and an RTX 5070 Ti (16 GB, PCIe 5.0 x16, driving the display).
- **Models:** ISTA-DASLab's GSQ-RCO **IQ3_S** (shard 1 54.8 GB: experts 46.8 GiB, i-quants with Q2_0 and IQ4_NL down
  projections) and unsloth's **UD-Q4_K_XL** (111 GB: experts 71.7 GiB, Q4_K / Q5_K with Q5_1 / Q8_0 downs). Each has
  a 28.8 GB IQ4_NL n-gram table (the per-layer embeddings) besides.
- **llama.cpp** build 11606 (1b70350c9, MSVC), as mainline builds it. Both GPUs with the layers split (the 5070 Ti
  first), the experts that do not fit in VRAM in RAM: UD-Q4_K_XL by `--fit`, IQ3_S by fit's placement with its Q2_0 down
  projections kept on the GPUs (llama.cpp has no x86 SIMD kernel for Q2_0: 30.5 instead of 27.2 t/s). 8-bit KV cache, 8
  threads, 1024-token micro-batches, the n-gram table in RAM. No speculative decoding: llama.cpp has no graph for this
  model's MTP layer.
- **Upstream Strata** a0116b27 (engine 0.1.43.1) with the configs its setup writes for each file on this PC: MTP
  drafts, 8-bit KV kept in RAM past 32K cells, the expert cache sized by the free VRAM; for UD-Q4_K_XL every expert in
  RAM (its RAM budget of 71 GiB) and its own pack (`tools/iq_pack.py --compat-bf16`). On the 3090, and on both GPUs
  with a layer split, the newer 5070 Ti first (its setup's choice here for IQ3_S; for UD-Q4_K_XL it recommends one GPU
  and offers both). Its route tail skip is on, as by default on CUDA: in a verify window, a missed expert that every
  token ranks 7th or lower is not computed (an approximation; this fork computes every routed expert).
- **This fork** with its configs ([examples](../examples/README.md)): the 3090 runs the model and the 5070 Ti is a
  second expert tier (12.2-12.4 GiB of experts; its share of each layer's experts, the next layer's likely experts
  copied ahead, the prompt path's experts the 3090 lacks, and with IQ3_S part of the output head). MTP drafts and
  prompt lookup (the draft layer's dense part trained here on both models' own output, `mtp/rt-distilled`; the
  examples keep basalt's `rt-tuned`, which thinks 1.3% (IQ3_S) / 4.5% (UD-Q4_K_XL) slower on the fixed text below),
  KV in VRAM (IQ3_S 8-bit keys and 4-bit values, k8v4; UD-Q4_K_XL 8-bit), 16K-token prompt chunks. A prompt of more
  than one chunk runs in two stages (`--prompt-split 26`): the 5070 Ti runs each chunk's layers 26-47 whole while the
  3090 runs the next chunk's first 26. Both keep the experts the CPU reads out of the RAM's slower region
  (`--arena-place`: the unpaired DIMMs leave 32 GB on one channel; timed at the start, placed again between
  requests); UD-Q4_K_XL reads its experts past the OS file cache (`--no-file-cache`), whose copy of its 72 GB file
  would hold the RAM's faster pages; for IQ3_S the 5070 Ti also computes a fifth of the experts the 3090 holds in long
  prompts (`--prompt-shift 0.2`). And on the 3090 alone.
- **Settings calibrated for this PC.** The two-stage prompts and their layer, the arena's placement, UD-Q4_K_XL's
  reads past the file cache, `--prompt-shift`, `--head-split` and `--feed-max` were each measured on this PC's pair of
  GPUs, their links (the 3090 on PCIe 4.0 x4, the 5070 Ti on 5.0 x16) and its RAM, and much of this fork's lead in long
  prompts comes from them: split at layer 26
  the prompts of 32K / 64K / 128K tokens read IQ3_S 1.24 / 1.43 / 1.55x as fast as in one stage, UD-Q4_K_XL 1.13 /
  1.30 / 1.43x. On another machine they need measuring again, through the server: the split's layer alone moved
  IQ3_S's long prompts by up to 11% here, and generate mode and the server disagreed on the best one
  ([DETAILS.md](DETAILS.md), `--prompt-split`; the engine's examples leave the split, the placement and the reads past
  the cache out).

## Speed through the API

The same requests to each server's OpenAI API: greedy, one at a time, the prompt caches off except for the follow-up
turn; this fork's servers got them 4 s apart, the time they use to place their RAM arenas again. Where an arena lands
varies from start to start (these runs: 3.1 GiB of IQ3_S's 47 in the RAM's slower region, 2.2 of UD-Q4_K_XL's 72;
other starts here 0.1-27 GiB), and placing it moves the experts the CPU reads out of there (IQ3_S: 796 of its 12,332
were there, none after). Prompts: 16K-128K tokens
of llama.cpp's sources with a question (400 tokens of answer), a 28-token coding request (up to 300), a coding task with
thinking on (3,000), and a Python module to return with a class renamed (the edit, up to 3,000 tokens that mostly quote
the prompt). The follow-up turn sends the 16K prompt, its answer and a new question: its prompt time is the processing
of what the engine did not keep. Each engine writes its own text and the texts part at the first near-tie, so the
generation speeds compare different texts of one kind, one run each. The memory rows come from a second start of each
server (the model file then in the OS cache: this fork read IQ3_S's experts from it at 9.1-9.5 GiB/s against 4.4 from
the SSD, and UD-Q4_K_XL's past the cache from the SSD at every start, at 4.3-4.4; its start timed to a quarter of a
second, the other engines' to 3 s): the GPUs' memory taken, and the RAM taken as the drop in the system's available
memory.

### IQ3_S

<!-- matrix:api-iq3_s -->
| | llama.cpp, both GPUs | upstream Strata, 3090 | upstream Strata, both GPUs | this fork, both GPUs | this fork, 3090 |
|---|---|---|---|---|---|
| Prompt, 16K tokens (t/s) | 885 | 1172 | 2308 | 3119 | 2160 |
| Prompt, 32K tokens (t/s) | 851 | 1236 | 2927 | 4511 | 2355 |
| Prompt, 64K tokens (t/s) | 837 | 1232 | 3250 | 5536 | 2256 |
| Prompt, 128K tokens (t/s) | 805 | 1203 | 3371 | 5781 | 2246 |
| Generation: code, 28-token prompt (t/s) | 29.6 | 93.0 | 117.1 | 198.7 | 132.6 |
| Generation: thinking, 3,000 tokens | 31.2 | 82.0 | 117.8 | 171.6 | 137.2 |
| Generation: edit, the prompt's file returned (2.4K tokens) | 31.1 | 115.8 | 148.3 | 316.2 | 218.0 |
| Generation after the 16K prompt | 30.8 | 84.3 | 117.6 | 178.7 | 137.3 |
| Generation after the 32K prompt | 28.9 | 91.6 | 125.6 | 194.7 | 149.1 |
| Generation after the 64K prompt | 25.7 | 80.5 | 122.3 | 165.8 | 125.1 |
| Generation after the 128K prompt | 21.4 | 84.4 | 113.9 | 147.9 | 122.7 |
| Follow-up turn after the 16K prompt: new tokens, prompt time | 34 in 0.61 s | 33 in 0.26 s | 440 in 1.06 s | 34 in 0.11 s | 34 in 0.18 s |
| Start to the API, the model file cached (s) | 20 | 21 | 25 | 14 | 13 |
| VRAM taken: 3090 / 5070 Ti (GiB) | 22.5 / 12.4 | 22.7 / 0.0 | 22.5 / 14.6 | 22.5 / 13.5 | 22.6 / 0.0 |
| RAM taken (GB) | 57 | 56 | 58 | 54 | 52 |
<!-- /matrix:api-iq3_s -->

### UD-Q4_K_XL

<!-- matrix:api-ud-q4_k_xl -->
| | llama.cpp, both GPUs | upstream Strata, 3090 | upstream Strata, both GPUs | this fork, both GPUs | this fork, 3090 |
|---|---|---|---|---|---|
| Prompt, 16K tokens (t/s) | 258 | 558 | 828 | 3381 | 1491 |
| Prompt, 32K tokens (t/s) | 258 | 746 | 1416 | 4457 | 1566 |
| Prompt, 64K tokens (t/s) | 253 | 750 | 1484 | 5343 | 1465 |
| Prompt, 128K tokens (t/s) | 248 | 743 | 1519 | 5735 | 1467 |
| Generation: code, 28-token prompt (t/s) | 23.4 | 49.3 | 65.5 | 133.3 | 87.5 |
| Generation: thinking, 3,000 tokens | 24.2 | 39.5 | 54.4 | 121.3 | 83.3 |
| Generation: edit, the prompt's file returned (2.4K tokens) | 24.1 | 68.1 | 85.3 | 177.9 | 129.0 |
| Generation after the 16K prompt | 23.6 | 44.1 | 58.6 | 141.9 | 79.8 |
| Generation after the 32K prompt | 22.6 | 44.3 | 64.0 | 143.5 | 91.0 |
| Generation after the 64K prompt | 21.6 | 42.4 | 63.7 | 126.0 | 73.6 |
| Generation after the 128K prompt | 18.5 | 41.8 | 59.2 | 127.6 | 72.2 |
| Follow-up turn after the 16K prompt: new tokens, prompt time | 34 in 0.72 s | 33 in 0.42 s | 33 in 0.45 s | 34 in 0.16 s | 34 in 0.32 s |
| Start to the API, the model file cached (s) | 53 | 51 | 61 | 28 | 27 |
| VRAM taken: 3090 / 5070 Ti (GiB) | 22.3 / 12.9 | 22.7 / 0.0 | 22.8 / 14.6 | 22.7 / 13.4 | 22.7 / 0.0 |
| RAM taken (GB) | 84 | 69 | 69 | 81 | 80 |
<!-- /matrix:api-ud-q4_k_xl -->

## Sampled generation

This fork's configs sample the requests that send no sampling settings: temperature 0.7, top_p 0.95, top_k 20, min_p
0. The thinking, code and edit requests above, each greedy and sampled with these settings five times (seeds 1-5, the
same seed to both engines; this fork got per seed the three greedy requests, then the three sampled ones, as its
adaptive tiers keep the last request's experts): generation t/s as the median (range) of the five, the sampled median
against the greedy one, and this fork's tokens a verify pass (median). A sampled token is drawn by its probability, and
this fork keeps a draft only when it is the token drawn, so fewer drafts pass than when greedy takes the likeliest
token: on thinking text 1.9-2.1 instead of 2.2-2.3 tokens a pass. The sampled code answers run 129-300 tokens
(greedy's 257-300); the edit quotes its prompt, where both pick the same tokens. llama.cpp runs no drafts, and sampling leaves
its speed as it is. The sampled runs spread about as widely as the greedy ones.

<!-- matrix:sampled -->
| | IQ3_S: llama.cpp | IQ3_S: this fork | tokens a pass | UD-Q4_K_XL: llama.cpp | UD-Q4_K_XL: this fork | tokens a pass |
|---|---|---|---|---|---|---|
| Thinking, 3,000 tokens, greedy | 31.1 (29.2-31.8) | 161.9 (155.6-164.2) | 2.30 | 25.4 (25.3-25.6) | 121.7 (120.3-126.3) | 2.19 |
| Thinking, 3,000 tokens, sampled | 31.3 (29.4-31.9; 1.01x) | 146.7 (141.4-147.4; 0.91x) | 2.07 | 25.5 (25.3-25.7; 1.00x) | 109.1 (108.3-110.3; 0.90x) | 1.87 |
| Code, 28-token prompt, greedy | 30.7 (29.3-32.4) | 208.7 (197.4-210.4) | 3.47 | 25.1 (24.8-25.2) | 143.8 (141.2-153.7) | 3.45 |
| Code, 28-token prompt, sampled | 30.7 (26.8-32.6; 1.00x) | 198.8 (161.3-210.5; 0.95x) | 3.30 | 25.3 (24.9-25.9; 1.01x) | 131.3 (123.3-136.6; 0.91x) | 3.09 |
| Edit, the prompt's file returned, greedy | 31.4 (29.5-31.7) | 314.2 (313.1-314.8) | 7.54 | 25.2 (25.2-25.4) | 190.7 (189.0-193.7) | 7.54 |
| Edit, the prompt's file returned, sampled | 31.2 (29.6-31.6; 0.99x) | 306.8 (305.7-307.0; 0.98x) | 7.54 | 25.2 (25.2-25.5; 1.00x) | 189.8 (188.0-192.2; 1.00x) | 7.54 |
<!-- /matrix:sampled -->

## Generation on fixed text (this fork)

Each run emits the model's reference text (`--spec-follow`), so runs and builds compare on the same tokens and
experts: two runs, averaged, tokens a verify pass in parentheses. The code text's reference repeats its answer after
the end of the turn, which prompt lookup drafts and a server never generates. Read through the file cache,
UD-Q4_K_XL's 72 GB file settles in the cache on the RAM's faster pages over a session's starts, and each later start's
arena lands on slower ones: with the file settled its code and edit texts ran 119.6 / 164.1 t/s, 8-11% below these
runs, the CPU pool reading its experts and the 5070 Ti its rows slower alike. Its config reads past the cache
(`--no-file-cache`, [DETAILS.md](DETAILS.md)); IQ3_S's 47 GB file fits beside its arena.

<!-- matrix:fixed -->
| Text | IQ3_S t/s (tokens a pass) | UD-Q4_K_XL t/s (tokens a pass) |
|---|---|---|
| code (400 tokens) | 208.7 (4.06) | 130.2 (3.79) |
| thinking (1,200) | 168.1 (2.43) | 123.8 (2.04) |
| after the 16K prompt (400) | 178.9 (2.85) | 138.0 (2.83) |
| edit (2,397) | 309.9 (7.54) | 184.0 (7.54) |
<!-- /matrix:fixed -->

## Agreement with llama.cpp

Argmax agreement and mean KL divergence (nats) of the next-token distributions along llama.cpp's greedy text, which
this fork follows (`--spec-follow`, `--window-logits`), with `--adapt-every 0`. The two baselines differ from each
other only in rounding: llama.cpp's CPU backend against its CUDA backend, and this fork in one-token windows against
its four-token windows. Any change of rounding moves 2-4% of the thinking text's argmaxes, all near-ties.

<!-- matrix:parity -->
| | Code, ~260 tokens | Thinking, 1,200 | After the 16K prompt, 300 |
|---|---|---|---|
| IQ3_S: this fork vs llama.cpp | 99.6%, 6.9e-04 | 97.3%, 4.0e-03 | 98.7%, 2.3e-03 |
| llama.cpp's CPU vs CUDA backend | 100.0%, 1.5e-04 | 97.7%, 2.6e-03 | 98.7%, 1.5e-03 |
| this fork, one-token vs four-token windows | 100.0%, 1.7e-04 | 97.7%, 2.7e-03 | 99.3%, 8.6e-04 |
| UD-Q4_K_XL: this fork vs llama.cpp | 99.2%, 6.0e-04 | 95.8%, 6.2e-03 | 98.3%, 3.9e-03 |
| llama.cpp's CPU vs CUDA backend | 100.0%, 1.7e-04 | 97.7%, 2.8e-03 | 98.7%, 1.5e-03 |
| this fork, one-token vs four-token windows | 99.6%, 1.4e-04 | 97.9%, 2.1e-03 | 98.0%, 5.5e-04 |
<!-- /matrix:parity -->

## The two files

llama.cpp's perplexity (build 11116, 2026-09-30) over 28 chunks of 4,096 tokens of llama.cpp's own docs, C++ and
Python sources (lower is better): UD-Q4_K_XL 1.8849, IQ3_S 1.8946 (+0.5%), and ISTA-DASLab's IQ3_XXS, which IQ3_S
replaced in this fork's configs, 1.9215 (+1.9%). IQ3_S's experts take 46.8 GiB, UD-Q4_K_XL's 71.7: on this PC IQ3_S
generates 30-68% faster on fixed text; with both GPUs it reads prompts 8% slower to 4% faster, on the 3090 alone
1.45-1.54x as fast.

## Qwen3.6-35B-A3B

The engine's Qwen3.6 path (`src/qwen35`, upstream's driver on this fork's kernels) on unsloth's UD-Q4_K_XL (the MTP GGUF,
22.9 GB), measured on 2026-10-29: this fork with its config (`examples/qwen36-ud-q4_k_xl.json`: the RTX 3090 alone,
65,536 cells, 2 drafts) against mainline llama.cpp build 11606 on the same file with this PC's former Qwen3.6 preset
(64K cells, 8-bit KV, 2048-token micro-batches, MTP drafts up to 3, the token embedding on the CPU), on the 3090 alone
(the experts that do not fit in RAM, `--fit`) and with the 5070 Ti, layer split 15,25. The same requests through each
server's API, greedy, the prompt cache off (this fork's thinking: the server's two engine requests, the thinking cut at
its budget of 2,250 tokens and the answer after it, 2,982 tokens in 25.1 s).

| | llama.cpp, 3090 | llama.cpp, both GPUs | this fork, 3090 |
|---|---|---|---|
| Generation: code, 28-token prompt (t/s) | 179.3 | 214.4 | 124.6 |
| Generation: thinking (3,000 tokens) | 181.5 | 196.8 | 118.9 |
| Edit: prompt / generation (t/s) | 3,268 / 227.8 | 3,700 / 238.6 | 934 / 109.6 |
| 16K-token prompt / generation after it (t/s) | 3,847 / 165.4 | 4,431 / 183.8 | 1,433 / 44.8 |
| 32K-token prompt / generation after it (t/s) | 3,520 / 146.5 | 4,055 / 154.3 | 1,380 / 23.1 |

This path keeps K and V in fp32 (~45 KB a cell): at 65,536 cells 192 of the 256 experts a layer fit on the 3090 and the
CPU computes 16-18% of the routed entries; at 8,192 cells every main layer's experts fit, and on fixed text (generate
mode) it reads a 6,000-token prompt at 2,696 tok/s and writes 195 t/s on the thinking task with 2 drafts, 125 t/s after
the 6,000 tokens (110 without drafts). Its generation slows with the context more than llama.cpp's: a pass takes ~1 ms
longer per 1K cells of context with every expert in VRAM. A follow-up turn is read again from the start where its
history differs from what the engine holds (no prompt-cache checkpoints on this path).

Agreement with llama.cpp (`scratch/q36/parity36.sh`, every expert in VRAM, 8,192 cells): along llama.cpp's own greedy
text, the next-token distributions' mean KL against llama.cpp's CUDA backend is 4.2e-4 / 1.6e-3 / 1.3e-6 (a 27-token
coding request, the thinking task, the edit; 138 / 400 / 400 positions), where llama.cpp's CPU backend is 5.1e-4 /
1.8e-3 / 1.0e-6 from it; the argmax agrees at 100% / 98.75% / 100% of the positions, as the CPU backend's does.

## Features

| | llama.cpp | Upstream Strata | This fork |
|---|---|---|---|
| Model files | any GGUF of the model | ISTA-DASLab's GSQ-RCO sizes (Q2_0 to IQ3_S), their Coder, Swift 1.5 (i-quant and Q2_0 experts); unsloth's UD-Q4_K_XL and UD-IQ4_XS | those, and every quant of unsloth's (K-quant, Q5_1 and Q8_0 experts) |
| Speculative decoding | none: no graph for this model's MTP layer | the MTP layer's drafts, up to 3 a pass; prompt lookup (up to 5 tokens where it pays) | the MTP layer's drafts; prompt lookup (up to 7 tokens after a 16+-token match) |
| Several GPUs | a layer or row split | a layer split, each card holding its layers' experts; helper expert caches on up to three more cards (experimental) | the second card as an expert tier: its share of each layer, the next layer's likely experts copied ahead, the prompt's experts the first card lacks, part of the output head; prompts of several chunks in two stages, one on each card |
| Experts in VRAM | a fixed placement | ranked by use, updated every 4 verify passes | ranked by use on both cards, updated every pass |
| KV cache | 8-bit, in VRAM | 8-bit, in RAM past 32K cells; 4-bit options | 8-bit, in VRAM; 4-bit options (q4_0, k8v4) in the tensor-core attention |
| Prompt cache | the live slot, states saved to RAM | the live session, checkpoints at turns and every 16K tokens, a replaced conversation parked in RAM | the live sequence, checkpoints at turns and chunks spread back from the end, replaced conversations kept in RAM with their K/V, a kept conversation's start lent to another that shares it |
| Sampling, APIs | full; OpenAI and Anthropic | per request; OpenAI and Anthropic, a web app, MCP tools | upstream's |
