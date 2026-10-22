# Krait: Nemotron 3.5 Super

Krait is based on Strata by Niko1221 (github.com/Niko1221/Strata, MIT).

This port runs **NVIDIA Nemotron 3.5 Super VL 120B-A12B** (GGUF architecture `nemotron_h_moe`: Mamba-2 + attention
without RoPE + LatentMoE, one MTP block) on a single 32 GB NVIDIA card with the routed experts of most layers in host
RAM. It has its own engine, `krait-engine` (src/nemotron_h/), built on llama.cpp's ggml kernels (b11806) with a
graph, expert cache and scheduler of its own. The server (`serve/server.py`, OpenAI + Anthropic API, web page) and the
vision encoder (upstream `strata-vision` on llama.cpp's mtmd, installed as `krait-vision`) are Strata's, with the
Nemotron changes described below (Run, Pictures).

Tested file: Unsloth's `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q4_K_XL` (3 shards) and `mmproj-BF16.gguf`.

## Correctness

The standard is llama.cpp b11806 (`0682a576a`) itself. All of these were measured; raw outputs are in the port's
work log (reports/).

| check | result |
|---|---|
| Gate 2: 32 golden prompts, greedy, vs a llama.cpp capture | 32/32 token-identical; 42,075/42,075 logits rows bit-identical (also with MTP speculation on) |
| speculation on vs off (`--spec-check`, every logits row hashed) | identical for MTP k = 1, 2, 3 and n-gram k = 1, 2, 3 |
| server: served greedy tokens vs the CLI | solo 6/6, `--batch 4` 6/6, 1-4 concurrent sessions all identical, HTTP 13/13 |
| pictures: engine vs `llama-mtmd-cli` (2 images, solo and batched) | 4/4 identical |

## Measured speed

The test machine (one 32 GB NVIDIA card, WSL2). Every run with
`--vram-limit-mib 29346`. Real prompts: 6 chat prompts x 3 runs, 256 tokens, greedy, median of the 18 runs, decode
tok/s = (n - 1) / (t_last - t_first). llama.cpp b11806 was run fresh on the same machine with the same placement
(`--n-cpu-moe 68`).

| | Krait | llama.cpp b11806 |
|---|---|---|
| decode, real prompts (CLI) | **23.016 tok/s** (1.609x) | 14.300 tok/s |
| decode, one HTTP stream (6 prompts, thinking off) | 24.6 tok/s median | - |
| time to first token, HTTP, 34-41 token prompts | 0.757-0.873 s | - |
| prompt read 1K / 4K / 32K tokens (synthetic) | 455 / 510 / 513 tok/s | - |
| time to first token at 32K | 63.378 s | - |
| MTP acceptance (k = 1) | 0.872 (1.872 tokens per verify step) | - |
| several requests at once (`--batch 4`), total | 1 / 2 / 4 requests: 23.3 / 21.25 / 23.7 tok/s | - |
| picture encode on the CPU (640x480 -> 300 rows / 640x400 -> 260 rows) | 1.17 s / 0.98 s | - |
| peak VRAM / host RAM (engine RSS) | 29.2 GB / 81.3 GiB | - |

More requests at once do not raise the total: every extra row needs its own experts, and 68 MoE layers run on the
CPU. `--batch` lets several users share the model; it does not make it faster.

## Build

Linux (or WSL2) with a CUDA toolkit, cmake >= 3.21, ninja, gcc and python3:

```bash
krait/build.sh            # KRAIT_ROOT=<work dir> CUDA_ARCH=120 JOBS=16 (defaults: two levels up, 120, nproc)
```

It fetches llama.cpp b11806 (`0682a576a`, the engine's ggml and the encoder's mtmd) and the commit `setup.py` pins
for the rest of Strata (`3cf03257`), then builds `build/ggml-nh`, `build/krait/krait-engine`,
`build/krait/krait-layout` and `build/vision/krait-vision` under KRAIT_ROOT.

Two llama.cpp versions on purpose: Strata's own CUDA kernels (src/prefill/moe_mmq.cu and the other MMQ files) compile
only against the commit setup.py pins. Pointing `STRATA_GGML_DIR` at a newer llama.cpp breaks them (for example
`ggml_cuda_host.cu: declaration is incompatible with ... new_pool_for_device`). The Nemotron engine needs b11806, so it
takes its ggml from `KRAIT_GGML_INCLUDE` / `KRAIT_GGML_LIBDIR` instead.

The model and the tokenizer pack:

```
KRAIT_ROOT/models/UD-Q4_K_XL/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q4_K_XL-0000{1,2,3}-of-00003.gguf
KRAIT_ROOT/models/mmproj-BF16.gguf
python3 tools/strata_tokenizer.py --gguf KRAIT_ROOT/models/UD-Q4_K_XL/...-00001-of-00003.gguf --out KRAIT_ROOT/pack/nemotron/tokenizer
build/krait/krait-layout --model <shard 1> --mmproj models/mmproj-BF16.gguf     # checks every tensor
```

## Run

```bash
KRAIT_ROOT=<work dir> krait/krait            # http://127.0.0.1:8095/v1, also HOST=, PORT=, KRAIT_CONFIG=
KRAIT_API_KEY=<key> HOST=0.0.0.0 krait/krait   # on the network: set a key
```

`krait/config.json` holds the engine flags (paths relative to KRAIT_ROOT). Its `vision` section turns
pictures on. The encoder runs on the CPU (`threads`, `"flash_attn": "on"`, which matches llama.cpp's `-fa on`), so it
uses no VRAM.

| engine flag | in the config | meaning |
|---|---|---|
| `--n-cpu-moe 68` | yes | the routed experts of the first 68 MoE layers stay in host RAM (read by the CPU or cached on the card) |
| `--cache all` | yes | a VRAM cache of hot experts for those layers (123 slots per layer solo, 111 with `--batch 4`) |
| `--pin` | yes | page-locks the host expert ranges (needs a memlock limit of unlimited, as in docker/krait/compose.yml) |
| `--spec mtp --spec-k 1 --spec-min-acc 0.6` | yes | MTP speculation with the model's own draft block; drafting pauses while the acceptance average is under 0.6 |
| `--chunk 512`, `--max-context 32768`, `--threads 16` | yes | prompt chunk, context, CPU threads |
| `--vram-limit-mib N` | 29346 | the engine keeps its whole VRAM use under N MiB |
| `--batch 2..4` | 4 | up to 4 requests decode together (rows mode); without it requests take turns |
| `--prefix-reuse on\|off` | **off** | see below |
| `KRAIT_TRACE=<file>` (environment) | - | writes each request's prompt ids and generated ids as one JSON line (debugging) |

### `--prefix-reuse`

With `on` (the engine's default, as in upstream Strata), a request that continues a conversation the engine still
holds (same token prefix, for example turn 2 = turn 1 + answer + a new user turn) keeps that session's state and only
reads the new tokens. With `off`, every request reads its whole prompt from scratch.

`on` is faster for follow-up turns, but its output is **not bit-identical** to a fresh read. The reused state was built
by single-token decode steps, while a fresh read builds it from one prompt chunk, so the float sums run in a different
order. Measured: turn 1 6/6 identical in both modes; turn 2 with `on`, 4 of 6 conversations differed (first
difference at token 21 or 61, the same two prompts solo and batched); with `off`, 12/12 identical. The shipped config
uses `off` so that served output equals the CLI exactly. Set `on` if faster follow-ups matter more than that. Sessions
that hold a picture are never reused in either mode.

### Pictures

The chat template writes `<image>` for a picture. For this model the server turns each one into `<img>`, one
`<image>` per picture row and `</img>`, which is the layout llama.cpp's mtmd (nemotron_omni) uses. The engine reads the
encoder's rows as input embeddings in their place. There is no M-RoPE: one position per row. Example: a 640x480 picture
becomes 300 rows.

## Docker Compose (Krait + Open WebUI + embedder for RAG)

`docker/krait/`: `stage.sh` builds a runtime image (`krait:local`) from your build, and `compose.yml`
starts Krait (GPU), Open WebUI and a llama.cpp embedding server (CPU, nomic-embed-text-v1.5) that Open WebUI uses
for document search. Steps: `cp docker/krait/env.example docker/krait/.env` (set KRAIT_MODELS and the keys), run
`docker/krait/stage.sh`, then `docker compose -f docker/krait/compose.yml --env-file docker/krait/.env up -d`,
then open http://127.0.0.1:3095. `docker/krait/smoke_openwebui.py` checks the stack through Open WebUI's API: chat,
streaming, a picture, and a document upload -> embedding -> retrieval -> answer round trip. It passed 8/8 on the test machine with
Docker Engine 29.8.1 in WSL2.

## Tests

Model-free (CTest, `-DSTRATA_BUILD_TESTS=ON`; also part of the full `ctest`):

- `nemotron_h_layout_test`: geometry from metadata, block kinds, the layout check and its refusals, GGUF arrays, family name.
- `nh_layout_cli_*`, `nh_engine_cli_*`: exit codes of `krait-layout` and `krait-engine` on tiny synthetic files (the engine refuses a broken layout with exit 3 before it touches the GPU).
- `nemotron_serve_test`: serve/test_nemotron_frontend.py and serve/test_nemotron_vision.py (effort mapping, sampling card, picture expansion, encoder flags).

With the model: `krait-vecdot-test --model <shard>` (the CPU-exact expert kernels vs ggml-cpu, bit for bit) and
`krait/bench_serve.py` (HTTP benchmark).

## Names

Krait's names live in `krait/project.env`: PROJECT_NAME (Krait), PROJECT_SLUG (krait: image tag, compose project),
ENGINE_BIN (krait-engine), LAYOUT_BIN (krait-layout), VECDOT_BIN (krait-vecdot-test), VISION_BIN (krait-vision, upstream
Strata's strata-vision target installed under this name). CMakeLists.txt (target names; the binaries print the name
they are built with), krait/build.sh, krait/krait and docker/krait/stage.sh read it. Plain text that names them too:
this file, README.md, THIRD_PARTY_NOTICES.md, docker/krait/compose.yml and krait/config.json (`exe`). Environment:
`KRAIT_ROOT`, `KRAIT_PYTHON`, `KRAIT_CONFIG`, `KRAIT_API_KEY` (krait/krait passes it on as `STRATA_API_KEY`, the
variable the upstream server reads), `KRAIT_TRACE`, and in compose `KRAIT_MODELS`, `KRAIT_IMAGE`, `KRAIT_PORT`. The
upstream code keeps its own names (serve/, strata_cli, strata_control, the C++ namespace `strata`, `STRATA_*` build
options).

## Limits

- One NVIDIA GPU (tested sm_120, 32 GB). Host RAM: the engine's RSS is 81.3 GiB with `--pin`.
- `--batch` is capped at 4, and batch slots decode without MTP (solo requests use it).
- Against `llama-server`, greedy tokens on the real prompts drift apart after 40-197 tokens (this predates speculation).
  The correctness standard is the Gate 2 llama.cpp capture, which the engine matches bit for bit.
