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

The standard is llama.cpp b11806 (`0682a576a`) itself, through a "golden" capture made by a reference tool built on
it. All of these were measured (raw numbers: [BENCHMARKS.md](BENCHMARKS.md#exactness-checks)).

| check | result |
|---|---|
| 32 test prompts, greedy, vs the golden capture | 32/32 token-identical; 42,075/42,075 logits rows bit-identical (also with MTP speculation on) |
| speculation on vs off (`--spec-check`, every logits row hashed) | identical for MTP k = 1, 2, 3 and n-gram k = 1, 2, 3 |
| server: served greedy tokens vs the CLI | solo 6/6, `--batch 4` 6/6, 1-4 concurrent sessions all identical, HTTP 13/13 |
| pictures: engine vs `llama-mtmd-cli` (2 images, solo and batched) | 4/4 identical |

This is exactness against the golden capture, for greedy decoding. Against `llama-server` b11806, greedy text on 6 real
prompts drifts apart after 40 to 197 tokens (this predates speculation). Sampled output is random by design.

## Measured speed

One RTX 5090 (VRAM capped at 29,346 MiB) + Ryzen 9 7950X3D, WSL2 with 116,713 MiB of RAM, greedy decoding: 23.0 tok/s
on real prompts from the CLI (llama.cpp b11806 on the same machine: 14.3 tok/s), 25.0 tok/s for one HTTP stream, a 32K
prompt in 63.4 s. MTP speculation is used only for greedy requests decoded alone; batch slots and sampled requests
decode without it, and sampled speed was not measured. Methods and every raw number: [BENCHMARKS.md](BENCHMARKS.md).

More requests at once do not raise the total: every extra row needs its own experts, and 68 MoE layers run on the
CPU. `--batch` lets several users share the model; it does not make it faster.

## Requirements

- One NVIDIA GPU with 32 GB (tested: RTX 5090, sm_120). The engine sizes its expert cache from the free VRAM (minus
  `--vram-reserve-mib`, 2,048 MiB) or from `--vram-limit-mib`. If too few cache slots fit for the configured
  `--batch` / `--spec-k`, it refuses to start and says how many slots it has and needs; lower `--batch`, `--spec-k`
  or `--max-context`, or free VRAM. Smaller cards are untested.
- Host RAM: engine resident set 81.3-81.8 GiB measured, of which `--pin` locks 72,261.3 MiB; tested only with
  116,713 MiB visible to Linux. With `--pin` the engine first reads the RAM available to it (MemAvailable and the
  cgroup limit) and locks only what fits while leaving `--pin-reserve-mib` (default 8,192 MiB) free; if not everything
  fits it prints how much it locked, and the rest stays unpinned (slower). Without `--pin` it warns if the host experts
  are larger than the available RAM.
- WSL2: the VM gets half the Windows RAM by default; raise `memory=` under `[wsl2]` in `%UserProfile%\.wslconfig` and
  run `wsl --shutdown` once.
- `--pin` may need a memlock limit of unlimited (`ulimit -l unlimited`; compose.yml sets it for the container). If
  page-locking fails, the engine prints the CUDA error and the rest of the experts stay pageable.

## Build

Linux (or WSL2) with a CUDA toolkit, cmake >= 3.21, ninja, gcc, curl and python3:

```bash
krait/build.sh     # KRAIT_ROOT=<dir> CUDA_ARCH=120 JOBS=16 KRAIT_NATIVE=1 (defaults: this repository, 120, nproc, 1)
```

It fetches llama.cpp b11806 (`0682a576a`, the engine's ggml and the encoder's mtmd) and the commit `setup.py` pins
for the rest of Strata (`3cf03257`) into `KRAIT_ROOT/third_party/`, then builds `build/ggml-nh`,
`build/krait/krait-engine`, `build/krait/krait-layout` and `build/vision/krait-vision` under KRAIT_ROOT.

Portability: `KRAIT_NATIVE=1` (default, the tested build) compiles ggml's CPU code for this machine's CPU
(`GGML_NATIVE=ON`, and the vision encoder with `STRATA_PORTABLE=OFF`); the binaries, and a Docker image staged from
them, may not run on a different CPU. `KRAIT_NATIVE=0` uses ggml's generic x86-64 flags (`GGML_NATIVE=OFF`,
`STRATA_PORTABLE=ON`) for an image meant to run elsewhere; that build is untested and its speed unmeasured.

Two llama.cpp versions on purpose: Strata's own CUDA kernels (src/prefill/moe_mmq.cu and the other MMQ files) compile
only against the commit setup.py pins. Pointing `STRATA_GGML_DIR` at a newer llama.cpp breaks them (for example
`ggml_cuda_host.cu: declaration is incompatible with ... new_pool_for_device`). The Nemotron engine needs b11806, so it
takes its ggml from `KRAIT_GGML_INCLUDE` / `KRAIT_GGML_LIBDIR` instead.

The model and the tokenizer pack:

```
$KRAIT_MODELS/UD-Q4_K_XL/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q4_K_XL-0000{1,2,3}-of-00003.gguf
$KRAIT_MODELS/mmproj-BF16.gguf
python3 tools/strata_tokenizer.py --gguf <shard 1> --out $KRAIT_ROOT/pack/nemotron    # writes pack/nemotron/tokenizer/
build/krait/krait-layout --model <shard 1> --mmproj $KRAIT_MODELS/mmproj-BF16.gguf      # checks every tensor
```

`KRAIT_MODELS` defaults to `$KRAIT_ROOT/models`.

## Run

```bash
krait/krait                                       # http://127.0.0.1:8095/v1
KRAIT_API_KEY=<key> HOST=0.0.0.0 krait/krait      # on the network: a key is required
```

`krait/krait` runs `krait/launch_config.py` first. It checks the Python packages (`pip install -r requirements.txt`
into `.venv`, which the launcher uses if present, or `KRAIT_PYTHON`), the binaries, the three shards and the mmproj,
and the tokenizer pack, and prints what to do for anything missing. It refuses a non-loopback `HOST` without an API key
(`KRAIT_ALLOW_OPEN=1` overrides that) and any key that starts with `change-me`; the server refuses such keys too. It
then writes the config the server runs with to `$KRAIT_ROOT/logs/krait-run.json`.

| environment | default | meaning |
|---|---|---|
| `KRAIT_ROOT` | the repository | holds build/, pack/, logs/ and krait.local.json; config paths are relative to it |
| `KRAIT_MODELS` | `$KRAIT_ROOT/models` | the model directory (config paths that start with `models/` point here) |
| `KRAIT_PYTHON` | `.venv/bin/python3`, else `python3` | the server's Python |
| `KRAIT_CONFIG` | `krait/config.json` | the base config |
| `KRAIT_LOCAL_CONFIG` | `$KRAIT_ROOT/krait.local.json` if present | this machine's engine flags: `{"engine_args": [...]}` (it may also set `max_queue`, `queue_timeout`, `max_tokens_cap`, `request_timeout`); see `krait/krait.local.example.json` |
| `KRAIT_ENGINE_ARGS` | - | more engine flags (`--flag value ...`), applied last |
| `KRAIT_THREADS` | physical cores | `--threads auto` and the encoder's `"threads": "auto"` |
| `KRAIT_API_KEY` | - | the API key (passed to the server as `STRATA_API_KEY`) |
| `HOST`, `PORT` | 127.0.0.1, 8095 | where the server listens |
| `KRAIT_IMAGE_FILES=1`, `KRAIT_IMAGE_URLS=1` | off | allow pictures from local files / from public URLs (below) |
| `KRAIT_TRACE=<file>` | - | the engine appends each request's prompt and generated token ids to that file as one JSON line. Debugging only: the ids decode to the full conversation, so the file holds every user's prompts and answers in effect. Keep it off in production |
| `STRATA_MAX_BODY_MIB` | the server's default | largest request body |
| `STRATA_ALLOWED_HOSTS` | - | extra Host names the server answers to without an API key (DNS-rebinding guard) |

`krait/config.json` holds the engine flags (paths relative to KRAIT_ROOT) and the server limits. Its `vision` section
turns pictures on. The encoder runs on the CPU (`"flash_attn": "on"`, which matches llama.cpp's `-fa on`), so it uses
no VRAM.

| engine flag | in the config | meaning |
|---|---|---|
| `--n-cpu-moe 68` | yes | the routed experts of the first 68 MoE layers stay in host RAM (read by the CPU or cached on the card) |
| `--cache all` | yes | a VRAM cache of hot experts for those layers (measured: 123 slots per layer alone, 111 with `--batch 4`, under a 29,346 MiB cap) |
| `--pin` | yes | page-locks the host expert ranges that fit (see Requirements) |
| `--pin-reserve-mib N` | 8192 | host RAM `--pin` leaves free |
| `--spec mtp --spec-k 1 --spec-min-acc 0.6` | yes | MTP speculation with the model's own draft block, for greedy requests decoded alone; drafting pauses while the acceptance average is under 0.6 |
| `--chunk 512`, `--max-context 32768`, `--threads auto` | yes | prompt chunk, context, CPU threads |
| `--vram-limit-mib N` | no | keep the engine's whole VRAM use under N MiB (e.g. on a card that also drives a desktop; every published number used 29346) |
| `--vram-reserve-mib N` | 2048 | without a limit: VRAM left free |
| `--batch 2..4` | 4 | up to 4 requests decode together (rows mode, no MTP); without it requests take turns |
| `--prefix-reuse on\|off` | **off** | see below |

| server setting (config.json) | shipped | meaning |
|---|---|---|
| `max_queue` | 8 | requests that may wait beyond the ones decoding; more are refused |
| `queue_timeout` | 300 | seconds a request may wait in the queue |
| `max_tokens_cap` | 8192 | the most tokens one request may generate (a larger `max_tokens` is lowered to it; also the default) |
| `request_timeout` | 1800 | seconds after which a request is stopped |
| `image_files`, `image_urls` | false | picture sources besides `data:` URLs (also `--image-files`, `--image-urls`) |

A request the engine cannot process (for example a corrupt picture) gets an error; the engine resets that request's
slot and keeps serving the others. Only a CUDA error stops the engine.

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

Picture sources: by default only inline `data:` URLs. With `image_files` (`KRAIT_IMAGE_FILES=1`) the server also opens
local paths, with the server's own file permissions. With `image_urls` (`KRAIT_IMAGE_URLS=1`) it downloads http(s)
URLs, but only from public addresses: a host name that resolves to a private, loopback, link-local, multicast or
otherwise non-global address is refused, and so is every redirect to one. The download then connects to the
address that was checked (the Host header, TLS SNI and the certificate check still use the name), so a name that
answers differently on a second lookup (DNS rebinding) cannot redirect it; it ignores HTTP(S)_PROXY. Change from
upstream Strata: `serve/server.py` itself now treats a config without `image_files` / `image_urls` as off (upstream
allowed both by default).

## Docker Compose (Krait + Open WebUI + embedder for RAG)

`docker/krait/`: `stage.sh` builds a runtime image (`krait:local`) from your build (its context goes to a temporary
directory that is removed afterwards; `KRAIT_KEEP_CONTEXT=1` keeps it), and `compose.yml` starts Krait (one GPU), Open
WebUI and a llama.cpp embedding server (CPU, nomic-embed-text-v1.5) that Open WebUI uses for document search. The
Krait container runs as an unprivileged user (uid 10001). Images are pinned by digest. Steps: `cp
docker/krait/env.example docker/krait/.env` (set KRAIT_MODELS and the three keys, for example from `openssl rand -hex
24`; machine flags such as a VRAM cap go in `KRAIT_ENGINE_ARGS`), run `docker/krait/stage.sh`, then `docker compose -f
docker/krait/compose.yml up -d`, then open http://127.0.0.1:3095.

`docker/krait/smoke_openwebui.py` checks the stack through Open WebUI's API: chat, streaming, a picture, and a document
upload -> embedding -> retrieval -> answer round trip. It makes a real account, so it needs a password:
`KRAIT_SMOKE_PASSWORD=<12+ characters> python3 docker/krait/smoke_openwebui.py`. It passed 8/8 on the test machine
with Docker Engine 29.8.1 in WSL2.

Upgrading from an image that ran as root: the `krait-logs` volume it made is owned by root, and the new image cannot
write to it. Recreate the volume (`docker compose -f docker/krait/compose.yml down -v` deletes it, together with Open
WebUI's data volume) or chown it to 10001.

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
the table under Run (krait/krait passes `KRAIT_API_KEY` on as `STRATA_API_KEY`, the variable the upstream server
reads), and in compose `KRAIT_MODELS`, `KRAIT_IMAGE`, `KRAIT_PORT`, `KRAIT_ENGINE_ARGS`. Build: `KRAIT_ROOT`,
`KRAIT_NATIVE`, `KRAIT_LLAMA_DIR`, `KRAIT_GGML_INCLUDE` / `KRAIT_GGML_LIBDIR`; stage.sh: `KRAIT_STAGE_DIR`,
`KRAIT_KEEP_CONTEXT`. The
upstream code keeps its own names (serve/, strata_cli, strata_control, the C++ namespace `strata`, `STRATA_*` build
options).

## Limits

- One NVIDIA GPU (tested: sm_120, 32 GB). Host RAM: the engine's resident set was 81.3-81.8 GiB (see Requirements).
- `--batch` is capped at 4, and batch slots decode without MTP (greedy requests decoded alone use it).
- Against `llama-server`, greedy tokens on the real prompts drift apart after 40-197 tokens (this predates speculation).
  The correctness standard is the golden llama.cpp b11806 capture, which the engine matches bit for bit.
- Measured on one machine only (BENCHMARKS.md).
