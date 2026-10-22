# Krait: fast local inference for NVIDIA Nemotron 3.5 Super (hybrid Mamba-2/MoE)

Krait is based on [Strata](https://github.com/Niko1221/Strata) by Niko1221 (MIT).

Krait runs **NVIDIA Nemotron 3.5 Super VL 120B-A12B** (GGUF architecture `nemotron_h_moe`: Mamba-2 + attention +
LatentMoE, with one MTP block; Unsloth's UD-Q4_K_XL GGUF) on a single 32 GB NVIDIA card, with the routed experts of
most layers in host RAM. Its engine, `krait-engine`, runs on llama.cpp's ggml kernels (b11806) with its own graph,
expert cache, scheduler and MTP speculation. It serves text and pictures over the OpenAI and Anthropic APIs, with up to
4 requests at once.

**Exactness, as measured:** with greedy decoding, Krait's tokens and logits are identical to a capture from a
reference tool built on llama.cpp b11806 (the "golden" capture) on 32 test prompts. Against `llama-server` the greedy text
can drift apart after some tokens (on 6 real prompts: after 40 to 197 tokens). Details: [docs/KRAIT.md](docs/KRAIT.md#correctness).

**Speed, as measured** (one RTX 5090 32 GB with the engine capped at 29,346 MiB of VRAM, Ryzen 9 7950X3D, WSL2 with
116,713 MiB of RAM visible; greedy decoding, the only mode that uses MTP speculation; sampled requests decode without
it and were not benchmarked): **23.0 tok/s decode on 6 real prompts, 1.61x llama.cpp b11806's 14.3 tok/s** on the same
machine; one HTTP stream 25.0 tok/s; a 32K-token prompt is read in 63.4 s (llama.cpp: 156.9 s). Raw numbers and
methods: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Requirements

- An NVIDIA GPU with 32 GB of VRAM (tested: RTX 5090, sm_120 only) and a CUDA toolkit that supports it (sm_120 needs
  CUDA 12.8 or newer; tested with 12.9). The engine
  refuses to start, and says why, if its expert cache does not fit; cards with less VRAM are untested.
- Host RAM: the engine keeps the routed experts of 68 layers in RAM. Measured: engine resident set 81.3-81.8 GiB, of
  which `--pin` page-locks 72,261.3 MiB; tested only with 116,713 MiB of RAM visible to Linux. Before pinning, the
  engine checks the RAM available to it and locks only what fits while leaving `--pin-reserve-mib` (default 8,192 MiB)
  free; if not everything fits it says so and runs with the rest unpinned (slower).
- WSL2: the WSL VM gets only part of the Windows RAM by default (half). Give it more in `%UserProfile%\.wslconfig`, for
  example `[wsl2]` / `memory=116GB` on a 128 GB PC, then `wsl --shutdown` once. (Docker Desktop has its own memory
  setting.)
- Linux or WSL2 with a CUDA toolkit (nvcc), cmake >= 3.21, ninja, gcc/g++, curl, and Python 3.10+ with venv.

## Quick start

```bash
git clone -b port/nemotron <this repository> krait && cd krait
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
krait/build.sh     # fetches llama.cpp b11806 + the pinned commit; builds krait-engine, krait-layout, krait-vision
                   # (CUDA_ARCH=120 by default; set it for another card - untested)

# the model (not included): Unsloth's NVIDIA-Nemotron-3.5-Super-VL-120B-A12B GGUF
#   $KRAIT_MODELS/UD-Q4_K_XL/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q4_K_XL-0000{1,2,3}-of-00003.gguf
#   $KRAIT_MODELS/mmproj-BF16.gguf
export KRAIT_MODELS=/path/to/models

# the tokenizer pack (written to pack/nemotron/tokenizer/)
.venv/bin/python3 tools/strata_tokenizer.py \
  --gguf "$KRAIT_MODELS/UD-Q4_K_XL/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q4_K_XL-00001-of-00003.gguf" \
  --out pack/nemotron

krait/krait        # http://127.0.0.1:8095/v1 (OpenAI) and /v1/messages (Anthropic)
```

Then, from another shell:

```bash
curl http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":64}'
```

`krait/krait` checks the install first and names whatever is missing (Python packages, binaries, model files, the
tokenizer pack). Its settings are environment variables listed at the top of `krait/krait` (`KRAIT_ROOT`,
`KRAIT_MODELS`, `KRAIT_PYTHON`, `KRAIT_THREADS`, `KRAIT_ENGINE_ARGS`, ...). Engine flags that belong to one machine,
such as a VRAM cap on a card that also drives a desktop, go in `krait.local.json` (see
`krait/krait.local.example.json`; git ignores the copy).

On the network: `KRAIT_API_KEY=<long random key> HOST=0.0.0.0 krait/krait`. Krait refuses a non-loopback HOST without
a key, and refuses placeholder keys (`change-me...`).

Pictures: by default the server accepts pictures only inline as `data:` URLs. `KRAIT_IMAGE_FILES=1` allows local file
paths, `KRAIT_IMAGE_URLS=1` allows downloads from public http(s) addresses (private, loopback and link-local addresses
are refused, also after a redirect).

With Open WebUI and a RAG embedder (Docker Engine + NVIDIA Container Toolkit):

```bash
cp docker/krait/env.example docker/krait/.env     # set KRAIT_MODELS and the keys
docker/krait/stage.sh                              # builds krait:local from your build
docker compose -f docker/krait/compose.yml up -d   # Open WebUI on http://127.0.0.1:3095
```

The image contains your build's binaries. `krait/build.sh` builds for the CPU it runs on by default
(`KRAIT_NATIVE=1`), so the image may not run on a machine with an older CPU; `KRAIT_NATIVE=0` builds generic x86-64
code (untested).

## `--prefix-reuse`

The shipped config (`krait/config.json`) uses `--prefix-reuse off`: every request reads its whole prompt, so served
greedy output equals the CLI token for token. `--prefix-reuse on` keeps a conversation's state between turns and reads
only the new tokens. Follow-ups are faster, but the output is not bit-identical to a fresh read (measured: 4 of 6 second
turns differed; with `off`, 12/12 identical). Sessions that hold a picture are never reused.

## More

- Build, flags, server limits, correctness checks, Docker and tests: [docs/KRAIT.md](docs/KRAIT.md).
- Measured numbers: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
- Names (binaries, image, environment) live in `krait/project.env`.
- Tests: `cmake -DSTRATA_BUILD_TESTS=ON ...` then `ctest -R "nemotron|nh_"` (no model needed).
- Third-party licenses, including the model's: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- This repository still holds upstream Strata (its Qwen engine, app and tools). Krait changed the shared server and
  vision encoder where Nemotron needed it (see docs/KRAIT.md). Upstream's README is
  [STRATA_README.md](STRATA_README.md).

## License

MIT ([LICENSE](LICENSE)). Krait is based on Strata by Niko1221 (MIT), Copyright (c) 2026 Niko1221 and the Strata
contributors. The model weights are not part of Krait and have their own license (OpenMDW-1.1, see
THIRD_PARTY_NOTICES.md).
