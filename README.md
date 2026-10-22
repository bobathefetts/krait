# Krait: fast local inference for NVIDIA Nemotron 3.5 Super (hybrid Mamba-2/MoE)

Krait is based on [Strata](https://github.com/Niko1221/Strata) by Niko1221 (MIT).

Krait runs **NVIDIA Nemotron 3.5 Super VL 120B-A12B** (GGUF architecture `nemotron_h_moe`: Mamba-2 + attention +
LatentMoE, with one MTP block; Unsloth's UD-Q4_K_XL GGUF) on a single 32 GB NVIDIA card, with the routed experts of
most layers in host RAM. Its engine, `krait-engine`, runs on llama.cpp's ggml kernels (b11806) with its own graph,
expert cache, scheduler and MTP speculation. It serves text and pictures over the OpenAI and Anthropic APIs, with up to
4 requests at once. Greedy output is token-identical to llama.cpp b11806.

Measured on an RTX 5090 + Ryzen 9 7950X3D (127 GB RAM, VRAM capped at 29,346 MiB): **23.0 tok/s decode on real
prompts, 1.61x llama.cpp's 14.3 tok/s on the same machine**. Over HTTP it does 24.6 tok/s, and it reads a 32K prompt in
63.4 s (llama.cpp: 156.9 s).

## Quick start (Linux or WSL2, CUDA)

```bash
krait/build.sh            # fetches llama.cpp b11806 + the pinned commit; builds krait-engine, krait-layout, krait-vision
# models: KRAIT_ROOT/models/UD-Q4_K_XL/*.gguf (3 shards) and KRAIT_ROOT/models/mmproj-BF16.gguf; tokenizer pack: docs/KRAIT.md
krait/krait               # http://127.0.0.1:8095/v1 (OpenAI) and /v1/messages (Anthropic)
KRAIT_API_KEY=<key> HOST=0.0.0.0 krait/krait      # on the network: set a key
```

With Open WebUI and a RAG embedder (Docker Engine + NVIDIA Container Toolkit):

```bash
cp docker/krait/env.example docker/krait/.env     # set KRAIT_MODELS and the keys
docker/krait/stage.sh                              # builds krait:local from your build
docker compose -f docker/krait/compose.yml --env-file docker/krait/.env up -d   # Open WebUI on http://127.0.0.1:3095
```

## `--prefix-reuse`

The shipped config (`krait/config.json`) uses `--prefix-reuse off`: every request reads its whole prompt, so served
output equals the CLI bit for bit. `--prefix-reuse on` keeps a conversation's state between turns and reads only the
new tokens. Follow-ups are faster, but the output is not bit-identical to a fresh read (measured: 4 of 6 second turns
differed; with `off`, 12/12 identical). Sessions that hold a picture are never reused.

## More

- Build, flags, measured speed, correctness checks, Docker and tests: [docs/KRAIT.md](docs/KRAIT.md).
- Names (binaries, image, environment) live in `krait/project.env`.
- Tests: `cmake -DSTRATA_BUILD_TESTS=ON ...` then `ctest -R "nemotron|nh_"` (no model needed).
- Third-party licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- This repository still holds upstream Strata (its Qwen engine, app and tools). Krait changed the shared server and
  vision encoder where Nemotron needed it (see docs/KRAIT.md). Upstream's README is
  [STRATA_README.md](STRATA_README.md).

## License

MIT ([LICENSE](LICENSE)). Krait is based on Strata by Niko1221 (MIT), Copyright (c) 2026 Niko1221 and the Strata
contributors.
