# Third-party notices

Krait: Copyright (c) 2026 the Krait contributors, MIT (LICENSE).

Krait is based on Strata by Niko1221 (github.com/Niko1221/Strata), MIT-licensed (LICENSE, Copyright (c)
2026 Niko1221 and the Strata contributors). It builds on, links or downloads the components below. Each stays under its own
license. This file lists the ones added by the Nemotron 3.5 Super port. Upstream Strata's own third-party code is under
third_party/ with its licenses (for example third_party/ggml/LICENSE).

**Code derived from llama.cpp.** Parts of src/nemotron_h/ (nh_engine.cu's forward pass: the Mamba-2, attention and
LatentMoE blocks; the CPU-exact expert kernels in nh_cpuexact.cuh; the reference code in nhref.cu) were transcribed
from llama.cpp b11806 (`0682a576a`: src/models/nemotron-h.cpp and mamba-base.cpp, src/llama-graph.cpp,
ggml/src/ggml-cpu/ops.cpp and arch/x86/quants.c) so that the engine computes the same thing in the same order. That code is under llama.cpp's MIT license, Copyright (c) 2023-2026
The ggml authors; the MIT notice is kept with it by this file and by the llama.cpp sources krait/build.sh fetches
(their LICENSE file). The pre-tokenizer pattern in tools/strata_tokenizer.py is likewise transcribed from llama.cpp.

| component | used for | license |
|---|---|---|
| llama.cpp / ggml (ggml-org/llama.cpp), b11806 `0682a576a` and `3cf03257` | krait-engine runs on ggml (linked as shared libraries); krait-vision (upstream strata-vision) links mtmd + llama; Strata's other CUDA kernels compile against the pinned commit; fetched by krait/build.sh, not vendored | MIT, Copyright (c) 2023-2026 The ggml authors |
| NVIDIA CUDA runtime, cuBLAS, cuBLASLt | linked by the engine; docker/krait/stage.sh copies libcudart, libcublas and libcublasLt into the locally built image | NVIDIA CUDA Toolkit EULA (these libraries are in its redistributable list) |
| GCC runtime (libstdc++, libgcc_s, libgomp) | linked; copied into the locally built image by stage.sh | GPL-3.0 with the GCC Runtime Library Exception |
| NVIDIA Nemotron 3.5 Super VL 120B-A12B (weights), Unsloth's GGUF conversion | the model; not included; downloaded by the user | OpenMDW License Agreement, version 1.1 (OpenMDW-1.1), per the model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 . Unsloth's GGUF repository carries the model's terms; check its card |
| nomic-embed-text-v1.5 (nomic-ai/nomic-embed-text-v1.5-GGUF) | the embedder in docker/krait/compose.yml; not included | Apache-2.0 |
| Open WebUI (ghcr.io/open-webui/open-webui) | the chat UI in docker/krait/compose.yml; pulled as an image, not included or modified | Open WebUI License (BSD-3-Clause based, with a branding clause) |
| llama.cpp server image (ghcr.io/ggml-org/llama.cpp:server) | the embedder container; pulled, not included | MIT |
| Ubuntu 24.04 base image and its python3 packages (jinja2, regex, numpy, psutil, yaml, requests, pillow, jsonschema) | base of the locally built image | their Debian/Ubuntu package licenses |

The locally built image (`krait:local`) is meant for your own machine. If you redistribute it, the CUDA and
GCC runtime terms above apply to the libraries inside it.
