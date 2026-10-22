# Third-party notices

Strata is MIT-licensed (LICENSE). It builds on, links or downloads the components below. Each stays under its own
license. This file lists the ones added by the Nemotron 3.5 Super port. Upstream Strata's own third-party code is under
third_party/ with its licenses (for example third_party/ggml/LICENSE).

| component | used for | license |
|---|---|---|
| llama.cpp / ggml (ggml-org/llama.cpp), b11806 `0682a576a` and `3cf03257` | strata-nh-engine runs on ggml (linked as shared libraries); strata-vision links mtmd + llama; Strata's other CUDA kernels compile against the pinned commit; fetched by nemotron/build-nemotron.sh, not vendored | MIT, Copyright (c) 2023-2026 The ggml authors |
| NVIDIA CUDA runtime, cuBLAS, cuBLASLt | linked by the engine; docker/nemotron/stage.sh copies libcudart, libcublas and libcublasLt into the locally built image | NVIDIA CUDA Toolkit EULA (these libraries are in its redistributable list) |
| GCC runtime (libstdc++, libgcc_s, libgomp) | linked; copied into the locally built image by stage.sh | GPL-3.0 with the GCC Runtime Library Exception |
| NVIDIA Nemotron 3.5 Super VL 120B-A12B (weights), Unsloth's GGUF conversion | the model; not included; downloaded by the user | the license on the model's card (NVIDIA's model license); see the model page |
| nomic-embed-text-v1.5 (nomic-ai/nomic-embed-text-v1.5-GGUF) | the embedder in docker/nemotron/compose.yml; not included | Apache-2.0 |
| Open WebUI (ghcr.io/open-webui/open-webui) | the chat UI in docker/nemotron/compose.yml; pulled as an image, not included or modified | Open WebUI License (BSD-3-Clause based, with a branding clause) |
| llama.cpp server image (ghcr.io/ggml-org/llama.cpp:server) | the embedder container; pulled, not included | MIT |
| Ubuntu 24.04 base image and its python3 packages (jinja2, regex, numpy, psutil, yaml, requests, pillow, jsonschema) | base of the locally built image | their Debian/Ubuntu package licenses |

The locally built image (`strata-nemotron:local`) is meant for your own machine. If you redistribute it, the CUDA and
GCC runtime terms above apply to the libraries inside it.
