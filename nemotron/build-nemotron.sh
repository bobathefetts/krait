#!/usr/bin/env bash
# Builds the Nemotron 3.5 Super pieces from source on Linux with CUDA (docs/NEMOTRON.md):
#   1. llama.cpp b11806 (0682a576a) - the ggml the engine runs on (shared CUDA + CPU libraries) and the mtmd the
#      vision encoder uses (nemotron_omni projector);
#   2. strata-nh-engine + strata-nh-layout (this repository's CMakeLists.txt; its other targets need STRATA_GGML_DIR =
#      llama.cpp at the commit setup.py pins, 3cf03257, which is fetched too);
#   3. strata-vision (tools/vision, CPU build: the server runs the encoder on the CPU).
# The results land where nemotron/strata-nemotron.json expects them, relative to NH_ROOT: build/strata/strata-nh-engine,
# build/vision/strata-vision.
#
#   NH_ROOT=<dir>  CUDA_ARCH=120  JOBS=16  nemotron/build-nemotron.sh [extra cmake args for the CUDA builds]
#   (cmake >= 3.21, ninja, a CUDA toolkit (nvcc on PATH or CUDACXX), gcc/g++, python3 for unpacking)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA="$(cd "$HERE/.." && pwd)"
NH_ROOT="${NH_ROOT:-$(cd "$STRATA/../.." && pwd)}"
ARCH="${CUDA_ARCH:-120}"
JOBS="${JOBS:-$(nproc)}"
TP="$NH_ROOT/third_party"
NH_LLAMA_COMMIT=0682a576a772de436caa5d1c1e94036486d6ebd0                       # llama.cpp b11806
PIN_LLAMA_COMMIT=$(sed -n 's/^LLAMA_CPP_COMMIT = "\([0-9a-f]*\)"/\1/p' "$STRATA/setup.py")
LLAMA_NH="${LLAMA_NH_DIR:-$TP/llama.cpp-$NH_LLAMA_COMMIT}"
LLAMA_PIN="${STRATA_GGML_DIR:-$TP/llama.cpp-$PIN_LLAMA_COMMIT}"
B="$NH_ROOT/build"

fetch() {   # $1 commit, $2 directory: llama.cpp's source archive at that commit, unpacked
  [ -d "$2/ggml" ] && return
  mkdir -p "$TP"
  local zip="$TP/llama.cpp-$1.zip"
  [ -f "$zip" ] || curl -fL --retry 3 -o "$zip" "https://github.com/ggml-org/llama.cpp/archive/$1.zip"
  python3 -m zipfile -e "$zip" "$TP"
  [ "$TP/llama.cpp-$1" = "$2" ] || mv "$TP/llama.cpp-$1" "$2"
}
fetch "$NH_LLAMA_COMMIT" "$LLAMA_NH"
fetch "$PIN_LLAMA_COMMIT" "$LLAMA_PIN"

echo "== 1/3 ggml (llama.cpp b11806, shared, CUDA $ARCH + CPU) -> $B/ggml-nh"
cmake -S "$LLAMA_NH" -B "$B/ggml-nh" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_CURL=OFF "$@"
cmake --build "$B/ggml-nh" -j "$JOBS" --target ggml ggml-base ggml-cpu ggml-cuda

echo "== 2/3 strata-nh-engine, strata-nh-layout -> $B/strata"
cmake -S "$STRATA" -B "$B/strata" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DSTRATA_GGML_DIR="$LLAMA_PIN" \
  -DSTRATA_NH_GGML_INCLUDE="$LLAMA_NH/ggml/include" -DSTRATA_NH_GGML_LIBDIR="$B/ggml-nh/bin" "$@"
cmake --build "$B/strata" -j "$JOBS" --target strata-nh-engine strata-nh-layout

echo "== 3/3 strata-vision (CPU, llama.cpp b11806 mtmd) -> $B/vision"
cmake -S "$STRATA/tools/vision" -B "$B/vision" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$LLAMA_NH" \
  -DSTRATA_PORTABLE=OFF
cmake --build "$B/vision" -j "$JOBS" --target strata-vision

ls -l "$B/strata/strata-nh-engine" "$B/strata/strata-nh-layout" "$B/vision/strata-vision"
echo "done: NH_ROOT=$NH_ROOT nemotron/run-nemotron.sh (models in $NH_ROOT/models, tokenizer pack in $NH_ROOT/pack/nemotron)"
