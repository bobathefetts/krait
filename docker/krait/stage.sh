#!/usr/bin/env bash
# Builds the runtime image (<PROJECT_SLUG>:local, krait/project.env) from a local build: stages the engine, the vision encoder,
# the shared libraries they load (as ldd resolves them here; libcuda comes from the NVIDIA container runtime), the
# server's Python sources and the tokenizer pack into a build context, then runs docker build.  The model is not in
# the image: compose.yml mounts it.
#   KRAIT_ROOT=<dir with build/ and pack/>  docker/krait/stage.sh [context dir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KRAIT_ROOT="${KRAIT_ROOT:-$(cd "$REPO/../.." && pwd)}"
. "$REPO/krait/project.env"
CTX="${1:-$KRAIT_ROOT/_tmp/docker-krait}"
TAG="${KRAIT_IMAGE:-$PROJECT_SLUG:local}"
if [ -e "$CTX" ]; then mv "$CTX" "$CTX.old.$(date +%Y%m%d-%H%M%S)"; fi   # never deleted; old contexts can be removed by hand
O="$CTX/opt/krait"
mkdir -p "$O/libexec" "$O/lib/engine" "$O/lib/vision" "$O/build/krait" "$O/build/vision" "$O/pack/nemotron" "$O/src/krait" "$O/logs"
copy_libs() {  # $1 binary, $2 destination: every resolved library except glibc's and the driver's
  ldd "$1" | awk '/=> \//{print $3}' | while read -r l; do
    case "$(basename "$l")" in
      libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|ld-linux*|libcuda.so*|libnvidia*|libdxcore*) ;;
      *) cp -L "$l" "$2/" ;;
    esac
  done
}
for b in "krait/$ENGINE_BIN" "vision/$VISION_BIN"; do
  [ -x "$KRAIT_ROOT/build/$b" ] || { echo "stage: $KRAIT_ROOT/build/$b is missing (build it first)" >&2; exit 1; }
done
cp "$KRAIT_ROOT/build/krait/$ENGINE_BIN" "$O/libexec/"; copy_libs "$KRAIT_ROOT/build/krait/$ENGINE_BIN" "$O/lib/engine"
cp "$KRAIT_ROOT/build/vision/$VISION_BIN" "$O/libexec/"; copy_libs "$KRAIT_ROOT/build/vision/$VISION_BIN" "$O/lib/vision"
# the two binaries load different ggml builds with the same sonames: each gets its own library directory
printf '#!/bin/sh\nLD_LIBRARY_PATH=/opt/krait/lib/engine exec /opt/krait/libexec/%s "$@"\n' "$ENGINE_BIN" > "$O/build/krait/$ENGINE_BIN"
printf '#!/bin/sh\nLD_LIBRARY_PATH=/opt/krait/lib/vision exec /opt/krait/libexec/%s "$@"\n' "$VISION_BIN" > "$O/build/vision/$VISION_BIN"
chmod +x "$O/build/krait/$ENGINE_BIN" "$O/build/vision/$VISION_BIN"
cp -r "$KRAIT_ROOT/pack/nemotron/tokenizer" "$O/pack/nemotron/"
( cd "$REPO" && git ls-files -z serve strata_control strata_cli tools/strata_tokenizer.py krait \
    | grep -zv -e '^serve/tests/' -e '\.exe$' -e '\.dll$' | xargs -0 cp --parents -t "$O/src/krait" )
cp "$HERE/Dockerfile" "$CTX/"
du -sh "$CTX"
docker build -t "$TAG" "$CTX"
