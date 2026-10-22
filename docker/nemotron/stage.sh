#!/usr/bin/env bash
# Builds the Strata Nemotron image (strata-nemotron:local) from a local build: stages strata-nh-engine, strata-vision,
# the shared libraries they load (as ldd resolves them here; libcuda comes from the NVIDIA container runtime), the
# server's Python sources and the tokenizer pack into a build context, then runs docker build.  The model is not in
# the image: compose.yml mounts it.
#   NH_ROOT=<dir with build/ and pack/>  docker/nemotron/stage.sh [context dir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA="$(cd "$HERE/../.." && pwd)"
NH_ROOT="${NH_ROOT:-$(cd "$STRATA/../.." && pwd)}"
CTX="${1:-$NH_ROOT/_tmp/docker-nemotron}"
TAG="${STRATA_IMAGE:-strata-nemotron:local}"
if [ -e "$CTX" ]; then mv "$CTX" "$CTX.old.$(date +%Y%m%d-%H%M%S)"; fi   # never deleted; old contexts can be removed by hand
O="$CTX/opt/strata"
mkdir -p "$O/libexec" "$O/lib/engine" "$O/lib/vision" "$O/build/strata" "$O/build/vision" "$O/pack/nemotron" "$O/src/strata" "$O/logs"
copy_libs() {  # $1 binary, $2 destination: every resolved library except glibc's and the driver's
  ldd "$1" | awk '/=> \//{print $3}' | while read -r l; do
    case "$(basename "$l")" in
      libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|ld-linux*|libcuda.so*|libnvidia*|libdxcore*) ;;
      *) cp -L "$l" "$2/" ;;
    esac
  done
}
for b in strata/strata-nh-engine vision/strata-vision; do
  [ -x "$NH_ROOT/build/$b" ] || { echo "stage: $NH_ROOT/build/$b is missing (build it first)" >&2; exit 1; }
done
cp "$NH_ROOT/build/strata/strata-nh-engine" "$O/libexec/"; copy_libs "$NH_ROOT/build/strata/strata-nh-engine" "$O/lib/engine"
cp "$NH_ROOT/build/vision/strata-vision" "$O/libexec/"; copy_libs "$NH_ROOT/build/vision/strata-vision" "$O/lib/vision"
# the two binaries load different ggml builds with the same sonames: each gets its own library directory
printf '#!/bin/sh\nLD_LIBRARY_PATH=/opt/strata/lib/engine exec /opt/strata/libexec/strata-nh-engine "$@"\n' > "$O/build/strata/strata-nh-engine"
printf '#!/bin/sh\nLD_LIBRARY_PATH=/opt/strata/lib/vision exec /opt/strata/libexec/strata-vision "$@"\n' > "$O/build/vision/strata-vision"
chmod +x "$O/build/strata/strata-nh-engine" "$O/build/vision/strata-vision"
cp -r "$NH_ROOT/pack/nemotron/tokenizer" "$O/pack/nemotron/"
( cd "$STRATA" && git ls-files -z serve strata_control strata_cli tools/strata_tokenizer.py nemotron \
    | grep -zv -e '^serve/tests/' -e '\.exe$' -e '\.dll$' | xargs -0 cp --parents -t "$O/src/strata" )
cp "$HERE/Dockerfile" "$CTX/"
du -sh "$CTX"
docker build -t "$TAG" "$CTX"
