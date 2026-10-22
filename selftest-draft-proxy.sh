#!/usr/bin/env bash
# Build and run THIS fork's draft_proxy parity selftest: correctness over the shipped 40,525-token draft
# head, one tile past the old 49,152-row cap (49,184, the shape that used to throw "too many rows"), and
# the 106,299 CJK subset - for Q6_K and Q8_0 - plus the copy-vs-exact draft-head timing at each size.
#
# It can't run inside `docker build` (no GPU) or in the runtime image (built STRATA_BUILD_TESTS=OFF), so
# this runs it in a CUDA devel image with --gpus all. A cached dev image (docker/Dockerfile.selftest) and
# a gitignored build-selftest/ cache make repeat runs incremental: only what changed recompiles.
#
#   sudo ./selftest-draft-proxy.sh                     # build + run draft_proxy_parity --selftest
#   sudo CUDA_ARCH=89-real ./selftest-draft-proxy.sh   # one card (86 = RTX 30, 89 = RTX 40); default both
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_IMAGE="${DEV_IMAGE:-strata-selftest-dev:latest}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:13.0.0-devel-ubuntu24.04}"
ARCH="${CUDA_ARCH:-86-real;89-real}"

# bake the deps + venv once (source and the build cache are mounted at run time, so this rarely rebuilds)
if ! docker image inspect "$DEV_IMAGE" >/dev/null 2>&1; then
  echo "building the selftest dev image $DEV_IMAGE (once) ..."
  docker build -f "$HERE/docker/Dockerfile.selftest" --build-arg CUDA_IMAGE="$CUDA_IMAGE" -t "$DEV_IMAGE" "$HERE"
fi

mkdir -p "$HERE/build-selftest"
exec docker run --rm --gpus all \
  -e "CUDA_ARCH=$ARCH" \
  -v "$HERE":/src -v "$HERE/build-selftest":/src/build -w /src "$DEV_IMAGE" bash -c '
    set -e
    [ -d third_party/llama.cpp ] || /venv/bin/python -c "import setup; setup.get_llama_cpp()"
    if [ ! -f build/CMakeCache.txt ]; then
      cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DSTRATA_GGML_DIR=/src/third_party/llama.cpp
    fi
    cmake --build build --target draft_proxy_parity -j"$(nproc)"
    ./build/draft_proxy_parity --selftest
  '
