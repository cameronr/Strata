# Lamina in a container (Linux, NVIDIA GPUs of the RTX 30 series or newer, CUDA 13). The first stage builds the engine
# and the image encoder with setup.py for the CUDA architectures below (docker build has no GPU to read them from); the
# runtime stage starts with setup.py --no-build, which downloads and prepares the model in /data once and checks it on
# later starts, then serves it on port 8080 with the API key the container is given.
#
#   docker build -t lamina .                                                       (RTX 30, 40 and 50)
#   docker build -t lamina --build-arg CUDA_ARCHITECTURES="86-real;86-virtual" .   (RTX 30 only: a faster build)
#   docker build -t lamina --build-arg JOBS=2 .                                    (a VM with little RAM)
#   docker run --rm --gpus all --ulimit memlock=-1 -p 127.0.0.1:8081:8080 -v lamina-data:/data \
#              -e MODEL=iq3_s -e API_KEY=<a long random key> lamina
#
# The settings (environment variables): MODEL (iq3_s, ud-q4_k_xl or qwen36-ud-q4_k_xl), API_KEY, and optionally
# VISION (on or off), CONTEXT (tokens), MAIN_GPU (a name, UUID or nvidia-smi index). The image runs on any x86-64 CPU
# with AVX2 (--portable). Pinned memory needs --ulimit memlock=-1.

ARG CUDA_VERSION=13.0.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates libssl-dev python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/lamina
COPY . .
ARG CUDA_ARCHITECTURES="86-real;89-real;120-real;120-virtual"
# compile jobs (default: half the CPU's threads); nvcc takes 1-2 GB of RAM a job
ARG JOBS=
RUN python3 -m venv .venv && \
    CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" .venv/bin/python setup.py --cuda-archs "$CUDA_ARCHITECTURES" --portable

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libssl3 python3 python3-venv tini \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/lamina
COPY . .
RUN python3 -m venv .venv && .venv/bin/pip install --no-cache-dir -r requirements.txt
COPY --from=build /opt/lamina/build/lamina /opt/lamina/build/lamina-server build/
COPY --from=build /opt/lamina/build-vision/bin/lamina-vision build-vision/bin/
COPY --from=build /opt/lamina/third_party/llama.cpp/gguf-py third_party/llama.cpp/gguf-py/
COPY --from=build /opt/lamina/third_party/llama.cpp/.lamina-commit third_party/llama.cpp/
VOLUME ["/data"]
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/lamina/docker/entrypoint.sh"]
