# ---- builder ----
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

ARG ORT_VERSION=1.20.1
ARG ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates curl \
      libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN curl -fL "${ORT_URL}" -o ort.tgz \
    && tar -xzf ort.tgz \
    && mv onnxruntime-linux-x64-gpu-${ORT_VERSION} onnxruntime \
    && rm ort.tgz


# Pre-fetch CMake FetchContent deps as separate cached layers
RUN curl -fL "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz" \
      -o /tmp/json.tar.xz \
    && mkdir -p /opt/json \
    && tar -xJf /tmp/json.tar.xz -C /opt/json --strip-components=1 \
    && rm /tmp/json.tar.xz

RUN git clone --depth=1 --branch v3.0 \
      https://github.com/p-ranav/argparse.git /opt/argparse

WORKDIR /src
COPY CMakeLists.txt /src/
COPY include /src/include
COPY src /src/src
RUN cmake -S /src -B /build \
      -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT=/opt/onnxruntime \
      -DFETCHCONTENT_SOURCE_DIR_JSON=/opt/json \
      -DFETCHCONTENT_SOURCE_DIR_ARGPARSE=/opt/argparse \
    && cmake --build /build -j"$(nproc)"

# ---- runtime ----
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libopencv-core4.5d \
      libopencv-imgproc4.5d \
      libopencv-imgcodecs4.5d \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/onnxruntime/lib/libonnxruntime*.so* /usr/local/lib/
COPY --from=builder /build/wood_knot_detector /usr/local/bin/wood_knot_detector
COPY model/yolo26m.onnx /model/yolo26m.onnx

RUN ldconfig

ENTRYPOINT ["wood_knot_detector"]
CMD ["--help"]
