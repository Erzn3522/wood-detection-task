# Wood Knot Detection

Detects knots in lumber board images using a YOLO26m end-to-end ONNX model. Frames from the same board are stitched horizontally and processed as a unit; detections are output as axis-aligned bounding boxes in board coordinates.

---

## Contents

- [Prerequisites](#prerequisites)
- [1. Clone the repository](#1-clone-the-repository)
- [2. Pull the Docker image](#2-pull-the-docker-image)
- [3. Run the commands (Docker)](#3-run-the-commands-docker)
  - [3a. Test](#3a-test--inference--evaluation-against-ground-truth)
  - [3b. Predict](#3b-predict--inference-only-no-ground-truth-needed)
  - [3c. Visualize](#3c-visualize--annotated-per-frame-images)
- [GPU inference](#gpu-inference)
- [4. Build from source](#4-build-from-source) *(alternative to Docker)*
  - [Linux (Debian / Ubuntu / Codespaces)](#linux-debian--ubuntu--codespaces)
  - [Google Colab](#google-colab)
  - [macOS (Apple Silicon)](#macos-apple-silicon)
- [CLI reference](#cli-reference)
- [Output schemas](#output-schemas)
- [Test set results](#test-set-results)
- [Notes](#notes)

---

## Prerequisites

### Docker

Install Docker for your platform: [https://docs.docker.com/get-docker](https://docs.docker.com/get-docker/)

### Git LFS

Required to download the ONNX model and test images from the repository.

**macOS**
```bash
brew install git-lfs
```

**Linux (Debian / Ubuntu)**
```bash
apt-get update && apt-get install -y git curl
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
```

**Windows**

Download and run the installer from [https://git-lfs.com](https://git-lfs.com), or:
```powershell
winget install GitHub.GitLFS
```

---

## 1. Clone the repository

Install Git LFS first (see above), then clone:

```bash
git lfs install
git clone https://github.com/erzn3522/wood-detection-task.git
cd wood-detection-task
```

`git lfs install` must run before `git clone` — LFS files (ONNX model and test images, ~170 MB) are downloaded automatically during the clone.

---

## 2. Pull the Docker image

> If Docker is not available, skip to [Build from source](#4-build-from-source).

The model is baked into the image — no separate model download needed at runtime.

**Linux / Windows (x86-64)**
```bash
docker pull ghcr.io/erzn3522/wood-detection-task:latest
```

**macOS (Apple Silicon)**
```bash
docker pull --platform linux/amd64 ghcr.io/erzn3522/wood-detection-task:latest
```

> The image is built for `linux/amd64`. On Apple Silicon, Docker Desktop runs it via Rosetta emulation — add `--platform linux/amd64` to all `docker run` commands as well.

---

## 3. Run the commands (Docker)

> These commands use the Docker image. If you built from source, the run commands are included in the [Build from source](#4-build-from-source) section below.

### 3a. Test — inference + evaluation against ground truth

Runs the full pipeline on the test set and produces a metrics report.

**Linux / Windows**
```bash
docker run --rm \
  -v $PWD/wood-dataset/test/images:/data/images \
  -v $PWD/wood-dataset/test/labels:/data/labels \
  -v $PWD/out:/out \
  ghcr.io/erzn3522/wood-detection-task:latest \
  test \
    --frames-dir  /data/images \
    --labels-dir  /data/labels \
    --model       /model/yolo26m.onnx \
    --out         /out
```

**macOS (Apple Silicon)**
```bash
docker run --rm --platform linux/amd64 \
  -v $PWD/wood-dataset/test/images:/data/images \
  -v $PWD/wood-dataset/test/labels:/data/labels \
  -v $PWD/out:/out \
  ghcr.io/erzn3522/wood-detection-task:latest \
  test \
    --frames-dir  /data/images \
    --labels-dir  /data/labels \
    --model       /model/yolo26m.onnx \
    --out         /out
```

**Outputs written to `out/`:**

```
out/
├── predictions/
│   ├── {board}.json          # per-board knot detections
│   └── images/{board}.png    # tiled board image with green bboxes
├── metrics.json              # aggregate precision / recall / F1 / mAP50
└── REPORT.md                 # human-readable report with per-board table
```

> After running `test`, run the `visualize` subcommand with `--mode per_frame` to inspect detections frame by frame with TP/FP/FN coloring. See [3c. Visualize](#3c-visualize--annotated-per-frame-images).

---

### 3b. Predict — inference only, no ground truth needed

Use this when you have your own board images without labels.

**Linux / Windows**
```bash
docker run --rm \
  -v $PWD/wood-dataset/test/images:/data/images \
  -v $PWD/out:/out \
  ghcr.io/erzn3522/wood-detection-task:latest \
  predict \
    --frames-dir /data/images \
    --model      /model/yolo26m.onnx \
    --out        /out
```

**macOS (Apple Silicon)**
```bash
docker run --rm --platform linux/amd64 \
  -v $PWD/wood-dataset/test/images:/data/images \
  -v $PWD/out:/out \
  ghcr.io/erzn3522/wood-detection-task:latest \
  predict \
    --frames-dir /data/images \
    --model      /model/yolo26m.onnx \
    --out        /out
```

**Outputs:** same as test but without `metrics.json` / `REPORT.md`.

---

### 3c. Visualize — annotated per-frame images

Run this after `test` or `predict` to inspect detections frame by frame.

**Linux / Windows**
```bash
docker run --rm \
  -v $PWD/wood-dataset/test/images:/data/images \
  -v $PWD/wood-dataset/test/labels:/data/labels \
  -v $PWD/out/predictions:/predictions \
  -v $PWD/out/vis:/vis \
  ghcr.io/erzn3522/wood-detection-task:latest \
  visualize \
    --predictions-dir /predictions \
    --frames-dir      /data/images \
    --labels-dir      /data/labels \
    --out             /vis \
    --mode            per_frame
```

**macOS (Apple Silicon)**
```bash
docker run --rm --platform linux/amd64 \
  -v $PWD/wood-dataset/test/images:/data/images \
  -v $PWD/wood-dataset/test/labels:/data/labels \
  -v $PWD/out/predictions:/predictions \
  -v $PWD/out/vis:/vis \
  ghcr.io/erzn3522/wood-detection-task:latest \
  visualize \
    --predictions-dir /predictions \
    --frames-dir      /data/images \
    --labels-dir      /data/labels \
    --out             /vis \
    --mode            per_frame
```

**Output:** one PNG per input frame in `out/vis/`. Each image has a 40 px header strip with the frame name, frame-local TP/FP/FN counts and a color legend.

**Color scheme:**

| Color | Meaning |
|---|---|
| Green | TP — correct detection |
| Yellow | FP — false positive |
| Red | FN — missed ground truth |
| Cyan dashed | Matched GT box (reference) |
| Dashed edge + arrow | Bbox crossing a frame boundary |

Each bbox carries a confidence score and a stable knot ID (`#K{n}`) so the same detection can be tracked across adjacent frames. Knots that span frame boundaries are stored as a single entry in the board JSON and marked with `→` / `←` arrows on the relevant edges.

Omit `--labels-dir` to skip GT comparison — all predictions are drawn in green.

For a grid view instead of per-frame:

**Linux / Windows**
```bash
docker run --rm \
  -v $PWD/wood-dataset/test/images:/data/images \
  -v $PWD/wood-dataset/test/labels:/data/labels \
  -v $PWD/out/predictions:/predictions \
  -v $PWD/out/vis:/vis \
  ghcr.io/erzn3522/wood-detection-task:latest \
  visualize \
    --predictions-dir /predictions \
    --frames-dir      /data/images \
    --labels-dir      /data/labels \
    --out             /vis \
    --mode            tiled \
    --cols            3
```

**macOS (Apple Silicon)**
```bash
docker run --rm --platform linux/amd64 \
  -v $PWD/wood-dataset/test/images:/data/images \
  -v $PWD/wood-dataset/test/labels:/data/labels \
  -v $PWD/out/predictions:/predictions \
  -v $PWD/out/vis:/vis \
  ghcr.io/erzn3522/wood-detection-task:latest \
  visualize \
    --predictions-dir /predictions \
    --frames-dir      /data/images \
    --labels-dir      /data/labels \
    --out             /vis \
    --mode            tiled \
    --cols            3
```

---

## GPU inference

Add `--gpus all` to the Docker run command and `--device gpu` to the subcommand:

```bash
docker run --rm --gpus all \
  -v $PWD/wood-dataset/test/images:/data/images \
  -v $PWD/out:/out \
  ghcr.io/erzn3522/wood-detection-task:latest \
  test \
    --frames-dir /data/images \
    --labels-dir /data/labels \
    --model      /model/yolo26m.onnx \
    --out        /out \
    --device     gpu
```

If no CUDA-capable GPU is available the runtime falls back to CPU automatically with a warning on stderr.

---

## 4. Build from source

Use this if Docker is not available.

### Linux (Debian / Ubuntu / Codespaces)

> Run these commands from inside the cloned repository (section 1 must be complete).
> GitHub Codespaces users: the repo is already open. If the model file is missing run `git lfs pull` once, then continue from the `apt-get` step below.

```bash
sudo apt-get install -y build-essential cmake libopencv-dev

curl -fL https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz | tar -xz

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-x64-gpu-1.20.1

cmake --build build -j$(nproc)

export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-gpu-1.20.1/lib:$LD_LIBRARY_PATH

# predict — inference only
./build/wood_knot_detector predict \
  --frames-dir wood-dataset/test/images \
  --model      model/yolo26m.onnx \
  --out        out

# test — inference + evaluation against ground truth
./build/wood_knot_detector test \
  --frames-dir wood-dataset/test/images \
  --labels-dir wood-dataset/test/labels \
  --model      model/yolo26m.onnx \
  --out        out

# visualize — annotated per-frame images (run after predict or test)
./build/wood_knot_detector visualize \
  --predictions-dir out/predictions \
  --frames-dir      wood-dataset/test/images \
  --labels-dir      wood-dataset/test/labels \
  --out             out/vis \
  --mode            per_frame   # alternative: --mode tiled --cols 3  (one grid image per board)
```

---

### Google Colab

Run each step in a separate cell. Paths are set for the `/content/` working directory.

```python
!apt-get install -y build-essential cmake libopencv-dev
```

```python
!curl -fL https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz | tar -xz -C /content
```

```python
!git lfs install
!git clone https://github.com/erzn3522/wood-detection-task.git /content/wood-detection-task
%cd /content/wood-detection-task
!git lfs pull
```

```python
!cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/content/onnxruntime-linux-x64-gpu-1.20.1

!cmake --build build -j$(nproc)
```

```python
import os
os.environ["LD_LIBRARY_PATH"] = "/content/onnxruntime-linux-x64-gpu-1.20.1/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

# predict — inference only
!./build/wood_knot_detector predict \
  --frames-dir wood-dataset/test/images \
  --model      model/yolo26m.onnx \
  --out        out \
  --device     gpu
```

```python
# test — inference + evaluation against ground truth
!./build/wood_knot_detector test \
  --frames-dir wood-dataset/test/images \
  --labels-dir wood-dataset/test/labels \
  --model      model/yolo26m.onnx \
  --out        out \
  --device     gpu
```

```python
# visualize — annotated per-frame images (run after predict or test)
!./build/wood_knot_detector visualize \
  --predictions-dir out/predictions \
  --frames-dir      wood-dataset/test/images \
  --labels-dir      wood-dataset/test/labels \
  --out             out/vis \
  --mode            per_frame   # alternative: --mode tiled --cols 3  (one grid image per board)
```

To display the output images inline in Colab:

```python
import glob
from IPython.display import Image, display

for path in sorted(glob.glob("out/vis/*.png"))[:5]:
    display(Image(path))
```

---

### macOS (Apple Silicon)

```bash
brew install cmake opencv

curl -fL https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-osx-arm64-1.20.1.tgz | tar -xz

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=$PWD/onnxruntime-osx-arm64-1.20.1

cmake --build build -j$(nproc)

export DYLD_LIBRARY_PATH=$PWD/onnxruntime-osx-arm64-1.20.1/lib

# test — inference + evaluation against ground truth
./build/wood_knot_detector test \
  --frames-dir wood-dataset/test/images \
  --labels-dir wood-dataset/test/labels \
  --model      model/yolo26m.onnx \
  --out        out

# visualize — annotated per-frame images (run after test)
./build/wood_knot_detector visualize \
  --predictions-dir out/predictions \
  --frames-dir      wood-dataset/test/images \
  --labels-dir      wood-dataset/test/labels \
  --out             out/vis \
  --mode            per_frame
```

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--frames-dir` | required | Directory of `{board}_{frame}.png` files |
| `--labels-dir` | required (test) | Directory of `{board}_{frame}.txt` YOLO labels |
| `--model` | required | Path to ONNX model |
| `--out` | required | Output directory |
| `--device` | `gpu` | `gpu` (falls back to `cpu` if no CUDA device is available) or `cpu` |
| `--conf-threshold` | `0.25` | Minimum confidence to keep a detection |
| `--iou-threshold` | `0.5` | IoU threshold for TP/FP matching (test only) |
| `--mode` | `per_frame` | Visualization mode: `per_frame` or `tiled` |
| `--cols` | `3` | Columns in tiled mode |

---

## Output schemas

**`{board}.json`**

```json
{
  "board_index": 1041,
  "frames": ["1041_0.png", "1041_1.png"],
  "frame_widths": [640, 640],
  "board_width_px": 17920,
  "board_height_px": 157,
  "knots": [
    {
      "polygon": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
      "confidence": 0.91,
      "spans_frames": [0]
    }
  ]
}
```

**`metrics.json`**

```json
{
  "iou_threshold": 0.5,
  "num_boards": 27,
  "num_frames": 852,
  "totals": {"tp": 1455, "fp": 288, "fn": 384},
  "precision": 0.835,
  "recall": 0.791,
  "f1": 0.812,
  "map50": 0.751
}
```

---

## Test set results

Evaluated on the sealed test split (27 boards, 852 frames). Matching criterion: `max(IoU, IoMin) ≥ 0.5` at confidence threshold 0.25. IoMin = intersection / min(area) handles the case where a small GT box is fully contained inside a larger prediction.

| Metric | Value |
|---|---|
| Precision | 0.835 |
| Recall | 0.791 |
| F1 | 0.812 |
| mAP50 | 0.751 |

<details>
<summary>Per-board results (worst → best F1)</summary>

| Board | Frames | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1091 | 32 | 42 | 13 | 59 | 0.764 | 0.416 | 0.538 |
| 1092 | 37 | 63 | 39 | 44 | 0.618 | 0.589 | 0.603 |
| 1205 | 33 | 59 | 47 | 21 | 0.557 | 0.738 | 0.634 |
| 1187 | 30 | 40 | 18 | 21 | 0.690 | 0.656 | 0.672 |
| 1136 | 33 | 61 | 24 | 14 | 0.718 | 0.813 | 0.763 |
| 1152 | 35 | 54 | 27 | 6 | 0.667 | 0.900 | 0.766 |
| 1221 | 32 | 43 | 9 | 16 | 0.827 | 0.729 | 0.775 |
| 1075 | 35 | 62 | 12 | 19 | 0.838 | 0.765 | 0.800 |
| 1218 | 29 | 26 | 4 | 9 | 0.867 | 0.743 | 0.800 |
| 1173 | 36 | 57 | 16 | 7 | 0.781 | 0.891 | 0.832 |
| 1188 | 32 | 45 | 8 | 10 | 0.849 | 0.818 | 0.833 |
| 1165 | 35 | 74 | 18 | 11 | 0.804 | 0.871 | 0.836 |
| 1190 | 34 | 40 | 2 | 13 | 0.952 | 0.755 | 0.842 |
| 1141 | 35 | 93 | 11 | 19 | 0.894 | 0.830 | 0.861 |
| 1067 | 31 | 45 | 6 | 8 | 0.882 | 0.849 | 0.865 |
| 1070 | 35 | 75 | 5 | 18 | 0.938 | 0.806 | 0.867 |
| 1151 | 28 | 33 | 3 | 7 | 0.917 | 0.825 | 0.868 |
| 1232 | 32 | 78 | 6 | 16 | 0.929 | 0.830 | 0.876 |
| 1189 | 36 | 82 | 0 | 23 | 1.000 | 0.781 | 0.877 |
| 1113 | 30 | 41 | 3 | 8 | 0.932 | 0.837 | 0.882 |
| 1169 | 37 | 78 | 5 | 13 | 0.940 | 0.857 | 0.897 |
| 1052 | 31 | 61 | 5 | 9 | 0.924 | 0.871 | 0.897 |
| 1049 | 36 | 77 | 5 | 7 | 0.939 | 0.917 | 0.928 |
| 1224 | 29 | 36 | 1 | 4 | 0.973 | 0.900 | 0.935 |
| 1177 | 30 | 46 | 0 | 2 | 1.000 | 0.958 | 0.979 |
| 1041 | 28 | 42 | 1 | 0 | 0.977 | 1.000 | 0.988 |
| 126 | 1 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 |

</details>

The low recall on board 1091 (F1=0.538) is driven by many small knots near frame edges that fall below the confidence threshold at this scale.

---

## Notes

**Training approach**

I came home after the first interview, split the provided dataset immediately, and started training before receiving confirmation on whether using it was allowed. The only uncertainty I had about the task instructions was whether I was permitted to train on the provided data.

I decided to start anyway. If it had turned out to be off-limits, my fallback plan was to use the model trained on the provided data only as a teacher model, collect around 40k wood images with different datasets from the internet, run inference on them, select the top 5k highest-confidence crops, and train a new model only on those selected images.

In that case, the model trained on the provided data would not have been included in the final deliverable. Only the distilled student model would have been used. This is why the model file was named with "teacher" in the early commits.

**Model selection**

I chose YOLO26m for two reasons: it is the latest release in the YOLO26 family, and it exports to ONNX with end-to-end NMS embedded in the graph. The embedded NMS means the model outputs post-NMS detections directly, with no custom NMS needed on the C++ host side, which greatly simplified the inference code.

I started with the medium (`m`) variant. Once its metrics looked good enough at the end of the training, I didn't train other variants, as mentioned in the first interview, maximising the detection score was not the primary goal of the task.

**Development process**

I used Claude as a coding assistant for the C++ implementation. I designed the full architecture, defined the data flow between components, and drove every implementation decision. Claude translated those decisions into C++ code. I chose this approach because while I have systems programming experience, I do not write C++ regularly. I understand what the code needs to do and why, but writing C++ would have taken significantly longer for me. I reviewed the code, tested the pipeline end-to-end, diagnosed and fixed issues (such as the IoMin matching bug and the FetchContent Docker failure), and iterated until the results were correct.

**GPU inference**

The default device is `gpu` with automatic CPU fallback. I was not able to test GPU inference directly — my development machine is a Mac (Apple Silicon), which does not support CUDA. The environments I tested on were macOS local, GitHub Codespaces (bare Linux server, CPU only), and Google Colab with T4 GPU

**Repository visibility**

I initially used a private repository, but due to GitHub free-tier limitations for building Docker images with GitHub Actions in a private repo, I moved the project to a public repository.
