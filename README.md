# Wood Knot Detection

Detects knots in lumber board images using a YOLO26m end-to-end ONNX model. Frames from the same board are stitched horizontally and processed as a unit; detections are output as axis-aligned bounding boxes in board coordinates.

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
git lfs pull
```

`git lfs pull` downloads the ONNX model (`model/yolo26m.onnx`) and the sealed test split (`wood-dataset/test/`). This is ~170 MB.

---

## 2. Pull the Docker image

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

## 3. Run the commands

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

---

### 3b. Predict — inference only, no ground truth needed

Use this when you have your own board images without labels.

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

**Outputs:** same as test but without `metrics.json` / `REPORT.md`.

---

### 3c. Visualize — annotated per-frame images

Run this after `test` or `predict` to inspect detections frame by frame.

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

## Build from source

Use this if Docker is not available.

### Linux (Debian / Ubuntu / bare server)

```bash
apt-get install -y build-essential cmake libopencv-dev

curl -fL https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz | tar -xz

git lfs install
git clone https://github.com/erzn3522/wood-detection-task.git
cd wood-detection-task
git lfs pull

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=$PWD/../onnxruntime-linux-x64-gpu-1.20.1

cmake --build build -j$(nproc)

export LD_LIBRARY_PATH=$PWD/../onnxruntime-linux-x64-gpu-1.20.1/lib:$LD_LIBRARY_PATH

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
  --mode            per_frame
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
  --mode            per_frame
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
./build/wood_knot_detector --help
```

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--frames-dir` | required | Directory of `{board}_{frame}.png` files |
| `--labels-dir` | required (test) | Directory of `{board}_{frame}.txt` YOLO labels |
| `--model` | required | Path to ONNX model |
| `--out` | required | Output directory |
| `--device` | `cpu` | `cpu` or `gpu` |
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
  "totals": {"tp": 1311, "fp": 102, "fn": 528},
  "precision": 0.928,
  "recall": 0.713,
  "f1": 0.806,
  "map50": 0.695
}
```

---

## Test set results

Evaluated on the sealed test split (27 boards, 852 frames). Matching criterion: `max(IoU, IoMin) ≥ 0.5` at confidence threshold 0.25. IoMin = intersection / min(area) handles the case where a small GT box is fully contained inside a larger prediction.

| Metric | Value |
|---|---|
| Precision | 0.928 |
| Recall | 0.713 |
| F1 | 0.806 |
| mAP50 | 0.695 |

<details>
<summary>Per-board results (worst → best F1)</summary>

| Board | Frames | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 1091 | 32 | 39 | 1 | 62 | 0.975 | 0.386 | 0.553 |
| 1221 | 32 | 34 | 6 | 25 | 0.850 | 0.576 | 0.687 |
| 1113 | 30 | 28 | 3 | 21 | 0.903 | 0.571 | 0.700 |
| 1205 | 33 | 58 | 23 | 22 | 0.716 | 0.725 | 0.720 |
| 1092 | 37 | 66 | 7 | 41 | 0.904 | 0.617 | 0.733 |
| 1187 | 30 | 37 | 2 | 24 | 0.949 | 0.607 | 0.740 |
| 1152 | 35 | 47 | 16 | 13 | 0.746 | 0.783 | 0.764 |
| 1232 | 32 | 59 | 1 | 35 | 0.983 | 0.628 | 0.766 |
| 1189 | 36 | 67 | 0 | 38 | 1.000 | 0.638 | 0.779 |
| 1075 | 35 | 55 | 3 | 26 | 0.948 | 0.679 | 0.791 |
| 1190 | 34 | 35 | 0 | 18 | 1.000 | 0.660 | 0.795 |
| 1070 | 35 | 65 | 4 | 28 | 0.942 | 0.699 | 0.802 |
| 1141 | 35 | 81 | 6 | 31 | 0.931 | 0.723 | 0.814 |
| 1218 | 29 | 26 | 2 | 9 | 0.929 | 0.743 | 0.825 |
| 1165 | 35 | 63 | 4 | 22 | 0.940 | 0.741 | 0.829 |
| 1052 | 31 | 51 | 2 | 19 | 0.962 | 0.729 | 0.829 |
| 1173 | 36 | 53 | 8 | 11 | 0.869 | 0.828 | 0.848 |
| 1151 | 28 | 32 | 2 | 8 | 0.941 | 0.800 | 0.865 |
| 1067 | 31 | 43 | 3 | 10 | 0.935 | 0.811 | 0.869 |
| 1136 | 33 | 61 | 4 | 14 | 0.938 | 0.813 | 0.871 |
| 1224 | 29 | 31 | 0 | 9 | 1.000 | 0.775 | 0.873 |
| 1169 | 37 | 75 | 2 | 16 | 0.974 | 0.824 | 0.893 |
| 1188 | 32 | 47 | 1 | 8 | 0.979 | 0.855 | 0.913 |
| 1177 | 30 | 42 | 0 | 6 | 1.000 | 0.875 | 0.933 |
| 1041 | 28 | 38 | 1 | 4 | 0.974 | 0.905 | 0.938 |
| 1049 | 36 | 76 | 1 | 8 | 0.987 | 0.905 | 0.944 |
| 126 | 1 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 |

</details>

The low recall on board 1091 (F1=0.553) is driven by many small knots near frame edges that fall below the confidence threshold at this scale.

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

I completed the task end-to-end with Claude as my primary coding assistant. I planned the solution, understood the implementation path clearly, and used Claude to speed up the C++ coding part. I then reviewed, tested, debugged, and integrated the generated code myself.

**Repository visibility**

I initially used a private repository, but due to GitHub free-tier limitations for building Docker images with GitHub Actions in a private repo, I moved the project to a public repository.
