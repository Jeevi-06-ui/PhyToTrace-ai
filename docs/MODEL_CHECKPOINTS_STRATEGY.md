# PhytoTrace AI — Model Checkpoints & Deployment Artifact Strategy

This document details the preservation, versioning, and distribution strategy for PhytoTrace's trained deep learning checkpoints.

---

## 1. Trained Checkpoint Inventory

PhytoTrace operates with three genuine, validated PyTorch checkpoints:

| Model Identity | Architecture | Checkpoint Path | Checkpoint Size | Validation Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Leaf Boundary Segmenter** | Custom Lightweight U-Net | `models/checkpoints/leaf_segmenter/leaf_unet_best.pt` | 1.9 MB | **Val Dice: 0.7923** |
| **Mine Lesion Localizer** | YOLOv8n-Seg Instance Segmenter | `models/checkpoints/mine_segmenter/best.pt` | 6.7 MB | **224 Polygons; mAP50: 0.53** |
| **Supplementary Screener** | MobileNetV3-Small | `models/checkpoints/baseline/baseline_mobilenetv3_best.pt` | 5.9 MB | **Macro F1: 0.5667** |

---

## 2. Git & Source Control Policy

In accordance with best software engineering practices:
1. Binary model weights (`*.pt`, `*.pth`, `*.onnx`) are **excluded from GitHub tracking** via `.gitignore` to prevent repository bloat and bandwidth throttling.
2. The folder structure is preserved via `.gitkeep` files in `models/checkpoints/leaf_segmenter/`, `models/checkpoints/mine_segmenter/`, and `models/checkpoints/baseline/`.

---

## 3. Production Deployment & Cloud Storage Options

For deploying to production environments (such as Google Cloud Run, AWS ECS, or Kubernetes):

### Option A: Direct Docker Container Packaging (Recommended for Cloud Run)
Because the three model weights total only **14.5 MB**, the root `Dockerfile` packages the `models/` directory directly into the application container layer:
```dockerfile
COPY models/ ./models/
```
This guarantees instant container startup with zero runtime dependency on external network downloads during auto-scaling.

### Option B: Cloudinary / Cloudflare R2 / S3 Checkpoint Sync
For CI/CD pipelines that build containers without local weight access, checkpoints can be fetched via signed download during image build:
```bash
python scripts/download_checkpoints.py --provider cloudinary
```

### Option C: GitHub Releases
For open-source distribution:
1. Create a GitHub Release tag: `v1.0.0-models`.
2. Attach `leaf_unet_best.pt`, `best.pt`, and `baseline_mobilenetv3_best.pt` as release assets.
3. Download using `curl -L -o ... https://github.com/<owner>/<repo>/releases/download/...`.

