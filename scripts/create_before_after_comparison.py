#!/usr/bin/env python3
"""PhytoTrace AI — Generate Visual Before/After Evidence Grid."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from phytotrace.utils.paths import resolve_path


def create_comparison():
    out_dir = resolve_path("reports/debug_segmentation")

    img_orig = cv2.imread(str(out_dir / "01_original_failing_image.jpg"))
    img_unet_bad = cv2.imread(str(out_dir / "04_unet_overlay_on_original.jpg"))
    img_yolo_bad = cv2.imread(str(out_dir / "05_raw_yolo_detections.jpg"))
    img_leaf_good = cv2.imread(str(out_dir / "10_heuristic_overlay.jpg"))
    img_after = cv2.imread(str(out_dir / "13_verified_auto_annotated.jpg"))

    # Create figure with 5 panels
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    fig.patch.set_facecolor("#0a1814")

    panels = [
        ("1. Original Smartphone Input\n(3024x4032 Phone Photo)", cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)),
        ("2. BEFORE: Uncalibrated U-Net\n(Background Segmented as Leaf)", cv2.cvtColor(img_unet_bad, cv2.COLOR_BGR2RGB)),
        ("3. BEFORE: YOLO Low-Conf Glitch\n(1.86M px Box @ 16.4% Conf)", cv2.cvtColor(img_yolo_bad, cv2.COLOR_BGR2RGB)),
        ("4. AFTER: Quality-Aware Canopy\n(Foliage Accurately Isolated)", cv2.cvtColor(img_leaf_good, cv2.COLOR_BGR2RGB)),
        ("5. AFTER: Validated Pipeline\n(0% Damage / Clean Decision)", cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)),
    ]

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, color="#6ae6a0", fontsize=11, fontweight="bold", pad=10)
        ax.axis("off")

    plt.tight_layout()
    comp_path = out_dir / "before_vs_after_comparison.png"
    plt.savefig(str(comp_path), dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved before/after comparative figure to {comp_path}")


if __name__ == "__main__":
    create_comparison()

