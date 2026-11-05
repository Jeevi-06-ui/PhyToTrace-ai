"""Visualization utilities for leaf canopy masks and localized leaf-miner damage overlays."""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phytotrace.utils.paths import resolve_path


def render_segmentation_overlay(
    image_bgr: np.ndarray,
    leaf_mask: Optional[np.ndarray] = None,
    mine_mask: Optional[np.ndarray] = None,
    alpha_leaf: float = 0.20,
    alpha_mine: float = 0.45,
) -> np.ndarray:
    """Render semi-transparent colored overlays and contours on an RGB/BGR image.
    
    - Leaf Canopy: Soft green tint (#2e7d32) with crisp contour.
    - Mine Damage: Vivid crimson/amber overlay (#d32f2f) with yellow border.
    
    Returns:
        BGR image array containing the rendered overlay.
    """
    overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    # 1. Render leaf canopy
    if leaf_mask is not None and np.sum(leaf_mask) > 0:
        lm_bin = (leaf_mask > 0).astype(np.uint8)
        # Green tint: B=46, G=125, R=50
        leaf_color = np.zeros_like(image_bgr)
        leaf_color[:] = (46, 125, 50)
        cv2.copyTo(leaf_color, lm_bin, overlay)
        cv2.addWeighted(overlay, alpha_leaf, image_bgr, 1.0 - alpha_leaf, 0, overlay)

        # Draw leaf contour outline
        contours, _ = cv2.findContours(lm_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (46, 204, 113), 2)  # Bright green outline

    # 2. Render mine damage
    if mine_mask is not None and np.sum(mine_mask) > 0:
        mm_bin = (mine_mask > 0).astype(np.uint8)
        mine_overlay = overlay.copy()
        # Red/coral tint: B=30, G=30, R=220
        mine_color = np.zeros_like(image_bgr)
        mine_color[:] = (30, 30, 220)
        cv2.copyTo(mine_color, mm_bin, mine_overlay)
        cv2.addWeighted(mine_overlay, alpha_mine, overlay, 1.0 - alpha_mine, 0, overlay)

        # Draw mine contour outline
        m_contours, _ = cv2.findContours(mm_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, m_contours, -1, (0, 215, 255), 2)  # Bright amber/gold outline

    return overlay


def save_segmentation_comparison_figure(
    image_bgr: np.ndarray,
    leaf_mask: Optional[np.ndarray],
    mine_mask: Optional[np.ndarray],
    save_path: Path,
    image_title: str = "Leaf Damage Segmentation",
) -> None:
    """Generate and save side-by-side comparison: Original | Leaf Mask | Mine Mask | Combined Overlay."""
    save_path = resolve_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay_bgr = render_segmentation_overlay(image_bgr, leaf_mask, mine_mask)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    h, w = image_bgr.shape[:2]
    leaf_m = leaf_mask if leaf_mask is not None else np.zeros((h, w), dtype=np.uint8)
    mine_m = mine_mask if mine_mask is not None else np.zeros((h, w), dtype=np.uint8)

    leaf_pixels = int(np.sum(leaf_m > 0))
    mine_pixels = int(np.sum(mine_m > 0))
    severity_pct = (mine_pixels / leaf_pixels * 100.0) if leaf_pixels > 0 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Photograph", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(leaf_m, cmap="Greens")
    axes[1].set_title(f"Leaf Mask ({leaf_pixels:,} px)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title(
        f"Damage Overlay (Mines: {mine_pixels:,} px | Severity: {severity_pct:.2f}%)",
        fontsize=11, fontweight="bold",
    )
    axes[2].axis("off")

    plt.suptitle(image_title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

