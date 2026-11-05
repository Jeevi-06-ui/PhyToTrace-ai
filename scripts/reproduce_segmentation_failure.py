#!/usr/bin/env python3
"""PhytoTrace AI — Reproduce and Diagnostic Log of Failing Smartphone Image.

Executes real-model inference on the failing inspection image, logging every
intermediate stage and saving visual artifacts to reports/debug_segmentation/.
"""

import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from phytotrace.inference.pipeline import InferencePipeline
from phytotrace.models.leaf_segmenter import LeafSegmenter
from phytotrace.models.mine_segmenter import MineSegmenter
from phytotrace.utils.paths import resolve_path


def reproduce():
    img_path = resolve_path("data/inspections/raw/INSP_20261105_024600_44601_PLANT_TOMATO_A07_orig.jpg")
    out_dir = resolve_path("reports/debug_segmentation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHYTOTRACE — CRITICAL SEGMENTATION FAILURE REPRODUCTION")
    print("=" * 80)
    print(f"Target Image: {img_path}")

    # 1. Load Image and Compute Physical/Color Properties
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w, c = img_bgr.shape
    aspect_ratio = round(w / h, 4)
    bgr_mean = img_bgr.mean(axis=(0, 1)).tolist()
    bgr_std = img_bgr.std(axis=(0, 1)).tolist()
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_mean = hsv.mean(axis=(0, 1)).tolist()

    print(f"\n[Image Properties]")
    print(f"  Dimensions    : {w}x{h} (Aspect Ratio: {aspect_ratio})")
    print(f"  Total Pixels  : {w * h:,}")
    print(f"  Mean BGR      : {[round(x, 2) for x in bgr_mean]}")
    print(f"  Std BGR       : {[round(x, 2) for x in bgr_std]}")
    print(f"  Mean HSV      : {[round(x, 2) for x in hsv_mean]}")

    # Save 01_original_failing_image.jpg (resized copy for fast viewing + full size)
    cv2.imwrite(str(out_dir / "01_original_failing_image.jpg"), img_bgr)

    # 2. U-Net Leaf Segmentation Internal Inspection
    leaf_model_path = resolve_path("models/checkpoints/leaf_segmenter/leaf_unet_best.pt")
    leaf_seg = LeafSegmenter(str(leaf_model_path), device="cpu")
    
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(rgb, (256, 256))
    tensor = torch.from_numpy(resized_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        logits = leaf_seg.model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        raw_logits_np = logits.squeeze().cpu().numpy()

    print(f"\n[U-Net Intermediate Analysis]")
    print(f"  Logits Range       : [{raw_logits_np.min():.4f}, {raw_logits_np.max():.4f}]")
    print(f"  Logits Mean / Std  : {raw_logits_np.mean():.4f} / {raw_logits_np.std():.4f}")
    print(f"  Probs Range        : [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Probs Mean / Median: {probs.mean():.4f} / {np.median(probs):.4f}")
    print(f"  Probs Std Dev      : {probs.std():.4f}")
    print(f"  Probs > 0.50 Count : {np.count_nonzero(probs > 0.50)} / {256 * 256} ({np.count_nonzero(probs > 0.50)/(256*256)*100:.2f}%)")

    # Generate Heatmap of U-Net Probabilities
    # Normalize probabilities for visual colormap
    prob_norm = ((probs - probs.min()) / (probs.max() - probs.min() + 1e-8) * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
    heatmap_full = cv2.resize(heatmap_colored, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(out_dir / "02_raw_unet_probability_map.png"), heatmap_full)

    # Thresholded mask (at 0.50)
    binary_256 = (probs > 0.50).astype(np.uint8) * 255
    unet_mask_full = cv2.resize(binary_256, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / "03_thresholded_unet_mask.png"), unet_mask_full)

    # Overlay U-Net green on original
    unet_overlay = img_bgr.copy()
    green_layer = np.zeros_like(img_bgr)
    green_layer[:] = [0, 255, 0]
    mask_bool = unet_mask_full > 0
    unet_overlay[mask_bool] = cv2.addWeighted(img_bgr[mask_bool], 0.6, green_layer[mask_bool], 0.4, 0)
    cv2.imwrite(str(out_dir / "04_unet_overlay_on_original.jpg"), unet_overlay)

    # 3. YOLO Mine Damage Segmentation Inspection (Raw & Unfiltered)
    yolo_model_path = resolve_path("models/checkpoints/mine_segmenter/best.pt")
    mine_seg = MineSegmenter(str(yolo_model_path), confidence_threshold=0.10, device="cpu")
    
    # Run YOLO raw inference
    from ultralytics import YOLO
    raw_yolo = YOLO(str(yolo_model_path))
    results = raw_yolo.predict(source=img_bgr, conf=0.10, iou=0.45, device="cpu", verbose=False)
    
    res0 = results[0]
    boxes = res0.boxes
    masks = res0.masks
    
    print(f"\n[Raw YOLO Predictions (conf=0.10, no leaf mask filter)]")
    print(f"  Model Names Dict   : {raw_yolo.names}")
    print(f"  Detections Count   : {len(boxes) if boxes is not None else 0}")

    raw_yolo_vis = img_bgr.copy()
    combined_raw_yolo_mask = np.zeros((h, w), dtype=np.uint8)

    yolo_dets = []
    if boxes is not None and len(boxes) > 0:
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        masks_data = masks.data.cpu().numpy() if masks is not None else []

        for i in range(len(boxes)):
            bx = [int(v) for v in boxes_xyxy[i]]
            conf = float(round(confs[i], 4))
            cls_id = int(cls_ids[i])
            cls_name = raw_yolo.names.get(cls_id, f"class_{cls_id}")
            
            # Mask
            if i < len(masks_data):
                m_single = (masks_data[i] > 0.5).astype(np.uint8) * 255
                m_single_res = cv2.resize(m_single, (w, h), interpolation=cv2.INTER_NEAREST)
                combined_raw_yolo_mask = cv2.bitwise_or(combined_raw_yolo_mask, m_single_res)
                area_px = int(np.count_nonzero(m_single_res))
            else:
                area_px = (bx[2] - bx[0]) * (bx[3] - bx[1])

            yolo_dets.append({
                "index": i + 1,
                "class": cls_name,
                "confidence": conf,
                "bbox": bx,
                "area_pixels": area_px,
                "area_pct_of_frame": round(100.0 * area_px / (w * h), 2),
            })

            # Draw on visualization
            color = (0, 0, 255) if conf < 0.20 else ((0, 165, 255) if conf < 0.30 else (0, 255, 255))
            cv2.rectangle(raw_yolo_vis, (bx[0], bx[1]), (bx[2], bx[3]), color, 4)
            label = f"#{i+1} {cls_name} conf:{conf:.3f} area:{area_px:,}px"
            cv2.putText(raw_yolo_vis, label, (bx[0] + 10, bx[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        for d in yolo_dets:
            print(f"  Det #{d['index']}: class={d['class']}, conf={d['confidence']:.4f}, bbox={d['bbox']}, area={d['area_pixels']:,} px ({d['area_pct_of_frame']}%)")

    cv2.imwrite(str(out_dir / "05_raw_yolo_detections.jpg"), raw_yolo_vis)
    cv2.imwrite(str(out_dir / "06_raw_yolo_masks.png"), combined_raw_yolo_mask)

    # 4. Intersected Damage Mask (Leaf Mask AND YOLO Damage Mask)
    intersected_mask = cv2.bitwise_and(unet_mask_full, combined_raw_yolo_mask)
    intersected_px = int(np.count_nonzero(intersected_mask))
    leaf_px = int(np.count_nonzero(unet_mask_full))
    calc_dmg_pct = round(100.0 * intersected_px / max(1, leaf_px), 2)

    cv2.imwrite(str(out_dir / "07_intersected_damage_mask.png"), intersected_mask)

    print(f"\n[Mask Intersection Analysis]")
    print(f"  Leaf Pixels (U-Net)     : {leaf_px:,} ({leaf_px/(w*h)*100:.2f}% of frame)")
    print(f"  YOLO Damage Pixels (Raw): {np.count_nonzero(combined_raw_yolo_mask):,}")
    print(f"  Intersected Mine Pixels : {intersected_px:,}")
    print(f"  Reported Damage Pct     : {calc_dmg_pct}%")

    # 5. Full Pipeline Output
    pipeline = InferencePipeline("inference.yaml")
    pipe_res = pipeline.predict(img_bgr, image_id="FAILED_SCAN_TEST")

    print(f"\n[Pipeline Result]")
    print(f"  Damage Detected : {pipe_res.get('damage_detected')}")
    print(f"  Damage Pct      : {pipe_res.get('damage_percentage')}%")
    print(f"  Severity Tier   : {pipe_res.get('severity_tier')}")
    print(f"  Decision        : {pipe_res.get('decision')}")
    print(f"  Regions Count   : {len(pipe_res.get('evidence_regions', []))}")
    for reg in pipe_res.get("evidence_regions", []):
        print(f"    Region {reg['label']}: area={reg['area_pixels']:,} px, conf={reg['confidence']}, bbox={reg['bbox']}")

    if "annotated_image" in pipe_res and pipe_res["annotated_image"] is not None:
        cv2.imwrite(str(out_dir / "08_pipeline_annotated_result.jpg"), pipe_res["annotated_image"])

    # Save summary report json
    summary_data = {
        "image_file": str(img_path.name),
        "dimensions": [w, h],
        "aspect_ratio": aspect_ratio,
        "unet_probs_stats": {
            "min": float(probs.min()),
            "max": float(probs.max()),
            "mean": float(probs.mean()),
            "median": float(np.median(probs)),
            "std": float(probs.std()),
        },
        "yolo_detections": yolo_dets,
        "intersection": {
            "leaf_pixels": leaf_px,
            "raw_mine_pixels": int(np.count_nonzero(combined_raw_yolo_mask)),
            "intersected_pixels": intersected_px,
            "damage_percentage": calc_dmg_pct,
        },
        "pipeline_result": {
            "damage_detected": pipe_res.get("damage_detected"),
            "damage_percentage": pipe_res.get("damage_percentage"),
            "severity_tier": pipe_res.get("severity_tier"),
            "decision": pipe_res.get("decision"),
            "region_count": len(pipe_res.get("evidence_regions", [])),
        }
    }
    with open(out_dir / "diagnostic_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n✅ Diagnostic reproduction complete! Artifacts saved to reports/debug_segmentation/")


if __name__ == "__main__":
    reproduce()

