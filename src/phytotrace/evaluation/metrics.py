"""Evaluation metrics for classification, segmentation, and pest-evidence."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, any]:
    """Compute rigorous classification metrics for 3-class baseline or single-label evaluation."""
    if len(y_true) == 0:
        return {"status": "NOT_EVALUATED", "message": "No samples available for evaluation."}

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    labels = list(range(len(class_names)))
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    class_stats = {}
    for i, name in enumerate(class_names):
        class_stats[name] = {
            "precision": float(round(per_class_precision[i], 4)),
            "recall": float(round(per_class_recall[i], 4)),
            "f1": float(round(per_class_f1[i], 4)),
            "support": int(np.sum(y_true == i)),
        }

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": class_stats,
        "total_samples": len(y_true),
    }


def compute_binary_segmentation_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Compute IoU and Dice coefficient between binary prediction and ground truth masks."""
    p = (pred_mask > 0).astype(np.float32)
    g = (gt_mask > 0).astype(np.float32)

    intersection = np.sum(p * g)
    union = np.sum(p) + np.sum(g) - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (np.sum(p) + np.sum(g) + smooth)

    return {
        "iou": float(round(iou, 4)),
        "dice": float(round(dice, 4)),
        "pred_pixels": int(np.sum(p)),
        "gt_pixels": int(np.sum(g)),
    }


def compute_multilabel_evidence_metrics(
    y_true: np.ndarray,      # (N, 2)
    y_pred_probs: np.ndarray,# (N, 2)
    valid_mask: np.ndarray,  # (N, 2)
    threshold: float = 0.60,
) -> Dict[str, any]:
    """Compute precision, recall, and F1 for multi-label pest evidence scores,
    evaluating only verified targets.
    """
    if len(y_true) == 0:
        return {"status": "NOT_EVALUATED", "message": "No verified pest annotations available."}

    results = {}
    targets = ["tuta_evidence", "liriomyza_evidence"]

    for i, target in enumerate(targets):
        mask_i = valid_mask[:, i] > 0
        if not np.any(mask_i):
            results[target] = {"status": "NOT_EVALUATED", "message": "No verified ground truth samples."}
            continue

        true_i = y_true[mask_i, i]
        pred_i = (y_pred_probs[mask_i, i] >= threshold).astype(int)

        results[target] = {
            "precision": float(round(precision_score(true_i, pred_i, zero_division=0), 4)),
            "recall": float(round(recall_score(true_i, pred_i, zero_division=0), 4)),
            "f1": float(round(f1_score(true_i, pred_i, zero_division=0), 4)),
            "verified_support": int(np.sum(mask_i)),
            "positive_count": int(np.sum(true_i)),
        }

    return results

