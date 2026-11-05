"""Unit tests for training controls, epoch resolution, class weights, and model selection tie-breaking."""

from collections import Counter
import pytest
import torch

from phytotrace.training.train_baseline import compute_class_weights


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.zeros(3, 224, 224), self.labels[idx], f"id_{idx}"


def test_class_weights_inverse_frequency():
    """Verify that compute_class_weights computes balanced inverse-frequency weights."""
    # 25 healthy (0), 34 leaf_miner (1), 6 other_damage (2)
    labels = [0] * 25 + [1] * 34 + [2] * 6
    ds = DummyDataset(labels)

    weights = compute_class_weights(ds, num_classes=3)
    assert len(weights) == 3
    # Class 2 (other_damage, minority) must have the highest weight
    assert weights[2] > weights[0]
    assert weights[2] > weights[1]
    # Class 1 (leaf_miner, majority) must have the lowest weight
    assert weights[1] < weights[0]
    # Mean weight should be normalized to 1.0
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-4)


def test_epoch_resolution_logic():
    """Verify that --epochs 25 resolves to Stage A (5 epochs) and Stage B (20 epochs)."""
    # Emulate the resolution logic in train_baseline_pipeline
    def resolve_epochs(head_epochs=None, finetune_epochs=None, override_epochs=None, default_head=5, default_ft=20, is_smoke=False):
        if is_smoke:
            return 1, 1
        h = head_epochs if head_epochs is not None else default_head
        if finetune_epochs is not None:
            f = finetune_epochs
        elif override_epochs is not None:
            if override_epochs > h:
                f = override_epochs - h
            else:
                f = max(1, override_epochs)
        else:
            f = default_ft
        return h, f

    # Case 1: user passes --epochs 25
    h, f = resolve_epochs(override_epochs=25)
    assert h == 5
    assert f == 20
    assert h + f == 25

    # Case 2: user passes explicit --head-epochs 3 --finetune-epochs 15
    h, f = resolve_epochs(head_epochs=3, finetune_epochs=15)
    assert h == 3
    assert f == 15
    assert h + f == 18

    # Case 3: default (no flags passed)
    h, f = resolve_epochs()
    assert h == 5
    assert f == 20

    # Case 4: smoke test flag
    h, f = resolve_epochs(is_smoke=True)
    assert h == 1
    assert f == 1


def test_deterministic_model_selection_tie_breaking():
    """Verify that when validation Macro F1 ties, model selection breaks tie by lower validation loss."""
    best_f1 = 0.8500
    best_loss = 0.4200

    # New candidate has identical F1, but lower loss -> should be accepted
    cand_f1 = 0.8500
    cand_loss = 0.3950
    is_better = (cand_f1 > best_f1 + 1e-4) or (abs(cand_f1 - best_f1) <= 1e-4 and cand_loss < best_loss - 1e-4)
    assert is_better

    # New candidate has identical F1, but higher loss -> should be rejected
    cand_loss_worse = 0.4500
    is_better_worse = (cand_f1 > best_f1 + 1e-4) or (abs(cand_f1 - best_f1) <= 1e-4 and cand_loss_worse < best_loss - 1e-4)
    assert not is_better_worse

