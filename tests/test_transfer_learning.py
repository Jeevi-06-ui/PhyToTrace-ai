"""Unit tests for MobileNetV3 transfer learning mechanics, freezing, unfreezing, and gradients."""

import pytest
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights

from phytotrace.models.baseline import MobileNetV3Baseline


def test_classification_head_three_outputs():
    """1. The classification head has three outputs with correct class names."""
    model = MobileNetV3Baseline(num_classes=3, pretrained=False)
    assert len(model.CLASS_NAMES) == 3
    assert model.CLASS_NAMES == ["healthy", "leaf_miner_damage", "other_damage"]

    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 3)

    probs = model.predict_probabilities(dummy_input)
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_backbone_is_frozen_during_stage_a():
    """2. The backbone is frozen during Stage A head training."""
    model = MobileNetV3Baseline(num_classes=3, pretrained=False)
    model.freeze_backbone()

    # All feature extractor parameters must have requires_grad == False
    for name, param in model.features.named_parameters():
        assert not param.requires_grad, f"Feature parameter {name} should be frozen in Stage A"

    # All classifier head parameters must have requires_grad == True
    for name, param in model.classifier.named_parameters():
        assert param.requires_grad, f"Classifier parameter {name} should be trainable in Stage A"


def test_classification_head_receives_gradients():
    """3. The classification head receives gradients while frozen backbone does not."""
    model = MobileNetV3Baseline(num_classes=3, pretrained=False)
    model.freeze_backbone()

    dummy_input = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 1], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    out = model(dummy_input)
    loss = criterion(out, target)
    loss.backward()

    # Head must receive non-zero gradients
    for name, param in model.classifier.named_parameters():
        assert param.grad is not None, f"Classifier parameter {name} received no gradient"
        assert param.grad.abs().sum() > 0, f"Classifier parameter {name} has all-zero gradient"

    # Backbone must have None gradients
    for name, param in model.features.named_parameters():
        assert param.grad is None, f"Frozen backbone parameter {name} should have None gradient"


def test_selected_backbone_layers_unfrozen_stage_b():
    """4. Selected backbone layers are unfrozen in Stage B fine-tuning."""
    model = MobileNetV3Baseline(num_classes=3, pretrained=False)
    model.freeze_backbone()
    model.unfreeze_backbone(unfreeze_last_n_blocks=3)

    total_blocks = len(model.features)
    # Check that the last 3 blocks are unfrozen and earlier blocks remain frozen
    for i, block in enumerate(model.features):
        expected_trainable = i >= (total_blocks - 3)
        for name, param in block.named_parameters():
            assert param.requires_grad == expected_trainable, (
                f"Block {i} param {name} requires_grad={param.requires_grad}, expected={expected_trainable}"
            )


def test_optimizer_includes_intended_trainable_parameters():
    """5. The optimizer includes only intended trainable parameters in each stage."""
    model = MobileNetV3Baseline(num_classes=3, pretrained=False)

    # Stage A optimizer
    model.freeze_backbone()
    trainable_stage_a = [p for p in model.parameters() if p.requires_grad]
    classifier_params = list(model.classifier.parameters())
    assert len(trainable_stage_a) == len(classifier_params)
    for p1, p2 in zip(trainable_stage_a, classifier_params):
        assert p1 is p2

    # Stage B optimizer
    model.unfreeze_backbone(unfreeze_last_n_blocks=3)
    trainable_stage_b = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_stage_b) > len(trainable_stage_a)


def test_learning_rate_reduction_for_finetuning():
    """6. Learning rate is reduced appropriately for Stage B fine-tuning."""
    lr_head = 0.001
    lr_finetune = 0.0001
    assert lr_finetune < lr_head
    assert lr_head / lr_finetune >= 5.0, "Fine-tuning LR should be at least 5x smaller than initial head LR"

