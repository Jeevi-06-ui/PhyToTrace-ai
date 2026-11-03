"""Unit tests verifying PyTorch model tensor shapes and loss functions."""

import pytest
import torch

from phytotrace.models.baseline import MobileNetV3Baseline
from phytotrace.models.leaf_segmenter import LightweightUNet
from phytotrace.models.species_classifier import MaskedBCEWithLogitsLoss, PestEvidenceClassifier


def test_baseline_mobilenetv3_shapes():
    model = MobileNetV3Baseline(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 3)

    probs = model.predict_probabilities(x)
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_baseline_freeze_unfreeze():
    model = MobileNetV3Baseline(num_classes=3, pretrained=False)
    model.freeze_backbone()
    for param in model.features.parameters():
        assert not param.requires_grad
    for param in model.classifier.parameters():
        assert param.requires_grad

    model.unfreeze_backbone()
    for param in model.features.parameters():
        assert param.requires_grad


def test_leaf_unet_shapes():
    unet = LightweightUNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    out = unet(x)
    assert out.shape == (2, 1, 256, 256)


def test_species_evidence_classifier_shapes():
    model = PestEvidenceClassifier(pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (4, 2)

    scores = model.predict_evidence_scores(x)
    assert scores.shape == (4, 2)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_masked_bce_loss():
    criterion = MaskedBCEWithLogitsLoss()
    logits = torch.tensor([[2.0, -2.0], [0.5, 0.5]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    
    # Second sample is unverified (mask=0)
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    loss = criterion(logits, targets, mask)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)

