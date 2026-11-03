"""Unit tests for the PhytoTrace uncertainty and decision engine."""

import pytest

from phytotrace.models.decision_engine import PhytoTraceDecisionEngine


@pytest.fixture
def engine():
    return PhytoTraceDecisionEngine(
        tuta_threshold=0.60,
        liriomyza_threshold=0.60,
        uncertainty_margin=0.15,
        min_leaf_ratio=0.05,
    )


def test_quality_rejection(engine):
    qc = {"is_acceptable": False, "reasons": ["Severe blur detected"]}
    res = engine.evaluate(
        quality_assessment=qc,
        leaf_ratio=0.5,
        damage_detected=True,
        damage_percentage=10.0,
        tuta_evidence=0.8,
        liriomyza_evidence=0.1,
    )
    assert res["decision"] == "image_quality_insufficient"
    assert res["uncertainty_status"] == "quality_rejected"


def test_insufficient_leaf_ratio(engine):
    qc = {"is_acceptable": True, "reasons": []}
    res = engine.evaluate(
        quality_assessment=qc,
        leaf_ratio=0.02,  # < 0.05
        damage_detected=False,
        damage_percentage=0.0,
        tuta_evidence=None,
        liriomyza_evidence=None,
    )
    assert res["decision"] == "unsupported_image"


def test_healthy_decision(engine):
    qc = {"is_acceptable": True, "reasons": []}
    res = engine.evaluate(
        quality_assessment=qc,
        leaf_ratio=0.4,
        damage_detected=False,
        damage_percentage=0.0,
        tuta_evidence=None,
        liriomyza_evidence=None,
        baseline_class="healthy",
    )
    assert res["decision"] == "no_visible_mine"


def test_tuta_dominant_decision(engine):
    qc = {"is_acceptable": True, "reasons": []}
    res = engine.evaluate(
        quality_assessment=qc,
        leaf_ratio=0.4,
        damage_detected=True,
        damage_percentage=12.5,
        tuta_evidence=0.85,
        liriomyza_evidence=0.10,
    )
    assert res["decision"] == "possible_tuta"
    assert res["uncertainty_status"] == "tuta_evidence_dominant"


def test_liriomyza_dominant_decision(engine):
    qc = {"is_acceptable": True, "reasons": []}
    res = engine.evaluate(
        quality_assessment=qc,
        leaf_ratio=0.4,
        damage_detected=True,
        damage_percentage=7.0,
        tuta_evidence=0.15,
        liriomyza_evidence=0.78,
    )
    assert res["decision"] == "possible_liriomyza"
    assert res["uncertainty_status"] == "liriomyza_evidence_dominant"


def test_mixed_infestation_decision(engine):
    qc = {"is_acceptable": True, "reasons": []}
    res = engine.evaluate(
        quality_assessment=qc,
        leaf_ratio=0.4,
        damage_detected=True,
        damage_percentage=15.0,
        tuta_evidence=0.75,
        liriomyza_evidence=0.72,
    )
    assert res["decision"] == "possible_mixed"
    assert res["uncertainty_status"] == "mixed_evidence_detected"


def test_abstention_on_uncertain_boundary(engine):
    qc = {"is_acceptable": True, "reasons": []}
    # Tuta score 0.52 is in [0.60 - 0.15, 0.60) = [0.45, 0.60)
    res = engine.evaluate(
        quality_assessment=qc,
        leaf_ratio=0.4,
        damage_detected=True,
        damage_percentage=6.0,
        tuta_evidence=0.52,
        liriomyza_evidence=0.20,
    )
    assert res["decision"] == "mine_detected_species_uncertain"
    assert res["uncertainty_status"] == "evidence_boundary_abstention"


def test_species_model_unavailable_fallback(engine):
    qc = {"is_acceptable": True, "reasons": []}
    res = engine.evaluate(
        quality_assessment=qc,
        leaf_ratio=0.4,
        damage_detected=True,
        damage_percentage=8.0,
        tuta_evidence=None,
        liriomyza_evidence=None,
    )
    assert res["decision"] == "mine_detected_species_uncertain"
    assert res["uncertainty_status"] == "species_model_unavailable"
    assert "Visible leaf-mining damage may be present" in res["explanation"]

