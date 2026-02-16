"""Basic import and metadata smoke tests."""

import openerf
import OpenERF


def test_public_api_exports() -> None:
    assert hasattr(openerf, "compute_erf")
    assert hasattr(openerf, "save_erf")
    assert hasattr(openerf, "compute_erf_metrics")
    assert hasattr(openerf, "get_supported_models")


def test_supported_models_has_common_preset() -> None:
    assert "resnet50.a1_in1k" in openerf.get_supported_models()


def test_legacy_top_level_alias() -> None:
    assert OpenERF.save_ERF is openerf.save_erf
    assert OpenERF.compute_ERF is openerf.compute_erf
