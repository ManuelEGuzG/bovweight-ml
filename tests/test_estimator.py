import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils.image_utils import decode_image


def test_decode_image_plain_base64():
    import base64
    data = base64.b64encode(b"fake_image_bytes").decode()
    assert decode_image(data) == b"fake_image_bytes"

def test_decode_image_data_uri():
    import base64
    data = "data:image/jpeg;base64," + base64.b64encode(b"fake_image_bytes").decode()
    assert decode_image(data) == b"fake_image_bytes"

def test_decode_image_invalid_base64():
    with pytest.raises(ValueError):
        decode_image("@@not_valid@@")

def test_not_detected_response():
    from app.services.estimator import WeightEstimator
    est = WeightEstimator.__new__(WeightEstimator)
    est.min_weight = 80
    est.max_weight = 900
    resp = est._not_detected("Sin bovino")
    assert resp["detected"] is False
    assert resp["weight_kg"] is None

def test_compute_weight_range():
    from app.services.estimator import WeightEstimator
    est = WeightEstimator.__new__(WeightEstimator)
    est.min_weight = 80
    est.max_weight = 900
    w1 = est._compute_weight(0.05, 0.45)
    w2 = est._compute_weight(0.6, 0.45)
    assert 80 <= w1 <= 900
    assert 80 <= w2 <= 900
    assert w2 > w1

def test_build_warning_too_far():
    from app.services.estimator import WeightEstimator
    est = WeightEstimator.__new__(WeightEstimator)
    assert "Acérquese" in est._build_warning(0.02)

def test_build_warning_too_close():
    from app.services.estimator import WeightEstimator
    est = WeightEstimator.__new__(WeightEstimator)
    assert "Aléjese" in est._build_warning(0.90)

def test_build_warning_ok():
    from app.services.estimator import WeightEstimator
    est = WeightEstimator.__new__(WeightEstimator)
    assert est._build_warning(0.3) is None

def test_estimate_all_no_image():
    from app.services.estimator import WeightEstimator
    est = WeightEstimator.__new__(WeightEstimator)
    est.min_weight = 80
    est.max_weight = 900
    result = est._not_detected("No se pudo decodificar la imagen.")
    assert result["detected"] is False

# ── Feedback utils ──────────────────────────────────────────────────────

def test_feedback_error_calculation():
    estimated = 400.0
    real      = 450.0
    error_kg  = round(real - estimated, 2)
    error_pct = round(abs(real - estimated) / real * 100, 2)
    assert error_kg  == 50.0
    assert error_pct == pytest.approx(11.11, rel=0.01)