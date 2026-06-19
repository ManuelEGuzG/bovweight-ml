import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.estimator import WeightEstimator


def crear_estimador():
    est = WeightEstimator.__new__(WeightEstimator)
    est.min_weight = 80
    est.max_weight = 900
    return est


def test_warning_para_area_muy_pequena():
    est = crear_estimador()

    warning = est._build_warning(0.01)

    assert warning is not None
    assert "Acérquese" in warning


def test_warning_para_area_muy_grande():
    est = crear_estimador()

    warning = est._build_warning(0.95)

    assert warning is not None
    assert "Aléjese" in warning


def test_sin_warning_para_area_adecuada():
    est = crear_estimador()

    warning = est._build_warning(0.35)

    assert warning is None
