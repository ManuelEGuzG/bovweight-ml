import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.estimator import WeightEstimator


def crear_estimador():
    est = WeightEstimator.__new__(WeightEstimator)
    est.min_weight = 80
    est.max_weight = 900
    return est


def test_peso_estimado_no_baja_del_minimo():
    est = crear_estimador()

    peso = est._compute_weight(0.001, 0.45)

    assert peso >= est.min_weight


def test_peso_estimado_no_supera_el_maximo():
    est = crear_estimador()

    peso = est._compute_weight(10.0, 0.45)

    assert peso <= est.max_weight


def test_peso_aumenta_si_el_area_detectada_aumenta():
    est = crear_estimador()

    peso_pequeno = est._compute_weight(0.05, 0.45)
    peso_grande = est._compute_weight(0.50, 0.45)

    assert peso_grande > peso_pequeno
