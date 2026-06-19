import base64
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils.image_utils import decode_image


def test_decode_image_con_png_data_uri():
    contenido = b"imagen_png_prueba"
    data = "data:image/png;base64," + base64.b64encode(contenido).decode()

    assert decode_image(data) == contenido


def test_decode_image_con_cadena_vacia():
    with pytest.raises(ValueError):
        decode_image("")


def test_decode_image_con_texto_sin_base64():
    with pytest.raises(ValueError):
        decode_image("esto-no-es-base64")
