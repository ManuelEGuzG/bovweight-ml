import base64
import binascii
import numpy as np
import cv2


def decode_image(b64_string: str) -> bytes:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    try:
        return base64.b64decode(b64_string)
    except binascii.Error as e:
        raise ValueError(f"Base64 inválido: {e}")


def bytes_to_cv2(img_bytes: bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def cv2_to_bytes(img, ext: str = ".jpg") -> bytes:
    _, buf = cv2.imencode(ext, img)
    return buf.tobytes()