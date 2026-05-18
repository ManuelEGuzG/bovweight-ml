from flask import Blueprint, request, jsonify, current_app
from app.services.estimator import WeightEstimator
from app.utils.image_utils import decode_image
import traceback

estimate_bp = Blueprint("estimate", __name__)

_estimator: WeightEstimator | None = None


def get_estimator() -> WeightEstimator:
    global _estimator
    if _estimator is None:
        _estimator = WeightEstimator(
            model_path=current_app.config["MODEL_PATH"],
            confidence_threshold=current_app.config["CONFIDENCE_THRESHOLD"],
            min_weight=current_app.config["MIN_WEIGHT_KG"],
            max_weight=current_app.config["MAX_WEIGHT_KG"],
        )
    return _estimator


def _get_image_bytes(req) -> tuple[bytes, str | None, float | None, str | None]:
    """Extrae imagen, animal_id, distancia y ángulo del request."""
    if req.content_type and "multipart/form-data" in req.content_type:
        if "image" not in req.files and "photo" not in req.files:
            raise ValueError("No se encontró el campo 'image' o 'photo'")
        file      = req.files.get("image") or req.files.get("photo")
        img_bytes = file.read()
        animal_id = req.form.get("animal_id")
        distance  = req.form.get("distance_meters")
        angle     = req.form.get("photo_angle", "lateral")
        return img_bytes, animal_id, float(distance) if distance else None, angle
    else:
        data = req.get_json(force=True)
        if not data or "image_base64" not in data:
            raise ValueError("Se requiere 'image_base64' en el body JSON")
        img_bytes = decode_image(data["image_base64"])
        return (
            img_bytes,
            data.get("animal_id"),
            float(data["distance_meters"]) if data.get("distance_meters") else None,
            data.get("photo_angle", "lateral"),
        )


@estimate_bp.post("/estimate")
def estimate():
    """Estima el peso del bovino más prominente en la imagen."""
    try:
        img_bytes, animal_id, distance, angle = _get_image_bytes(request)
        result = get_estimator().estimate(img_bytes, distance_meters=distance, photo_angle=angle)
        return jsonify({
            "animal_id":           animal_id,
            "estimated_weight_kg": result["weight_kg"],
            "confidence":          result["confidence"],
            "bovine_detected":     result["detected"],
            "bounding_box":        result["bbox"],
            "warning":             result.get("warning"),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Error interno", "detail": str(e)}), 500


@estimate_bp.post("/estimate/all")
def estimate_all():
    """Estima el peso de TODOS los bovinos en la imagen."""
    try:
        img_bytes, _, distance, angle = _get_image_bytes(request)
        result = get_estimator().estimate_all(img_bytes, distance_meters=distance, photo_angle=angle)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Error interno", "detail": str(e)}), 500