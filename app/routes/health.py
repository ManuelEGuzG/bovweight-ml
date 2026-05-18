from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    return jsonify({
        "status":  "ok",
        "service": "bovweight-ml",
        "version": "1.0.0"
    })