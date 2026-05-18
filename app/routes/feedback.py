from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import json, os, traceback

feedback_bp = Blueprint("feedback", __name__)

FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", "data/feedback.jsonl")


@feedback_bp.post("/feedback")
def feedback():
    try:
        data = request.get_json(force=True)
        for field in ["animal_id", "estimated_weight_kg", "real_weight_kg"]:
            if field not in data:
                return jsonify({"error": f"Campo requerido: '{field}'"}), 400

        estimated = float(data["estimated_weight_kg"])
        real      = float(data["real_weight_kg"])

        if not (20 <= real <= 1200):
            return jsonify({"error": "real_weight_kg fuera de rango (20–1200 kg)"}), 400

        record = {
            "animal_id":           data["animal_id"],
            "estimated_weight_kg": estimated,
            "real_weight_kg":      real,
            "error_kg":            round(real - estimated, 2),
            "error_pct":           round(abs(real - estimated) / real * 100, 2),
            "notes":               data.get("notes", ""),
            "timestamp":           datetime.now(timezone.utc).isoformat(),
        }

        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return jsonify({
            "message":   "Feedback registrado correctamente.",
            "error_kg":  record["error_kg"],
            "error_pct": record["error_pct"],
        }), 201

    except ValueError:
        return jsonify({"error": "Los pesos deben ser números válidos"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Error interno", "detail": str(e)}), 500


@feedback_bp.get("/feedback/stats")
def feedback_stats():
    try:
        if not os.path.exists(FEEDBACK_FILE):
            return jsonify({"total": 0, "message": "Sin feedback registrado aún."})

        records = []
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            return jsonify({"total": 0})

        errors = [abs(r["error_kg"]) for r in records]
        pcts   = [r["error_pct"] for r in records]

        return jsonify({
            "total":         len(records),
            "mae_kg":        round(sum(errors) / len(errors), 2),
            "avg_error_pct": round(sum(pcts) / len(pcts), 2),
            "max_error_kg":  round(max(errors), 2),
            "min_error_kg":  round(min(errors), 2),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Error interno", "detail": str(e)}), 500