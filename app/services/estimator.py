"""
WeightEstimator — versión corregida
------------------------------------
Bug corregido: la distancia ahora NORMALIZA el rel_area (perspectiva)
en lugar de multiplicar el peso final (lógica invertida).

Fix adicional: filtro estricto de clases — solo COCO 19 (cow) pasa.
La clase 0 (person) ya NO se incluye en las detecciones de bovinos.
Se detecta por separado para dar mensajes de error específicos.

Calibración:
  Bovino adulto CR a 3m de flanco ≈ 25-35% del frame => ~380-450 kg
  Ternero a 3m ≈ 8-15% del frame  => ~180-250 kg
"""
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.utils.image_utils import bytes_to_cv2
from app.utils.image_preprocessing import preprocess_image


class WeightEstimator:

    COCO_COW_CLASS    = 19   # única clase válida para estimación
    COCO_PERSON_CLASS = 0    # detectada pero rechazada con mensaje específico
    REF_DISTANCE      = 3.0  # metros de referencia para calibración

    def __init__(self, model_path: str, confidence_threshold: float = 0.1,
                 min_weight: float = 80.0, max_weight: float = 900.0):
        self.confidence_threshold = confidence_threshold
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._model = self._load_model(model_path)

    # ── Carga ────────────────────────────────────────────────────────────────

    def _load_model(self, model_path: str):
        from ultralytics import YOLO
        if Path(model_path).exists():
            print(f"[ML] Cargando modelo: {model_path}")
            return YOLO(model_path)
        print("[ML] Modelo no encontrado. Usando yolov8n.pt (fallback COCO)")
        return YOLO("yolov8n.pt")

    # ── API pública: bovino más prominente ───────────────────────────────────

    def estimate(self, img_bytes: bytes,
                 distance_meters: float | None = None,
                 photo_angle: str | None = None) -> dict[str, Any]:

        detections = self.estimate_all(img_bytes,
                                       distance_meters=distance_meters,
                                       photo_angle=photo_angle)
        if not detections["bovines"]:
            return {
                "weight_kg":  None,
                "confidence": 0.0,
                "detected":   False,
                "bbox":       None,
                "warning":    detections.get("warning", "No se detectó ningún bovino."),
            }

        best = detections["bovines"][0]
        return {
            "weight_kg":  best["weight_kg"],
            "confidence": best["confidence"],
            "detected":   True,
            "bbox":       best["bbox"],
            "warning":    detections.get("warning"),
        }

    # ── API pública: todos los bovinos ───────────────────────────────────────

    def estimate_all(self, img_bytes: bytes,
                     distance_meters: float | None = None,
                     photo_angle: str | None = None) -> dict[str, Any]:

        img = bytes_to_cv2(img_bytes)
        if img is None:
            return {"count": 0, "bovines": [], "warning": "No se pudo decodificar la imagen."}

        h, w     = img.shape[:2]
        img_proc = preprocess_image(img)
        results  = self._model(img_proc, conf=self.confidence_threshold, verbose=False)

        # ── Separar bovinos de personas ───────────────────────────────────────
        bovine_dets, person_detected = self._get_all_detections(results, w, h)

        # ── Sin bovinos: mensaje específico según qué había ───────────────────
        if not bovine_dets:
            if person_detected:
                warning = (
                    "Se detectó una persona en la imagen, no un bovino. "
                    "Por favor, tome la foto directamente del animal."
                )
            else:
                warning = (
                    "No se detectó ningún bovino en la imagen. "
                    "Asegúrese de que el animal esté completamente visible, "
                    "con buena iluminación y sin objetos que lo tapen."
                )
            return {"count": 0, "bovines": [], "warning": warning}

        # ── Calcular peso para cada bovino detectado ──────────────────────────
        bovines = []
        for i, det in enumerate(bovine_dets):
            weight = self._compute_weight(
                det, w, h,
                distance_meters=distance_meters,
                photo_angle=photo_angle,
            )
            bovines.append({
                "id":         i + 1,
                "weight_kg":  round(weight, 1),
                "confidence": round(det["conf"], 3),
                "bbox":       det["bbox"],
            })

        # ── Advertencia global ────────────────────────────────────────────────
        warning = None
        if person_detected:
            warning = (
                "Se detectó una persona en la imagen además del bovino. "
                "Para mayor precisión, tome la foto solo del animal."
            )
        else:
            warning = self._build_warning(bovine_dets[0]["rel_area"])

        return {"count": len(bovines), "bovines": bovines, "warning": warning}

    # ── Detecciones: separa bovinos de personas ───────────────────────────────

    def _get_all_detections(
        self, results, img_w: int, img_h: int
    ) -> tuple[list[dict], bool]:
        """
        Retorna (bovine_dets, person_detected).

        bovine_dets : lista de dicts con info del bbox, ordenada por rel_area desc.
        person_detected : True si se detectó al menos una persona (clase 0).

        SOLO la clase 19 (cow) entra en bovine_dets.
        La clase 0 (person) se registra en el flag pero NO se estima su peso.
        Cualquier otra clase se ignora silenciosamente.
        """
        bovine_dets     = []
        person_detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])

                # ── Persona detectada: marcar flag y continuar ────────────────
                if cls_id == self.COCO_PERSON_CLASS:
                    person_detected = True
                    continue

                # ── Solo procesar la clase cow ────────────────────────────────
                if cls_id != self.COCO_COW_CLASS:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                bw   = x2 - x1
                bh   = y2 - y1
                area = bw * bh

                bovine_dets.append({
                    "conf":         conf,
                    "bbox":         [round(x1), round(y1), round(x2), round(y2)],
                    "rel_area":     area / (img_w * img_h),
                    "aspect_ratio": bh / bw if bw > 0 else 1.0,
                    "bbox_w":       bw,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

        # Ordenar por área relativa descendente (bovino más prominente primero)
        bovine_dets.sort(key=lambda d: d["rel_area"], reverse=True)
        return bovine_dets, person_detected

    # ── Cálculo de peso (calibrado) ──────────────────────────────────────────

    def _compute_weight(self, det: dict, img_w: int, img_h: int,
                        distance_meters: float | None = None,
                        photo_angle: str | None = None) -> float:
        """
        Lógica corregida:

        1. NORMALIZAR rel_area por distancia real (perspectiva):
           Si la foto fue a 1m, el animal parece más grande en píxeles.
           Normalizamos al equivalente de 3m de referencia:
             rel_area_norm = rel_area * (distancia / 3)²

           Ejemplo: animal a 1m ocupa 60% del frame.
           A 3m ese mismo animal ocuparía: 0.60 * (1/3)² = 0.067 (7%)
           => peso estimado correcto para su tamaño real.

        2. Si no hay distancia conocida: inferir por posición Y y rel_width.

        3. CALIBRACIÓN (A=50, B=660):
           f(rel_area) = 50 + sqrt(rel_area) * 660
           f(0.25) = 380 kg  ← bovino adulto promedio CR a 3m de flanco
           f(0.10) = 259 kg  ← animal mediano
           f(0.05) = 198 kg  ← ternero
        """
        rel_area     = det["rel_area"]
        aspect_ratio = det["aspect_ratio"]
        bbox_w       = det["bbox_w"]
        img_w_f      = float(img_w)
        y_center     = (det["y1"] + det["y2"]) / 2
        img_h_f      = float(img_w)  # conservado del original

        # ── 1. Normalizar rel_area por distancia ──────────────────────────────
        if distance_meters and distance_meters > 0:
            norm_factor   = (distance_meters / self.REF_DISTANCE) ** 2
            rel_area_norm = rel_area * norm_factor
        else:
            # Inferir distancia por posición Y y ancho relativo
            rel_width = bbox_w / img_w_f
            rel_y     = y_center / float(img_w)

            if rel_y < 0.30:
                inferred_dist = 7.0
            elif rel_y < 0.45:
                inferred_dist = 5.0
            elif rel_y < 0.60:
                inferred_dist = 3.5
            else:
                inferred_dist = 2.5

            if rel_width > 0.70:
                inferred_dist = max(1.5, inferred_dist * 0.5)
            elif rel_width > 0.45:
                inferred_dist = inferred_dist * 0.75

            norm_factor   = (inferred_dist / self.REF_DISTANCE) ** 2
            rel_area_norm = rel_area * norm_factor

        rel_area_norm = min(rel_area_norm, 1.0)

        # ── 2. Factor de ángulo ───────────────────────────────────────────────
        angle = (photo_angle or "lateral").lower()
        angle_map = {
            "lateral":   1.00,
            "diagonal":  0.90,
            "frontal":   0.78,
            "posterior": 0.80,
        }
        angle_factor = angle_map.get(angle, 1.0)

        if angle in ("lateral", "diagonal"):
            if 0.35 <= aspect_ratio <= 0.65:
                angle_factor *= 1.0
            elif aspect_ratio < 0.30:
                angle_factor *= 0.88
            elif aspect_ratio > 0.85:
                angle_factor *= 0.85

        # ── 3. Peso base ──────────────────────────────────────────────────────
        base   = 50.0 + math.sqrt(rel_area_norm) * 660.0
        weight = base * angle_factor

        return float(max(self.min_weight, min(self.max_weight, weight)))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_warning(self, rel_area: float) -> str | None:
        if rel_area < 0.03:
            return "El animal está muy lejos. Acérquese más para mejor precisión."
        if rel_area < 0.08:
            return "El animal ocupa poco espacio en la imagen. Indique la distancia para mayor precisión."
        if rel_area > 0.80:
            return "El animal está muy cerca. Aléjese un poco para mejor precisión."
        return None