from pathlib import Path
from ultralytics import YOLO

MODEL_DIR = Path("app/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Descargando YOLOv8n...")
model = YOLO("yolov8n.pt")
dest  = MODEL_DIR / "bovine_yolov8.pt"
model.save(str(dest))
print(f"Modelo guardado en: {dest}")