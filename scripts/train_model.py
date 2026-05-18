"""
Fine-tuning de YOLOv8 con imágenes de ganado bovino.

Estructura esperada del dataset:
  data/
    dataset.yaml
    images/
      train/   *.jpg
      val/     *.jpg
    labels/
      train/   *.txt  (formato YOLO: class cx cy w h)
      val/     *.txt

Uso:
  python scripts/train_model.py --data data/dataset.yaml --epochs 50
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/dataset.yaml", help="Path al dataset.yaml")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--base",    default="yolov8n.pt", help="Modelo base para fine-tuning")
    parser.add_argument("--output",  default="app/models/bovine_yolov8.pt")
    args = parser.parse_args()

    print(f"[Train] Iniciando fine-tuning sobre: {args.base}")
    model = YOLO(args.base)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="bovweight_finetune",
        project="runs/train",
        patience=10,          # early stopping
        save=True,
        plots=True,
    )

    # Copiar el mejor modelo al path de producción
    best = Path("runs/train/bovweight_finetune/weights/best.pt")
    if best.exists():
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(best, dest)
        print(f"[Train] Modelo guardado en: {dest}")
    else:
        print("[Train] No se encontró best.pt — revisa los logs de entrenamiento.")


if __name__ == "__main__":
    main()