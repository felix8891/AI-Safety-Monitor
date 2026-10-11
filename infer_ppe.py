from pathlib import Path
from collections import Counter
import argparse
import sys

import cv2
import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "yolo8n.pt"
ALLOWED_CLASSES = {"person", "helmet", "safety-vest"}


def main():
    parser = argparse.ArgumentParser(
        description="AI Safety Monitor - PPE image inference"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="factory_test.jpg",
        help="Path to the input image",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Minimum detection confidence (default: 0.35)",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = PROJECT_DIR / source_path

    if not MODEL_PATH.is_file():
        sys.exit(f"Model tidak ditemukan: {MODEL_PATH}")

    if not source_path.is_file():
        sys.exit(f"Gambar tidak ditemukan: {source_path}")

    if not 0 < args.conf < 1:
        sys.exit("Confidence harus lebih dari 0 dan kurang dari 1.")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"GPU: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Gambar: {source_path.name}")

    model = YOLO(str(MODEL_PATH))
    names = model.names

    class_ids = [
        class_id
        for class_id, name in names.items()
        if str(name).lower() in ALLOWED_CLASSES
    ]

    found_classes = {str(names[i]).lower() for i in class_ids}
    missing_classes = ALLOWED_CLASSES - found_classes
    if missing_classes:
        print(f"Peringatan: kelas tidak ditemukan: {sorted(missing_classes)}")

    if not class_ids:
        sys.exit("Tidak ada kelas target yang cocok dengan model.")

    results = model.predict(
        source=str(source_path),
        device=device,
        imgsz=640,
        conf=args.conf,
        iou=0.5,
        classes=class_ids,
        verbose=False,
    )

    result = results[0]
    output_dir = PROJECT_DIR / "outputs" / "ppe_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_path.stem}_detected.jpg"

    annotated = result.plot()
    if not cv2.imwrite(str(output_path), annotated):
        sys.exit(f"Gagal menyimpan gambar: {output_path}")

    counts = Counter()
    print("\nHasil deteksi:")
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = str(names[class_id])
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (
                int(value) for value in box.xyxy[0].tolist()
            )
            counts[label] += 1
            print(
                f"- {label}: {confidence:.2f} "
                f"| bbox=({x1}, {y1}, {x2}, {y2})"
            )
    else:
        print("- Tidak ada objek yang terdeteksi.")

    print("\nRingkasan:")
    for label in ("person", "helmet", "safety-vest"):
        print(f"- {label}: {counts[label]}")

    print(f"\nGambar hasil disimpan di:\n{output_path}")
    print(
        "\nCatatan: hasil deteksi objek belum merupakan "
        "keputusan kepatuhan SOP."
    )


if __name__ == "__main__":
    main()
