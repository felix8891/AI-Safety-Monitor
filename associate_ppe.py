from pathlib import Path
import argparse
import sys

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "yolo8n.pt"

MIN_CONF = {
    "person": 0.35,
    "helmet": 0.35,
    "safety-vest": 0.15,
}

COLORS = {
    "person": (70, 200, 70),
    "helmet": (255, 200, 0),
    "safety-vest": (220, 0, 220),
    "uncertain": (0, 165, 255),
}


def coords(box):
    return tuple(int(round(v)) for v in box.xyxy[0].tolist())


def center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    return inter / max(1, area_a + area_b - inter)


def overlap_of_item(item, person):
    x1, y1, x2, y2 = item
    px1, py1, px2, py2 = person

    ix1, iy1 = max(x1, px1), max(y1, py1)
    ix2, iy2 = min(x2, px2), min(y2, py2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    item_area = max(1, (x2 - x1) * (y2 - y1))
    return inter / item_area


def deduplicate(detections, iou_threshold=0.50):
    """Extra class-wise NMS; higher-confidence box is kept first."""
    detections = sorted(
        detections, key=lambda item: item["conf"], reverse=True
    )
    kept = []

    for item in detections:
        if all(
            iou(item["box"], other["box"]) <= iou_threshold
            for other in kept
        ):
            kept.append(item)

    return kept


def association_score(item_box, person_box, kind):
    px1, py1, px2, py2 = person_box
    width = max(1, px2 - px1)
    height = max(1, py2 - py1)

    cx, cy = center(item_box)
    xr = (cx - px1) / width
    yr = (cy - py1) / height

    if not -0.08 <= xr <= 1.08:
        return None

    overlap = overlap_of_item(item_box, person_box)
    if overlap < 0.25:
        return None

    if kind == "helmet":
        if not -0.08 <= yr <= 0.38:
            return None
        target_y = 0.16
    else:
        if not 0.12 <= yr <= 0.80:
            return None
        target_y = 0.48

    return (
        abs(xr - 0.5)
        + 1.5 * abs(yr - target_y)
        + 0.8 * (1 - overlap)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path gambar asli")
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    if not 0 < args.iou < 1:
        sys.exit("--iou harus lebih dari 0 dan kurang dari 1.")

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    source = source.resolve()

    if not source.is_file():
        sys.exit(f"Gambar tidak ditemukan: {source}")
    if not MODEL_PATH.is_file():
        sys.exit(f"Model tidak ditemukan: {MODEL_PATH}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(
        "GPU:",
        torch.cuda.get_device_name(0) if device == 0 else "CPU",
    )

    model = YOLO(str(MODEL_PATH))
    names = model.names
    class_ids = [
        idx for idx, name in names.items()
        if str(name).lower() in MIN_CONF
    ]

    if not class_ids:
        sys.exit("Kelas person/helmet/safety-vest tidak ditemukan.")

    results = model.predict(
        source=str(source),
        device=device,
        imgsz=640,
        conf=min(MIN_CONF.values()),
        iou=0.50,
        classes=class_ids,
        verbose=False,
    )

    result = results[0]
    image = cv2.imread(str(source))
    if image is None:
        sys.exit("Gambar tidak dapat dibaca.")

    detections = {name: [] for name in MIN_CONF}

    if result.boxes is not None:
        for box in result.boxes:
            label = str(names[int(box.cls.item())]).lower()
            confidence = float(box.conf.item())

            if confidence < MIN_CONF[label]:
                continue

            detections[label].append({
                "box": coords(box),
                "conf": confidence,
            })

    for label in detections:
        detections[label] = deduplicate(
            detections[label], args.iou
        )

    persons = []
    for item in detections["person"]:
        persons.append({
            **item,
            "helmet": None,
            "safety-vest": None,
        })

    # Match each PPE box to at most one person.
    for kind in ("helmet", "safety-vest"):
        candidates = []

        for item in detections[kind]:
            for person_idx, person in enumerate(persons):
                score = association_score(
                    item["box"], person["box"], kind
                )
                if score is not None:
                    candidates.append((score, person_idx, item))

        used_people = set()
        used_items = set()

        for score, person_idx, item in sorted(
            candidates, key=lambda candidate: candidate[0]
        ):
            item_key = tuple(item["box"])
            if person_idx in used_people or item_key in used_items:
                continue

            persons[person_idx][kind] = item
            used_people.add(person_idx)
            used_items.add(item_key)

    for idx, person in enumerate(persons, start=1):
        x1, y1, x2, y2 = person["box"]
        has_h = person["helmet"] is not None
        has_v = person["safety-vest"] is not None

        color = (
            COLORS["person"]
            if has_h and has_v
            else COLORS["uncertain"]
        )

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        h_text = "FOUND" if has_h else "CHECK"
        v_text = "FOUND" if has_v else "CHECK"
        cv2.putText(
            image,
            f"P{idx} H:{h_text} V:{v_text}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )

        for kind in ("helmet", "safety-vest"):
            item = person[kind]
            if item is not None:
                ax1, ay1, ax2, ay2 = item["box"]
                cv2.rectangle(
                    image, (ax1, ay1), (ax2, ay2),
                    COLORS[kind], 2
                )

    print("\nDeteksi setelah threshold dan deduplikasi:")
    for label, items in detections.items():
        print(f"- {label}: {len(items)}")
        for item in items:
            print(f"    confidence={item['conf']:.2f}, bbox={item['box']}")

    print("\nAsosiasi per pekerja:")
    for idx, person in enumerate(persons, start=1):
        helmet = "FOUND" if person["helmet"] is not None else "CHECK"
        vest = "FOUND" if person["safety-vest"] is not None else "CHECK"
        print(f"- P{idx}: H:{helmet}, V:{vest}")

    out_dir = ROOT / "outputs" / "ppe_association"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.stem}_association.jpg"

    if not cv2.imwrite(str(out_path), image):
        sys.exit(f"Gagal menyimpan hasil: {out_path}")

    print(f"\nHasil gambar: {out_path}")
    print(
        "CHECK berarti APD belum berhasil diasosiasikan. "
        "Ini bukan bukti pekerja melanggar SOP."
    )


if __name__ == "__main__":
    main()
