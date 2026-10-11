from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from PySide6.QtCore import Qt, QThread, Signal, QRectF, QPointF
from PySide6.QtGui import QFont, QPixmap, QPainter, QPen, QBrush, QColor, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "yolo8n.pt"
OUTPUT_DIR = APP_DIR / "outputs" / "ppe_gui"
ZONES_PATH = APP_DIR / "zones.json"

# Provisional thresholds based on the initial tests; validate against a labeled test set.
CLASS_THRESHOLDS = {"person": 0.35, "helmet": 0.35, "safety-vest": 0.15}
COLORS = {
    "person": (65, 185, 85),
    "helmet": (255, 190, 25),
    "safety-vest": (205, 60, 205),
    "uncertain": (0, 165, 255),
    "zone": (40, 210, 250),
    "white": (255, 255, 255),
}


def load_zones() -> list[dict]:
    if not ZONES_PATH.is_file():
        return []
    try:
        data = json.loads(ZONES_PATH.read_text(encoding="utf-8"))
        zones = data.get("zones", []) if isinstance(data, dict) else data
        if not isinstance(zones, list):
            return []
        valid = []
        for zone in zones:
            if not isinstance(zone, dict):
                continue
            points = zone.get("points", [])
            required = zone.get("required_classes", [])
            if len(points) < 3 or not isinstance(required, list):
                continue
            if not all(
                isinstance(point, (list, tuple)) and len(point) == 2
                and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in point)
                for point in points
            ):
                continue
            valid.append({
                "name": str(zone.get("name", f"Zona {len(valid) + 1}")),
                "points": [[float(p[0]), float(p[1])] for p in points],
                "required_classes": [
                    str(c) for c in required if c in ("helmet", "safety-vest")
                ],
            })
        return valid
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return []


def save_zones(zones: list[dict]) -> None:
    payload = {"version": 1, "coordinate_system": "normalized_0_to_1", "zones": zones}
    ZONES_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def box_coordinates(box) -> tuple[int, int, int, int]:
    return tuple(int(round(v)) for v in box.xyxy[0].tolist())


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, area_a + area_b - inter)


def item_overlap(item_box: tuple[int, int, int, int], person_box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = item_box
    px1, py1, px2, py2 = person_box
    ix1, iy1, ix2, iy2 = max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (x2 - x1) * (y2 - y1))
    return inter / area


def score_association(item_box: tuple[int, int, int, int], person_box: tuple[int, int, int, int], kind: str) -> float | None:
    px1, py1, px2, py2 = person_box
    width, height = max(1, px2 - px1), max(1, py2 - py1)
    cx, cy = box_center(item_box)
    xr, yr = (cx - px1) / width, (cy - py1) / height
    if not -0.08 <= xr <= 1.08:
        return None
    overlap = item_overlap(item_box, person_box)
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
    return abs(xr - 0.5) + 1.5 * abs(yr - target_y) + 0.8 * (1.0 - overlap)


def deduplicate(detections: list[dict], threshold: float = 0.70) -> list[dict]:
    kept: list[dict] = []
    for item in sorted(detections, key=lambda obj: obj["conf"], reverse=True):
        if all(iou(item["box"], other["box"]) <= threshold for other in kept):
            kept.append(item)
    return kept


def draw_label(image, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.45) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    height, width = image.shape[:2]
    x = max(0, min(int(x), max(0, width - tw - 8)))
    top = max(0, min(int(y - th - baseline - 6), max(0, height - th - baseline - 6)))
    cv2.rectangle(image, (x, top), (min(width - 1, x + tw + 8), top + th + baseline + 6), color, -1)
    cv2.putText(image, text, (x + 4, top + th + 1), font, scale, COLORS["white"], thickness, cv2.LINE_AA)


def build_associations(detections: dict[str, list[dict]]) -> list[dict]:
    people = [{**item, "helmet": None, "safety-vest": None} for item in detections["person"]]
    # This is a geometric heuristic; it does not track identity or guarantee correct matching.
    for kind in ("helmet", "safety-vest"):
        candidates: list[tuple[float, int, dict]] = []
        for item in detections[kind]:
            for person_index, person in enumerate(people):
                score = score_association(item["box"], person["box"], kind)
                if score is not None:
                    candidates.append((score, person_index, item))
        used_people: set[int] = set()
        used_items: set[tuple[int, int, int, int]] = set()
        for score, person_index, item in sorted(candidates, key=lambda c: c[0]):
            key = tuple(item["box"])
            if person_index in used_people or key in used_items:
                continue
            people[person_index][kind] = item
            used_people.add(person_index)
            used_items.add(key)
    return people


def polygon_for_zone(zone: dict, width: int, height: int) -> list[tuple[int, int]]:
    return [(int(round(x * width)), int(round(y * height))) for x, y in zone["points"]]


def point_inside_zone(point: tuple[float, float], zone: dict, width: int, height: int) -> bool:
    polygon = polygon_for_zone(zone, width, height)
    if len(polygon) < 3:
        return False
    poly = __import__("numpy").array(polygon, dtype="int32")
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def render_detection_view(image, detections: dict[str, list[dict]]) -> None:
    short = {"person": "P", "helmet": "H", "safety-vest": "V"}
    for label in ("person", "helmet", "safety-vest"):
        for item in detections[label]:
            x1, y1, x2, y2 = item["box"]
            color = COLORS[label]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            draw_label(image, f"{short[label]} {item['conf']:.2f}", x1, max(y1, y2 - 2), color, 0.42)


def render_association_view(image, people: list[dict]) -> None:
    for index, person in enumerate(people, start=1):
        x1, y1, x2, y2 = person["box"]
        has_h, has_v = person["helmet"] is not None, person["safety-vest"] is not None
        color = COLORS["person"] if has_h and has_v else COLORS["uncertain"]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_label(image, f"P{index} H:{'F' if has_h else '?'} V:{'F' if has_v else '?'}", x1, y1, color, 0.40)
        for kind in ("helmet", "safety-vest"):
            item = person[kind]
            if item is not None:
                ax1, ay1, ax2, ay2 = item["box"]
                cv2.rectangle(image, (ax1, ay1), (ax2, ay2), COLORS[kind], 2)


def render_zone_view(image, detections: dict[str, list[dict]], people: list[dict], zones: list[dict]) -> list[str]:
    height, width = image.shape[:2]
    report_lines: list[str] = []
    for idx, zone in enumerate(zones, start=1):
        pts = polygon_for_zone(zone, width, height)
        color = (40, 210, 250) if idx % 2 else (255, 170, 60)
        polygon = __import__("numpy").array(pts, dtype="int32")
        cv2.polylines(image, [polygon], True, color, 3, cv2.LINE_AA)
        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, 0.09, image, 0.91, 0, image)
        required_text = ", ".join("Helmet" if c == "helmet" else "Vest" for c in zone["required_classes"])
        label_y = max(26, min(p[1] for p in pts) + 22)
        draw_label(image, f"ZONE {zone['name']} | {required_text or 'no PPE rule'}", min(p[0] for p in pts), label_y, color, 0.48)

    # Draw associated PPE boxes and worker status. Worker location uses bottom-centre point.
    for idx, person in enumerate(people, start=1):
        x1, y1, x2, y2 = person["box"]
        foot = ((x1 + x2) / 2, max(y1, y2 - 1))
        applicable = [zone for zone in zones if point_inside_zone(foot, zone, width, height)]
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS["person"], 2)
        if not applicable:
            draw_label(image, f"P{idx} OUTSIDE ZONE", x1, y1, (100, 100, 100), 0.40)
            continue
        fragments = []
        for zone in applicable:
            status_bits = []
            missing = []
            for kind in zone["required_classes"]:
                found = person.get(kind) is not None
                short = "H" if kind == "helmet" else "V"
                status_bits.append(f"{short}:{'OK' if found else 'CHECK'}")
                if not found:
                    missing.append(short)
            fragments.append(f"{zone['name']} " + " ".join(status_bits))
            report_lines.append(
                f"P{idx} in '{zone['name']}': "
                + (", ".join(status_bits) if status_bits else "no PPE required")
                + (" — review needed; absence of detection is not proof of violation" if missing else " — required PPE detections found")
            )
        label = f"P{idx} " + " | ".join(fragments)
        person_color = COLORS["uncertain"] if "CHECK" in label else COLORS["person"]
        draw_label(image, label, x1, y1, person_color, 0.38)
        for kind in ("helmet", "safety-vest"):
            item = person.get(kind)
            if item is not None:
                ax1, ay1, ax2, ay2 = item["box"]
                cv2.rectangle(image, (ax1, ay1), (ax2, ay2), COLORS[kind], 2)

    return report_lines


class ZoneCanvas(QWidget):
    """Image canvas; clicks add normalized polygon vertices for the selected scene."""
    def __init__(self, image_path: Path, zones: list[dict]) -> None:
        super().__init__()
        self.pixmap = QPixmap(str(image_path))
        self.zones = zones
        self.current_points: list[list[float]] = []
        self.setMinimumSize(560, 330)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def image_rect(self) -> QRectF:
        if self.pixmap.isNull():
            return QRectF()
        area = self.rect().adjusted(10, 10, -10, -10)
        iw, ih = self.pixmap.width(), self.pixmap.height()
        scale = min(area.width() / max(1, iw), area.height() / max(1, ih))
        w, h = iw * scale, ih * scale
        return QRectF(area.center().x() - w / 2, area.center().y() - h / 2, w, h)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b1220"))
        rect = self.image_rect()
        if not self.pixmap.isNull():
            painter.drawPixmap(rect, self.pixmap, QRectF(self.pixmap.rect()))
        palette = [QColor("#22d3ee"), QColor("#fb923c"), QColor("#a78bfa"), QColor("#34d399")]
        for i, zone in enumerate(self.zones):
            color = palette[i % len(palette)]
            points = [QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height()) for x, y in zone["points"]]
            if len(points) >= 3:
                painter.setPen(QPen(color, 3))
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 35)))
                painter.drawPolygon(QPolygonF(points))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                if points:
                    painter.drawText(points[0] + QPointF(4, -5), zone["name"])
        if self.current_points:
            points = [QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height()) for x, y in self.current_points]
            painter.setPen(QPen(QColor("#facc15"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor("#facc15")))
            for point in points:
                painter.drawEllipse(point, 4, 4)
            if len(points) > 1:
                painter.drawPolyline(QPolygonF(points))
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self.pixmap.isNull():
            return
        rect = self.image_rect()
        pos = event.position()
        if not rect.contains(pos):
            return
        x = min(1.0, max(0.0, (pos.x() - rect.x()) / rect.width()))
        y = min(1.0, max(0.0, (pos.y() - rect.y()) / rect.height()))
        self.current_points.append([x, y])
        self.update()

    def clear_current(self) -> None:
        self.current_points.clear()
        self.update()


class ZoneEditorDialog(QDialog):
    def __init__(self, image_path: Path, initial_zones: list[dict], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Safety Zone Editor")
        self.resize(1100, 720)
        self.zones = [dict(z, points=[list(p) for p in z["points"]], required_classes=list(z["required_classes"])) for z in initial_zones]
        root = QVBoxLayout(self)
        heading = QLabel("Editor Safety Zone")
        heading.setObjectName("heading")
        instructions = QLabel(
            "Klik minimal 3 titik pada gambar untuk membentuk poligon. Zona hanya cocok untuk sudut pandang kamera/scene yang sama."
        )
        instructions.setWordWrap(True)
        instructions.setObjectName("subtitle")
        root.addWidget(heading)
        root.addWidget(instructions)

        body = QHBoxLayout()
        self.canvas = ZoneCanvas(image_path, self.zones)
        body.addWidget(self.canvas, 3)
        side = QVBoxLayout()
        side.addWidget(QLabel("Zona tersimpan"))
        self.zone_list = QListWidget()
        side.addWidget(self.zone_list, 1)
        form = QFormLayout()
        self.name_edit = QLineEdit(f"Zona {len(self.zones) + 1}")
        form.addRow("Nama zona:", self.name_edit)
        side.addLayout(form)
        side.addWidget(QLabel("APD wajib di zona baru:"))
        self.helmet_check = QCheckBox("Helm keselamatan")
        self.vest_check = QCheckBox("Rompi keselamatan")
        self.helmet_check.setChecked(True)
        self.vest_check.setChecked(True)
        side.addWidget(self.helmet_check)
        side.addWidget(self.vest_check)
        self.add_zone_button = QPushButton("Tambahkan Zona dari Titik")
        self.clear_points_button = QPushButton("Hapus Titik Baru")
        self.delete_zone_button = QPushButton("Hapus Zona Terpilih")
        self.save_button = QPushButton("Simpan Zona & Tutup")
        self.add_zone_button.setObjectName("runButton")
        side.addWidget(self.add_zone_button)
        side.addWidget(self.clear_points_button)
        side.addWidget(self.delete_zone_button)
        side.addStretch(1)
        side.addWidget(self.save_button)
        body.addLayout(side, 1)
        root.addLayout(body, 1)

        self.add_zone_button.clicked.connect(self.add_zone)
        self.clear_points_button.clicked.connect(self.canvas.clear_current)
        self.delete_zone_button.clicked.connect(self.delete_zone)
        self.save_button.clicked.connect(self.save_and_close)
        self._refresh_zone_list()

    def _refresh_zone_list(self) -> None:
        self.zone_list.clear()
        for zone in self.zones:
            required = ", ".join(zone["required_classes"]) or "tidak ada"
            self.zone_list.addItem(QListWidgetItem(f"{zone['name']} — wajib: {required}"))
        self.canvas.zones = self.zones
        self.canvas.update()

    def add_zone(self) -> None:
        points = list(self.canvas.current_points)
        if len(points) < 3:
            QMessageBox.information(self, "Titik belum cukup", "Klik minimal 3 titik pada gambar untuk membentuk zona.")
            return
        required = []
        if self.helmet_check.isChecked():
            required.append("helmet")
        if self.vest_check.isChecked():
            required.append("safety-vest")
        if not required:
            QMessageBox.information(self, "Pilih APD", "Pilih minimal satu jenis APD wajib.")
            return
        name = self.name_edit.text().strip() or f"Zona {len(self.zones) + 1}"
        if any(z["name"].casefold() == name.casefold() for z in self.zones):
            QMessageBox.information(self, "Nama zona sudah ada", "Gunakan nama zona yang berbeda.")
            return
        self.zones.append({"name": name, "points": points, "required_classes": required})
        self.canvas.clear_current()
        self.name_edit.setText(f"Zona {len(self.zones) + 1}")
        self._refresh_zone_list()

    def delete_zone(self) -> None:
        index = self.zone_list.currentRow()
        if index < 0:
            return
        del self.zones[index]
        self._refresh_zone_list()

    def save_and_close(self) -> None:
        try:
            save_zones(self.zones)
        except OSError as exc:
            QMessageBox.critical(self, "Gagal menyimpan", f"Tidak dapat menyimpan konfigurasi zona:\n{exc}")
            return
        self.accept()


class AnalysisWorker(QThread):
    completed = Signal(str, str)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, image_path: Path, mode: str, zones: list[dict]) -> None:
        super().__init__()
        self.image_path = image_path
        self.mode = mode
        self.zones = zones

    def run(self) -> None:
        try:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"Model tidak ditemukan:\n{MODEL_PATH}\nLetakkan yolo8n.pt di folder yang sama dengan app_v3.py.")
            if not self.image_path.is_file():
                raise FileNotFoundError(f"Gambar tidak ditemukan:\n{self.image_path}")
            device = 0 if torch.cuda.is_available() else "cpu"
            device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
            self.status.emit(f"Menjalankan analisis pada {device_name}...")
            model = YOLO(str(MODEL_PATH))
            names = model.names
            class_ids = [idx for idx, name in names.items() if str(name).lower() in CLASS_THRESHOLDS]
            if not class_ids:
                raise RuntimeError("Model tidak memiliki kelas person, helmet, atau safety-vest.")
            result = model.predict(
                source=str(self.image_path), device=device, imgsz=640,
                conf=min(CLASS_THRESHOLDS.values()), iou=0.50,
                classes=class_ids, verbose=False,
            )[0]
            image = cv2.imread(str(self.image_path))
            if image is None:
                raise RuntimeError("OpenCV tidak dapat membaca gambar ini.")
            detections: dict[str, list[dict]] = {key: [] for key in CLASS_THRESHOLDS}
            if result.boxes is not None:
                for box in result.boxes:
                    label = str(names[int(box.cls.item())]).lower()
                    confidence = float(box.conf.item())
                    if label not in CLASS_THRESHOLDS or confidence < CLASS_THRESHOLDS[label]:
                        continue
                    detections[label].append({"box": box_coordinates(box), "conf": confidence})
            for label in detections:
                detections[label] = deduplicate(detections[label], 0.70)
            people = build_associations(detections)
            zone_lines: list[str] = []
            if self.mode == "Asosiasi APD":
                render_association_view(image, people)
            elif self.mode == "Analisis Safety Zone":
                zone_lines = render_zone_view(image, detections, people, self.zones)
            else:
                render_detection_view(image, detections)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            safe_mode = {"Deteksi Objek": "detection", "Asosiasi APD": "association", "Analisis Safety Zone": "zones"}.get(self.mode, "analysis")
            output_path = OUTPUT_DIR / f"{self.image_path.stem}_{safe_mode}.jpg"
            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(f"Gagal menyimpan hasil ke:\n{output_path}")

            counts = Counter({label: len(items) for label, items in detections.items()})
            lines = [
                "AI SAFETY MONITOR — HASIL ANALISIS",
                f"Model: {MODEL_PATH.name}", f"Device: {device_name}",
                f"Gambar: {self.image_path.name}", f"Mode: {self.mode}", "",
                "Ambang confidence sementara:", "  person: 0.35", "  helmet: 0.35", "  safety-vest: 0.15", "",
                "Ringkasan deteksi:", f"  Person: {counts['person']}",
                f"  Helmet: {counts['helmet']}", f"  Safety vest: {counts['safety-vest']}",
            ]
            if self.mode == "Asosiasi APD":
                lines.extend(["", "Asosiasi berdasarkan posisi kotak:"])
                for idx, person in enumerate(people, 1):
                    lines.append(f"  P{idx}: Helmet={'FOUND' if person['helmet'] is not None else 'CHECK'}; Vest={'FOUND' if person['safety-vest'] is not None else 'CHECK'}")
            elif self.mode == "Analisis Safety Zone":
                lines.extend(["", f"Zona dikonfigurasi: {len(self.zones)}"])
                if not self.zones:
                    lines.append("Belum ada zona. Klik Editor Safety Zone untuk menggambar poligon dan menentukan APD wajib.")
                else:
                    for zone in self.zones:
                        required = ", ".join(zone["required_classes"]) or "tidak ada"
                        lines.append(f"  {zone['name']}: APD wajib = {required}")
                lines.extend(["", "Hasil pekerja dalam zona:"] + (zone_lines or ["  Tidak ada pekerja yang terdeteksi berada dalam zona yang dipetakan."]))
            lines.extend(["", "Peringatan: asosiasi adalah heuristik geometris sederhana.", "CHECK/belum terdeteksi bukan bukti pelanggaran SOP.", "Zona dinyatakan dalam koordinat relatif dan hanya sesuai untuk sudut pandang kamera/scene yang sama.", f"Output: {output_path}"])
            self.completed.emit(str(output_path), "\n".join(lines))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ImagePreview(QLabel):
    def __init__(self, placeholder: str) -> None:
        super().__init__(placeholder)
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("imagePreview")
        self.setWordWrap(True)

    def set_image(self, image_path: str | Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._pixmap = None
            self.setText(f"Gagal membuka gambar:\n{image_path}")
            return
        self._pixmap = pixmap
        self._update_scaled_pixmap()

    def clear_image(self, placeholder: str) -> None:
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(placeholder)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(self.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Safety Monitor")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)
        self.selected_image: Path | None = None
        self.worker: AnalysisWorker | None = None
        self.zones = load_zones()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)
        heading = QLabel("AI Safety Monitor")
        heading.setObjectName("heading")
        subtitle = QLabel("Prototype desktop · PPE detection, association, and configurable safety zones")
        subtitle.setObjectName("subtitle")
        layout.addWidget(heading)
        layout.addWidget(subtitle)

        controls = QHBoxLayout()
        self.choose_button = QPushButton("Pilih Gambar")
        self.choose_button.setObjectName("primaryButton")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Deteksi Objek", "Asosiasi APD", "Analisis Safety Zone"])
        self.mode_combo.setMinimumWidth(190)
        self.zone_button = QPushButton("Editor Safety Zone")
        self.zone_button.setObjectName("zoneButton")
        self.run_button = QPushButton("Jalankan Analisis")
        self.run_button.setObjectName("runButton")
        self.run_button.setEnabled(False)
        self.status_label = QLabel("Pilih gambar untuk memulai.")
        self.status_label.setObjectName("status")
        controls.addWidget(self.choose_button)
        controls.addWidget(QLabel("Mode:"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(self.zone_button)
        controls.addWidget(self.run_button)
        controls.addWidget(self.status_label, 1)
        layout.addLayout(controls)

        previews = QHBoxLayout()
        previews.setSpacing(14)
        original_column, result_column = QVBoxLayout(), QVBoxLayout()
        original_title, result_title = QLabel("Gambar Asli"), QLabel("Hasil Analisis")
        original_title.setObjectName("sectionTitle")
        result_title.setObjectName("sectionTitle")
        self.original_preview = ImagePreview("Belum ada gambar dipilih")
        self.result_preview = ImagePreview("Hasil analisis akan muncul di sini")
        original_column.addWidget(original_title)
        original_column.addWidget(self.original_preview, 1)
        result_column.addWidget(result_title)
        result_column.addWidget(self.result_preview, 1)
        previews.addLayout(original_column, 1)
        previews.addLayout(result_column, 1)
        layout.addLayout(previews, 1)

        report_title = QLabel("Ringkasan")
        report_title.setObjectName("sectionTitle")
        layout.addWidget(report_title)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(150)
        self.report.setMaximumHeight(220)
        self.report.setPlainText("Model: yolo8n.pt\nGPU digunakan jika CUDA tersedia.\n\nPilih gambar lalu jalankan analisis.")
        layout.addWidget(self.report)
        disclaimer = QLabel("CHECK/belum terdeteksi bukan bukti pelanggaran SOP. Safety zone cocok untuk sudut pandang kamera yang tetap.")
        disclaimer.setObjectName("disclaimer")
        layout.addWidget(disclaimer)

        self.setCentralWidget(root)
        self.choose_button.clicked.connect(self.choose_image)
        self.zone_button.clicked.connect(self.open_zone_editor)
        self.run_button.clicked.connect(self.run_analysis)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)

    def on_mode_changed(self, mode: str) -> None:
        self.result_preview.clear_image(f"Tekan Jalankan Analisis untuk mode: {mode}")

    def choose_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih gambar untuk dianalisis", str(APP_DIR), "Image files (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not file_path:
            return
        self.selected_image = Path(file_path)
        self.original_preview.set_image(self.selected_image)
        self.result_preview.clear_image("Tekan Jalankan Analisis")
        self.run_button.setEnabled(True)
        self.status_label.setText(f"Dipilih: {self.selected_image.name}")
        self.report.setPlainText(f"Gambar: {self.selected_image}\nMode: {self.mode_combo.currentText()}\n\nSiap menjalankan analisis.")

    def open_zone_editor(self) -> None:
        if self.selected_image is None:
            QMessageBox.information(self, "Pilih gambar", "Pilih gambar yang mewakili sudut pandang kamera terlebih dahulu.")
            return
        dialog = ZoneEditorDialog(self.selected_image, self.zones, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.zones = load_zones()
            self.status_label.setText(f"Konfigurasi tersimpan: {len(self.zones)} zona")
            self.report.setPlainText(f"Zona tersimpan di:\n{ZONES_PATH}\nJumlah zona: {len(self.zones)}\n\nZona memakai koordinat relatif dan sebaiknya digunakan hanya dengan kamera/scene yang sama.")

    def run_analysis(self) -> None:
        if self.selected_image is None:
            QMessageBox.information(self, "Pilih gambar", "Pilih gambar terlebih dahulu.")
            return
        mode = self.mode_combo.currentText()
        if mode == "Analisis Safety Zone" and not self.zones:
            answer = QMessageBox.question(self, "Belum ada zona", "Belum ada Safety Zone. Mau membuka editor zona sekarang?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if answer == QMessageBox.StandardButton.Yes:
                self.open_zone_editor()
            if not self.zones:
                return
        if self.worker is not None and self.worker.isRunning():
            return
        self.run_button.setEnabled(False)
        self.choose_button.setEnabled(False)
        self.zone_button.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.status_label.setText("Menjalankan model...")
        self.report.setPlainText("Analisis sedang berjalan. Tunggu hingga selesai...")
        self.worker = AnalysisWorker(self.selected_image, mode, self.zones)
        self.worker.status.connect(self.status_label.setText)
        self.worker.completed.connect(self.on_analysis_completed)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_analysis_completed(self, output_path: str, report: str) -> None:
        self.result_preview.set_image(output_path)
        self.report.setPlainText(report)
        self.status_label.setText(f"Selesai: {output_path}")

    def on_analysis_failed(self, message: str) -> None:
        self.status_label.setText("Analisis gagal.")
        self.report.setPlainText(message)
        QMessageBox.critical(self, "Kesalahan analisis", message)

    def on_worker_finished(self) -> None:
        self.choose_button.setEnabled(True)
        self.zone_button.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.run_button.setEnabled(self.selected_image is not None)
        self.worker = None


def apply_style(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(
        "QMainWindow, QWidget { background: #111827; color: #e5e7eb; }"
        "QLabel#heading { font-size: 27px; font-weight: 700; color: #f9fafb; }"
        "QLabel#subtitle { color: #9ca3af; margin-bottom: 4px; }"
        "QLabel#sectionTitle { font-size: 14px; font-weight: 650; color: #d1d5db; }"
        "QLabel#status { color: #93c5fd; padding-left: 8px; }"
        "QLabel#imagePreview { background: #0b1220; border: 1px solid #374151; border-radius: 8px; color: #9ca3af; padding: 8px; }"
        "QLabel#disclaimer { color: #9ca3af; font-size: 9pt; }"
        "QPushButton, QComboBox { background: #374151; color: #f9fafb; border: 1px solid #4b5563; padding: 8px 11px; border-radius: 7px; }"
        "QPushButton:hover { background: #4b5563; }"
        "QPushButton:disabled { background: #1f2937; color: #6b7280; }"
        "QPushButton#primaryButton { background: #2563eb; border-color: #2563eb; font-weight: 600; }"
        "QPushButton#runButton { background: #047857; border-color: #047857; font-weight: 600; }"
        "QPushButton#zoneButton { background: #7c3aed; border-color: #7c3aed; font-weight: 600; }"
        "QComboBox QAbstractItemView { background: #1f2937; color: #f9fafb; selection-background-color: #2563eb; }"
        "QTextEdit, QListWidget, QLineEdit { background: #0b1220; border: 1px solid #374151; border-radius: 8px; padding: 7px; color: #d1d5db; }"
        "QCheckBox { padding: 4px; }"
    )


def main() -> int:
    app = QApplication(sys.argv)
    apply_style(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
