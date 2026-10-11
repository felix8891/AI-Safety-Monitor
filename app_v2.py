from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
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

# Start with class-specific thresholds based on the tests performed so far.
# These are provisional settings, not validated operating thresholds.
CLASS_THRESHOLDS = {
    "person": 0.35,
    "helmet": 0.35,
    "safety-vest": 0.15,
}

# OpenCV colors are BGR.
COLORS = {
    "person": (65, 185, 85),
    "helmet": (255, 190, 25),
    "safety-vest": (205, 60, 205),
    "uncertain": (0, 165, 255),
    "white": (255, 255, 255),
}


def box_coordinates(box) -> tuple[int, int, int, int]:
    return tuple(int(round(v)) for v in box.xyxy[0].tolist())


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(1, area_a + area_b - intersection)


def item_overlap(item_box: tuple[int, int, int, int], person_box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = item_box
    px1, py1, px2, py2 = person_box
    ix1, iy1 = max(x1, px1), max(y1, py1)
    ix2, iy2 = min(x2, px2), min(y2, py2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    item_area = max(1, (x2 - x1) * (y2 - y1))
    return intersection / item_area


def score_association(item_box: tuple[int, int, int, int], person_box: tuple[int, int, int, int], kind: str) -> float | None:
    px1, py1, px2, py2 = person_box
    width = max(1, px2 - px1)
    height = max(1, py2 - py1)
    cx, cy = box_center(item_box)
    x_rel = (cx - px1) / width
    y_rel = (cy - py1) / height

    if not -0.08 <= x_rel <= 1.08:
        return None

    overlap = item_overlap(item_box, person_box)
    if overlap < 0.25:
        return None

    if kind == "helmet":
        if not -0.08 <= y_rel <= 0.38:
            return None
        target_y = 0.16
    else:
        if not 0.12 <= y_rel <= 0.80:
            return None
        target_y = 0.48

    return abs(x_rel - 0.5) + 1.5 * abs(y_rel - target_y) + 0.8 * (1.0 - overlap)


def deduplicate(detections: list[dict], threshold: float = 0.70) -> list[dict]:
    """Conservative class-wise duplicate suppression after model NMS."""
    kept: list[dict] = []
    for item in sorted(detections, key=lambda obj: obj["conf"], reverse=True):
        if all(iou(item["box"], other["box"]) <= threshold for other in kept):
            kept.append(item)
    return kept


def draw_label(image, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.48) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1 if scale < 0.55 else 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    height, width = image.shape[:2]
    x = max(0, min(x, max(0, width - text_w - 8)))
    top = max(0, min(y - text_h - baseline - 6, max(0, height - text_h - baseline - 6)))
    cv2.rectangle(image, (x, top), (min(width - 1, x + text_w + 8), top + text_h + baseline + 6), color, -1)
    cv2.putText(image, text, (x + 4, top + text_h + 1), font, scale, COLORS["white"], thickness, cv2.LINE_AA)


def build_associations(detections: dict[str, list[dict]]) -> list[dict]:
    people = [
        {**item, "helmet": None, "safety-vest": None}
        for item in detections["person"]
    ]

    # Match each PPE detection to at most one person and each person to at most
    # one PPE detection of a given class. This is a geometric prototype only.
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


def render_detection_view(image, detections: dict[str, list[dict]]) -> None:
    """Draw compact object labels; dense scenes can still naturally overlap."""
    for label in ("person", "helmet", "safety-vest"):
        for item in detections[label]:
            x1, y1, x2, y2 = item["box"]
            color = COLORS[label]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # Short labels take less space than verbose class names.
            short_name = {"person": "P", "helmet": "H", "safety-vest": "V"}[label]
            draw_label(image, f"{short_name} {item['conf']:.2f}", x1, y1, color, 0.44)


def render_association_view(image, people: list[dict]) -> None:
    for index, person in enumerate(people, start=1):
        x1, y1, x2, y2 = person["box"]
        has_helmet = person["helmet"] is not None
        has_vest = person["safety-vest"] is not None
        color = COLORS["person"] if has_helmet and has_vest else COLORS["uncertain"]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        h_text = "FOUND" if has_helmet else "CHECK"
        v_text = "FOUND" if has_vest else "CHECK"
        draw_label(image, f"P{index} H:{h_text} V:{v_text}", x1, y1, color, 0.45)
        for kind in ("helmet", "safety-vest"):
            item = person[kind]
            if item is None:
                continue
            ax1, ay1, ax2, ay2 = item["box"]
            cv2.rectangle(image, (ax1, ay1), (ax2, ay2), COLORS[kind], 2)
            short_name = "H" if kind == "helmet" else "V"
            draw_label(image, f"{short_name} {item['conf']:.2f}", ax1, ay1, COLORS[kind], 0.43)


class DetectionWorker(QThread):
    completed = Signal(str, str)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, image_path: Path, mode: str) -> None:
        super().__init__()
        self.image_path = image_path
        self.mode = mode

    def run(self) -> None:
        try:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(
                    f"Model tidak ditemukan:\n{MODEL_PATH}\n\n"
                    "Pastikan yolo8n.pt berada di folder yang sama dengan app.py."
                )
            if not self.image_path.is_file():
                raise FileNotFoundError(f"Gambar tidak ditemukan:\n{self.image_path}")

            device = 0 if torch.cuda.is_available() else "cpu"
            device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
            self.status.emit(f"Menjalankan deteksi pada {device_name}...")

            model = YOLO(str(MODEL_PATH))
            names = model.names
            class_ids = [
                int(class_id)
                for class_id, name in names.items()
                if str(name).lower() in CLASS_THRESHOLDS
            ]
            if not class_ids:
                raise RuntimeError("Model tidak memiliki kelas person, helmet, atau safety-vest.")

            result = model.predict(
                source=str(self.image_path),
                device=device,
                imgsz=640,
                conf=min(CLASS_THRESHOLDS.values()),
                iou=0.50,
                classes=class_ids,
                verbose=False,
            )[0]

            image = cv2.imread(str(self.image_path))
            if image is None:
                raise RuntimeError("OpenCV tidak dapat membaca gambar ini.")

            detections: dict[str, list[dict]] = {key: [] for key in CLASS_THRESHOLDS}
            if result.boxes is not None:
                for box in result.boxes:
                    label = str(names[int(box.cls.item())]).lower()
                    conf = float(box.conf.item())
                    if label not in CLASS_THRESHOLDS or conf < CLASS_THRESHOLDS[label]:
                        continue
                    detections[label].append({"box": box_coordinates(box), "conf": conf})

            # The model already applies NMS; this conservative extra pass is only
            # for highly overlapping same-class boxes in the display pipeline.
            for label in detections:
                detections[label] = deduplicate(detections[label], 0.70)

            people = build_associations(detections)
            if self.mode == "Asosiasi APD":
                render_association_view(image, people)
            else:
                render_detection_view(image, detections)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            safe_mode = "association" if self.mode == "Asosiasi APD" else "detection"
            output_path = OUTPUT_DIR / f"{self.image_path.stem}_{safe_mode}.jpg"
            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(f"Gagal menyimpan hasil ke:\n{output_path}")

            counts = Counter({label: len(items) for label, items in detections.items()})
            lines = [
                "AI SAFETY MONITOR — HASIL ANALISIS",
                f"Model: {MODEL_PATH.name}",
                f"Device: {device_name}",
                f"Gambar: {self.image_path.name}",
                f"Mode: {self.mode}",
                "",
                "Ambang confidence sementara:",
                "  person: 0.35",
                "  helmet: 0.35",
                "  safety-vest: 0.15",
                "",
                "Ringkasan deteksi:",
                f"  Person: {counts['person']}",
                f"  Helmet: {counts['helmet']}",
                f"  Safety vest: {counts['safety-vest']}",
            ]
            if self.mode == "Asosiasi APD":
                lines.extend(["", "Asosiasi berdasarkan posisi kotak:"])
                for index, person in enumerate(people, start=1):
                    helmet = "FOUND" if person["helmet"] is not None else "CHECK"
                    vest = "FOUND" if person["safety-vest"] is not None else "CHECK"
                    lines.append(f"  P{index}: Helmet={helmet}; Vest={vest}")
                lines.extend([
                    "",
                    "CHECK berarti APD belum berhasil diasosiasikan, bukan bukti pelanggaran.",
                ])
            lines.extend([
                "",
                "Peringatan: asosiasi ini merupakan heuristik geometris sederhana.",
                "Hasil deteksi bukan keputusan kepatuhan SOP.",
                f"Output: {output_path}",
            ])
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

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt event name
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.contentsRect().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Safety Monitor")
        self.resize(1360, 860)
        self.setMinimumSize(980, 680)
        self.selected_image: Path | None = None
        self.worker: DetectionWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        heading = QLabel("AI Safety Monitor")
        heading.setObjectName("heading")
        subtitle = QLabel("Prototype desktop · PPE detection for industrial safety review")
        subtitle.setObjectName("subtitle")
        layout.addWidget(heading)
        layout.addWidget(subtitle)

        controls = QHBoxLayout()
        self.choose_button = QPushButton("Pilih Gambar")
        self.choose_button.setObjectName("primaryButton")
        self.run_button = QPushButton("Jalankan Analisis")
        self.run_button.setObjectName("runButton")
        self.run_button.setEnabled(False)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Deteksi Objek", "Asosiasi APD"])
        self.mode_combo.setMinimumWidth(175)
        self.status_label = QLabel("Pilih gambar untuk memulai.")
        self.status_label.setObjectName("status")
        controls.addWidget(self.choose_button)
        controls.addWidget(QLabel("Mode:"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(self.run_button)
        controls.addWidget(self.status_label, 1)
        layout.addLayout(controls)

        previews = QHBoxLayout()
        previews.setSpacing(14)
        original_column = QVBoxLayout()
        result_column = QVBoxLayout()
        original_title = QLabel("Gambar Asli")
        original_title.setObjectName("sectionTitle")
        result_title = QLabel("Hasil Analisis")
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
        self.report.setMinimumHeight(155)
        self.report.setMaximumHeight(220)
        self.report.setPlainText(
            "Model: yolo8n.pt\n"
            "GPU digunakan jika CUDA tersedia.\n\n"
            "Pilih gambar, pilih mode analisis, lalu tekan Jalankan Analisis."
        )
        layout.addWidget(self.report)

        disclaimer = QLabel(
            "Catatan: prototipe awal. CHECK berarti APD belum berhasil diasosiasikan, bukan bukti pelanggaran SOP."
        )
        disclaimer.setObjectName("disclaimer")
        layout.addWidget(disclaimer)

        self.setCentralWidget(root)
        self.choose_button.clicked.connect(self.choose_image)
        self.run_button.clicked.connect(self.run_analysis)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)

    def on_mode_changed(self, mode: str) -> None:
        self.result_preview.clear_image(f"Pilih Jalankan Analisis untuk mode: {mode}")

    def choose_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih gambar untuk dianalisis",
            str(APP_DIR),
            "Image files (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not file_path:
            return
        self.selected_image = Path(file_path)
        self.original_preview.set_image(self.selected_image)
        self.result_preview.clear_image("Tekan Jalankan Analisis")
        self.run_button.setEnabled(True)
        self.status_label.setText(f"Dipilih: {self.selected_image.name}")
        self.report.setPlainText(
            f"Gambar: {self.selected_image}\nMode: {self.mode_combo.currentText()}\n\nSiap menjalankan analisis."
        )

    def run_analysis(self) -> None:
        if self.selected_image is None:
            QMessageBox.information(self, "Pilih gambar", "Pilih gambar terlebih dahulu.")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.run_button.setEnabled(False)
        self.choose_button.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.status_label.setText("Menjalankan model...")
        self.report.setPlainText("Analisis sedang berjalan. Tunggu hingga selesai...")
        self.worker = DetectionWorker(self.selected_image, self.mode_combo.currentText())
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
        self.mode_combo.setEnabled(True)
        self.run_button.setEnabled(self.selected_image is not None)
        self.worker = None


def apply_style(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(
        "QMainWindow, QWidget { background: #111827; color: #e5e7eb; }"
        "QLabel#heading { font-size: 28px; font-weight: 700; color: #f9fafb; }"
        "QLabel#subtitle { color: #9ca3af; margin-bottom: 4px; }"
        "QLabel#sectionTitle { font-size: 14px; font-weight: 650; color: #d1d5db; }"
        "QLabel#status { color: #93c5fd; padding-left: 8px; }"
        "QLabel#imagePreview { background: #0b1220; border: 1px solid #374151; border-radius: 8px; color: #9ca3af; padding: 8px; }"
        "QLabel#disclaimer { color: #9ca3af; font-size: 9pt; }"
        "QPushButton, QComboBox { background: #374151; color: #f9fafb; border: 1px solid #4b5563; padding: 9px 13px; border-radius: 7px; }"
        "QPushButton:hover { background: #4b5563; }"
        "QPushButton:disabled { background: #1f2937; color: #6b7280; }"
        "QPushButton#primaryButton { background: #2563eb; border-color: #2563eb; font-weight: 600; }"
        "QPushButton#primaryButton:hover { background: #1d4ed8; }"
        "QPushButton#runButton { background: #047857; border-color: #047857; font-weight: 600; }"
        "QPushButton#runButton:hover { background: #065f46; }"
        "QComboBox QAbstractItemView { background: #1f2937; color: #f9fafb; selection-background-color: #2563eb; }"
        "QTextEdit { background: #0b1220; border: 1px solid #374151; border-radius: 8px; padding: 8px; color: #d1d5db; }"
    )


def main() -> int:
    app = QApplication(sys.argv)
    apply_style(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
