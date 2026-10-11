from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import cv2
import torch
from ultralytics import YOLO

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
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

# Different thresholds because the model showed lower-confidence vest detections.
CLASS_THRESHOLDS = {
    "person": 0.35,
    "helmet": 0.35,
    "safety-vest": 0.15,
}

# OpenCV uses BGR colors.
CLASS_COLORS = {
    "person": (55, 190, 90),
    "helmet": (255, 190, 30),
    "safety-vest": (205, 65, 215),
}


class DetectionWorker(QThread):
    completed = Signal(str, str)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, image_path: Path) -> None:
        super().__init__()
        self.image_path = image_path

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
            gpu_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
            self.status.emit(f"Memuat model dan menjalankan deteksi pada {gpu_name}...")

            model = YOLO(str(MODEL_PATH))
            names = model.names
            class_ids = [
                int(class_id)
                for class_id, name in names.items()
                if str(name).lower() in CLASS_THRESHOLDS
            ]
            if not class_ids:
                raise RuntimeError(
                    "Model tidak memiliki kelas person, helmet, atau safety-vest."
                )

            # The lowest class threshold is used for inference; class-specific
            # thresholds are then applied below.
            results = model.predict(
                source=str(self.image_path),
                device=device,
                imgsz=640,
                conf=min(CLASS_THRESHOLDS.values()),
                iou=0.50,
                classes=class_ids,
                verbose=False,
            )
            result = results[0]

            image = cv2.imread(str(self.image_path))
            if image is None:
                raise RuntimeError("OpenCV tidak dapat membaca gambar ini.")

            counts: Counter[str] = Counter()
            lines = [
                "AI SAFETY MONITOR — HASIL DETEKSI",
                f"Model: {MODEL_PATH.name}",
                f"Device: {gpu_name}",
                f"Gambar: {self.image_path.name}",
                "",
                "Ambang confidence per kelas:",
                "  person: 0.35",
                "  helmet: 0.35",
                "  safety-vest: 0.15",
                "",
                "Deteksi yang ditampilkan:",
            ]

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    label = str(names[class_id]).lower()
                    confidence = float(box.conf.item())
                    if label not in CLASS_THRESHOLDS:
                        continue
                    if confidence < CLASS_THRESHOLDS[label]:
                        continue

                    x1, y1, x2, y2 = (
                        int(round(value)) for value in box.xyxy[0].tolist()
                    )
                    color = CLASS_COLORS[label]
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{label} {confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.55
                    thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label_text, font, font_scale, thickness
                    )
                    text_top = max(0, y1 - text_h - baseline - 6)
                    cv2.rectangle(
                        image,
                        (x1, text_top),
                        (x1 + text_w + 6, text_top + text_h + baseline + 6),
                        color,
                        -1,
                    )
                    cv2.putText(
                        image,
                        label_text,
                        (x1 + 3, text_top + text_h + 2),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
                    counts[label] += 1
                    lines.append(f"  {label}: {confidence:.2f}")

            if not any(counts.values()):
                lines.append("  Tidak ada objek target yang melewati ambang.")

            lines.extend(
                [
                    "",
                    "RINGKASAN",
                    f"  Person: {counts['person']}",
                    f"  Helmet: {counts['helmet']}",
                    f"  Safety vest: {counts['safety-vest']}",
                    "",
                    "Catatan: deteksi objek bukan keputusan kepatuhan SOP.",
                    "Asosiasi APD dan safety zone belum diterapkan pada GUI versi awal.",
                ]
            )

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / f"{self.image_path.stem}_detected.jpg"
            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(f"Gagal menyimpan hasil ke:\n{output_path}")

            self.completed.emit(str(output_path), "\n".join(lines))
        except Exception as exc:  # Surface inference errors to the GUI.
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
        self.resize(1240, 820)
        self.setMinimumSize(920, 650)
        self.selected_image: Path | None = None
        self.worker: DetectionWorker | None = None
        self.output_image: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        heading = QLabel("AI Safety Monitor")
        heading.setObjectName("heading")
        subtitle = QLabel(
            "Prototype desktop · PPE detection for industrial safety review"
        )
        subtitle.setObjectName("subtitle")
        layout.addWidget(heading)
        layout.addWidget(subtitle)

        controls = QHBoxLayout()
        self.choose_button = QPushButton("Pilih Gambar")
        self.choose_button.setObjectName("primaryButton")
        self.run_button = QPushButton("Jalankan Deteksi")
        self.run_button.setObjectName("runButton")
        self.run_button.setEnabled(False)
        self.status_label = QLabel("Pilih gambar untuk memulai.")
        self.status_label.setObjectName("status")
        controls.addWidget(self.choose_button)
        controls.addWidget(self.run_button)
        controls.addWidget(self.status_label, 1)
        layout.addLayout(controls)

        previews = QHBoxLayout()
        previews.setSpacing(14)
        original_column = QVBoxLayout()
        result_column = QVBoxLayout()

        original_title = QLabel("Gambar Asli")
        original_title.setObjectName("sectionTitle")
        result_title = QLabel("Hasil Deteksi")
        result_title.setObjectName("sectionTitle")

        self.original_preview = ImagePreview("Belum ada gambar dipilih")
        self.result_preview = ImagePreview("Hasil deteksi akan muncul di sini")
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
        self.report.setMinimumHeight(145)
        self.report.setMaximumHeight(190)
        self.report.setPlainText(
            "Model: yolo8n.pt\n"
            "GPU akan digunakan jika CUDA tersedia.\n\n"
            "Pilih gambar, kemudian tekan Jalankan Deteksi."
        )
        layout.addWidget(self.report)

        disclaimer = QLabel(
            "Catatan: ini prototipe awal. Hasil deteksi bukan penetapan pelanggaran SOP."
        )
        disclaimer.setObjectName("disclaimer")
        layout.addWidget(disclaimer)

        self.setCentralWidget(root)
        self.choose_button.clicked.connect(self.choose_image)
        self.run_button.clicked.connect(self.run_detection)

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
        self.output_image = None
        self.original_preview.set_image(self.selected_image)
        self.result_preview.clear_image("Tekan Jalankan Deteksi")
        self.run_button.setEnabled(True)
        self.status_label.setText(f"Dipilih: {self.selected_image.name}")
        self.report.setPlainText(
            f"Gambar: {self.selected_image}\n\n"
            "Siap menjalankan deteksi pada model SH17."
        )

    def run_detection(self) -> None:
        if self.selected_image is None:
            QMessageBox.information(self, "Pilih gambar", "Pilih gambar terlebih dahulu.")
            return
        if self.worker is not None and self.worker.isRunning():
            return

        self.run_button.setEnabled(False)
        self.choose_button.setEnabled(False)
        self.status_label.setText("Menjalankan model...")
        self.report.setPlainText("Inferensi sedang berjalan. Tunggu hingga selesai...")

        self.worker = DetectionWorker(self.selected_image)
        self.worker.status.connect(self.status_label.setText)
        self.worker.completed.connect(self.on_detection_completed)
        self.worker.failed.connect(self.on_detection_failed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_detection_completed(self, output_path: str, report: str) -> None:
        self.output_image = Path(output_path)
        self.result_preview.set_image(self.output_image)
        self.report.setPlainText(report)
        self.status_label.setText(f"Selesai: {self.output_image}")

    def on_detection_failed(self, message: str) -> None:
        self.status_label.setText("Deteksi gagal.")
        self.report.setPlainText(message)
        QMessageBox.critical(self, "Kesalahan deteksi", message)

    def on_worker_finished(self) -> None:
        self.choose_button.setEnabled(True)
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
        "QLabel#imagePreview { background: #0b1220; border: 1px solid #374151;"
        " border-radius: 8px; color: #9ca3af; padding: 8px; }"
        "QLabel#disclaimer { color: #9ca3af; font-size: 9pt; }"
        "QPushButton { background: #374151; color: #f9fafb; border: 1px solid #4b5563;"
        " padding: 10px 16px; border-radius: 7px; }"
        "QPushButton:hover { background: #4b5563; }"
        "QPushButton:disabled { background: #1f2937; color: #6b7280; }"
        "QPushButton#primaryButton { background: #2563eb; border-color: #2563eb; font-weight: 600; }"
        "QPushButton#primaryButton:hover { background: #1d4ed8; }"
        "QPushButton#runButton { background: #047857; border-color: #047857; font-weight: 600; }"
        "QPushButton#runButton:hover { background: #065f46; }"
        "QTextEdit { background: #0b1220; border: 1px solid #374151;"
        " border-radius: 8px; padding: 8px; color: #d1d5db; }"
    )


def main() -> int:
    app = QApplication(sys.argv)
    apply_style(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
