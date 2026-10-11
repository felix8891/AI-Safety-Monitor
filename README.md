# AI Safety Monitoring

Aplikasi desktop untuk mendeteksi penggunaan PPE/APD (person, helmet, safety-vest) menggunakan YOLO dan PySide6.

## Jalankan di Windows (one-click)

1. Install **Python 3.10 atau 3.11** dari https://www.python.org/downloads/windows/ dan centang **Add Python to PATH**.
2. Pastikan koneksi internet aktif saat instalasi dependency.
3. Klik dua kali `setup.bat`. Script akan membuat `.venv`, memasang dependency, lalu menjalankan aplikasi.
4. Untuk penggunaan berikutnya, klik dua kali `run.bat`.

Model `yolo8n.pt` harus berada di folder yang sama dengan `app.py` (sudah disertakan di repo ini).

## Dependency

Dependency utama tercantum di `requirements.txt`: Ultralytics/YOLO, PySide6, dan OpenCV. Instalasi PyTorch dapat memerlukan unduhan cukup besar.

## Catatan Git

Folder `outputs/`, `runs/`, dan `.venv/` diabaikan oleh Git karena berisi hasil inferensi atau file environment lokal. Jangan commit video/data pribadi atau file berukuran besar kecuali memang diperlukan kelompok.

## Troubleshooting

- Jika `Python tidak ditemukan`, install Python 3.10/3.11 dan aktifkan opsi Add Python to PATH, lalu buka ulang terminal.
- Jika instalasi gagal, pastikan internet aktif, lalu jalankan `setup.bat` lagi.
- Jika Windows memblokir file `.bat`, klik kanan file lalu pilih **Run** / **Open** sesuai peringatan Windows.
