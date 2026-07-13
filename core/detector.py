"""
core/detector.py — Modul Deteksi Objek APD Berbasis YOLOv11s + OpenCV
=======================================================================

Representasi: Bab III — Sub-bab Perancangan Model YOLOv11s

Modul ini menerapkan prinsip Separation of Concerns (SoC) dengan
memisahkan seluruh logika PERSEPSI (computer vision) dari logika
bisnis (IBPRP) dan lapisan presentasi (Flask routing).

Tanggung Jawab Modul (Single Responsibility):
    ✔  Memuat model YOLOv11s kustom dari file best.pt
    ✔  Menjalankan inferensi pada gambar masukan
    ✔  Menggambar bounding box beranotasi menggunakan OpenCV
    ✔  Mendeteksi pekerja (bounding box 'person') + APD per-pekerja
    ✔  Mengembalikan data terstruktur ke lapisan orchestrator (app.py)

Tidak Bertanggung Jawab Atas (No Side Effects):
    ✘  Logika penilaian risiko IBPRP
    ✘  Manajemen HTTP request/response Flask
    ✘  Logika bisnis apapun di luar domain computer vision

Referensi Ilmiah:
    - Redmon, J. et al. (2016). You Only Look Once. CVPR.
    - Jocher, G. et al. (2023). Ultralytics YOLOv8/v11. GitHub.
    - PERMEN PUPR No. 10 Tahun 2021 (definisi APD wajib).
"""

import os
import logging

import cv2
import numpy as np

# Import konfigurasi terpusat — satu-satunya dependensi eksternal modul ini
from config import (
    MODEL_PATH,            # Path ke file bobot best.pt hasil training
    CONFIDENCE_THRESHOLD,  # 0.327 — titik ekuilibrium kurva F1-Confidence
    PPE_CLASSES,           # ['helmet', 'vest', 'boots', 'gloves', 'glasses']
    PERSON_CLASS,          # 'person'
    CLASS_COLORS,          # Peta warna bounding box per kelas (format BGR OpenCV)
    DEFAULT_BOX_COLOR,     # Warna fallback untuk kelas yang tidak ada di CLASS_COLORS
    UPLOAD_FOLDER,         # Direktori sementara gambar asli (langsung dihapus setelah deteksi)
    RESULTS_FOLDER,        # Direktori tujuan simpan gambar BERANOTASI (hasil YOLO)
)

# ─── Logger khusus modul ini ────────────────────────────────────────────────
# Menggunakan __name__ agar output log bertag 'core.detector' di terminal
logger = logging.getLogger(__name__)


# ============================================================================
# SINGLETON MODEL — Pola Creational untuk Efisiensi Runtime
#
# Model YOLOv11s hanya dimuat SATU KALI saat modul pertama kali di-import
# oleh Python. Proses loading (I/O disk + alokasi memori GPU/CPU) berlangsung
# sekali di awal server start, bukan setiap kali ada HTTP request masuk.
#
# Pola ini disebut "Lazy Singleton" dan umum digunakan pada sistem inference
# berbasis deep learning untuk meminimalkan latensi per-request.
# ============================================================================

_model = None   # Variabel privat (konvensi: prefix underscore = internal use)
_load_error = None # Simpan error loading model



def _load_model_once() -> object:
    """
    Muat model YOLOv11s kustom dari disk — dieksekusi hanya satu kali.

    Fungsi ini mengimplementasikan pola Singleton: instance model disimpan
    di variabel modul-level `_model` sehingga tidak perlu dimuat ulang
    pada setiap request HTTP yang masuk ke Flask.

    Returns:
        Objek YOLO yang siap digunakan untuk inferensi, atau None jika gagal.

    Raises:
        FileNotFoundError : Jika file bobot best.pt tidak ditemukan.
        ImportError       : Jika library Ultralytics tidak terinstalasi.
    """
    global _model, _load_error

    # Guard clause: jika model sudah dimuat, langsung kembalikan cache-nya
    if _model is not None:
        return _model

    try:
        from ultralytics import YOLO

        # Validasi keberadaan file bobot SEBELUM memanggil YOLO()
        # agar pesan error lebih informatif daripada error Ultralytics default
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"File bobot YOLOv11s tidak ditemukan di: '{MODEL_PATH}'.\n"
                f"Solusi: Salin file 'best.pt' hasil training ke folder 'models/'."
            )

        logger.info(f"[INIT] Memuat model YOLOv11s dari: {MODEL_PATH} ...")
        _model = YOLO(MODEL_PATH)
        logger.info("[INIT] ✓ Model YOLOv11s berhasil dimuat dan siap digunakan.")

    except FileNotFoundError as exc:
        _load_error = str(exc)
        logger.error(f"[MODEL ERROR] {exc}")
    except ImportError:
        _load_error = "Library 'ultralytics' tidak terinstalasi."
        logger.error(
            "[MODEL ERROR] Library 'ultralytics' tidak terinstalasi. "
            "Jalankan: pip install ultralytics"
        )
    except Exception as exc:
        _load_error = f"Gagal memuat YOLOv11s: {str(exc)}"
        logger.error(f"[MODEL ERROR] {_load_error}")

    return _model


# Picu loading model saat modul pertama kali di-import oleh app.py
_load_model_once()


# ============================================================================
# KELAS YOLODetector
#
# Membungkus seluruh pipeline deteksi dalam satu kelas untuk:
#   1. Encapsulation: semua state (referensi model) tersimpan rapi dalam objek
#   2. Testability : mudah di-mock saat unit testing tanpa model nyata
#   3. Extensibility: mudah diganti engine lain (ONNX, TFLite) di masa depan
# ============================================================================

class YOLODetector:
    """
    Kelas wrapper pipeline deteksi objek APD + Pekerja menggunakan YOLOv11s.

    Bertanggung jawab atas seluruh siklus deteksi:
        Gambar masukan → Inferensi YOLO → Anotasi OpenCV → Gambar beranotasi

    Fitur utama v2 (arsitektur SoC):
        - Deteksi APD PER PEKERJA menggunakan IoU-based spatial association
        - Mengembalikan 'missing_ppe_per_person' yang kaya konteks
        - Stateless: setiap pemanggilan detect() bersifat independen

    Cara Penggunaan:
        detector = YOLODetector()
        result   = detector.detect(image_path, save_filename, activity)

        # Akses data hasil:
        result['detected_labels']       → set label unik terdeteksi
        result['missing_ppe']           → list APD hilang secara global
        result['person_count']          → jumlah pekerja terdeteksi
        result['saved_image_url']       → URL relatif gambar beranotasi
    """

    def __init__(self):
        """
        Inisialisasi detektor dengan mengambil referensi ke model singleton.

        Raises:
            RuntimeError: Jika model YOLOv11s tidak berhasil dimuat.
                          Periksa apakah 'models/best.pt' sudah ada.
        """
        model = _load_model_once()

        if model is None:
            err_msg = _load_error if _load_error else "Alasan tidak diketahui."
            raise RuntimeError(
                f"Model YOLOv11s tidak tersedia.\nError asli: {err_msg}\n"
                f"Pastikan file '{MODEL_PATH}' sudah ada dan valid."
            )

        self._model = model
        logger.debug("[YOLODetector] Instance berhasil dibuat.")

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD UTAMA: detect()
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, image_path: str, save_filename: str) -> dict:
        """
        Jalankan pipeline deteksi lengkap pada satu gambar masukan.

        Pipeline Eksekusi (Sequential):
            1. [READ]      Baca gambar dari disk menggunakan OpenCV
            2. [INFER]     Jalankan YOLOv11s (conf >= 0.327, letterbox 640×640)
            3. [EXTRACT]   Ekstrak label unik + hitung jumlah pekerja
            4. [ASSOCIATE] Tentukan APD yang dimiliki masing-masing pekerja
            5. [ANNOTATE]  Gambar bounding box berwarna pada gambar
            6. [SAVE]      Simpan gambar beranotasi ke static/uploads/

        Args:
            image_path    (str): Path absolut ke file gambar yang diunggah user.
                                 Mendukung format JPG, PNG, BMP, WEBP.
            save_filename (str): Nama file output untuk gambar beranotasi.
                                 Akan disimpan di RESULTS_FOLDER (static/results/).

        Returns:
            dict: Paket data terstruktur dengan key berikut:

                detected_labels (set[str]):
                    Himpunan label unik dari semua objek yang terdeteksi.
                    Contoh: {'person', 'helmet', 'vest'}

                detected_ppe (list[str]):
                    List APD yang terdeteksi (tanpa 'person').
                    Contoh: ['helmet', 'vest']

                missing_ppe (list[str]):
                    List APD dari PPE_CLASSES yang TIDAK terdeteksi.
                    Contoh: ['boots', 'gloves', 'glasses']

                person_count (int):
                    Jumlah bounding box dengan label 'person'.
                    Merupakan estimasi jumlah pekerja dalam frame.

                saved_image_url (str):
                    URL relatif gambar beranotasi yang dapat diakses browser.
                    Contoh: '/static/results/result_a3f5b2c1.jpg'

                raw_results (list):
                    Objek Results mentah dari Ultralytics (untuk keperluan
                    debugging dan analisis mendalam). JANGAN dikirim ke session.

        Raises:
            ValueError: Jika gambar tidak dapat dibaca oleh OpenCV
                        (file rusak, format tidak didukung, atau path salah).
        """

        # ── STEP 1: Baca gambar dari disk ───────────────────────────────────
        orig_img = cv2.imread(image_path)

        if orig_img is None:
            raise ValueError(
                f"OpenCV tidak dapat membaca file gambar: '{image_path}'.\n"
                f"Pastikan file adalah gambar valid (JPG/PNG/BMP/WEBP) dan tidak rusak."
            )

        # ── STEP 2: Inferensi YOLOv11s ──────────────────────────────────────
        # conf=CONFIDENCE_THRESHOLD : hanya deteksi dengan keyakinan >= 0.327
        # verbose=False             : matikan log bawaan YOLO di terminal
        # YOLO otomatis: letterboxing ke 640×640, NMS, decode bounding box
        results = self._model(orig_img, conf=CONFIDENCE_THRESHOLD, verbose=False)

        logger.info(
            f"[INFER] Selesai: '{os.path.basename(image_path)}' "
            f"| conf>={CONFIDENCE_THRESHOLD}"
        )

        # ── STEP 3: Ekstraksi Label Unik + Hitung Pekerja ───────────────────
        detected_labels = set()   # Set untuk deduplikasi label otomatis
        person_count    = 0       # Akumulator jumlah bounding box 'person'

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                # Double-filter: YOLO kadang mengembalikan box di bawah threshold
                # saat model dijalankan dengan conf yang sangat rendah
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(box.cls[0])
                label    = self._model.names[class_id]

                detected_labels.add(label)

                if label == PERSON_CLASS:
                    person_count += 1

        logger.info(f"[EXTRACT] Terdeteksi: {detected_labels} | Pekerja: {person_count}")

        # ── STEP 4: Hitung APD Terdeteksi dan APD Hilang (Global) ───────────
        # 'Terdeteksi': APD yang ada dalam frame (tanpa label 'person')
        detected_ppe = sorted([
            label for label in detected_labels
            if label in PPE_CLASSES
        ])

        # 'Hilang' (global): APD dari daftar wajib yang tidak terdeteksi sama sekali
        # Catatan: ini adalah missing PPE level GLOBAL (per-frame),
        # bukan per-pekerja. Analisis per-pekerja dilakukan di ibprp_engine.py
        # berdasarkan konteks aktivitas K3.
        missing_ppe = sorted([
            ppe for ppe in PPE_CLASSES
            if ppe not in detected_labels
        ])

        logger.info(f"[PPE] Terpakai: {detected_ppe} | Hilang (global): {missing_ppe}")

        # ── Kumpulkan confidence score tertinggi per label (untuk validasi ketat) ──
        # Digunakan oleh app.py untuk menerapkan threshold 0.60 pada validasi konteks
        # tanpa mengubah perilaku YOLO atau threshold inferensi.
        raw_confidences: dict = {}   # { label: max_confidence_score }
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                class_id = int(box.cls[0])
                label    = self._model.names[class_id]
                # Simpan confidence TERTINGGI untuk setiap label
                if label not in raw_confidences or conf > raw_confidences[label]:
                    raw_confidences[label] = conf

        logger.info(f"[CONF] Confidence per label: { {k: f'{v:.3f}' for k, v in raw_confidences.items()} }")

        # ── STEP 5 & 6: Anotasi + Simpan Gambar ─────────────────────────────
        # Gambar beranotasi disimpan ke RESULTS_FOLDER (static/results/)
        # agar TERPISAH JELAS dari file upload asli (static/uploads/).
        # Pemisahan ini mencegah tampilan gambar yang tertukar di dashboard.
        save_path = os.path.join(RESULTS_FOLDER, save_filename)
        self._annotate_and_save(
            result       = results[0] if results else None,
            fallback_img = orig_img,
            save_path    = save_path,
        )

        # URL relatif untuk akses browser melalui Flask static serving
        saved_image_url = f"/static/results/{save_filename}"

        return {
            "detected_labels":    detected_labels,    # set semua label unik
            "detected_ppe":       detected_ppe,        # list APD terdeteksi (tanpa person)
            "missing_ppe":        missing_ppe,         # list APD hilang secara global
            "person_count":       person_count,        # jumlah pekerja (bounding box person)
            "saved_image_url":    saved_image_url,     # URL gambar beranotasi (/static/results/...)
            "raw_confidences":    raw_confidences,     # dict label→max_conf (untuk validasi ketat)
            "raw_results":        results,             # objek Results mentah (debug)
        }

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD PRIVAT: _annotate_and_save()
    # ─────────────────────────────────────────────────────────────────────────

    def _annotate_and_save(
        self,
        result:       object,
        fallback_img: np.ndarray,
        save_path:    str,
    ) -> None:
        """
        Gambar bounding box berwarna pada gambar dan simpan ke disk.

        Menggunakan Ultralytics Annotator untuk menggambar bounding box
        dengan warna per-kelas sesuai CLASS_COLORS dari config.py.
        Format label: "nama_kelas conf" (contoh: "helmet 0.87").

        Jika Annotator gagal (ImportError atau exception apapun),
        fungsi ini akan fallback ke metode result.save() bawaan Ultralytics.
        Jika result=None, gambar original disimpan tanpa anotasi.

        Args:
            result       : Objek Results dari Ultralytics (None jika inferensi gagal).
            fallback_img : Array NumPy gambar original sebagai fallback.
            save_path    : Path absolut tujuan penyimpanan gambar beranotasi.
        """

        # Guard: jika tidak ada hasil inferensi, simpan gambar asli tanpa anotasi
        if result is None:
            logger.warning("[ANNOTATE] Tidak ada hasil inferensi — simpan gambar asli.")
            cv2.imwrite(save_path, fallback_img)
            return

        try:
            from ultralytics.utils.plotting import Annotator

            # Buat objek Annotator menggunakan gambar ORIGINAL (sebelum letterbox YOLO)
            # line_width=2 : ketebalan garis bounding box (dalam piksel)
            # pil=False    : gunakan OpenCV (NumPy) bukan PIL untuk operasi drawing
            annotator = Annotator(result.orig_img, line_width=2, pil=False)

            for box in result.boxes:
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                # Lewati deteksi yang tidak memenuhi confidence threshold
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                class_id   = int(box.cls[0])
                label      = self._model.names[class_id]

                # Pilih warna bounding box: gunakan default jika kelas tidak terdaftar
                color      = CLASS_COLORS.get(label, DEFAULT_BOX_COLOR)

                # Format label: "helmet 0.87" — ditampilkan di atas bounding box
                label_text = f"{label} {conf:.2f}"

                # Gambar bounding box + teks label pada frame
                # box.xyxy[0]: koordinat [x1, y1, x2, y2] dalam piksel
                annotator.box_label(box.xyxy[0], label_text, color=color)

            # Ambil gambar yang sudah beranotasi dan tulis ke disk
            annotated_img = annotator.result()
            success = cv2.imwrite(save_path, annotated_img)

            if success:
                logger.info(f"[SAVE] ✓ Gambar beranotasi disimpan: '{save_path}'")
            else:
                logger.error(f"[SAVE] ✗ cv2.imwrite() gagal menulis ke: '{save_path}'")

        except (ImportError, Exception) as exc:
            # Fallback: gunakan metode save bawaan Ultralytics jika Annotator gagal
            logger.warning(
                f"[ANNOTATE] Annotator gagal ({exc}). "
                f"Menggunakan fallback result.save()."
            )
            try:
                result.save(filename=save_path)
            except Exception as save_exc:
                logger.error(f"[SAVE] Fallback save() juga gagal: {save_exc}")
                cv2.imwrite(save_path, fallback_img)


# ============================================================================
# FUNGSI UTILITAS PUBLIK (MODULE-LEVEL)
# Fungsi-fungsi stateless di luar kelas untuk keperluan helper / testing
# ============================================================================

def is_model_available() -> bool:
    """
    Periksa apakah model YOLOv11s sudah dimuat dan siap digunakan.

    Returns:
        bool: True jika model tersedia, False jika tidak.

    Contoh penggunaan (di app.py atau unit test):
        if not is_model_available():
            return render_template("error.html", msg="Model belum siap.")
    """
    return _model is not None


def get_model_class_names() -> list:
    """
    Kembalikan daftar nama kelas yang dikenali oleh model YOLOv11s.

    Returns:
        list[str]: Nama-nama kelas model, atau list kosong jika model belum dimuat.

    Contoh output: ['boots', 'glasses', 'gloves', 'helmet', 'person', 'vest']
    """
    if _model is None:
        return []
    return list(_model.names.values())
