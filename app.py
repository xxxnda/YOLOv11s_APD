# app.py — Routing Controller Utama Aplikasi Flask SAFEWATCH
# ===========================================================
#
# Representasi: Bab III — Sub-bab Perancangan Sistem (Modul Integrasi)
#
# File ini bertindak HANYA sebagai Orchestrator (pengatur lalu lintas):
#   1. Terima request HTTP dari browser (upload foto + pilihan aktivitas)
#   2. Delegasikan deteksi gambar ke core/detector.py
#   3. Delegasikan evaluasi risiko ke core/ibprp_engine.py
#   4. Kemas hasil dan kirim ke template dashboard.html
#
# Prinsip Separation of Concerns (SoC) yang diterapkan:
#   ┌─────────────────────────────────────────────────────────────────┐
#   │  LAYER PRESENTASI   ←  app.py  →  core/detector.py            │
#   │  (Flask HTTP)           │              (YOLOv11s + OpenCV)      │
#   │                         ↓                                       │
#   │                  core/ibprp_engine.py                           │
#   │                  (Rule-based K3 Engine)                         │
#   └─────────────────────────────────────────────────────────────────┘
#
# app.py TIDAK mengandung:
#   ✘ Logika deteksi gambar / computer vision (→ core/detector.py)
#   ✘ Logika penilaian risiko IBPRP (→ core/ibprp_engine.py)
#   ✘ Konfigurasi threshold atau aturan K3 (→ config.py)
#
# Regulasi acuan: PERMEN PUPR No. 10 Tahun 2021 tentang K3 Konstruksi
# Model AI      : YOLOv11s Kustom (fine-tuned pada dataset APD)

import os
import uuid
import logging

from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    session,
    jsonify,
)

# ── Import konfigurasi terpusat ──────────────────────────────────────────────
from config import (
    UPLOAD_FOLDER,               # Folder sementara gambar asli (dihapus setelah deteksi)
    RESULTS_FOLDER,              # Folder gambar BERANOTASI hasil YOLO (static/results/)
    CONFIDENCE_THRESHOLD,        # 0.227 — untuk metadata laporan di dashboard
    VALIDATION_CONF_THRESHOLD,   # 0.60  — threshold ketat khusus validasi konteks foto
    PPE_CLASSES,                 # Untuk validasi dan tampilan UI
    RULES_IBPRP,                 # Untuk validasi dropdown aktivitas
    ACTIVITY_REQUIRED_PPE,
    get_activity_names,          # Helper: ambil list nama aktivitas untuk dropdown
)

# ── Daftar TETAP aktivitas yang diterima sistem (3 aktivitas utama saja) ──────
# Harus konsisten dengan kunci dalam RULES_IBPRP di config.py.
# Sistem bersifat STRICT — aktivitas di luar daftar ini LANGSUNG ditolak
# dengan JSON 400 dan tidak diarahkan ke kategori fallback apapun.
ALLOWED_ACTIVITIES = frozenset(RULES_IBPRP.keys())

# ── Import modul inti (core package) — SoC ──────────────────────────────────
from core.detector import YOLODetector, is_model_available, summarize_compliance
from core.ibprp_engine import (
    evaluate_ibprp_risk,             # Hitung risiko IBPRP per APD hilang
    summarize_risk,                  # Agregasi statistik hasil IBPRP
)

# ============================================================================
# INISIALISASI FLASK + LOGGING
# ============================================================================

# Konfigurasi logging global — format informatif untuk debugging & demo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Inisialisasi aplikasi Flask
# Folder 'templates/' dan 'static/' secara default dicari di direktori app.py
app = Flask(__name__)

# Secret key untuk Flask session — sebaiknya diambil dari environment variable
# Jika FLASK_SECRET_KEY tidak di-set, gunakan fallback (aman untuk development)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "safewatch-k3-skripsi-2025-soc")

# Konfigurasi Flask tambahan
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Batas upload maks 16 MB
app.config["TEMPLATES_AUTO_RELOAD"] = True            # Reload template tanpa restart

# Pastikan folder uploads/ dan results/ sudah ada sebelum server menerima request
os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ============================================================================
# MIDDLEWARE: Nonaktifkan Cache Browser
# Memastikan gambar beranotasi yang baru selalu ditampilkan (bukan cache lama)
# ============================================================================

@app.after_request
def disable_browser_cache(response):
    """Tambahkan header HTTP untuk menonaktifkan cache browser."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "-1"
    return response


# ============================================================================
# ROUTE 1: Halaman Utama (GET /)
# ============================================================================

@app.route("/")
def index():
    """
    Tampilkan halaman utama — form upload foto K3.

    Mengirimkan data kontekstual ke template:
        activities      : List nama aktivitas K3 untuk dropdown
        model_available : Status ketersediaan model YOLOv11s
        ppe_classes     : Daftar kelas APD untuk tampilan informatif di UI
    """
    activities = get_activity_names()   # ['Pengecoran Lantai', 'Pemasangan Besi Lantai', ...]

    return render_template(
        "index.html",
        activities      = activities,
        model_available = is_model_available(),   # Tampilkan peringatan jika model tidak ada
        ppe_classes     = PPE_CLASSES,
    )


# ============================================================================
# ROUTE 2: Endpoint Prediksi (POST /predict)
# ============================================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint utama — orkestrasi pipeline deteksi APD dan evaluasi risiko IBPRP.

    Alur Orkestrasi (Orchestrator Pattern):
        1. [VALIDATE]   Validasi file upload dan pilihan aktivitas dari form
        2. [SAVE]       Simpan gambar ke UPLOAD_FOLDER dengan nama UUID unik
        3. [DETECT]     Delegasi ke core/detector.py → deteksi APD + pekerja
        4. [EVALUATE]   Delegasi ke core/ibprp_engine.py → kalkulasi risiko
        5. [SUMMARIZE]  Agregasi statistik hasil IBPRP
        6. [SESSION]    Simpan hasil ke Flask session
        7. [REDIRECT]   Redirect ke /dashboard untuk tampilkan hasil

    HTTP Method: POST
    Form Fields:
        file     (file) : File gambar yang diunggah (JPG/PNG/BMP)
        activity (str)  : Nama aktivitas K3 dari dropdown

    Returns:
        Redirect ke /dashboard (sukses) atau render index.html (gagal).
    """

    # ── STEP 1: Validasi file upload ─────────────────────────────────────────
    if "file" not in request.files:
        logger.warning("[PREDICT] Request tidak mengandung file.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if not file or file.filename == "" or file.filename is None:
        logger.warning("[PREDICT] Nama file kosong — redirect ke index.")
        return redirect(url_for("index"))

    # ── STEP 2: Validasi STRICT pilihan aktivitas ──────────────────────────────
    # Sistem HANYA menerima 3 aktivitas utama. Aktivitas tidak dikenal
    # LANGSUNG diblokir dengan JSON 400 — tidak ada fallback kategori.
    activity = request.form.get("activity", "").strip()

    if activity not in ALLOWED_ACTIVITIES:
        logger.warning(
            f"[PREDICT] Aktivitas tidak valid atau tidak terklasifikasi: '{activity}'. "
            f"Aktivitas yang diterima: {list(ALLOWED_ACTIVITIES)}"
        )
        return jsonify({
            "status":  "error",
            "message": "Foto tidak valid atau tidak sesuai dengan konteks aktivitas konstruksi yang dipilih!"
        }), 400

    logger.info(f"[PREDICT] Aktivitas valid: '{activity}'")

    # ── STEP 3: Simpan file dengan nama UUID unik ────────────────────────────
    # UUID mencegah tabrakan nama file saat banyak pengguna menggunakan sistem
    ext         = os.path.splitext(file.filename)[1].lower() or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"                    # Contoh: "a3f5...jpg"
    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)     # Path absolut

    file.save(upload_path)
    logger.info(f"[SAVE] Gambar tersimpan sementara: '{upload_path}'")

    # ── STEP 4: Deteksi APD menggunakan core/detector.py ────────────────────
    result_filename = f"result_{unique_name}"   # Nama file gambar beranotasi

    try:
        detector = YOLODetector()

        # Delegasi PENUH ke core/detector.py — app.py tidak tahu cara deteksi
        det_output = detector.detect(
            image_path    = upload_path,
            save_filename = result_filename,
        )

        detected_labels  = det_output["detected_labels"]   # set{'person', 'helmet', ...}
        detected_ppe     = det_output["detected_ppe"]       # ['helmet'] (tanpa 'person')
        person_count     = det_output["person_count"]       # jumlah pekerja
        saved_image_url  = det_output["saved_image_url"]   # '/static/results/result_...'
        raw_confidences  = det_output["raw_confidences"]   # {'helmet': 0.82, 'person': 0.91, ...}

        logger.info(
            f"[DETECT] Selesai | Pekerja: {person_count} | "
            f"APD terdeteksi: {detected_ppe}"
        )

    except RuntimeError as exc:
        # Model tidak tersedia — file best.pt mungkin belum ada
        logger.error(f"[ERR] RuntimeError saat deteksi: {exc}")
        return jsonify({
            "status": "error",
            "message": f"Model error: {str(exc)}"
        }), 500

    except ValueError as exc:
        # File gambar tidak dapat dibaca oleh OpenCV
        logger.error(f"[ERR] ValueError saat deteksi: {exc}")
        return jsonify({
            "status": "error",
            "message": f"File gambar tidak valid: {str(exc)}"
        }), 400

    except Exception as exc:
        # Error tak terduga lainnya
        logger.error(f"[ERR] Deteksi gagal: {exc}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan sistem: {str(exc)}"
        }), 500

    finally:
        # Hapus file upload ASLI — hanya gambar beranotasi yang perlu disimpan
        # Blok finally memastikan cleanup terjadi meski ada exception
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
                logger.debug(f"[CLEANUP] File asli dihapus: '{upload_path}'")
            except OSError as exc:
                logger.warning(f"[CLEANUP] Gagal hapus file asli: {exc}")

    # ── STEP 5: Validasi Berlapis dengan Filter Confidence Ketat ───────────────
    #
    # Validasi menggunakan VALIDATION_CONF_THRESHOLD (0.60) yang LEBIH KETAT
    # dari threshold inferensi YOLO (0.227). Tujuannya: mencegah false positive
    # seperti jilbab/topi/rambut yang terdeteksi sebagai 'helmet' dengan
    # confidence rendah (0.33–0.45) lolos ke perhitungan IBPRP.
    #
    # raw_confidences = confidence score TERTINGGI per label dari detector:
    #   {'person': 0.91, 'helmet': 0.36}  → helmet 0.36 < 0.60 → dianggap tidak valid
    #   {'person': 0.88, 'helmet': 0.74}  → helmet 0.74 ≥ 0.60 → valid

    # Kumpulkan label yang LOLOS filter 0.60 (confidence cukup tinggi untuk dipercaya)
    high_conf_labels: set = {
        label
        for label, conf in raw_confidences.items()
        if conf >= VALIDATION_CONF_THRESHOLD
    }

    logger.info(
        f"[VAL] raw_confidences={raw_confidences} | "
        f"high_conf_labels (>={VALIDATION_CONF_THRESHOLD})={high_conf_labels}"
    )

    # ── VALIDASI 1: Person harus terdeteksi dengan confidence ≥ 0.60 ──────────
    # Menolak: foto benda mati, hewan, makanan, lanskap, dll.
    # JUGA menolak: foto yang hanya mendeteksi 'person' dengan conf rendah
    # (bisa terjadi saat YOLO salah deteksi objek berbentuk manusia).
    if "person" not in high_conf_labels:
        logger.warning(
            f"[VAL-1] DITOLAK: 'person' tidak terdeteksi dengan conf ≥{VALIDATION_CONF_THRESHOLD} "
            f"| raw_confidences={raw_confidences}"
        )
        return jsonify({
            "status":  "error",
            "message": "Foto tidak valid atau tidak sesuai dengan konteks aktivitas konstruksi yang dipilih!"
        }), 400

    # ── VALIDASI 2: Minimal 1 APD harus terdeteksi dengan confidence ≥ 0.60 ─────
    # Menolak: foto orang di mal, pantai, kantor, atau area non-konstruksi
    # yang mungkin memiliki satu APD dengan confidence sangat rendah.
    high_conf_ppe: set = high_conf_labels & set(PPE_CLASSES)
    if not high_conf_ppe:
        logger.warning(
            f"[VAL-2] DITOLAK: Tidak ada APD konstruksi dengan conf ≥{VALIDATION_CONF_THRESHOLD} "
            f"| high_conf_labels={high_conf_labels}"
        )
        return jsonify({
            "status":  "error",
            "message": "Foto tidak valid atau tidak sesuai dengan konteks aktivitas konstruksi yang dipilih!"
        }), 400

    logger.info(
        f"[VAL-OK] Foto lolos validasi ketat "
        f"| Person conf={raw_confidences.get('person', 0):.3f} "
        f"| High-conf APD: {high_conf_ppe}"
    )

    # ── STEP 5b: Per-Person APD Counting (BARU) ────────────────────────────
    # Hitung compliance APD per pekerja menggunakan containment-based
    # spatial association. Setiap APD di-assign ke person terdekat yang
    # bounding box-nya mencakup ≥50% area APD.
    all_detections = det_output["all_detections"]
    required_ppe   = ACTIVITY_REQUIRED_PPE.get(activity, [])

    compliance_result = summarize_compliance(
        detections        = all_detections,
        activity          = activity,
        required_apd_list = required_ppe,
    )

    logger.info(
        f"[PER-PERSON] Selesai | "
        f"Total pekerja: {compliance_result['total_persons']} | "
        f"Summary: {compliance_result['compliance_summary']}"
    )

    # ── STEP 6: Evaluasi risiko IBPRP menggunakan core/ibprp_engine.py ───────
    # Delegasi PENUH ke ibprp_engine.py — app.py hanya mengoper parameter
    # Menggunakan compliance_summary untuk evaluasi per-pekerja (v3)
    ibprp_rows = evaluate_ibprp_risk(
        activity           = activity,
        compliance_summary = compliance_result['compliance_summary'],
        total_persons      = person_count,
    )

    # ── STEP 7: Agregasi statistik hasil IBPRP ───────────────────────────────
    # Delegasi ke ibprp_engine.py — menghasilkan ringkasan untuk header dashboard
    risk_summary = summarize_risk(ibprp_rows)

    logger.info(
        f"[IBPRP] Selesai | Baris risiko: {len(ibprp_rows)} | "
        f"Ringkasan: {risk_summary}"
    )

    # ── STEP 8: Simpan hasil ke Flask session + redirect ─────────────────────
    # Session digunakan agar URL /dashboard bersih (tanpa query string panjang)
    # dan data tersedia saat user me-refresh halaman dashboard
    session["result"] = {
        # ─ Metadata request
        "activity":          activity,           # Nama aktivitas K3 yang dipilih
        "conf_threshold":    CONFIDENCE_THRESHOLD,  # 0.227 (untuk metadata laporan)

        # ─ Hasil deteksi
        "required_ppe":       required_ppe,       # APD yang wajib untuk aktivitas ini
        "person_count":       person_count,       # Jumlah pekerja terdeteksi
        "image_url":          saved_image_url,    # URL gambar beranotasi bounding box

        # ─ Hasil evaluasi
        "ibprp_rows":         ibprp_rows,         # Baris tabel IBPRP (list of dict)
        "risk_summary":       risk_summary,       # Ringkasan statistik risiko
        "person_details":     compliance_result.get("person_details", []), # Detail per pekerja
        "compliance_summary": compliance_result.get("compliance_summary", {}), # Summary per APD
    }

    return redirect(url_for("dashboard"))


# ============================================================================
# ROUTE 3: Halaman Dashboard Hasil (GET /dashboard)
# ============================================================================

@app.route("/dashboard")
def dashboard():
    """
    Tampilkan hasil deteksi dan tabel laporan IBPRP di halaman dashboard.

    Membaca data dari Flask session yang diisi oleh route /predict.
    Jika session kosong (pengguna mengakses langsung tanpa upload),
    redirect kembali ke halaman utama.

    Template Variables yang dikirim ke dashboard.html:
        activity       (str)  : Nama aktivitas K3
        detected_ppe   (list) : APD yang terdeteksi
        missing_ppe    (list) : APD yang hilang (spesifik aktivitas)
        required_ppe   (list) : APD yang wajib untuk aktivitas ini
        ibprp_rows     (list) : Baris tabel IBPRP
        risk_summary   (dict) : Ringkasan statistik (TR tertinggi, distribusi)
        image_url      (str)  : URL gambar beranotasi
        conf_threshold (float): 0.227 (untuk footer laporan)
        person_count   (int)  : Jumlah pekerja terdeteksi
    """
    result = session.get("result")

    if result is None:
        logger.warning("[DASHBOARD] Session kosong — redirect ke halaman utama.")
        return redirect(url_for("index"))

    return render_template(
        "dashboard.html",
        activity           = result["activity"],
        required_ppe       = result.get("required_ppe", []),
        ibprp_rows         = result["ibprp_rows"],
        risk_summary       = result.get("risk_summary", {}),
        image_url          = result["image_url"],
        conf_threshold     = result["conf_threshold"],
        person_count       = result.get("person_count", 0),
        person_details     = result.get("person_details", []),
        compliance_summary = result.get("compliance_summary", {}),
        activity_rules     = RULES_IBPRP.get(result["activity"], {}),
    )


# ============================================================================
# ROUTE 4: API Health Check (GET /api/health) — Opsional
# Endpoint ini berguna untuk memverifikasi status sistem secara programatik
# (misalnya oleh monitoring tool atau saat debugging deployment)
# ============================================================================

@app.route("/api/health")
def health_check():
    """
    Kembalikan status sistem dalam format JSON.

    Response JSON:
        {
            "status"          : "ok" | "degraded",
            "model_available" : bool,
            "confidence_threshold": float,
            "ppe_classes"     : list[str],
            "activities"      : list[str]
        }
    """
    model_ok = is_model_available()
    return jsonify({
        "status":               "ok" if model_ok else "degraded",
        "model_available":      model_ok,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "ppe_classes":          PPE_CLASSES,
        "activities":           get_activity_names(),
    })


# ============================================================================
# ENTRY POINT — Jalankan Server Flask Development
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("  SAFEWATCH — Sistem Deteksi APD Berbasis YOLOv11s")
    logger.info("  Skripsi K3 | PERMEN PUPR No. 10 Tahun 2021")
    logger.info("  Arsitektur: Separation of Concerns (SoC) + DRM")
    logger.info("=" * 65)
    logger.info(f"  Confidence Threshold  : {CONFIDENCE_THRESHOLD}")
    logger.info(f"  Kelas APD Wajib       : {PPE_CLASSES}")
    logger.info(f"  Status Model          : {'✓ Siap' if is_model_available() else '✗ Tidak tersedia'}")
    logger.info(f"  URL Server            : http://localhost:5000")
    logger.info("=" * 65)

    # debug=False untuk demo sidang — hindari stack trace di browser
    # Ganti ke debug=True selama pengembangan / testing
    app.run(debug=False, host="0.0.0.0", port=5000)

