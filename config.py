# config.py — Konfigurasi Terpusat Sistem Deteksi APD K3
# =======================================================
#
# Regulasi acuan : PERMEN PUPR No. 10 Tahun 2021 tentang K3 Konstruksi
# Model AI       : YOLOv11s Kustom (fine-tuned pada dataset APD proyek konstruksi)
#
# Catatan untuk sidang:
#   - Semua nilai L (Likelihood) dan S (Severity) di RULES_IBPRP
#     diturunkan langsung dari tabel IBPRP PERMEN PUPR No. 10 Tahun 2021.
#   - Confidence threshold 0.327 adalah titik ekuilibrium kurva F1-Confidence
#     hasil eksperimen training model YOLOv11s pada dataset kustom ini.

import os

# ============================================================================
# KONFIGURASI PATH & FOLDER
# ============================================================================

# Direktori root proyek = folder tempat file config.py ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder penyimpanan gambar yang diunggah oleh pengguna (ASLI, sementara)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

# Folder penyimpanan gambar HASIL DETEKSI (beranotasi bounding box YOLO)
# Terpisah dari UPLOAD_FOLDER agar tidak tertukar dengan gambar asli
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")

# Path file bobot model YOLOv11s kustom hasil training skripsi
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# ============================================================================
# PARAMETER DETEKSI MODEL YOLOV11s
# ============================================================================

# Confidence threshold dikunci pada nilai EKSAK 0.327
# Nilai ini adalah titik ekuilibrium optimal dari kurva F1-Confidence
# yang diperoleh saat eksperimen training model YOLOv11s kustom.
# Mengubah nilai ini akan mempengaruhi akurasi deteksi secara signifikan.
CONFIDENCE_THRESHOLD = 0.327

# Threshold confidence KHUSUS VALIDASI KONTEKS FOTO (lebih ketat)
# Digunakan HANYA pada guard clause di app.py untuk menolak foto non-konstruksi.
# Nilai 0.60 mencegah false positive: jilbab, topi biasa, atau rambut
# yang mungkin terdeteksi sebagai 'helmet' dengan confidence rendah (0.33–0.45)
# tidak akan lolos validasi.
# CATATAN: Nilai ini TIDAK mengubah threshold inferensi YOLO — deteksi
# dan anotasi bounding box tetap menggunakan CONFIDENCE_THRESHOLD (0.327).
VALIDATION_CONF_THRESHOLD = 0.60

# Daftar 5 kelas APD wajib yang dikenali oleh model YOLOv11s kustom
# Urutan ini mengikuti urutan kelas dalam file data.yaml saat training
PPE_CLASSES = ['helmet', 'vest', 'boots', 'gloves', 'glasses']

# Label kelas "orang" dari model — digunakan sebagai denominat hitung APD
PERSON_CLASS = "person"

# Warna bounding box per kelas untuk anotasi OpenCV (format BGR, bukan RGB)
CLASS_COLORS = {
    "person":  (255,   0, 255),   # Magenta — untuk pekerja
    "helmet":  (  0, 128, 255),   # Oranye  — untuk helm
    "vest":    (  0, 255,   0),   # Hijau   — untuk rompi
    "gloves":  (  0, 165, 255),   # Oranye muda — untuk sarung tangan
    "boots":   (255,   0,   0),   # Biru    — untuk sepatu
    "glasses": (  0, 255, 255),   # Kuning  — untuk kacamata
}

# Warna fallback jika label tidak ditemukan dalam CLASS_COLORS
DEFAULT_BOX_COLOR = (200, 200, 200)

# ============================================================================
# TABEL LOOKUP IBPRP (DETERMINISTIC RULE-BASED ENGINE)
# Sumber: PERMEN PUPR No. 10 Tahun 2021 tentang K3 Konstruksi
#
# Struktur:
#   RULES_IBPRP[nama_aktivitas][nama_ppe] = {
#       'L'     : int,  # Likelihood / Kekerapan (1–5)
#       'S'     : int,  # Severity / Keparahan  (1–5)
#       'hazard': str,  # Deskripsi potensi bahaya dari regulasi
#   }
#
# PENTING:
#   - Kunci level pertama adalah nama AKTIVITAS K3 (string tampilan untuk UI)
#   - Kunci level kedua adalah nama APD yang HILANG (sesuai kelas model)
#   - Hanya APD yang WAJIB dan HILANG yang akan masuk ke tabel laporan
# ============================================================================

RULES_IBPRP = {

    # ── AKTIVITAS 1: Pengecoran Lantai ─────────────────────────────────────
    # Sumber Ground Truth: Validator Pakar Bapak Zulkifli
    # Bahaya dominan: kepala tertimpa material coran, terpeleset beton basah,
    #                 iritasi semen korosif, cipratan mortar ke mata/tangan.
    "Pengecoran Lantai": {
        "helmet": {
            "L": 4,   # Sangat mungkin terjadi (1 kali dalam 1 tahun terakhir)
            "S": 5,   # >1 orang meninggal atau cacat tetap
            "hazard": "Kepala pekerja tertimpa material coran atau runtuhan dari elevasi atas",
            "catatan_risiko": (
                "Risiko fatal akibat kepala tertimpa bucket beton, boom pump, selang "
                "concrete pump, atau benda jatuh dari elevasi atas saat pengecoran lantai."
            ),
        },
        "vest": {
            "L": 3,   # Mungkin terjadi (2 kali dalam 3 tahun terakhir)
            "S": 3,   # Cedera serius (>1 pekerja rawat inap, kehilangan waktu kerja)
            "hazard": "Visibilitas pekerja berkurang di area pengecoran lantai",
            "catatan_risiko": (
                "Mobilitas alat berat dan concrete mixer membuat visibilitas pekerja lebih "
                "penting — risiko terserempet kendaraan atau alat berat meningkat signifikan."
            ),
        },
        "boots": {
            "L": 5,   # Hampir pasti terjadi (terjadi berulang kali)
            "S": 4,   # 1 orang meninggal atau cacat tetap 1 orang
            "hazard": "Pekerja terpeleset permukaan beton basah atau kaki tertusuk besi tulangan",
            "catatan_risiko": (
                "Lantai licin akibat beton basah, risiko tertimpa alat, serta kontak kaki "
                "dengan semen basah yang korosif sangat tinggi di area pengecoran aktif."
            ),
        },
        "gloves": {
            "L": 5,   # Hampir pasti terjadi (terjadi berulang kali)
            "S": 3,   # Cedera serius (>1 pekerja rawat inap, kehilangan waktu kerja)
            "hazard": "Tangan kontak langsung dengan material semen basah yang korosif",
            "catatan_risiko": (
                "Kontak langsung dengan beton basah dapat menyebabkan iritasi, chemical burn "
                "ringan, serta gesekan dengan alat pengecoran — risiko terjadi hampir pasti."
            ),
        },
        "glasses": {
            "L": 3,   # Mungkin terjadi (2 kali dalam 3 tahun terakhir)
            "S": 3,   # Cedera serius (>1 pekerja rawat inap, kehilangan waktu kerja)
            "hazard": "Percikan material semen korosif mengenai mata pekerja",
            "catatan_risiko": (
                "Percikan beton, slurry semen, dan debu semen dapat mengenai mata pekerja "
                "selama proses penuangan dan perataan adukan cor berlangsung."
            ),
        },
    },

    # ── AKTIVITAS 2: Pemasangan Besi Lantai ───────────────────────────────
    # Sumber Ground Truth: Validator Pakar Bapak Zulkifli
    # Bahaya dominan: tertimpa/terbentur besi tulangan, tertusuk kawat bendrat,
    #                 luka sayatan tajam, koordinasi visual dalam jaringan besi.
    "Pemasangan Besi Lantai": {
        "helmet": {
            "L": 4,   # Sangat mungkin terjadi (1 kali dalam 1 tahun terakhir)
            "S": 4,   # 1 orang meninggal atau cacat tetap 1 orang
            "hazard": "Kepala terbentur atau tertimpa potongan besi tulangan saat dipindahkan",
            "catatan_risiko": (
                "Potensi benturan dengan besi tulangan, material jatuh dari aktivitas di "
                "atas, dan pergerakan pekerja yang cukup tinggi meningkatkan risiko cedera kepala."
            ),
        },
        "vest": {
            "L": 2,   # Kecil kemungkinan terjadi (1 kali dalam 3 tahun terakhir)
            "S": 3,   # Cedera serius (>1 pekerja rawat inap, kehilangan waktu kerja)
            "hazard": "Visibilitas pekerja rendah di tengah jaringan besi lantai yang padat",
            "catatan_risiko": (
                "Dibutuhkan untuk visibilitas, terutama jika ada crane, forklift, atau "
                "concrete pump di area — risiko terserempet meningkat tanpa rompi reflektif."
            ),
        },
        "boots": {
            "L": 5,   # Hampir pasti terjadi (terjadi berulang kali)
            "S": 4,   # 1 orang meninggal atau cacat tetap 1 orang
            "hazard": "Kaki tertusuk besi, terpeleset, tertimpa material, menginjak kawat bendrat",
            "catatan_risiko": (
                "Risiko tertusuk besi, terpeleset, tertimpa material, dan menginjak kawat "
                "bendrat sangat tinggi — hampir pasti terjadi tanpa pelindung kaki yang memadai."
            ),
        },
        "gloves": {
            "L": 5,   # Hampir pasti terjadi (terjadi berulang kali)
            "S": 3,   # Cedera serius (>1 pekerja rawat inap, kehilangan waktu kerja)
            "hazard": "Tangan luka gores atau tusuk akibat besi tulangan dan kawat bendrat",
            "catatan_risiko": (
                "Hampir seluruh pekerjaan melibatkan besi tulangan dan kawat bendrat yang "
                "dapat menyebabkan luka gores atau tusuk — kontak tangan hampir tidak terhindarkan."
            ),
        },
        "glasses": {
            "L": 2,   # Kecil kemungkinan terjadi (1 kali dalam 3 tahun terakhir)
            "S": 3,   # Cedera serius (>1 pekerja rawat inap, kehilangan waktu kerja)
            "hazard": "Serpihan karat besi atau debu masuk ke mata saat pemotongan besi",
            "catatan_risiko": (
                "Risiko terutama berasal dari debu, serpihan kawat, atau saat pemotongan "
                "besi tulangan lantai yang dapat mengenai mata pekerja."
            ),
        },
    },

    # ── AKTIVITAS 3: Pemasangan Besi Kolom ────────────────────────────────
    # Sumber Ground Truth: Validator Pakar Bapak Zulkifli
    # Bahaya dominan: material jatuh dari ketinggian, pijakan tidak aman pada
    #                 sengkang vertikal, kawat bendrat, debu di ketinggian.
    "Pemasangan Besi Kolom": {
        "helmet": {
            "L": 5,   # Hampir pasti terjadi (terjadi berulang kali)
            "S": 5,   # >1 orang meninggal atau cacat tetap
            "hazard": "Kepala tertimpa material jatuh atau benturan langsung dengan rangka besi kolom",
            "catatan_risiko": (
                "Banyak pekerjaan di atas kepala, besi vertikal, alat angkat, dan potensi "
                "benda jatuh — risiko cedera kepala fatal hampir pasti terjadi tanpa helm."
            ),
        },
        "vest": {
            "L": 2,   # Kecil kemungkinan terjadi (1 kali dalam 3 tahun terakhir)
            "S": 3,   # Cedera serius (>1 pekerja rawat inap, kehilangan waktu kerja)
            "hazard": "Deteksi pekerja yang memanjat rangka kolom sulit teridentifikasi tim lain",
            "catatan_risiko": (
                "Membantu visibilitas, terutama saat lifting cage kolom atau aktivitas crane — "
                "tanpa rompi, pekerja sulit terdeteksi dan rawan tertabrak alat angkat."
            ),
        },
        "boots": {
            "L": 5,   # Hampir pasti terjadi (terjadi berulang kali)
            "S": 4,   # 1 orang meninggal atau cacat tetap 1 orang
            "hazard": "Kaki tertusuk besi starter, terpeleset, atau tertimpa material besi kolom",
            "catatan_risiko": (
                "Risiko tertusuk besi starter, terpeleset pada area kerja besi kolom, dan "
                "tertimpa material tetap sangat tinggi — hampir pasti terjadi tanpa boots."
            ),
        },
        "gloves": {
            "L": 5,   # Hampir pasti terjadi (terjadi berulang kali)
            "S": 3,   # Cedera serius (>1 pekerja rawat inap, kehilangan waktu kerja)
            "hazard": "Tangan terluka mengikat kawat bendrat dan mencengkeram besi pada posisi vertikal",
            "catatan_risiko": (
                "Mengikat tulangan kolom menyebabkan kontak intensif dengan besi dan kawat "
                "bendrat — luka tangan hampir pasti terjadi jika tidak menggunakan sarung tangan."
            ),
        },
        "glasses": {
            "L": 3,   # Mungkin terjadi (2 kali dalam 3 tahun terakhir)
            "S": 3,   # Cedera serius (>1 pekerja rawat inap, kehilangan waktu kerja)
            "hazard": "Mata terpapar debu material atau serpihan besi saat penyetelan kolom",
            "catatan_risiko": (
                "Risiko meningkat karena proses pemotongan, pengikatan, dan serpihan kawat "
                "bendrat yang dapat langsung mengenai mata saat penyetelan kolom."
            ),
        },
    },
    # CATATAN: Sistem hanya menerima 3 aktivitas di atas (strict).
    # Aktivitas tidak valid ditolak di app.py sebelum masuk ke engine ini.
}

# ============================================================================
# KLASIFIKASI TINGKAT RISIKO
# Sumber: PERMEN PUPR No. 10 Tahun 2021 (Matriks Risiko 5×5)
#
# Rumus: Total Risk (TR) = Likelihood (L) × Severity (S)
# Rentang TR: 1 (minimum) s/d 25 (maksimum)
# ============================================================================

RISK_LEVELS = {
    "Kecil":  {"min": 1,  "max": 4,  "color": "green"},   # TR 1–4
    "Sedang": {"min": 5,  "max": 12, "color": "orange"},  # TR 5–12
    "Besar":  {"min": 15, "max": 25, "color": "red"},     # TR 15–25
}


def classify_risk(tr: int) -> str:
    # Tentukan Tingkat Risiko berdasarkan nilai Total Risk (TR = L x S).
    # Args:
    #   tr: Nilai Total Risk hasil perkalian L x S
    # Returns:
    #   String tingkat risiko: 'Kecil', 'Sedang', atau 'Besar'
    # Catatan sidang (PERMEN PUPR No. 10 Tahun 2021):
    #   TR 1-4   -> Kecil  (tindakan: pengawasan rutin)
    #   TR 5-12  -> Sedang (tindakan: enforcement & training)
    #   TR 15-25 -> Besar  (tindakan: hentikan pekerjaan bila perlu)
    if 1 <= tr <= 4:
        return "Kecil"
    elif 5 <= tr <= 12:
        return "Sedang"
    else:
        # TR ≥ 13 masuk kategori Besar (termasuk TR 13 dan 14 yang secara
        # teknis ada di antara Sedang dan Besar dalam matriks 5×5)
        return "Besar"


def get_activity_names() -> list:
    # Kembalikan daftar nama aktivitas K3 yang tersedia dalam RULES_IBPRP.
    # Returns:
    #   List string nama aktivitas untuk mengisi dropdown UI
    return list(RULES_IBPRP.keys())
