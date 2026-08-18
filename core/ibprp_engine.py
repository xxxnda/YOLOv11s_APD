"""
core/ibprp_engine.py — Mesin Penilaian Risiko K3 Berbasis Aturan IBPRP
========================================================================

Representasi: Bab III — Sub-bab Perancangan Penilaian Risiko IBPRP

Modul ini menerapkan prinsip Separation of Concerns (SoC) dengan
memisahkan seluruh logika KOGNISI (penilaian risiko K3) dari logika
persepsi (YOLO) dan lapisan presentasi (Flask routing).

Regulasi Acuan:
    PERMEN PUPR No. 10 Tahun 2021 tentang Pedoman Sistem Manajemen
    Keselamatan Konstruksi (SMKK) — Lampiran I Formulir IBPRP.

    Metode penilaian risiko yang diimplementasikan:
        IBPRP (Identifikasi Bahaya, Penilaian, dan Pengendalian Risiko)

Sumber Data (Ground Truth Matrix):
    Seluruh nilai L, S, dan narasi catatan_risiko pada RULES_IBPRP
    telah divalidasi oleh Validator Pakar (Bapak Zulkifli) dan
    mengacu ketat pada PERMEN PUPR No. 10 Tahun 2021.

Aktivitas K3 yang Dicakup (3 Aktivitas Konstruksi Gedung):
    1. Pengecoran Lantai
    2. Pemasangan Besi Lantai
    3. Pemasangan Besi Kolom

Item APD yang Dievaluasi (5 Kelas Model YOLOv11):
    helmet, vest, boots, gloves, glasses

Metodologi Perhitungan:
    Skor Tingkat Risiko (TR) dihitung menggunakan formula matriks 5×5:

        TR = L × S

    Di mana:
        L  = Likelihood (Kekerapan/Kemungkinan terjadinya kejadian)
             Skala ordinal 1–5:
               1 = Hampir tidak pernah (Terakhir >3 tahun lalu)
               2 = Kecil kemungkinan (1× dalam 3 tahun terakhir)
               3 = Mungkin terjadi (2× dalam 3 tahun terakhir)
               4 = Sangat mungkin (1× dalam 1 tahun terakhir)
               5 = Hampir pasti (>2× dalam 1 tahun terakhir)
        S  = Severity / Keparahan (Fokus Dampak Manusia)
             Skala ordinal 1–5:
               1 = Cedera ringan (Cukup P3K, tanpa kehilangan waktu kerja)
               2 = Cedera sedang (1 pekerja rawat inap)
               3 = Cedera serius (>1 pekerja rawat inap)
               4 = 1 orang meninggal atau cacat tetap 1 orang
               5 = Fatality/cacat tetap >1 orang
        TR = Total Risk (Skor Risiko Keseluruhan), rentang 1–25

    Klasifikasi TR berdasarkan matriks risiko PERMEN PUPR No. 10/2021:
        TR  1–4  → Kecil  (tindakan: pengawasan rutin)
        TR  5–12 → Sedang (tindakan: enforcement & safety training)
        TR 13–25 → Besar  (tindakan: hentikan pekerjaan bila perlu)

Tanggung Jawab Modul (Single Responsibility):
    ✔  Evaluasi risiko K3 berdasarkan APD yang hilang + aktivitas
    ✔  Lookup nilai L dan S dari tabel Ground Truth RULES_IBPRP
    ✔  Kalkulasi TR = L × S
    ✔  Klasifikasi tingkat risiko ke 'Kecil' / 'Sedang' / 'Besar'
    ✔  Menyediakan narasi catatan_risiko terkorelasi per APD & aktivitas
    ✔  Menyediakan ringkasan statistik hasil evaluasi
    ✔  Serialisasi output JSON terstruktur via generate_ibprp_json()

Tidak Bertanggung Jawab Atas (No Side Effects):
    ✘  Deteksi objek atau pemrosesan gambar (domain core/detector.py)
    ✘  Manajemen HTTP request/response Flask (domain app.py)
    ✘  Penyimpanan data ke database atau file

Catatan Desain (Purely Functional):
    Semua fungsi dalam modul ini adalah FUNGSI MURNI (pure functions):
        - Tidak memiliki state internal
        - Output hanya bergantung pada input (deterministik)
        - Tidak ada side effects (tidak menulis ke disk, tidak ke network)
    Desain ini memudahkan pengujian unit (unit testing) secara terisolasi.

Referensi:
    - PERMEN PUPR No. 10 Tahun 2021, Lampiran I: Formulir IBPRP
    - Ground Truth Matrix — Validator Pakar Bapak Zulkifli
    - Ramli, S. (2010). Sistem Manajemen Keselamatan & Kesehatan Kerja.
"""

import json
import logging
from typing import Optional

# Import konfigurasi terpusat — satu-satunya dependensi eksternal modul ini
from config import (
    RULES_IBPRP,    # Tabel Ground Truth IBPRP: aktivitas → APD → {L, S, hazard, catatan_risiko}
    PPE_CLASSES,    # Daftar 5 kelas APD wajib dari model YOLOv11s
    CLASS_COLORS,   # Bounding box colors
    classify_risk,  # Fungsi klasifikasi TR → 'Kecil'/'Sedang'/'Besar'
)

# ─── Logger khusus modul ini ────────────────────────────────────────────────
# Menggunakan __name__ agar output log bertag 'core.ibprp_engine' di terminal
logger = logging.getLogger(__name__)

# ── Mapping nama tampilan APD (Bahasa Inggris) ────────────────────────────
# Digunakan untuk kolom 'apd_hilang' di output tabel IBPRP dashboard
APD_DISPLAY_NAMES = {
    'helmet':  'Helmet',
    'vest':    'Vest',
    'boots':   'Boots',
    'gloves':  'Gloves',
    'glasses': 'Glasses',
}


# ============================================================================
# FUNGSI UTAMA — evaluate_ibprp_risk()
# ============================================================================

def evaluate_ibprp_risk(
    activity: str,
    compliance_summary: dict,
    total_persons: int,
) -> list:
    """
    Evaluasi risiko K3 menggunakan metode IBPRP untuk satu skenario deteksi.

    Ini adalah fungsi inti dari sistem — implementasi deterministic rule-based
    reasoning sesuai metodologi IBPRP pada PERMEN PUPR No. 10 Tahun 2021.

    Versi v3 (Compliance Summary):
        Menerima compliance_summary yang berasal dari detector.py.
        - APD dianggap "hilang" jika statusnya "tidak_ada" atau "sebagian" (n < N)
        - Field 'pekerja_terdampak' ditambahkan ke output (format: "M dari N")

    Args:
        activity (str):
            Nama aktivitas K3 yang dipilih pengguna dari dropdown UI.

        compliance_summary (dict):
            Hasil dari summarize_compliance() untuk sinkronisasi dengan panel kiri.

        total_persons (int):
            Jumlah pekerja terdeteksi (digunakan untuk format "M dari N").

    Returns:
        list[dict]: List baris tabel IBPRP, terurut berdasarkan nomor urut.
            Setiap elemen dict memiliki struktur:
            {
                'no'                (int)      : Nomor urut baris tabel (1-indexed)
                'ppe'               (str)      : Nama APD yang hilang (kelas model)
                'apd_hilang'        (str)      : Nama tampilan APD Bahasa Indonesia
                'pekerja_terdampak' (str)      : "M dari N" (BARU)
                'status_deteksi'    (str)      : Selalu "Tidak Terdeteksi"
                'hazard'            (str)      : Deskripsi singkat potensi bahaya
                'catatan_risiko'    (str)      : Narasi korelasi aktivitas–APD
                'L'                 (int|None) : Likelihood / Kekerapan (1–5)
                'S'                 (int|None) : Severity / Keparahan (1–5)
                'TR'                (int|None) : Total Risk = L × S (1–25)
                'risk_level'        (str)      : 'Kecil', 'Sedang', 'Besar', 'Perlu Kalibrasi'
                'color'             (str)      : Warna badge UI
                'keterangan'        (str|None) : Catatan tambahan (untuk Perlu Kalibrasi)
            }
    """

    # ── Ambil tabel aturan untuk aktivitas yang dipilih ─────────────────────
    activity_rules: dict = RULES_IBPRP.get(activity, {})

    if not activity_rules:
        logger.warning(
            f"[IBPRP] Aktivitas '{activity}' tidak ditemukan dalam RULES_IBPRP. "
            f"Kembalikan list kosong."
        )
        return []

    # ── Tentukan APD mana yang perlu dievaluasi ─────────────────────────────
    # Gunakan keys dari compliance_summary (yang mana merupakan required APD dari aktivitas ini)
    apd_to_check = list(compliance_summary.keys())
    
    rows      = []   # Akumulator baris hasil evaluasi
    row_index = 1    # Nomor urut baris tabel (1-indexed, untuk UI)
    N         = total_persons

    # ── Iterasi setiap APD yang perlu dievaluasi ────────────────────────────
    for ppe_name in apd_to_check:
        comp = compliance_summary[ppe_name]

        # ─ Tentukan jumlah pekerja yang melanggar (M) ────────────────────────
        if comp["status"] == "lengkap":
            # Semua pekerja punya APD ini → tidak ada risiko → lewati
            logger.debug(f"[IBPRP] APD TERPENUHI (semua pekerja): '{ppe_name}' — dilewati.")
            continue
            
        M = N - comp["n"]  # Jumlah yang kehilangan APD

        # ─ Format string pekerja terdampak ───────────────────────────────────
        pekerja_terdampak = f"{M} dari {N}" if N > 0 else "0 dari 0"

        # ─ Nama tampilan APD (Bahasa Indonesia) ──────────────────────────────
        apd_display_name = APD_DISPLAY_NAMES.get(ppe_name, ppe_name.title())

        # ─ Cek apakah kombinasi (activity, ppe) ada di RULES_IBPRP ──────────
        rule_params = activity_rules.get(ppe_name)

        if rule_params is None:
            # Kombinasi TIDAK ADA di IBPRP_LOOKUP → Perlu Kalibrasi
            row = {
                "no":                row_index,
                "ppe":               ppe_name,
                "apd_hilang":        apd_display_name,
                "pekerja_terdampak": pekerja_terdampak,
                "status_deteksi":    "Tidak Terdeteksi",
                "hazard":            "-",
                "catatan_risiko":    "Kombinasi belum tervalidasi Praktisi K3 pada Lampiran 3",
                "L":                 None,
                "S":                 None,
                "TR":                None,
                "risk_level":        "Perlu Kalibrasi",
                "color":             "gray",
                "keterangan":        "Kombinasi belum tervalidasi Praktisi K3 pada Lampiran 3",
            }
            rows.append(row)
            logger.info(
                f"[IBPRP] ⚠ APD HILANG (tidak terkalibrasi): '{ppe_name}' | "
                f"Pekerja terdampak: {pekerja_terdampak} → PERLU KALIBRASI"
            )
            row_index += 1
            continue

        # ─ APD ini HILANG dan ADA di RULES → Hitung risiko IBPRP ────────────
        L  = rule_params["L"]      # Likelihood — Ground Truth Pakar (PERMEN PUPR No. 10/2021)
        S  = rule_params["S"]      # Severity   — Ground Truth Pakar (PERMEN PUPR No. 10/2021)
        TR = L * S                 # Total Risk = L × S (rumus matriks K3 standar)

        # Klasifikasikan TR ke dalam kategori tingkat risiko
        risk_level = classify_risk(TR)   # 'Kecil', 'Sedang', atau 'Besar'

        # Mapping warna untuk badge/chip di UI (sesuai konvensi warna K3 internasional)
        color_map   = {"Kecil": "green", "Sedang": "orange", "Besar": "red"}
        badge_color = color_map.get(risk_level, "gray")

        # ─ Ambil dan Konversi Warna dari BGR (OpenCV) ke RGB (CSS) ────────────
        bgr = CLASS_COLORS.get(ppe_name, (200, 200, 200))
        r, g, b = bgr[2], bgr[1], bgr[0]
        
        # RGB murni untuk background
        box_color_rgb = f"{r}, {g}, {b}"
        
        # Gelapkan warna khusus untuk teks jika warnanya sangat terang (seperti kuning/hijau muda)
        # Ambang batas luminance sederhana (r + g + b)
        if (r + g + b) > 400:
            text_r, text_g, text_b = int(r * 0.6), int(g * 0.6), int(b * 0.6)
        else:
            text_r, text_g, text_b = r, g, b
            
        text_color_rgb = f"{text_r}, {text_g}, {text_b}"

        # Ambil narasi catatan_risiko dari Ground Truth — sudah divalidasi pakar
        catatan = rule_params.get("catatan_risiko", rule_params.get("hazard", "-"))

        # ─ Tambahkan baris ke hasil ──────────────────────────────────────────
        row = {
            "no":                row_index,              # Nomor urut untuk kolom tabel
            "ppe":               ppe_name,               # Nama APD hilang (contoh: 'helmet')
            "apd_hilang":        apd_display_name,       # Nama tampilan Bahasa Indonesia
            "pekerja_terdampak": pekerja_terdampak,      # "M dari N" (per-person)
            "status_deteksi":    "Tidak Terdeteksi",     # Status APD sesuai format output JSON
            "hazard":            rule_params["hazard"],  # Deskripsi singkat bahaya (untuk UI)
            "catatan_risiko":    catatan,                # Narasi korelasi pakar (detail laporan)
            "L":                 L,                      # Likelihood (Kekerapan)
            "S":                 S,                      # Severity (Keparahan)
            "TR":                TR,                     # Total Risk = L × S
            "risk_level":        risk_level,             # 'Kecil' / 'Sedang' / 'Besar'
            "color":             badge_color,            # Warna UI: 'green'/'orange'/'red'
            "box_color_rgb":     box_color_rgb,          # Warna bounding box untuk UI (R, G, B)
            "text_color_rgb":    text_color_rgb,         # Warna teks agar kontras
            "keterangan":        None,                   # Tidak ada catatan tambahan
        }
        rows.append(row)

        logger.info(
            f"[IBPRP] ✗ APD HILANG: '{ppe_name}' | "
            f"Pekerja terdampak: {pekerja_terdampak} | "
            f"L={L}, S={S}, TR={TR} → {risk_level.upper()}"
        )
        row_index += 1

    # ─ Log ringkasan jika tidak ada risiko ──────────────────────────────────
    if not rows:
        logger.info(
            f"[IBPRP] ✓ Semua APD wajib terpenuhi untuk aktivitas '{activity}'. "
            f"Tidak ada risiko teridentifikasi."
        )

    return rows


# ============================================================================
# FUNGSI AGREGASI — summarize_risk()
# ============================================================================

def summarize_risk(ibprp_rows: list) -> dict:
    """
    Hasilkan ringkasan statistik dari hasil evaluasi IBPRP.

    Fungsi ini mengagregasi data dari output evaluate_ibprp_risk()
    menjadi metrik-metrik ringkas yang berguna untuk ditampilkan
    di header/summary section pada dashboard.

    Args:
        ibprp_rows (list[dict]):
            Output dari evaluate_ibprp_risk() — list baris tabel IBPRP.
            Menerima list kosong [] dengan aman (tidak error).

    Returns:
        dict: Ringkasan statistik evaluasi IBPRP:
            {
                'total_risks'    (int) : Total APD yang hilang (jumlah baris tabel)
                'highest_tr'     (int) : Nilai TR tertinggi dari semua APD hilang
                'highest_level'  (str) : Level risiko tertinggi: 'Kecil'/'Sedang'/'Besar'
                'highest_color'  (str) : Warna UI untuk level tertinggi
                'count_besar'    (int) : Jumlah temuan risiko level 'Besar'
                'count_sedang'   (int) : Jumlah temuan risiko level 'Sedang'
                'count_kecil'    (int) : Jumlah temuan risiko level 'Kecil'
                'is_safe'        (bool): True jika tidak ada APD yang hilang
            }

    Catatan:
        Jika ibprp_rows kosong (semua APD terpenuhi), is_safe=True dan
        seluruh count akan bernilai 0.
    """

    # Guard: tangani list kosong tanpa error
    if not ibprp_rows:
        return {
            "total_risks":   0,
            "highest_tr":    0,
            "highest_level": "N/A",
            "highest_color": "green",
            "count_besar":   0,
            "count_sedang":  0,
            "count_kecil":   0,
            "is_safe":       True,
        }

    # ── Hitung statistik agregat ────────────────────────────────────────────
    total_risks = len(ibprp_rows)

    # Temukan baris dengan nilai TR tertinggi (None dianggap 0 untuk Perlu Kalibrasi)
    max_row     = max(ibprp_rows, key=lambda r: r["TR"] or 0)
    highest_tr  = max_row["TR"] or 0

    # Tentukan level dan warna untuk TR tertinggi
    highest_level = classify_risk(highest_tr)
    color_map     = {"Kecil": "green", "Sedang": "orange", "Besar": "red"}
    highest_color = color_map.get(highest_level, "gray")

    # Hitung distribusi berdasarkan level risiko
    count_besar  = sum(1 for r in ibprp_rows if r["risk_level"] == "Besar")
    count_sedang = sum(1 for r in ibprp_rows if r["risk_level"] == "Sedang")
    count_kecil  = sum(1 for r in ibprp_rows if r["risk_level"] == "Kecil")

    summary = {
        "total_risks":   total_risks,
        "highest_tr":    highest_tr,
        "highest_level": highest_level,
        "highest_color": highest_color,
        "count_besar":   count_besar,
        "count_sedang":  count_sedang,
        "count_kecil":   count_kecil,
        "is_safe":       False,   # Ada APD yang hilang → tidak aman
    }

    logger.info(
        f"[SUMMARY] Total risiko: {total_risks} | "
        f"TR tertinggi: {highest_tr} ({highest_level}) | "
        f"Besar: {count_besar}, Sedang: {count_sedang}, Kecil: {count_kecil}"
    )

    return summary





# ============================================================================
# FUNGSI SERIALISASI — generate_ibprp_json()
# ============================================================================

def generate_ibprp_json(activity: str, ibprp_row: dict) -> str:
    """
    Serialisasi satu baris hasil evaluasi IBPRP ke format JSON terstruktur.

    Menghasilkan representasi JSON canonical yang sesuai dengan spesifikasi
    output sistem IBPRP (format yang divalidasi pakar). Berguna untuk:
        - Logging terstruktur ke file audit
        - Response API endpoint
        - Ekspor laporan digital

    Format JSON yang dihasilkan mengikuti skema berikut (per APD yang hilang):
        {
          "aktivitas"      : str  — Nama aktivitas K3
          "item_apd"       : str  — Nama item APD yang tidak terdeteksi
          "status_deteksi" : str  — Selalu "Tidak Terdeteksi"
          "nilai_l"        : int  — Likelihood (1–5)
          "nilai_s"        : int  — Severity (1–5)
          "nilai_tr"       : int  — Total Risk = L × S
          "tingkat_risiko" : str  — "Kecil (1-4)" / "Sedang (5-12)" / "Besar (15-25)"
          "catatan_risiko" : str  — Narasi korelasi aktivitas–APD dari Ground Truth
        }

    Args:
        activity   (str) : Nama aktivitas K3 (contoh: "Pengecoran Lantai").
        ibprp_row  (dict): Satu elemen dari output evaluate_ibprp_risk().
                           Harus memiliki kunci: 'ppe', 'status_deteksi',
                           'L', 'S', 'TR', 'risk_level', 'catatan_risiko'.

    Returns:
        str: String JSON yang diformat dengan indentasi 2 spasi (pretty-printed),
             siap untuk logging atau disimpan ke file.

    Contoh output:
        {
          "aktivitas": "Pengecoran Lantai",
          "item_apd": "helmet",
          "status_deteksi": "Tidak Terdeteksi",
          "nilai_l": 3,
          "nilai_s": 4,
          "nilai_tr": 12,
          "tingkat_risiko": "Sedang (5-12)",
          "catatan_risiko": "Risiko fatal akibat kepala tertimpa material coran ..."
        }
    """
    # Mapping label tingkat risiko ke format dengan rentang TR
    # sesuai spesifikasi output JSON yang divalidasi pakar
    level_label_map = {
        "Kecil":  "Kecil (1-4)",
        "Sedang": "Sedang (5-12)",
        "Besar":  "Besar (15-25)",
    }
    risk_level  = ibprp_row.get("risk_level", "")
    level_label = level_label_map.get(risk_level, risk_level)

    payload = {
        "aktivitas":      activity,
        "item_apd":       ibprp_row.get("ppe", ""),
        "status_deteksi": ibprp_row.get("status_deteksi", "Tidak Terdeteksi"),
        "nilai_l":        ibprp_row.get("L", 0),
        "nilai_s":        ibprp_row.get("S", 0),
        "nilai_tr":       ibprp_row.get("TR", 0),
        "tingkat_risiko": level_label,
        "catatan_risiko": ibprp_row.get("catatan_risiko", "-"),
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)
