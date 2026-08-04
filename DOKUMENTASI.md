# Panduan Presentasi & Dokumentasi Sistem SAFEWATCH
*(Sistem Deteksi APD Berbasis YOLOv11s untuk Audit K3 Konstruksi)*

Dokumen ini disusun khusus untuk membantu Anda memahami arsitektur proyek dan mempersiapkan presentasi/sidang di depan dosen penguji. 

---

## 1. Konsep Utama & Nilai Jual (Value Proposition)
Saat presentasi, sangat penting untuk menekankan **dua hal utama** yang membuat sistem ini unggul dan dapat dipertanggungjawabkan secara akademis:
1. **AI sebagai "Mata"**: YOLOv11s digunakan semata-mata untuk deteksi visual (mencari objek orang dan APD pada gambar).
2. **Rule-Based System sebagai "Otak" Regulasi**: Penilaian risiko K3 tidak ditebak oleh AI, melainkan dihitung menggunakan mesin aturan pasti (deterministik) yang merujuk pada **PERMEN PUPR No. 10 Tahun 2021**. Sistem ini menghitung Total Risk (TR) = *Likelihood* (L) × *Severity* (S).

---

## 2. Alur Kerja Sistem (System Workflow)
Jelaskan alur ini saat mendemokan aplikasi ke dosen:

1. **Input Pengguna**: Pengguna mengunggah foto pekerja dan memilih jenis aktivitas K3 (misalnya: "Pengecoran Lantai"). Data ini dikirim ke server Flask (`app.py`).
2. **Deteksi Objek (AI)**: Server memanggil `core/detector.py`. Gambar akan diproses oleh model YOLOv11s kustom (`best.pt`). Model akan mendeteksi kelas yang ada di gambar (`person`, `helmet`, `vest`, `boots`, `gloves`, `glasses`).
3. **Pengecekan Kepatuhan (Rule-Based)**: 
   - Sistem melihat *tabel aturan* di `config.py` untuk aktivitas yang dipilih.
   - Sistem membandingkan: **Apa saja APD yang WAJIB** vs **Apa saja APD yang TERDETEKSI**.
   - Jika ada APD wajib yang *tidak terdeteksi*, maka APD tersebut dianggap **HILANG/TIDAK DIPAKAI**.
4. **Evaluasi IBPRP**: Fungsi `evaluate_ibprp_risk` (di `core/ibprp_engine.py`) akan mengambil nilai Kekerapan (L) dan Keparahan (S) untuk setiap APD yang hilang. Lalu menghitung **Total Risk = L × S**, dan mengelompokkannya (Kecil, Sedang, Besar).
5. **Output**: Pengguna diarahkan ke halaman Dashboard (`dashboard.html`) yang menampilkan gambar beranotasi bounding box serta tabel matriks risiko (IBPRP).

---

## 3. Penjelasan Struktur Kode (Arsitektur Separation of Concerns / SoC)

Sistem ini dirancang menggunakan arsitektur **Separation of Concerns (SoC)** yang memisahkan logika aplikasi menjadi beberapa bagian yang mandiri (modular). Ini menunjukkan pemahaman *Software Engineering* yang baik.

### A. `config.py` (Pusat Konfigurasi & Aturan)
Dosen sering bertanya soal parameter. Di file ini terdapat tiga poin krusial:
- **`CONFIDENCE_THRESHOLD = 0.227`** 
  *Penjelasan untuk Dosen:* "Angka ini bukan asal tebak, Pak/Bu. 0.227 adalah titik ekuilibrium (pertemuan optimal) pada kurva F1-Confidence hasil *training* model saya. Di titik ini, model memberikan keseimbangan terbaik antara *Precision* (tidak asal menebak) dan *Recall* (tidak ada objek yang terlewat)."
- **`VALIDATION_CONF_THRESHOLD = 0.60`**
  *Penjelasan untuk Dosen:* "Ini adalah lapisan keamanan ganda (Dual Filter). Sementara anotasi deteksi YOLO tetap pakai 0.227, sistem memiliki filter validasi kontekstual yang ketat di angka 0.60. Tujuannya agar false-positive dari foto non-konstruksi (seperti jilbab atau topi biasa yang mungkin keliru terdeteksi sebagai 'helmet' dengan confidence 0.35) akan langsung tertolak sebelum masuk ke perhitungan risiko."
- **`RULES_IBPRP` & Pemisahan Folder Hasil**
  *Penjelasan untuk Dosen:* "Sistem dirancang *Strict* dan fokus pada **hanya 3 aktivitas utama** (Pengecoran Lantai, Pemasangan Besi Lantai, Pemasangan Besi Kolom). Folder gambar juga dienkapsulasi dengan baik: `static/uploads/` khusus file mentah user, sedangkan `static/results/` khusus gambar hasil deteksi YOLO. Hal ini agar tidak ada kasus gambar asli tertukar dengan gambar beranotasi di dashboard."

### B. `app.py` (Orchestrator & Routing)
File ini adalah *entry point* dari aplikasi Flask. Tugas utamanya hanyalah sebagai pengatur lalu lintas (*orchestrator*), **tidak memuat logika AI atau K3**:
- Menerima request gambar dan menangani routing.
- **Melakukan Validasi Konteks Berlapis (Strict Mode):** Mengamankan sistem dengan mengecek apakah aktivitas valid (hanya dari 3 pilihan) dan apakah foto benar-benar memiliki 'person' + APD di atas batas validasi 0.60. Jika gagal, server melempar status error (HTTP 400).
- Menyerahkan tugas deteksi ke `core/detector.py`.
- Menyerahkan tugas hitung risiko ke `core/ibprp_engine.py`.
- Mengemas hasil dan menampilkannya di halaman web secara efisien.

### C. `core/detector.py` (Modul Persepsi AI)
File ini khusus mengurus *Computer Vision* (AI).
- Memuat file bobot model (`models/best.pt`) menggunakan *library* `ultralytics`.
- Mengeksekusi *inference* pada gambar untuk mencari objek APD dan pekerja.
- Menggambar *bounding box* (kotak warna-warni) menggunakan OpenCV.
- *Tidak tahu menahu soal aturan K3, hanya peduli soal melihat objek.*

### D. `core/ibprp_engine.py` (Modul Kognisi / Mesin Aturan K3)
File ini khusus mengurus evaluasi risiko K3.
- Mengambil data APD yang hilang dari AI, dan membandingkannya dengan `RULES_IBPRP`.
- Menghitung **Total Risk (TR) = L × S** untuk setiap pelanggaran.
- Mengelompokkan tingkat risiko (Kecil, Sedang, Besar).
- *Modul ini berisi "fungsi murni" (pure functions) yang bekerja secara deterministik tanpa campur tangan AI.*

