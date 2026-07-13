"""
core/__init__.py — Package Initializer Modul Inti Sistem
=========================================================

Package 'core' adalah jantung dari sistem SAFEWATCH.
Berisi dua sub-modul utama yang menerapkan prinsip Separation of Concerns (SoC):

    ┌─────────────────────────────────────────────────────────────┐
    │  core/                                                      │
    │  ├── detector.py     → Lapisan Persepsi (YOLOv11s + OpenCV) │
    │  └── ibprp_engine.py → Lapisan Kognisi  (Rule-based K3)     │
    └─────────────────────────────────────────────────────────────┘

Catatan Arsitektur (Design Research Methodology — DRM):
    - Setiap modul di package ini bersifat MURNI dan STATELESS:
      tidak ada dependensi terhadap Flask, session, atau HTTP request.
    - Modul-modul ini dapat diuji secara independen (unit test)
      tanpa perlu menjalankan server Flask.
    - app.py bertindak sebagai 'Orchestrator' yang menghubungkan
      dua modul ini ke lapisan HTTP (Flask routing).

Digunakan di skripsi untuk:
    Bab III — Sub-bab Perancangan Model (detector.py)
    Bab III — Sub-bab Perancangan Penilaian Risiko IBPRP (ibprp_engine.py)
"""
