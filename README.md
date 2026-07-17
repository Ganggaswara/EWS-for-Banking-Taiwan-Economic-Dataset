# 🏦 EWS — Early Warning System Prediksi Kebangkrutan Bank

Sistem *Early Warning* berbasis Machine Learning untuk memprediksi risiko kebangkrutan bank berdasarkan rasio-rasio keuangan, membandingkan performa **Logistic Regression**, **Random Forest**, dan **XGBoost** pada dataset Taiwan Economic Journal — lengkap dengan pipeline zero-leakage, feature selection berbasis permutation importance, dan aplikasi web interaktif untuk simulasi prediksi.

---

## ✨ Fitur Utama

- 🔬 **3 Model Dibandingkan** — Logistic Regression, Random Forest, dan XGBoost dievaluasi secara paralel
- 🧹 **Pipeline Zero-Leakage** — seluruh preprocessing (imputasi, capping outlier, filter korelasi, seleksi fitur) di-fit hanya pada data train, mencegah data leakage
- 🎯 **Feature Selection via Permutation Importance** — reduksi dimensi berbasis kontribusi nyata fitur terhadap prediksi (bukan seleksi manual), diambil hingga kumulatif 90% importance
- ⚖️ **Threshold Optimization (F2-score)** — threshold keputusan tiap model dioptimasi untuk memaksimalkan Recall, karena melewatkan bank yang benar-benar bangkrut jauh lebih mahal daripada false alarm
- 🖥️ **Aplikasi Web Interaktif (Streamlit)** — input manual rasio keuangan → prediksi probabilitas kebangkrutan dari ketiga model sekaligus, lengkap visualisasi perbandingan
- 📦 **Model Artifacts Terpisah** — pipeline, statistik fitur, threshold, dan daftar fitur terpilih diekspor sebagai artefak siap-deploy

---

## 🏗️ Arsitektur Sistem

```
┌────────────────────────────────────────────────────────┐
│              NOTEBOOK RISET (UAS_final.ipynb)            │
│                                                            │
│  Dataset: Taiwan Economic Journal (rasio keuangan bank)   │
│         │                                                  │
│         ▼                                                  │
│  Custom Transformers (custom_transformers.py)              │
│  ┌────────────────────────────────────────────┐           │
│  │ MedianImputer      → isi missing value       │           │
│  │ IQRCapper           → capping outlier (1.5×IQR)│         │
│  │ CorrelationFilter   → drop fitur korelasi >0.95│         │
│  │ ImportanceSelector  → pilih fitur cum. 90%      │           │
│  │                        (permutation importance,  │           │
│  │                         scoring='recall')          │           │
│  └────────────────────────────────────────────┘           │
│         │                                                  │
│         ▼                                                  │
│  Training 3 Model + Threshold Tuning (F2-score)             │
│  • Logistic Regression   • Random Forest   • XGBoost         │
│         │                                                  │
│         ▼  (Bagian 10: Export Artefak)                      │
│  model_deploy/                                              │
│    ├── logreg_pipeline.pkl                                  │
│    ├── rf_pipeline.pkl                                       │
│    ├── xgb_pipeline.pkl                                      │
│    ├── feature_stats.csv                                     │
│    ├── thresholds.json                                       │
│    └── selected_features.json                                │
└────────────────────┬───────────────────────────────────────┘
                      ▼
┌────────────────────────────────────────────────────────┐
│               APLIKASI WEB (app.py — Streamlit)          │
│                                                            │
│  Sidebar Form → input manual rasio keuangan (11 fitur)     │
│         │                                                  │
│         ▼                                                  │
│  Load 3 model pipeline (cached)                             │
│         │                                                  │
│         ▼                                                  │
│  Prediksi probabilitas kebangkrutan per model                │
│  → bandingkan dengan threshold optimal masing-masing          │
│         │                                                  │
│         ▼                                                  │
│  Output: status (⚠️ Berisiko / ✅ Sehat) + bar chart          │
│  perbandingan probabilitas antar model                         │
└────────────────────────────────────────────────────────┘
```

---

## 🧪 Pipeline Preprocessing (Zero-Leakage)

Seluruh transformer custom dirancang sebagai scikit-learn `Pipeline` yang **hanya di-`fit()` pada data training**, lalu diterapkan (`transform()`) ke data test/inference tanpa mengintip statistik dari luar data train:

| Transformer | Fungsi |
|---|---|
| `MedianImputer` | Mengisi nilai kosong dengan median (dihitung dari train) |
| `IQRCapper` | Meng-capping outlier di luar rentang `Q1 - 1.5×IQR` s.d. `Q3 + 1.5×IQR` |
| `CorrelationFilter` | Membuang fitur dengan korelasi antar-fitur > 0.95 (menghindari multikolinearitas) |
| `ImportanceSelector` | Memilih subset fitur berdasarkan *permutation importance* kumulatif hingga 90%, dievaluasi dengan `scoring='recall'` menggunakan Random Forest screener |

> ⚠️ **Catatan teknis penting:** seluruh custom transformer wajib didefinisikan di `custom_transformers.py` (bukan langsung di notebook/`__main__`), karena `joblib`/`pickle` menyimpan referensi `module.ClassName`. Jika class didefinisikan di notebook, `app.py` tidak akan bisa unpickle model (`AttributeError: Can't get attribute 'X' on module '__main__'`).

---

## ⚖️ Threshold Optimization

Threshold keputusan tiap model **tidak menggunakan default 0.5**, melainkan hasil optimasi **F2-score** — metrik yang memberi bobot lebih besar pada Recall dibanding Precision. Rasionalnya: dalam konteks Early Warning System, melewatkan bank yang benar-benar akan bangkrut (*false negative*) jauh lebih berbahaya secara sistemik daripada memunculkan false alarm (*false positive*).

---

## 🖥️ Aplikasi Streamlit

- Form input di sidebar untuk 11 fitur rasio keuangan terpilih, dengan nilai default = median data training dan validasi rentang min–max
- Prediksi probabilitas kebangkrutan ditampilkan berdampingan untuk ketiga model, masing-masing dengan status **⚠️ Berisiko Bangkrut** / **✅ Sehat** berdasarkan threshold optimalnya
- Bar chart perbandingan probabilitas antar model (Plotly)
- Panel edukatif tentang proses seleksi fitur dan statistik masing-masing fitur

---

## 🛠️ Tech Stack

| Kategori | Library |
|---|---|
| **Data & Modeling** | `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn` |
| **Model Persistence** | `joblib` |
| **Visualisasi** | `plotly` |
| **Aplikasi Web** | `streamlit` |

---

## 📁 Struktur Proyek

```
.
├── UAS_final.ipynb              # Notebook riset lengkap: EDA → preprocessing →
│                                 # training 3 model → threshold tuning → export artefak
├── custom_transformers.py       # Custom sklearn transformers (wajib di-import kedua sisi)
├── app.py                       # Aplikasi Streamlit untuk prediksi interaktif
├── model_deploy/                # Artefak hasil export dari Bagian 10 notebook
│   ├── logreg_pipeline.pkl
│   ├── rf_pipeline.pkl
│   ├── xgb_pipeline.pkl
│   ├── feature_stats.csv
│   ├── thresholds.json
│   └── selected_features.json
├── requirements.txt
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Notebook Riset (opsional — untuk retrain model)
Jalankan `UAS_final.ipynb` dari awal hingga Bagian 10 untuk menghasilkan ulang folder `model_deploy/`.

### 3. Jalankan Aplikasi Streamlit
Pastikan folder `model_deploy/` berada di direktori yang sama dengan `app.py`, lalu:
```bash
streamlit run app.py
```

### 4. Gunakan Aplikasi
1. Isi form rasio keuangan di sidebar kiri (nilai default = median data training)
2. Klik **🔍 Prediksi**
3. Lihat hasil probabilitas kebangkrutan dari ketiga model beserta status dan interpretasinya

---

## 📊 Dataset

**Taiwan Economic Journal** — dataset rasio keuangan perusahaan/bank yang umum digunakan untuk riset prediksi kebangkrutan (bankruptcy prediction), berisi puluhan rasio keuangan (profitabilitas, likuiditas, leverage, dll.) dengan label status bangkrut/tidak.

---

## ⚠️ Disclaimer

Aplikasi ini adalah **simulasi akademik** untuk keperluan tugas akhir/UAS, bukan alat keputusan regulasi resmi. Prediksi tidak boleh dijadikan dasar tunggal pengambilan keputusan finansial atau regulasi terhadap institusi perbankan.

---

## 👤 Author

**I Gusti Bagus Pradnyana Gangga Swara**
Mahasiswa Ilmu Komputer, Universitas Pendidikan Ganesha
🔗 [GitHub](https://github.com/Ganggaswara) · [LinkedIn](https://www.linkedin.com/in/ganggaswara11)
