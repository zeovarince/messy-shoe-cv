# Model 2.0: Klasifikasi Kondisi Susunan Sepatu (YOLO Deep Features vs Pure YOLO)

Proyek ini merupakan pembaruan arsitektur model klasifikasi kondisi susunan sepatu (`berantakan` vs `tidak_berantakan`) yang meninggalkan metode fitur *handcrafted* tradisional dan beralih ke representasi *Deep Learning* menggunakan YOLO.

---

## 1. Perubahan Paradigma (Model 1.0 vs Model 2.0)

| Karakteristik | Model 1.0 (PRD Base) | Model 2.0 (Deep Learning Shift) |
|---|---|---|
| **Ekstraksi Fitur** | Manual / Handcrafted (9 Fitur RGB + 5 Fitur Tekstur GLCM) | **Dihapus / Dilupakan**. Digantikan oleh representasi laten (*deep embeddings*) YOLO |
| **Dimensi Fitur** | 14 kolom tabular | Fitur representasi laten tingkat tinggi dari backbone YOLO |
| **Pendekatan Pemodelan** | Baseline ML tradisional pada fitur 14 kolom | **Dua Pendekatan**: (1) **Model 2.0** (YOLO Embeddings + ML Classifier) dan (2) **Pure YOLO** (End-to-End Classification) |
| **Evaluasi** | Stratified 5-Fold CV + Holdout Test | Stratified Group Split (80/20) + 5-Fold CV + Head-to-Head Comparison |

---

## 2. Dua Pendekatan yang Dibandingkan

### A. Model 2.0 (YOLO Feature Extractor + Machine Learning)
1. **Feature Extraction**: Citra diproses melalui model YOLO pretrained (`yolov8n-cls`) untuk mengekstrak vektor representasi semantik laten. Fitur ini menangkap kerapian, orientasi spasial, dan kepadatan objek sepatu tanpa dipengaruhi variasi warna minor.
2. **Model Selection**: Dilakukan evaluasi 6 model machine learning dengan Stratified 5-Fold Cross Validation:
   - **Support Vector Classifier (SVM RBF)** dengan `StandardScaler`
   - **Logistic Regression** dengan `StandardScaler`
   - **Random Forest Classifier**
   - **Extra Trees Classifier**
   - **K-Nearest Neighbors (k=5)** dengan `StandardScaler`
   - **Gradient Boosting Classifier**
3. **Penyimpulan Model**: Model dengan rata-rata akurasi CV tertinggi dipilih sebagai model final, dilatih pada seluruh data train, dan diuji pada 36 citra holdout test.

### B. Pure YOLO (End-to-End Deep Learning Classifier)
1. Model YOLOv8-cls dilatih secara murni *end-to-end* langsung dari citra piksel sepatu menuju kelas target (`berantakan` vs `tidak_berantakan`).
2. Tidak melibatkan pemodelan tabular maupun model ML eksternal.
3. Seluruh bobot konvolusi dan pooling dilatih/di-*fine-tune* selama 25 epoch pada dataset sepatu.

### C. Head-to-Head Comparison (Model 2.0 vs Pure YOLO)
Kedua model diuji pada set holdout test yang identik (36 citra, 13 grup tanpa kebocoran data) untuk mengukur:
- Akurasi (Accuracy)
- Macro Precision, Recall, F1-Score
- Matriks Konfusi (Confusion Matrix) berdampingan
- Waktu Inferensi per Citra (Latency)
- Analisis sampel yang menghasilkan perbedaan prediksi (*discrepancy analysis*)

---

## 3. Struktur Folder & File

```text
model2.0/
├── citra-sepatu.ipynb            # Notebook interaktif lengkap (seluruh tahapan mandiri di sini)
├── dataset_yolo_cls/             # Folder dataset terstruktur untuk Pure YOLO (train & val)
│   ├── train/ (berantakan, tidak_berantakan)
│   └── val/   (berantakan, tidak_berantakan)
├── outputs/
│   ├── features/                 # Penyimpanan metadata split dan fitur laten YOLO (CSV)
│   ├── models/                   # Penyimpanan bobot model terbaik
│   └── evaluations/              # Hasil grafik evaluasi dan matriks konfusi
└── README.md                     # Dokumentasi teknis proyek
```

---

## 4. Cara Menjalankan

Seluruh pipeline bersifat mandiri (*self-contained*) dan dijalankan langsung dari Jupyter Notebook:

1. Buka file notebook:
   `model2.0/citra-sepatu.ipynb`
2. Pilih kernel Python dari environment yang sudah aktif (`venv`).
3. Jalankan menu **Kernel -> Restart & Run All**.

Seluruh tahapan mulai dari pembagian dataset (anti-leakage), ekstraksi representasi laten YOLO, visualisasi sebaran fitur PCA/t-SNE, pelatihan & seleksi 6 model ML, pelatihan Pure YOLO end-to-end, hingga visualisasi perbandingan komparatif akan dieksekusi secara otomatis.
