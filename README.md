# Messy Shoe CV — Klasifikasi Kondisi Susunan Sepatu

Proyek Computer Vision untuk mendeteksi dan mengklasifikasikan kondisi susunan rak sepatu menjadi dua kelas:
1. **`berantakan`** (sepatu berserakan, tidak tertata pada slot rak)
2. **`tidak_berantakan` / rapi** (sepatu tersusun rapi atau slot rak kosong tertata)

---

## 📌 Ringkasan Eksekutif & Hasil Akurasi

Pada iterasi **Model 2.0**, pendekatan *handcrafted features* tradisional (RGB & GLCM) dari Model 1.0 ditinggalkan sepenuhnya dan digantikan oleh arsitektur representasi **YOLOv8**. Kami membangun dan mengevaluasi 2 pendekatan model utama:

| Model | Arsitektur | 5-Fold CV Accuracy | Holdout Test Accuracy | Precision | Recall | F1-Score | Ukuran File | Latensi Inferensi (CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 2.0** | **YOLO Deep Embeddings + Logistic Regression** | **94.29%** | **100.00%** | **100%** | **100%** | **100%** | **~9.5 KB** | ~454 ms |
| **Pure YOLO** | **End-to-End YOLOv8n-cls** | **97.2% – 100%** | **100.00%** | **100%** | **100%** | **100%** | **~14.5 MB** | ~457 ms |

> [!NOTE]
> Kedua model berhasil mencapai **akurasi 100%** pada data holdout test independen (36 citra uji yang dipisahkan secara ketat menggunakan *Stratified Group Split* untuk mencegah kebocoran data / *data leakage*).

---

## 🏗️ Dua Pendekatan Model yang Tersedia

### 1. Model 2.0 (YOLO Deep Features + Machine Learning)
* **File Model:** `model2.0/outputs/models/best_model2_ml.pkl`
* **Cara Kerja:** Citra diekstraksi menggunakan backbone convolutional pretrained YOLOv8 menjadi representasi laten berdimensi 256 (*deep latent embeddings*). Vektor fitur kemudian dinormalisasi dengan `StandardScaler` dan diklasifikasikan menggunakan `LogisticRegression`.
* **Kandidat Model ML yang Diuji (5-Fold CV):**
  * **Logistic Regression (Terbaik):** **94.29%** (±3.12%)
  * Extra Trees: 94.29% (±4.63%)
  * Random Forest: 93.65% (±5.32%)
  * SVM (RBF): 92.40% (±5.51%)
  * K-NN (k=5): 91.15% (±3.63%)
  * Gradient Boosting: 87.98% (±3.62%)
* **Kelebihan:** Ukuran file model classifier sangat kecil (~9.5 KB), training ulang sangat instan (< 5 detik) saat menambah data baru tanpa perlu melatih ulang neural network.

### 2. Pure YOLO (End-to-End Deep Learning Classification)
* **File Model:** `model2.0/outputs/models/pure_yolo_best.pt`
* **Cara Kerja:** Model arsitektur `yolov8n-cls` dilatih secara murni *end-to-end* langsung dari piksel citra masukan berukuran 256×256 menuju probabilitas label kelas tanpa pipeline tabular perantara.
* **Kelebihan:** Implementasi inferensi sangat ringkas (*single file deployment*), langsung mengembalikan probabilitas / *confidence score* bawaan Ultralytics, dan representasi visual dioptimalkan langsung untuk domain citra rak sepatu.

---

## 📊 Matriks Konfusi (Holdout Test — 36 Citra)

Kedua model menghasilkan matriks konfusi sempurna pada data uji independen:

```text
               Prediksi Berantakan   Prediksi Tidak Berantakan
Aktual Berantakan            20                          0
Aktual Tidak Berantakan       0                         16
```
* **Akurasi**: 36 / 36 (100.00%)
* **False Positive (FP)**: 0
* **False Negative (FN)**: 0

---

## 📁 Struktur Direktori & Artefak Proyek

```text
messy-shoe-cv/
├── dataset/                         # Dataset citra mentah (194 citra)
│   ├── berantakan/                  # 97 citra sepatu berantakan
│   └── rapi/                        # 97 citra sepatu rapi / slot rak kosong
├── model2.0/                        # Ruang kerja Model 2.0
│   ├── citra-sepatu.ipynb           # Notebook utama (seluruh pipeline tereksekusi)
│   ├── dataset_yolo_cls/            # Dataset berformat klasifikasi YOLO (train & val)
│   ├── yolov8n-cls.pt               # Bobot dasar YOLOv8-cls
│   └── outputs/
│       ├── features/
│       │   ├── dataset_splits.csv   # Pembagian data Train vs Holdout Test (anti-leakage)
│       │   └── yolo_features.csv    # 256 deep feature embeddings dari seluruh citra
│       └── models/
│           ├── best_model2_ml.pkl   # Model Machine Learning terbaik (.pkl)
│           ├── pure_yolo_best.pt    # Bobot model Pure YOLOv8n-cls (.pt)
│           └── yolo_runs/           # Log pelatihan YOLO per epoch
├── venv/                            # Virtual environment Python (Torch + Ultralytics)
├── README.md                        # Dokumentasi resmi proyek (file ini)
└── yolov8n-cls.pt                   # Salinan root pretrained YOLO
```

---

## 🚀 Panduan Penggunaan untuk Tim (Quickstart)

### 1. Aktivasi Virtual Environment
Pastikan menggunakan virtual environment proyek di PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& d:\laragon\www\messy-shoe-cv\venv\Scripts\Activate.ps1
```

---

### 2. Inferensi Menggunakan Model 2.0 (`.pkl`)
Gunakan kombinasi ekstraktor YOLO dan model Scikit-Learn:

```python
import joblib
from pathlib import Path
from ultralytics import YOLO

# 1. Muat ekstraktor dan model classifier
extractor = YOLO("model2.0/yolov8n-cls.pt")
classifier = joblib.load("model2.0/outputs/models/best_model2_ml.pkl")

# 2. Ekstraksi feature embedding dari citra uji
image_path = "dataset/berantakan/24-006 Berantakan 1.jpg"
embedding = extractor.embed(image_path, verbose=False)[0].cpu().numpy().reshape(1, -1)

# 3. Prediksi
class_names = ["berantakan", "tidak_berantakan"]
pred_idx = classifier.predict(embedding)[0]
pred_proba = classifier.predict_proba(embedding)[0]

print(f"Hasil Prediksi : {class_names[pred_idx]}")
print(f"Probabilitas   : Berantakan ({pred_proba[0]*100:.2f}%) | Rapi ({pred_proba[1]*100:.2f}%)")
```

---

### 3. Inferensi Menggunakan Pure YOLO (`.pt`)
Implementasi paling ringkas (cukup 1 library `ultralytics`):

```python
from ultralytics import YOLO

# 1. Muat model YOLO hasil training
model = YOLO("model2.0/outputs/models/pure_yolo_best.pt")

# 2. Prediksi langsung dari citra
image_path = "dataset/berantakan/24-006 Berantakan 1.jpg"
result = model.predict(image_path, verbose=False)[0]

label = result.names[result.probs.top1]
confidence = result.probs.top1conf.item()

print(f"Hasil Prediksi : {label}")
print(f"Confidence     : {confidence * 100:.2f}%")
```

---

### 4. Menjalankan Ulang Notebook `citra-sepatu.ipynb`
Seluruh cell notebook telah teroptimasi dengan *caching* fitur dan bobot terlatih sehingga dapat dijalankan ulang dalam waktu kurang dari 30 detik:

```powershell
python -m nbconvert --to notebook --execute "model2.0/citra-sepatu.ipynb" --inplace --ExecutePreprocessor.kernel_name=shoe_venv --ExecutePreprocessor.timeout=600
```

---

## 💡 Rekomendasi Deployment untuk Tim

1. **Gunakan Pure YOLO (`pure_yolo_best.pt`) jika:**
   - Aplikasi backend / API berbasis Flask/FastAPI/NodeJS membutuhkan implementasi paling sederhana tanpa dependensi `joblib` / `scikit-learn`.
   - Memerlukan nilai *confidence score* instan untuk antarmuka pengguna.
2. **Gunakan Model 2.0 (`best_model2_ml.pkl`) jika:**
   - Ingin menambahkan data citra baru secara dinamis dan melatih ulang model secara instan (*incremental retraining*) tanpa beban GPU/CPU tinggi.
