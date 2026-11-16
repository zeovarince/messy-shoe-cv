# Klasifikasi Kondisi Susunan Sepatu — Hybrid Feature Extraction

Notebook: `citra-sepatu.ipynb`

- Jumlah citra valid: 194
- Kelas: berantakan, tidak_berantakan
- Fitur baseline: 33 (RGB 9, GLCM 5, HOG 9, LBP 10)
- Fitur YOLO: tidak tersedia (Mode 2 tidak aktif)
- Model yang dibandingkan: SVM (RBF), Random Forest, Extra Trees, K-NN (k=5), Logistic Regression, Gradient Boosting, LightGBM
- Model terbaik: SVM (RBF)
- Eksperimen fitur terbaik: E5 (RGB + GLCM + HOG + LBP)
- CV mean accuracy: 0.8194 +/- 0.1052
- Test accuracy: 0.7179
- Feature selection: tidak
- Parameter terbaik: {'model__C': 100, 'model__gamma': 0.01}

## Struktur output
```text
outputs/
├── features/     # CSV per kelompok fitur, fitur_lengkap, fitur_hybrid, Excel
├── evaluations/  # comparison_models, comparison_features, report, confusion matrix
└── models/       # best_classifier.pkl
```

## Keterbatasan
- Dataset kecil dan label tingkat gambar bersifat subjektif.
- Duplikat file dilaporkan, tidak dihapus otomatis.
- Fitur YOLO hanya tersedia bila model berkelas `shoe` tersedia; COCO tidak memilikinya.
- Split acak dapat mencampur foto dari sepatu yang sama; `SPLIT_BY_GROUP=True` tersedia sebagai opsi.
