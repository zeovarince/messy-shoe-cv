# PRD — Ekstraksi Fitur Citra & Klasifikasi (Tugas Mingdep)

| | |
|---|---|
| **Kelompok** | (isi nama anggota) |
| **Format kerja** | Python, Jupyter Notebook (`.ipynb`) |
| **Acuan rumus** | Hallur & Gavade (2025), *Image feature extraction techniques: A comprehensive review*, Franklin Open 12, 100366 |
| **Acuan alur klasifikasi** | Tugas audio sebelumnya (bandingkan model baseline, pakai yang terbaik) |

---

## 1. Ringkasan

Membangun pipeline yang mengubah dataset citra menjadi tabel fitur (CSV), lalu mengklasifikasikannya. Dataset terdiri dari dua kelas: **berantakan** dan **tidak berantakan** (masing-masing ±100 citra). Fitur yang diekstrak mengikuti rumus pada jurnal acuan: statistik warna per channel dan tekstur GLCM dari citra grayscale.

## 2. Tujuan

1. Mengekstrak fitur warna (mean, std, skewness) per channel RGB dan menyimpannya ke CSV.
2. Mengubah citra ke grayscale dan mengekstrak fitur tekstur GLCM (energi, entropi, kontras, korelasi, homogenitas).
3. Menghasilkan CSV final berisi 14 fitur + label.
4. Membandingkan 5 model baseline, memilih yang terbaik, lalu mengevaluasinya dengan confusion matrix dan accuracy.

## 3. Ruang lingkup

**Termasuk:** ekstraksi fitur, pembuatan CSV, klasifikasi biner, perbandingan 5 model, confusion matrix, accuracy, laporan singkat.

**Tidak termasuk:** deep learning/CNN, augmentasi data, deployment.

## 4. Kebutuhan data

| Item | Spesifikasi |
|---|---|
| Jumlah kelas | 2: `berantakan`, `tidak_berantakan` |
| Jumlah citra | ±100 per kelas (±200 total) |
| Format | `.jpg`, `.jpeg`, `.png` |
| Struktur folder | `dataset/berantakan/*.jpg` dan `dataset/tidak_berantakan/*.jpg` (nama folder = label) |
| Ukuran citra | Seragam. Disarankan resize ke ukuran tetap (mis. 256×256) agar fitur tidak dipengaruhi resolusi |

**Yang perlu disiapkan:** dataset sudah terkumpul, dibersihkan dari duplikat/gambar rusak, dan jumlah kedua kelas relatif seimbang.

## 5. Spesifikasi fitur (14 fitur + label)

### 5.1 Fitur warna — dihitung per channel R, G, B dari citra RGB (9 fitur)

| Fitur | Eq. jurnal | Rumus | Kolom CSV |
|---|---|---|---|
| Mean | (2) | μ = 1/(M×N) ΣΣ C(i,j) | `mean_R`, `mean_G`, `mean_B` |
| Std | (3) | σ = √(1/N Σ (x − μ)²) | `std_R`, `std_G`, `std_B` |
| Skewness | (4) | 1/(M×N) ΣΣ ((I − μ)/σ)³ | `skew_R`, `skew_G`, `skew_B` |

### 5.2 Fitur tekstur — dihitung dari GLCM citra grayscale (5 fitur)

Langkah: citra RGB → grayscale → GLCM → normalisasi (tiap elemen dibagi total elemen, sehingga P(i,j) adalah probabilitas) → hitung fitur.

| Fitur | Eq. jurnal | Rumus | Kolom CSV |
|---|---|---|---|
| Energy | (8) | ΣΣ P(i,j)² | `energy` |
| Entropy | (5) | −ΣΣ P(i,j) log₂ P(i,j) | `entropy` |
| Contrast | (6) | ΣΣ (i − j)² · P(i,j) | `contrast` |
| Correlation | (7) | ΣΣ (i−μᵢ)(j−μⱼ)P(i,j) / (σᵢσⱼ) | `correlation` |
| Homogeneity | (9) | ΣΣ P(i,j) / (1 + \|i − j\|) | `homogeneity` |

> Entropi dan kontras disebut dua kali di soal (poin GLCM dan poin "tambah 1 fitur"). Keduanya fitur yang sama, dihitung sekali, total tetap **14 fitur**.

### 5.3 Parameter GLCM (keputusan kelompok, jurnal tidak menetapkan)

| Parameter | Nilai default |
|---|---|
| Level abu-abu | 256 |
| Jarak (distance) | 1 |
| Sudut | 0°, 45°, 90°, 135° (fitur dirata-rata) |
| Symmetric | True |
| Normalisasi | True (jumlah semua elemen = 1) |
| Konversi grayscale | Luminance standar (0.299 R + 0.587 G + 0.114 B) |

### 5.4 Catatan implementasi penting

- **Std** memakai pembagi N (populasi), bukan N−1 → `numpy.std` default (`ddof=0`).
- **Skewness** tanpa koreksi sampel → sama dengan `scipy.stats.skew` default.
- **Energy** jurnal = ΣP² (ASM). Di scikit-image `graycoprops('energy')` = √ASM, jadi **hitung manual** atau pakai `'ASM'`.
- **Homogeneity** jurnal memakai |i−j|, scikit-image memakai (i−j)². **Hitung manual** dari matriks P agar sesuai jurnal.
- **Entropy** tidak tersedia di `graycoprops`, wajib manual.
- **Korelasi** tidak terdefinisi jika σ = 0 (citra seragam). Tentukan nilai pengganti (mis. 1.0) dan tuliskan di laporan.

## 6. Output yang dihasilkan

| File | Isi |
|---|---|
| `fitur_warna.csv` | `filename`, 9 fitur warna, `label` (hasil tahap 1) |
| `fitur_lengkap.csv` | `filename`, 14 fitur, `label` (hasil akhir) |

Kolom `filename` hanya sebagai ID dan **tidak dipakai** saat klasifikasi.

Urutan kolom `fitur_lengkap.csv`:
`filename, mean_R, mean_G, mean_B, std_R, std_G, std_B, skew_R, skew_G, skew_B, energy, entropy, contrast, correlation, homogeneity, label`

## 7. Spesifikasi klasifikasi

### 7.1 Perbandingan 5 model baseline

Mengikuti alur tugas audio: semua model dievaluasi dengan **Stratified 5-fold cross-validation** pada seluruh data, metrik **accuracy** (rata-rata ± std).

| No | Model | Konfigurasi baseline |
|---|---|---|
| 1 | SVM (RBF) | `StandardScaler` + `SVC(C=10, gamma=0.01)` |
| 2 | Logistic Regression | `StandardScaler` + `LogisticRegression(max_iter=5000, C=0.5)` |
| 3 | Random Forest | `RandomForestClassifier(500)` |
| 4 | Extra Trees | `ExtraTreesClassifier(500)` |
| 5 | K-NN (k=5) | `StandardScaler` + `KNeighborsClassifier(5, weights='distance')` |

> Samakan 5 model ini dengan yang dipakai di tugas audio. Jika berbeda, ganti sesuai tugas audio.

Aturan: `StandardScaler` **wajib** di dalam pipeline untuk SVM, Logistic Regression, dan K-NN (skala mean/std 0–255 jauh berbeda dari korelasi −1..1). Model berbasis pohon tidak memerlukannya.

### 7.2 Pemilihan dan evaluasi model terbaik

1. Pilih model dengan rata-rata accuracy CV tertinggi.
2. Bagi data 80/20 (`train_test_split`, `stratify=y`, `random_state` tetap).
3. (Opsional) Tuning hyperparameter model terbaik dengan `GridSearchCV` hanya pada data train.
4. Evaluasi akhir pada data test: **accuracy**, **confusion matrix**, dan `classification_report` (precision, recall, F1).

### 7.3 Visualisasi yang diminta

- Bar chart perbandingan accuracy CV 5 model.
- Confusion matrix model terbaik (heatmap, label kelas terbaca).

## 8. Kebutuhan teknis

| Kebutuhan | Keterangan |
|---|---|
| Python | 3.9+ |
| Library | `numpy`, `pandas`, `matplotlib`, `seaborn`, `Pillow`, `scikit-image`, `scikit-learn`, `tqdm` |
| Lingkungan | Jupyter / Google Colab (jika Colab, mount Google Drive tempat dataset) |
| Reproducibility | Tetapkan `RANDOM_STATE` (mis. 42) dan pakai di semua split/model |

## 9. Struktur notebook

1. Judul, anggota kelompok, tujuan
2. Import library & konfigurasi (path dataset, ukuran citra, parameter GLCM)
3. Muat daftar citra + label
4. Ekstraksi fitur warna (mean, std, skewness per channel) → simpan `fitur_warna.csv`
5. Konversi grayscale (tampilkan contoh sebelum/sesudah)
6. Ekstraksi fitur GLCM (energi, entropi, kontras, korelasi, homogenitas)
7. Gabung fitur → simpan `fitur_lengkap.csv`
8. Eksplorasi singkat (cek missing value, distribusi kelas, statistik fitur)
9. Perbandingan 5 model baseline (CV + bar chart)
10. Evaluasi model terbaik (accuracy, confusion matrix, classification report)
11. Kesimpulan & analisis

## 10. Kriteria penerimaan (checklist)

- [ ] Semua citra terbaca, tidak ada error/NaN pada CSV
- [ ] `fitur_warna.csv` berisi 9 fitur + label
- [ ] `fitur_lengkap.csv` berisi 14 fitur + label
- [ ] Rumus fitur sesuai Eq. (2)–(9) jurnal (energy = ΣP², homogeneity memakai |i−j|)
- [ ] 5 model dibandingkan dengan CV yang sama
- [ ] Model terbaik dipilih berdasarkan hasil perbandingan, bukan dipilih di awal
- [ ] Confusion matrix dan accuracy ditampilkan
- [ ] Notebook bisa dijalankan dari awal sampai akhir (Restart & Run All)

## 11. Risiko & catatan

| Risiko | Mitigasi |
|---|---|
| Data kecil (±200), hasil CV bervariasi | Pakai stratified k-fold dan laporkan mean ± std, bukan satu angka |
| Fitur hasil `graycoprops` berbeda dari jurnal | Hitung GLCM feature manual dari matriks P yang sudah dinormalisasi |
| Resolusi citra beragam memengaruhi fitur | Resize ke ukuran tetap sebelum ekstraksi |
| Kelas tidak seimbang | Cek jumlah per kelas; jika timpang, laporkan juga precision/recall/F1 |
| Kebocoran data (scaler di-fit pada seluruh data) | Letakkan `StandardScaler` di dalam `Pipeline` |
| Citra "berantakan" vs "tidak berantakan" mirip secara warna | Andalkan fitur tekstur (entropi, kontras) dan bahas di analisis |

## 12. Pembagian tugas kelompok

| Peran | Tugas | Penanggung jawab |
|---|---|---|
| Data | Kumpulkan dan rapikan dataset, struktur folder, cek kualitas | |
| Fitur warna | Mean, std, skewness per channel → `fitur_warna.csv` | |
| Fitur tekstur | Grayscale, GLCM, 5 fitur tekstur → `fitur_lengkap.csv` | |
| Klasifikasi | Perbandingan 5 model, confusion matrix, accuracy | |
| Laporan | Analisis hasil, kesimpulan, kompilasi notebook | |

## 13. Deliverable

1. Notebook `.ipynb` lengkap (dijalankan, output terlihat)
2. `fitur_warna.csv` dan `fitur_lengkap.csv`
3. Ringkasan hasil: tabel accuracy 5 model, model terpilih, confusion matrix, dan analisis singkat
