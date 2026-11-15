# PROMPT — Poles & Sesuaikan Notebook dengan PRD

> **Cara pakai:** lampirkan 3 file ke AI (Claude / ChatGPT / dll.):
> 1. `PRD_Ekstraksi_Fitur_Citra.md`
> 2. `Tugas_Mingdep_Ekstraksi_Fitur_Citra.ipynb` (notebook base)
> 3. Jurnal PDF (Hallur & Gavade, 2025), plus notebook audio kemarin jika ada
>
> Lalu isi bagian **[ISI DULU]** di bawah, dan tempel seluruh isi di antara garis `=====` sebagai pesan.

---

## [ISI DULU] — data kelompok

- Nama kelompok / anggota: `...`
- Lokasi dataset: `...` (contoh: `/content/drive/MyDrive/dataset`)
- Lingkungan: `Google Colab` / `Jupyter lokal` (pilih satu)
- Lima model di tugas audio (jika berbeda dari notebook base): `...` (kosongkan jika sama)
- Hal lain yang diminta dosen (format laporan, jumlah fitur tambahan, dll.): `...`

---

=====

Kamu adalah asisten data science yang membantu mahasiswa menyelesaikan tugas kelompok **ekstraksi fitur citra dan klasifikasi** dengan Python (Jupyter Notebook).

## Konteks

Kami mengklasifikasikan citra ke dua kelas: **berantakan** dan **tidak_berantakan** (±100 citra per kelas). Fitur diekstrak mengikuti rumus pada jurnal terlampir (Hallur & Gavade, 2025, Franklin Open 12, 100366). Alur klasifikasi mengikuti tugas audio kami sebelumnya: membandingkan 5 model baseline dengan Stratified 5-fold cross-validation, lalu memakai model yang terbaik.

Terlampir:
- **PRD** (sumber kebenaran untuk kebutuhan dan kriteria penerimaan)
- **Notebook base** (`.ipynb`) yang akan dipoles
- **Jurnal PDF** (sumber kebenaran untuk rumus)

## Tugasmu

Poles notebook base dan sesuaikan dengan PRD, tanpa mengubah tujuan dan struktur utamanya. Hasil akhir harus bisa dijalankan dari awal sampai akhir (Restart & Run All) oleh mahasiswa yang baru belajar.

## Aturan yang TIDAK boleh dilanggar

1. **Rumus fitur harus tetap sesuai jurnal** (Eq. 2-9):
   - Std memakai pembagi N (populasi), skewness tanpa koreksi sampel.
   - Energy = ΣP(i,j)², **bukan** akar dari ASM.
   - Homogeneity memakai penyebut `1 + |i−j|`, **bukan** `(i−j)²`.
   - Entropy memakai log basis 2.
   - Fitur GLCM tetap dihitung **manual** dari matriks P yang sudah dinormalisasi. Jangan diganti dengan `graycoprops` untuk energy dan homogeneity.
2. **Total fitur tetap 14** + label, dengan urutan kolom:
   `filename, mean_R, mean_G, mean_B, std_R, std_G, std_B, skew_R, skew_G, skew_B, energy, entropy, contrast, correlation, homogeneity, label`
   Kolom `filename` tidak dipakai saat klasifikasi.
3. **Dua file CSV** tetap dihasilkan: `fitur_warna.csv` (9 fitur + label) dan `fitur_lengkap.csv` (14 fitur + label).
4. **Lima model baseline** dibandingkan dengan **Stratified 5-fold CV yang sama**, dan model terbaik dipilih dari hasil perbandingan (bukan ditentukan di awal). `StandardScaler` harus berada di dalam `Pipeline` untuk SVM, Logistic Regression, dan K-NN (cegah data leakage).
5. **Reproducible:** satu `RANDOM_STATE` dipakai di semua split dan model.
6. **Jangan mengarang hasil.** Jangan menulis angka accuracy, output, atau kesimpulan yang tidak benar-benar berasal dari hasil eksekusi. Kolom kesimpulan dibiarkan sebagai template untuk diisi kelompok.
7. **Jangan menambah fitur atau model baru** di luar PRD. Ide tambahan boleh disarankan di bagian akhir jawaban, tapi tidak dimasukkan ke notebook kecuali ditandai jelas sebagai "(opsional)".

## Yang perlu dikerjakan

### A. Cek kesesuaian dengan PRD
Telusuri **checklist kriteria penerimaan** (PRD bagian 10) satu per satu dan laporkan statusnya (sudah / belum / sebagian). Perbaiki yang belum.

### B. Kualitas kode
- Tangani kasus error umum dengan pesan yang jelas: folder dataset kosong atau path salah, gambar rusak/tidak terbaca (lewati dan catat di daftar), kelas hanya satu, nilai NaN.
- Hindari duplikasi; satu fungsi untuk satu tugas; semua parameter di satu sel konfigurasi.
- Komentar dalam Bahasa Indonesia, singkat, dan merujuk ke nomor Eq. jurnal bila relevan.
- Pastikan perhitungan GLCM cukup cepat untuk ±200 citra (hindari komputasi berulang di dalam loop).
- Kode tetap sederhana dan mudah dibaca mahasiswa. Jangan over-engineering (tanpa class rumit atau framework tambahan).

### C. Kualitas notebook
- Setiap bagian diawali sel markdown yang menjelaskan **apa** yang dilakukan dan **kenapa**, singkat.
- Rumus ditulis dalam LaTeX dengan nomor Eq. jurnal.
- Tabel hasil dan grafik berjudul jelas, label sumbu terbaca, kelas bernama (bukan 0/1).
- Sel "sanity check" GLCM (manual vs `graycoprops`) dipertahankan, dengan penjelasan kenapa homogeneity memang berbeda.
- Tambahkan sel ringkasan di akhir: tabel accuracy 5 model (mean ± std), nama model terbaik, accuracy holdout, dan lokasi file CSV.
- Bagian "Kesimpulan & Analisis" tetap berupa **template kosong** dengan pertanyaan pemandu.

### D. Penyesuaian khusus
- Jika saya mengisi daftar model audio yang berbeda di atas, ganti `baseline_models` sesuai itu dan jelaskan perubahannya.
- Jika lingkungan = Google Colab, aktifkan dan rapikan sel mount Drive; jika lokal, hapus bagian Colab.
- Jika ada permintaan dosen tambahan, jelaskan dulu bagaimana kamu akan memenuhinya sebelum mengubah notebook.

## Format jawaban

1. **Ringkasan perubahan** (maksimal 10 poin, tiap poin satu kalimat).
2. **Hasil cek PRD:** tabel checklist kriteria penerimaan dengan status.
3. **Notebook final** lengkap dalam format `.ipynb` (jika kamu bisa membuat file), atau sel demi sel yang bisa langsung disalin dengan penanda `[MARKDOWN]` / `[CODE]` dan nomor urut.
4. **Asumsi dan pertanyaan:** hal yang kamu tebak atau tidak bisa pastikan (mis. parameter GLCM, kriteria pelabelan, model audio).
5. **Catatan eksekusi:** apakah notebook sudah benar-benar dijalankan. Jika belum karena tidak punya akses ke dataset, katakan terus terang, dan sebutkan hal yang paling mungkin error saat pertama kali dijalankan.

Jika ada informasi penting yang kurang, **ajukan maksimal 3 pertanyaan** di awal, lalu lanjutkan dengan asumsi yang kamu tulis jelas.

=====
