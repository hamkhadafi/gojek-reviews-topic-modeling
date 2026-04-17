# 📊 Topic Modeling Ulasan Pengguna Aplikasi Gojek
### Menggunakan Latent Dirichlet Allocation (LDA)

**Mata Kuliah:** SCST603106 — Analisis Data Tidak Terstruktur  
**Program Studi:** Statistika, FMIPA Universitas Indonesia  
**Nama:** Ilham Khadafi  
**NIM:** 2306260952  
**Kelas:** A  

---

## 📁 Struktur Folder

```
📦 UTS_TopicModeling_Gojek/
├── 📄 README.md                                          ← Dokumen ini
├── 📊 GojekAppReviewV4.0.0-V4.9.3_Cleaned.csv           ← Dataset ulasan Gojek
├── 📓 Ilham_Khadafi_UTS_AndatTidakTerstruktur.ipynb     ← Kode analisis (Jupyter Notebook)
└── 📝 Ilham_Khadafi_2306260952_A_UTS.pdf                ← Laporan UTS
```

---

## 📖 Deskripsi Proyek

Proyek ini menerapkan **Topic Modeling** berbasis **Latent Dirichlet Allocation (LDA)** untuk mengidentifikasi topik-topik tersembunyi dalam ulasan pengguna aplikasi Gojek yang diambil dari Google Play Store. Dengan menganalisis 30.000 sampel dari 225.000 ulasan berbahasa Indonesia, model berhasil menemukan **5 topik utama** yang merepresentasikan berbagai aspek pengalaman pengguna.

---

## 🗂️ Dataset

| Atribut | Tipe Data | Keterangan |
|---|---|---|
| `userName` | String | Nama pengguna yang memberikan ulasan |
| `content` | String | Teks ulasan pengguna (**variabel utama**) |
| `score` | Integer (1–5) | Rating bintang yang diberikan |
| `at` | Datetime | Waktu ulasan diberikan |
| `appVersion` | String | Versi aplikasi (V4.0.0 – V4.9.3) |

- **Total data:** 225.000 ulasan
- **Setelah preprocessing:** 87.930 ulasan (39,1%)
- **Digunakan untuk modeling:** 30.000 ulasan (sampling)
- **Sumber:** [Kaggle — Gojek App Reviews Bahasa Indonesia](https://www.kaggle.com/datasets/ucupsedaya/gojek-app-reviews-bahasa-indonesia/)

---

## ⚙️ Pipeline Analisis

```
Raw Data (225.000 ulasan)
        │
        ▼
┌─────────────────────────────┐
│     TEXT PREPROCESSING      │
│  1. Case folding            │
│  2. Hapus URL/mention/      │
│     hashtag                 │
│  3. Hapus angka & simbol    │
│  4. Hapus kata < 3 huruf    │
│  5. Stopwords removal       │
│     (Sastrawi + custom,     │
│      867 kata)              │
│  6. Stemming (Sastrawi)     │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   DOCUMENT-TERM MATRIX      │
│  CountVectorizer            │
│  max_df=0.90, min_df=10     │
│  max_features=3.000         │
│  ngram_range=(1,2)          │
│  → Shape: 30.000 × 3.000    │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   PEMILIHAN K OPTIMAL       │
│  Evaluasi Perplexity        │
│  K ∈ {3,4,5,6,7,8,10}      │
│  → Dipilih K=5              │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   LDA TOPIC MODELING        │
│  n_components=5             │
│  max_iter=30                │
│  learning_method='online'   │
│  learning_decay=0.7         │
│  Perplexity final: 991.97   │
└─────────────────────────────┘
        │
        ▼
   5 TOPIK UTAMA
```

---

## 🏷️ Hasil: 5 Topik yang Ditemukan

| Topik | Label | Kata Kunci Utama | Proporsi |
|:---:|---|---|:---:|
| 0 | **Promo & Diskon** | promo, voucher, diskon, ongkir, harga | 18,4% |
| 1 | **Masalah Akun & GoPay** | gopay, akun, saldo, blokir, transaksi | 20,6% |
| 2 | **Kepuasan Layanan** | bantu, mudah, ramah, nyaman, aman | 18,7% |
| 3 | **Layanan GoFood & Driver** | driver, order, pesan, makan, gofood | **26,7%** |
| 4 | **Pembayaran & GoPay Later** | bayar, biaya, gopaylater, tagih, paylater | 15,5% |

### 💡 Temuan Utama
- **Topik paling dominan:** Layanan GoFood & Driver (26,7%)
- **Topik paling positif:** Kepuasan Layanan — 40,0% ulasan bintang ⭐⭐⭐⭐⭐ berasal dari topik ini
- **Topik paling banyak dikeluhkan:** Masalah Akun & GoPay (29,3%) dan Layanan GoFood & Driver (32,9%) mendominasi ulasan bintang ⭐

---

## 🛠️ Library yang Digunakan

```python
# Core
pandas, numpy

# NLP & Preprocessing
PySastrawi          # Stemming & stopwords bahasa Indonesia
nltk

# Modeling
scikit-learn        # CountVectorizer, LatentDirichletAllocation

# Visualisasi
matplotlib, seaborn
wordcloud
```

### Instalasi
```bash
pip install PySastrawi wordcloud scikit-learn pandas numpy matplotlib seaborn tqdm
```

---

## 📚 Referensi

1. M. Wankhade, A. C. S. Rao, and C. Kulkarni, "A survey on sentiment analysis methods, applications, and challenges," *Artificial Intelligence Review*, vol. 55, pp. 5731–5780, 2022.
2. D. M. Blei, A. Y. Ng, and M. I. Jordan, "Latent Dirichlet Allocation," *Journal of Machine Learning Research*, vol. 3, pp. 993–1022, Jan. 2003.
3. A. Tala, "A Study of Stemming Effects on Information Retrieval in Bahasa Indonesia," M.S. thesis, Universiteit van Amsterdam, 2003.
4. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
5. U. Chauhan and A. Shah, "Topic Modeling using Latent Dirichlet Allocation: A Survey," *ACM Computing Surveys*, vol. 54, no. 7, pp. 1–35, 2021.

---

*SCST603217 Analisis Data Tidak Terstruktur — ATA 2025/2026 | FMIPA Universitas Indonesia*