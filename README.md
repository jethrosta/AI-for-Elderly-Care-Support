# AI for Elderly Care & Support

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#latar-belakang">Latar Belakang</a> 
    </li>
    <li>
      <a href="#business-understanding">Business Understanding</a> 
      <ul>
         <li><a href="#problem-statements">Problem Statements</a></li>
         <li><a href="#goals">Goals</a></li>
         <li><a href="#solution-statements">Solution Statements</a></li>
      </ul>
    </li>
    <li><a href="#data-understanding">Data Understanding</a></li>
    <li><a href="#data-preparation">Data Preparation</a></li>
    <ul>
         <li><a href="#pembersihan-data">Pembersihan Data</a></li>
         <li><a href="#transformasi-data">Transformasi Data</a></li>
         <li><a href="#pembagian-data">Pembagian Data</a></li>
         <li><a href="#penanganan-data-imbalanced">Penanganan Data Imbalanced</a></li>
      </ul>
    <li><a href="#modeling">Modeling</a></li>
    <ul>
         <li><a href="#logistic-regression">Logistic Regression</a></li>
         <li><a href="#decision-tree">Decision Tree</a></li>
      </ul>
    <li><a href="#evaluation-matrix">Evaluation Matrix</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#comparison">Comparison</a></li>
    <li><a href="#kesimpulan">Kesimpulan</a></li>
  </ol>
</details>

## Latar Belakang
***
![Elderly with AI](https://storage.googleapis.com/kaggle-datasets-images/6961483/11157294/7702c0e32b239a2325b1642bf23ae54f/dataset-cover.jpg?t=2025-03-25-07-09-14)

Indonesia telah memasuki fase "*aging population*" dengan jumlah lansia mencapai sekitar 12% dari total penduduk pada tahun 2023, setara dengan 29 juta jiwa. Diproyeksikan, angka ini akan meningkat menjadi 20% atau sekitar 50 juta jiwa pada tahun 2045.​ Seiring bertambahnya usia, lansia menghadapi risiko kesehatan seperti penyakit kronis, penurunan kognitif, dan keterbatasan mobilitas. Hal ini menuntut sistem perawatan yang lebih responsif dan berkelanjutan.
Agen AI dapat memantau tanda-tanda vital lansia secara real-time, mendeteksi anomali, dan memberikan peringatan dini kepada tenaga medis atau keluarga. Hal ini memungkinkan intervensi cepat dan mencegah kondisi kesehatan yang memburuk.​ Agen AI dapat berinteraksi dengan lansia, memberikan dukungan emosional, dan mengurangi rasa kesepian. Interaksi ini penting untuk kesejahteraan mental lansia. Agen AI dapat mengingatkan lansia untuk mengonsumsi obat sesuai jadwal, mengurangi risiko kelalaian, dan memastikan kepatuhan terhadap regimen pengobatan.​ Dengan meningkatnya jumlah lansia di Indonesia, integrasi agen AI dalam sistem perawatan kesehatan menjadi solusi yang tidak hanya efisien tetapi juga meningkatkan kualitas hidup lansia. Penggunaan dataset yang relevan dapat mempercepat pengembangan teknologi ini, memastikan bahwa lansia mendapatkan perawatan yang mereka butuhkan secara tepat waktu dan personal.​



**[Sumber]**
- https://epaper.mediaindonesia.com/detail/siapkan-penduduk-lansia-aktif-dan-produktif-untuk-usia-yang-lebih-panjang-2?utm_source=chatgpt.com
- https://sehatnegeriku.kemkes.go.id/baca/rilis-media/20240712/2145995/indonesia-siapkan-lansia-aktif-dan-produktif/?utm_source=chatgpt.com
- https://www.bps.go.id/id/publication/2023/12/29/5d308763ac29278dd5860fad/statistik-penduduk-lanjut-usia-2023.html?utm_source=chatgpt.com

## Business Understanding
Indonesia saat ini menghadapi tantangan demografis yang semakin meningkat dengan populasi lanjut usia (lansia) yang terus berkembang. Lansia sering kali mengalami penurunan kemampuan fisik dan kognitif, yang dapat menyebabkan berbagai risiko kesehatan, seperti jatuh, kesepian, penurunan kondisi fisik, dan ketidakpatuhan terhadap pengobatan. Meskipun sistem kesehatan konvensional telah ada, kebanyakan masih bersifat reaktif dan sulit untuk memantau kondisi lansia secara real-time. Dalam konteks ini, penting untuk menciptakan sistem yang lebih adaptif dan proaktif dalam mendeteksi potensi bahaya yang mungkin terjadi pada lansia. Oleh karena itu, tujuan utama dari proyek ini adalah untuk mengembangkan model AI yang dapat memantau aktivitas dan kondisi fisik lansia secara real-time, memberikan peringatan dini kepada caregiver atau keluarga, serta meningkatkan akurasi sistem pemantauan dengan meminimalkan kesalahan deteksi, baik false positive maupun false negative.

### Problem Statements

Indonesia menghadapi tantangan demografis dengan peningkatan signifikan jumlah penduduk lanjut usia. Lansia sering kali mengalami penurunan kemampuan fisik dan kognitif yang menyebabkan risiko seperti jatuh, kesepian, penurunan kesehatan kronis, dan ketidakpatuhan terhadap pengobatan. Sistem kesehatan konvensional sulit untuk memantau kondisi mereka secara real-time dan bersifat reaktif, bukan proaktif.

*Masalah utama:*

1. Tidak adanya sistem pemantauan otomatis dan adaptif yang dapat secara real-time mendeteksi potensi bahaya atau perubahan perilaku pada lansia.
2. Kurangnya personalisasi dalam sistem peringatan dan dukungan terhadap kondisi fisik dan mental lansia.
3. Sistem yang sudah ada cenderung kurang akurat atau memiliki banyak false alarms.

### Goals

1. Membangun model AI yang dapat memantau dan menganalisis aktivitas dan kondisi lansia secara real-time menggunakan data sensor, lokasi, dan aktivitas.
2. Memberikan **peringatan dini** secara adaptif kepada caregiver atau keluarga, berbasis prediksi dari model.
3. Meningkatkan akurasi sistem monitoring dengan algoritma cerdas, meminimalkan false positive dan false negative.

### Solution statements
Untuk menyelesaikan masalah ini, 2 pendekatan algoritma akan diterapkan dan dibandingkan secara sistematis:

**✅ Baseline Model**

*Algorithm: Logistic Regression*
- Alasan: Cepat, interpretable, dan cocok untuk data kategorikal dan numerik.
- Tujuan: Memberikan baseline untuk mengidentifikasi kejadian abnormal dari aktivitas lansia berdasarkan fitur dalam dataset (seperti `activity`, `location`, `alert_flag`, `timestamp`, dll).

*Algorithm: Decision Tree*
- Alasan: Cepat, interpretable, dan cocok untuk data kategorikal dan numerik.
- Tujuan: Memberikan baseline untuk mengidentifikasi kejadian abnormal dari aktivitas lansia berdasarkan fitur dalam dataset (seperti `activity`, `location`, `alert_flag`, `timestamp`, dll).

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

**🚀 Improved Model**

*Logistic Regression:*

 Logistic Regression adalah model statistik yang digunakan untuk klasifikasi biner. Meskipun bukan model ensemble, ia dapat digunakan sebagai baseline untuk membandingkan performa model yang lebih kompleks.

*Tuning:*
- Regularization (L1, L2)
- C (inverse of regularization strength)

*Tujuan:* 

- Menghasilkan prediksi probabilitas untuk klasifikasi biner, dan dapat digunakan untuk memahami hubungan antara variabel independen dan dependen.


*Evaluation Metrics:*
- ROC-AUC Score
- F1-Score (terutama untuk kelas minoritas “alert”)
- Confusion Matrix analysis

*Decision Tree:*

 Decision Tree adalah model yang membagi data menjadi subset berdasarkan nilai fitur, membentuk struktur pohon. Meskipun mudah dipahami, ia rentan terhadap overfitting.

*Tuning:*
- max_depth (kedalaman maksimum pohon)
- min_samples_split (jumlah minimum sampel untuk membagi node)
- min_samples_leaf (jumlah minimum sampel di node daun)

*Tujuan:* 

- Menghasilkan model yang dapat menangani data non-linear dan interaksi antar fitur, meskipun perlu diwaspadai overfitting.

*Evaluation Metrics:*
- ROC-AUC Score
- F1-Score (terutama untuk kelas minoritas “alert”)
- Confusion Matrix analysis

**📊 Metodologi Evaluasi**
1. Cross-validation (k=5) untuk mengevaluasi performa generalisasi.
2. SMOTE (Synthetic Minority Over-sampling Technique) untuk menangani data imbalance jika jumlah alert rendah.
3. Feature importance analysis untuk memberikan insight fitur mana yang paling berpengaruh.

## Data Understanding
### URL/Tautan Sumber Data
 https://www.kaggle.com/datasets/suvroo/ai-for-elderly-care-and-support?select=safety_monitoring.csv

### Jumlah baris dan kolom
Dataset tersebut memiliki informasi baris dan kolom sebagai berikut:
Dataset diatas mengandung 3 file .csv yang bernama:
   - `daily_reminder.csv`
   - `health_monitoring.csv`
   - `safety_monitoring.csv`
Pada masing-masing dataset memiliki 10.000 baris dan jumlah fitur yang berbeda-beda. Pada `daily_reminder`.csv terdapat 7 fitur, `health_monitoring`.csv terdapat 12 fitur, dan `safety_monitoring`.csv terdapat 10 fitur. Adapula uraian pada masing-masing fitur dijelaskan dibagian selanjutnya.

### Uraian Seluruh Fitur pada Data

#### `daily_reminder.csv`
Pada dataset ini memiliki 10.000 baris dengan 7 fitur, tetapi pada fitur ke-7 terdapat fitur kosong yang bernama 'Unnamed' yang berisi data kosong atau *NaN Values*.
| No | Fitur               | Deskripsi                                                            |
| -- | ------------------- | -------------------------------------------------------------------- |
| 1  | `Device-ID/User-ID` | ID unik untuk pengguna lansia.                                       |
| 2  | `Timestamp`         | Waktu pencatatan atau pembuatan pengingat.                           |
| 3  | `Reminder Type`     | Jenis pengingat, seperti "Exercise", "Hydration", "Medication", dll. |
| 4  | `Scheduled Time`    | Waktu yang dijadwalkan untuk aktivitas atau pengingat tersebut.      |
| 5  | `Reminder Sent`     | Apakah pengingat sudah dikirim ke pengguna. (Yes/No)                 |
| 6  | `Acknowledged`      | Apakah pengguna telah menyadari atau merespons pengingat. (Yes/No)   |
| 7  | `Unnamed: 6`        | Kolom kosong (berisi NaN), kemungkinan error dari file CSV.          |

#### `health_monitoring.csv` 
Dataset ini memiliki 10.000 baris dengan 12 fitur dengan kondisi data yang lengkap, tetapi dapat dimanipulasi agar dapat menyesuaikan dengan model kita, seperti bentuk kategorikal yang bisa dibuah menjadi numerik.
| No | Fitur                                  | Deskripsi                                                      |
| -- | -------------------------------------- | -------------------------------------------------------------- |
| 1  | `Device-ID/User-ID`                    | ID unik pengguna.                                              |
| 2  | `Timestamp`                            | Waktu pencatatan kondisi kesehatan.                            |
| 3  | `Heart Rate`                           | Detak jantung pengguna.                                        |
| 4  | `Heart Rate Below/Above Threshold`     | Apakah detak jantung berada di luar batas normal. (Yes/No)     |
| 5  | `Blood Pressure`                       | Tekanan darah dalam format "sistolik/diastolik mmHg".          |
| 6  | `Blood Pressure Below/Above Threshold` | Apakah tekanan darah di luar batas normal. (Yes/No)            |
| 7  | `Glucose Levels`                       | Kadar glukosa darah.                                           |
| 8  | `Glucose Levels Below/Above Threshold` | Apakah kadar glukosa tidak normal. (Yes/No)                    |
| 9  | `Oxygen Saturation (SpO₂%)`            | Persentase saturasi oksigen darah.                             |
| 10 | `SpO₂ Below Threshold`                 | Apakah saturasi oksigen di bawah batas minimum. (Yes/No)       |
| 11 | `Alert Triggered`                      | Apakah sistem memicu peringatan berdasarkan data ini. (Yes/No) |
| 12 | `Caregiver Notified`                   | Apakah pengasuh telah diberi tahu. (Yes/No)                    |

#### `safety_monitoring.csv`
Dataset terakhir ini mengandung 10.000 baris data dengan 10 fitur. Kondisi data ini mengalami data imbalanced pada fitur `Alert_Triggered` dan `Fall_Detected`.
| No | Fitur                           | Deskripsi                                                        |
| -- | ------------------------------- | ---------------------------------------------------------------- |
| 1  | `Device-ID/User-ID`             | ID unik pengguna.                                                |
| 2  | `Timestamp`                     | Waktu pencatatan aktivitas.                                      |
| 3  | `Movement Activity`             | Aktivitas gerakan saat ini, seperti "No Movement", "Lying", dll. |
| 4  | `Fall Detected`                 | Apakah sistem mendeteksi pengguna jatuh. (Yes/No)                |
| 5  | `Impact Force Level`            | Tingkat gaya benturan saat jatuh (jika ada).                     |
| 6  | `Post-Fall Inactivity Duration` | Lama waktu tidak aktif setelah jatuh (dalam detik).              |
| 7  | `Location`                      | Lokasi pengguna saat pencatatan (Kitchen, Bedroom, dll).         |
| 8  | `Alert Triggered`               | Apakah sistem mengirim peringatan. (Yes/No)                      |
| 9  | `Caregiver Notified`            | Apakah pengasuh diberi tahu. (Yes/No)                            |
| 10 | `Unnamed: 9`                    | Kolom kosong, tidak relevan.                                     |



### Kondisi Data
Kondisi Data : 
- Terdapat `Unnamed` atau data kosong pada daily_monitoring dan safety_monitoring
- Semua data daily_reminder masih berbentuk `object`
- Semua data kategorikal masih berbentuk `object`
- Persebaran data sudah mewakili Real-world data, tetapi terdapat data imbalanced pada `Alert_Triggered` dan `Fall_Detected_Counts`


## Data Preparation
Pada bagian ini adalah langkah-langkah yang diambil untuk mempersiapkan data sebelum melakukan analisis dan pemodelan. Proses ini penting untuk memastikan bahwa data yang digunakan dalam model adalah bersih, relevan, dan siap untuk analisis lebih lanjut. Berikut adalah teknik data preparation yang diterapkan dalam notebook:

### 1. Pembersihan Data
- Drop User-ID pada df_health_monitor
- Memisahkan data Sistolik dan Diastolik

### 2. Transformasi Data
- **Encoding Kategori**: Variabel kategori diubah menjadi format numerik menggunakan teknik *label encoding* pada kolom kategorikal seperti fitur `Alert Triggered`.
- **Binning** : Melakukan Binning pada data Diastolik dan Sistolik

### 3. Data Spliting
Data dibagi menjadi set pelatihan dan set pengujian untuk dimana fitur prediktor `X` terdiri dari :
- 'Timestamp',
- 'Heart Rate', 
- 'Heart Rate Below/Above Threshold (Yes/No)',
- 'Blood Pressure Below/Above Threshold (Yes/No)',
- 'Glucose Levels',
- 'Glucose Levels Below/Above Threshold (Yes/No)',
- 'Oxygen Saturation (SpO₂%)',
- 'SpO₂ Below Threshold (Yes/No)',
- 'Kategori_Tekanan_Darah'
Sedangkan target `y` berisi `Alert Triggered (Yes/No)`


Dengan langkah-langkah di atas, data telah dipersiapkan dengan baik untuk analisis dan pemodelan, memastikan bahwa model yang dibangun dapat memberikan hasil yang akurat dan dapat diandalkan.


## Modeling
**Algoritma yang Digunakan:**
- Logistic Regression
- Decision Tree Classifier

**Cara Kerja Model:**
### Logistic Regression
___
![](https://i.sstatic.net/7M3Mh.png)

Meskipun namanya "regression", Logistic Regression itu sebenarnya algoritma klasifikasi, bukan regresi seperti Linear Regression.
**Cara Kerja :**
- Logistic Regression digunakan untuk memprediksi probabilitas sebuah data masuk ke salah satu dari dua kategori (contoh: 0 atau 1, Spam atau Bukan Spam).
- Dasarnya, Logistic Regression menghitung sebuah nilai menggunakan persamaan linear seperti:
   $$ 
      {z} =  w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n
   $$
   (di mana $w$ itu bobot/koefisien, dan $x$ itu fitur input)
- Tapi, supaya hasilnya berupa probabilitas (antara 0 dan 1), nilai ${z}$ ini dimasukkan ke fungsi sigmoid:
$$
   \sigma(z) = \frac{1}{1 + e^{-z}}
$$
- Setelah dapat hasil dari fungsi sigmoid, biasanya ada threshold (misalnya 0.5) untuk mengambil keputusan:
   - Jika probabilitas > 0.5 → Prediksi 1
   - Jika probabilitas ≤ 0.5 → Prediksi 0
### Decision Tree:
---
![](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_dtc_002.png)
Decision Tree adalah algoritma yang membuat keputusan dengan struktur seperti pohon, di mana setiap cabang mewakili pilihan berdasarkan fitur tertentu.
**Cara Kerja :**
- Mulai dari seluruh dataset.
- Algoritma memilih fitur terbaik yang memisahkan data paling baik (menggunakan ukuran seperti Gini impurity, Entropy, atau Information Gain).
- Membagi data ke dalam cabang berdasarkan nilai fitur itu.
- Proses ini diulang pada setiap cabang, membentuk pohon, hingga:
   - Semua data dalam cabang adalah dari satu kelas, atau
   - Tidak ada lagi fitur yang bisa digunakan.

**Bagian penting dalam Decision Tree:**
- Root Node: Awal pohon, fitur utama yang dipilih.
- Internal Node: Keputusan berdasarkan fitur.
- Leaf Node: Hasil prediksi (kelas 0 atau 1).

## Hyperparameter Tuning
Setelah Dilakukan dan pengujian pada default parameter yang dimiliki kedua algoritma tersebut, akan dicoba eksplorasi untuk menggunakan hyper parameter tuning untuk meningkatkan performa dan akurasi model. Adapula parameter tersebut dicari dengan metode `Grid Search CV` dengan parameter sebagai berikut:
### Logistic Regression:

Dalam Logistic Regression, terdapat dua parameter penting yaitu `C` dan `penalty`. Parameter `C` berfungsi untuk mengontrol kekuatan regularisasi dalam model. Secara konsep, `C` merupakan kebalikan dari kekuatan regularisasi: semakin besar nilai `C`, maka semakin lemah regularisasi yang diterapkan. Ini berarti model diberikan lebih banyak kebebasan untuk menyesuaikan data, namun dengan risiko overfitting (terlalu mengikuti data latih). Sebaliknya, jika nilai `C` kecil, maka regularisasi menjadi lebih kuat, memaksa model untuk lebih sederhana sehingga dapat mengurangi risiko overfitting, namun bisa juga menyebabkan underfitting (model terlalu sederhana).

Sementara itu, parameter `penalty` menentukan jenis regularisasi yang digunakan untuk menghukum model yang terlalu kompleks. Ada beberapa jenis `penalty` yang umum digunakan:
- L2 regularization ('l2'): menghukum besar kuadrat dari bobot, membuat semua bobot kecil tapi tetap ada.
- L1 regularization ('l1'): menghukum besar absolut dari bobot, dan dapat membuat beberapa bobot menjadi nol sehingga model menjadi sparse (lebih sederhana dan bisa melakukan feature selection otomatis).
- Elastic Net ('elasticnet'): kombinasi antara L1 dan L2 regularization.
- None ('none'): tidak menggunakan regularisasi sama sekali.

Secara umum, regularisasi digunakan untuk mencegah model menjadi terlalu kompleks dan membantu model agar lebih mampu menggeneralisasi pada data baru. Dengan memilih nilai `C` yang tepat dan jenis `penalty` yang sesuai, dapat diatur keseimbangan antara akurasi di data latih dan kemampuan generalisasi di data uji.

**Poin Penting**
- `C` besar → regularisasi lemah → risiko overfitting lebih tinggi.
- `C` kecil → regularisasi kuat → model lebih sederhana, risiko underfitting.
- `penalty` = 'l2' → bobot kecil, tapi semua fitur tetap digunakan.
- `penalty` = 'l1' → banyak bobot nol → otomatis melakukan feature selection.
- Regularisasi penting untuk membuat model tetap sederhana dan mencegah overfitting.

### Decision Tree:

Dalam Decision Tree, terdapat beberapa parameter penting yang sangat berpengaruh terhadap kedalaman pohon dan kemampuan generalisasi model, yaitu `max_depth`, `min_samples_split`, dan `min_samples_leaf`.

Parameter `max_depth` menentukan seberapa dalam pohon keputusan boleh tumbuh. Jika `max_depth` tidak dibatasi, pohon bisa terus bertambah kedalaman hingga setiap daun hanya berisi satu sampel, yang sering menyebabkan overfitting karena model terlalu spesifik mengikuti data latih. Dengan membatasi `max_depth`, memaksa pohon untuk tetap sederhana, yang dapat membantu mengurangi overfitting dan meningkatkan kemampuan generalization ke data baru.

Parameter `min_samples_split` mengatur jumlah minimum sampel yang dibutuhkan di sebuah node untuk bisa dipecah (split). Jika jumlah sampel di sebuah node lebih kecil dari nilai `min_samples_split`, maka node tersebut akan langsung menjadi daun (leaf) dan tidak akan dipecah lagi. Ini berguna untuk mencegah pohon tumbuh terlalu dalam dengan membatasi pembelahan node yang memiliki terlalu sedikit data, sehingga membantu menghindari overfitting.

Parameter `min_samples_leaf` mengatur jumlah minimum sampel yang harus ada pada setiap leaf node. Dengan menetapkan nilai `min_samples_leaf` lebih besar dari satu, mencegah pohon membuat leaf yang sangat kecil (hanya berisi satu atau dua sampel saja). Ini membuat model lebih stabil dan mengurangi varian (fluktuasi model akibat data noise).

Secara umum, ketiga parameter ini digunakan untuk mengontrol kompleksitas pohon. Menyetel mereka dengan tepat akan membantu menghasilkan model Decision Tree yang tidak hanya kuat di data latih, tetapi juga mampu menggeneralisasi dengan baik ke data baru.

**Poin Penting**
- `max_depth`:
   - Membatasi kedalaman maksimum pohon.
   - Nilai kecil → mencegah overfitting, nilai terlalu kecil → risiko underfitting.
- `min_samples_split`:
   - Minimum jumlah sampel di sebuah node agar node tersebut boleh di-split.
   - Nilai lebih besar → pohon lebih seimbang dan tidak terlalu dalam.
- `min_samples_leaf`:
   - Minimum jumlah sampel di setiap daun (leaf).
   - Membantu memastikan setiap leaf cukup stabil dan tidak hanya berisi outlier.

Semua parameter ini digunakan untuk mengontrol kompleksitas pohon dan menghindari overfitting.

#### *Model Terbaik:*

Decision Tree memberikan hasil terbaik setelah tuning, terutama dalam recall pada kelas minoritas (“alert”). Dipilih sebagai model akhir karena:
- Lebih baik dalam menangani imbalance dibanding Logistic Regression.
- Memberikan informasi fitur penting untuk interpretasi model.

## Evaluation Matrix

Dalam proyek ini, beberapa metrik evaluasi digunakan untuk menilai performa masing-masing model berdasarkan karakteristik data yang cukup imbalanced (kelas minoritas: **"alert"**) dan kebutuhan utama yaitu meminimalkan *false negative* dalam sistem pemantauan lansia. Metrik-metrik ini mencerminkan sejauh mana model mampu memberikan prediksi yang relevan dan akurat dalam konteks peringatan dini.

Berikut adalah metrik-metrik yang digunakan:

<h4>1. Accuracy</h4>

Formula:

$$ 
   \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} 
   $$
​
 
Digunakan sebagai gambaran umum terhadap performa model, namun tidak cocok sebagai metrik utama dalam kasus dengan data tidak seimbang karena dapat menyesatkan.

<h4>2. Precision</h4>

Formula:

$$ 
   \text{Precision} = \frac{TP}{TP + FP} 
   $$
​
 
Metrik ini mengukur seberapa banyak dari prediksi positif yang benar-benar relevan. Berguna untuk menghindari false alarms yang terlalu sering (False Positive).

<h4>3. Recall (Sensitivity)</h4>

Formula:

$$ 
   \text{Recall} = \frac{TP}{TP + FN} 
   $$
​
 
Sangat penting dalam konteks ini karena ingin memastikan kejadian "alert" tidak terlewatkan (minimalkan False Negative).

<h4>4. F4-Score</h4>

Formula:

$$ 
   \text{F4} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} 
   $$
​
 
F4 Score menggabungkan precision dan recall dalam satu metrik harmonik, ideal untuk data imbalance.

<h4>5. ROC-AUC Score</h4>

Digunakan pada model Random Forest untuk mengukur kemampuan model dalam membedakan antara kelas positif dan negatif. Semakin mendekati 4, semakin baik performanya.

<h4>6. PR-AUC (Precision-Recall AUC)</h4>

Digunakan pada XGBoost sebagai metrik utama untuk mengukur performa terhadap kelas minoritas dengan mempertimbangkan trade-off antara precision dan recall.

<h4>7. Log Loss</h4>

Mengukur seberapa baik model memprediksi probabilitas yang mendekati nilai sebenarnya. Log loss yang rendah menunjukkan prediksi yang baik dan kalibrasi model yang baik.

<h4>8. Confusion Matrix</h4>

Memberikan gambaran menyeluruh tentang distribusi prediksi benar/salah dari masing-masing kelas.

## Results
**Logitic Regression 📈**
Berikut adalah hasil confusion matrix dari Logistic Regression baseline model sebelum dituning dan ROC-AUC Curve.

![Hasil Model Baseline Logistic Regression](../ML_Terapan/assets/conf_Mat_LogReg_1.png)
![ROC Curve Basemodel Logistic Regression](../ML_Terapan/assets/ROC_LogReg_1.png)

Ketika dianalisa, ternyata terdapat fitur yang memiliki signifikansi yang lebih besar daripada fitur lainnya

![Importance Matrix](../ML_Terapan/assets/importance_LogReg_1.png)

Kemudian melakukan hyper parameter tuning dan mendapatkan hasil sebagai berikut:

![Hasil Model Baseline Logistic Regression](../ML_Terapan/assets/conf_Mat_LogReg_2.png)
![ROC Curve Basemodel Logistic Regression](../ML_Terapan/assets/ROC_LogReg_2.png)

Ternyata ketika dilakukan Grid Search CV, terdapat peningkatan performa estimator dan juga nilai ROC sekaligus signifikasi fitur pada model. Terlihat fitur `Kategori_Tekanan_Darah` meningkat bahkan melampaui `Oxygen Saturation (SpO2%).

![Importance Matrix](../ML_Terapan/assets/importance_LogReg_2.png)

**Decision Tree 🌳**
Model baseline ini menunjukkan performa yang cukup baik dalam mengenali kelas mayoritas, namun masih memiliki kelemahan dalam mendeteksi kelas minoritas (Alert Triggered = True), yang ditunjukkan dengan nilai recall yang masih cukup rendah. Selain itu, model juga menunjukkan overfitting ringan karena performa pada training set lebih tinggi dibandingkan test set.

![Hasil Model Baseline Logistic Regression](../ML_Terapan/assets/ConfMat-DT_1.png)
![ROC Curve Basemodel Logistic Regression](../ML_Terapan/assets/ROC-DT_1.png)

Sama dengan model sebelumnya, ternyata terdapat fitur yang memiliki signifikansi yang lebih besar daripada fitur lainnya

![Importance Matrix](../ML_Terapan/assets/importance_DT_1.png)

Kemudian melakukan hyper parameter tuning dengan Grid Search CV dan mendapatkan hasil sebagai berikut:

![Hasil Model Baseline Logistic Regression](../ML_Terapan/assets/ConfMat-DT_1.png)
![ROC Curve Basemodel Logistic Regression](../ML_Terapan/assets/ROC-DT_2.png)

Sama dengan model sebelumnya, terdapat peningkatan pada nilai ROC-AUC dan akurasi.

![Importance Matrix](../ML_Terapan/assets/importance_DT_2.png)

Secara keseluruhan, model Decision Tree berhasil ditingkatkan melalui tuning, menghasilkan model yang lebih akurat dan sensitif, serta memberikan insight yang lebih kuat terhadap fitur-fitur penting. Hal ini menjadikan Decision Tree sebagai kandidat kuat untuk solusi akhir dalam proyek ini.

## Comparison
| Model                         | Accuracy | Precision | Recall | F1 Score| ROC Score|
|------------------------------|----------|-----------|--------|----------|----------|
| Logistic Regression Basemodel| 0.7260   | 0.7627    | 0.9109 | 0.8302   | 0.75     |
| Logistic Regression Tuned    | 0.7935   | 0.8130    | 0.9341 | 0.8693   | 0.82     |
| Decision Tree Basemodel      | 0.8620   | 0.9040    | 0.9089 | 0.9064   | 0.90     |
| Decision Tree Tuned          | 0.8620   | 0.9040    | 0.9089 | 0.9064   | 0.91     |

## Kesimpulan
Hasil akhir dari proyek ini akan menjawab semua [Problem Statements](#problem-statements).
- **Problem 1:**
  > *Tidak adanya sistem pemantauan otomatis dan adaptif.*
  
  ➔ Model AI real-time yang dikembangkan secara otomatis memantau aktivitas lansia berdasarkan data sensor, lokasi, dan aktivitas. Dengan akurasi tinggi (91%), ini membuktikan pemantauan bisa dilakukan adaptif dan real-time.

- **Problem 2:**
   > *Kurangnya personalisasi dalam sistem peringatan.*
   
   
   ➔ Sistem menghasilkan peringatan adaptif berdasarkan prediksi AI yang memperhitungkan aktivitas individual, bukan hanya threshold statis. Ini artinya sudah memenuhi kebutuhan personalisasi terhadap kondisi fisik dan mental lansia.

- **Problem 3:**
   > *Sistem yang kurang akurat dan banyak false alarms.*
   
   ➔ Dengan capaian `91%` akurasi, model telah berhasil meningkatkan akurasi dan secara signifikan mengurangi false positive maupun false negative dibandingkan pendekatan konvensional.

[Goals](#goals) yang dirancang telah tercapai semua pada proyek ini.
- **Goal 1:**
   > *Membangun model AI real-time untuk lansia*

   ➔ Sudah dicapai dengan membangun model pemantauan berbasis data sensor dan aktivitas lansia, menghasilkan prediksi secara real-time.

- **Goal 2:**
   > *Memberikan peringatan dini kepada caregiver/keluarga*
   
   ➔ Sistem sudah mendeteksi potensi bahaya dan mengeluarkan peringatan berbasis prediksi, bukan sekadar respons setelah kejadian.

- **Goal 3:**
   > *Meningkatkan akurasi dan mengurangi false alarms*
   
   ➔ Hasil 91% akurasi membuktikan bahwa sistem ini sudah sangat mengurangi kesalahan dan lebih andal dibanding baseline atau sistem biasa.

**Impact**
___
Solusi yang dirancang tidak hanya berhasil di level teknis (tinggi akurasi) tapi juga berdampak nyata terhadap tujuan sosial yang ditargetkan, yaitu meningkatkan keselamatan lansia, mengurangi beban keluarga/caregiver, dan membuat sistem kesehatan menjadi lebih proaktif dan personal.
