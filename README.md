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
![dataset-cover](https://github.com/user-attachments/assets/f786b626-3002-4270-b3a0-1d68d9407484)

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
Data Source : https://www.kaggle.com/datasets/suvroo/ai-for-elderly-care-and-support?select=safety_monitoring.csv

Dataset tersebut memiliki 3 file yang terdiri dari 10.000 baris data dimana masing-masing file memiliki jumlah kolom/features yang berbeda. Pada data `daily_reminder.csv` memiliki 10.000 baris data dengan 6 kolom, pada data `health_monitoring.csv` memiliki 10.000 baris data dengan 10 kolom, dan pada data `safety_monitoring.csv` memiliki 10.000 baris data dengan 9 kolom.

Kemudian ketika dieksplorasi, data tersebut 9 baris yang memiliki *missing value*, duplikat, tetapi memiliki outliers dan juga mengalami data imbalanced.

Dataset tersebut memiliki 4 komponen utama :
- Health Data : Heart rate, blood pressure, glucose levels, and other vitals recorded from wearable devices.
- Activity & Movement Tracking : Sensor-based logs of movement, fall detection, and inactivity periods.
- Emergency Alerts : Records of triggered safety alerts due to unusual behavior or health anomalies.
- Reminder Logs : Medication schedules, appointment reminders, and daily task notifications.
This dataset enables AI-driven insights for personalized elderly care, predictive health monitoring, and automated reminders, ensuring safety, well-being, and improved quality of life for aging individuals.

Keempat komponen diatas terkandung dalam 3 file .csv yang bernama:
   - `daily_reminder.csv`
   - `health_monitoring.csv`
   - `safety_monitoring.csv`

**`daily_reminder.csv`**
- User ID
- Timestamp
- Reminder type : 'Appointment', 'Hydration', 'Other'
- Scheduled Time
- Reminder Sent : 'Yes' and 'No'
- Acknowledged : 'Yes' and 'No'

**`health_monitoring.csv`**
- User ID
- Timestamp
- Heart Rate : 60 - 120
- Hear Rate Below/Above Threshold : 'Yes' and 'No'
- Blood Pressure : integer
- Blood Pressure Below/Above Threshold : 'Yes' and 'No'
- Glucose Level : 70 - 150
- Glucose Level Below/Above Threshold : 'Yes' and 'No'
- Oxygen Saturation Level : 90 - 100
- Oxygen Saturation Level Below/Above Threshold : 'Yes' and 'No'

**`safety_monitoring.csv`**
- User ID
- Timestamp
- Movement Activity : 'Sitting', 'No Movement', 'Other'
- Fall Detected : 'Yes' and 'No'
- Impact Force Level : 'High', 'Medium', dan 'Low'
- Post Fall Inactivity Duration (seconds)
- Location : 'Bedroom', 'Bathroom', dan 'other'
- Alert Triggered : 'True' and 'false'
- Caregiver Notified : 'Yes' and 'No'


## Data Preparation
Pada bagian ini adalah langkah-langkah yang diambil untuk mempersiapkan data sebelum melakukan analisis dan pemodelan. Proses ini penting untuk memastikan bahwa data yang digunakan dalam model adalah bersih, relevan, dan siap untuk analisis lebih lanjut. Berikut adalah teknik data preparation yang diterapkan dalam notebook:

### Pembersihan Data
- Missing Values: Dataset `daily_monitoring` dan `safety_monitoring` memiliki missing value dengan tulisan `unnamed` berjumlah 9. Hal ini dapat dihapus saja karena jumlahnya tidak mempengaruhi data keseluruhan.
- Data Convertion : Konversi Timestamp ke Datetime
- Menghapus ID User
- Memisahkan angka antara Diastolik dan Sistolik
- Data Type Fixing: Kolom Post Fall Inactivity Duration dikonversi dari string ke numerik.

### Transformasi Data
- **Encoding Kategori**: Variabel kategori diubah menjadi format numerik menggunakan teknik *label encoding* pada kolom kategorikal seperti fitur `Alert Triggered`.
- **Normalisasi/Standarisasi**: Fitur numerik dinormalisasi atau distandarisasi untuk memastikan bahwa semua fitur berada dalam skala yang sama.
- **Binning** : Melakukan Binning pada data Diastolik dan Sistolik

### Pembagian Data
Data dibagi menjadi set pelatihan dan set pengujian untuk memastikan bahwa model dapat dievaluasi dengan baik. Metode pembagian menggunakan library `train_test_split`dari Scikit Learn

### Penanganan Data Imbalanced
Jika dataset memiliki kelas yang tidak seimbang, maka perlu menerapkan teknik berikut untuk menangani masalah ini:
- **Oversampling**: Meningkatkan jumlah sampel dari kelas minoritas menggunakan teknik seperti SMOTE (Synthetic Minority Over-sampling Technique).
- **Undersampling**: Mengurangi jumlah sampel dari kelas mayoritas untuk menciptakan keseimbangan.

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
![conf_Mat_LogReg_1](https://github.com/user-attachments/assets/65d9d5a1-6a21-4fa6-b106-1059b3c18868)

![ROC_LogReg_1](https://github.com/user-attachments/assets/b0112b51-f01c-4373-b119-034f2ac48b29)
<br>Ketika dianalisa, ternyata terdapat fitur yang memiliki signifikansi yang lebih besar daripada fitur lainnya
![importance_LogReg_1](https://github.com/user-attachments/assets/c29be15c-ac6b-41a8-bd7b-f8bb8d860b8e)

Kemudian melakukan hyper parameter tuning dan mendapatkan hasil sebagai berikut:
![conf_Mat_LogReg_2](https://github.com/user-attachments/assets/cbd5d280-76f3-4d49-8d2f-a9dde862b47c)
![ROC_LogReg_2](https://github.com/user-attachments/assets/6c620a21-9b57-4088-ab20-2dfccb2a7b12)
<br>Ternyata ketika dilakukan Grid Search CV, terdapat peningkatan performa estimator dan juga nilai ROC sekaligus signifikasi fitur pada model. Terlihat fitur `Kategori_Tekanan_Darah` meningkat bahkan melampaui `Oxygen Saturation (SpO2%).
![importance_LogReg_2](https://github.com/user-attachments/assets/26449ff9-1493-441a-b642-529758968bf8)

**Decision Tree 🌳**
Model baseline ini menunjukkan performa yang cukup baik dalam mengenali kelas mayoritas, namun masih memiliki kelemahan dalam mendeteksi kelas minoritas (Alert Triggered = True), yang ditunjukkan dengan nilai recall yang masih cukup rendah. Selain itu, model juga menunjukkan overfitting ringan karena performa pada training set lebih tinggi dibandingkan test set.

![ConfMat-DT_1](https://github.com/user-attachments/assets/2700d72e-d420-4dd3-8c10-13caf14b4ed7)
![ROC-DT_1](https://github.com/user-attachments/assets/1472a5ad-e500-451e-915f-b933d6c70254)
<br>Sama dengan model sebelumnya, ternyata terdapat fitur yang memiliki signifikansi yang lebih besar daripada fitur lainnya
![importance_DT_1](https://github.com/user-attachments/assets/9f5b3b92-81a2-4f54-813e-48fe23c0493a)

Kemudian melakukan hyper parameter tuning dengan Grid Search CV dan mendapatkan hasil sebagai berikut:
![ConfMat-DT_1](https://github.com/user-attachments/assets/ca022380-f079-4a9a-872a-a76190842157)
![ROC-DT_2](https://github.com/user-attachments/assets/d7f37f3b-6648-4568-9133-1b6880f7182c)
<br>Sama dengan model sebelumnya, terdapat peningkatan pada nilai ROC-AUC dan akurasi.
![importance_DT_2](https://github.com/user-attachments/assets/be949fdb-302f-4462-81e6-505fc0f1dc16)

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
