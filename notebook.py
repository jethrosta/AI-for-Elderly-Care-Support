#!/usr/bin/env python
# coding: utf-8

# # Domain Proyek - Kesehatan
# 
# Proyek : AI for Elderly Cara & Support
# 
# **🙋 WHY?**
# 
# *1. Peningkatan Populasi Lansia*
# 
# Indonesia telah memasuki fase "aging population" dengan jumlah lansia mencapai sekitar 12% dari total penduduk pada tahun 2023, setara dengan 29 juta jiwa. Diproyeksikan, angka ini akan meningkat menjadi 20% atau sekitar 50 juta jiwa pada tahun 2045 .​
# 
# *2. Tantangan Kesehatan dan Kemandirian*
# 
# Seiring bertambahnya usia, lansia menghadapi risiko kesehatan seperti penyakit kronis, penurunan kognitif, dan keterbatasan mobilitas. Hal ini menuntut sistem perawatan yang lebih responsif dan berkelanjutan.
# 
# **🤖 Peran Agen AI dalam Menyelesaikan Masalah**
# 
# *1. Pemantauan Kesehatan Real-Time*
# 
# Agen AI dapat memantau tanda-tanda vital lansia secara real-time, mendeteksi anomali, dan memberikan peringatan dini kepada tenaga medis atau keluarga. Hal ini memungkinkan intervensi cepat dan mencegah kondisi kesehatan yang memburuk .​
# 
# *2. Dukungan Emisional dan Sosial*
# 
# Agen AI dapat berinteraksi dengan lansia, memberikan dukungan emosional, dan mengurangi rasa kesepian. Interaksi ini penting untuk kesejahteraan mental lansia .​
# 
# *3. Manajemen Obat dan Jadwal*
# 
# Agen AI dapat mengingatkan lansia untuk mengonsumsi obat sesuai jadwal, mengurangi risiko kelalaian, dan memastikan kepatuhan terhadap regimen pengobatan .​
# 
# **📚 Kesimpulan**
# 
# Dengan meningkatnya jumlah lansia di Indonesia, integrasi agen AI dalam sistem perawatan kesehatan menjadi solusi yang tidak hanya efisien tetapi juga meningkatkan kualitas hidup lansia. Penggunaan dataset yang relevan dapat mempercepat pengembangan teknologi ini, memastikan bahwa lansia mendapatkan perawatan yang mereka butuhkan secara tepat waktu dan personal.​
# 
# 
# 
# **[Sumber]**
# - https://epaper.mediaindonesia.com/detail/siapkan-penduduk-lansia-aktif-dan-produktif-untuk-usia-yang-lebih-panjang-2?utm_source=chatgpt.com
# - https://sehatnegeriku.kemkes.go.id/baca/rilis-media/20240712/2145995/indonesia-siapkan-lansia-aktif-dan-produktif/?utm_source=chatgpt.com
# - https://www.bps.go.id/id/publication/2023/12/29/5d308763ac29278dd5860fad/statistik-penduduk-lanjut-usia-2023.html?utm_source=chatgpt.com

# #  Business Understanding
# 
# **Problem Statements**
# 
# Indonesia menghadapi tantangan demografis dengan peningkatan signifikan jumlah penduduk lanjut usia. Lansia sering kali mengalami penurunan kemampuan fisik dan kognitif yang menyebabkan risiko seperti jatuh, kesepian, penurunan kesehatan kronis, dan ketidakpatuhan terhadap pengobatan. Sistem kesehatan konvensional sulit untuk memantau kondisi mereka secara real-time dan bersifat reaktif, bukan proaktif.
# 
# *Masalah utama:*
# 
# - Tidak adanya sistem pemantauan otomatis dan adaptif yang dapat secara real-time mendeteksi potensi bahaya atau perubahan perilaku pada lansia.
# - Kurangnya personalisasi dalam sistem peringatan dan dukungan terhadap kondisi fisik dan mental lansia.
# - Sistem yang sudah ada cenderung kurang akurat atau memiliki banyak false alarms.
# 
# **🎯 Project Goals**
# 1. Membangun model AI yang dapat memantau dan menganalisis aktivitas dan kondisi lansia secara real-time menggunakan data sensor, lokasi, dan aktivitas.
# 2. Mengklasifikasikan potensi risiko seperti jatuh, tidak aktif terlalu lama, atau kelainan pola aktivitas.
# 3. Memberikan **peringatan dini** secara adaptif kepada caregiver atau keluarga, berbasis prediksi dari model.
# 4. Meningkatkan akurasi sistem monitoring dengan algoritma cerdas, meminimalkan false positive dan false negative.
# 5. Mengembangkan sistem yang scalable dan dapat dievaluasi dengan metrik standar AI/ML.
# 
# **🧩 Solution Statement**
# 
# Untuk menyelesaikan masalah ini, 2 pendekatan algoritma akan diterapkan dan dibandingkan secara sistematis:
# 
# **✅ Baseline Model**
# 
# *Algorithm: Logistic Regression*
# - Alasan: Cepat, interpretable, dan cocok untuk data kategorikal dan numerik.
# - Tujuan: Memberikan baseline untuk mengidentifikasi kejadian abnormal dari aktivitas lansia berdasarkan fitur dalam dataset (seperti `activity`, `location`, `alert_flag`, `timestamp`, dll).
# 
# *Algorithm: Decision Tree*
# - Alasan: Cepat, interpretable, dan cocok untuk data kategorikal dan numerik.
# - Tujuan: Memberikan baseline untuk mengidentifikasi kejadian abnormal dari aktivitas lansia berdasarkan fitur dalam dataset (seperti `activity`, `location`, `alert_flag`, `timestamp`, dll).
# 
# Evaluation Metrics:
# 
# - Accuracy
# - Precision
# - Recall
# - F1-Score
# - ROC-AUC Score
# - Confusion Matrix
# 
# **🚀 Improved Model**
# 
# *Logistic Regression:*
# 
#  Logistic Regression adalah model statistik yang digunakan untuk klasifikasi biner. Meskipun bukan model ensemble, ia dapat digunakan sebagai baseline untuk membandingkan performa model yang lebih kompleks.
# 
# *Tuning:*
# - Regularization (L1, L2)
# - C (inverse of regularization strength)
# 
# *Tujuan:* 
# 
# - Menghasilkan prediksi probabilitas untuk klasifikasi biner, dan dapat digunakan untuk memahami hubungan antara variabel independen dan dependen.
# 
# 
# *Evaluation Metrics:*
# - ROC-AUC Score
# - F1-Score (terutama untuk kelas minoritas “alert”)
# - Confusion Matrix analysis
# 
# *Decision Tree:*
# 
#  Decision Tree adalah model yang membagi data menjadi subset berdasarkan nilai fitur, membentuk struktur pohon. Meskipun mudah dipahami, ia rentan terhadap overfitting.
# 
# *Tuning:*
# - max_depth (kedalaman maksimum pohon)
# - min_samples_split (jumlah minimum sampel untuk membagi node)
# - min_samples_leaf (jumlah minimum sampel di node daun)
# 
# *Tujuan:* 
# 
# - Menghasilkan model yang dapat menangani data non-linear dan interaksi antar fitur, meskipun perlu diwaspadai overfitting.
# 
# *Evaluation Metrics:*
# - ROC-AUC Score
# - F1-Score (terutama untuk kelas minoritas “alert”)
# - Confusion Matrix analysis
# 
# **📊 Metodologi Evaluasi**
# 1. Cross-validation (k=5) untuk mengevaluasi performa generalisasi.
# 2. SMOTE (Synthetic Minority Over-sampling Technique) untuk menangani data imbalance jika jumlah alert rendah.
# 3. Feature importance analysis untuk memberikan insight fitur mana yang paling berpengaruh.

# # Data Understanding
# 
# Data Source : https://www.kaggle.com/datasets/suvroo/ai-for-elderly-care-and-support?select=safety_monitoring.csv
# 
# Jumlah Data : 
# - daily_reminder -> 10.000 baris dan 6 kolom
# - health_monitoring -> 10.000 baris dan 10 kolom
# - safety_monitoring -> 10.000 baris dan 9 kolom
# 
# Kondisi Data : 
# - Terdapat `Unnamed` atau data kosong pada daily_monitoring dan safety_monitoring
# - Semua data daily_reminder masih berbentuk `object`
# - Semua data kategorikal masih berbentuk `object`
# - Terdapat data imbalanced pada `Alert_Triggered` dan `Fall_Detected_Counts`
# 
# 
# Dataset tersebut memiliki 4 komponen utama :
# - Health Data : Heart rate, blood pressure, glucose levels, and other vitals recorded from wearable devices.
# - Activity & Movement Tracking : Sensor-based logs of movement, fall detection, and inactivity periods.
# - Emergency Alerts : Records of triggered safety alerts due to unusual behavior or health anomalies.
# - Reminder Logs : Medication schedules, appointment reminders, and daily task notifications.
# This dataset enables AI-driven insights for personalized elderly care, predictive health monitoring, and automated reminders, ensuring safety, well-being, and improved quality of life for aging individuals.
# 
# Keempat komponen diatas terkandung dalam 3 file .csv yang bernama:
#    - `daily_reminder.csv`
#    - `health_monitoring.csv`
#    - `safety_monitoring.csv`
# 
# **`daily_reminder.csv`**
# - User ID
# - Timestamp
# - Reminder type : 'Appointment', 'Hydration', 'Other'
# - Scheduled Time
# - Reminder Sent : 'Yes' and 'No'
# - Acknowledged : 'Yes' and 'No'
# 
# **`health_monitoring.csv`**
# - User ID
# - Timestamp
# - Heart Rate : 60 - 120
# - Hear Rate Below/Above Threshold : 'Yes' and 'No'
# - Blood Pressure : integer
# - Blood Pressure Below/Above Threshold : 'Yes' and 'No'
# - Glucose Level : 70 - 150
# - Glucose Level Below/Above Threshold : 'Yes' and 'No'
# - Oxygen Saturation Level : 90 - 100
# - Oxygen Saturation Level Below/Above Threshold : 'Yes' and 'No'
# 
# **`safety_monitoring.csv`**
# - User ID
# - Timestamp
# - Movement Activity : 'Sitting', 'No Movement', 'Other'
# - Fall Detected : 'Yes' and 'No'
# - Impact Force Level : 'High', 'Medium', dan 'Low'
# - Post Fall Inactivity Duration (seconds)
# - Location : 'Bedroom', 'Bathroom', dan 'other'
# - Alert Triggered : 'True' and 'false'
# - Caregiver Notified : 'Yes' and 'No'

# # Data Preperation

# ## 1. Import Data

# In[1]:


# Import Dataset
import pandas as pd

df_daily = pd.read_csv('dataset/daily_reminder.csv')
df_health_monitor = pd.read_csv('dataset/health_monitoring.csv')
df_safety = pd.read_csv('dataset/safety_monitoring.csv')


# In[2]:


df_daily.head()


# In[3]:


df_daily.info()


# In[4]:


df_health_monitor.head()


# In[5]:


df_health_monitor.info()


# In[6]:


df_safety.head()


# In[7]:


df_safety.info()


# ## Exploratory Data Analysis

# Terdapat Fitur yang tidak seharusnya masuk ke dalam dataset, yaitu fitur `Unnamed`. Mari kita hapus terlebih dahulu sebelum digabungkan

# In[8]:


df_daily_temp = df_daily.dropna(how='all', axis="columns")
df_safety_temp = df_safety.dropna(how='all', axis="columns")

df_daily = df_daily_temp
df_safety = df_safety_temp

df_daily.info()
df_safety.info()


# Mari kita lihat terlebih dahulu untuk masing-masing dataset

# In[9]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Daily Reminder Data Exploration
print('Daily Reminder Data Columns:', df_daily.columns.tolist())
print(df_daily.head())

# Parsing timestamp columns
df_daily['Timestamp'] = pd.to_datetime(df_daily['Timestamp'], errors='coerce')
df_daily['Scheduled Time'] = pd.to_datetime(df_daily['Scheduled Time'], errors='coerce')

# Plot a countplot for 'Reminder Sent (Yes/No)'
plt.figure(figsize=(6,4))
sns.countplot(x='Reminder Sent (Yes/No)', data=df_daily, color='purple')
plt.title('Count of Reminder Sent Status')
plt.tight_layout()
plt.show()

# Display numeric columns if any
print(df_daily.select_dtypes(include=[np.number]).columns.tolist())


# In[10]:


# Health Monitoring Data Exploration
print('Health Monitoring Data Columns:', df_health_monitor.columns.tolist())
print(df_health_monitor.head())

# Convert Timestamp to datetime
df_health_monitor['Timestamp'] = pd.to_datetime(df_health_monitor['Timestamp'], errors='coerce')

# List of numeric columns for visualization
numeric_cols = ['Heart Rate', 'Glucose Levels', 'Oxygen Saturation (SpO₂%)']

# Histograms for numeric columns
plt.figure(figsize=(12,4))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, 3, i+1)
    sns.histplot(df_health_monitor[col].dropna(), kde=True)
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

# Pair plot
sns.pairplot(df_health_monitor[numeric_cols].dropna())
plt.suptitle('Pair Plot of Numeric Features', y=1.02)
plt.show()

# Count plot for Alert Triggered
plt.figure(figsize=(6,4))
sns.countplot(x='Alert Triggered (Yes/No)', data=df_health_monitor, color='yellow')
plt.title('Alert Triggered Counts')
plt.tight_layout()
plt.show()


# In[11]:


# Safety Monitoring Data Exploration
print('Safety Monitoring Data Columns:', df_safety.columns.tolist())
print(df_safety.head())

# Convert Timestamp to datetime
df_safety['Timestamp'] = pd.to_datetime(df_safety['Timestamp'], errors='coerce')

# Histogram for Post-Fall Inactivity Duration
plt.figure(figsize=(6,4))
sns.histplot(df_safety['Post-Fall Inactivity Duration (Seconds)'].dropna(), bins=20, kde=True)
plt.title('Post-Fall Inactivity Duration Distribution')
plt.tight_layout()
plt.show()

# Count plot for Fall Detected
plt.figure(figsize=(6,4))
sns.countplot(x='Fall Detected (Yes/No)', data=df_safety)
plt.title('Fall Detected Counts')
plt.tight_layout()
plt.show()


# Melihat hasil visualisasi data, kita dapat menggunakan `df_health_monitor` sebagai dataset kita dikarenakan berdasarkan **business understanding** poin 3, kita dapat membuat *early warning* system dengan dataset tersebut.

# In[12]:


df_health_monitor.describe()


# In[13]:


df_health_monitor.info()


# In[14]:


df_health_monitor.head()


# ## 2. Menyeleksi Data
# 
# Membersihkan data :
# 
# - Menyesuaikan bentuk data 
# - Mengatasi Nan value
# - Mengatasi outliers
# - Menghapus data yang tidak penting
# 
# ✅ To Do :
# 1. Melihat Unique values pada masing-masing features
# 4. Mengubah Blood Pressure menjadi float
# 5. Mengubah object menjadi categorical menggunakan metode Integer Encoding berdasarkan Unique Value

# In[15]:


# Melihat unique value
import pandas as pd

# Make sure df_main is a DataFrame
if isinstance(df_health_monitor, pd.DataFrame):
    for col in df_health_monitor.columns:
        print(f"Unique values in '{col}':")
        try:
            print(df_health_monitor[col].unique())
        except AttributeError as e:
            print(f"⚠️ Error for column '{col}': {e}")
        print("-" * 40)
else:
    print("⚠️ df_health_monitor is not a DataFrame!")


# In[16]:


# Menghapus User-ID
df_health_monitor = df_health_monitor.drop(columns=['Device-ID/User-ID'])
# Timestamp menjadi date time
#df_health_monitor = df_health_monitor.drop(columns=['Timestamp'])


# In[17]:


df_health_monitor.info()


# In[18]:


# 1. Memisahkan Sistolik dan Diastolik
df_health_monitor[['Sistolik', 'Diastolik']] = df_health_monitor['Blood Pressure'].str.extract(r'(\d+)/(\d+)').astype(int)

# 2. Binning Tekanan Darah
conditions = [
    # Normal: Sistolik < 120 DAN Diastolik < 80
    (df_health_monitor['Sistolik'] < 120) & (df_health_monitor['Diastolik'] < 80),
    
    # Tinggi: Sistolik 120-129 DAN Diastolik < 80
    (df_health_monitor['Sistolik'].between(120, 129)) & (df_health_monitor['Diastolik'] < 80),
    
    # Hipertensi Stage 1: Sistolik 130-139 ATAU Diastolik 80-90
    (df_health_monitor['Sistolik'].between(130, 139)) | (df_health_monitor['Diastolik'].between(80, 90)),
    
    # Hipertensi Stage 2: Sistolik ≥140 ATAU Diastolik ≥90
    (df_health_monitor['Sistolik'] >= 140) | (df_health_monitor['Diastolik'] >= 90),
    
    # Krisis Hipertensi: Sistolik >180 DAN/ATAU Diastolik >120
    (df_health_monitor['Sistolik'] > 180) | (df_health_monitor['Diastolik'] > 120)
]

categories = [
    'Normal',
    'Tinggi',
    'Hipertensi Stage 1',
    'Hipertensi Stage 2',
    'Krisis Hipertensi'
]

# 3. Membuat kolom kategori
df_health_monitor['Kategori_Tekanan_Darah'] = np.select(conditions, categories, default='Tidak Terdefinisi')

# 4. Hapus kolom numerik jika diperlukan
df_health_monitor = df_health_monitor.drop(columns=['Sistolik', 'Diastolik', 'Blood Pressure'])

# Hasil akhir
print(df_health_monitor)


# In[19]:


df_health_monitor.info()


# ## 3. Transformasi Data
# 
# Transformasi data berupa normalisasi, Standarisasi, dan Label Encoding apabila terdapat jenis data kategorikal untuk proses yang lebih lanjut. Melihat apakah terdapat outliers atau tidak

# In[20]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Buat copy dataframe
df_main_ob = df_health_monitor.copy()

# Inisialisasi LabelEncoder
le = LabelEncoder()

# Daftar untuk menyimpan mapping hasil encoding
encoding_mappings = {}

# Loop melalui semua kolom bertipe object
for column in df_main_ob.select_dtypes(include=['object']).columns:
    try:
        # Lakukan Label Encoding per kolom
        df_main_ob[column] = le.fit_transform(df_main_ob[column].astype(str))
        
        # Simpan mapping antara nilai asli dan encoded value
        encoding_mappings[column] = dict(zip(le.classes_, le.transform(le.classes_)))
        
        print(f"Kolom '{column}' telah diencode:")
        print(f"  Nilai unik: {encoding_mappings[column]}\n")
    except Exception as e:
        print(f"Gagal mengencode kolom {column}: {str(e)}")
        continue

# Tampilkan DataFrame hasil encoding
print("\nDataFrame setelah Label Encoding:")
print(df_main_ob.head())

# Tampilkan semua mapping
print("\nMapping Lengkap:")
for col, mapping in encoding_mappings.items():
    print(f"{col}: {mapping}")


# In[21]:


df_main_ob.columns


# ## 4. Data Spliting

# 📌 Struktur Kelompok yang Masuk Akal:
# 
# - Threshold fisiologis: Heart Rate, Blood Pressure, SpO₂
# - Respons sistem: Reminder, Acknowledgement, Notification
# - Aktivitas jatuh & dampak: Fall Detected, Impact Force, Inactivity
# - Kategori: Tekanan darah
# 
# Sejauh ini, kita bisa dapat membangun sistem keselamatan pasien dimana kita dapat membuat prediksi Deteksi Jatuh. Kenapa?
# 
# - Langsung menyangkut keselamatan hidup.
# - Datanya cukup lengkap.
# - Bisa digunakan untuk sistem alarm otomatis real-time.

# In[22]:


# Fitur prediktor
X = df_main_ob[[
    'Timestamp',
    'Heart Rate', 
    'Heart Rate Below/Above Threshold (Yes/No)',
    'Blood Pressure Below/Above Threshold (Yes/No)',
    'Glucose Levels',
    'Glucose Levels Below/Above Threshold (Yes/No)',
    'Oxygen Saturation (SpO₂%)',
    'SpO₂ Below Threshold (Yes/No)',
    'Kategori_Tekanan_Darah'
]].values

# Target
#y = df_cleaned['Fall Detected (Yes/No)'].values
y = df_main_ob['Alert Triggered (Yes/No)'].values


# In[23]:


from sklearn.model_selection import train_test_split

# Split dengan rasio 80:20 dan random_state untuk replikasi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# # Model Development
# 
# - Membuat model machine learning untuk menyelesaikan permasalahan.
# - Menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.
# - Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
# - Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. Jelaskan proses improvement yang dilakukan.
# - Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. Jelaskan mengapa memilih model tersebut sebagai model terbaik.
# 
# **Pilihan Model yang ingin digunakan**
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. K-Nearest Neighbors

# ## Logistic Regression

# In[24]:


import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

# Set plotting style
sns.set(style="whitegrid")

# Prepare data for prediction on health monitoring data
health_model_df = df_main_ob.copy()

# Engineering: extract hour from Timestamp
health_model_df['Hour'] = health_model_df['Timestamp'].dt.hour

# Define target variable: Alert Triggered (Yes/No) has been converted to binary already
target = 'Alert Triggered (Yes/No)'

# Define features: numeric vital signs and the hour
features = [
    'Heart Rate', 
    #'Heart Rate Below/Above Threshold (Yes/No)',
    #'Blood Pressure Below/Above Threshold (Yes/No)',
    #'Glucose Levels',
    #'Glucose Levels Below/Above Threshold (Yes/No)',
    'Oxygen Saturation (SpO₂%)',
    #'SpO₂ Below Threshold (Yes/No)',
    'Kategori_Tekanan_Darah']
    #'Hour']

# Drop rows with missing feature or target values
health_model_df = health_model_df.dropna(subset=features + [target])

# Split into X and y
X = health_model_df[features]
y = health_model_df[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training using Logistic Regression
model = LogisticRegression(      
    max_iter=10)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Measure Model Performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Prediction Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1 Score:  {f1:.4f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Permutation Importance
r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = r.importances_mean.argsort()

plt.figure(figsize=(6,4))
plt.barh(np.array(features)[sorted_idx], r.importances_mean[sorted_idx])
plt.xlabel('Mean Importance')
plt.title('Permutation Importance')
plt.tight_layout()
plt.show()


# Melihat performa base model yang belum baik, kita perlu mencari parameter atau Hyper Parameter Tuning. Oleh karena itu, kita perlu menggunakan Grid Search CV untuk mencari parameter terbaik untuk Model kita.

# In[25]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter' : [10, 100, 1000, 10000, 100000]
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)


# In[26]:


import seaborn as sns

# Additional libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

# Set plotting style
sns.set(style="whitegrid")

# Prepare data for prediction on health monitoring data
health_model_df = df_main_ob.copy()

# Engineering: extract hour from Timestamp
health_model_df['Hour'] = health_model_df['Timestamp'].dt.hour

# Define target variable: Alert Triggered (Yes/No) has been converted to binary already
target = 'Alert Triggered (Yes/No)'

# Define features: numeric vital signs and the hour
features = [
    'Heart Rate', 
    #'Heart Rate Below/Above Threshold (Yes/No)',
    #'Blood Pressure Below/Above Threshold (Yes/No)',
    #'Glucose Levels',
    #'Glucose Levels Below/Above Threshold (Yes/No)',
    'Oxygen Saturation (SpO₂%)',
    #'SpO₂ Below Threshold (Yes/No)',
    'Kategori_Tekanan_Darah']
    #'Hour']

# Drop rows with missing feature or target values
health_model_df = health_model_df.dropna(subset=features + [target])

# Split into X and y
X = health_model_df[features]
y = health_model_df[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training using Logistic Regression
model = LogisticRegression(      
    max_iter = 1000,
    C = 10,
    penalty='l1',
    solver='liblinear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Measure Model Performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Prediction Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1 Score:  {f1:.4f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Permutation Importance
r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = r.importances_mean.argsort()

plt.figure(figsize=(6,4))
plt.barh(np.array(features)[sorted_idx], r.importances_mean[sorted_idx])
plt.xlabel('Mean Importance')
plt.title('Permutation Importance')
plt.tight_layout()
plt.show()


# Terlihat dengan mencari parameter terbaik, akurasi dan kurva ROC terlihat lebih baik, tetapi masih belum menyentuk 90%. Peneliti ingin memastikan para lansia mendapatkan akurasi yang tinggi sehingga dapat menikmati kualitas pengawasan yang baik. Untuk itu mari kita coba algoritma lain, yaitu Decision Tree

# ## Decission Tree

# In[27]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Inisialisasi Model Decision Tree (Basemodel)
dt_model = DecisionTreeClassifier(
    max_depth=3,           
)

# 2. Training Model
dt_model.fit(X_train, y_train)

# 3. Predictions
y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]  # Probabilitas kelas positif

# 4. Evaluasi
# Measure Model Performance
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print(f'Decision Tree Accuracy: {accuracy_dt:.4f}')
print(f'Precision: {precision_dt:.4f}')
print(f'Recall:    {recall_dt:.4f}')
print(f'F1 Score:  {f1_dt:.4f}')

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6,4))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Decision Tree Confusion Matrix')
plt.tight_layout()
plt.show()

# ROC Curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

plt.figure(figsize=(6,4))
plt.plot(fpr_dt, tpr_dt, label=f'DT ROC (AUC = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Feature Importance (Natural pada Decision Tree)
feature_importance = dt_model.feature_importances_
sorted_idx_dt = np.argsort(feature_importance)

plt.figure(figsize=(6,4))
plt.barh(np.array(features)[sorted_idx_dt], feature_importance[sorted_idx_dt])
plt.xlabel('Importance Score')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.show()


# Terlihat dengan base model Decision Tree, model mendapatkan nilai ROC 90%. Sekarang mari kita coba menggunakan Grid Search CV untuk mendapatkan Hyper Parameter tuning yang lebih baik. Apakah kita dapat meningkatkan akurasinya.

# In[28]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],              # Kontrol kedalaman pohon untuk hindari overfitting
    'min_samples_split':[1, 2, 4, 6, 8, 10, 14, 20, 30, 100],      # Minimum sampel untuk split node
    'min_samples_leaf':[1, 2, 3, 4, 5],        # Minimum sampel di leaf node
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Inisialisasi Model Decision Tree (Basemodel)
dt_model = DecisionTreeClassifier(
    max_depth=5,               # Kontrol kedalaman pohon untuk hindari overfitting
    min_samples_split=100,      # Minimum sampel untuk split node
)

# 2. Training Model
dt_model.fit(X_train, y_train)

# 3. Predictions
y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]  # Probabilitas kelas positif

# 4. Evaluasi
# Measure Model Performance
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print(f'Decision Tree Accuracy: {accuracy_dt:.4f}')
print(f'Precision: {precision_dt:.4f}')
print(f'Recall:    {recall_dt:.4f}')
print(f'F1 Score:  {f1_dt:.4f}')

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6,4))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Decision Tree Confusion Matrix')
plt.tight_layout()
plt.show()

# ROC Curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

plt.figure(figsize=(6,4))
plt.plot(fpr_dt, tpr_dt, label=f'DT ROC (AUC = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Feature Importance (Natural pada Decision Tree)
feature_importance = dt_model.feature_importances_
sorted_idx_dt = np.argsort(feature_importance)

plt.figure(figsize=(6,4))
plt.barh(np.array(features)[sorted_idx_dt], feature_importance[sorted_idx_dt])
plt.xlabel('Importance Score')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.show()


# Yups!!! Ternyata dapat meningkatkan akurasi, walaupun hanya 1%, tetapi ini berarti baik. Para lansia dapat menikmati *Early Warning System* diatas 90%. Sehingga mereka dapat tenang dalam kegiatan sehari-hari.

# # Evaluation Results & Insights
# 
# **Berdasarkan evaluasi model:**
# 
# <h2>Logistic Regression:</h2>
# 
# Accuracy cukup tinggi (~88%) tetapi F1 Score untuk kelas "alert" rendah (<0.5), mengindikasikan bahwa model terlalu bias terhadap kelas mayoritas. False Negative tinggi, sehingga tidak cocok untuk implementasi *real-world*.
# 
# <h2>Decission Tree:</h2>
# 
# Meningkatkan F1-Score dan ROC-AUC secara signifikan (>0.75). Lebih seimbang dalam mengklasifikasi kelas minoritas. Penggunaan tuning hyperparameter memberikan hasil generalisasi yang lebih baik.
