import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import imblearn
import sklearn
import scipy
import io

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.tree import plot_tree
from scipy.stats import zscore

st.title("""PREDIKSI TELCO CUSTOMER CHURN""")
st.write("Nabila Atira Qurratul Aini")

tabs = st.tabs(["Business Understanding", "Data Understanding", "Data Preprocessing", "Modeling", "Evaluation", "Deployment", "Informasi"])
business_understanding, data_understanding, data_preprocessing, modeling, evaluation, deployment, informasi = tabs

with business_understanding:
    st.write("# BUSINESS UNDERSTANDING")

    st.write("### Latar Belakang")
    st.write("Industri telekomunikasi merupakan salah satu sektor yang sangat kompetitif, di mana perusahaan berlomba-lomba untuk mendapatkan dan mempertahankan pelanggan. Churn pelanggan, atau hilangnya pelanggan yang beralih ke penyedia layanan lain, adalah masalah utama yang dihadapi oleh perusahaan telekomunikasi. Churn tidak hanya mengakibatkan hilangnya pendapatan, tetapi juga meningkatkan biaya akuisisi pelanggan baru. Oleh karena itu, memahami dan memprediksi perilaku churn pelanggan sangat penting untuk menjaga pangsa pasar dan meningkatkan profitabilitas perusahaan.")

    st.write("### Tujuan")
    st.write("Tujuan dari penelitian ini adalah untuk mengembangkan model prediksi churn pelanggan dengan menggunakan data pelanggan Telco. Model ini bertujuan untuk mengidentifikasi pelanggan yang berisiko tinggi untuk churn dengan akurasi tinggi, sehingga perusahaan dapat mengambil tindakan proaktif untuk mempertahankan mereka. Selain itu, penelitian ini juga bertujuan untuk memahami faktor-faktor utama yang mempengaruhi churn pelanggan, sehingga perusahaan dapat merancang strategi retensi yang lebih efektif dan sesuai dengan kebutuhan pelanggan. Dengan pemahaman yang lebih baik tentang perilaku pelanggan, perusahaan dapat meningkatkan retensi, mengurangi biaya akuisisi pelanggan baru, dan pada akhirnya meningkatkan loyalitas serta profitabilitas.")

    st.write("### Manfaat")
    st.write("Penelitian ini diharapkan dapat mengidentifikasi pelanggan yang berisiko tinggi untuk churn, memungkinkan tindakan preventif untuk mengurangi churn. Selain itu, penelitian ini memberikan wawasan tentang faktor-faktor penyebab churn, membantu perusahaan memperbaiki layanan dan strategi pemasaran. Dengan meningkatkan retensi pelanggan, perusahaan dapat mengurangi biaya akuisisi pelanggan baru, serta meningkatkan pendapatan dan loyalitas pelanggan. Pada akhirnya, perusahaan dapat membangun hubungan yang lebih kuat dengan pelanggan, meningkatkan citra merek, dan memperkuat posisi kompetitif di pasar telekomunikasi yang semakin ketat.")

with data_understanding:
    st.write("# DATA UNDERSTANDING")
    
    st.write("### Pemahaman Data")
    st.write("Dataset ini terdiri dari 7043 baris (rows) dan 21 kolom (columns) yang berisi informasi tentang pelanggan perusahaan telekomunikasi. Data ini mencakup berbagai fitur demografis dan perilaku pelanggan, yang bertujuan untuk memprediksi apakah pelanggan akan berhenti berlangganan (churn) atau tidak. Informasi yang tersedia mencakup demografi pelanggan, layanan yang mereka gunakan, lama berlangganan, metode pembayaran, dan biaya yang mereka keluarkan.")
    
    st.write("### Fitur-fitur")
    st.write("- customerID : ID unik untuk setiap pelanggan, digunakan sebagai pengenal utama dalam dataset. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, memastikan setiap pelanggan memiliki identitas yang unik.")
    st.write("- gender : Jenis kelamin pelanggan, biasanya berisi nilai 'F', 'Female', 'M', atau 'Male'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, membantu dalam analisis demografis.")
    st.write("- SeniorCitizen : Menunjukkan apakah pelanggan merupakan warga senior, diwakili dengan nilai 1 (senior) atau 0 (non-senior). Data yang terkait dengan fitur ini memiliki tipe data numerik, memudahkan pengenalan kelompok usia yang lebih tua.")
    st.write("- Partner : Menunjukkan apakah pelanggan memiliki pasangan atau tidak, dengan nilai 'Yes' atau 'No'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, berguna dalam memahami kondisi rumah tangga pelanggan.")
    st.write("- Dependents : Menunjukkan apakah pelanggan memiliki tanggungan (anak-anak atau orang lain) dengan nilai 'Yes' atau 'No'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, membantu mengidentifikasi pelanggan dengan tanggungan.")
    st.write("- tenure : Lama waktu (dalam bulan) pelanggan telah berlangganan dengan perusahaan. Data yang terkait dengan fitur ini memiliki tipe data numerik, menunjukkan durasi keterlibatan pelanggan.")
    st.write("- PhoneService : Menunjukkan apakah pelanggan memiliki layanan telepon, dengan nilai 'Yes' atau 'No'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, mengindikasikan ketersediaan layanan telepon.")
    st.write("- MultipleLines : Menunjukkan apakah pelanggan memiliki lebih dari satu saluran telepon, dengan nilai 'Yes', 'No', atau 'No phone service'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, memberikan informasi tentang layanan telepon tambahan.")
    st.write("- InternetService : Jenis layanan internet yang digunakan oleh pelanggan, dengan nilai 'DSL', 'Fiber optic', atau 'No'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, menunjukkan jenis koneksi internet.")
    st.write("- OnlineSecurity : Menunjukkan apakah pelanggan memiliki layanan keamanan online, dengan nilai 'Yes', 'No', atau 'No internet service'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, menggambarkan penggunaan layanan keamanan online.")
    st.write("- OnlineBackup : Menunjukkan apakah pelanggan memiliki layanan pencadangan data online, dengan nilai 'Yes', 'No', atau 'No internet service'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, memberikan informasi tentang layanan pencadangan data.")
    st.write("- DeviceProtection : Menunjukkan apakah pelanggan memiliki perlindungan perangkat, dengan nilai 'Yes', 'No', atau 'No internet service'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, menunjukkan penggunaan layanan perlindungan perangkat.")
    st.write("- TechSupport : Menunjukkan apakah pelanggan memiliki layanan dukungan teknis, dengan nilai 'Yes', 'No', atau 'No internet service'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, menggambarkan penggunaan layanan dukungan teknis.")
    st.write("- StreamingTV : Menunjukkan apakah pelanggan memiliki layanan streaming TV, dengan nilai 'Yes', 'No', atau 'No internet service'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, memberikan informasi tentang layanan streaming TV.")
    st.write("- StreamingMovies : Menunjukkan apakah pelanggan memiliki layanan streaming film, dengan nilai 'Yes', 'No', atau 'No internet service'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, menggambarkan penggunaan layanan streaming film.")
    st.write("- Contract : Jenis kontrak yang diambil oleh pelanggan, dengan nilai 'Month-to-month', 'One year', atau 'Two year'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, menunjukkan durasi komitmen pelanggan.")
    st.write("- PaperlessBilling : Menunjukkan apakah pelanggan menggunakan penagihan tanpa kertas (elektronik), dengan nilai 'Yes' atau 'No'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, memberikan informasi tentang preferensi penagihan pelanggan.")
    st.write("- PaymentMethod : Metode pembayaran yang digunakan oleh pelanggan, seperti 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', atau 'Credit card (automatic)'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, menggambarkan cara pembayaran pelanggan.")
    st.write("- MonthlyCharges : Biaya bulanan yang dibayar oleh pelanggan untuk layanan mereka. Data yang terkait dengan fitur ini memiliki tipe data numerik, menunjukkan jumlah tagihan bulanan pelanggan.")
    st.write("- TotalCharges : Total biaya yang telah dibayar oleh pelanggan sejak mereka berlangganan. Data yang terkait dengan fitur ini memiliki tipe data numerik, memberikan informasi akumulatif tentang pengeluaran pelanggan.")
    st.write("- Churn : Menunjukkan apakah pelanggan berhenti berlangganan (churn) atau tidak, dengan nilai 'Yes' atau 'No'. Data yang terkait dengan fitur ini memiliki tipe data kategorikal, menjadi target utama dalam analisis prediksi churn.")

    data = pd.read_csv("https://raw.githubusercontent.com/NabilaAtiraQurratulAini/Dataset/main/Telco-Customer-Churn.csv")
    st.write("### Dataset Telco Customer Churn")
    data

    st.write("### Informasi Dataset")
    informasi_data = pd.DataFrame({"Column": data.columns, "Non-Null Count": [data[col].notnull().sum() for col in data.columns], "Dtype": [data[col].dtype for col in data.columns]})
    informasi_data

with data_preprocessing:
    st.write("# DATA PREPROCESSING")

    st.write("### Konversi Kolom TotalCharges Menjadi Tipe Data Numerik")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data

    st.write("### Data Missing Value")
    null_counts = data.isnull().sum().reset_index()
    null_counts.columns = ["Column", "Null Count"]
    st.dataframe(null_counts)

    st.write("### Data Setelah Pengisian Missing Value")
    modus_gender = data["gender"].mode()[0]
    data["gender"].fillna(modus_gender, inplace=True)
    mean_tenure = data["tenure"].mean()
    data["tenure"].fillna(mean_tenure, inplace=True)
    mean_totalcharges = data["TotalCharges"].mean()
    data["TotalCharges"].fillna(mean_totalcharges, inplace=True)
    data

    st.write("### Informasi Pengisian Nilai Hilang")
    st.write("Modus gender yang digunakan :", modus_gender)
    st.write("Rata-rata tenure yang digunakan :", mean_tenure)
    st.write("Rata-rata TotalCharges yang digunakan :", mean_totalcharges)

    st.write("### Data Missing Value")
    null_counts = data.isnull().sum().reset_index()
    null_counts.columns = ["Column", "Null Count"]
    st.dataframe(null_counts)

    st.write("### Dataset Setelah Penghapusan Kolom 'customerID'")
    data.drop('customerID', axis=1, inplace=True)
    data

    st.write("### Data Duplikat")
    num_duplicates = data.duplicated().sum()
    st.write("Jumlah baris duplikat :", num_duplicates)

    st.write("### Dataset Setelah Penghapusan Data Duplikat")
    data = data.drop_duplicates()
    data

    st.write("### Blox Plot")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for i, feature in enumerate(features):
        axs[i].boxplot(data[feature])
        axs[i].set_title(feature)
        axs[i].set_ylabel('Value')
        axs[i].set_xticklabels([feature], rotation=45)
    fig.suptitle('Boxplots of Features')
    st.pyplot(fig)

    st.write("### Data Outlier")
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    z_scores = np.abs(zscore(data[numeric_columns]))
    threshold = 3
    outliers = (z_scores > threshold).any(axis=1)
    jumlah_outlier = outliers.sum()
    jumlah_tanpa_outlier = len(data) - jumlah_outlier
    st.write("Jumlah outlier :", jumlah_outlier)
    st.write("Jumlah data tanpa outlier :", jumlah_tanpa_outlier)

    st.write("### Data Bersih")
    df = data[~outliers]
    df

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for i, feature in enumerate(features):
        axs[i].boxplot(df[feature])
        axs[i].set_title(feature)
        axs[i].set_ylabel('Value')
        axs[i].set_xticklabels([feature], rotation=45)
    fig.suptitle('Boxplots of Features')
    st.pyplot(fig)

    st.write("### Statistik Deskriptif")
    deskripsi_df = df.describe()
    st.dataframe(deskripsi_df)

    st.write("### Diagram Batang")
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'] 
    fig, axs = plt.subplots(4, 4, figsize=(15, 20))
    for i, column in enumerate(categorical_columns):
        row = i // 4
        col = i % 4
        counts = df[column].value_counts()
        axs[row, col].bar(counts.index, counts.values)
        axs[row, col].set_title(f'Distribution of {column}')
        axs[row, col].set_ylabel('Count')
        axs[row, col].tick_params(axis='x', rotation=45)
    for i in range(len(categorical_columns), 4*4):
        row = i // 4
        col = i % 4
        fig.delaxes(axs[row, col])
    plt.tight_layout()
    st.pyplot(fig)

    st.write("### Encoding")
    df['gender'].replace(['Female', 'F'], 0, inplace=True)
    df['gender'].replace(['Male', 'M'], 1, inplace=True)
    df['Partner'].replace('No', 0, inplace=True)
    df['Partner'].replace('Yes', 1, inplace=True)
    df['Dependents'].replace('No', 0, inplace=True)
    df['Dependents'].replace('Yes', 1, inplace=True)
    df['PhoneService'].replace('No', 0, inplace=True)
    df['PhoneService'].replace('Yes', 1, inplace=True)
    df['MultipleLines'].replace('No', 0, inplace=True)
    df['MultipleLines'].replace('Yes', 1, inplace=True)
    df['MultipleLines'].replace('No phone service', 2, inplace=True)
    df['InternetService'].replace('No', 0, inplace=True)
    df['InternetService'].replace('DSL', 1, inplace=True)
    df['InternetService'].replace('Fiber optic', 2, inplace=True)
    df['OnlineSecurity'].replace('No', 0, inplace=True)
    df['OnlineSecurity'].replace('Yes', 1, inplace=True)
    df['OnlineSecurity'].replace('No internet service', 2, inplace=True)
    df['OnlineBackup'].replace('No', 0, inplace=True)
    df['OnlineBackup'].replace('Yes', 1, inplace=True)
    df['OnlineBackup'].replace('No internet service', 2, inplace=True)
    df['DeviceProtection'].replace('No', 0, inplace=True)
    df['DeviceProtection'].replace('Yes', 1, inplace=True)
    df['DeviceProtection'].replace('No internet service', 2, inplace=True)
    df['TechSupport'].replace('No', 0, inplace=True)
    df['TechSupport'].replace('Yes', 1, inplace=True)
    df['TechSupport'].replace('No internet service', 2, inplace=True)
    df['StreamingTV'].replace('No', 0, inplace=True)
    df['StreamingTV'].replace('Yes', 1, inplace=True)
    df['StreamingTV'].replace('No internet service', 2, inplace=True)
    df['StreamingMovies'].replace('No', 0, inplace=True)
    df['StreamingMovies'].replace('Yes', 1, inplace=True)
    df['StreamingMovies'].replace('No internet service', 2, inplace=True)
    df['Contract'].replace('One year', 0, inplace=True)
    df['Contract'].replace('Two year', 1, inplace=True)
    df['Contract'].replace('Month-to-month', 2, inplace=True)
    df['PaperlessBilling'].replace('No', 0, inplace=True)
    df['PaperlessBilling'].replace('Yes', 1, inplace=True)
    df['PaymentMethod'].replace('Credit card (automatic)', 0, inplace=True)
    df['PaymentMethod'].replace('Bank transfer (automatic)', 1, inplace=True)
    df['PaymentMethod'].replace('Mailed check', 2, inplace=True)
    df['PaymentMethod'].replace('Electronic check', 3, inplace=True)
    df['Churn'].replace('No', 0, inplace=True)
    df['Churn'].replace('Yes', 1, inplace=True)
    st.write("Dataset setelah binary encoding :")
    df

    st.write("### Korelasi Matriks")
    correlation = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    correlation_matrix = correlation.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.title("Correlation Matrix")
    st.pyplot(fig)

    st.write("### Feature Selection")
    Y = df['Churn']
    X = df.drop(columns=['Churn'])
    importances = mutual_info_classif(X, Y)
    feat_importances = pd.Series(importances, X.columns)
    fig, ax = plt.subplots()
    feat_importances.plot(kind='barh', color='teal', ax=ax)
    st.pyplot(fig)

    st.write("### Rank Fitur")
    k_best = SelectKBest(mutual_info_classif, k='all')
    X_new = k_best.fit_transform(X, Y)
    feature_ranks = k_best.scores_
    feature_rank_df = pd.DataFrame({'Feature': X.columns, 'Rank': feature_ranks})
    feature_rank_df = feature_rank_df.sort_values(by='Rank', ascending=False)
    feature_rank_df

    st.write("### Korelasi Matriks")
    correlation = df[['tenure', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    correlation_matrix = correlation.corr()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.title("Correlation Matrix")
    st.pyplot(fig)

    st.write("### Normalisasi Data")
    st.write("Normalisasi data adalah proses untuk mengubah rentang data sehingga memiliki skala yang konsisten, yang penting untuk banyak algoritma machine learning. Salah satu metode normalisasi yang umum digunakan adalah Standard Scaler.")
    st.markdown("""
    Standard Scaler, juga dikenal sebagai Z-score normalization atau standar deviasi normalisasi, bekerja dengan cara berikut :
    - Mean-Centering (Pengaturan Rata-Rata ke Nol) : Data diubah sehingga rata-ratanya (mean) menjadi nol. Ini dilakukan dengan mengurangkan rata-rata dari setiap nilai data.
    - Scaling (Pengaturan Variansi) : Setelah mean diatur ke nol, data juga diubah sehingga memiliki deviasi standar (standard deviation) sebesar satu. Ini dicapai dengan membagi deviasi setiap nilai dari rata-rata dengan deviasi standar data.
    """)
    
    st.write("### Rumus Z-Score")
    st.write("Rumus untuk normalisasi menggunakan Standard Scaler adalah : ")
    st.latex(r"""
    z = \frac{\sigma}{x - \mu}
    """)

    st.markdown(r"""
    Keterangan :
    - 𝑧 adalah Z-score dari nilai individual. Z-score menunjukkan seberapa jauh dan dalam arah apa (positif atau negatif) sebuah nilai berada dari rata-rata dataset dalam satuan deviasi standar. Z-score adalah hasil akhir dari normalisasi.
    - 𝜎 adalah Deviasi standar (standard deviation) dari dataset. Deviasi standar mengukur sebaran data di sekitar rata-rata. Semakin besar deviasi standar, semakin besar penyebaran data.
    - 𝑥 adalah Nilai individual dari data yang ingin dinormalisasi. Ini adalah nilai asli yang akan dikonversi menjadi Z-score.
    - 𝜇 adalah Rata-rata (mean) dari dataset. Rata-rata adalah jumlah semua nilai data dibagi dengan jumlah total data. Ini adalah pusat distribusi data.
    """)

    st.write("### Dataset Hasil Normalisasi")
    scaler = StandardScaler()
    fitur = ["tenure", "MonthlyCharges", "TotalCharges"]
    df.loc[:, fitur] = scaler.fit_transform(df[fitur])
    df

    st.write("### Label Data Churn")
    churn_counts = df["Churn"].value_counts()
    churn_counts

    st.write("### Imbalanced Resampling")
    st.write("Imbalanced resampling adalah teknik yang digunakan untuk menangani ketidakseimbangan kelas dalam dataset, di mana satu kelas memiliki lebih banyak contoh dibandingkan kelas lainnya. Teknik ini bertujuan untuk menyeimbangkan distribusi kelas, sehingga model machine learning dapat belajar dengan lebih baik dan memberikan kinerja yang lebih baik pada kedua kelas.")
    
    st.write("### Metode SMOTE (Synthetic Minority Over-sampling Technique)")
    st.write("SMOTE adalah metode oversampling yang digunakan untuk menghasilkan contoh sintetik bagi kelas minoritas. Alih-alih menduplikasi contoh yang ada, SMOTE membuat contoh baru dengan menggunakan interpolasi antara contoh yang ada dalam kelas minoritas.")
    
    st.write("### Rumus SMOTE")
    st.write("Interpolasi sintetik dalam SMOTE menggunakan rumus berikut : ")
    st.latex(r"""
    x_{\text{new}} = x_i + \theta \cdot (x_j - x_i)
    """)

    st.markdown(r"""
    Keterangan :
    - 𝑥 new adalah contoh sintetik baru. Ini adalah contoh baru yang dihasilkan oleh SMOTE untuk kelas minoritas, yang berada di antara contoh asli 𝑥𝑖 dan tetangga terdekat 𝑥𝑗.
    - 𝑥𝑖 adalah contoh asli dari kelas minoritas. Ini adalah titik data yang ada dalam dataset asli dan digunakan sebagai basis untuk pembuatan contoh sintetik.
    - 𝑥𝑗 adalah tetangga terdekat dari 𝑥𝑖 dalam kelas minoritas. Ini adalah titik data yang paling dekat dengan 𝑥𝑖 di kelas minoritas, yang digunakan untuk membuat contoh sintetik. 𝑥𝑗 dipilih dari k-nearest neighbors (kNN) dari 𝑥𝑖.
    - 𝜃 adalah nilai acak antara 0 dan 1. Ini adalah parameter interpolasi yang digunakan untuk menentukan posisi contoh sintetik baru antara 𝑥𝑖 dan 𝑥𝑗. Nilai 𝜃 diambil secara acak dari distribusi uniform dalam rentang [0, 1].
    """)

    st.write("### Dataset Hasil Resampling")
    X = df.drop(columns=["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "Churn"])
    y = df["Churn"]
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    X_size = X.shape[0]
    st.write("Jumlah baris data setelah diresampling :", X_size)

    st.write("### Data Duplikat Hasil Resampling")
    num_duplicates = df.duplicated().sum()
    st.write("Jumlah data duplikat :", num_duplicates)

    st.write("### Total Churn")
    total_churn = y.value_counts()
    total_churn

    st.write("### Split Data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    st.write("Jumlah data train :", train_size)
    st.write("Jumlah data test :", test_size)

with modeling:
    st.write("# MODELING")

    st.write("### Decision Tree")
    st.write("Decision Tree adalah algoritma pembelajaran mesin yang digunakan untuk klasifikasi dan regresi. Model ini membagi dataset menjadi subset berdasarkan fitur untuk membuat keputusan yang dapat digunakan untuk memprediksi hasil. Struktur pohon terdiri dari node, cabang, dan daun :")
    st.markdown(r"""
    - Akar (Root) adalah node pertama yang mewakili seluruh dataset.
    - Node adalah titik keputusan yang membagi data berdasarkan fitur tertentu.
    - Cabang (Branch) adalah garis yang menghubungkan node, menunjukkan hasil dari keputusan yang dibuat di node.
    - Daun (Leaf) adalah node terminal yang memberikan hasil akhir (kelas untuk klasifikasi, nilai untuk regresi).
    """)

    st.write("### Rumus Decision Tree Entropy")
    st.write("Entropy mengukur ketidakpastian atau kekacauan dalam dataset :")
    st.latex(r"""
    \text{Entropy}(S) = - \sum_{i=1}^{c} p_i \log_{2} p_i
    """)

    st.markdown(r"""
    Keterangan :
    - 𝑝𝑖 adalah proporsi dari kelas 𝑖 dalam subset data 𝑆. Proporsi ini dihitung sebagai rasio jumlah contoh dari kelas 𝑖 terhadap jumlah total contoh dalam subset data 𝑆.
    - 𝑐 adalah jumlah kelas dalam subset data 𝑆. Ini menunjukkan total kategori atau kelas yang berbeda yang ada dalam subset data.
    """)

    dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    st.write("### Matriks Evaluasi")
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    precision_dt = precision_score(y_test, y_pred_dt)
    recall_dt = recall_score(y_test, y_pred_dt)
    f1_dt = f1_score(y_test, y_pred_dt)

    metrics_df_dt = pd.DataFrame({"Akurasi": [accuracy_dt], "Precision": [precision_dt], "Recall": [recall_dt], "F1-Score": [f1_dt]})
    metrics_df_dt

    st.write("### Tabel Prediksi")
    dt_results_df = pd.DataFrame({"Actual Label": y_test, "Prediksi Decision Tree": y_pred_dt})
    dt_results_df

    st.write("### Confusion Matriks")
    confusion_matrix_dt = confusion_matrix(y_test, y_pred_dt)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_dt, annot=True, fmt="d", cmap="Blues", xticklabels=dt_model.classes_, yticklabels=dt_model.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")
    st.pyplot(plt)

    st.write("### Naive Bayes Gaussian")
    st.write("Naive Bayes Gaussian adalah salah satu varian dari algoritma Naive Bayes yang digunakan khusus untuk data kontinu. Model ini berasumsi bahwa fitur-fitur dalam dataset mengikuti distribusi Gaussian (normal) dalam setiap kelas.")
    
    st.write("### Konsep Dasar Naive Bayes Gaussian")
    st.markdown(r"""
    Naive Bayes Gaussian mengasumsikan bahwa :
    - Fitur-fitur bersifat kontinu dan mengikuti distribusi Gaussian.
    - Fitur-fitur independen secara kondisional terhadap kelas yang diberikan (asumsi Naive Bayes).
    """)

    st.write("### Rumus Naive Bayes Gaussian")
    st.write("Teorema Bayes digunakan untuk menghitung probabilitas posterior dari kelas 𝐶𝑘 berdasarkan fitur 𝑋 :")
    st.latex(r"""
    P(C_k \mid X) = \frac{P(X \mid C_k) \cdot P(C_k)}{P(X)}
    """)

    st.markdown(r"""
    Keterangan :
    - 𝑃(𝐶𝑘 ∣ 𝑋) adalah probabilitas bahwa contoh data 𝑋 termasuk dalam kelas 𝐶𝑘, setelah mempertimbangkan informasi dari fitur 𝑋.
    - 𝑃(𝑋 ∣ 𝐶𝑘) adalah probabilitas terjadinya fitur 𝑋 ketika kelas yang sebenarnya adalah 𝐶𝑘.
    - 𝑃(𝐶𝑘) adalah probabilitas awal dari kelas 𝐶𝑘 sebelum melihat fitur 𝑋.
    - 𝑃(𝑋) adalah probabilitas bahwa fitur 𝑋 terjadi secara keseluruhan.
    """)

    gnb_model = GaussianNB(priors=None, var_smoothing=1e-9)
    gnb_model.fit(X_train, y_train)
    y_pred_gnb = gnb_model.predict(X_test)

    st.write("### Matriks Evaluasi")
    accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
    precision_gnb = precision_score(y_test, y_pred_gnb)
    recall_gnb = recall_score(y_test, y_pred_gnb)
    f1_gnb = f1_score(y_test, y_pred_gnb)

    metrics_df_gnb = pd.DataFrame({"Akurasi": [accuracy_gnb], "Precision": [precision_gnb], "Recall": [recall_gnb], "F1-Score": [f1_gnb]})
    metrics_df_gnb

    st.write("### Tabel Prediksi")
    gnb_results_df = pd.DataFrame({"Actual Label": y_test, "Prediksi Naive Bayes Gaussian": y_pred_gnb})
    gnb_results_df

    st.write("### Confusion Matriks")
    confusion_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_gnb, annot=True, fmt="d", cmap="Blues", xticklabels=gnb_model.classes_, yticklabels=gnb_model.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")
    st.pyplot(plt)

with evaluation:
    st.write("# EVALUATION")

    st.write("### Visualisasi Pohon Keputusan")
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt_model, feature_names=X.columns, class_names=["Not Churn", "Churn"], filled=True, rounded=True, fontsize=10, ax=ax)
    plt.title("Pohon Keputusan")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    st.image(buf, caption="Pohon Keputusan", use_column_width=True)

    results = permutation_importance(dt_model, X_test, y_test, scoring="accuracy")
    importances = results.importances_mean
    feature_names = X.columns.tolist()
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance using Permutation Importance Decision Tree")
    ax.invert_yaxis()
    st.pyplot(fig)
    st.dataframe(importance_df)
    
    st.write("### Feature Importance Naive Bayes Gaussian")
    results = permutation_importance(gnb_model, X_test, y_test, scoring="accuracy")
    importances = results.importances_mean
    feature_names = X.columns.tolist()
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance using Permutation Importance Naive Bayes Gaussian")
    ax.invert_yaxis()
    st.pyplot(fig)
    st.dataframe(importance_df)

    st.write("# REKOMENDASI")
    st.write("### Fitur 'Contract'")
    st.markdown(r"""
    Rekomendasi untuk mengatasi masalah 'Churn' terkait fitur 'Contract' :
    - Memberikan penawaran khusus, diskon atau paket tambahan kepada pelanggan yang memiliki kontrak agar mereka tetap setia.
    - Memastikan bahwa pelanggan dengan kontrak mendapatkan layanan yang terbaik.
    - Memberikan edukasi kepada pelanggan tentang manfaat menggunakan layanan ini dalam jangka panjang.
    - Mengembangkan produk atau paket yang lebih menarik untuk pelanggan dengan kontrak.
    """)

    st.write("### Fitur 'OnlineSecurity'")
    st.markdown(r"""
    Rekomendasi untuk mengatasi masalah 'Churn' terkait fitur 'OnlineSecurity' :
    - Menyediakan konten edukatif secara berkala tentang pentingnya keamanan online dan bagaimana pelanggan dapat melindungi diri mereka sendiri saat menggunakan layanan.
    - Menawarkan paket keamanan tambahan dengan fitur-fitur yang lebih canggih untuk pelanggan yang menginginkan perlindungan ekstra.
    - Menyediakan layanan pemantauan keamanan online 24 jam sehari untuk memberikan ketenangan pikiran kepada pelanggan.
    - Membagikan testimoni pelanggan dan studi kasus tentang bagaimana layanan keamanan telah membantu mereka.
    """)

    st.write("### Fitur 'tenure'")
    st.markdown(r"""
    Rekomendasi untuk mengatasi masalah 'Churn' terkait fitur 'tenure' :
    - Memperkenalkan program poinloyalitas dimana pelanggan mendapatkan poin untuk setiap bulan berlangganan yang dapat ditukar dengan layanan gratis.
    - Menawarkan paket berlangganan jangka panjang dengan harga yang lebih murah per bulan dibandingkan paket bulanan.
    - Memberikan diskon atau penawaran khusus kepada pelanggan yang sudah lama berlanganan.
    - Lebih aktif dalam menangani keluhan pelanggan sebelum mereka memutuskan untuk berhenti.
    """)

with deployment:
    st.write("# APLIKASI PREDIKSI CUSTOMER CHURN")

    def preprocess_input(data, feature_names):
        missing_cols = set(feature_names) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
        data = data[feature_names]
        return data

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=0)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=0.0)

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "gender_Male": [1 if gender == "Male" else 0],
        "SeniorCitizen": [senior_citizen],
        "Partner_Yes": [1 if partner == "Yes" else 0],
        "Dependents_Yes": [1 if dependents == "Yes" else 0],
        "PhoneService_Yes": [1 if phone_service == "Yes" else 0],
        "MultipleLines_No phone service": [1 if multiple_lines == "No phone service" else 0],
        "MultipleLines_Yes": [1 if multiple_lines == "Yes" else 0],
        "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
        "InternetService_No": [1 if internet_service == "No" else 0],
        "OnlineSecurity_Yes": [1 if online_security == "Yes" else 0],
        "OnlineSecurity_No internet service": [1 if online_security == "No internet service" else 0],
        "OnlineBackup_Yes": [1 if online_backup == "Yes" else 0],
        "OnlineBackup_No internet service": [1 if online_backup == "No internet service" else 0],
        "DeviceProtection_Yes": [1 if device_protection == "Yes" else 0],
        "DeviceProtection_No internet service": [1 if device_protection == "No internet service" else 0],
        "TechSupport_Yes": [1 if tech_support == "Yes" else 0],
        "TechSupport_No internet service": [1 if tech_support == "No internet service" else 0],
        "StreamingTV_Yes": [1 if streaming_tv == "Yes" else 0],
        "StreamingTV_No internet service": [1 if streaming_tv == "No internet service" else 0],
        "StreamingMovies_Yes": [1 if streaming_movies == "Yes" else 0],
        "StreamingMovies_No internet service": [1 if streaming_movies == "No internet service" else 0],
        "Contract_One year": [1 if contract == "One year" else 0],
        "Contract_Two year": [1 if contract == "Two year" else 0],
        "PaperlessBilling_Yes": [1 if paperless_billing == "Yes" else 0],
        "PaymentMethod_Credit card (automatic)": [1 if payment_method == "Credit card (automatic)" else 0],
        "PaymentMethod_Electronic check": [1 if payment_method == "Electronic check" else 0],
        "PaymentMethod_Mailed check": [1 if payment_method == "Mailed check" else 0]
    })

    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[X.columns]

    def predict(model, data):
        prediction = model.predict(data)
        probability = model.predict_proba(data)[0][1]
        return prediction, probability

    if st.button("Predict Churn"):
        input_data = preprocess_input(input_data, X.columns)
        prediction_dt, probability_dt = predict(dt_model, input_data)
        st.write("### Prediksi Decision Tree")
        if prediction_dt[0] == 1:
            st.write("Hasil Prediksi : Customer Churn")
            st.write("Pelanggan kemungkinan besar akan berhenti menggunakan layanan dengan probabilitas", probability_dt)
        else:
            st.write("Hasil Prediksi : Customer Tidak Churn")
            st.write("Pelanggan kemungkinan besar akan tetap menggunakan layanan dengan probabilitas :", probability_dt)
        
        prediction_gnb, probability_gnb = predict(gnb_model, input_data)
        st.write("### Prediksi Gaussian Naive Bayes")
        if prediction_gnb[0] == 1:
            st.write("Hasil Prediksi : Customer Churn")
            st.write("Pelanggan kemungkinan besar akan berhenti menggunakan layanan dengan probabilitas :", probability_gnb)
        else:
            st.write("Hasil Prediksi : Customer Tidak Churn")
            st.write("Pelanggan kemungkinan besar akan tetap menggunakan layanan dengan probabilitas :", probability_gnb)

with informasi:
    st.write("# INFORMASI DATASET")
    
    st.write("### Sumber Dataset di GitHub")
    st.write("https://github.com/NabilaAtiraQurratulAini/Dataset/blob/main/Telco-Customer-Churn.csv")

    st.write("### Source Code di GitHub")
    st.write("https://github.com/NabilaAtiraQurratulAini/Prediksi-Telco-Customer-Churn.git")
