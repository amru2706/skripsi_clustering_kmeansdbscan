# app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Judul aplikasi
st.title("📊 Aplikasi Clustering Pemakaian Air Tanah (KMeans & DBSCAN)")

# Upload file Excel
uploaded_file = st.file_uploader("📂 Upload file Excel (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    # Baca data
    df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
    st.write("### 📌 Data Awal", df.head())

    # Agregasi per kecamatan
    agg_df = df.groupby("nama_kecamatan")["jumlah_pemakaianairtanah"].agg(
        rata2_bulanan="mean",
        total_5tahun="sum",
        maksimum="max",
        minimum="min",
        std_dev="std"
    ).reset_index()

    st.write("### 📈 Data Agregat", agg_df)

    # Normalisasi & clustering
    features = ['rata2_bulanan', 'total_5tahun', 'maksimum', 'minimum', 'std_dev']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg_df[features])

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    agg_df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=2)
    agg_df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

    # Hasil akhir
    st.write("### 🎯 Hasil Clustering", agg_df)

    # Visualisasi KMeans
    st.write("### 🔍 Visualisasi KMeans Clustering")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(
        x=agg_df['rata2_bulanan'], y=agg_df['total_5tahun'],
        hue=agg_df['kmeans_cluster'], palette='Set2', s=100, ax=ax1
    )
    for i, row in agg_df.iterrows():
        ax1.text(row['rata2_bulanan'], row['total_5tahun'], row['nama_kecamatan'], fontsize=8)
    ax1.set_title("KMeans Clustering")
    st.pyplot(fig1)

    # Visualisasi DBSCAN
    st.write("### 🔍 Visualisasi DBSCAN Clustering")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x=agg_df['rata2_bulanan'], y=agg_df['total_5tahun'],
        hue=agg_df['dbscan_cluster'], palette='Dark2', s=100, ax=ax2
    )
    for i, row in agg_df.iterrows():
        ax2.text(row['rata2_bulanan'], row['total_5tahun'], row['nama_kecamatan'], fontsize=8)
    ax2.set_title("DBSCAN Clustering")
    st.pyplot(fig2)

    # Unduh hasil
    st.write("### 💾 Unduh Hasil")
    csv = agg_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download sebagai CSV", csv, file_name="hasil_klaster_airtanah.csv", mime='text/csv')
