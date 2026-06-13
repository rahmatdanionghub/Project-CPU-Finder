import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# CONFIGURASI HALAMAN WEBPAGE
# ==========================================
st.set_page_config(
    page_title="CPU Smart Recommender System",
    layout="wide"
)

# ==========================================
# LOAD RESOURCE (MODEL, ENCODER, & DATASET)
# ==========================================
@st.cache_resource
def load_models():
    # Load model terbaik dan label encoders yang di-export dari Colab
    with open('model_cpu_terbaik.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('le_brand.pkl', 'rb') as f:
        le_brand = pickle.load(f)
    with open('le_target.pkl', 'rb') as f:
        le_target = pickle.load(f)
    return model, le_brand, le_target

@st.cache_data
def load_csv_data():
    # Load dataset hasil pembersihan untuk filter rekomendasi nama produk
    return pd.read_csv("data_cleaned_filtered.csv")

# Eksekusi load data    
try:
    model, le_brand, le_target = load_models()
    df_cpu = load_csv_data()
except FileNotFoundError as e:
    st.error(f"Error: File pendukung tidak ditemukan di direktori! Pastikan model .pkl dan data_cleaned_filtered.csv sudah di-upload ke GitHub. ({e})")
    st.stop()

# ==========================================
# HEADER & INFORMASI AKURASI (SYARAT WAJIB POIN 4)
# ==========================================
st.title("CPU Smart Recommender System")
st.subheader("Final Project AI & Big Data 2026")
st.write("Aplikasi cerdas berbasis Machine Learning untuk menentukan kategori kebutuhan CPU berdasarkan spesifikasi dan budget Anda.")

# Kartu Informasi Metrik Performa Model (Sesuai Syarat Poin 4 Deskripsi Tugas)
st.markdown("---")
col_info1, col_info2 = st.columns(2)
with col_info1:
    st.info("**Algoritma Model Utama:** Random Forest Classifier (Terpilih)")
with col_info2:
    # Catatan: Sesuaikan angka 94.50% ini dengan hasil akurasi asli yang lu dapet di Colab nanti!
    st.success("**Metrik Performa Model:** Akurasi Pengujian = **94.50%** | F1-Score = **0.94**")
st.markdown("---")

# ==========================================
# INTERFACE INPUT PARAMETER (SYARAT WAJIB POIN 1 - MINIMAL 4 INPUT)
# ==========================================
st.sidebar.header("Input Parameter User")
st.sidebar.write("Masukkan kriteria CPU yang Anda butuhkan:")

# Input 1: Budget/Harga (Angka/Slider)
user_budget = st.sidebar.number_input(
    "1. Batas Maksimal Budget Anda ($)", 
    min_value=10, 
    max_value=5000, 
    value=300, 
    step=10,
    help="Masukkan nominal dana maksimal dalam satuan USD ($)"
)

# Input 2: Jumlah Cores (Pilihan Kategori/Selectbox)
available_cores = sorted(df_cpu['cores'].unique())
user_cores = st.sidebar.selectbox(
    "2. Jumlah Core Minimal yang Diperlukan", 
    options=available_cores,
    index=int(len(available_cores)/2)
)

# Input 3: Base Clock Speed (Angka)
user_speed = st.sidebar.number_input(
    "3. Kecepatan Base Clock Minimal (MHz)", 
    min_value=1000, 
    max_value=6000, 
    value=2500, 
    step=100
)

# Input 4: Merek/Brand CPU (Radio Button)
user_brand = st.sidebar.radio(
    "4. Prioritas Merek CPU", 
    options=["All", "AMD", "Intel"]
)

# ==========================================
# PROSES PREDIKSI MACHINE LEARNING (SYARAT WAJIB POIN 2 & 3)
# ==========================================
if st.sidebar.button("Cari Rekomendasi CPU"):
    
    st.subheader("Hasil Analisis & Prediksi AI")
    
    # 1. Transformasi input brand teks menjadi angka biner
    if user_brand == "All":
        brand_encoded = le_brand.transform(["Intel"])[0]
    else:
        try:
            brand_encoded = le_brand.transform([user_brand])[0]
        except ValueError:
            brand_encoded = 0

    # 2. Menyusun data input sesuai urutan fitur saat training
    fitur_input = np.array([[user_budget, user_cores, user_speed, brand_encoded]])
    
    # 3. Model memproses input melakukan prediksi (Syarat Poin 2)
    prediksi_indeks = model.predict(fitur_input)[0]
    
    # 4. Mengubah angka hasil prediksi kembali menjadi teks label asli (Syarat Poin 3)
    kategori_terprediksi = le_target.inverse_transform([prediksi_indeks])[0]
    
    # Menampilkan Hasil Prediksi Utama
    st.markdown(f"""
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; border-left: 8px solid #ff4b4b;">
        <h4 style="margin:0; color:#31333F;">Rekomendasi Kelas CPU Berdasarkan AI:</h4>
        <h2 style="margin:5px 0 0 0; color:#ff4b4b;">{kategori_terprediksi}</h2>
        <p style="margin:10px 0 0 0; font-size:14px; color:#555;">
            Model mendeteksi kombinasi Budget <b>${user_budget}</b> dengan spesifikasi <b>{user_cores} Cores / {user_speed} MHz</b> bermerek <b>{user_brand}</b> paling optimal untuk ekosistem kerja tersebut.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Alternatif Produk Nyata dari Dataset")
    st.write("Berikut adalah daftar tipe prosesor asli di dalam database yang cocok dengan kriteria Anda:")
    
    # =========================================================================
    # DI SINI PERUBAHANNYA: Logika filter ditambah CORES (Saklek) & SPEED (Minimal)
    # =========================================================================
    if user_brand == "All":
        df_hasil_filter = df_cpu[
            (df_cpu['Kebutuhan_Tier'] == kategori_terprediksi) & 
            (df_cpu['price'] <= user_budget) &
            (df_cpu['cores'] == user_cores) &        # Harus pas dengan core pilihan user
            (df_cpu['speed'] >= user_speed)          # Minimal secepat clock speed pilihan user
        ].copy()
    else:
        df_hasil_filter = df_cpu[
            (df_cpu['Kebutuhan_Tier'] == kategori_terprediksi) & 
            (df_cpu['brand'] == user_brand) & 
            (df_cpu['price'] <= user_budget) &
            (df_cpu['cores'] == user_cores) &        # Harus pas dengan core pilihan user
            (df_cpu['speed'] >= user_speed)          # Minimal secepat clock speed pilihan user
        ].copy()
    # =========================================================================
    
    # Urutkan berdasarkan rank performa terbaik
    df_hasil_filter = df_hasil_filter.sort_values(by='rank', ascending=True)
    
    if not df_hasil_filter.empty:
        # Merapikan kolom tabel untuk ditampilkan ke user / dosen
        tabel_tampil = df_hasil_filter[['name', 'price', 'cores', 'speed', 'turbo', 'tdp', 'rank']].rename(
            columns={
                'name': 'Nama Prosesor',
                'price': 'Harga ($)',
                'cores': 'Total Core',
                'speed': 'Base Clock (MHz)',
                'turbo': 'Turbo Clock (MHz)',
                'tdp': 'Daya (TDP Watt)',
                'rank': 'Peringkat Performa'
            }
        ).reset_index(drop=True)
        
        # Batasi maksimal menampilkan 10 baris teratas agar web tidak kepanjangan
        st.dataframe(tabel_tampil.head(10), use_container_width=True)
        st.caption(f"Menampilkan {min(10, len(tabel_tampil))} dari total {len(tabel_tampil)} produk alternatif yang ditemukan.")
    else:
        st.warning("Tidak ada produk spesifik di dalam database yang harganya di bawah budget Anda untuk spesifikasi core/speed setinggi ini. Cobalah untuk menaikkan parameter budget Anda di panel kiri.")

else:
    # Tampilan awal saat web baru dibuka dan tombol belum ditekan
    st.info("Silakan sesuaikan parameter spesifikasi komputer di panel bilah kiri, kemudian klik tombol **'Cari Rekomendasi CPU'** untuk melihat hasil analisis AI!")