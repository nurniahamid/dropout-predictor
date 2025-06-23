import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# --- 1. Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Prediksi Dropout Mahasiswa Jaya-jaya Institute", layout="wide")

st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.markdown("Aplikasi ini memprediksi kemungkinan seorang mahasiswa akan *dropout* berdasarkan berbagai faktor demografi dan akademik.")

# --- 2. Pemuatan Model ---
# Pastikan jalur ke model benar relatif terhadap `app.py`
model_path = 'model/best_model.pkl'
try:
    pipeline_model = joblib.load(model_path)
    st.sidebar.success(f"Model '{model_path}' berhasil dimuat!")
except FileNotFoundError:
    st.sidebar.error(f"Error: Model tidak ditemukan di {model_path}. Pastikan model sudah tersimpan di folder 'model/'.")
    st.stop() # Hentikan eksekusi jika model tidak ditemukan
except Exception as e:
    st.sidebar.error(f"Error saat memuat model: {e}")
    st.stop()

# --- 3. Definisi Fitur dan Preprocessor (HARUS SAMA DENGAN SAAT TRAINING) ---
# Daftar kolom kategorikal dan numerik yang digunakan di `X_train`
cat_cols = [
    'Application_mode',
    'Course',
    'Daytime_evening_attendance',
    'Previous_qualification',
    'Debtor',
    'Gender',
    'Scholarship_holder',
    'Marital_status',
    'Mothers_qualification',
    'Fathers_qualification',
    'Mothers_occupation',
    'Fathers_occupation',
    'Displaced',
    'Tuition_fees_up_to_date'
]

num_cols = ['Age_at_enrollment']

# --- 4. Fungsi untuk Membuat Input UI ---
st.subheader("Masukkan Data Mahasiswa:")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Usia saat Pendaftaran", min_value=17, max_value=70, value=20)
    gender = st.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
    marital_status = st.selectbox("Status Perkawinan", options=[1, 2, 3, 4, 5, 6], format_func=lambda x: {1: 'Single', 2: 'Menikah', 3: 'Duda/Janda', 4: 'Cerai', 5: 'Serikat Fakta', 6: 'Pisah Hukum'}.get(x, f'Kode {x}'))
    application_mode = st.selectbox("Mode Aplikasi", options=[1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57], format_func=lambda x: {
        1: 'Fase 1 - Umum', 2: 'Ordinansi 612/93', 5: 'Fase 1 - Azores', 7: 'Kursus Pendidikan Tinggi Lain',
        10: 'Ordinansi 854-B/99', 15: 'Mahasiswa Internasional', 16: 'Fase 1 - Madeira', 17: 'Fase 2 - Umum',
        18: 'Fase 3 - Umum', 26: 'Rencana Berbeda', 27: 'Institusi Lain', 39: 'Usia > 23 tahun',
        42: 'Transfer', 43: 'Perubahan Kursus', 44: 'Pemegang Diploma Spesialisasi Teknologi',
        51: 'Perubahan Institusi/Kursus', 53: 'Pemegang Diploma Siklus Pendek', 57: 'Perubahan Institusi/Kursus (Int.)'
    }.get(x, f'Kode {x}'))

with col2:
    course = st.selectbox("Kursus", options=[33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991], format_func=lambda x: {
        33: 'Teknologi Produksi Biofuel', 171: 'Desain Animasi & Multimedia', 8014: 'Pelayanan Sosial (Malam)',
        9003: 'Agronomi', 9070: 'Desain Komunikasi', 9085: 'Keperawatan Hewan', 9119: 'Teknik Informatika',
        9130: 'Equinculture', 9147: 'Manajemen', 9238: 'Pelayanan Sosial', 9254: 'Pariwisata',
        9500: 'Keperawatan', 9556: 'Kebersihan Mulut', 9670: 'Manajemen Periklanan & Pemasaran',
        9773: 'Jurnalisme & Komunikasi', 9853: 'Pendidikan Dasar', 9991: 'Manajemen (Malam)'
    }.get(x, f'Kode {x}'))
    daytime_evening_attendance = st.selectbox("Kehadiran", options=[0, 1], format_func=lambda x: "Malam" if x == 0 else "Siang")
    previous_qualification = st.selectbox("Kualifikasi Sebelumnya", options=[1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43], format_func=lambda x: {
        1: 'Pendidikan Menengah', 2: 'Pendidikan Tinggi - Sarjana', 3: 'Pendidikan Tinggi - Gelar',
        4: 'Pendidikan Tinggi - Magister', 5: 'Pendidikan Tinggi - Doktor', 6: 'Pendidikan Tinggi Berlangsung',
        9: 'Kelas 12 - Tidak Selesai', 10: 'Kelas 11 - Tidak Selesai', 12: 'Lainnya - Kelas 11',
        14: 'Kelas 10', 15: 'Kelas 10 - Tidak Selesai', 19: 'Pendidikan Dasar Siklus 3',
        38: 'Pendidikan Dasar Siklus 2', 39: 'Kursus Spesialisasi Teknologi',
        40: 'Pendidikan Tinggi - Gelar (Siklus 1)', 42: 'Kursus Teknis Tinggi Profesional',
        43: 'Pendidikan Tinggi - Magister (Siklus 2)'
    }.get(x, f'Kode {x}'))
    debtor = st.selectbox("Debitur (Punya Tunggakan)", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    scholarship_holder = st.selectbox("Penerima Beasiswa", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

with col3:
    mothers_qualification = st.selectbox("Kualifikasi Ibu", options=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44], format_func=lambda x: {
        1: 'Pendidikan Menengah - Kelas 12 atau Setara', 2: 'Pendidikan Tinggi - Sarjana',
        3: 'Pendidikan Tinggi - Gelar', 4: 'Pendidikan Tinggi - Magister',
        5: 'Pendidikan Tinggi - Doktor', 6: 'Pendidikan Tinggi Berlangsung',
        9: 'Kelas 12 - Tidak Selesai', 10: 'Kelas 11 - Tidak Selesai', 11: 'Kelas 7 (Lama)',
        12: 'Lainnya - Kelas 11', 14: 'Kelas 10', 18: 'Kursus Umum Perdagangan',
        19: 'Pendidikan Dasar Siklus 3', 22: 'Kursus Teknis-Profesional',
        26: 'Kelas 7', 27: 'Siklus 2 Kursus SMA Umum', 29: 'Kelas 9 - Tidak Selesai',
        30: 'Kelas 8', 34: 'Tidak Diketahui', 35: 'Tidak Bisa Membaca/Menulis',
        36: 'Bisa Membaca Tanpa Kelas 4', 37: 'Pendidikan Dasar Siklus 1 (Kelas 4/5)',
        38: 'Pendidikan Dasar Siklus 2 (Kelas 6-8)', 39: 'Kursus Spesialisasi Teknologi',
        40: 'Pendidikan Tinggi - Gelar (Siklus 1)', 41: 'Kursus Studi Tinggi Spesialisasi',
        42: 'Kursus Teknis Tinggi Profesional', 43: 'Pendidikan Tinggi - Magister (Siklus 2)',
        44: 'Pendidikan Tinggi - Doktor (Siklus 3)'
    }.get(x, f'Kode {x}'))

    fathers_qualification = st.selectbox("Kualifikasi Ayah", options=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44], format_func=lambda x: {
        1: 'Pendidikan Menengah - Kelas 12 atau Setara', 2: 'Pendidikan Tinggi - Sarjana',
        3: 'Pendidikan Tinggi - Gelar', 4: 'Pendidikan Tinggi - Magister',
        5: 'Pendidikan Tinggi - Doktor', 6: 'Pendidikan Tinggi Berlangsung',
        9: 'Kelas 12 - Tidak Selesai', 10: 'Kelas 11 - Tidak Selesai', 11: 'Kelas 7 (Lama)',
        12: 'Lainnya - Kelas 11', 13: 'Kursus SMA Komplementer Kelas 2', 14: 'Kelas 10',
        18: 'Kursus Umum Perdagangan', 19: 'Pendidikan Dasar Siklus 3', 20: 'Kursus SMA Komplementer',
        22: 'Kursus Teknis-Profesional', 25: 'SMA Komplementer - Tidak Selesai', 26: 'Kelas 7',
        27: 'Siklus 2 Kursus SMA Umum', 29: 'Kelas 9 - Tidak Selesai', 30: 'Kelas 8',
        31: 'Kursus Administrasi & Perdagangan', 33: 'Akuntansi & Administrasi Suplementer',
        34: 'Tidak Diketahui', 35: 'Tidak Bisa Membaca/Menulis',
        36: 'Bisa Membaca Tanpa Kelas 4', 37: 'Pendidikan Dasar Siklus 1 (Kelas 4/5)',
        38: 'Pendidikan Dasar Siklus 2 (Kelas 6-8)', 39: 'Kursus Spesialisasi Teknologi',
        40: 'Pendidikan Tinggi - Gelar (Siklus 1)', 41: 'Kursus Studi Tinggi Spesialisasi',
        42: 'Kursus Teknis Tinggi Profesional', 43: 'Pendidikan Tinggi - Magister (Siklus 2)',
        44: 'Pendidikan Tinggi - Doktor (Siklus 3)'
    }.get(x, f'Kode {x}'))
    
    mothers_occupation = st.selectbox("Pekerjaan Ibu", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125, 131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173, 175, 191, 192, 193, 194], format_func=lambda x: {
        0: 'Pelajar', 1: 'Perwakilan Legislatif/Eksekutif/Direktur', 2: 'Spesialis Aktivitas Intelektual/Ilmiah',
        3: 'Teknisi', 4: 'Staf Administrasi', 5: 'Pekerja Layanan/Keamanan/Penjual',
        6: 'Pertanian/Perikanan/Kehutanan', 7: 'Industri/Konstruksi/Pengrajin', 8: 'Operator Mesin',
        9: 'Pekerja Tidak Terampil', 10: 'Angkatan Bersenjata', 90: 'Lainnya', 99: 'Kosong',
        122: 'Profesional Kesehatan', 123: 'Guru', 125: 'Spesialis TIK',
        131: 'Teknisi Ilmu Pengetahuan/Teknik (Menengah)', 132: 'Teknisi Kesehatan (Menengah)',
        134: 'Teknisi Hukum/Sosial/Olahraga/Budaya (Menengah)', 141: 'Pekerja Kantor/Sekretaris/Operator Data',
        143: 'Operator Keuangan/Statistik/Data', 144: 'Staf Pendukung Administrasi Lain',
        151: 'Layanan Pribadi', 152: 'Penjual', 153: 'Perawatan Pribadi', 171: 'Pekerja Konstruksi Terampil',
        173: 'Pencetak/Perhiasan/Pengrajin', 175: 'Industri Makanan/Kayu/Tekstil', 191: 'Pekerja Pembersih',
        192: 'Pekerja Tidak Terampil Pertanian/Perikanan/Kehutanan', 193: 'Pekerja Tidak Terampil Industri/Transportasi',
        194: 'Asisten Persiapan Makanan'
    }.get(x, f'Kode {x}'))

    fathers_occupation = st.selectbox("Pekerjaan Ayah", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 101, 102, 103, 112, 114, 121, 122, 123, 124, 131, 132, 134, 135, 141, 143, 144, 151, 152, 153, 154, 161, 163, 171, 172, 174, 175, 181, 182, 183, 192, 193, 194, 195], format_func=lambda x: {
        0: 'Pelajar', 1: 'Perwakilan Legislatif/Eksekutif/Direktur', 2: 'Spesialis Aktivitas Intelektual/Ilmiah',
        3: 'Teknisi', 4: 'Staf Administrasi', 5: 'Pekerja Layanan/Keamanan/Penjual',
        6: 'Pertanian/Perikanan/Kehutanan', 7: 'Industri/Konstruksi/Pengrajin', 8: 'Operator Mesin',
        9: 'Pekerja Tidak Terampil', 10: 'Angkatan Bersenjata', 90: 'Lainnya', 99: 'Kosong',
        101: 'Perwira Angkatan Bersenjata', 102: 'Sersan Angkatan Bersenjata', 103: 'Personil Angkatan Bersenjata Lain',
        112: 'Direktur Administrasi/Komersial', 114: 'Direktur Hotel/Perdagangan',
        121: 'Ilmu Fisika/Matematika/Teknik', 122: 'Profesional Kesehatan', 123: 'Guru',
        124: 'Spesialis Keuangan/Administrasi', 131: 'Teknisi Ilmu Pengetahuan/Teknik (Menengah)',
        132: 'Teknisi Kesehatan (Menengah)', 134: 'Teknisi Hukum/Sosial/Budaya (Menengah)',
        135: 'Teknisi TIK', 141: 'Pekerja Kantor/Sekretaris/Operator Data',
        143: 'Operator Keuangan/Statistik/Data', 144: 'Staf Pendukung Administrasi Lain',
        151: 'Layanan Pribadi', 152: 'Penjual', 153: 'Perawatan Pribadi', 154: 'Personil Keamanan',
        161: 'Petani/Pekerja Produksi Hewan Terampil', 163: 'Petani Subsisten/Nelayan',
        171: 'Pekerja Konstruksi Terampil', 172: 'Pekerja Logam', 174: 'Pekerja Listrik',
        175: 'Industri Makanan/Kayu/Tekstil', 181: 'Operator Pabrik/Mesin', 182: 'Pekerja Perakitan',
        183: 'Pengemudi Kendaraan/Operator Peralatan Mobile', 192: 'Pekerja Tidak Terampil Pertanian/Perikanan/Kehutanan',
        193: 'Pekerja Tidak Terampil Industri/Transportasi', 194: 'Asisten Persiapan Makanan',
        195: 'Pedagang Kaki Lima'
    }.get(x, f'Kode {x}'))
    
    displaced = st.selectbox("Mengungsi (Pindah dari Luar)", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    tuition_fees_up_to_date = st.selectbox("Biaya Kuliah Lunas", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

# --- 5. Logika Prediksi ---
if st.button("Prediksi Risiko Dropout"):
    # Buat DataFrame input dengan SEMUA kolom asli dari df.head()
    # dan pastikan urutannya sama persis.
    # Kolom yang tidak digunakan di UI diisi dengan np.nan atau default 0 jika itu lebih masuk akal.
    
    # Daftar semua kolom original DataFrame Anda (dari df.info() di notebook)
    original_df_columns = [
        'Marital_status', 'Application_mode', 'Application_order', 'Course',
        'Daytime_evening_attendance', 'Previous_qualification', 'Previous_qualification_grade',
        'Nacionality', 'Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation',
        'Fathers_occupation', 'Admission_grade', 'Displaced', 'Educational_special_needs',
        'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
        'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_credited',
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
        'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
        'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited',
        'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
        'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
        'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate',
        'Inflation_rate', 'GDP', 'Status'
    ]

    # Buat satu baris data input
    input_data_dict = {col: np.nan for col in original_df_columns} # Inisialisasi dengan NaN

    # Isi nilai dari UI untuk fitur-fitur yang dipilih
    input_data_dict['Age_at_enrollment'] = age
    input_data_dict['Application_mode'] = application_mode
    # Application_order tidak ada di selected_features, tapi jika penting untuk preprocessor
    # berikan nilai default yang masuk akal, misal 1 atau rata-rata.
    input_data_dict['Application_order'] = 1 # Nilai default, sesuaikan jika perlu
    input_data_dict['Course'] = course
    input_data_dict['Daytime_evening_attendance'] = daytime_evening_attendance
    input_data_dict['Previous_qualification'] = previous_qualification
    input_data_dict['Debtor'] = debtor
    input_data_dict['Gender'] = gender
    input_data_dict['Scholarship_holder'] = scholarship_holder
    input_data_dict['Marital_status'] = marital_status
    input_data_dict['Mothers_qualification'] = mothers_qualification
    input_data_dict['Fathers_qualification'] = fathers_qualification
    input_data_dict['Mothers_occupation'] = mothers_occupation
    input_data_dict['Fathers_occupation'] = fathers_occupation
    input_data_dict['Displaced'] = displaced
    input_data_dict['Tuition_fees_up_to_date'] = tuition_fees_up_to_date
    
    # Buat DataFrame dari input_data_dict dengan urutan kolom yang benar
    input_df = pd.DataFrame([input_data_dict], columns=original_df_columns)

    # Filtering input_df agar hanya berisi kolom yang dipakai untuk training model
    # (selected_features). Ini penting karena pipeline_model dilatih hanya pada kolom ini.
    input_df_for_prediction = input_df[cat_cols + num_cols]


    try:
        prediction = pipeline_model.predict(input_df_for_prediction)[0]
        prediction_proba = pipeline_model.predict_proba(input_df_for_prediction)[0]

        st.subheader("Hasil Prediksi:")
        if prediction == 1:
            st.error(f"**Mahasiswa ini kemungkinan besar akan DROP OUT!** 😥")
            st.write(f"Probabilitas Dropout: **{prediction_proba[1]:.2%}**")
            st.write(f"Probabilitas Tidak Dropout: **{prediction_proba[0]:.2%}**")
        else:
            st.success(f"**Mahasiswa ini kemungkinan TIDAK akan DROP OUT.** 🎉")
            st.write(f"Probabilitas Tidak Dropout: **{prediction_proba[0]:.2%}**")
            st.write(f"Probabilitas Dropout: **{prediction_proba[1]:.2%}**")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.info("Mohon cek kembali input Anda dan pastikan semua kolom yang diperlukan telah diisi dengan benar.")


st.markdown("---")
st.markdown("Aplikasi ini dibuat sebagai bagian dari proyek Machine Learning. Prediksi ini adalah hasil dari model yang dilatih dan tidak menjamin kebenaran mutlak. Selalu gunakan pertimbangan profesional dalam mengambil keputusan.")
