import streamlit as st
import pandas as pd
import joblib

# Load model dan nama fitur
@st.cache_resource
def load_model():
    model = joblib.load("dropout_model.pkl")
    feature_names = joblib.load("feature_names.pkl")  # pastikan file ini ada
    return model, feature_names

model, feature_names = load_model()

st.title("🎓 Prediksi Dropout Mahasiswa - Jaya Jaya Institut")
st.markdown("Masukkan data mahasiswa untuk memprediksi kemungkinan dropout:")

# Form input pengguna
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
age = st.number_input("Usia Saat Mendaftar", min_value=15, max_value=60, value=18)
semester1 = st.number_input("Nilai Semester 1", min_value=0.0, max_value=20.0, value=12.0)
semester2 = st.number_input("Nilai Semester 2", min_value=0.0, max_value=20.0, value=13.0)
scholarship = st.selectbox("Penerima Beasiswa", ["Ya", "Tidak"])
debtor = st.selectbox("Memiliki Hutang", ["Ya", "Tidak"])
tuition_paid = st.selectbox("Biaya Kuliah Terbayar", ["Ya", "Tidak"])
course = st.selectbox("Program Studi", ["Science", "Arts", "Engineering", "Law", "Business"])  # Sesuaikan dengan aslinya
marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced", "Widowed"])  # Sesuaikan juga

if st.button("🔍 Prediksi Dropout"):
    try:
        # Susun DataFrame sesuai input
        input_df = pd.DataFrame([{
            'Curricular_units_2nd_sem_grade': semester2,
            'Curricular_units_1st_sem_grade': semester1,
            'Age_at_enrollment': age,
            'Tuition_fees_up_to_date': 1 if tuition_paid == "Ya" else 0,
            'Debtor': 1 if debtor == "Ya" else 0,
            'Scholarship_holder': 1 if scholarship == "Ya" else 0,
            'Gender': gender,
            'Course': course,
            'Marital_status': marital_status
        }])

        # One-hot encoding untuk fitur kategorikal (harus sama dengan saat training)
        input_df = pd.get_dummies(input_df)

        # Tambahkan kolom yang hilang, isi dengan 0
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        # Urutkan kolom sesuai dengan urutan pelatihan
        input_df = input_df[feature_names]

        # Prediksi
        y_pred = model.predict(input_df)[0]
        y_prob = model.predict_proba(input_df)[0][1]

        if y_pred == 1:
            st.error(f"⚠️ Mahasiswa ini BERISIKO dropout (Probabilitas: {y_prob:.2%})")
        else:
            st.success(f"✅ Mahasiswa ini diprediksi TIDAK dropout (Probabilitas dropout: {y_prob:.2%})")

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat memproses prediksi.")
        st.text(f"Detail: {e}")
