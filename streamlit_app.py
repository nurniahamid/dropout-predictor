import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('dropout_model.pkl')

st.title("🎓 Prediksi Dropout Mahasiswa - Jaya Jaya Institut")
st.markdown("Masukkan data mahasiswa untuk memprediksi kemungkinan dropout:")

# Input form
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
age = st.number_input("Usia Saat Mendaftar", min_value=15, max_value=60, value=18)
admission_grade = st.number_input("Nilai Masuk (Admission Grade)", min_value=0.0, max_value=200.0, value=140.0)
semester1 = st.number_input("Nilai Semester 1", min_value=0.0, max_value=20.0, value=12.0)
semester2 = st.number_input("Nilai Semester 2", min_value=0.0, max_value=20.0, value=13.0)
scholarship = st.selectbox("Penerima Beasiswa", ["Ya", "Tidak"])
debtor = st.selectbox("Memiliki Hutang", ["Ya", "Tidak"])
tuition_paid = st.selectbox("Biaya Kuliah Terbayar", ["Ya", "Tidak"])

# Tambahan input baru sesuai fitur model
course = st.selectbox("Program Studi", [
    "Science", "Arts", "Engineering", "Business", "Law", "Medicine", "Education"
])  # Sesuaikan dengan kategori asli saat training

marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced", "Widowed"])  # Sesuaikan juga

# Tombol prediksi
if st.button("🔍 Prediksi Dropout"):
    try:
        # Buat DataFrame input
        input_dict = {
            'Gender': 1 if gender == "Laki-laki" else 0,
            'Age_at_enrollment': age,
            'Admission_grade': admission_grade,
            'Curricular_units_1st_sem_grade': semester1,
            'Curricular_units_2nd_sem_grade': semester2,
            'Scholarship_holder': 1 if scholarship == "Ya" else 0,
            'Debtor': 1 if debtor == "Ya" else 0,
            'Tuition_fees_up_to_date': 1 if tuition_paid == "Ya" else 0,
            'Course': course,
            'Marital_status': marital_status
        }

        input_df = pd.DataFrame([input_dict])

        # Pastikan urutan kolom sesuai model
        input_df = input_df[model.feature_names_in_]

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



