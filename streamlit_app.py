import streamlit as st
import pandas as pd
import pickle

# Load model pipeline yang sudah menyertakan preprocessing
with open("dropout_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎓 Prediksi Dropout Mahasiswa - Jaya Jaya Institut")
st.markdown("Masukkan data mahasiswa untuk memprediksi kemungkinan dropout:")

# Form input dari pengguna
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
age = st.number_input("Usia Saat Mendaftar", min_value=15, max_value=60, value=18)
admission_grade = st.number_input("Nilai Masuk (Admission Grade)", min_value=0.0, max_value=200.0, value=140.0)
semester1 = st.number_input("Nilai Semester 1", min_value=0.0, max_value=20.0, value=12.0)
semester2 = st.number_input("Nilai Semester 2", min_value=0.0, max_value=20.0, value=13.0)

scholarship = st.selectbox("Penerima Beasiswa", ["Ya", "Tidak"])
debtor = st.selectbox("Memiliki Hutang", ["Ya", "Tidak"])
tuition_paid = st.selectbox("Biaya Kuliah Terbayar", ["Ya", "Tidak"])

course = st.selectbox("Program Studi", ["Science", "Arts", "Engineering", "Law", "Business"])  # sesuaikan dengan dataset aslinya
marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced", "Widowed"])  # sesuaikan juga

if st.button("🔍 Prediksi Dropout"):
    try:
        # Susun DataFrame sesuai input
        input_df = pd.DataFrame([{
            'Gender': gender,
            'Age_at_enrollment': age,
            'Admission_grade': admission_grade,
            'Curricular_units_1st_sem_grade': semester1,
            'Curricular_units_2nd_sem_grade': semester2,
            'Scholarship_holder': "Yes" if scholarship == "Ya" else "No",
            'Debtor': "Yes" if debtor == "Ya" else "No",
            'Tuition_fees_up_to_date': "Yes" if tuition_paid == "Ya" else "No",
            'Course': course,
            'Marital_status': marital_status
        }])

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


    except Exception as e:
        st.error("❌ Terjadi kesalahan saat memproses prediksi.")
        st.text(f"Detail: {e}")



