import streamlit as st
import pandas as pd
import joblib

# Load model dan fitur
@st.cache_resource
def load_model():
    model = joblib.load("dropout_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names

model, feature_names = load_model()

st.title("🎓 Prediksi Dropout Mahasiswa")

st.markdown("Masukkan data mahasiswa untuk memprediksi kemungkinan dropout:")

# Form input
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age at Enrollment", min_value=15, max_value=60, value=18)
semester1 = st.number_input("1st Semester Grade", min_value=0.0, max_value=20.0, value=12.0)
semester2 = st.number_input("2nd Semester Grade", min_value=0.0, max_value=20.0, value=13.0)
scholarship = st.selectbox("Scholarship Holder", ["Yes", "No"])
debtor = st.selectbox("Debtor", ["Yes", "No"])
tuition_paid = st.selectbox("Tuition Fees Up To Date", ["Yes", "No"])
course = st.selectbox("Course", ["Course1", "Course2", "Course3"])  # Sesuaikan ini
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])  # Sesuaikan juga

if st.button("🔍 Prediksi Dropout"):
    try:
        # Gunakan nama kolom sesuai pelatihan (perhatikan huruf besar)
        input_df = pd.DataFrame([{
            'Curricular_units_2nd_sem_grade': semester2,
            'Curricular_units_1st_sem_grade': semester1,
            'Age_at_enrollment': age,
            'Tuition_fees_up_to_date': 1 if tuition_paid == "Yes" else 0,
            'Debtor': 1 if debtor == "Yes" else 0,
            'Scholarship_holder': 1 if scholarship == "Yes" else 0,
            'Gender': gender,
            'Course': course,
            'Marital_status': marital_status
        }])

        # Rename kolom agar cocok persis dengan saat training (title-case)
        input_df.columns = [col[0].upper() + col[1:] if col.lower() in [f.lower() for f in feature_names] else col for col in input_df.columns]

        # One-hot encoding
        input_df = pd.get_dummies(input_df)

        # Tambah kolom yang hilang
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_names]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Mahasiswa ini diprediksi **Dropout** dengan probabilitas {probability:.2%}")
        else:
            st.success(f"✅ Mahasiswa ini **TIDAK Dropout** (Probabilitas dropout: {probability:.2%})")

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat memproses prediksi.")
        st.text(f"Detail: {e}")

