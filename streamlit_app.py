import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_assets()

# UI
st.title("🎓 Prediksi Dropout Mahasiswa")

gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
age = st.number_input("Usia", 15, 60, 18)
admission_grade = st.number_input("Nilai Masuk", 0.0, 200.0, 140.0)
semester1 = st.number_input("Nilai Semester 1", 0.0, 20.0, 12.0)
semester2 = st.number_input("Nilai Semester 2", 0.0, 20.0, 13.0)
scholarship = st.selectbox("Beasiswa", ["Ya", "Tidak"])
debtor = st.selectbox("Hutang", ["Ya", "Tidak"])
tuition_paid = st.selectbox("Biaya Kuliah Terbayar", ["Ya", "Tidak"])
course = st.selectbox("Program Studi", ["Science", "Arts", "Engineering", "Law", "Business"])
marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced", "Widowed"])

if st.button("🔍 Prediksi Dropout"):
    df = pd.DataFrame([{
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
    
    try:
        X = preprocessor.transform(df)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        if pred == 1:
            st.error(f"⚠️ Mahasiswa ini BERISIKO dropout (Prob: {prob:.2%})")
        else:
            st.success(f"✅ Mahasiswa ini diprediksi TIDAK dropout (Prob dropout: {prob:.2%})")
    except Exception as e:
        st.error("❌ Terjadi kesalahan.")
        st.text(str(e))
