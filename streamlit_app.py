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
semester2 = st.number_input("Nilai Semest_


