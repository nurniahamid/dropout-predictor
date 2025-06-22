import streamlit as st
import pandas as pd
import joblib

# Load model terbaik hasil GridSearchCV
model = joblib.load('model/best_model.pkl')

st.set_page_config(page_title="Prediksi Dropout Siswa", layout="centered")
st.title("🎓 Prediksi Dropout Siswa Jaya Jaya Institut")

st.write("""
Upload file CSV berisi data siswa yang ingin Anda prediksi.  
File harus memiliki kolom yang sama seperti data training (fitur `df_selected`, kecuali `Dropout_flag`).
""")

# Upload file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📋 Data Siswa yang Diupload")
    st.dataframe(data.head())

    # Prediksi
    if st.button("🔍 Jalankan Prediksi Dropout"):
        try:
            predictions = model.predict(data)
            data["Dropout_Prediction"] = predictions
            data["Keterangan"] = data["Dropout_Prediction"].map({0: "✅ Tidak Dropout", 1: "❌ Potensi Dropout"})

            st.subheader("📊 Hasil Prediksi")
            st.dataframe(data)

            # Download button
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil Prediksi",
                data=csv,
                file_name='hasil_prediksi_dropout.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
else:
    st.info("Silakan upload file CSV terlebih dahulu.")

