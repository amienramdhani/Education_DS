import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# LOAD MODEL & PIPELINE
# ===============================
model = joblib.load("model/gboost_model.joblib")
target_encoder = joblib.load("model/encoder_target.joblib")
pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")

# Fitur yang digunakan
numerical_pca_1 = [
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
]

numerical_pca_2 = [
    'Previous_qualification_grade', 'Admission_grade', 'Age_at_enrollment',
    'Unemployment_rate', 'Inflation_rate', 'GDP', 
]

categorical_columns = [
    "Marital_status", "Application_mode", "Course", "Previous_qualification",
    "Mothers_qualification", "Fathers_qualification", "Mothers_occupation",
    "Fathers_occupation", "Displaced", "Debtor",
    "Tuition_fees_up_to_date", "Gender", "Scholarship_holder",
]

# Load preprocessing tools
scalers = {col: joblib.load(f"model/scaler_{col}.joblib") for col in numerical_pca_1 + numerical_pca_2}
encoders = {col: joblib.load(f"model/encoder_{col}.joblib") for col in categorical_columns}

# ===============================
# UI STARTS HERE
# ===============================
st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="wide")
st.title("🎓 Prediksi Status Mahasiswa")
st.markdown("Isi form di bawah ini untuk memprediksi status mahasiswa berdasarkan data pendaftaran dan akademik.")

# Helper untuk parsing nilai dari dropdown
def extract_int(value):
    try:
        return int(str(value).split(" - ")[0])
    except:
        return value

# ===============================
# FORM INPUT
# ===============================
with st.form("student_form"):
    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        inputs["Marital_status"] = st.selectbox("Status Pernikahan", [
            "1 - Single", "2 - Married", "3 - Widower", "4 - Divorced", "5 - Facto Union", "6 - Legally Separated"
        ])
        inputs["Application_mode"] = st.selectbox("Mode Aplikasi", [
            "1 - 1st phase - general contingent",
            "2 - Ordinance No. 612/93", "5 - 1st phase - special contingent (Azores Island)",
            "7 - Holders of other higher courses", "10 - Ordinance No. 854-B/99",
            "15 - International student (bachelor)", "16 - 1st phase - special contingent (Madeira Island)",
            "17 - 2nd phase - general contingent", "18 - 3rd phase - general contingent",
            "26 - Ordinance No. 533-A/99, item b2) (Different Plan)",
            "27 - Ordinance No. 533-A/99, item b3 (Other Institution)", "39 - Over 23 years old",
            "42 - Transfer", "43 - Change of course", "44 - Technological specialization diploma holders",
            "51 - Change of institution/course", "53 - Short cycle diploma holders",
            "57 - Change of institution/course (International)"
        ])
        inputs["Course"] = st.selectbox("Program Studi", [
            "33 - Biofuel Production Technologies", "171 - Animation and Multimedia Design",
            "8014 - Social Service (evening attendance)", "9003 - Agronomy", "9070 - Communication Design",
            "9085 - Veterinary Nursing", "9119 - Informatics Engineering", "9130 - Equinculture",
            "9147 - Management", "9238 - Social Service", "9254 - Tourism", "9500 - Nursing",
            "9556 - Oral Hygiene", "9670 - Advertising and Marketing Management",
            "9773 - Journalism and Communication", "9853 - Basic Education",
            "9991 - Management (evening attendance)"
        ])
        inputs["Mothers_qualification"] = st.text_input("Kualifikasi Ibu", "1")
        inputs["Fathers_qualification"] = st.text_input("Kualifikasi Ayah", "1")
        inputs["Mothers_occupation"] = st.text_input("Pekerjaan Ibu", "0")
        inputs["Fathers_occupation"] = st.text_input("Pekerjaan Ayah", "0")
        inputs["Previous_qualification"] = st.text_input("Kualifikasi Sebelumnya", "1")
        inputs["Educational_special_needs"] = st.selectbox("Kebutuhan Khusus", ["0 - Ya", "1 - Tidak"])
        inputs["Debtor"] = st.selectbox("Peminjam", ["0 - Tidak", "1 - Ya"])
        inputs["Tuition_fees_up_to_date"] = st.selectbox("Biaya Lunas", ["0 - Tidak", "1 - Ya"])
        inputs["Gender"] = st.selectbox("Jenis Kelamin", ["0 - Perempuan", "1 - Laki-laki"])
        inputs["Scholarship_holder"] = st.selectbox("Beasiswa", ["0 - Tidak", "1 - Ya"])
        inputs["Displaced"] = st.selectbox("Displaced", ["0 - Tidak", "1 - Ya"])

    with col2:
        for col in numerical_pca_1 + numerical_pca_2:
            label = col.replace("_", " ").capitalize()
            inputs[col] = st.number_input(label, step=0.1, format="%.2f")

    submitted = st.form_submit_button("🔍 Prediksi Status Mahasiswa")

# ===============================
# PREDIKSI
# ===============================
if submitted:
    input_df = pd.DataFrame([inputs])

    # Convert dropdown text to int
    input_df = input_df.applymap(extract_int)

    # Scaling
    for col in numerical_pca_1 + numerical_pca_2:
        input_df[[col]] = scalers[col].transform(input_df[[col]])

    # Encoding
    for col in categorical_columns:
        input_df[[col]] = input_df[[col]].astype(str)
        input_df[[col]] = encoders[col].transform(input_df[[col]])

    # PCA transformasi
    pc1 = pca_1.transform(input_df[numerical_pca_1])
    pc2 = pca_2.transform(input_df[numerical_pca_2])
    pc1_df = pd.DataFrame(pc1, columns=[f"pc1_{i+1}" for i in range(pc1.shape[1])])
    pc2_df = pd.DataFrame(pc2, columns=[f"pc2_{i+1}" for i in range(pc2.shape[1])])

    # Gabungkan semua fitur
    final_df = input_df[categorical_columns].copy()
    final_df = pd.concat([final_df, pc1_df, pc2_df], axis=1)

    # Susun urutan kolom sesuai pelatihan model
    try:
        final_df = final_df[model.feature_names_in_]
    except:
        expected_cols = categorical_columns + [f"pc1_{i+1}" for i in range(5)] + [f"pc2_{i+1}" for i in range(2)]
        final_df = final_df[expected_cols]

    # Prediksi
    pred = model.predict(final_df)
    pred_label = target_encoder.inverse_transform(pred)[0]

    st.success(f"🎯 Status Mahasiswa Diprediksi: **{pred_label}**")
