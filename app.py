import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan preprocessing
model = joblib.load("model/gboost_model.joblib")
target_encoder = joblib.load("model/encoder_target.joblib")

# Daftar fitur numerik untuk PCA
numerical_pca_1 = [
    'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations'
]

numerical_pca_2 = [
    'Previous_qualification_grade', 'Admission_grade', 'Age_at_enrollment',
    'Unemployment_rate', 'Inflation_rate', 'GDP'
]

categorical_columns = [
    "Marital_status", "Application_mode", 'Application_order', "Course",
    "Daytime_evening_attendance", "Previous_qualification", "Nacionality",
    "Mothers_qualification", "Fathers_qualification", "Mothers_occupation",
    "Fathers_occupation", "Displaced", "Educational_special_needs", "Debtor",
    "Tuition_fees_up_to_date", "Gender", "Scholarship_holder", "International"
]

# Load scaler dan encoder
scalers = {col: joblib.load(f"model/scaler_{col}.joblib") for col in numerical_pca_1 + numerical_pca_2}
encoders = {col: joblib.load(f"model/encoder_{col}.joblib") for col in categorical_columns}
pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="wide")
st.title("🎓 Prediksi Status Mahasiswa")
st.markdown("Isi form di bawah ini untuk memprediksi status mahasiswa.")

# Fungsi bantu untuk dropdown
def get_options(mapping):
    return [f"{k} - {v}" for k, v in mapping.items()]

# Mapping label
qual_dict = {
    1: "Secondary Education", 2: "Bachelor's Degree", 3: "Degree", 4: "Master's",
    5: "Doctorate", 6: "Frequency of Higher Education", 9: "12th Year Not Completed",
    10: "11th Year Not Completed", 11: "7th Year (Old)", 12: "Other - 11th Year",
    14: "10th Year", 18: "General Commerce Course", 19: "Basic Education 3rd Cycle",
    22: "Technical-professional course", 26: "7th year", 27: "2nd cycle high school",
    29: "9th Year Not Completed", 30: "8th year", 34: "Unknown", 35: "Can't read or write",
    36: "Can read (no 4th year)", 37: "Basic Education 1st Cycle", 38: "Basic Education 2nd Cycle",
    39: "Tech specialization course", 40: "Degree (1st cycle)", 41: "Specialized higher studies",
    42: "Professional higher technical course", 43: "Master (2nd cycle)", 44: "Doctorate (3rd cycle)"
}

occupation_dict = {
    0: "Student", 1: "Executive Managers", 2: "Scientific Professionals", 3: "Technicians",
    4: "Admin Staff", 5: "Services/Sellers", 6: "Farmers", 7: "Construction", 8: "Operators",
    9: "Unskilled", 10: "Armed Forces", 90: "Other", 99: "Blank", 122: "Health Professionals",
    123: "Teachers", 125: "ICT Specialists", 131: "Sci/Eng Technicians", 132: "Health Techs",
    134: "Legal/Cultural", 141: "Office workers", 143: "Finance Operators", 144: "Support Staff",
    151: "Service Workers", 152: "Sellers", 153: "Care Workers", 171: "Construction Workers",
    173: "Artisans", 175: "Crafts", 191: "Cleaning", 192: "Unskilled Agri", 193: "Unskilled Industry",
    194: "Food Prep"
}

# Form input
with st.form("student_form"):
    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        inputs["Marital_status"] = st.selectbox("Status Pernikahan", [
            "1 - Single", "2 - Married", "3 - Widower", "4 - Divorced", "5 - Facto Union", "6 - Legally Separated"
        ])
        inputs["Daytime_evening_attendance"] = st.selectbox("Jadwal Kehadiran", ["0 - Siang", "1 - Sore"])
        inputs["Application_mode"] = st.selectbox("Mode Aplikasi", [
            "1 - 1st phase - general contingent", "2 - Ordinance No. 612/93",
            "5 - 1st phase - special contingent (Azores Island)", "7 - Holders of other higher courses",
            "10 - Ordinance No. 854-B/99", "15 - International student (bachelor)",
            "16 - 1st phase - special contingent (Madeira Island)", "17 - 2nd phase - general contingent",
            "18 - 3rd phase - general contingent", "26 - Ordinance No. 533-A/99, item b2) (Different Plan)",
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
        inputs["Nacionality"] = st.selectbox("Kebangsaan", [
            "1 - Portuguese", "2 - German", "6 - Spanish", "11 - Italian", "13 - Dutch",
            "14 - English", "17 - Lithuanian", "21 - Angolan", "22 - Cape Verdean",
            "24 - Guinean", "25 - Mozambican", "26 - Santomean", "32 - Turkish",
            "41 - Brazilian", "62 - Romanian", "100 - Moldova (Republic of)",
            "101 - Mexican", "103 - Ukrainian", "105 - Russian", "108 - Cuban", "109 - Colombian"
        ])
        inputs["Mothers_qualification"] = st.selectbox("Kualifikasi Ibu", get_options(qual_dict))
        inputs["Fathers_qualification"] = st.selectbox("Kualifikasi Ayah", get_options(qual_dict))
        inputs["Mothers_occupation"] = st.selectbox("Pekerjaan Ibu", get_options(occupation_dict))
        inputs["Fathers_occupation"] = st.selectbox("Pekerjaan Ayah", get_options(occupation_dict))
        inputs["Previous_qualification"] = st.selectbox("Kualifikasi Sebelumnya", get_options(qual_dict))
        inputs["Educational_special_needs"] = st.selectbox("Kebutuhan Khusus", ["0 - Iya", "1 - Tidak"])
        inputs["Debtor"] = st.selectbox("Peminjam", ["0 - Tidak", "1 - Ya"])
        inputs["Tuition_fees_up_to_date"] = st.selectbox("Biaya Lunas", ["0 - Tidak", "1 - Ya"])
        inputs["Gender"] = st.selectbox("Jenis Kelamin", ["0 - Perempuan", "1 - Laki-laki"])
        inputs["Scholarship_holder"] = st.selectbox("Beasiswa", ["0 - Tidak", "1 - Ya"])
        inputs["International"] = st.selectbox("Internasional", ["0 - Tidak", "1 - Ya"])
        inputs["Displaced"] = st.selectbox("Displaced", ["0 - Tidak", "1 - Ya"])

    with col2:
        for col in numerical_pca_1 + numerical_pca_2:
            inputs[col] = st.number_input(col.replace("_", " "), step=0.1)

    submitted = st.form_submit_button("Prediksi Status Mahasiswa")

# Prediksi ketika tombol ditekan
if submitted:
    input_df = pd.DataFrame([inputs])

    # Pisahkan kode dari dropdown "x - y"
    for col in input_df.columns:
        if input_df[col].dtype == object and " - " in str(input_df[col][0]):
            input_df[col] = input_df[col].str.split(" - ").str[0].astype(int)

    # Encoding kategori
    for col in categorical_columns:
        input_df[col] = encoders[col].transform(input_df[col])

    # Scaling numerik
    for col in numerical_pca_1 + numerical_pca_2:
        input_df[col] = scalers[col].transform(input_df[[col]])

    # PCA transformasi
    pc1 = pca_1.transform(input_df[numerical_pca_1])
    pc2 = pca_2.transform(input_df[numerical_pca_2])

    pc_df = pd.DataFrame(pc1, columns=[f"pc1_{i+1}" for i in range(pc1.shape[1])])
    pc_df[[f"pc2_{i+1}" for i in range(pc2.shape[1])]] = pc2

    final_df = input_df.drop(columns=numerical_pca_1 + numerical_pca_2).reset_index(drop=True)
    final_df = pd.concat([final_df, pc_df], axis=1)

    # Prediksi dan tampilkan hasil
    pred = model.predict(final_df)
    pred_label = target_encoder.inverse_transform(pred)[0]
    st.success(f"🎯 Prediksi Status Mahasiswa: **{pred_label}**")
