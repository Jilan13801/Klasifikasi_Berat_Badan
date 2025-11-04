import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------
# Konfigurasi Halaman
# ----------------------------------
st.set_page_config(page_title="Klasifikasi Berat Badan - Decision Tree", page_icon="🌳", layout="wide")

st.title("🌳 Aplikasi Klasifikasi Berat Badan Menggunakan Decision Tree")
st.markdown("""
Aplikasi ini memprediksi kategori berat badan seseorang **Underweight (Kurus)**, **Normal Weight (Ideal)**, **Overweight (Gemuk)**, **Obesity (Obesitas)**.
Berdasarkan **tinggi badan**, **berat badan**, dan **jenis kelamin** menggunakan algoritma **Decision Tree Classifier**.
""")

# ----------------------------------
# Load Dataset Langsung (tanpa upload)
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity_Data.csv")
    # Mapping label angka ke teks
    label_map = {1: "Underweight (Kurus)", 2: "Normal Weight (Ideal)", 3: "Overweight (Gemuk)", 4: "Obesity (Obesitas)"}
    df["Kategori"] = df["Label"].map(label_map)
    return df

df = load_data()

# ----------------------------------
# Tampilkan Dataset
# ----------------------------------
st.subheader("📊 Dataset Berat Badan")
st.dataframe(df, use_container_width=True)

# ----------------------------------
# Visualisasi Data Awal
# ----------------------------------
st.subheader("📈 Visualisasi Data")

fig = px.scatter(
    df, x="Tinggi_Badan", y="Berat_Badan", color="Kategori",
    symbol="Jenis_Kelamin",
    title="Sebaran Data Tinggi vs Berat Badan",
    color_discrete_map={
        "Underweight (Kurus)": "#1f77b4",
        "Normal Weight (Ideal)": "#2ca02c",
        "Overweight (Gemuk)": "#ff7f0e",
        "Obesity (Obesitas)": "#d62728"
    }
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# Preprocessing & Split Data
# ----------------------------------
X = df[["Tinggi_Badan", "Berat_Badan"]]
y = df["Kategori"]

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ----------------------------------
# Training Model
# ----------------------------------
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# ----------------------------------
# Evaluasi Model
# ----------------------------------
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

with st.expander("📋 Hasil Evaluasi Model"):
    st.write(f"**Akurasi Model:** {akurasi * 100:.2f}%")
    st.text("Laporan Klasifikasi:")
    st.text(classification_report(y_test, y_pred))

# ----------------------------------
# Visualisasi Pohon Keputusan
# ----------------------------------
st.subheader("🌿 Visualisasi Pohon Keputusan")
fig_tree, ax = plt.subplots(figsize=(10, 6))
plot_tree(model, feature_names=["Tinggi_Badan", "Berat_Badan"], class_names=model.classes_, filled=True)
st.pyplot(fig_tree)

# ----------------------------------
# Prediksi Data Baru
# ----------------------------------
st.subheader("🔍 Prediksi Kategori Berat Badan Baru")

col1, col2, col3 = st.columns(3)
with col1:
    jenis_kelamin = st.selectbox("Pilih Jenis Kelamin:", ["Laki - Laki", "Perempuan"])
with col2:
    tinggi = st.number_input("Masukkan Tinggi Badan (cm):", min_value=100, max_value=220, value=170)
with col3:
    berat = st.number_input("Masukkan Berat Badan (kg):", min_value=30, max_value=150, value=60)


if st.button("Prediksi", key="prediksi_button"):
    try:
        # --- Siapkan data baru ---
        data_baru = pd.DataFrame({
            "Tinggi_Badan": [tinggi],
            "Berat_Badan": [berat]
        })
        data_scaled = scaler.transform(data_baru)
        hasil_prediksi = model.predict(data_scaled)[0]  # hasil berupa teks kategori

        # --- Saran berdasarkan hasil prediksi ---
        saran_dict = {
            "Underweight (Kurus)": "💡 Disarankan untuk meningkatkan asupan nutrisi dan kalori. Konsumsi makanan bergizi seimbang dan cukup istirahat.",
            "Normal Weight (Ideal)": "✅ Berat badan Anda ideal! Pertahankan pola makan dan gaya hidup sehat agar tetap bugar.",
            "Overweight (Gemuk)": "⚠️ Anda termasuk kategori gemuk. Disarankan untuk mulai memperhatikan pola makan, kurangi makanan berlemak, dan rutin berolahraga.",
            "Obesity (Obesitas)": "🚨 Anda termasuk kategori obesitas. Segera atur pola makan sehat, kurangi gula dan lemak berlebih, serta lakukan olahraga ringan secara rutin."
        }

        saran = saran_dict.get(hasil_prediksi, "Tidak ada saran tersedia untuk kategori ini.")

        # --- Tampilkan hasil ---
        st.success(f"✅ Berdasarkan data yang dimasukkan, kategori berat badan Anda adalah: **{hasil_prediksi}**")
        st.info(saran)

        # --- Visualisasi posisi pada grafik ---
        fig_pred = px.scatter(
            df, x="Tinggi_Badan", y="Berat_Badan", color="Kategori",
            symbol="Jenis_Kelamin",
            title="Posisi Data Baru pada Grafik",
            color_discrete_map={
                "Underweight (Kurus)": "#1f77b4",
                "Normal Weight (Ideal)": "#2ca02c",
                "Overweight (Gemuk)": "#ff7f0e",
                "Obesity (Obesitas)": "#d62728"
            }
        )
        fig_pred.add_scatter(
            x=[tinggi], y=[berat],
            mode="markers+text", name="Data Baru",
            marker=dict(size=15, color="black", symbol="star"),
            text=["Data Baru"], textposition="top center"
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")