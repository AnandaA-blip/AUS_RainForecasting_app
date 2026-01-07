import streamlit as st
import pandas as pd
import joblib
import numpy as np
import math

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌦️",
    layout="centered"
)

# --- 2. LOAD ASET ---
# Menggunakan st.cache versi lama sesuai sistem Anda
@st.cache(allow_output_mutation=True)
def load_assets():
    try:
        model = joblib.load('model/xgb_weather_model.joblib')
        scaler = joblib.load('model/scaler.joblib')
        features = joblib.load('model/feature_columns.joblib')
        return model, scaler, features
    except Exception as e:
        st.error(f"Gagal memuat aset model: {e}")
        return None, None, None

model, scaler, feature_columns = load_assets()

# --- 3. JUDUL ---
st.title("🌦️ Australia Rain Forecasting")
st.write(" byGroup 3 Data Logos.")
st.write("Masukkan data cuaca hari ini untuk memprediksi besok.")

# --- 4. FORM INPUT USER ---
with st.form("prediction_form"):
    st.header("Input Data")

    col1, col2 = st.columns(2)
    
    # Opsi Dropdown (Sesuai dataset WeatherAUS)
    lokasi_list = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru']
    arah_angin_list = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    with col1:
        date_input = st.date_input("Tanggal")
        location = st.selectbox("Lokasi", options=lokasi_list)
        min_temp = st.number_input("Min Temp (°C)", value=15.0)
        max_temp = st.number_input("Max Temp (°C)", value=25.0)
        rainfall = st.number_input("Curah Hujan Hari Ini (mm)", value=0.0, min_value=0.0)
        evaporation = st.number_input("Evaporation (mm)", value=5.0)
        sunshine = st.number_input("Sunshine (hours)", value=7.0)
        
        wind_gust_dir = st.selectbox("Arah Angin Gust", options=arah_angin_list)
        wind_gust_speed = st.number_input("Kecepatan Angin Gust (km/h)", value=35.0)

    with col2:
        wind_dir_9am = st.selectbox("Arah Angin 9am", options=arah_angin_list)
        wind_dir_3pm = st.selectbox("Arah Angin 3pm", options=arah_angin_list)
        wind_speed_9am = st.number_input("Kecepatan Angin 9am", value=10.0)
        wind_speed_3pm = st.number_input("Kecepatan Angin 3pm", value=15.0)
        
        humidity_9am = st.slider("Kelembaban 9am (%)", 0, 100, 60)
        humidity_3pm = st.slider("Kelembaban 3pm (%)", 0, 100, 50)
        pressure_9am = st.number_input("Tekanan 9am (hPa)", value=1015.0)
        pressure_3pm = st.number_input("Tekanan 3pm (hPa)", value=1012.0)
        
        cloud_9am = st.slider("Awan 9am (0-8)", 0, 8, 4)
        cloud_3pm = st.slider("Awan 3pm (0-8)", 0, 8, 4)
        temp_9am = st.number_input("Suhu 9am", value=18.0)
        temp_3pm = st.number_input("Suhu 3pm", value=22.0)

    submitted = st.form_submit_button("🔍 Prediksi Sekarang")

# --- 5. LOGIKA PREDIKSI ---
if submitted:
    if model is None:
        st.error("Model belum dimuat.")
    else:
        # A. Buat DataFrame Awal dari Input
        input_data = {
            'Location': location,
            'MinTemp': min_temp,
            'MaxTemp': max_temp,
            'Rainfall': rainfall,
            'Evaporation': evaporation,
            'Sunshine': sunshine,
            'WindGustDir': wind_gust_dir,
            'WindGustSpeed': wind_gust_speed,
            'WindDir9am': wind_dir_9am,
            'WindDir3pm': wind_dir_3pm,
            'WindSpeed9am': wind_speed_9am,
            'WindSpeed3pm': wind_speed_3pm,
            'Humidity9am': humidity_9am,
            'Humidity3pm': humidity_3pm,
            'Pressure9am': pressure_9am,
            'Pressure3pm': pressure_3pm,
            'Cloud9am': cloud_9am,
            'Cloud3pm': cloud_3pm,
            'Temp9am': temp_9am,
            'Temp3pm': temp_3pm
        }
        
        input_df = pd.DataFrame([input_data])

        # B. FEATURE ENGINEERING (Rekayasa Fitur)
        # Bagian ini HARUS SAMA dengan preprocessing saat training Anda
        
        # 1. Date Feature (Month Sin/Cos)
        month = date_input.month
        input_df['Month_Sin'] = np.sin(2 * np.pi * month / 12)
        input_df['Month_Cos'] = np.cos(2 * np.pi * month / 12)
        
        # 2. TempRange
        input_df['TempRange'] = input_df['MaxTemp'] - input_df['MinTemp']
        
        # 3. Rainfall_Log (Log Transform)
        input_df['Rainfall_Log'] = np.log1p(input_df['Rainfall'])
        
        # 4. RainToday (Yes/No menjadi 1/0)
        input_df['RainToday'] = 1 if input_df['Rainfall'].iloc[0] > 1.0 else 0
        
        # C. ONE-HOT ENCODING & ALIGNMENT
        # Ubah kategori jadi dummy variables
        input_df_encoded = pd.get_dummies(input_df)
        
        # --- LANGKAH PENTING: REINDEXING ---
        # Memaksa input user memiliki kolom yang SAMA PERSIS dengan data training.
        # Kolom yang tidak dipilih user (misal: Location_Sydney saat user pilih Albany) akan diisi 0.
        input_df_ready = input_df_encoded.reindex(columns=feature_columns, fill_value=0)

        # D. SCALING & PREDIKSI
        try:
            input_scaled = scaler.transform(input_df_ready)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.write("---")
            if prediction == 1:
                st.error(f"🌧️ **Hujan** (Probabilitas: {probability:.1%})")
                st.write("Besok kemungkinan hujan.")
            else:
                st.success(f"☀️ **Tidak Hujan** (Probabilitas Hujan: {probability:.1%})")
                st.write("Besok kemungkinan cerah.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")