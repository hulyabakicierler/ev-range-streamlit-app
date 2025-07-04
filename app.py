import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Modelleri ve dönüştürücüleri yükle
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Elektrikli Araç Segment ve Menzil Tahmin Aracı")
st.sidebar.header("Araç Özelliklerini Giriniz:")

# Kullanıcıdan veri al
battery_capacity = st.sidebar.slider("Batarya Kapasitesi (kWh)", 30, 120, 60)
efficiency = st.sidebar.slider("Verimlilik (Wh/km)", 130, 250, 160)
top_speed = st.sidebar.slider("Maksimum Hız (km/h)", 100, 250, 160)
torque = st.sidebar.slider("Tork (Nm)", 150, 700, 300)
acceleration = st.sidebar.slider("0-100 km/s Hızlanma (saniye)", 2.5, 15.0, 7.0)
fast_charge_power = st.sidebar.slider("Hızlı Şarj Gücü (kW)", 50, 300, 100)
towing_capacity = st.sidebar.slider("Çeki Kapasitesi (kg)", 0, 2000, 500)
cargo_volume = st.sidebar.slider("Bagaj Hacmi (L)", 100, 1000, 400)
seats = st.sidebar.slider("Koltuk Sayısı", 2, 8, 5)
length = st.sidebar.slider("Araç Uzunluğu (mm)", 3500, 5500, 4500)
width = st.sidebar.slider("Araç Genişliği (mm)", 1600, 2200, 1800)
height = st.sidebar.slider("Araç Yüksekliği (mm)", 1300, 2000, 1600)

# Hacim ve tahmini menzil
volume_mm3 = length * width * height
estimated_range = battery_capacity * 1000 / efficiency

# Dummy kategorik değerler – örnek olarak
drivetrain_FWD = 1
car_body_type_SUV = 1

# Giriş verisi
input_data = pd.DataFrame([{
    'top_speed_kmh': top_speed,
    'battery_capacity_kWh': battery_capacity,
    'torque_nm': torque,
    'efficiency_wh_per_km': efficiency,
    'acceleration_0_100_s': acceleration,
    'fast_charging_power_kw_dc': fast_charge_power,
    'towing_capacity_kg': towing_capacity,
    'cargo_volume_l': cargo_volume,
    'seats': seats,
    'volume_mm3': volume_mm3,
    'drivetrain_FWD': drivetrain_FWD,
    'car_body_type_SUV': car_body_type_SUV
}])

# Eksik sütunları doldur ve sırala (scaler için)
expected_columns = scaler.feature_names_in_
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[expected_columns]

# Tahmin için ölçekle
scaled_input = scaler.transform(input_data)

# Menzil tahmini (ölçeklenmiş veri ile)
predicted_range = xgb_model.predict(scaled_input)[0]

# Segment tahmini için girdi hazırlanıyor
# Bu sefer modelin beklediği sıralamaya göre veri veriyoruz
expected_rf_columns = rf_model.feature_names_in_
segment_input = input_data.copy()
segment_input["predicted_range_km"] = predicted_range

# Eksik kolonlar varsa 0 ile tamamla
for col in expected_rf_columns:
    if col not in segment_input.columns:
        segment_input[col] = 0
segment_input = segment_input[expected_rf_columns]

# Segment tahmini
predicted_segment_encoded = rf_model.predict(segment_input)[0]
predicted_segment = label_encoder.inverse_transform([predicted_segment_encoded])[0]

# Sonuçları göster
st.metric("Tahmin Edilen Menzil", f"{predicted_range:.0f} km")
st.success(f"Araç Segmenti: {predicted_segment}")
