import streamlit as st
import numpy as np
import joblib

# Load trained Extra Trees model
model = joblib.load("extra_trees_model.pkl")

st.title("☀️ Solar Power Generation Predictor (Extra Trees Model)")
st.markdown("Fill in the environmental parameters below to estimate power generation:")

# Input widgets
distance_to_solar_noon = st.slider("Distance to Solar Noon", 0.0, 1.0, 0.5)
temperature = st.slider("Temperature (°F)", 0, 120, 75)
wind_direction = st.slider("Wind Direction (°)", 0, 360, 180)
wind_speed = st.slider("Wind Speed (mph)", 0.0, 50.0, 10.0)
sky_cover = st.slider("Sky Cover (%)", 0, 100, 25)
visibility = st.slider("Visibility (miles)", 0.0, 10.0, 10.0)
humidity = st.slider("Humidity (%)", 0, 100, 60)
avg_wind_speed = st.slider("Avg Wind Speed (mph)", 0.0, 50.0, 5.0)
avg_pressure = st.slider("Avg Pressure", 20.0, 35.0, 29.9)

# Calculate additional feature
solar_proximity = 1 - distance_to_solar_noon

# Combine inputs into feature array
features = np.array([[distance_to_solar_noon, temperature, wind_direction,
                      wind_speed, sky_cover, visibility, humidity,
                      avg_wind_speed, avg_pressure, solar_proximity]])

# Predict on button click
if st.button("🔮 Predict Power"):
    prediction = model.predict(features)[0]
    st.success(f"Estimated Power Generated: **{int(prediction)} watts**")

# At the end of your script
st.markdown("---")
st.markdown("📘 *Project by Lokesh Gadhi | July 2025*")

