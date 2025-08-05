import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="NYC Taxi Trip Duration", layout="wide")

@st.cache_resource
def load_pipeline():
    return joblib.load("BaseLine_pipeline.pkl", mmap_mode="r")
    #return joblib.load("advancedModel_pipeline.pkl", mmap_mode="r") # But you need to optimize the Features as in the Model

pipeline = load_pipeline()

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return R * (2 * np.arcsin(np.sqrt(a)))

st.session_state.setdefault("pickup", None)
st.session_state.setdefault("dropoff", None)

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚕 NYC Taxi Trip Duration Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Click on the map to set pickup and dropoff points.</p>", unsafe_allow_html=True)
st.divider()

left, right = st.columns((2, 1))

with left:
    click_mode = st.radio("Click Mode", ["Pickup", "Dropoff"], horizontal=True)

    m = folium.Map(location=[40.75, -73.98], zoom_start=12)

    if st.session_state.pickup:
        folium.Marker(st.session_state.pickup, popup="Pickup", icon=folium.Icon(color="green")).add_to(m)
    if st.session_state.dropoff:
        folium.Marker(st.session_state.dropoff, popup="Dropoff", icon=folium.Icon(color="red")).add_to(m)
    if st.session_state.pickup and st.session_state.dropoff:
        folium.PolyLine([st.session_state.pickup, st.session_state.dropoff], color="orange", weight=3).add_to(m)

    map_data = st_folium(m, height=550, width=1200)

    if map_data and map_data["last_clicked"]:
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        st.session_state[click_mode.lower()] = [lat, lon]
        st.success(f"✅ {click_mode}: [{lat:.5f}, {lon:.5f}]")

with right:
    st.subheader("🚖 Ride Info")
    vendor_id = st.selectbox("Vendor ID", [1, 2])
    passenger_count = st.slider("Passenger Count", 1, 6, 1)
    date = st.date_input("Pickup Date", value=datetime.now().date())
    time = st.time_input("Pickup Time", value=datetime.now().time())
    pickup_datetime = datetime.combine(date, time)

    st.divider()
    st.subheader("🕒 Prediction")

    if st.button("🚦 Predict Duration"):
        if not st.session_state.pickup or not st.session_state.dropoff:
            st.warning("⚠️ Set both pickup and dropoff.")
        else:
            p_lat, p_lon = st.session_state.pickup
            d_lat, d_lon = st.session_state.dropoff
            hour = pickup_datetime.hour
            day_of_week = pickup_datetime.weekday()
            distance_km = haversine_distance(p_lat, p_lon, d_lat, d_lon)

            features = pd.DataFrame(
             [[passenger_count, vendor_id, hour, day_of_week, distance_km]],
             columns=['passenger_count', 'vendor_id', 'hour', 'day_of_week', 'distance_km']
             )

            try:
                log_prediction = pipeline.predict(features)[0]
                prediction = np.expm1(log_prediction)
                minutes, seconds = divmod(int(prediction), 60)

                st.markdown(
                    f"""
                    <div style='background-color:#000000;padding:18px;border-radius:18px;text-align:center;'>
                        <h3>🕒 {minutes} min {seconds} sec</h3>
                        <small>({int(prediction)} seconds total)</small>
                        <p>Distance: {distance_km:.2f} km</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

st.markdown("""
<div style='margin-top: 12px; font-size: 18px; color: #555; text-align: center;'>
    🚕 Created by <strong style="color:#4CAF50;">Anas Tamer</strong>
</div>
""", unsafe_allow_html=True)
