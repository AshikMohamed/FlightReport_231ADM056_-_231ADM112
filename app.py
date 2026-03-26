import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

# =========================
# LOAD MODEL
# =========================
model = joblib.load("rf_model.pkl")
features = joblib.load("features.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Flight AI", page_icon="✈️", layout="wide")

st.title("✈️ Flight Delay Prediction System")

# =========================
# FLIGHT DATA (NEW)
# =========================
flight_list = [
    "AI101", "AI202", "AI303",
    "6E201", "6E305",
    "UK811", "UK955",
    "SG401", "SG502"
]

# Flight → Route mapping
flight_routes = {
    "AI101": ("DFW", "MIA"),
    "AI202": ("MIA", "DFW"),
    "AI303": ("DFW", "JFK"),
    "6E201": ("JFK", "LAX"),
    "6E305": ("LAX", "JFK"),
    "UK811": ("ORD", "ATL"),
    "UK955": ("ATL", "ORD"),
    "SG401": ("MIA", "JFK"),
    "SG502": ("JFK", "MIA"),
}

# =========================
# INPUT SECTION
# =========================
st.subheader("🛫 Flight Information")

col1, col2, col3 = st.columns(3)

with col1:
    flight_no = st.selectbox("Flight Number", flight_list)

with col2:
    hour = st.selectbox("Hour", list(range(0, 24)))
    minute = st.selectbox("Minute", list(range(0, 60)))
    dep_time_input = datetime.strptime(f"{hour}:{minute}", "%H:%M").time()

with col3:
    day_name = st.selectbox(
        "Day",
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    )

# =========================
# AUTO ROUTE FROM FLIGHT
# =========================
origin, destination = flight_routes[flight_no]

st.subheader("🌍 Route")
st.info(f"✈️ {origin} → {destination}")

# =========================
# TIME PROCESSING
# =========================
dep_minutes = hour * 60 + minute
dep_hour = hour

day_map = {
    "Monday":0,"Tuesday":1,"Wednesday":2,
    "Thursday":3,"Friday":4,"Saturday":5,"Sunday":6
}
day = day_map[day_name]

# =========================
# ARRIVAL TIME
# =========================
st.subheader("🛬 Flight Timing")

duration_map = {
    ("DFW","MIA"): 2.5,
    ("MIA","DFW"): 2.5,
    ("JFK","LAX"): 6,
    ("LAX","JFK"): 6,
    ("DFW","JFK"): 3,
    ("ORD","ATL"): 2,
    ("ATL","ORD"): 2
}

duration = duration_map.get((origin, destination), 3)

arrival_time = (
    datetime.combine(datetime.today(), dep_time_input)
    + timedelta(hours=duration)
).time()

st.info(f"🛫 Departure: {dep_time_input} → 🛬 Arrival: {arrival_time} ({duration} hrs)")

# =========================
# FEATURE ENGINEERING
# =========================
morning = 1 if 5 <= dep_hour < 12 else 0
afternoon = 1 if 12 <= dep_hour < 17 else 0
evening = 1 if 17 <= dep_hour < 21 else 0

input_data = pd.DataFrame([{
    "Dep_Time_Minutes": dep_minutes,
    "Dep_Hour": dep_hour,
    "DayOfWeek": day,

    "Time_Bucket_Morning": morning,
    "Time_Bucket_Afternoon": afternoon,
    "Time_Bucket_Evening": evening,

    "Origin Airport_DFW": 1 if origin == "DFW" else 0,
    "Origin Airport_MIA": 1 if origin == "MIA" else 0,

    "Destination Airport_DFW": 1 if destination == "DFW" else 0,
    "Destination Airport_MIA": 1 if destination == "MIA" else 0,
}])

input_data = input_data.reindex(columns=features, fill_value=0)

# =========================
# PREDICTION
# =========================
st.subheader("📊 Prediction")

if st.button("🚀 Predict"):

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.metric("Delay Probability", f"{prob:.2%}")
    st.progress(int(prob * 100))

    if pred == 1:
        st.error("⚠️ Flight likely DELAYED")
    else:
        st.success("✅ Flight likely ON TIME")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("✈️ Built with Machine Learning + Streamlit")