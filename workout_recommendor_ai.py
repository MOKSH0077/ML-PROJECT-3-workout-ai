import streamlit as st
import pandas as pd
import joblib
import random

# ---------------- LOAD FILES ---------------- #
model = joblib.load("workout_model.pkl")
ct = joblib.load("workout_column_transformer.pkl")
scaler = joblib.load("workout_scaler.pkl")

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AI Workout Planner", layout="wide")

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
body { background-color: #0f172a; color: white; }
.block-container { padding-top: 2rem; }
h1, h2, h3 { color: #38bdf8; }
.stButton>button {
    background: linear-gradient(90deg, #0ea5e9, #22c55e);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.card {
    background-color: #111827;
    padding: 1.2rem;
    border-radius: 16px;
    box-shadow: 0 0 10px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LABEL MAP ---------------- #
split_map = {
    0: "Full Body",
    1: "Upper / Lower",
    2: "Push Pull Legs (PPL)",
    3: "Hybrid Split",
    4: "Bro Split",
    5: "Strength Focused",
    6: "Home Bodyweight"
}

# ---------------- SMART WORKOUT ENGINE ---------------- #
BEGINNER = {
    "Push": ["Pushups", "Machine Chest Press", "DB Shoulder Press"],
    "Pull": ["Lat Pulldown", "Seated Row", "Band Curls"],
    "Legs": ["Bodyweight Squat", "Leg Press", "Glute Bridge"]
}

INTERMEDIATE = {
    "Push": ["Bench Press", "Incline DB Press", "Dips"],
    "Pull": ["Pullups", "Barbell Row", "Hammer Curls"],
    "Legs": ["Back Squat", "Lunges", "RDL"]
}

ADVANCED = {
    "Push": ["Barbell Bench", "Overhead Press", "Weighted Dips"],
    "Pull": ["Weighted Pullups", "Pendlay Rows", "Barbell Curls"],
    "Legs": ["Front Squat", "Romanian Deadlift", "Bulgarian Split Squat"]
}

HOME_ONLY = ["Pushups", "Bodyweight Squat", "Plank", "Mountain Climbers", "Burpees"]

def generate_workout(pred, experience, equipment):
    plan = {}

    if equipment == 0:
        plan["Home Workout"] = random.sample(HOME_ONLY, 4)
        return plan

    pool = BEGINNER if experience == 0 else INTERMEDIATE if experience == 1 else ADVANCED

    if pred == 2:
        plan["Push Day"] = random.sample(pool["Push"], 3)
        plan["Pull Day"] = random.sample(pool["Pull"], 3)
        plan["Leg Day"] = random.sample(pool["Legs"], 3)

    elif pred == 0:
        combined = pool["Push"] + pool["Pull"] + pool["Legs"]
        plan["Full Body Day"] = random.sample(combined, 5)

    elif pred == 1:
        plan["Upper Day"] = random.sample(pool["Push"] + pool["Pull"], 4)
        plan["Lower Day"] = random.sample(pool["Legs"], 3)

    else:
        combined = pool["Push"] + pool["Pull"] + pool["Legs"]
        plan["Workout"] = random.sample(combined, 5)

    return plan

# ---------------- UI HEADER ---------------- #
st.title("🏋️ AI Workout Recommendation System")
st.write("Personalized gym-style workout planner powered by Machine Learning")

# ---------------- INPUT SECTIONS ---------------- #
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👤 Body Profile")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 16, 60, 22)

    with c2:
        feet = st.slider("Height (feet)", 4, 7, 5)
        inches = st.slider("Extra inches", 0, 11, 7)

    with c3:
        weight = st.slider("Weight (kg)", 40, 130, 70)

    st.markdown("</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 Lifestyle & Experience")

    c1, c2, c3 = st.columns(3)

    with c1:
        activity = {"Sedentary":0, "Lightly Active":1, "Active":2, "Very Active":3}[st.selectbox("Activity Level", ["Sedentary","Lightly Active","Active","Very Active"])]
        stress = {"Low":1, "Moderate":2, "High":3, "Very High":4, "Extreme":5}[st.selectbox("Stress Level", ["Low","Moderate","High","Very High","Extreme"])]

    with c2:
        sleep = st.slider("Sleep (hours)", 4, 10, 7)
        experience = {"Beginner":0, "Intermediate":1, "Advanced":2}[st.selectbox("Training Experience", ["Beginner","Intermediate","Advanced"])]

    with c3:
        days = st.slider("Workout Days / Week", 2, 6, 4)
        duration = st.select_slider("Session Duration (min)", [30, 40, 45, 60, 75, 90])

    st.markdown("</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Goals & Setup")

    c1, c2 = st.columns(2)

    with c1:
        equipment = {"Home Only":0, "Dumbbells":1, "Full Gym":2}[st.selectbox("Equipment Access", ["Home Only","Dumbbells","Full Gym"])]
        goal = {"Fat Loss":0, "Muscle Gain":1, "General Fitness":2}[st.selectbox("Primary Goal", ["Fat Loss","Muscle Gain","General Fitness"])]

    with c2:
        target_weight = st.slider("Target Weight (kg)", 40, 130, 68)
        target_duration = st.slider("Target Duration (months)", 1, 12, 4)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CENTERED PREDICTION ---------------- #
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🚀 Generate Your Plan")

    if st.button("Generate Workout Plan"):
        height_cm = (feet * 30.48) + (inches * 2.54)
        bmi = round(weight / ((height_cm / 100) ** 2), 2)

        user_input = pd.DataFrame([[
            age, height_cm, weight, bmi,
            activity, stress, sleep,
            experience, days, duration,
            equipment, goal,
            target_weight, target_duration
        ]], columns=[
            "age", "height", "weight", "bmi",
            "activity_level", "stress", "sleep",
            "experience_level", "days_per_week",
            "session_duration_min", "equipment_access",
            "primary_goal", "target_weight_change_kg",
            "target_duration_months"
        ])

        num_cols = [
            "age", "height", "weight", "bmi", "sleep",
            "days_per_week", "session_duration_min",
            "target_weight_change_kg", "target_duration_months"
        ]

        user_input[num_cols] = scaler.transform(user_input[num_cols])
        final_input = ct.transform(user_input)

        pred = model.predict(final_input)[0]

        st.success(f"🔥 Recommended Plan: **{split_map[pred]}**")

        plan = generate_workout(pred, experience, equipment)

        st.subheader("📅 Your Weekly Workout")
        for day, exs in plan.items():
            with st.expander(day):
                for e in exs:
                    st.write(f"• {e}")

    st.markdown("</div>", unsafe_allow_html=True)




