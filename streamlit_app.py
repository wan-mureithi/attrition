import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/wan-mureithi/datasets/refs/heads/main/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    df["EducationLabel"] = df["Education"].map(
        {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
    )
    return df


df = load_data()

tab1, tab2 = st.tabs(["🏠 Key Attrition Drivers", "🔮 Predict Attrition"])

with tab1:
    st.title("Key Attrition Drivers")
    st.markdown("""
    This dashboard explores key factors influencing employee attrition.
    """)

    # Plot 1: Distance from Home by Job Role and Attrition
    st.subheader("📍 Distance from Home by Job Role and Attrition")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="JobRole", y="DistanceFromHome", hue="Attrition", ax=ax1)
    ax1.set_title("Distance from Home by Job Role and Attrition")
    ax1.tick_params(axis="x", rotation=45)
    st.pyplot(fig1)

    # Plot 2: Monthly Income by Education and Attrition
    st.subheader("💰 Monthly Income by Education Level and Attrition")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="EducationLabel", y="MonthlyIncome", hue="Attrition", ax=ax2)
    ax2.set_title("Average Monthly Income by Education and Attrition")
    st.pyplot(fig2)

with tab2:
    st.title("Employee Attrition Predictor")
    st.subheader("Make smarter HR decisions")
    st.write(
        "This app predicts whether an employee is at risk of leaving based on their profile."
    )

    # @st.cache_resource
    # def load_model():
    #     with open("model/ensemble_model.pkl", "rb") as f:
    #         model, features = cloudpickle.load(f)
    #     return model, features

    @st.cache_resource
    def load_model():
        model, features = joblib.load("model/ensemble_model.pkl")
        return model, features

    model, feature_names = load_model()

    st.sidebar.title("Employee Details")

    job_level = st.sidebar.slider("Job Level", 1, 5, 2)
    overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
    env_satisfaction = st.sidebar.slider("Environment Satisfaction", 1, 4, 3)
    job_involvement = st.sidebar.slider("Job Involvement", 1, 4, 3)
    years_in_role = st.sidebar.slider("Years in Current Role", 0, 15, 3)
    age = st.sidebar.slider("Age", 18, 60, 35)
    job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
    monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 6000)
    total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)

    overtime_binary = 1 if overtime == "Yes" else 0
    input_data = pd.DataFrame(
        [
            [
                job_level,
                overtime_binary,
                env_satisfaction,
                job_involvement,
                years_in_role,
                age,
                job_satisfaction,
                monthly_income,
                total_working_years,
            ]
        ],
        columns=feature_names,
    )

    if st.button("Predict Attrition Risk"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.markdown("### 🧾 Prediction Result:")
        if prediction == 1:
            st.error(
                f"🚨 This employee is at **risk of attrition** (probability: {probability:.2f})"
            )
        else:
            st.success(
                f"✅ This employee is **not likely to leave** (probability: {probability:.2f})"
            )
