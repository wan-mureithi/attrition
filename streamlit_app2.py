import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

# ---- TABS ----
tab1, tab2 = st.tabs(["🏠 Key Attrition Drivers", "🔮 Predict Attrition"])

# --- Tab 1: Key Attrition Drivers ---
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

# --- Tab 2: Predict Attrition ---
with tab2:
    st.title("Employee Attrition Predictor")
    st.subheader("Make smarter HR decisions")
    st.write(
        "This app predicts whether an employee is at risk of leaving based on their profile."
    )

    st.sidebar.title("Employee Details")

    age = st.sidebar.slider("Age", 18, 60, 35)
    monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 6000)
    overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
    job_level = st.sidebar.slider("Job Level", 1, 5, 2)
    total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
    years_at_company = st.sidebar.slider("Years at Company", 0, 20, 5)

    business_travel = st.sidebar.selectbox(
        "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )
    department = st.sidebar.selectbox(
        "Department", ["Sales", "Research & Development", "Human Resources"]
    )

    st.write(
        "🔍 **Model prediction placeholder:** (You can connect this to your API or model)"
    )
    st.info("Feature values collected — ready to run prediction.")
