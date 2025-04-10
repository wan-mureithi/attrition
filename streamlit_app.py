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


# Load model and features
@st.cache_resource
def load_model():
    model, features = joblib.load("voting_model.pkl")
    return model, features


model, feature_names = load_model()

# Split tabs
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

    st.sidebar.title("Employee Details")

    # Input capture matching trained features
    BusinessTravel_Non_Travel = (
        st.sidebar.selectbox("Business Travel (Is Non-Travel?)", ["Yes", "No"]) == "Yes"
    )
    Department_Research_Development = (
        st.sidebar.selectbox("Department: Research & Dev?", ["Yes", "No"]) == "Yes"
    )
    JobRole_SalesExecutive = (
        st.sidebar.selectbox("Job Role: Sales Executive?", ["Yes", "No"]) == "Yes"
    )
    MaritalStatus_Single = st.sidebar.selectbox("Is Single?", ["Yes", "No"]) == "Yes"

    age = st.sidebar.slider("Age", 18, 60, 35)
    distance_from_home = st.sidebar.slider("Distance From Home", 1, 30, 10)
    monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 6000)
    num_companies_worked = st.sidebar.slider("Number of Companies Worked", 0, 10, 2)
    percent_salary_hike = st.sidebar.slider("% Salary Hike", 10, 30, 15)
    total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)

    # Assemble input vector
    input_dict = {
        "Age": age,
        "DistanceFromHome": distance_from_home,
        "MonthlyIncome": monthly_income,
        "NumCompaniesWorked": num_companies_worked,
        "PercentSalaryHike": percent_salary_hike,
        "TotalWorkingYears": total_working_years,
        "BusinessTravel_Non-Travel": int(BusinessTravel_Non_Travel),
        "Department_Research & Development": int(Department_Research_Development),
        "JobRole_Sales Executive": int(JobRole_SalesExecutive),
        "MaritalStatus_Single": int(MaritalStatus_Single),
    }

    input_data = pd.DataFrame([input_dict])[feature_names]

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
