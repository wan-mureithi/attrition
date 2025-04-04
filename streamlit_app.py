import streamlit as st

st.title("Employee Attrition Predictor")
st.subheader("Make smarter HR decisions")
st.write("This app predicts whether an employee is at risk of leaving.")


st.sidebar.title("Employee Details")

age = st.sidebar.slider("Age", 18, 60, 35)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 6000)
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
job_level = st.sidebar.slider("Job Level", 1, 5, 2)
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
years_at_company = st.sidebar.slider("Years at Company", 0, 20, 5)

# One-hot categorical variables (must match training)
business_travel = st.sidebar.selectbox(
    "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
)
department = st.sidebar.selectbox(
    "Department", ["Sales", "Research & Development", "Human Resources"]
)
