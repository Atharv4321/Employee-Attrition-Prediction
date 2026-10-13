import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
EXPECTED_FEATURE_COUNT = 44
PREDICTION_THRESHOLD = 0.40


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    with open(BASE_DIR / "employee_attrition_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(BASE_DIR / "feature_columns.pkl", "rb") as f:
        columns = pickle.load(f)

    if len(columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected 44 features, but found {len(columns)}."
        )

    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate feature names found.")

    return model, columns


try:
    model, feature_columns = load_model()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()


# -------------------- STYLING --------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1>📊 Employee Attrition Predictor</h1>
        <p>
            Explore a machine-learning prediction using employee data.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "Educational demo only. Predictions are estimates and must not "
    "be used to make employment decisions about individuals."
)


# -------------------- INPUT FORM --------------------
with st.form("employee_form"):

    st.header("Employee Information")

    # Personal and job details
    st.subheader("Personal and Job Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Female", "Male"])

        marital = st.selectbox(
            "Marital Status",
            ["Divorced", "Married", "Single"]
        )

        distance = st.number_input(
            "Distance From Home", 1, 29, 7
        )

        education = st.selectbox(
            "Education Level", [1, 2, 3, 4, 5]
        )

        education_field = st.selectbox(
            "Education Field",
            [
                "Human Resources",
                "Life Sciences",
                "Marketing",
                "Medical",
                "Other",
                "Technical Degree"
            ]
        )

    with c2:
        department = st.selectbox(
            "Department",
            [
                "Human Resources",
                "Research & Development",
                "Sales"
            ]
        )

        job_role = st.selectbox(
            "Job Role",
            [
                "Healthcare Representative",
                "Human Resources",
                "Laboratory Technician",
                "Manager",
                "Manufacturing Director",
                "Research Director",
                "Research Scientist",
                "Sales Executive",
                "Sales Representative"
            ]
        )

        business_travel = st.selectbox(
            "Business Travel",
            [
                "Non-Travel",
                "Travel_Frequently",
                "Travel_Rarely"
            ]
        )

        overtime = st.selectbox(
            "Works Overtime?",
            ["No", "Yes"]
        )

        job_level = st.selectbox(
            "Job Level", [1, 2, 3, 4, 5]
        )

    with c3:
        daily_rate = st.number_input(
            "Daily Rate", 102, 1499, 800
        )

        hourly_rate = st.number_input(
            "Hourly Rate", 30, 100, 65
        )

        monthly_income = st.number_input(
            "Monthly Income", 1009, 19999, 5000
        )

        monthly_rate = st.number_input(
            "Monthly Rate", 2094, 26999, 14000
        )

        num_companies = st.number_input(
            "Number of Previous Companies", 0, 9, 2
        )

    # Satisfaction and work environment
    st.subheader("Satisfaction and Work Environment")
    c1, c2, c3 = st.columns(3)

    with c1:
        environment = st.selectbox(
            "Environment Satisfaction", [1, 2, 3, 4]
        )

        job_sat = st.selectbox(
            "Job Satisfaction", [1, 2, 3, 4]
        )

        involvement = st.selectbox(
            "Job Involvement", [1, 2, 3, 4]
        )

        relationship = st.selectbox(
            "Relationship Satisfaction", [1, 2, 3, 4]
        )

    with c2:
        work_life = st.selectbox(
            "Work-Life Balance", [1, 2, 3, 4]
        )

        stock_option = st.selectbox(
            "Stock Option Level", [0, 1, 2, 3]
        )

        performance = st.selectbox(
            "Performance Rating", [1, 2, 3, 4]
        )

        salary_hike = st.number_input(
            "Percent Salary Hike", 11, 25, 15
        )

    with c3:
        training = st.number_input(
            "Training Times Last Year", 0, 6, 3
        )

        total_years = st.number_input(
            "Total Working Years", 0, 40, 8
        )

        years_company = st.number_input(
            "Years at Company", 0, 40, 5
        )

        years_role = st.number_input(
            "Years in Current Role", 0, 18, 3
        )

    # Career history
    st.subheader("Career History")
    c1, c2 = st.columns(2)

    with c1:
        years_promotion = st.number_input(
            "Years Since Last Promotion", 0, 15, 1
        )

    with c2:
        years_manager = st.number_input(
            "Years With Current Manager", 0, 17, 3
        )

    submitted = st.form_submit_button(
        "🔍 Predict Attrition",
        type="primary",
        use_container_width=True
    )


# -------------------- PREDICTION --------------------
if submitted:

    # Create all features in the exact training order.
    employee = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns,
        dtype=float
    )

    # Numerical features
    numeric_values = {
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance,
        "Education": education,
        "EnvironmentSatisfaction": environment,
        "HourlyRate": hourly_rate,
        "JobInvolvement": involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_sat,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies,
        "PercentSalaryHike": salary_hike,
        "PerformanceRating": performance,
        "RelationshipSatisfaction": relationship,
        "StockOptionLevel": stock_option,
        "TotalWorkingYears": total_years,
        "TrainingTimesLastYear": training,
        "WorkLifeBalance": work_life,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promotion,
        "YearsWithCurrManager": years_manager
    }

    for name, value in numeric_values.items():
        if name not in employee.columns:
            st.error(f"Missing numerical feature: {name}")
            st.stop()

        employee.loc[0, name] = value

    # Categorical features
    categorical_values = {
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital,
        "OverTime": overtime
    }

    # Set the corresponding one-hot encoded column to 1.
    # Baseline categories remain 0 because training used drop_first=True.
    for name, value in categorical_values.items():
        encoded_name = f"{name}_{value}"

        if encoded_name in employee.columns:
            employee.loc[0, encoded_name] = 1

    # Validate model inputs
    if list(employee.columns) != feature_columns:
        st.error(
            "Feature order mismatch. Check feature_columns.pkl."
        )
        st.stop()

    if employee.shape != (1, EXPECTED_FEATURE_COUNT):
        st.error("Expected exactly 44 model features.")
        st.stop()

    if employee.isnull().any().any():
        st.error("Missing values found in model inputs.")
        st.stop()

    st.caption(
        "Input validation passed: all 44 model features are aligned."
    )

    # Predict
    try:
        probabilities = model.predict_proba(employee)[0]

        if len(probabilities) != 2:
            st.error(
                "The loaded model does not return two class probabilities."
            )
            st.stop()

        probability = float(probabilities[1])
        prediction = probability >= PREDICTION_THRESHOLD

        if not 0 <= probability <= 1:
            st.error("The model returned an invalid probability.")
            st.stop()

        # Display results
        st.divider()
        st.header("Prediction Result")

        left, right = st.columns(2)

        with left:
            if prediction:
                st.error(
                    "Higher estimated likelihood of attrition"
                )
            else:
                st.success(
                    "Lower estimated likelihood of attrition"
                )

        with right:
            st.metric(
                "Estimated Attrition Probability",
                f"{probability:.1%}"
            )

        st.progress(probability)

        st.caption(
            "The 0.40 threshold was explored during model development "
            "and has not been independently validated. This prediction "
            "is uncertain and must not be treated as a reliable assessment "
            "of an individual employee."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

