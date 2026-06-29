import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model files
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Helper functions
def prepare_input(age, marital_status, dependents,
                  contract_type, monthly_charges,
                  tenure_months, phone_service,
                  internet_service, streaming_service,
                  customer_rating, number_of_complaints,
                  late_payments):

    marital_encoded = 1 if marital_status == "Married" else 0
    dependents_encoded = 1 if dependents == "Yes" else 0
    phone_encoded = 1 if phone_service == "Yes" else 0
    internet_encoded = 1 if internet_service == "Yes" else 0
    streaming_encoded = 1 if streaming_service == "Yes" else 0
    contract_one_year = 1 if contract_type == "One Year" else 0
    contract_two_year = 1 if contract_type == "Two Year" else 0
    num_services = (phone_encoded + internet_encoded +
                   streaming_encoded)
    total_charges = monthly_charges * tenure_months

    input_data = pd.DataFrame({
        'age': [age],
        'marital_status': [marital_encoded],
        'dependents': [dependents_encoded],
        'monthly_charges': [monthly_charges],
        'tenure_months': [tenure_months],
        'phone_service': [phone_encoded],
        'internet_service': [internet_encoded],
        'streaming_service': [streaming_encoded],
        'customer_rating': [customer_rating],
        'number_of_complaints': [number_of_complaints],
        'late_payments': [late_payments],
        'contract_type_One Year': [contract_one_year],
        'contract_type_Two Year': [contract_two_year],
        'number_of_services': [num_services],
        'total_charges': [total_charges]
    })

    numerical_cols = ['age', 'monthly_charges',
                     'tenure_months', 'customer_rating',
                     'number_of_complaints', 'late_payments']
    input_data[numerical_cols] = scaler.transform(
        input_data[numerical_cols])

    return input_data, num_services, total_charges


def get_risk_category(probability):
    if probability >= 0.7:
        return "🔴 HIGH RISK", "error"
    elif probability >= 0.4:
        return "🟡 MEDIUM RISK", "warning"
    else:
        return "🟢 LOW RISK", "success"


def show_result(probability):
    risk_label, risk_style = get_risk_category(probability)
    if risk_style == "error":
        st.error(risk_label)
    elif risk_style == "warning":
        st.warning(risk_label)
    else:
        st.success(risk_label)
    st.progress(float(probability))
    st.markdown(f"**Churn Probability: {probability:.1%}**")
    st.markdown(
        f"**Retention Probability: {1-probability:.1%}**")


def get_recommendations(r_contract, r_charges,
                        r_complaints, r_rating,
                        r_tenure, r_late, r_services):
    recommendations = []
    if r_contract == "Month-to-Month":
        recommendations.append({
            'Priority': '🔴 High',
            'Action': 'Offer Contract Upgrade',
            'Detail': 'Provide 20% discount on One Year contract'
        })
    if r_charges > 5000:
        recommendations.append({
            'Priority': '🔴 High',
            'Action': 'Offer Loyalty Discount',
            'Detail': 'Provide 15% monthly bill discount'
        })
    if r_complaints > 3:
        recommendations.append({
            'Priority': '🔴 High',
            'Action': 'Assign Dedicated Support Agent',
            'Detail': 'Personal support call within 24 hours'
        })
    if r_rating < 3:
        recommendations.append({
            'Priority': '🟡 Medium',
            'Action': 'Service Quality Survey',
            'Detail': 'Send satisfaction survey and '
                     'offer service upgrade'
        })
    if r_tenure < 12:
        recommendations.append({
            'Priority': '🟡 Medium',
            'Action': 'New Customer Welcome Program',
            'Detail': 'Free premium support for 3 months'
        })
    if r_late > 5:
        recommendations.append({
            'Priority': '🟡 Medium',
            'Action': 'Flexible Payment Plan',
            'Detail': 'Offer installment plan option'
        })
    if r_services < 2:
        recommendations.append({
            'Priority': '🟢 Low',
            'Action': 'Service Bundle Offer',
            'Detail': 'Discounted bundle to increase '
                     'services used'
        })
    return recommendations


# Title
st.title("📊 Customer Churn Prediction System")
st.markdown(
    "### Pakistani Telecom Customer Retention Dashboard")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("📋 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Select Page",
    ["🔍 Single Prediction & What-If Simulator",
     "💡 Retention Recommendations",
     "📂 Batch Prediction"],
    key="main_nav"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("Model: Logistic Regression")
st.sidebar.markdown("ROC-AUC: 0.715")
st.sidebar.markdown("Churn Recall: 66%")
st.sidebar.markdown("Training Records: 1600")


# ─────────────────────────────────────
# PAGE 1: SINGLE PREDICTION + WHAT-IF
# ─────────────────────────────────────
if page == "🔍 Single Prediction & What-If Simulator":

    st.markdown("## 🔍 Single Customer Prediction")
    st.markdown(
        "Enter customer details below and click "
        "Predict to see churn risk.")
    st.markdown("---")

    with st.form(key="prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demographics**")
            age = st.slider("Age", 18, 70, 30)
            marital_status = st.selectbox(
                "Marital Status", ["Single", "Married"])
            dependents = st.selectbox(
                "Dependents", ["No", "Yes"])

        with col2:
            st.markdown("**📋 Account Details**")
            contract_type = st.selectbox(
                "Contract Type",
                ["Month-to-Month", "One Year", "Two Year"])
            monthly_charges = st.slider(
                "Monthly Charges (PKR)", 500, 8000, 3000)
            tenure_months = st.slider(
                "Tenure (Months)", 1, 72, 12)

        with col3:
            st.markdown("**📡 Services & Satisfaction**")
            phone_service = st.selectbox(
                "Phone Service", ["No", "Yes"])
            internet_service = st.selectbox(
                "Internet Service", ["No", "Yes"])
            streaming_service = st.selectbox(
                "Streaming Service", ["No", "Yes"])
            customer_rating = st.slider(
                "Customer Rating", 1, 5, 3)
            number_of_complaints = st.slider(
                "Complaints", 0, 5, 0)
            late_payments = st.slider(
                "Late Payments", 0, 10, 0)

        submitted = st.form_submit_button(
            "🔍 Predict Churn",
            use_container_width=True)

    if submitted:
        input_data, num_services, total_charges = \
            prepare_input(
                age, marital_status, dependents,
                contract_type, monthly_charges,
                tenure_months, phone_service,
                internet_service, streaming_service,
                customer_rating, number_of_complaints,
                late_payments)

        probability = model.predict_proba(input_data)[0][1]
        risk_label, risk_style = get_risk_category(probability)

        st.markdown("---")
        st.markdown("## Prediction Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("### Churn Risk")
            show_result(probability)

            st.markdown("### Customer Summary")
            st.table(pd.DataFrame({
                'Attribute': [
                    'Contract Type',
                    'Monthly Charges',
                    'Tenure',
                    'Services Used',
                    'Total Value Paid to Date'],
                'Value': [
                    contract_type,
                    f"PKR {monthly_charges:,}",
                    f"{tenure_months} months",
                    f"{num_services} services",
                    f"PKR {total_charges:,}"]
            }))

        with res_col2:
            st.markdown("### 🔄 What-If Simulator")
            st.markdown(
                "What happens if we change this "
                "customer's contract type?")

            whatif_results = []
            for contract in [
                    "Month-to-Month", "One Year", "Two Year"]:
                sim_data, _, _ = prepare_input(
                    age, marital_status, dependents,
                    contract, monthly_charges,
                    tenure_months, phone_service,
                    internet_service, streaming_service,
                    customer_rating, number_of_complaints,
                    late_payments)
                sim_prob = model.predict_proba(sim_data)[0][1]
                current = " ← Current" if \
                    contract == contract_type else ""
                whatif_results.append({
                    'Contract Type': contract + current,
                    'Churn Probability': f"{sim_prob:.1%}",
                    'Risk Level': get_risk_category(sim_prob)[0]
                })
            st.table(pd.DataFrame(whatif_results))
            st.caption(
                "💡 Upgrading contract type is the single "
                "most effective retention action.")


# ─────────────────────────────────────
# PAGE 2: RETENTION RECOMMENDATIONS
# ─────────────────────────────────────
elif page == "💡 Retention Recommendations":

    st.markdown("## 💡 Retention Recommendation Engine")
    st.markdown(
        "Get personalized retention strategies "
        "for at-risk customers.")
    st.markdown("---")

    with st.form(key="recommendation_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demographics**")
            r_age = st.slider("Age", 18, 70, 30)
            r_marital = st.selectbox(
                "Marital Status", ["Single", "Married"])
            r_dependents = st.selectbox(
                "Dependents", ["No", "Yes"])

        with col2:
            st.markdown("**📋 Account Details**")
            r_contract = st.selectbox(
                "Contract Type",
                ["Month-to-Month", "One Year", "Two Year"])
            r_charges = st.slider(
                "Monthly Charges (PKR)", 500, 8000, 3000)
            r_tenure = st.slider(
                "Tenure (Months)", 1, 72, 12)

        with col3:
            st.markdown("**📡 Services & Satisfaction**")
            r_phone = st.selectbox(
                "Phone Service", ["No", "Yes"])
            r_internet = st.selectbox(
                "Internet Service", ["No", "Yes"])
            r_streaming = st.selectbox(
                "Streaming Service", ["No", "Yes"])
            r_rating = st.slider(
                "Customer Rating", 1, 5, 3)
            r_complaints = st.slider(
                "Complaints", 0, 5, 0)
            r_late = st.slider(
                "Late Payments", 0, 10, 0)

        rec_submitted = st.form_submit_button(
            "💡 Get Retention Recommendations",
            use_container_width=True)

    if rec_submitted:
        r_input, r_services, r_total = prepare_input(
            r_age, r_marital, r_dependents, r_contract,
            r_charges, r_tenure, r_phone, r_internet,
            r_streaming, r_rating, r_complaints, r_late)

        r_prob = model.predict_proba(r_input)[0][1]
        r_label, r_style = get_risk_category(r_prob)

        st.markdown("---")
        st.markdown("## Results")

        prob_col, rec_col = st.columns(2)

        with prob_col:
            st.markdown("### Customer Risk Level")
            show_result(r_prob)

        with rec_col:
            st.markdown("### Recommended Actions")
            recommendations = get_recommendations(
                r_contract, r_charges, r_complaints,
                r_rating, r_tenure, r_late, r_services)

            if not recommendations:
                st.success(
                    "✅ This customer is low risk. "
                    "Consider a loyalty reward to "
                    "maintain their satisfaction.")
            else:
                st.table(pd.DataFrame(recommendations))


# ─────────────────────────────────────
# PAGE 3: BATCH PREDICTION
# ─────────────────────────────────────
elif page == "📂 Batch Prediction":

    st.markdown("## 📂 Batch Churn Prediction")
    st.markdown(
        "Upload a CSV file to predict churn for "
        "multiple customers at once.")
    st.markdown("---")

    st.info(
        "📋 Required CSV columns: age, marital_status, "
        "dependents, contract_type, monthly_charges, "
        "tenure_months, phone_service, internet_service, "
        "streaming_service, customer_rating, "
        "number_of_complaints, late_payments")

    uploaded_file = st.file_uploader(
        "Upload your CSV file here", type=['csv'])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(
            f"✅ Successfully loaded "
            f"{len(batch_df)} customer records.")
        st.markdown("### Preview")
        st.dataframe(batch_df.head())

        if st.button("🚀 Run Batch Prediction",
                    use_container_width=True,
                    key="batch_run"):

            results = []
            progress_bar = st.progress(0)
            status = st.empty()
            total_rows = len(batch_df)

            for idx, (_, row) in enumerate(
                    batch_df.iterrows()):
                try:
                    inp, nsvc, tchg = prepare_input(
                        row['age'],
                        row['marital_status'],
                        row['dependents'],
                        row['contract_type'],
                        row['monthly_charges'],
                        row['tenure_months'],
                        row['phone_service'],
                        row['internet_service'],
                        row['streaming_service'],
                        row['customer_rating'],
                        row['number_of_complaints'],
                        row['late_payments'])

                    prob = model.predict_proba(inp)[0][1]
                    risk, _ = get_risk_category(prob)

                    results.append({
                        'Customer ID': row.get(
                            'customer_id',
                            f'CUST-{idx+1}'),
                        'Churn Probability': f"{prob:.1%}",
                        'Risk Level': risk
                    })
                except Exception as e:
                    results.append({
                        'Customer ID': row.get(
                            'customer_id',
                            f'CUST-{idx+1}'),
                        'Churn Probability': 'Error',
                        'Risk Level': str(e)
                    })

                progress_bar.progress(
                    (idx + 1) / total_rows)
                status.text(
                    f"Processing {idx+1} "
                    f"of {total_rows} customers...")

            status.text("✅ Complete!")
            results_df = pd.DataFrame(results)

            st.markdown("### Prediction Results")
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

            st.markdown("### Batch Summary")
            total = len(results_df)
            high = results_df[
                results_df['Risk Level'].str.contains(
                    'HIGH')].shape[0]
            medium = results_df[
                results_df['Risk Level'].str.contains(
                    'MEDIUM')].shape[0]
            low = results_df[
                results_df['Risk Level'].str.contains(
                    'LOW')].shape[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Total Customers", total)
            c2.metric("🔴 High Risk", high)
            c3.metric("🟡 Medium Risk", medium)
            c4.metric("🟢 Low Risk", low)

    else:
        st.markdown("### How Batch Prediction Works")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Steps:**
            1. Prepare your CSV with required columns
            2. Upload the file above
            3. Click Run Batch Prediction
            4. Download results with risk levels
            """)
        with col2:
            st.markdown("""
            **Output includes:**
            - Churn probability for each customer
            - Risk category (High/Medium/Low)
            - Downloadable CSV with all results
            - Summary metrics dashboard
            """)
        st.info(
            "💡 Tip: You can use your generated "
            "telecom_churn_dataset.csv to test "
            "batch prediction on all 2000 customers.")