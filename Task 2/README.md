# Customer Churn Prediction System
## Teyzix Core Internship — Task 2

**Intern:** Hina
**Reference ID:** TC-INT-20260607-697
**Submission Date:** June 29, 2026

---

## Project Overview

This project builds a complete customer churn prediction system
for a Pakistani telecom company from scratch. Unlike Task 1 which
used a provided dataset, this project involved designing and
generating a realistic synthetic dataset, building ML models, and
creating an interactive web dashboard.

---

## Dataset

- Records: 2000 synthetic customers
- Features: 18 columns
- Target: Churn (0 = Stayed, 1 = Churned)
- Churn Rate: ~30% (realistic class imbalance)
- Currency: Pakistani Rupees (PKR)
- Cities: Karachi, Lahore, Islamabad, Peshawar, Quetta

The dataset was generated using probability-based churn logic
where churn is influenced by contract type, monthly charges,
tenure, complaints, late payments, and customer rating.

---

## Project Structure

Task2/
├── churn_prediction.ipynb    # Main notebook
├── app.py                    # Streamlit dashboard
├── telecom_churn_dataset.csv # Generated dataset
├── churn_model.pkl           # Trained model
├── scaler.pkl                # Fitted scaler
├── feature_names.pkl         # Feature names
└── README.md                 # This file

---

## How to Run

1. Install Requirements
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn

2. Run the Notebook
Open churn_prediction.ipynb in Jupyter Notebook and run all cells.

3. Launch the Dashboard
streamlit run app.py

---

## ML Pipeline

Phase 1 - Dataset Creation: Synthetic data generation with realistic churn logic
Phase 2 - EDA: Distribution analysis, churn rates, correlation heatmap
Phase 3 - Data Preparation: Encoding, scaling, train-test split
Phase 4 - Feature Engineering: Created number_of_services and total_charges
Phase 5 - Model Building: Logistic Regression and Random Forest
Phase 6 - Model Evaluation: Classification report, confusion matrix, ROC-AUC
Phase 7 - Prediction Interface: Streamlit web dashboard

---

## Model Performance

Logistic Regression:
- Accuracy: 64%
- Churn Recall: 66%
- ROC-AUC: 0.715

Random Forest:
- Accuracy: 70%
- Churn Recall: 17%
- ROC-AUC: 0.695

Selected Model: Logistic Regression

Reason: Higher recall for churners (66% vs 17%). In churn
prediction, missing a churner is more costly than a false
alarm. Logistic Regression catches significantly more
at-risk customers despite lower overall accuracy.

---

## Dashboard Features

Page 1 - Single Prediction and What-If Simulator:
- Enter customer details and get instant churn probability
- What-If Simulator shows how contract upgrade reduces churn risk

Page 2 - Retention Recommendations:
- Personalized action recommendations based on risk factors
- Priority-based suggestions for retention team

Page 3 - Batch Prediction:
- Upload CSV with multiple customers
- Get churn predictions for all at once
- Download results as CSV

---

## Key Findings

- Contract type is the strongest churn predictor
- Month-to-Month customers churn at 40%+ vs 16% for Two Year
- High monthly charges and low tenure increase churn risk
- Upgrading contract type is the single most effective retention action

---

## Business Recommendations

1. Prioritize outreach to Month-to-Month customers
2. Offer contract upgrade discounts to high-risk customers
3. New customers under 12 months need special attention
4. High complaint customers should get dedicated support
5. Use batch prediction monthly to identify at-risk customers