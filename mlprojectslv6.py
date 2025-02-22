import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load trained model
model_path = "random_forest_heloc.pkl"
best_rf = joblib.load(model_path)

# Load dataset for feature reference
data_path = "heloc_dataset_cleaned.csv"
df = pd.read_csv(data_path)
X = df.drop(columns=['RiskPerformance'])

# Manually defined business-friendly feature names
feature_mapping = {
    "ExternalRiskEstimate": "Risk Estimate Score",
    "MSinceOldestTradeOpen": "Months Since Oldest Credit Account",
    "MSinceMostRecentTradeOpen": "Months Since Most Recent Credit Account",
    "AverageMInFile": "Average Age of Credit Accounts (Months)",
    "NumSatisfactoryTrades": "Number of Satisfactory Trades",
    "NumTrades60Ever2DerogPubRec": "Trades 60+ Days Delinquent",
    "NumTrades90Ever2DerogPubRec": "Trades 90+ Days Delinquent",
    "PercentTradesNeverDelq": "Percent of Trades Never Delinquent",
    "MSinceMostRecentDelq": "Months Since Most Recent Delinquency",
    "MaxDelq2PublicRecLast12M": "Max Delinquency Last 12 Months",
    "MaxDelqEver": "Max Delinquency Ever",
    "NumTotalTrades": "Total Number of Trades",
    "NumTradesOpeninLast12M": "Trades Opened in Last 12 Months",
    "PercentInstallTrades": "Percent Installment Trades",
    "MSinceMostRecentInqexcl7days": "Months Since Most Recent Inquiry",
    "NumInqLast6M": "Inquiries in Last 6 Months",
    "NumInqLast6Mexcl7days": "Inquiries in Last 6 Months (Excl. Last 7 Days)",
    "NetFractionRevolvingBurden": "Revolving Credit Utilization",
    "NetFractionInstallBurden": "Installment Credit Utilization",
    "NumRevolvingTradesWBalance": "Revolving Trades with Balance",
    "NumInstallTradesWBalance": "Installment Trades with Balance",
    "NumBank2NatlTradesWHighUtilization": "Bank/National Trades with High Utilization",
    "PercentTradesWBalance": "Percent of Trades with Balance"
}

# Rename features for display
X_display = X.rename(columns=feature_mapping)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(best_rf)

# Streamlit App
st.title("HELOC Loan Decision Support System")
st.write("Enter customer details below to predict loan approval.")

def predict_loan(user_input):
    prediction = best_rf.predict(user_input)[0]
    probability = best_rf.predict_proba(user_input)[0, 1]
    explanation = explainer.shap_values(user_input)
    return prediction, probability, explanation

# User Input Form
user_data = []
for col in X.columns:
    user_data.append(st.number_input(f"Enter {feature_mapping.get(col, col)}", value=float(df[col].median())))

if st.button("Predict Loan Decision"):
    user_df = pd.DataFrame([user_data], columns=X.columns)
    
    # Make Prediction
    prediction, probability, explanation = predict_loan(user_df)
    decision = "Approved" if prediction == 1 else "Denied"
    
    st.write(f"**Loan Decision:** {decision}")
    st.write(f"**Approval Probability:** {probability:.2f}")

    # SHAP Bar Chart Explanation
    try:
        shap_values = explanation[1] if isinstance(explanation, list) else explanation
        shap_values_sample = shap_values.flatten()[:X.shape[1]]  # Fix shape issue

        # Create SHAP Bar Chart
        feature_importance = pd.DataFrame({
            "Feature": X_display.columns,
            "SHAP Value": shap_values_sample
        }).sort_values(by="SHAP Value", ascending=True)  # Sorting for better visualization

        plt.figure(figsize=(12, 8))  # Increase figure size
        plt.barh(feature_importance["Feature"], feature_importance["SHAP Value"], color='skyblue')
        plt.xlabel("SHAP Value (Impact on Decision)")
        plt.ylabel("Feature")
        plt.title("Top Features Influencing Loan Decision")
        plt.xticks(rotation=0)  # Keep x-axis text horizontal
        plt.yticks(fontsize=10)  # Increase font size for readability
        plt.tight_layout()  # Adjust layout to fit text
        plt.savefig("shap_bar_plot.png")
        st.image("shap_bar_plot.png")

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
