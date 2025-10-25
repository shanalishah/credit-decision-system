import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ------------------
# 1. Load assets
# ------------------
MODEL_PATH = "random_forest_heloc.pkl"
DATA_PATH = "heloc_dataset_cleaned.csv"

best_rf = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["RiskPerformance"])  # model features
y = df["RiskPerformance"]

# Map raw feature names -> business language for display
FEATURE_MAPPING = {
    "ExternalRiskEstimate": "Risk Estimate Score",
    "MSinceOldestTradeOpen": "Months Since Oldest Credit Account",
    "MSinceMostRecentTradeOpen": "Months Since Most Recent Credit Account",
    "AverageMInFile": "Avg Age of Credit Accounts (Months)",
    "NumSatisfactoryTrades": "Satisfactory Trades",
    "NumTrades60Ever2DerogPubRec": "Trades 60+ Days Delinquent",
    "NumTrades90Ever2DerogPubRec": "Trades 90+ Days Delinquent",
    "PercentTradesNeverDelq": "% Trades Never Delinquent",
    "MSinceMostRecentDelq": "Months Since Most Recent Delinquency",
    "MaxDelq2PublicRecLast12M": "Worst Delinquency in Last 12M",
    "MaxDelqEver": "Worst Delinquency Ever",
    "NumTotalTrades": "Total Trades",
    "NumTradesOpeninLast12M": "Trades Opened (12M)",
    "PercentInstallTrades": "% Installment Trades",
    "MSinceMostRecentInqexcl7days": "Months Since Most Recent Inquiry",
    "NumInqLast6M": "Credit Inquiries (6M)",
    "NumInqLast6Mexcl7days": "Credit Inquiries (6M, excl last 7d)",
    "NetFractionRevolvingBurden": "Revolving Utilization",
    "NetFractionInstallBurden": "Installment Utilization",
    "NumRevolvingTradesWBalance": "Revolving Trades w/ Balance",
    "NumInstallTradesWBalance": "Installment Trades w/ Balance",
    "NumBank2NatlTradesWHighUtilization": "High-Utilization Bank/National Trades",
    "PercentTradesWBalance": "% Trades w/ Balance"
}

# Pre-build SHAP explainer once
explainer = shap.TreeExplainer(best_rf)

# ------------------
# 2. Helper functions
# ------------------

def make_prediction(one_row_df):
    """Return model class (0/1), probability of approval class=1, and shap values."""
    pred_class = best_rf.predict(one_row_df)[0]
    proba = best_rf.predict_proba(one_row_df)[0, 1]  # P(approved)
    shap_values = explainer.shap_values(one_row_df)
    # shap for "approved" class 1 if model is a classifier with list output
    shap_for_row = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    return pred_class, proba, shap_for_row

def build_risk_summary(one_row_df, shap_for_row):
    """Return 2-4 plain-English bullets explaining main risk drivers."""
    # Get absolute impact sort
    impact_df = pd.DataFrame({
        "feature": X.columns,
        "friendly": [FEATURE_MAPPING.get(f, f) for f in X.columns],
        "shap_value": shap_for_row,
        "abs_impact": [abs(v) for v in shap_for_row],
        "raw_value": one_row_df.iloc[0].values
    }).sort_values("abs_impact", ascending=False)

    bullets = []
    for _, r in impact_df.head(4).iterrows():
        friendly = r["friendly"]
        val = r["raw_value"]
        if "Utilization" in friendly:
            bullets.append(f"{friendly} is high ({val:.0f}%), which increases risk.")
        elif "Delinquency" in friendly or "Delinquent" in friendly or "Inquiries" in friendly:
            bullets.append(f"{friendly} = {val:.0f}, which signals recent credit stress.")
        else:
            bullets.append(f"{friendly} = {val:.0f} is a key driver in this decision.")
    return bullets

def shap_bar_plot(shap_for_row):
    # Build dataframe for bar chart
    plot_df = pd.DataFrame({
        "Feature": [FEATURE_MAPPING.get(f, f) for f in X.columns],
        "SHAP Value": shap_for_row
    }).sort_values("SHAP Value", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["Feature"], plot_df["SHAP Value"])
    plt.xlabel("Impact on Approval (positive = helps)")
    plt.title("Model Explanation - Feature Impact")
    plt.tight_layout()
    return plt

# ------------------
# 3. Streamlit UI
# ------------------

st.set_page_config(
    page_title="HELOC Underwriting Assistant",
    layout="wide"
)

st.title("HELOC Underwriting Assistant")
st.markdown(
    "This tool predicts approval likelihood for a Home Equity Line of Credit and highlights the top risk drivers. "
    "Use an example applicant from the dataset or enter a custom profile."
)

# --- Sidebar: Applicant selection / manual override
st.sidebar.header("Applicant Input")

mode = st.sidebar.radio(
    "How would you like to provide borrower data?",
    ["Use Sample Applicant", "Enter Manually"]
)

if mode == "Use Sample Applicant":
    # let user pick an index from the dataset (you can prettify by making an ApplicantID column)
    sample_index = st.sidebar.number_input(
        "Sample Applicant # (row index from dataset)",
        min_value=0,
        max_value=len(X)-1,
        value=0,
        step=1
    )
    user_row = X.iloc[[sample_index]].copy()

else:
    # Show only ~6 high-level business-facing inputs
    # We'll fill the rest with medians so the model can still run
    medians = X.median(numeric_only=True)

    RevolvingUtil = st.sidebar.number_input(
        "Revolving Utilization (%)",
        min_value=0.0, max_value=200.0,
        value=float(medians["NetFractionRevolvingBurden"])
    )

    Recent90Delq = st.sidebar.number_input(
        "Trades 90+ Days Delinquent",
        min_value=0.0, max_value=20.0,
        value=float(medians["NumTrades90Ever2DerogPubRec"])
    )

    Inquiries6M = st.sidebar.number_input(
        "Credit Inquiries (Last 6M)",
        min_value=0.0, max_value=20.0,
        value=float(medians["NumInqLast6M"])
    )

    MonthsSinceDelq = st.sidebar.number_input(
        "Months Since Most Recent Delinquency",
        min_value=0.0, max_value=200.0,
        value=float(medians["MSinceMostRecentDelq"])
    )

    PctNeverDelq = st.sidebar.number_input(
        "% Trades Never Delinquent",
        min_value=0.0, max_value=100.0,
        value=float(medians["PercentTradesNeverDelq"])
    )

    RiskScore = st.sidebar.number_input(
        "Risk Estimate Score",
        min_value=0.0, max_value=100.0,
        value=float(medians["ExternalRiskEstimate"])
    )

    # Build a full row for the model:
    user_row = medians.to_frame().T  # 1-row df of medians
    user_row["NetFractionRevolvingBurden"] = RevolvingUtil
    user_row["NumTrades90Ever2DerogPubRec"] = Recent90Delq
    user_row["NumInqLast6M"] = Inquiries6M
    user_row["MSinceMostRecentDelq"] = MonthsSinceDelq
    user_row["PercentTradesNeverDelq"] = PctNeverDelq
    user_row["ExternalRiskEstimate"] = RiskScore

# --- Predict button
run_pred = st.sidebar.button("Run Underwriting Decision")

# --- Main area
col1, col2, col3 = st.columns(3)

if run_pred:
    pred_class, proba, shap_row = make_prediction(user_row)

    decision_text = "Approved" if pred_class == 1 else "Denied"
    approval_pct = proba * 100.0

    # Card 1: Decision
    with col1:
        st.subheader("Decision")
        st.metric(
            label="Model Decision",
            value=decision_text,
            delta=f"Approval likelihood: {approval_pct:.1f}%"
        )
        if pred_class == 0:
            st.caption("This applicant would require manual review / collateral reassessment.")
        else:
            st.caption("This applicant meets automated approval criteria.")

    # Card 2: Risk Summary (plain English)
    bullets = build_risk_summary(user_row, shap_row)
    with col2:
        st.subheader("Key Risk / Support Factors")
        for b in bullets:
            st.write(f"- {b}")

    # Card 3: Compliance View (Top Drivers Table)
    drivers_df = (
        pd.DataFrame({
            "Feature": [FEATURE_MAPPING.get(f, f) for f in X.columns],
            "Applicant Value": user_row.iloc[0].values,
            "Impact on Approval": shap_row
        })
        .reindex(
            pd.Series(abs(shap_row)).sort_values(ascending=False).index
        )
        .head(5)
    )
    with col3:
        st.subheader("Top Drivers (For Audit)")
        st.dataframe(
            drivers_df,
            use_container_width=True
        )

    # Advanced explainer
    with st.expander("Show Model Explanation Chart"):
        fig = shap_bar_plot(shap_row)
        st.pyplot(fig)

else:
    st.info("Select or enter an applicant in the left panel, then click 'Run Underwriting Decision'.")
    st.caption("You’ll get approval decision, approval likelihood, and top risk drivers.")
