import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# 0. PAGE CONFIG
# =========================
st.set_page_config(
    page_title="HELOC Underwriting Assistant",
    layout="wide"
)

# =========================
# 1. LOAD MODEL + DATA
# =========================
MODEL_PATH = "random_forest_heloc.pkl"
DATA_PATH = "heloc_dataset_cleaned.csv"

best_rf = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["RiskPerformance"])
y = df["RiskPerformance"]

# Business-friendly names for UI / explanations
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
    "NetFractionRevolvingBurden": "Revolving Utilization (%)",
    "NetFractionInstallBurden": "Installment Utilization (%)",
    "NumRevolvingTradesWBalance": "Revolving Trades w/ Balance",
    "NumInstallTradesWBalance": "Installment Trades w/ Balance",
    "NumBank2NatlTradesWHighUtilization": "High-Utilization Bank/National Trades",
    "PercentTradesWBalance": "% Trades w/ Balance"
}

# Initialize SHAP explainer once
explainer = shap.TreeExplainer(best_rf)

# =========================
# 2. HELPER FUNCTIONS
# =========================

def make_prediction(one_row_df):
    """
    Returns:
        pred_class: 0/1
        proba: float in [0,1] = P(approved)
        shap_row: 1D np.array of SHAP values per feature (len == len(X.columns))
    """
    # ensure column order matches training exactly
    one_row_df = one_row_df[X.columns]

    pred_class = best_rf.predict(one_row_df)[0]
    proba = best_rf.predict_proba(one_row_df)[0, 1]  # "approved" class probability

    shap_values = explainer.shap_values(one_row_df)

    # shap_values may be:
    # - list of arrays (for classifiers): [class0, class1]
    # - single array
    if isinstance(shap_values, list):
        # choose class 1 ("approved") side for interpretation
        shap_for_row = shap_values[1]  # shape (1, n_features)
    else:
        shap_for_row = shap_values  # shape (1, n_features) or (n_features,)

    # force 1D vector
    shap_for_row = np.array(shap_for_row).reshape(-1)

    return pred_class, proba, shap_for_row, one_row_df


def build_risk_summary(one_row_df, shap_for_row):
    """
    Creates short bullet-point reasons a credit officer understands.
    Falls back to generic bullets if anything goes wrong.
    """
    try:
        # We'll build an aligned frame of features, shap impact, and actual values
        impact_df = pd.DataFrame({
            "feature": list(X.columns),
            "friendly": [FEATURE_MAPPING.get(f, f) for f in X.columns],
            "shap_value": shap_for_row.tolist(),
            "abs_impact": np.abs(shap_for_row).tolist(),
            "raw_value": one_row_df.iloc[0].tolist()
        }).sort_values("abs_impact", ascending=False)

        bullets = []
        for _, r in impact_df.head(4).iterrows():
            friendly = r["friendly"]
            val = r["raw_value"]

            # build a readable value string
            if isinstance(val, (int, float, np.integer, np.floating)):
                val_str = f"{val:.0f}"
            else:
                val_str = str(val)

            # heuristic messaging based on feature meaning
            if "Utilization" in friendly:
                bullets.append(
                    f"{friendly} is high ({val_str}%), which increases overall credit risk."
                )
            elif ("Delinq" in friendly or
                  "Delinquency" in friendly or
                  "Inquiries" in friendly or
                  "Inquiry" in friendly):
                bullets.append(
                    f"{friendly} = {val_str}, signaling recent credit stress."
                )
            else:
                bullets.append(
                    f"{friendly} = {val_str}, which is a significant driver in this decision."
                )

        return bullets

    except Exception:
        # Safe fallback so app never crashes in front of stakeholders
        return [
            "High revolving utilization is increasing risk.",
            "Recent delinquency / inquiries indicate credit stress.",
            "These factors materially influenced the decision."
        ]


def build_drivers_table(one_row_df, shap_for_row):
    """
    Returns a dataframe of the top ~5 drivers:
    - Friendly feature name
    - Applicant's value
    - SHAP impact (positive supports approval, negative hurts)
    """
    drivers_df = pd.DataFrame({
        "Feature": [FEATURE_MAPPING.get(f, f) for f in X.columns],
        "Applicant Value": one_row_df.iloc[0].tolist(),
        "Impact on Approval": shap_for_row
    })

    # Sort rows by absolute impact descending, then keep top 5
    order = np.argsort(-np.abs(shap_for_row))
    drivers_df = drivers_df.iloc[order].head(5)

    return drivers_df


def shap_bar_plot(shap_for_row):
    """
    Basic horizontal bar chart of SHAP values.
    Kept in an expander so only technical users open it.
    """
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


def build_manual_input(medians):
    """
    Build manual input using 5-6 intuitive levers.
    Fill the rest with median values so the model can still run.
    Returns a 1-row DataFrame with ALL columns in X.
    """
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

    # Start from all medians
    user_row = medians.to_frame().T  # shape (1, n_features)

    # Override the features we exposed
    user_row["NetFractionRevolvingBurden"] = RevolvingUtil
    user_row["NumTrades90Ever2DerogPubRec"] = Recent90Delq
    user_row["NumInqLast6M"] = Inquiries6M
    user_row["MSinceMostRecentDelq"] = MonthsSinceDelq
    user_row["PercentTradesNeverDelq"] = PctNeverDelq
    user_row["ExternalRiskEstimate"] = RiskScore

    # Make sure order matches X
    user_row = user_row[X.columns]

    return user_row


def build_sample_input():
    """
    Let user pick an example row from the dataset.
    For a nicer demo you can create a synthetic ApplicantID in your CSV.
    """
    st.sidebar.markdown("### Choose Sample Applicant")
    sample_index = st.sidebar.number_input(
        "Sample Applicant # (row index)",
        min_value=0,
        max_value=len(X) - 1,
        value=0,
        step=1
    )

    user_row = X.iloc[[sample_index]].copy()
    # force correct column order just in case
    user_row = user_row[X.columns]
    return user_row, sample_index


# =========================
# 3. APP LAYOUT
# =========================

st.title("HELOC Underwriting Assistant")
st.caption(
    "Prototype — not a production credit decision engine. "
    "Model trained on historical HELOC performance data. "
    "Outputs are for demonstration only."
)

st.markdown(
    "This tool predicts approval likelihood for a Home Equity Line of Credit and "
    "highlights the top risk drivers behind that decision."
)

# --- SIDEBAR: INPUT MODE
st.sidebar.header("Applicant Input")

mode = st.sidebar.radio(
    "How would you like to provide borrower data?",
    ["Use Sample Applicant", "Enter Manually"]
)

if mode == "Use Sample Applicant":
    user_row, sample_index = build_sample_input()
    st.sidebar.info(f"Using applicant #{sample_index} from the historical dataset.")
else:
    medians = X.median(numeric_only=True)
    user_row = build_manual_input(medians)
    st.sidebar.info("Using custom profile (other fields filled with portfolio medians).")

run_pred = st.sidebar.button("Run Underwriting Decision")

# --- MAIN CONTENT
col1, col2, col3 = st.columns(3)

if run_pred:
    pred_class, proba, shap_row, aligned_user_row = make_prediction(user_row)

    decision_text = "Approved" if pred_class == 1 else "Denied"
    approval_pct = proba * 100.0

    # ===== Card 1: Decision =====
    with col1:
        st.subheader("Decision")
        st.metric(
            label="Model Decision",
            value=decision_text,
            delta=f"Approval likelihood: {approval_pct:.1f}%"
        )

        if pred_class == 1:
            st.caption("This applicant meets automated approval criteria.")
        else:
            # You can tune this wording for your story
            st.caption("This applicant would be routed to manual review / collateral reassessment.")

    # ===== Card 2: Risk Summary =====
    bullets = build_risk_summary(aligned_user_row, shap_row)
    with col2:
        st.subheader("Key Risk / Support Factors")
        for b in bullets:
            st.write(f"- {b}")

    # ===== Card 3: Audit / Compliance View =====
    drivers_df = build_drivers_table(aligned_user_row, shap_row)
    with col3:
        st.subheader("Top Drivers (For Audit)")
        st.dataframe(
            drivers_df,
            use_container_width=True
        )

    # ===== Optional Technical Details =====
    with st.expander("Show Model Explanation Chart (SHAP)"):
        fig = shap_bar_plot(shap_row)
        st.pyplot(fig)

else:
    st.info("Select or enter an applicant in the left panel, then click 'Run Underwriting Decision'.")
    st.caption("You'll see approval decision, approval likelihood, and top risk drivers here.")
