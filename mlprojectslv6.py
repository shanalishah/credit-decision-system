import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="HELOC Underwriting Assistant",
    layout="wide"
)

# =========================
# LOAD MODEL + DATA
# =========================
MODEL_PATH = "random_forest_heloc.pkl"
DATA_PATH = "heloc_dataset_cleaned.csv"

best_rf = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["RiskPerformance"])
y = df["RiskPerformance"]

# Mapping raw model features -> business readable labels
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

# Pre-build SHAP explainer
explainer = shap.TreeExplainer(best_rf)

# =========================
# HELPER FUNCTIONS
# =========================

def _align_vectors(one_row_df, shap_for_row):
    """
    Make sure features, applicant values, and SHAP values all have the same length.
    Returns aligned lists/arrays so downstream code never crashes.
    """
    # canonical feature list from training
    feature_list = list(X.columns)

    # applicant values in that order
    applicant_values = one_row_df.iloc[0][feature_list].tolist()

    # flatten SHAP and force into python list
    shap_flat = np.array(shap_for_row).reshape(-1).tolist()

    # lengths
    n_features = len(feature_list)
    n_values = len(applicant_values)
    n_shap = len(shap_flat)

    min_len = min(n_features, n_values, n_shap)

    # truncate consistently
    feature_list = feature_list[:min_len]
    applicant_values = applicant_values[:min_len]
    shap_flat = shap_flat[:min_len]

    # business-friendly names
    friendly_list = [FEATURE_MAPPING.get(f, f) for f in feature_list]

    return feature_list, friendly_list, applicant_values, np.array(shap_flat)


def make_prediction(one_row_df):
    """
    Run model prediction on a single-row DF.
    Returns:
        pred_class: int (0=deny,1=approve)
        proba: float (P(approve))
        shap_row: np.ndarray (1D, can be longer than features but we'll align later)
        aligned_row_df: row reordered to match X.columns
    """
    # ensure same columns/order as training data
    one_row_df = one_row_df[X.columns]

    pred_class = best_rf.predict(one_row_df)[0]
    proba = best_rf.predict_proba(one_row_df)[0, 1]  # probability of class 1 (approved)

    shap_values = explainer.shap_values(one_row_df)

    # TreeExplainer on classifier often returns list: [class0, class1]
    if isinstance(shap_values, list):
        shap_for_row = shap_values[1]  # explanation towards "approved"
    else:
        shap_for_row = shap_values

    # force 1D
    shap_for_row = np.array(shap_for_row).reshape(-1)

    return pred_class, proba, shap_for_row, one_row_df


def build_risk_summary(one_row_df, shap_for_row):
    """
    Generate plain-English bullets about top drivers.
    It's safe: if anything goes weird, returns fallback bullets.
    """
    try:
        (
            feature_list,
            friendly_list,
            applicant_values,
            shap_used
        ) = _align_vectors(one_row_df, shap_for_row)

        impact_df = pd.DataFrame({
            "feature": feature_list,
            "friendly": friendly_list,
            "raw_value": applicant_values,
            "shap_value": shap_used,
            "abs_impact": np.abs(shap_used)
        }).sort_values("abs_impact", ascending=False)

        bullets = []
        for _, r in impact_df.head(4).iterrows():
            friendly = r["friendly"]
            val = r["raw_value"]

            if isinstance(val, (int, float, np.integer, np.floating)):
                val_str = f"{val:.0f}"
            else:
                val_str = str(val)

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
        return [
            "High revolving utilization is increasing risk.",
            "Recent delinquency / inquiries indicate credit stress.",
            "These factors materially influenced the decision."
        ]


def build_drivers_table(one_row_df, shap_for_row):
    """
    Build a compliance/audit-friendly table:
    Feature | Applicant Value | Impact on Approval
    Always aligned + truncated so pandas doesn't explode.
    """
    (
        feature_list,
        friendly_list,
        applicant_values,
        shap_used
    ) = _align_vectors(one_row_df, shap_for_row)

    drivers_df = pd.DataFrame({
        "Feature": friendly_list,
        "Applicant Value": applicant_values,
        "Impact on Approval": shap_used
    })

    # sort by |impact|
    order = np.argsort(-np.abs(shap_used))
    drivers_df = drivers_df.iloc[order].head(5).reset_index(drop=True)

    return drivers_df


def shap_bar_plot(shap_for_row):
    """
    Horizontal bar chart of SHAP values.
    This is under an expander in the UI because it's more technical.
    """
    # For plotting we don't need alignment strictness, but it's nice
    # to only plot what we can label clearly.
    (
        feature_list,
        friendly_list,
        applicant_values,
        shap_used
    ) = _align_vectors(pd.DataFrame([X.iloc[0]]), shap_for_row)

    plot_df = pd.DataFrame({
        "Feature": friendly_list,
        "SHAP Value": shap_used
    }).sort_values("SHAP Value", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["Feature"], plot_df["SHAP Value"])
    plt.xlabel("Impact on Approval (positive = helps)")
    plt.title("Model Explanation - Feature Impact")
    plt.tight_layout()
    return plt


def build_manual_input(medians):
    """
    Sidebar manual input:
    We expose a few intuitive levers and fill the rest with medians.
    Returns a 1-row DataFrame with all model features.
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

    # start with medians for every feature
    user_row = medians.to_frame().T  # shape (1, n_features)

    # override the ones we asked for
    user_row["NetFractionRevolvingBurden"] = RevolvingUtil
    user_row["NumTrades90Ever2DerogPubRec"] = Recent90Delq
    user_row["NumInqLast6M"] = Inquiries6M
    user_row["MSinceMostRecentDelq"] = MonthsSinceDelq
    user_row["PercentTradesNeverDelq"] = PctNeverDelq
    user_row["ExternalRiskEstimate"] = RiskScore

    # ensure same col order
    user_row = user_row[X.columns]

    return user_row


def build_sample_input():
    """
    Sidebar sample input mode:
    User picks a row index from the historical dataset.
    You can later replace this with a dropdown of fake ApplicantIDs.
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
    user_row = user_row[X.columns]
    return user_row, sample_index


# =========================
# UI HEADER
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

# =========================
# SIDEBAR INPUT MODE
# =========================
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

# =========================
# MAIN OUTPUT AREA
# =========================
col1, col2, col3 = st.columns(3)

if run_pred:
    pred_class, proba, shap_row, aligned_user_row = make_prediction(user_row)

    decision_text = "Approved" if pred_class == 1 else "Denied"
    approval_pct = proba * 100.0

    # (Optional but helpful while you're polishing: shows if SHAP aligns with features)
    st.caption(
        f"Debug info (will not be in production): "
        f"features={len(X.columns)}, "
        f"shap={len(np.array(shap_row).reshape(-1))}, "
        f"row_vals={aligned_user_row.shape[1]}"
    )

    # ----- Card 1: Decision -----
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
            st.caption("This applicant would be routed to manual review / collateral reassessment.")

    # ----- Card 2: Risk Summary -----
    bullets = build_risk_summary(aligned_user_row, shap_row)
    with col2:
        st.subheader("Key Risk / Support Factors")
        for b in bullets:
            st.write(f"- {b}")

    # ----- Card 3: Compliance / Audit Drivers -----
    drivers_df = build_drivers_table(aligned_user_row, shap_row)
    with col3:
        st.subheader("Top Drivers (For Audit)")
        st.dataframe(
            drivers_df,
            use_container_width=True
        )

    # ----- Technical Explainability (Expandable) -----
    with st.expander("Show Model Explanation Chart (SHAP)"):
        fig = shap_bar_plot(shap_row)
        st.pyplot(fig)

else:
    st.info("Select or enter an applicant in the left panel, then click 'Run Underwriting Decision'.")
    st.caption("You'll see approval decision, approval likelihood, and top risk drivers here.")
