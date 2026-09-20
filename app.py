import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.api_client import CreditRiskApiClient, CreditRiskApiError
from src.utils import FeatureEngineering

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CreditIQ — Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.main { background: #F7F6F2; }

.topbar {
    background: #1A1A2E;
    padding: 14px 24px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}
.topbar-left { display: flex; align-items: baseline; gap: 12px; }
.topbar-title { font-family: 'DM Serif Display', serif; font-size: 22px; color: #E2E8F0; }
.topbar-sub { font-size: 12px; color: #6B7A99; }
.topbar-right { display: flex; align-items: center; gap: 8px; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #5DCAA5; display: inline-block; }
.status-text { font-size: 12px; color: #9FE1CB; }
.threshold-text { font-size: 12px; color: #6B7A99; }

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.25rem;
    border: 0.5px solid #EBEBEB;
}
.metric-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 6px;
}
.metric-value { font-size: 28px; font-weight: 500; line-height: 1; color: #1A1A2E; }
.metric-sub { font-size: 12px; color: #aaa; margin-top: 5px; }

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 18px;
    color: #1A1A2E;
    margin-bottom: 4px;
}
.section-sub { font-size: 13px; color: #888; margin-bottom: 1rem; }

.feature-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 0.5px solid #F0F0F0;
    font-size: 13px;
}
.feature-row:last-child { border-bottom: none; }
.sig-red   { color: #791F1F; font-weight: 500; }
.sig-amber { color: #633806; font-weight: 500; }
.sig-green { color: #27500A; font-weight: 500; }

.sug-card {
    background: #FFFBF5;
    border: 0.5px solid #FAC775;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.sug-title { font-size: 13px; font-weight: 500; color: #1A1A2E; margin-bottom: 3px; }
.sug-text  { font-size: 12px; color: #666; line-height: 1.5; }

.dist-bar {
    height: 20px;
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    margin: 8px 0;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 8px;
    border: 0.5px solid #EBEBEB;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 500;
    color: #666;
}
.stTabs [aria-selected="true"] {
    background: #1A1A2E !important;
    color: white !important;
    border-color: #1A1A2E !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Constants ───────────────────────────────────────────────────────────────────
FEATURES = [
    "limit",
    "gender",
    "edu",
    "marital_status",
    "age",
    "rep_status_mth_6",
    "rep_status_mth_5",
    "rep_status_mth_4",
    "rep_status_mth_3",
    "rep_status_mth_2",
    "rep_status_mth_1",
    "bill_amt_mth_6",
    "bill_amt_mth_5",
    "bill_amt_mth_4",
    "bill_amt_mth_3",
    "bill_amt_mth_2",
    "bill_amt_mth_1",
    "pmt_amt_mth_6",
    "pmt_amt_mth_5",
    "pmt_amt_mth_4",
    "pmt_amt_mth_3",
    "pmt_amt_mth_2",
    "pmt_amt_mth_1",
    "pay_streak",
    "avg_payment_delay",
    "utilization_rate",
    "pmt_ratio",
    "bill_trend",
    "avg_utilization",
    "worst_pmt_delay",
]

EDU_MAP = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other/Unknown"}
GENDER_MAP = {1: "Male", 2: "Female"}
MARITAL_MAP = {1: "Married", 2: "Single", 3: "Other"}
PAY_MAP = {
    -2: "No consumption",
    -1: "Paid in full",
    0: "On time",
    1: "1 month late",
    2: "2 months late",
    3: "3 months late",
    4: "4 months late",
    5: "5 months late",
    6: "6 months late",
    7: "7 months late",
    8: "8+ months late",
}


# ── Loaders ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_api_client():
    return CreditRiskApiClient(os.getenv("CREDIT_CARD_API_URL"))


api_client = load_api_client()
try:
    api_health = api_client.health()
    model_info = api_client.model_info()
    if not api_health.get("model_loaded"):
        raise CreditRiskApiError("The prediction API has no model package loaded.")
except CreditRiskApiError as exc:
    st.error(f"Prediction API unavailable: {exc}")
    st.stop()

feature_engineering = FeatureEngineering()
explainer = None
thresholds = {
    "lower_threshold": model_info["lower_threshold"],
    "upper_threshold": model_info["upper_threshold"],
}
LOWER_T = thresholds["lower_threshold"]
UPPER_T = thresholds["upper_threshold"]


@st.cache_data
def load_and_score_dashboard_dataset():
    """Load the raw local test dataset and score it through FastAPI."""
    raw_data = pd.read_csv("./data/test_only/test_only.csv")
    transformed_data = feature_engineering.transform(raw_data)
    predictions = api_client.predict_dataframe(raw_data)
    transformed_data["probability"] = [
        prediction["probability"] for prediction in predictions
    ]
    transformed_data["decision"] = [
        prediction["decision"] for prediction in predictions
    ]
    return transformed_data


try:
    scored_df = load_and_score_dashboard_dataset()
except (CreditRiskApiError, FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Could not load the dashboard dataset: {exc}")
    st.stop()


# ── Helpers ──────────────────────────────────────────────────────────────────────
def get_decision(prob):
    if prob < LOWER_T:
        return "APPROVE"
    elif prob >= UPPER_T:
        return "REJECT"
    return "REVIEW"


def get_suggestions(row):
    suggestions = []
    pay0 = row.get("rep_status_mth_6", 0)
    streak = row.get("pay_streak", 0)
    worst = row.get("worst_pmt_delay", 0)
    avg_del = row.get("avg_payment_delay", 0)
    util = row.get("utilization_rate", 0)
    pmt_r = row.get("pmt_ratio", 1)
    limit = row.get("limit", 0)

    if pay0 >= 2 and streak >= 3:
        suggestions.append(
            {
                "title": f"Critical: {int(streak)} consecutive late payments",
                "text": f"Your most recent payment was {int(pay0)} months late with "
                f"{int(streak)} consecutive late months. This is the strongest "
                f"default signal lenders use. Bringing your account current and "
                f"maintaining on-time payments for 3 months is your highest priority.",
            }
        )
    elif pay0 >= 2:
        suggestions.append(
            {
                "title": "Most recent payment was 2+ months late",
                "text": "This single factor carries the most weight in your risk score. "
                "Making a payment immediately signals intent and reduces your score meaningfully.",
            }
        )
    elif streak >= 2:
        suggestions.append(
            {
                "title": f"{int(streak)} consecutive late months detected",
                "text": "Paying on time next month is urgent. Setting up automatic "
                "minimum payments removes this risk entirely.",
            }
        )
    elif pay0 == 1:
        suggestions.append(
            {
                "title": "Most recent payment was 1 month late",
                "text": "One late payment has an outsized effect on your score. "
                "Getting current before the next billing cycle prevents compounding.",
            }
        )

    if worst >= 3 and pay0 <= 0:
        suggestions.append(
            {
                "title": "Historical delays still affecting your score",
                "text": f"Your worst delay was {int(worst)} months but recent payments improved. "
                f"Maintain on-time payments for 3 more months to counteract the history.",
            }
        )

    if avg_del >= 1.5:
        suggestions.append(
            {
                "title": f"Average delay of {avg_del:.1f} months across 6 months",
                "text": "Sustained lateness signals financial stress. Automatic minimum "
                "payments would immediately begin improving this average.",
            }
        )

    if util >= 0.80:
        target = limit * 0.30
        suggestions.append(
            {
                "title": f"Credit utilisation at {util:.0%} of limit",
                "text": f"Reduce balance below NT${target:,.0f} (30% of NT${limit:,.0f} limit). "
                f"Even a partial reduction helps significantly.",
            }
        )
    elif util >= 0.50:
        suggestions.append(
            {
                "title": f"Utilisation at {util:.0%} — above recommended threshold",
                "text": "Lenders prefer below 30%. An extra payment this month would help.",
            }
        )

    if pmt_r < 0.10:
        suggestions.append(
            {
                "title": f"Paying only {pmt_r:.0%} of your bill",
                "text": "Consistently paying above 25% — even if not the full amount — "
                "strongly improves your risk profile.",
            }
        )
    elif pmt_r < 0.25:
        suggestions.append(
            {
                "title": f"Payment ratio at {pmt_r:.0%} — below recommended level",
                "text": "Aim to pay at least 25% of your balance each month.",
            }
        )

    return suggestions[:3]


def plot_shap_local(shap_vals, feature_names, n=8):
    series = pd.Series(shap_vals, index=feature_names)
    top = series.abs().nlargest(n).index
    vals = series[top]
    colors = ["#E24B4A" if v > 0 else "#378ADD" for v in vals.values]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.barh(range(len(top)), vals.values, color=colors, height=0.55)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(vals.index, fontsize=9)
    ax.axvline(0, color="#EBEBEB", linewidth=1)
    ax.set_xlabel("SHAP value", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#EBEBEB")
    ax.tick_params(colors="#888", labelsize=8)
    ax.legend(
        handles=[
            mpatches.Patch(color="#E24B4A", label="Increases risk"),
            mpatches.Patch(color="#378ADD", label="Decreases risk"),
        ],
        fontsize=8,
        frameon=False,
    )
    plt.tight_layout()
    return fig


def plot_global_shap(df, n=10):
    feat_cols = [c for c in FEATURES if c in df.columns]
    if not feat_cols:
        return None
    sample = df[feat_cols].sample(min(500, len(df)), random_state=42)
    try:
        # x_proc = prep.transform(sample) if hasattr(prep, 'transform') else sample.values
        # x_proc = sample.values
        if explainer is None:
            return None
        sv = explainer.shap_values(sample)
        # sv_use = sv[1] if isinstance(sv, list) else sv
        sv_use = sv[:, :, 1]
        mean_abs = pd.Series(np.abs(sv_use).mean(axis=0), index=feat_cols).sort_values()
        top = mean_abs.tail(n)
        colors = [
            "#1A1A2E" if i >= len(top) - 3 else "#EBEBEB" for i in range(len(top))
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.barh(top.index, top.values, color=colors, height=0.6)
        ax.set_xlabel("Mean |SHAP value|", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#EBEBEB")
        ax.tick_params(colors="#888", labelsize=8)
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ── Topbar ───────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-title">CreditIQ</div>
    <div class="topbar-sub">Credit risk intelligence platform</div>
  </div>
  <div class="topbar-right">
    <span class="threshold-text">Thresholds: {LOWER_T} / {UPPER_T}</span>
    &nbsp;&nbsp;
    <span class="status-dot"></span>
    <span class="status-text">Model active</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋  Batch Scoring", "📊  Portfolio Dashboard"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — BATCH SCORING
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        "<div class='section-title'>Batch scoring</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='section-sub'>Upload a customer CSV or use the pre-scored test set. Select a row index to view the full risk breakdown.</div>",
        unsafe_allow_html=True,
    )

    source = st.radio(
        "Source",
        ["Use test dataset", "Upload new CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "Upload new CSV":
        uploaded = st.file_uploader(
            "Upload CSV", type="csv", label_visibility="collapsed"
        )
        if uploaded:
            raw = pd.read_csv(uploaded)
            try:
                predictions = api_client.predict_dataframe(raw)
                X_proc = feature_engineering.transform(raw)
                X_proc["probability"] = [
                    prediction["probability"] for prediction in predictions
                ]
                X_proc["decision"] = [
                    prediction["decision"] for prediction in predictions
                ]
                working_df = X_proc
            except (CreditRiskApiError, ValueError, KeyError) as exc:
                st.error(f"Could not score uploaded data: {exc}")
                st.stop()
        else:
            st.info("Upload a CSV file to score new customers.")
            st.stop()
    else:
        working_df = scored_df.copy()

    has_tgt = "target" in working_df.columns
    total = len(working_df)
    n_app = (working_df["decision"] == "APPROVE").sum()
    n_rev = (working_df["decision"] == "REVIEW").sum()
    n_rej = (working_df["decision"] == "REJECT").sum()

    # ── Metrics ──
    cols = st.columns(5)
    metrics = [
        ("Total customers", f"{total:,}", "in this batch", "#1A1A2E"),
        ("Auto-approve", f"{n_app:,}", f"prob < {LOWER_T}", "#27500A"),
        ("For review", f"{n_rev:,}", f"prob {LOWER_T}–{UPPER_T}", "#633806"),
        ("Auto-reject", f"{n_rej:,}", f"prob > {UPPER_T}", "#791F1F"),
        (
            "Actual default rate" if has_tgt else "Avg probability",
            f"{working_df['target'].mean():.1%}"
            if has_tgt
            else f"{working_df['probability'].mean():.1%}",
            "ground truth" if has_tgt else "across batch",
            "#1A1A2E",
        ),
    ]
    for col, (label, val, sub, color) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""<div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color}'>{val}</div>
                <div class='metric-sub'>{sub}</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filter + table ──
    fc, sc = st.columns([3, 1])
    with fc:
        dec_filter = st.multiselect(
            "Filter",
            ["APPROVE", "REVIEW", "REJECT"],
            default=["APPROVE", "REVIEW", "REJECT"],
            label_visibility="collapsed",
        )
    with sc:
        sort_asc = (
            st.selectbox(
                "Sort",
                ["Highest risk first", "Lowest risk first"],
                label_visibility="collapsed",
            )
            == "Lowest risk first"
        )

    display_df = working_df[working_df["decision"].isin(dec_filter)].copy()
    display_df = display_df.sort_values("probability", ascending=sort_asc).reset_index(
        drop=True
    )

    show_cols = [
        c
        for c in [
            "probability",
            "decision",
            "age",
            "limit",
            "rep_status_mth_6",
            "pay_streak",
            "utilization_rate",
            "pmt_ratio",
        ]
        if c in display_df.columns
    ]
    rename_map = {
        "probability": "Probability",
        "decision": "Decision",
        "age": "Age",
        "limit": "Credit limit (NT$)",
        "rep_status_mth_6": "Pay status (latest)",
        "pay_streak": "Pay streak",
        "utilization_rate": "Utilisation",
        "pmt_ratio": "Payment ratio",
    }
    tv = display_df[show_cols].rename(columns=rename_map).copy()
    tv["Probability"] = tv["Probability"].map(lambda x: f"{x:.1%}")
    if "Utilisation" in tv:
        tv["Utilisation"] = tv["Utilisation"].map(lambda x: f"{x:.1%}")
    if "Payment ratio" in tv:
        tv["Payment ratio"] = tv["Payment ratio"].map(lambda x: f"{x:.1%}")
    if "Pay status (latest)" in tv:
        tv["Pay status (latest)"] = tv["Pay status (latest)"].map(
            lambda x: PAY_MAP.get(int(x), str(x))
        )

    st.dataframe(tv, width="stretch", height=280, hide_index=True)
    st.caption(
        "Enter a row index below (starting at 0) to open the full customer breakdown."
    )

    # ── Row selector + drilldown ──
    if len(display_df) > 0:
        sel = st.number_input(
            "Customer row index",
            min_value=0,
            max_value=len(display_df) - 1,
            value=0,
            step=1,
        )
        customer = display_df.iloc[sel]

        st.markdown("---")
        st.markdown(
            "<div class='section-title'>Customer breakdown</div>",
            unsafe_allow_html=True,
        )

        prob = customer["probability"]
        decision = customer["decision"]
        sugs = get_suggestions(customer)

        dec_color = {"APPROVE": "#27500A", "REVIEW": "#633806", "REJECT": "#791F1F"}
        dec_bg = {"APPROVE": "#EAF3DE", "REVIEW": "#FAEEDA", "REJECT": "#FCEBEB"}

        left, right = st.columns(2)

        with left:
            st.markdown(
                f"""
            <div class='metric-card' style='margin-bottom:12px'>
                <div class='metric-label'>Default probability</div>
                <div style='font-size:36px;font-weight:500;color:{dec_color[decision]};line-height:1.1'>{prob:.1%}</div>
                <div style='height:8px;background:#F0F0F0;border-radius:4px;overflow:hidden;margin:10px 0 6px'>
                    <div style='width:{prob * 100:.1f}%;height:8px;background:{dec_color[decision]};border-radius:4px'></div>
                </div>
                <div style='display:flex;justify-content:space-between;font-size:10px;color:#aaa;margin-bottom:10px'>
                    <span>0%</span>
                    <span>{LOWER_T:.0%} — approve/review</span>
                    <span>{UPPER_T:.0%} — review/reject</span>
                </div>
                <span style='background:{dec_bg[decision]};color:{dec_color[decision]};
                      padding:4px 14px;border-radius:20px;font-size:12px;font-weight:500'>
                    {decision}
                </span>
            </div>""",
                unsafe_allow_html=True,
            )

            feat_rows = [
                ("Credit limit", f"NT${customer.get('limit', 0):,.0f}", ""),
                ("Age", f"{int(customer.get('age', 0))}", ""),
                ("Education", EDU_MAP.get(int(customer.get("edu", 4)), "?"), ""),
                (
                    "Pay status (latest)",
                    PAY_MAP.get(int(customer.get("rep_status_mth_6", 0)), "?"),
                    "sig-red"
                    if customer.get("rep_status_mth_6", 0) >= 2
                    else "sig-amber"
                    if customer.get("rep_status_mth_6", 0) == 1
                    else "sig-green",
                ),
                (
                    "Pay streak",
                    f"{int(customer.get('pay_streak', 0))} months",
                    "sig-red"
                    if customer.get("pay_streak", 0) >= 3
                    else "sig-amber"
                    if customer.get("pay_streak", 0) >= 1
                    else "sig-green",
                ),
                (
                    "Avg payment delay",
                    f"{customer.get('avg_payment_delay', 0):.2f} months",
                    "sig-red"
                    if customer.get("avg_payment_delay", 0) >= 1.5
                    else "sig-amber"
                    if customer.get("avg_payment_delay", 0) >= 0.5
                    else "sig-green",
                ),
                (
                    "Worst delay",
                    f"{int(customer.get('worst_pmt_delay', 0))} months",
                    "sig-red"
                    if customer.get("worst_pmt_delay", 0) >= 3
                    else "sig-amber"
                    if customer.get("worst_pmt_delay", 0) >= 1
                    else "sig-green",
                ),
                (
                    "Utilisation rate",
                    f"{customer.get('utilization_rate', 0):.1%}",
                    "sig-red"
                    if customer.get("utilization_rate", 0) >= 0.8
                    else "sig-amber"
                    if customer.get("utilization_rate", 0) >= 0.5
                    else "sig-green",
                ),
                (
                    "Payment ratio",
                    f"{customer.get('pmt_ratio', 0):.1%}",
                    "sig-red"
                    if customer.get("pmt_ratio", 0) < 0.10
                    else "sig-amber"
                    if customer.get("pmt_ratio", 0) < 0.25
                    else "sig-green",
                ),
                (
                    "Bill trend",
                    f"NT${customer.get('bill_trend', 0):,.0f}",
                    "sig-red"
                    if customer.get("bill_trend", 0) > 5000
                    else "sig-amber"
                    if customer.get("bill_trend", 0) > 0
                    else "sig-green",
                ),
            ]
            rows_html = "".join(
                [
                    f"<div class='feature-row'><span style='color:#666'>{l}</span>"
                    f"<span class='{c}'>{v}</span></div>"
                    for l, v, c in feat_rows
                ]
            )
            st.markdown(
                f"<div style='background:white;border:0.5px solid #EBEBEB;"
                f"border-radius:12px;padding:12px 16px'>{rows_html}</div>",
                unsafe_allow_html=True,
            )

        with right:
            st.markdown("**What is driving this risk?**")
            st.caption("SHAP values — underlying Random Forest model")
            try:
                feat_cols = [c for c in FEATURES if c in customer.index]
                x_row = pd.DataFrame([customer[feat_cols]])
                # x_proc = prep.transform(x_row) if hasattr(prep, 'transform') else x_row.values
                if explainer is None:
                    raise RuntimeError("SHAP explanations are not provided by the API yet.")
                sv = explainer.shap_values(x_row)
                sv_use = sv[:, :, 1][0]
                # sv_use = sv[1][0] if isinstance(sv, list) else sv[0]

                fig_s = plot_shap_local(sv_use, feat_cols)
                st.pyplot(fig_s)
                plt.close()
            except Exception as e:
                st.info(f"SHAP unavailable: {e}")

            st.markdown("**How to improve this score**")
            if not sugs:
                st.success(
                    "No significant risk factors detected — this customer's profile looks healthy."
                )
            else:
                for s in sugs:
                    st.markdown(
                        f"<div class='sug-card'>"
                        f"<div class='sug-title'>{s['title']}</div>"
                        f"<div class='sug-text'>{s['text']}</div></div>",
                        unsafe_allow_html=True,
                    )

            if has_tgt:
                actual = int(customer["target"])
                correct = (actual == 1 and decision == "REJECT") or (
                    actual == 0 and decision == "APPROVE"
                )
                label = "Actually defaulted" if actual == 1 else "Did not default"
                color = "#791F1F" if actual == 1 else "#27500A"
                bg = "#FCEBEB" if actual == 1 else "#EAF3DE"
                st.markdown(
                    f"<div style='background:{bg};border-radius:10px;padding:10px 14px;margin-top:8px'>"
                    f"<span style='font-size:12px;font-weight:500;color:{color}'>Ground truth: {label}</span>"
                    f"<span style='font-size:11px;color:#888;margin-left:8px'>"
                    f"{'Model was correct' if correct else 'Model was incorrect'}</span></div>",
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORTFOLIO DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<div class='section-title'>Portfolio dashboard</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='section-sub'>Risk distribution, segment breakdowns, and live threshold sensitivity across the full scored dataset.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Threshold sensitivity** — move sliders to see how the portfolio splits"
    )
    sl1, sl2, _ = st.columns([1, 1, 2])
    with sl1:
        t_low = st.slider(
            "Lower threshold", 0.05, 0.50, float(LOWER_T), 0.01, format="%.2f"
        )
    with sl2:
        t_high = st.slider(
            "Upper threshold", 0.20, 0.95, float(UPPER_T), 0.01, format="%.2f"
        )

    probs_all = scored_df["probability"]
    live_app = int((probs_all < t_low).sum())
    live_rev = int(((probs_all >= t_low) & (probs_all < t_high)).sum())
    live_rej = int((probs_all >= t_high).sum())
    n_tot = len(scored_df)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, sub, color in zip(
        [m1, m2, m3, m4],
        ["Portfolio size", "Auto-approve", "For review", "Auto-reject"],
        [f"{n_tot:,}", f"{live_app:,}", f"{live_rev:,}", f"{live_rej:,}"],
        [
            "scored customers",
            f"{live_app / n_tot:.1%} of portfolio",
            f"{live_rev / n_tot:.1%} of portfolio",
            f"{live_rej / n_tot:.1%} of portfolio",
        ],
        ["#1A1A2E", "#27500A", "#633806", "#791F1F"],
    ):
        with col:
            st.markdown(
                f"""<div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color}'>{val}</div>
                <div class='metric-sub'>{sub}</div></div>""",
                unsafe_allow_html=True,
            )

    ap = live_app / n_tot * 100
    rp = live_rev / n_tot * 100
    rjp = live_rej / n_tot * 100
    st.markdown(
        f"""<div class='dist-bar'>
        <div style='width:{ap:.1f}%;background:#639922;display:flex;align-items:center;
             justify-content:center;font-size:10px;color:#EAF3DE;font-weight:500'>{ap:.0f}%</div>
        <div style='width:{rp:.1f}%;background:#BA7517;display:flex;align-items:center;
             justify-content:center;font-size:10px;color:#FAEEDA;font-weight:500'>{rp:.0f}%</div>
        <div style='width:{rjp:.1f}%;background:#E24B4A;display:flex;align-items:center;
             justify-content:center;font-size:10px;color:#FCEBEB;font-weight:500'>{rjp:.0f}%</div>
    </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Probability distribution**")
        bins = np.arange(0, 1.05, 0.1)
        counts, edges = np.histogram(scored_df["probability"], bins=bins)
        bcolors = [
            "#639922"
            if (e + 0.05) < t_low
            else "#BA7517"
            if (e + 0.05) < t_high
            else "#E24B4A"
            for e in edges[:-1]
        ]
        fig_h, ax_h = plt.subplots(figsize=(6, 3))
        fig_h.patch.set_facecolor("white")
        ax_h.set_facecolor("white")
        ax_h.bar(
            edges[:-1],
            counts,
            width=0.09,
            color=bcolors,
            align="edge",
            edgecolor="white",
            linewidth=0.5,
        )
        ax_h.axvline(t_low, color="#1A1A2E", linestyle="--", linewidth=1, alpha=0.6)
        ax_h.axvline(t_high, color="#1A1A2E", linestyle="--", linewidth=1, alpha=0.6)
        ax_h.set_xlabel("Default probability", fontsize=9)
        ax_h.set_ylabel("Customers", fontsize=9)
        ax_h.spines[["top", "right"]].set_visible(False)
        ax_h.spines[["left", "bottom"]].set_color("#EBEBEB")
        ax_h.tick_params(colors="#888", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close()

    with ch2:
        st.markdown("**Global feature importance (SHAP)**")
        fig_g = plot_global_shap(scored_df)
        if fig_g:
            st.pyplot(fig_g)
            plt.close()
        else:
            st.info("Feature columns not found in scored dataset for SHAP.")

    # ── Segment breakdowns ──
    has_tgt2 = "target" in scored_df.columns
    if has_tgt2:
        st.markdown("<br>", unsafe_allow_html=True)
        seg1, seg2 = st.columns(2)

        with seg1:
            if "edu" in scored_df.columns:
                st.markdown("**Default rate by education level**")
                edu_dr = scored_df.groupby("edu")["target"].mean().reset_index()
                edu_dr["label"] = edu_dr["edu"].map(EDU_MAP)
                edu_dr = edu_dr.sort_values("target")
                fig_e, ax_e = plt.subplots(figsize=(5, 2.8))
                fig_e.patch.set_facecolor("white")
                ax_e.set_facecolor("white")
                bc = [
                    "#E24B4A" if v > 0.25 else "#BA7517" if v > 0.18 else "#639922"
                    for v in edu_dr["target"]
                ]
                bars = ax_e.barh(
                    edu_dr["label"], edu_dr["target"] * 100, color=bc, height=0.5
                )
                for bar, val in zip(bars, edu_dr["target"]):
                    ax_e.text(
                        bar.get_width() + 0.3,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1%}",
                        va="center",
                        fontsize=9,
                        color="#555",
                    )
                ax_e.set_xlabel("Default rate (%)", fontsize=9)
                ax_e.spines[["top", "right", "left"]].set_visible(False)
                ax_e.spines["bottom"].set_color("#EBEBEB")
                ax_e.tick_params(colors="#888", labelsize=9)
                plt.tight_layout()
                st.pyplot(fig_e)
                plt.close()

        with seg2:
            if "limit" in scored_df.columns:
                st.markdown("**Default rate by credit limit tier**")
                scored_df["limit_tier"] = pd.cut(
                    scored_df["limit"],
                    bins=[0, 50000, 200000, 1000000],
                    labels=["NT$0–50k", "NT$50k–200k", "NT$200k+"],
                )
                tier_dr = scored_df.groupby("limit_tier", observed=True)[
                    "target"
                ].mean()
                fig_t, ax_t = plt.subplots(figsize=(5, 2.8))
                fig_t.patch.set_facecolor("white")
                ax_t.set_facecolor("white")
                bc2 = [
                    "#E24B4A" if v > 0.25 else "#BA7517" if v > 0.18 else "#639922"
                    for v in tier_dr.values
                ]
                bars2 = ax_t.bar(
                    tier_dr.index.astype(str),
                    tier_dr.values * 100,
                    color=bc2,
                    width=0.5,
                )
                for bar, val in zip(bars2, tier_dr.values):
                    ax_t.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{val:.1%}",
                        ha="center",
                        fontsize=9,
                        color="#555",
                    )
                ax_t.set_ylabel("Default rate (%)", fontsize=9)
                ax_t.spines[["top", "right", "left"]].set_visible(False)
                ax_t.spines["bottom"].set_color("#EBEBEB")
                ax_t.tick_params(colors="#888", labelsize=9)
                plt.tight_layout()
                st.pyplot(fig_t)
                plt.close()

    # ── Model performance ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Model bake-off — performance summary**")
    st.dataframe(
        pd.DataFrame(
            {
                "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
                "ROC-AUC": ["0.763", "0.785", "0.784"],
                "Recall": ["0.615", "0.621", "0.607"],
                "Precision": ["0.457", "0.472", "0.471"],
                "F1": ["0.524", "0.536", "0.530"],
                "Selected": ["", "Yes", ""],
            }
        ).set_index("Model"),
        width="stretch",
    )
    st.caption(
        "Winner: Random Forest with engineered features — chosen on recall. "
        "Probabilities are calibrated. SHAP values reflect the underlying "
        "uncalibrated Random Forest, which is standard practice."
    )
