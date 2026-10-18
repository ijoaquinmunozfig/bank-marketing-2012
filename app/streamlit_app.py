import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "bank_marketing_rf.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
SCORES_PATH = ROOT / "models" / "test_scores.csv"
IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
CAT_COLS = ["job", "marital", "education", "poutcome", "month"]
NAVY = "#1E3A8A"
FEATURE_LABELS = {
    "age": "Age (years)",
    "job": "Job",
    "marital": "Marital status",
    "education": "Education",
    "default": "Credit in default",
    "balance": "Average yearly balance (EUR)",
    "housing": "Housing loan",
    "loan": "Personal loan",
    "contact_bool": "Last contact channel",
    "day_of_week": "Day of month of last contact",
    "month": "Month of last contact",
    "campaign": "Contacts during this campaign",
    "pdays": "Days since previous contact",
    "previous": "Contacts before this campaign",
    "poutcome": "Previous campaign outcome",
}

YES_NO = {True: "Yes", False: "No"}

VALUE_LABELS = {
    "job": {
        "admin.": "Administrative", "blue-collar": "Blue-collar", "entrepreneur": "Entrepreneur",
        "housemaid": "Housemaid", "management": "Management", "retired": "Retired",
        "self-employed": "Self-employed", "services": "Services", "student": "Student",
        "technician": "Technician", "unemployed": "Unemployed", "missing": "Not reported",
    },
    "marital": {"divorced": "Divorced", "married": "Married", "single": "Single"},
    "education": {"primary": "Primary", "secondary": "Secondary", "tertiary": "Tertiary", "missing": "Not reported"},
    "poutcome": {"failure": "Failure", "other": "Other", "success": "Success", "missing": "Not available"},
    "month": {
        "jan": "January", "feb": "February", "mar": "March", "apr": "April", "may": "May", "jun": "June",
        "jul": "July", "aug": "August", "sep": "September", "oct": "October", "nov": "November", "dec": "December",
    },
    "default": YES_NO,
    "housing": YES_NO,
    "loan": YES_NO,
    "contact_bool": {True: "Cellular", False: "Other"},
}


def feature_label(column):
    return FEATURE_LABELS.get(column, column.replace("_", " ").capitalize())


def value_label(column, value):
    text = str(value).replace("_", " ")
    return VALUE_LABELS.get(column, {}).get(value, text[:1].upper() + text[1:])


def friendly_frame(frame):
    out = frame.copy()
    for column in out.columns:
        if column in VALUE_LABELS:
            out[column] = out[column].map(lambda v, c=column: value_label(c, v))
    return out.rename(columns=feature_label)

st.set_page_config(page_title="Bank Marketing | Term Deposit Propensity", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_scores():
    return pd.read_csv(SCORES_PATH)


@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH)


@st.cache_data(show_spinner="Loading data from the UCI repository...")
def load_data():
    from ucimlrepo import fetch_ucirepo

    bank = fetch_ucirepo(id=222)
    data = pd.concat([bank.data.features, bank.data.targets], axis=1)
    data = data.assign(contact_bool=(data["contact"] == "cellular"))
    data = data.drop(columns=["contact", "duration"])
    for col in CAT_COLS:
        data[col] = data[col].fillna("missing")
    for col in ["default", "housing", "loan", "y"]:
        data[col] = data[col].eq("yes")
    return data


model = load_model()
meta = load_metadata()
FEATURES = meta["features"]
OPTIONS = meta["categorical_options"]
RANGES = meta["numeric_ranges"]


def int_range(name):
    r = RANGES[name]
    return int(r["min"]), int(r["max"]), int(r["median"])


def yes_no(column):
    return st.radio(feature_label(column), ["No", "Yes"], horizontal=True) == "Yes"


def pick(column, options):
    return st.selectbox(feature_label(column), options, format_func=lambda v: value_label(column, v))


with st.sidebar:
    st.header("Client profile")
    lo, hi, mid = int_range("age")
    age = st.slider(feature_label("age"), lo, hi, mid)
    job = pick("job", OPTIONS["job"])
    marital = pick("marital", OPTIONS["marital"])
    education = pick("education", OPTIONS["education"])
    default = yes_no("default")
    lo, hi, mid = int_range("balance")
    balance = st.number_input(feature_label("balance"), lo, hi, mid, step=100)
    housing = yes_no("housing")
    loan = yes_no("loan")
    contact_bool = st.radio(feature_label("contact_bool"), ["Cellular", "Other"], horizontal=True) == "Cellular"
    months = [m for m in MONTH_ORDER if m in OPTIONS["month"]]
    month = pick("month", months)
    lo, hi, mid = int_range("day_of_week")
    day = st.slider(feature_label("day_of_week"), lo, hi, mid)
    lo, hi, mid = int_range("campaign")
    campaign = st.number_input(feature_label("campaign"), lo, hi, mid)
    lo, hi, mid = int_range("pdays")
    pdays = st.number_input(feature_label("pdays"), lo, hi, mid, help="-1 means the client was not contacted in a previous campaign.")
    lo, hi, mid = int_range("previous")
    previous = st.number_input(feature_label("previous"), lo, hi, mid)
    poutcome = pick("poutcome", OPTIONS["poutcome"])
    st.divider()
    threshold = st.slider("Decision threshold", 0.05, 0.95, float(meta["default_threshold"]), 0.01)

client = pd.DataFrame([{
    "age": age, "job": job, "marital": marital, "education": education,
    "default": default, "balance": balance, "housing": housing, "loan": loan,
    "day_of_week": day, "month": month, "campaign": campaign, "pdays": pdays,
    "previous": previous, "poutcome": poutcome, "contact_bool": contact_bool,
}])[FEATURES]

st.title("Term Deposit Propensity Model")
st.caption(
    "Educational portfolio project built on historical campaign data from a Portuguese bank. "
    "It is not a tool for real customer decisions."
)

tab_pred, tab_model, tab_data = st.tabs(["Prediction", "Model performance", "Data exploration"])

with tab_pred:
    proba = float(model.predict_proba(client)[0, 1])
    base_rate = meta["dataset"]["train_positive_rate"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Probability of subscription", f"{proba:.1%}")
    col2.metric("Average in training data", f"{base_rate:.1%}")
    col3.metric("Relative to average", f"{proba / base_rate:.1f}x")

    if proba >= threshold:
        st.success(f"Above the {threshold:.0%} threshold: this profile would be prioritized for contact.")
    else:
        st.info(f"Below the {threshold:.0%} threshold: this profile would not be prioritized.")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": NAVY},
            "threshold": {"line": {"color": "#EF4444", "width": 3}, "value": threshold * 100},
        },
    ))
    gauge.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10))
    st.plotly_chart(gauge)

    with st.expander("Input features sent to the model"):
        st.dataframe(friendly_frame(client), hide_index=True)

with tab_model:
    scores = load_scores()
    y_true = scores["y_true"]
    y_prob = scores["y_prob"]
    y_hat = (y_prob >= threshold).astype(int)

    st.subheader("Performance on the held-out test set")
    st.caption(
        f"{len(scores):,} customers not used for training. "
        "The metrics change with the decision threshold in the sidebar."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision_score(y_true, y_hat, zero_division=0):.1%}")
    c2.metric("Recall", f"{recall_score(y_true, y_hat, zero_division=0):.1%}")
    c3.metric("F1 score", f"{f1_score(y_true, y_hat, zero_division=0):.1%}")
    c4.metric("ROC-AUC", f"{roc_auc_score(y_true, y_prob):.3f}")

    left, right = st.columns(2)

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    cm = px.imshow(
        [[tn, fp], [fn, tp]],
        x=["Predicted: no", "Predicted: yes"],
        y=["Actual: no", "Actual: yes"],
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion matrix",
    )
    cm.update_coloraxes(showscale=False)
    left.plotly_chart(cm)

    grid = [i / 100 for i in range(5, 96, 5)]
    curve = pd.DataFrame({
        "threshold": grid,
        "Precision": [precision_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid],
        "Recall": [recall_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid],
    })
    trade = px.line(
    curve, x="threshold", y=["Precision", "Recall"], markers=True,
    title="Precision and recall by threshold",
    labels={"threshold": "Decision threshold", "value": "Score", "variable": "Metric"})
    trade.add_vline(x=threshold, line_dash="dash")
    right.plotly_chart(trade)

    importance = load_importance().head(10).assign(feature=lambda d: d["feature"].map(feature_label))
    fig_imp = px.bar(
    importance, x="importance", y="feature", orientation="h",
    title="Top 10 variables by importance",
    labels={"importance": "Importance", "feature": "Variable"})
    fig_imp.update_traces(marker_color=NAVY)
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp)
    st.caption(
        "Impurity-based importance shows what the forest uses to split the data, not a causal effect. "
        "Continuous variables such as balance and age tend to rank higher."
    )

    with st.expander("About the model"):
        st.write(f"Model: {meta['model_name']}")
        st.write(f"Trained on: {meta['trained_on']}")
        st.write(f"Excluded to avoid data leakage: {', '.join(meta['excluded_features'])}")
        st.write(f"Data: {meta['dataset']['source']} ({meta['dataset']['license']})")

with tab_data:
    try:
        data = load_data()
    except Exception:
        data = None
        st.warning(
            "The dataset could not be downloaded from the UCI repository right now. "
            "The prediction and model tabs still work."
        )

    if data is not None:
        with st.expander("Data"):
            st.write(f"{data.shape[0]:,} rows and {data.shape[1]} columns")
            st.write("Variables")
            st.dataframe(friendly_frame(data.drop(columns="y").head(100)), hide_index=True)
            st.write("Target")
            st.write(f"y = subscribed a term deposit ({data['y'].mean():.1%} of customers)")

        by = st.selectbox(
            "Subscription rate by",
            CAT_COLS + ["housing", "loan", "default", "contact_bool"],
            format_func=feature_label,
        )
        rate = data.groupby(by)["y"].agg(rate="mean", n="count").reset_index().sort_values("rate")
        rate["rate"] = rate["rate"] * 100
        rate[by] = rate[by].map(lambda v: value_label(by, v))
        fig_rate = px.bar(rate, x="rate", y=by, orientation="h", hover_data=["n"])
        fig_rate.update_traces(marker_color=NAVY)
        fig_rate.add_vline(x=data["y"].mean() * 100, line_dash="dash")
        fig_rate.update_layout(xaxis_title="Subscription rate (%)", yaxis_title=feature_label(by))
        st.plotly_chart(fig_rate)

        numeric = ["age", "balance", "campaign", "pdays", "previous", "day_of_week"]
        sx, sy = st.columns(2)
        x_col = sx.selectbox("X axis", numeric, index=0, format_func=feature_label)
        y_col = sy.selectbox("Y axis", numeric, index=1, format_func=feature_label)
        sample = data.sample(min(3000, len(data)), random_state=42).assign(
            outcome=lambda d: d["y"].map({True: "Subscribed", False: "Did not subscribe"})
        )
        fig_scatter = px.scatter(
            sample, x=x_col, y=y_col, color="outcome", opacity=0.6,
            labels={x_col: feature_label(x_col), y_col: feature_label(y_col), "outcome": "Outcome"},
        )
        st.plotly_chart(fig_scatter)

st.divider()
st.caption("Data: UCI Machine Learning Repository, Bank Marketing dataset (CC BY 4.0).")
