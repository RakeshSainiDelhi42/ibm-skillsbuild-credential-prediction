import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="IBM SkillsBuild Credential Prediction",
    layout="wide",
)

MODEL_PATH = "models/best_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
FEATURES_PATH = "models/feature_names.pkl"
CATEGORICAL = ["learner_type", "learning_source", "delivery_type", "state", "age"]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, encoders, feature_names


def read_csvs(uploaded_files):
    frames = [pd.read_csv(f, low_memory=False) for f in uploaded_files]
    return pd.concat(frames, ignore_index=True)


def engineer_dates(trans):
    reg = pd.to_datetime(
        trans.get("User Registration Date"), format="%d-%m-%Y",
        errors="coerce", utc=True,
    )
    trans["_registration_date"] = reg

    if "time_on_platform_days" not in trans.columns:
        last_access = pd.to_datetime(
            trans.get("Learning Last Accessed Date"), format="ISO8601",
            errors="coerce", utc=True,
        )
        completion = pd.to_datetime(
            trans.get("Completion Date"), format="ISO8601",
            errors="coerce", utc=True,
        )
        last_activity = last_access.fillna(completion)
        trans["time_on_platform_days"] = (last_activity - reg).dt.days
        trans.loc[trans["time_on_platform_days"] < 0, "time_on_platform_days"] = np.nan

    trans["time_on_platform_days"] = pd.to_numeric(
        trans["time_on_platform_days"], errors="coerce")
    return trans


def build_learner_table(trans):
    keep = {}
    if "Learner - Name" in trans.columns:
        keep["learner_name"] = ("Learner - Name", "first")

    learner = trans.groupby("Learner - ID").agg(
        total_courses=("Learning activity - ID", "nunique"),
        time_on_platform=("time_on_platform_days", "max"),
        learner_type=("Learner - Type", "first"),
        learning_source=("Learning Source Name", "first"),
        delivery_type=("Delivery Type", "first"),
        state=("State", "first"),
        age=("Age At Registration", "first"),
        registration_date=("_registration_date", "first"),
        **keep,
    ).reset_index()

    learner["state"] = learner["state"].astype(str).str.replace(" - IN", "", regex=False)
    for col in CATEGORICAL:
        learner[col] = learner[col].replace("Not Available", "Unknown").fillna("Unknown")

    med = learner["time_on_platform"].median() if learner["time_on_platform"].notna().any() else 0
    learner["time_on_platform"] = learner["time_on_platform"].fillna(med)
    return learner


def score_learners(learner, model, encoders, feature_names):
    X = learner.copy()
    for col, le in encoders.items():
        known = set(le.classes_)
        X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
        X[col] = le.transform(X[col].astype(str))

    proba = model.predict_proba(X[feature_names])[:, 1]
    learner["credential_probability"] = (proba * 100).round(1)
    learner["outreach_priority_score"] = (100 - learner["credential_probability"]).round(1)
    learner["priority_category"] = pd.cut(
        learner["outreach_priority_score"],
        bins=[0, 33, 66, 100],
        labels=["Low Priority", "Medium Priority", "High Priority"],
        include_lowest=True,
    )
    return learner


def build_output(learner):
    out = learner.copy()
    out["registration_date"] = pd.to_datetime(
        out["registration_date"], errors="coerce", utc=True).dt.tz_localize(None).dt.date

    rename = {"Learner - ID": "Learner ID"}
    if "learner_name" in out.columns:
        rename["learner_name"] = "Learner Name"
    rename.update({
        "registration_date": "Registration Date",
        "outreach_priority_score": "Outreach Priority Score",
        "credential_probability": "Credential Probability (%)",
        "priority_category": "Priority",
        "total_courses": "Total Courses",
        "time_on_platform": "Days on Platform",
        "learner_type": "Learner Type",
        "learning_source": "Learning Source",
        "delivery_type": "Delivery Type",
        "state": "State",
    })
    out = out.rename(columns=rename)
    out = out[[c for c in rename.values() if c in out.columns]]
    out = out.sort_values("Outreach Priority Score", ascending=False).reset_index(drop=True)
    out.index = out.index + 1
    out.index.name = "Rank"
    return out


def to_excel_bytes(out):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="Priority List", index=True)
        high = out[out["Priority"] == "High Priority"]
        high.to_excel(writer, sheet_name="High Priority", index=True)
        summary = pd.DataFrame({
            "Metric": ["Report Generated", "Total Learners", "High Priority",
                       "Medium Priority", "Low Priority"],
            "Value": [datetime.today().strftime("%Y-%m-%d"), len(out),
                      int((out["Priority"] == "High Priority").sum()),
                      int((out["Priority"] == "Medium Priority").sum()),
                      int((out["Priority"] == "Low Priority").sum())],
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)
    buffer.seek(0)
    return buffer.getvalue()


with st.sidebar:
    st.header("Credential Prediction")
    st.caption("Built for Learning Links Foundation, an IBM SkillsBuild implementation partner.")
    st.subheader("How it works")
    st.write(
        "Upload one or more monthly learning-transcript reports. The app rebuilds "
        "the features used in training, scores every learner, and ranks them by "
        "outreach priority. Download the result as Excel."
    )
    st.subheader("Privacy")
    st.info("Files are processed in memory only. Nothing is stored or logged.")
    st.caption("Model: XGBoost. F1 0.89, AUC 0.98, 5-fold CV 0.93.")


st.title("IBM SkillsBuild Learner Outreach Priority")
st.write(
    "Predict which learners are least likely to earn a digital credential, so trainers "
    "can focus outreach where it matters most. The priority list includes each learner's "
    "registration date, so you can filter to a specific session."
)

try:
    model, encoders, feature_names = load_artifacts()
except Exception:
    st.error("Model files were not found in models/. Expected best_model.pkl, encoders.pkl, feature_names.pkl.")
    st.stop()

st.subheader("1. Upload transcript report(s)")
uploaded = st.file_uploader(
    "Learning-transcript CSV files",
    type=["csv"], accept_multiple_files=True,
)
st.caption("No file handy? A synthetic sample is included in sample_data/ so you can try the app.")

if not uploaded:
    st.info("Upload at least one transcript CSV to generate the priority list.")
    st.stop()

try:
    trans = read_csvs(uploaded)
    if "Learner - ID" not in trans.columns:
        st.error("These files do not look like transcript reports. No Learner - ID column found.")
        st.stop()

    trans = engineer_dates(trans)
    learner = build_learner_table(trans)
    learner = score_learners(learner, model, encoders, feature_names)
    out = build_output(learner)

    st.subheader("2. Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Learners scored", f"{len(out):,}")
    c2.metric("High priority", f"{int((out['Priority']=='High Priority').sum()):,}")
    c3.metric("Medium priority", f"{int((out['Priority']=='Medium Priority').sum()):,}")
    c4.metric("Low priority", f"{int((out['Priority']=='Low Priority').sum()):,}")

    st.write("Filter the list")
    f1, f2 = st.columns(2)
    with f1:
        bands = st.multiselect(
            "Priority band", ["High Priority", "Medium Priority", "Low Priority"],
            default=["High Priority"],
        )
    with f2:
        valid_dates = out["Registration Date"].dropna() if "Registration Date" in out.columns else pd.Series([], dtype=object)
        if len(valid_dates):
            dmin, dmax = valid_dates.min(), valid_dates.max()
            date_range = st.date_input(
                "Registration date range", value=(dmin, dmax),
                min_value=dmin, max_value=dmax,
            )
        else:
            date_range = None

    view = out.copy()
    if bands:
        view = view[view["Priority"].isin(bands)]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and "Registration Date" in view.columns:
        lo, hi = date_range
        view = view[view["Registration Date"].apply(
            lambda d: d is not None and not pd.isna(d) and lo <= d <= hi)]

    st.write(f"Showing {len(view):,} learners")
    st.dataframe(view, use_container_width=True, height=420)

    st.subheader("3. Download")
    st.download_button(
        "Download full priority list (Excel)",
        data=to_excel_bytes(out),
        file_name=f"priority_list_{datetime.today().strftime('%Y_%m_%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Three sheets: full Priority List, High Priority only, and Summary. "
               "Registration Date is a real date column, so Excel date filters work directly.")

except Exception as e:
    st.error(f"Something went wrong while processing the files: {e}")