"""
Page 1 – Data Upload
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import validate_data, fill_missing_values
from config import FEATURE_COLUMNS, FAULT_LABELS, FAULT_COLORS

st.set_page_config(page_title="Data Upload", page_icon="📤", layout="wide")

st.markdown("""
<style>
  .stApp { background-color: #0f0f1a; color: #e0e0ff; }
  section[data-testid="stSidebar"] { background-color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

st.title("📤 Data Upload")
st.markdown("Upload a PV sensor CSV file or generate a synthetic dataset for exploration.")

# ── Source selector ───────────────────────────────────────────────────────────
source = st.radio("Data source", ["Upload CSV", "Generate synthetic data"], horizontal=True)

df: pd.DataFrame | None = None

if source == "Upload CSV":
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, parse_dates=["timestamp"],
                             infer_datetime_format=True)
            st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")
        except Exception as exc:
            st.error(f"Failed to parse file: {exc}")

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        n_days = st.number_input("Days to generate", 30, 730, 365, step=30)
    with col2:
        interval = st.selectbox("Sample interval (min)", [5, 15, 30, 60], index=1)
    with col3:
        seed = st.number_input("Random seed", 0, 9999, 42)

    if st.button("⚡ Generate data", type="primary"):
        with st.spinner("Generating synthetic PV data…"):
            from data.data_generator import generate_pv_data
            df = generate_pv_data(n_days=int(n_days), interval_min=int(interval),
                                   seed=int(seed))
        st.success(f"Generated {len(df):,} rows.")

# ── Preview & statistics ──────────────────────────────────────────────────────
if df is not None:
    is_valid, issues = validate_data(df, FEATURE_COLUMNS)
    if not is_valid:
        st.warning("Data issues detected – applying auto-fix:")
        for iss in issues:
            st.caption(f"• {iss}")
        df = fill_missing_values(df, [c for c in FEATURE_COLUMNS if c in df.columns])

    # Store in session state
    st.session_state["df"] = df

    tab1, tab2, tab3 = st.tabs(["🔍 Preview", "📈 Statistics", "🥧 Fault Distribution"])

    with tab1:
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(df):,} rows.")

    with tab2:
        desc = df[FEATURE_COLUMNS].describe().T
        st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)

    with tab3:
        if "fault_label" in df.columns:
            counts = df["fault_label"].value_counts().sort_index()
            labels = [FAULT_LABELS.get(i, str(i)) for i in counts.index]
            colors = [FAULT_COLORS.get(i, "#999") for i in counts.index]
            fig = px.pie(
                values=counts.values,
                names=labels,
                color_discrete_sequence=colors,
                title="Fault Label Distribution",
            )
            fig.update_layout(
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#0f0f1a",
                font=dict(color="#e0e0ff"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'fault_label' column found in data.")

    st.success("✅ Data stored in session. Proceed to **Causal Discovery** →")
