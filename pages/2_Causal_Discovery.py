"""
Page 2 – Causal Discovery
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from config import FEATURE_COLUMNS, KNOWN_CAUSAL_RELATIONS, MAX_LAG, ALPHA_LEVEL
from models.causal_discovery import PVCausalDiscovery
from utils.metrics import compute_causal_physics_validation
from utils.theme import apply_theme
from utils.visualization import plot_causal_graph

st.set_page_config(page_title="Causal Discovery", page_icon="🔍", layout="wide")

apply_theme()

st.title("🔍 Causal Discovery")
st.markdown(
    "Discover causal relationships between PV sensors using **PCMCI** "
    "(tigramite) or a lagged-correlation fallback."
)

df: pd.DataFrame | None = st.session_state.get("df")

if df is None:
    st.warning("⚠️ No data loaded. Go to **Data Upload** first.")
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Discovery Parameters")
    max_lag = st.slider("Max lag", 1, 8, MAX_LAG)
    alpha = st.number_input("Alpha level", 0.001, 0.2, ALPHA_LEVEL, step=0.005,
                             format="%.3f")
    sample_n = st.slider("Max rows for discovery", 500, min(len(df), 10_000),
                          min(len(df), 5_000), step=500)
    run_btn = st.button("▶ Run Causal Discovery", type="primary")

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running causal discovery…"):
        subset = df.sample(n=sample_n, random_state=42).sort_index() if sample_n < len(df) else df
        cd = PVCausalDiscovery(max_lag=int(max_lag), alpha_level=float(alpha))
        cd.fit(subset, FEATURE_COLUMNS)
        st.session_state["causal_discovery"] = cd
        st.session_state["causal_edges"] = cd.get_causal_edges()
        st.session_state["physics_validation"] = cd.validate_against_physics(KNOWN_CAUSAL_RELATIONS)
    st.success("Causal discovery complete!")

cd: PVCausalDiscovery | None = st.session_state.get("causal_discovery")

if cd is None:
    st.info("Configure parameters and click **Run Causal Discovery**.")
    st.stop()

edges = st.session_state.get("causal_edges", [])
physics_val = st.session_state.get("physics_validation", [])
G = cd.get_networkx_graph()

# ── Layout ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🕸️ Causal Graph", "📋 Discovered Edges",
                               "✅ Physics Validation"])

with tab1:
    fig = plot_causal_graph(G, physics_val)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🟢 Green edges match known physics relations. "
        "🔵 Blue edges are data-driven discoveries."
    )

with tab2:
    if edges:
        edges_df = pd.DataFrame(edges).sort_values("strength", ascending=False)
        st.dataframe(edges_df.style.format({"strength": "{:.4f}", "p_value": "{:.4f}"}),
                     use_container_width=True)
        st.metric("Total edges discovered", len(edges))
    else:
        st.info("No significant causal edges found. Try lowering the alpha level.")

with tab3:
    summary = compute_causal_physics_validation(physics_val)
    c1, c2, c3 = st.columns(3)
    c1.metric("Known relations", summary["total"])
    c2.metric("Rediscovered", summary["found"])
    c3.metric("Physics precision", f"{summary['precision']:.1%}")

    st.subheader("Relation-level results")
    val_df = pd.DataFrame(physics_val)
    val_df["status"] = val_df["found"].map({True: "✅ Found", False: "❌ Missing"})
    val_df["relation"] = val_df["relation"].astype(str)
    st.dataframe(val_df[["relation", "status", "strength"]].style.format(
        {"strength": "{:.4f}"}), use_container_width=True)
