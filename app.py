"""
Solar PV Causal Fault Diagnosis – Home Page
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from utils.theme import apply_theme

st.set_page_config(
    page_title="Solar PV Fault Diagnosis",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme("""
  .nav-card {
    background: linear-gradient(180deg, rgba(18, 39, 62, 0.94), rgba(11, 25, 40, 0.96));
    border: 1px solid rgba(135, 170, 205, 0.18);
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 18px 45px rgba(2, 12, 24, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }
  .nav-card:hover {
    border-color: rgba(245, 185, 66, 0.45);
    transform: translateY(-2px);
  }
  .nav-card h3 {
    color: #f5f7fb;
    margin-bottom: 0.45rem;
  }
  .nav-card p { color: #bfd0e3; }
  .hero-title {
    font-size: 2.9rem;
    font-weight: 800;
    color: #f5b942;
    margin-bottom: 0.25rem;
  }
  .hero-sub {
    font-size: 1.08rem;
    color: #d7e3f0;
    line-height: 1.7;
    max-width: 70rem;
  }
  .stat-box {
    background: linear-gradient(180deg, rgba(17, 39, 62, 0.9), rgba(11, 25, 40, 0.95));
    border: 1px solid rgba(135, 170, 205, 0.18);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 16px 35px rgba(2, 12, 24, 0.22);
  }
  .stat-value { font-size: 2rem; font-weight: 700; color: #ffd27d; }
  .stat-label { font-size: 0.88rem; color: #bfd0e3; }
""")

# ── Hero section ─────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">☀️ Solar PV Causal Fault Diagnosis</p>',
            unsafe_allow_html=True)
st.markdown("""
<p class="hero-sub">
An end-to-end <strong>Causal Deep Learning</strong> system for diagnosing faults
in solar photovoltaic arrays. This app combines <em>PCMCI causal discovery</em>,
<em>DoWhy causal inference</em>, and <em>causal-informed neural networks</em>
to detect faults and explain their root causes.
</p>
""", unsafe_allow_html=True)

st.divider()

# ── Quick stats ───────────────────────────────────────────────────────────────
df = st.session_state.get("df")
pipeline = st.session_state.get("pipeline")

if df is not None:
    from utils.metrics import compute_system_health_score, compute_energy_loss
    health = compute_system_health_score(df)
    energy_loss = compute_energy_loss(df)
    total_loss = sum(energy_loss.values())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{len(df):,}</div>'
                    '<div class="stat-label">Samples loaded</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{health:.1f}%</div>'
                    '<div class="stat-label">System Health</div></div>',
                    unsafe_allow_html=True)
    with c3:
        n_faults = (df["fault_label"] != 0).sum() if "fault_label" in df.columns else 0
        st.markdown(f'<div class="stat-box"><div class="stat-value">{n_faults:,}</div>'
                    '<div class="stat-label">Fault events</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{total_loss:.1f}</div>'
                    '<div class="stat-label">kWh lost (est.)</div></div>',
                    unsafe_allow_html=True)
    st.divider()

# ── Navigation cards ──────────────────────────────────────────────────────────
st.subheader("📋 Navigation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="nav-card">
      <h3>📤 1 · Data Upload</h3>
      <p>Upload your PV sensor CSV or generate a synthetic dataset to get started.</p>
    </div>
    <div class="nav-card">
      <h3>🔍 2 · Causal Discovery</h3>
      <p>Discover causal relationships between sensors using PCMCI or lagged correlation.</p>
    </div>
    <div class="nav-card">
      <h3>🤖 3 · Fault Diagnosis</h3>
      <p>Train LSTM, Transformer, or Causal-Informed neural networks for fault classification.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nav-card">
      <h3>💡 4 · Causal Explanation</h3>
      <p>Get feature attributions, causal chains, and counterfactual analysis for predictions.</p>
    </div>
    <div class="nav-card">
      <h3>📊 5 · Dashboard</h3>
      <p>Monitor system health, energy loss, fault timelines, and sensor overviews.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.info(
        "👈 Use the sidebar to navigate between pages. "
        "Start with **Data Upload** if no data is loaded yet.",
        icon="ℹ️",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Solar PV Causal Deep Learning Fault Diagnosis · "
    "Powered by PyTorch · Plotly · Streamlit"
)
