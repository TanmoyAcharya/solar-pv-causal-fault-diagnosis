import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from config import FAULT_LABELS, FAULT_COLORS, FEATURE_COLUMNS
from utils.metrics import (compute_system_health_score, compute_energy_loss,
                            compute_causal_physics_validation)
from utils.visualization import (plot_fault_timeline, plot_fault_heatmap,
                                  plot_system_health_gauge, plot_energy_loss_bars,
                                  plot_sensor_grid, plot_causal_graph)

st.set_page_config(page_title="📈 Dashboard", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0f0f1a; }
section[data-testid="stSidebar"] { background-color: #12122a; border-right:1px solid #f39c12; }
.kpi-card { background:#1a1a2e; border:1px solid #333; border-left:4px solid #f39c12;
             border-radius:8px; padding:18px; text-align:center; }
.kpi-num { font-size:2rem; font-weight:900; color:#f39c12; }
.kpi-label { color:#aaa; font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 System Monitoring Dashboard")
st.caption("Full analytics dashboard for Solar PV fault monitoring, energy loss estimation, and causal insights.")

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.error("⚠️ No data loaded. Please go to **📂 Data Upload** first.")
    st.stop()

df = st.session_state['df']
pipeline = st.session_state.get('pipeline')
causal_graph = st.session_state.get('causal_graph')
physics_validation = st.session_state.get('physics_validation', [])

# ── KPIs ─────────────────────────────────────────────────────────────────────
st.subheader("🔑 Key Performance Indicators")
health = compute_system_health_score(df)
energy_loss = compute_energy_loss(df)
total_loss = sum(v for k, v in energy_loss.items() if k != 'Normal')
fault_rate = (df['fault_label'] > 0).mean() * 100 if 'fault_label' in df.columns else 0
n_faults = (df['fault_label'] > 0).sum() if 'fault_label' in df.columns else 0

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (f"{health:.1f}%.", "System Health Score"),
    (f"{fault_rate:.1f}%.", "Fault Rate"),
    (f"{n_faults:,}", "Fault Events"),
    (f"{total_loss:.1f} kWh", "Total Energy Lost"),
    (f"{len(df):,}", "Records Analysed"),
]
for col, (num, label) in zip([k1, k2, k3, k4, k5], kpis):
    with col:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-num">{num}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# System health gauge
gauge_col, loss_col = st.columns([1, 2])
with gauge_col:
    fig_gauge = plot_system_health_gauge(health)
    st.plotly_chart(fig_gauge, use_container_width=True)
with loss_col:
    fig_loss = plot_energy_loss_bars(energy_loss)
    st.plotly_chart(fig_loss, use_container_width=True)

# ── Timeline ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⚡ Fault Timeline")
fig_tl = plot_fault_timeline(df)
st.plotly_chart(fig_tl, use_container_width=True)

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.subheader("🗓️ Fault Frequency Heatmap")
if 'timestamp' in df.columns:
    fig_hm = plot_fault_heatmap(df)
    st.plotly_chart(fig_hm, use_container_width=True)

# ── Sensor Grid ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Sensor Data Overview")
fig_sg = plot_sensor_grid(df)
st.plotly_chart(fig_sg, use_container_width=True)

# ── Causal Graph ──────────────────────────────────────────────────────────────
if causal_graph is not None:
    st.markdown("---")
    st.subheader("🔗 Causal Structure")
    fig_cg = plot_causal_graph(causal_graph, physics_validation)
st.plotly_chart(fig_cg, use_container_width=True)

# ── Model Performance ─────────────────────────────────────────────────────────
if pipeline is not None and pipeline.is_fitted:
    st.markdown("---")
    st.subheader("🤖 Model Performance Summary")
    eval_r = pipeline.eval_results
    m1, m2, m3, m4 = st.columns(4)
    for col, num, label in [
        (m1, f"{eval_r['accuracy']*100:.1f}%.", "Test Accuracy"),
        (m2, f"{eval_r['f1_macro']*100:.1f}%.", "Macro F1"),
        (m3, f"{eval_r['precision_macro']*100:.1f}%.", "Precision"),
        (m4, f"{eval_r['recall_macro']*100:.1f}%.", "Recall"),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-num">{num}</div>'
                f'<div class="kpi-label">{label}</div></div>',
                unsafe_allow_html=True
            )

# ── Report Download ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Download System Report")
report = {
    'metric': ['System Health', 'Fault Rate (%)', 'Total Energy Loss (kWh)', 'Records'],
    'value': [f"{health:.1f}%.", f"{fault_rate:.1f}%.", f"{total_loss:.1f}", f"{len(df):,}"]
}
if pipeline is not None and pipeline.is_fitted:
    report['metric'].extend(['Model Accuracy', 'Macro F1'])
    report['value'].extend([f"{pipeline.eval_results['accuracy']*100:.1f}%.", f"{pipeline.eval_results['f1_macro']*100:.1f}%.".])
df_report = pd.DataFrame(report)
csv_report = df_report.to_csv(index=False).encode()
st.download_button("⬇️ Download Summary Report CSV", csv_report,
                   "pv_system_report.csv", "text/csv")
if 'fault_label' in df.columns:
    fault_counts = df['fault_label'].map(FAULT_LABELS).value_counts().reset_index()
    fault_counts.columns = ['Fault Type', 'Count']
    fault_csv = fault_counts.to_csv(index=False).encode()
    st.download_button("⬇️ Download Fault Log CSV", fault_csv,
                        "fault_log.csv", "text/csv")

st.markdown("---")
st.success("🎉 **Full analysis complete!** Navigate back to any section using the sidebar.")
