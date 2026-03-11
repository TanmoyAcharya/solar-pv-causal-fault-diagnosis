"""
Page 4 – Causal Explanation
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st

from config import FEATURE_COLUMNS, FAULT_LABELS, SEQ_LEN
from models.deep_learning import predict_single, get_gradient_attribution
from utils.visualization import (plot_probability_bars, plot_feature_attribution,
                                  plot_causal_chain)

st.set_page_config(page_title="Causal Explanation", page_icon="💡", layout="wide")

st.markdown("""
<style>
  .stApp { background-color: #0f0f1a; color: #e0e0ff; }
  section[data-testid="stSidebar"] { background-color: #1a1a2e; }
  .nl-box {
    background: #1a1a2e;
    border-left: 4px solid #f39c12;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 12px 0;
    line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)

st.title("💡 Causal Explanation")
st.markdown(
    "Inspect **why** the model made a specific prediction – "
    "using gradient attribution, causal chains, and counterfactual analysis."
)

model = st.session_state.get("model")
X_test: np.ndarray | None = st.session_state.get("X_test")
y_test: np.ndarray | None = st.session_state.get("y_test")
causal_edges: list = st.session_state.get("causal_edges", [])

if model is None:
    st.warning("⚠️ No trained model found. Go to **Fault Diagnosis** first.")
    st.stop()

if X_test is None or len(X_test) == 0:
    st.warning("⚠️ No test data found. Please retrain the model.")
    st.stop()

# ── Sample window selector ────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🎯 Sample Window")
    idx = st.slider("Test sample index", 0, len(X_test) - 1, 0)
    run_btn = st.button("🔍 Explain this sample", type="primary")

sequence = X_test[idx]
true_label = int(y_test[idx]) if y_test is not None else None

if run_btn or "last_explanation" in st.session_state:
    if run_btn:
        with st.spinner("Running inference and attribution…"):
            pred = predict_single(model, sequence, FAULT_LABELS)
            attribution = get_gradient_attribution(model, sequence, FEATURE_COLUMNS)
        st.session_state["last_explanation"] = {
            "pred": pred,
            "attribution": attribution,
            "idx": idx,
            "true_label": true_label,
        }

    exp = st.session_state.get("last_explanation", {})
    pred = exp.get("pred", {})
    attribution = exp.get("attribution", {})

    if not pred:
        st.info("Click **Explain this sample** to generate an explanation.")
        st.stop()

    # ── Header ─────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Fault", pred.get("class_name", "—"))
    c2.metric("Confidence", f"{pred.get('confidence', 0):.1%}")
    if true_label is not None:
        c3.metric("True Label", FAULT_LABELS.get(true_label, str(true_label)),
                   delta="Correct ✅" if true_label == pred.get("class_idx") else "Wrong ❌",
                   delta_color="normal" if true_label == pred.get("class_idx") else "inverse")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Probabilities", "🔬 Attribution", "🔗 Causal Chain", "🔄 Counterfactual"
    ])

    with tab1:
        fig = plot_probability_bars(pred.get("probabilities", {}))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if attribution:
            fig = plot_feature_attribution(attribution)
            st.plotly_chart(fig, use_container_width=True)
            top_feat = max(attribution, key=lambda k: attribution[k])
            st.info(f"🔑 Most influential feature: **{top_feat}** "
                    f"(score {attribution[top_feat]:.4f})")
        else:
            st.info("Attribution not available.")

    with tab3:
        fault_name = pred.get("class_name", "Normal")
        # Filter causal edges to those relevant to this fault
        chain = [{"cause": e["cause"], "effect": e["effect"], "strength": e["strength"]}
                 for e in causal_edges
                 if e["cause"] in FEATURE_COLUMNS and e["effect"] in FEATURE_COLUMNS][:8]
        if chain:
            fig = plot_causal_chain(chain)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Causal Discovery first to see the causal chain.")

    with tab4:
        ci = st.session_state.get("causal_inference")
        if ci is not None:
            t_val = st.slider(
                "Hypothetical module_temp (°C)", -10.0, 90.0,
                float(sequence[:, FEATURE_COLUMNS.index("module_temp")].mean()),
                step=1.0,
            )
            cf = ci.get_counterfactual(t_val)
            if cf:
                st.json(cf)
            else:
                st.info("No counterfactual result available.")
        else:
            st.info("Run Causal Discovery to enable counterfactual analysis.")

    # ── Natural language explanation ─────────────────────────────────────────
    st.divider()
    st.subheader("📝 Natural Language Explanation")

    top_features = sorted(attribution.items(), key=lambda kv: -kv[1])[:3]
    feat_str = ", ".join(f"**{f}** ({v:.3f})" for f, v in top_features)
    confidence_pct = pred.get("confidence", 0) * 100

    nl = (
        f"The model predicts **{pred.get('class_name', 'Unknown')}** with "
        f"**{confidence_pct:.1f}% confidence**.\n\n"
        f"The top contributing features are: {feat_str}.\n\n"
        f"The causal discovery analysis identified {len(causal_edges)} significant "
        f"causal edges in the sensor data. The gradient attribution shows which "
        f"sensors most influenced this specific prediction."
    )
    st.markdown(f'<div class="nl-box">{nl}</div>', unsafe_allow_html=True)

else:
    st.info("Select a test sample index in the sidebar and click **Explain this sample**.")
