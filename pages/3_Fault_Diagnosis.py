"""
Page 3 – Fault Diagnosis (model training & evaluation)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import pandas as pd

from config import (FEATURE_COLUMNS, FAULT_LABELS, EPOCHS, LEARNING_RATE,
                    BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, N_HEADS, DROPOUT,
                    N_FEATURES, N_CLASSES)
from models.deep_learning import create_model, train_model, evaluate_model
from utils.preprocessing import normalize_features, create_sequences, train_test_split_temporal
from utils.theme import apply_theme
from utils.visualization import plot_training_curves, plot_confusion_matrix

st.set_page_config(page_title="Fault Diagnosis", page_icon="🤖", layout="wide")

apply_theme()

st.title("🤖 Fault Diagnosis")
st.markdown("Train a deep learning model to classify PV faults from sensor sequences.")

df: pd.DataFrame | None = st.session_state.get("df")
if df is None:
    st.warning("⚠️ No data loaded. Go to **Data Upload** first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Model Configuration")
    model_type = st.selectbox(
        "Model architecture",
        ["transformer", "lstm", "causal"],
        format_func=lambda x: {
            "transformer": "Transformer",
            "lstm": "Bidirectional LSTM",
            "causal": "CausalInformedNet",
        }[x],
    )
    epochs = st.slider("Epochs", 1, 50, EPOCHS)
    lr = st.number_input("Learning rate", 1e-5, 1e-1, LEARNING_RATE,
                          format="%.5f", step=1e-4)
    batch_size = st.select_slider("Batch size", [16, 32, 64, 128, 256], value=BATCH_SIZE)
    hidden_size = st.select_slider("Hidden size", [32, 64, 128, 256], value=HIDDEN_SIZE)
    train_btn = st.button("🚀 Train model", type="primary")

# ── Training ──────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Preprocessing data…"):
        df_norm, scaler = normalize_features(df, FEATURE_COLUMNS)
        train_df, test_df = train_test_split_temporal(df_norm, test_ratio=0.2)
        train_df, val_df = train_test_split_temporal(train_df, test_ratio=0.15)

        X_tr, y_tr = create_sequences(
            train_df[FEATURE_COLUMNS].values, df.loc[train_df.index, "fault_label"].values, SEQ_LEN)
        X_val, y_val = create_sequences(
            val_df[FEATURE_COLUMNS].values, df.loc[val_df.index, "fault_label"].values, SEQ_LEN)
        X_te, y_te = create_sequences(
            test_df[FEATURE_COLUMNS].values, df.loc[test_df.index, "fault_label"].values, SEQ_LEN)

    causal_mask = None
    if model_type == "causal":
        cd = st.session_state.get("causal_discovery")
        if cd is not None:
            causal_mask = cd.get_adjacency_matrix()

    model = create_model(model_type, N_FEATURES, N_CLASSES,
                          hidden_size, NUM_LAYERS, N_HEADS, DROPOUT, causal_mask)

    progress_bar = st.progress(0, text="Training…")
    epoch_status = st.empty()

    history_store = []

    def _cb(epoch, total, hist):
        progress_bar.progress(epoch / total,
                               text=f"Epoch {epoch}/{total} – "
                                    f"val_loss={hist['val_loss'][-1]:.4f} "
                                    f"val_acc={hist['val_acc'][-1]:.2%}")
        epoch_status.caption(
            f"train_loss={hist['train_loss'][-1]:.4f}  "
            f"train_acc={hist['train_acc'][-1]:.2%}"
        )
        history_store.append(hist.copy())

    history = train_model(
        model, X_tr, y_tr, X_val, y_val,
        epochs=int(epochs), lr=float(lr), batch_size=int(batch_size),
        progress_callback=_cb,
    )
    progress_bar.progress(1.0, text="Training complete ✅")

    eval_results = evaluate_model(model, X_te, y_te)

    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["training_history"] = history
    st.session_state["eval_results"] = eval_results
    st.session_state["X_test"] = X_te
    st.session_state["y_test"] = y_te
    st.session_state["model_type"] = model_type

    st.success("✅ Model trained and evaluated!")

# ── Results display ───────────────────────────────────────────────────────────
history = st.session_state.get("training_history")
eval_results = st.session_state.get("eval_results")

if history and eval_results:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{eval_results['accuracy']:.2%}")
    col2.metric("F1 Macro", f"{eval_results['f1_macro']:.2%}")
    col3.metric("Precision", f"{eval_results['precision_macro']:.2%}")
    col4.metric("Recall", f"{eval_results['recall_macro']:.2%}")

    tab1, tab2, tab3 = st.tabs(["📉 Training Curves", "🔲 Confusion Matrix", "📄 Class Report"])

    with tab1:
        fig = plot_training_curves(history)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cm = np.array(eval_results["confusion_matrix"])
        fig = plot_confusion_matrix(cm, FAULT_LABELS)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.text(eval_results.get("class_report", "No report available."))
else:
    st.info("Configure the model in the sidebar and click **Train model** to begin.")
