"""
Metrics utilities for Solar PV fault diagnosis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)


# ── Fault taxonomy (local copy to avoid circular imports) ────────────────────
_FAULT_LABELS = {
    0: "Normal",
    1: "Partial Shading",
    2: "Soiling",
    3: "Hot Spot",
    4: "PID Effect",
    5: "Bypass Diode Failure",
    6: "String Disconnect",
}

# Rough efficiency loss percentages per fault type
_FAULT_EFFICIENCY_LOSS = {
    0: 0.00,
    1: 0.20,
    2: 0.15,
    3: 0.30,
    4: 0.25,
    5: 0.40,
    6: 0.50,
}


def compute_classification_metrics(y_true, y_pred, fault_labels: dict) -> dict:
    """Return standard classification metrics."""
    labels = sorted(fault_labels.keys())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                    labels=labels, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                  labels=labels, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                            labels=labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def compute_system_health_score(df: pd.DataFrame) -> float:
    """Return a 0-100 health score based on fault label distribution.

    Normal (label 0) = full health; other faults reduce the score proportional
    to their severity weight.
    """
    if "fault_label" not in df.columns or len(df) == 0:
        return 100.0
    counts = df["fault_label"].value_counts(normalize=True)
    penalty = 0.0
    for label, freq in counts.items():
        loss = _FAULT_EFFICIENCY_LOSS.get(int(label), 0.3)
        if int(label) != 0:
            penalty += freq * loss
    health = max(0.0, 100.0 * (1.0 - penalty))
    return round(health, 2)


def compute_energy_loss(df: pd.DataFrame) -> dict:
    """Estimate energy loss (kWh) per fault type.

    Assumes each row represents a 15-minute interval and uses the dc_power
    column (W) if available, otherwise defaults to 300 W nominal.
    """
    if "fault_label" not in df.columns or len(df) == 0:
        return {}

    interval_h = 15 / 60  # hours per sample
    result = {}
    for label, name in _FAULT_LABELS.items():
        if label == 0:
            continue
        mask = df["fault_label"] == label
        if mask.sum() == 0:
            result[name] = 0.0
            continue
        if "dc_power" in df.columns:
            avg_power_w = float(df.loc[mask, "dc_power"].mean())
        else:
            avg_power_w = 300.0
        loss_frac = _FAULT_EFFICIENCY_LOSS.get(label, 0.2)
        energy_lost_kwh = avg_power_w * loss_frac * mask.sum() * interval_h / 1000.0
        result[name] = round(energy_lost_kwh, 3)
    return result


def compute_causal_physics_validation(validation_results: list) -> dict:
    """Summarise physics-validation output from PVCausalDiscovery."""
    if not validation_results:
        return {"total": 0, "found": 0, "missing": 0, "precision": 0.0}
    total = len(validation_results)
    found = sum(1 for r in validation_results if r["found"])
    avg_strength = float(np.mean([r["strength"] for r in validation_results
                                   if r["found"]] or [0.0]))
    missing_rels = [r["relation"] for r in validation_results if not r["found"]]
    return {
        "total": total,
        "found": found,
        "missing": total - found,
        "precision": round(found / total, 3) if total else 0.0,
        "avg_strength": round(avg_strength, 4),
        "missing_relations": missing_rels,
    }


def compute_model_comparison(results_dict: dict) -> pd.DataFrame:
    """Build a comparison DataFrame from a dict of evaluate_model results.

    Parameters
    ----------
    results_dict : {model_name: eval_dict, ...}
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy", 0.0),
            "F1 Macro": metrics.get("f1_macro", 0.0),
            "Precision": metrics.get("precision_macro", 0.0),
            "Recall": metrics.get("recall_macro", 0.0),
        })
    return pd.DataFrame(rows)
