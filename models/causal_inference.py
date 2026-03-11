"""
Causal inference for Solar PV data using DoWhy with a linear-regression fallback.
"""

import numpy as np
import pandas as pd

try:
    import dowhy
    from dowhy import CausalModel
    _DOWHY_AVAILABLE = True
except ImportError:
    _DOWHY_AVAILABLE = False


class PVCausalInference:
    """Estimates causal effects between PV sensor variables."""

    def __init__(self):
        self._ate: float = 0.0
        self._treatment: str = ""
        self._outcome: str = ""
        self._data: pd.DataFrame | None = None
        self._causal_graph = None
        self._model = None   # linear regression fallback
        self._summary: dict = {}

    # ── public API ────────────────────────────────────────────────────────────

    def build_model(self, data: pd.DataFrame, causal_graph,
                    treatment: str, outcome: str) -> "PVCausalInference":
        self._data = data.copy()
        self._causal_graph = causal_graph
        self._treatment = treatment
        self._outcome = outcome

        if _DOWHY_AVAILABLE:
            self._build_dowhy(data, causal_graph, treatment, outcome)
        else:
            self._build_linear(data, treatment, outcome)

        return self

    def get_ate(self) -> float:
        return self._ate

    def get_counterfactual(self, treatment_value: float) -> dict:
        if self._data is None:
            return {}
        baseline_treatment = float(self._data[self._treatment].mean())
        delta = treatment_value - baseline_treatment
        counterfactual_outcome = float(self._data[self._outcome].mean()) + self._ate * delta
        return {
            "treatment": self._treatment,
            "outcome": self._outcome,
            "treatment_value": treatment_value,
            "baseline_treatment": baseline_treatment,
            "baseline_outcome": float(self._data[self._outcome].mean()),
            "counterfactual_outcome": counterfactual_outcome,
            "delta_treatment": delta,
            "delta_outcome": self._ate * delta,
        }

    def get_effect_summary(self) -> dict:
        return self._summary

    # ── private helpers ───────────────────────────────────────────────────────

    def _build_dowhy(self, data, causal_graph, treatment, outcome):
        try:
            import networkx as nx
            if isinstance(causal_graph, nx.DiGraph):
                dot_graph = "digraph { "
                for u, v in causal_graph.edges():
                    dot_graph += f"{u} -> {v}; "
                dot_graph += "}"
            else:
                dot_graph = None

            model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                graph=dot_graph,
            )
            identified = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                identified,
                method_name="backdoor.linear_regression",
            )
            self._ate = float(estimate.value)
            self._summary = {
                "method": "DoWhy linear regression",
                "ate": self._ate,
                "treatment": treatment,
                "outcome": outcome,
            }
        except Exception:
            self._build_linear(data, treatment, outcome)

    def _build_linear(self, data, treatment, outcome):
        from sklearn.linear_model import LinearRegression
        cols = [c for c in data.columns if c not in [treatment, outcome] and
                pd.api.types.is_numeric_dtype(data[c])]
        X = data[[treatment] + cols].dropna()
        y = data.loc[X.index, outcome]
        reg = LinearRegression().fit(X, y)
        self._ate = float(reg.coef_[0])
        self._model = reg
        self._summary = {
            "method": "linear regression fallback",
            "ate": self._ate,
            "treatment": treatment,
            "outcome": outcome,
            "r2": float(reg.score(X, y)),
        }
