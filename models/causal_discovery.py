"""
Causal discovery for Solar PV data using PCMCI (tigramite) with a lagged-correlation fallback.
"""

import numpy as np
import pandas as pd
import networkx as nx
from itertools import product

try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    _TIGRAMITE_AVAILABLE = True
except ImportError:
    _TIGRAMITE_AVAILABLE = False


class PVCausalDiscovery:
    """Discovers causal structure in PV sensor data."""

    def __init__(self, max_lag: int = 4, alpha_level: float = 0.05):
        self.max_lag = max_lag
        self.alpha_level = alpha_level

        self._feature_cols: list = []
        self._val_matrix: np.ndarray | None = None   # shape (N, N, max_lag+1)
        self._p_matrix: np.ndarray | None = None
        self._edges: list[dict] = []
        self._graph: nx.DiGraph | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, data: pd.DataFrame, feature_cols: list) -> "PVCausalDiscovery":
        self._feature_cols = feature_cols
        values = data[feature_cols].dropna().values.astype(float)

        if _TIGRAMITE_AVAILABLE:
            self._fit_pcmci(values)
        else:
            self._fit_lagged_correlation(values)

        self._build_graph()
        return self

    def get_networkx_graph(self) -> nx.DiGraph:
        if self._graph is None:
            raise RuntimeError("Call fit() first.")
        return self._graph

    def get_causal_edges(self) -> list[dict]:
        return self._edges

    def get_adjacency_matrix(self) -> np.ndarray:
        n = len(self._feature_cols)
        adj = np.zeros((n, n), dtype=float)
        for e in self._edges:
            i = self._feature_cols.index(e["cause"])
            j = self._feature_cols.index(e["effect"])
            adj[i, j] = max(adj[i, j], e["strength"])
        return adj

    def validate_against_physics(self, known_relations: list) -> list[dict]:
        results = []
        edge_map = {(e["cause"], e["effect"]): e["strength"] for e in self._edges}
        for rel in known_relations:
            found = rel in edge_map
            results.append({
                "relation": rel,
                "found": found,
                "strength": edge_map.get(rel, 0.0),
            })
        return results

    # ── private helpers ───────────────────────────────────────────────────────

    def _fit_pcmci(self, values: np.ndarray):
        dataframe = pp.DataFrame(values, var_names=self._feature_cols)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
        results = pcmci.run_pcmci(tau_max=self.max_lag, pc_alpha=self.alpha_level)
        self._val_matrix = results["val_matrix"]
        self._p_matrix = results["p_matrix"]
        self._extract_edges_from_matrices()

    def _fit_lagged_correlation(self, values: np.ndarray):
        n = len(self._feature_cols)
        lags = self.max_lag + 1
        self._val_matrix = np.zeros((n, n, lags))
        self._p_matrix = np.ones((n, n, lags))

        from scipy import stats

        for lag in range(1, lags):
            x = values[:-lag]
            y = values[lag:]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    r, p = stats.pearsonr(x[:, i], y[:, j])
                    self._val_matrix[i, j, lag] = r
                    self._p_matrix[i, j, lag] = p

        self._extract_edges_from_matrices()

    def _extract_edges_from_matrices(self):
        self._edges = []
        n = len(self._feature_cols)
        for i, j in product(range(n), range(n)):
            if i == j:
                continue
            for lag in range(1, self.max_lag + 1):
                p_val = self._p_matrix[i, j, lag]
                strength = abs(float(self._val_matrix[i, j, lag]))
                if p_val < self.alpha_level and strength > 0.05:
                    self._edges.append({
                        "cause": self._feature_cols[i],
                        "effect": self._feature_cols[j],
                        "strength": strength,
                        "lag": lag,
                        "p_value": float(p_val),
                    })

    def _build_graph(self):
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(self._feature_cols)
        for e in self._edges:
            u, v = e["cause"], e["effect"]
            if self._graph.has_edge(u, v):
                if e["strength"] > self._graph[u][v]["strength"]:
                    self._graph[u][v].update(e)
            else:
                self._graph.add_edge(u, v, **e)
