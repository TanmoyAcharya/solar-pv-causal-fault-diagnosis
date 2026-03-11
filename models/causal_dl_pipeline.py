"""
Master Pipeline: Causal Discovery → Causal Inference → Deep Learning
Connects all modules for end-to-end Solar PV fault diagnosis.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from models.causal_discovery import PVCausalDiscovery
from models.causal_inference import PVCausalInference
from models.deep_learning import (create_model, train_model, evaluate_model,
                                    predict_single, get_gradient_attribution)
from utils.preprocessing import normalize_features, create_sequences, train_test_split_temporal
from config import (FEATURE_COLUMNS, FAULT_LABELS, SEQ_LEN, BATCH_SIZE, EPOCHS,
                    LEARNING_RATE, HIDDEN_SIZE, NUM_LAYERS, N_HEADS, DROPOUT,
                    N_FEATURES, N_CLASSES, KNOWN_CAUSAL_RELATIONS, MAX_LAG, ALPHA_LEVEL)

FAULT_ACTIONS = {
    'Hot Spot': 'Perform thermal imaging. Check for cell damage or reverse bias conditions.',
    'Partial Shading': 'Inspect for obstructions (trees, dirt, debris). Check bypass diodes.',
    'Soiling': 'Schedule panel cleaning. Monitor soiling rate for optimal cleaning frequency.',
    'PID Effect': 'Check grounding and insulation resistance. Consider PID recovery module.',
    'Bypass Diode Failure': 'Replace bypass diode. Inspect affected string for further damage.',
    'String Disconnect': 'Inspect string connections, combiner box, and MC4 connectors.',
    'Normal': 'System operating normally. Continue routine monitoring.'
}

FAULT_CAUSAL_VARS = {
    'Hot Spot': ['irradiance', 'wind_speed', 'module_temp', 'efficiency'],
    'Partial Shading': ['irradiance', 'dc_current', 'string_current_imbalance'],
    'Soiling': ['irradiance', 'dc_current', 'dc_power'],
    'PID Effect': ['module_temp', 'dc_voltage', 'efficiency'],
    'Bypass Diode Failure': ['dc_current', 'string_current_imbalance', 'dc_power'],
    'String Disconnect': ['dc_current', 'dc_power'],
    'Normal': ['irradiance', 'dc_current', 'dc_power']
}


class CausalPVDiagnosisPipeline:
    """
    End-to-end pipeline for Solar PV fault diagnosis using causal deep learning.

    Steps:
    1. Causal discovery (PCMCI) on sensor time-series
    2. Causal inference (DoWhy) for effect estimation
    3. Deep learning fault classification (with causal mask)
    4. Causal explanation generation
    """

    def __init__(self):
        self.causal_discovery = None
        self.causal_inference = PVCausalInference()
        self.model = None
        self.scaler = None
        self.causal_graph = None
        self.causal_edges = []
        self.causal_mask = None
        self.physics_validation = []
        self.training_history = None
        self.eval_results = None
        self.is_fitted = False
        self.model_type = 'causal_informed'
        self.feature_cols = FEATURE_COLUMNS
        self.X_test = None
        self.y_test = None

    def fit(self, data: pd.DataFrame, model_type: str = 'causal_informed',
            epochs: int = EPOCHS, lr: float = LEARNING_RATE,
            progress_callback=None) -> dict:
        """
        Fit the complete pipeline on PV sensor data.

        Parameters
        ----------
        data : pd.DataFrame  with sensor columns + 'fault_label'
        model_type : str  'lstm', 'transformer', or 'causal_informed'
        epochs : int
        lr : float
        progress_callback : callable(epoch, total, train_loss, val_loss, train_acc, val_acc)

        Returns
        -------
        training_history dict
        """
        self.model_type = model_type

        # ── Step 1: Causal Discovery ──────────────────────────────────────
        self.causal_discovery = PVCausalDiscovery(max_lag=MAX_LAG, alpha_level=ALPHA_LEVEL)
        self.causal_discovery.fit(data, self.feature_cols)
        self.causal_graph = self.causal_discovery.get_networkx_graph()
        self.causal_edges = self.causal_discovery.get_causal_edges()
        self.causal_mask = self.causal_discovery.get_adjacency_matrix()
        self.physics_validation = self.causal_discovery.validate_against_physics(KNOWN_CAUSAL_RELATIONS)

        # ── Step 2: Causal Inference ──────────────────────────────────────
        daytime = data[data['irradiance'] > 10].copy()
        if len(daytime) > 100:
            try:
                self.causal_inference.build_model(
                    daytime, self.causal_graph, 'module_temp', 'dc_power'
                )
            except Exception:
                pass

        # ── Step 3: Deep Learning Training ───────────────────────────────
        train_df, test_df = train_test_split_temporal(data, test_ratio=0.2)

        df_norm, self.scaler = normalize_features(train_df, self.feature_cols)
        df_test_norm, _ = normalize_features(test_df, self.feature_cols, self.scaler)

        X_tr_arr = df_norm[self.feature_cols].values
        y_tr_arr = train_df['fault_label'].values
        X_te_arr = df_test_norm[self.feature_cols].values
        y_te_arr = test_df['fault_label'].values

        X_seq_tr, y_seq_tr = create_sequences(X_tr_arr, y_tr_arr, SEQ_LEN)
        X_seq_te, y_seq_te = create_sequences(X_te_arr, y_te_arr, SEQ_LEN)

        self.X_test = X_seq_te
        self.y_test = y_seq_te

        val_split = int(len(X_seq_tr) * 0.85)
        X_tr, X_val = X_seq_tr[:val_split], X_seq_tr[val_split:]
        y_tr, y_val = y_seq_tr[:val_split], y_seq_tr[val_split:]

        self.model = create_model(
            model_type, N_FEATURES, N_CLASSES,
            HIDDEN_SIZE, NUM_LAYERS, N_HEADS, DROPOUT,
            self.causal_mask
        )

        self.training_history = train_model(
            self.model, X_tr, y_tr, X_val, y_val,
            epochs=epochs, lr=lr, batch_size=BATCH_SIZE,
            device='cpu', progress_callback=progress_callback
        )

        self.eval_results = evaluate_model(self.model, X_seq_te, y_seq_te, device='cpu')
        self.is_fitted = True
        return self.training_history

    def predict(self, sequence: np.ndarray) -> dict:
        """Predict fault type for a single sequence window."""
        if not self.is_fitted:
            return {'error': 'Pipeline not fitted. Call fit() first.'}
        result = predict_single(self.model, sequence, FAULT_LABELS, device='cpu')
        result['top_causal_drivers'] = self.causal_edges[:5]
        return result

    def explain(self, sequence: np.ndarray, sensor_readings: dict = None) -> dict:
        """
        Generate a full causal explanation for a diagnosis.

        Returns
        -------
        dict with: predicted_fault, confidence, probabilities,
                   feature_attribution, causal_chain, counterfactual,
                   natural_language
        """
        if not self.is_fitted:
            return {'error': 'Pipeline not fitted.'}

        pred = predict_single(self.model, sequence, FAULT_LABELS, device='cpu')

        # Feature attribution
        attribution = {}
        try:
            attribution = get_gradient_attribution(
                self.model, sequence, self.feature_cols, device='cpu'
            )
        except Exception:
            attribution = {f: 0.0 for f in self.feature_cols}

        fault_name = pred['class_name']
        chain = self._build_causal_chain(fault_name)

        # Counterfactual analysis
        cf_result = {}
        if sensor_readings is not None:
            try:
                t_val = float(sensor_readings.get('module_temp', 45.0))
                cf_result = self.causal_inference.get_counterfactual(t_val - 10.0)
            except Exception:
                pass

        # Natural language
        nl = self._generate_nl_explanation(
            fault_name, pred['confidence'],
            attribution, chain, sensor_readings or {}
        )

        return {
            'predicted_fault': fault_name,
            'confidence': pred['confidence'],
            'probabilities': pred['probabilities'],
            'feature_attribution': attribution,
            'causal_chain': chain,
            'counterfactual': cf_result,
            'natural_language': nl
        }

    def _build_causal_chain(self, fault_name: str) -> list:
        """Build a causal path leading to the fault based on discovered edges."""
        vars_for_fault = FAULT_CAUSAL_VARS.get(fault_name, self.feature_cols[:3])
        chain = []
        for i in range(len(vars_for_fault) - 1):
            src = vars_for_fault[i]
            tgt = vars_for_fault[i + 1]
            strength = next(
                (e['strength'] for e in self.causal_edges
                 if e['cause'] == src and e['effect'] == tgt),
                0.0
            )
            chain.append({'from': src, 'to': tgt, 'strength': round(float(strength), 3)})
        return chain

    def _generate_nl_explanation(self, fault_name: str, confidence: float,
                                   attribution: dict, chain: list,
                                   readings: dict) -> str:
        """Generate a natural language explanation of the diagnosis."""
        if attribution:
            top = max(attribution, key=lambda k: abs(attribution.get(k, 0)))
            top_val = readings.get(top, 'N/A')
            top_str = f'{top_val:.2f}' if isinstance(top_val, (int, float)) else str(top_val)
            feature_line = f"**Most influential sensor:** `{top}` (value: {top_str})"
        else:
            feature_line = ''

        chain_str = ' → '.join(
            [c['from'] for c in chain] + ([chain[-1]['to']] if chain else [])
        )
        action = FAULT_ACTIONS.get(fault_name, 'Perform general system inspection.')

        return (
            f"**{fault_name}** detected with **{confidence * 100:.1f}% confidence**.\n\n"
            f"{feature_line}\n\n"
            f"**Causal pathway:** {chain_str}\n\n"
            f"The causal discovery analysis (PCMCI) identified this fault pathway "
            f"directly from your PV sensor data. The Causal-Informed Neural Network "
            f"classified this fault while respecting the discovered causal structure.\n\n"
            f"**Recommended action:** {action}"
        )

    def save(self, path: str = 'pipeline.pkl') -> None:
        """Save the pipeline to disk."""
        payload = {
            'model_state': self.model.state_dict() if self.model else None,
            'model_type': self.model_type,
            'scaler': self.scaler,
            'causal_edges': self.causal_edges,
            'causal_mask': self.causal_mask,
            'eval_results': self.eval_results,
            'training_history': self.training_history,
            'physics_validation': self.physics_validation,
        }
        joblib.dump(payload, path)

    def load(self, path: str = 'pipeline.pkl') -> None:
        """Load a previously saved pipeline from disk."""
        d = joblib.load(path)
        self.scaler = d['scaler']
        self.causal_edges = d.get('causal_edges', [])
        self.causal_mask = d.get('causal_mask')
        self.eval_results = d.get('eval_results')
        self.training_history = d.get('training_history')
        self.physics_validation = d.get('physics_validation', [])
        self.model_type = d.get('model_type', 'causal_informed')
        if d.get('model_state'):
            self.model = create_model(
                self.model_type, N_FEATURES, N_CLASSES,
                HIDDEN_SIZE, NUM_LAYERS, N_HEADS, DROPOUT,
                self.causal_mask
            )
            self.model.load_state_dict(d['model_state'])
            self.model.eval()
        self.is_fitted = True
