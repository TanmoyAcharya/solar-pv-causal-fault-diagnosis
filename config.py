"""
Central configuration for Solar PV Causal Deep Learning Fault Diagnosis
"""

# ── Fault taxonomy ─────────────────────────────────────────────────────────────
FAULT_LABELS = {
    0: "Normal",
    1: "Partial Shading",
    2: "Soiling",
    3: "Hot Spot",
    4: "PID Effect",
    5: "Bypass Diode Failure",
    6: "String Disconnect",
}

FAULT_COLORS = {
    0: "#2ecc71",   # green  – normal
    1: "#3498db",   # blue   – shading
    2: "#f39c12",   # orange – soiling
    3: "#e74c3c",   # red    – hot spot
    4: "#9b59b6",   # purple – PID
    5: "#e91e63",   # pink   – diode
    6: "#1abc9c",   # teal   – string disconnect
}

# ── Sensor / feature columns ───────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "irradiance",
    "module_temp",
    "ambient_temp",
    "wind_speed",
    "dc_voltage",
    "dc_current",
    "dc_power",
    "efficiency",
    "string_current_imbalance",
]

# ── Physics-based ground-truth causal relations ────────────────────────────────
# (cause, effect) pairs that we know must exist from solar physics
KNOWN_CAUSAL_RELATIONS = [
    ("irradiance", "dc_power"),
    ("irradiance", "module_temp"),
    ("module_temp", "dc_voltage"),
    ("dc_voltage", "dc_power"),
    ("dc_current", "dc_power"),
    ("wind_speed", "module_temp"),
    ("irradiance", "dc_current"),
    ("module_temp", "efficiency"),
    ("dc_power", "efficiency"),
]

# ── Causal discovery parameters ────────────────────────────────────────────────
MAX_LAG = 4          # max time lag (4 × 15 min = 1 hour look-back)
ALPHA_LEVEL = 0.05   # significance threshold for PCMCI

# ── Deep-learning hyperparameters ─────────────────────────────────────────────
SEQ_LEN = 24         # input window length (24 × 15 min = 6 hours)
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.2

# ── Derived constants ─────────────────────────────────────────────────────────
N_FEATURES = len(FEATURE_COLUMNS)   # 9
N_CLASSES  = len(FAULT_LABELS)      # 7
N_HEADS    = NUM_HEADS              # alias used by pipeline

# ── Data generation defaults ──────────────────────────────────────────────────
SAMPLE_INTERVAL_MIN = 15   # minutes between samples
RANDOM_SEED = 42
