"""
Synthetic Solar PV fault data generator.
"""

import numpy as np
import pandas as pd
from pathlib import Path

FAULT_LABELS = {
    0: "Normal",
    1: "Partial Shading",
    2: "Soiling",
    3: "Hot Spot",
    4: "PID Effect",
    5: "Bypass Diode Failure",
    6: "String Disconnect",
}

# Fault injection probabilities (during daytime), must sum < 1
_FAULT_PROBS = [0.70, 0.07, 0.07, 0.04, 0.04, 0.04, 0.04]


def _irradiance(hour: float, day_of_year: int) -> float:
    """Clear-sky irradiance model (W/m²)."""
    season = np.cos(2 * np.pi * (day_of_year - 172) / 365)  # peak at summer solstice
    peak = 950 + 50 * season
    if hour < 6 or hour > 20:
        return 0.0
    solar_noon = 12.5 - 0.5 * season
    irr = peak * np.exp(-((hour - solar_noon) ** 2) / 18)
    return max(0.0, irr)


def generate_pv_data(n_days: int = 365, interval_min: int = 15,
                     seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic PV dataset.

    Parameters
    ----------
    n_days       : simulation length in days
    interval_min : sampling interval (minutes)
    seed         : random seed

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, irradiance, module_temp, ambient_temp, wind_speed,
        dc_voltage, dc_current, dc_power, efficiency,
        string_current_imbalance, fault_label
    """
    rng = np.random.default_rng(seed)
    steps_per_day = 24 * 60 // interval_min
    n_steps = n_days * steps_per_day

    start = pd.Timestamp("2023-01-01")
    timestamps = pd.date_range(start, periods=n_steps, freq=f"{interval_min}min")

    records = []
    fault_state = 0
    fault_duration = 0

    for i, ts in enumerate(timestamps):
        hour = ts.hour + ts.minute / 60.0
        doy = ts.day_of_year

        # ── Environmental conditions ─────────────────────────────────────────
        irr = _irradiance(hour, doy) * (1 + rng.normal(0, 0.03))
        irr = max(0.0, irr)

        season_temp = 15 + 12 * np.cos(2 * np.pi * (doy - 200) / 365)
        ambient_temp = season_temp + 5 * np.sin(np.pi * hour / 12) + rng.normal(0, 1)
        wind_speed = max(0.0, rng.weibull(2) * 4 + rng.normal(0, 0.5))

        # Module temp = ambient + irradiance heating – wind cooling
        module_temp = ambient_temp + 0.03 * irr - 0.5 * wind_speed + rng.normal(0, 1)

        # ── Fault state machine ───────────────────────────────────────────────
        if fault_duration > 0:
            fault_duration -= 1
        else:
            if irr > 50:
                fault_state = int(rng.choice(len(_FAULT_PROBS), p=_FAULT_PROBS))
                fault_duration = int(rng.integers(4, 24))   # 1–6 hours
            else:
                fault_state = 0

        # ── Electrical model ──────────────────────────────────────────────────
        # Base values at STC + irradiance scaling
        irr_ratio = irr / 1000.0
        voc_base = 40.0 - 0.12 * (module_temp - 25)  # temperature coefficient
        isc_base = 8.5 * irr_ratio

        eff_base = 0.18  # 18% nominal efficiency

        # Fault modifiers
        if fault_state == 0:    # Normal
            v_mod, i_mod, eff_mod, imbalance = 1.0, 1.0, 1.0, 0.01
        elif fault_state == 1:  # Partial Shading
            v_mod, i_mod, eff_mod = 0.85, 0.70, 0.80
            imbalance = rng.uniform(0.15, 0.35)
        elif fault_state == 2:  # Soiling
            v_mod, i_mod, eff_mod = 0.96, 0.85, 0.85
            imbalance = rng.uniform(0.02, 0.08)
        elif fault_state == 3:  # Hot Spot
            v_mod, i_mod, eff_mod = 0.80, 0.90, 0.70
            imbalance = rng.uniform(0.10, 0.25)
            module_temp += rng.uniform(10, 25)   # localised hot spot
        elif fault_state == 4:  # PID Effect
            v_mod, i_mod, eff_mod = 0.75, 0.95, 0.75
            imbalance = rng.uniform(0.08, 0.20)
        elif fault_state == 5:  # Bypass Diode Failure
            v_mod, i_mod, eff_mod = 0.65, 0.60, 0.60
            imbalance = rng.uniform(0.25, 0.50)
        else:                   # String Disconnect (6)
            v_mod, i_mod, eff_mod = 0.50, 0.0, 0.0
            imbalance = rng.uniform(0.40, 0.80)

        dc_voltage = voc_base * v_mod * (1 + rng.normal(0, 0.01))
        dc_current = isc_base * i_mod * (1 + rng.normal(0, 0.01))
        dc_power = dc_voltage * dc_current
        efficiency = eff_base * eff_mod * (1 + rng.normal(0, 0.005))
        efficiency = max(0.0, min(1.0, efficiency))
        imbalance = max(0.0, imbalance + rng.normal(0, 0.005))

        records.append({
            "timestamp": ts,
            "irradiance": round(irr, 2),
            "module_temp": round(module_temp, 2),
            "ambient_temp": round(ambient_temp, 2),
            "wind_speed": round(wind_speed, 2),
            "dc_voltage": round(dc_voltage, 3),
            "dc_current": round(dc_current, 3),
            "dc_power": round(dc_power, 2),
            "efficiency": round(efficiency, 4),
            "string_current_imbalance": round(imbalance, 4),
            "fault_label": fault_state,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "sample_pv_data.csv"
    print(f"Generating synthetic PV data → {out_path}")
    df = generate_pv_data(n_days=365, interval_min=15, seed=42)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows.  Fault distribution:")
    print(df["fault_label"].value_counts().sort_index())
