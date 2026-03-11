# Solar PV Causal Fault Diagnosis

Streamlit application for solar photovoltaic fault diagnosis using synthetic or uploaded sensor data, causal discovery, causal inference, and deep learning.

## What the app does

- Upload PV sensor CSV data or generate a realistic synthetic dataset.
- Discover causal relationships between PV sensors with PCMCI or a lagged-correlation fallback.
- Train a sequence model for fault classification.
- Explain predictions with feature attribution, causal chains, and counterfactual views.
- Monitor system health, fault frequency, and estimated energy loss in a dashboard.

## Main entrypoint

- Streamlit app file: `app.py`

## Project structure

- `app.py`: Home page and high-level navigation.
- `pages/`: Streamlit multipage workflow.
- `data/data_generator.py`: Synthetic PV dataset generation.
- `models/`: Causal discovery, causal inference, and deep learning modules.
- `utils/`: Preprocessing, metrics, visualization, and theme helpers.
- `config.py`: Shared constants, labels, and model defaults.

## Run locally

1. Create and activate a Python environment.
2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

3. Start the app:

	```bash
	streamlit run app.py
	```

## Deploy to Streamlit Community Cloud

This repository is prepared for Streamlit Community Cloud:

- `requirements.txt` is in the repo root.
- `.streamlit/config.toml` is included.
- The app entrypoint is `app.py`.

### Deployment steps

1. Push this repository to GitHub.
2. Go to `https://share.streamlit.io/` and sign in with GitHub.
3. Click `Create app`.
4. Select this repository and branch.
5. Set the entrypoint file to `app.py`.
6. Open `Advanced settings` and choose Python `3.12`.
7. Click `Deploy`.

### Notes

- The app does not require secrets for basic use.
- Synthetic data generation works out of the box, so you can deploy without bundling a sample dataset.
- If the first build fails after dependency changes, use the Streamlit Cloud app settings to reboot or clear the build cache and redeploy.

## Expected input data

If you upload your own CSV, it should contain these columns:

- `timestamp`
- `irradiance`
- `module_temp`
- `ambient_temp`
- `wind_speed`
- `dc_voltage`
- `dc_current`
- `dc_power`
- `efficiency`
- `string_current_imbalance`
- `fault_label`

## Fault classes

- `0`: Normal
- `1`: Partial Shading
- `2`: Soiling
- `3`: Hot Spot
- `4`: PID Effect
- `5`: Bypass Diode Failure
- `6`: String Disconnect