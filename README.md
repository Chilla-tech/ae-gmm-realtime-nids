# AE-GMM Real-Time NIDS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](#)

A standalone **desktop application** for real-time network intrusion detection,
built on a hybrid **Autoencoder + Gaussian Mixture Model (AE-GMM)** with
two-level **SHAP explainability** and **incremental drift adaptation**
(MAS+Replay).

This repository is the deployment companion to the research project
*"AE-GMM: A Hybrid, Interpretable Approach for Robust Network Intrusion
Detection"* — see [CITATION.md](CITATION.md).

---

## Features

- **Live traffic capture** — pure-Python real-time flow feature extraction
  (`flow_extractor.py`, scapy-based). No external Java/CICFlowMeter
  dependency required for the default capture path.
- **Whole-network monitoring** — supports SPAN/mirror-port setups with an
  optional BPF subnet filter, so the app can monitor an entire LAN segment
  from a single monitoring host (not just traffic to/from the host itself).
- **Interpretable predictions** — two-level SHAP explanations:
  - Level 1: which features drive the AE reconstruction error
  - Level 2: which features drive the GMM anomaly score
- **Human-in-the-loop feedback** — right-click any flow to confirm/correct
  its label; corrected samples feed the adaptation pipeline.
- **Environment adaptation** — two-phase workflow for deploying to a new
  network:
  1. **Initial adaptation** (first deployment): fits a new scaler, trains a
     new AE and GMM from scratch on the new environment's traffic, and lets
     you pick an anomaly threshold from the live score distribution.
  2. **Incremental adaptation** (drift over time): uses a **MAS
     (Memory-Aware Synapses) + Replay** hybrid — the same technique used
     to train the model across CIC-IDS2018 → CIC-IDS2017 → USB-IDS →
     Local-Malware domains — so the model adapts to drift without
     forgetting previously learned attack patterns. Includes
     accept/rollback so a bad adaptation never corrupts the running model.
- **CSV import** — load CICFlowMeter-style CSVs for offline analysis.

## Model Summary (M3)

| | |
|---|---|
| Architecture | `Autoencoder(input_dim=18, encoder_layers=[64,48,24,18], latent_dim=16)` + BatchNorm, L1 reg, MAE loss |
| Anomaly scoring | `GaussianMixture(n_components=21, covariance_type='full')` on reconstruction-error vectors |
| Default threshold | 39.9265 (log-probability) |
| Training regime | Incremental across 4 domains (CIC-IDS2018 → CIC-IDS2017 → USB-IDS → Local-Malware) via MAS+Replay (λ=50, replay ratio=0.4) |
| Features (18) | `FWD Init Win Bytes, Fwd IAT Total, Protocol, Fwd Seg Size Min, Bwd Packet Length Mean, Flow IAT Max, Packet Length Max, Flow IAT Std, Fwd Packet Length Std, Idle Max, Packet Length Mean, Bwd Packet Length Max, Bwd Packet Length Std, Fwd Packet Length Mean, Packet Length Min, Dst Port, FIN Flag Count, Bwd IAT Std` |

## Screenshots

_Add screenshots of the desktop app (flow table, SHAP panel, threshold picker) here._

---

## Repository Structure

```
ae-gmm-realtime-nids/
├── README.md
├── LICENSE
├── CITATION.md
├── requirements.txt          # pinned dependencies
├── INSTALL.bat                # one-time setup (conda env + packages)
├── LAUNCH.bat                 # start the desktop app
├── run_dashboard.bat           # (optional) start the legacy Streamlit dashboard
├── desktop_app.py              # PRIMARY app — PySide6 desktop GUI
├── app.py                      # legacy Streamlit dashboard (alternative UI)
├── engine.py                   # M3InferenceEngine — model loading, predict(), explain()
├── flow_extractor.py           # real-time flow feature extractor (scapy)
├── capture.py                  # interface listing + legacy CICFlowMeter pipeline
├── config.py                   # paths & thresholds — edit before first run
├── retrain.py                  # legacy naive fine-tune adaptation (superseded by incremental.py)
├── incremental.py              # MAS+Replay incremental adaptation + from-scratch initial adaptation
├── utils/
│   └── match_to_ids2018.py     # CSV column standardization for CICFlowMeter variants
├── trained_models/
│   └── M3_deploy_20260602_105838/   # self-contained model package (~1.2 MB)
│       ├── aegmm_model_package.joblib
│       ├── ae_M3.keras
│       ├── ae_shap_explainer.joblib
│       ├── gmm_shap_explainer.joblib
│       ├── shap_precomputed_2000.joblib
│       └── manifest.json
├── buffers/                     # runtime sample buffers (created at runtime)
└── dev_scripts/                 # manual smoke-test / debug scripts (not required for normal use)
```

---

## Installation (Windows)

### Prerequisites

1. **Miniconda** — <https://docs.conda.io/en/latest/miniconda.html>
2. **Npcap** — <https://npcap.com/#download> — tick **"WinPcap API-compatible mode"**
   during install (required for live packet capture via scapy)

### Setup

```powershell
git clone <this-repo-url>
cd ae-gmm-realtime-nids
INSTALL.bat
```

`INSTALL.bat` creates a conda environment named `aegmm_ids` (Python 3.10) and
installs all pinned dependencies from `requirements.txt`.

### Run

```powershell
LAUNCH.bat
```

or manually:

```powershell
conda activate aegmm_ids
python desktop_app.py
```

> **Note on VS Code terminals:** if your workspace has a `.venv` that VS Code
> auto-activates, it will shadow the conda environment. Either open a plain
> terminal outside VS Code, or set `"python.terminal.activateEnvironment": false`
> in `.vscode/settings.json` and manually run `conda activate aegmm_ids`.

---

## Usage

### 1. Live capture

1. Select a network interface from the toolbar dropdown (interfaces with no
   IP address are shown as `[MONITOR]` — useful for SPAN/mirror ports).
2. Optionally set a subnet filter (e.g. `10.10.10.0/24`) to restrict capture
   to a specific network segment when using a SPAN port that sees multiple
   segments.
3. Click **Start**. Flows populate the table in real time, color-coded by
   prediction (Normal / Attack).
4. Right-click any row to view its SHAP explanation or submit human feedback.

### 2. Whole-network monitoring via SPAN/mirror port

To monitor an entire LAN segment (not just the host running the app), mirror
traffic from your router/firewall (e.g. pfSense) to a dedicated interface on
the monitoring host, then select that `[MONITOR]` interface with a subnet
filter. See the inline tooltip on the Subnet field for details.

### 3. Adapting to a new environment

Click **Adapt Model** after capturing a representative sample of traffic:

- **First time** (no baseline yet): runs the *initial adaptation* — trains a
  new scaler/AE/GMM from scratch, treating all captured traffic as normal
  baseline. You'll be shown the score distribution to pick a threshold.
- **Subsequent times** (drift adaptation): runs the *incremental MAS+Replay
  adaptation* — fine-tunes the existing AE with a Memory-Aware-Synapses
  penalty while replaying a buffer of previously accepted traffic, then
  refits the GMM. You review the new score distribution and either **accept**
  (swap in the updated model) or the run is **rolled back** automatically if
  you cancel.

### 4. CSV import

Use **Open CSV** to load a CICFlowMeter-format CSV for offline batch analysis
(column names are auto-standardized via `utils/match_to_ids2018.py`).

---

## Known Issues / Notes

- **PySide6 DLL load errors on Windows**: caused by a version mismatch
  between `PySide6-Essentials` and `shiboken6`. This repo pins both to
  `6.7.3` in `requirements.txt`, which resolves the issue observed with
  `6.11.x`. If you still see `DLL load failed while importing QtWidgets`,
  reinstall both packages together:
  ```
  pip install --force-reinstall --no-deps --ignore-installed shiboken6==6.7.3 PySide6-Essentials==6.7.3
  ```
- **CICFlowMeter (Java) path** (`capture.py`'s `CICFlowMeterRunner` /
  `LiveCapturePipeline`) is legacy and **not used** by the default capture
  path. It requires a separate Java-based CICFlowMeter project as a sibling
  directory and is only exercised by `dev_scripts/test_capture.py` /
  `test_e2e.py`.
- The **incremental drift-injection test harness** (used to validate the
  MAS+Replay adaptation loop under synthetic drift) is part of ongoing work
  and not included in this initial release.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

See [CITATION.md](CITATION.md).
