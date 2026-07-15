"""
M3 AE-GMM IDS — Deployment Configuration
Edit these settings before first run.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent              # repo root (this file's directory)
DEPLOY_DIR   = PROJECT_ROOT / 'trained_models' / 'M3_deploy_20260602_105838'
BUFFER_DIR   = PROJECT_ROOT / 'buffers'
WATCH_DIR    = PROJECT_ROOT / 'incoming'

# ── CICFlowMeter (optional / legacy) ──────────────────────
# Only needed if you use capture.py's CICFlowMeterRunner / LiveCapturePipeline
# (the Java-based capture path). The default live-capture path uses
# flow_extractor.py (pure Python, no external dependency) and does not
# require this directory to exist.
CICFLOWMETER_DIR = PROJECT_ROOT.parent / 'CICFlowMeter'

# ── Buffer settings ───────────────────────────────────────
BUFFER_MAX_SAMPLES          = 500_000   # rotate after this many rows per buffer
ATTACK_CONFIDENCE_THRESHOLD = 0.60      # only buffer attacks with confidence >= this

# ── Display ───────────────────────────────────────────────
SHAP_MAX_DISPLAY = 15                   # max features in SHAP waterfall plots
