"""
M3 AE-GMM IDS — Deployment Configuration
Edit these settings before first run.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent              # repo root (this file's directory)

# Model directory: check for a .current pointer (written after adaptation),
# fall back to the original pre-trained model if none exists.
_MODELS_DIR = PROJECT_ROOT / 'trained_models'
_CURRENT_PTR = _MODELS_DIR / '.current'
_DEFAULT_MODEL = 'M3_deploy_20260602_105838'

if _CURRENT_PTR.exists():
    _model_name = _CURRENT_PTR.read_text(encoding='utf-8').strip()
    if (_MODELS_DIR / _model_name).exists():
        DEPLOY_DIR = _MODELS_DIR / _model_name
    else:
        DEPLOY_DIR = _MODELS_DIR / _DEFAULT_MODEL
else:
    DEPLOY_DIR = _MODELS_DIR / _DEFAULT_MODEL

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
