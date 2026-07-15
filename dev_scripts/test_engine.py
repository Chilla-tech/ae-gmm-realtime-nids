"""Quick smoke test for the M3 inference engine.

Run from the repo root (or anywhere) with:
    python dev_scripts/test_engine.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine import M3InferenceEngine
from config import DEPLOY_DIR, BUFFER_DIR, BUFFER_MAX_SAMPLES, ATTACK_CONFIDENCE_THRESHOLD

print("Loading engine...")
e = M3InferenceEngine(
    deploy_dir=str(DEPLOY_DIR),
    buffer_dir=str(BUFFER_DIR),
    buffer_max=BUFFER_MAX_SAMPLES,
    attack_conf_threshold=ATTACK_CONFIDENCE_THRESHOLD,
)
print(f"Model     : {e.model_id}")
print(f"Features  : {len(e.features)}")
print(f"Threshold : {e.threshold:.4f}")
print(f"SHAP cache: {e.shap_cache['n_samples']} samples")
print(f"Buffers   : {e.buffer_stats}")
print(f"Feedback  : {e.feedback_count}")

# Test submit_feedback
import pandas as pd, numpy as np
fake_row = pd.Series(
    {"prediction": "Attack", "confidence": 0.85, "gmm_score": 20.0,
     "timestamp": "test", **{f: 0.0 for f in e.features}}
)
dest = e.submit_feedback(fake_row, "Normal")
print(f"\nFeedback test: Attack→Normal override → routed to '{dest}' buffer")
print(f"Feedback count after: {e.feedback_count}")

print("\n✓ All checks passed.")
