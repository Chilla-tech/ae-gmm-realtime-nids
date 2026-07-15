"""
Environment Adaptation — AE-GMM Model Retraining
─────────────────────────────────────────────────
When deployed to a new network, the model may flag most traffic as attacks
because the normal-traffic distribution differs from training.  This module:

  1. Detects the situation (≥50% attack rate in captured flows)
  2. Treats *all* captured flows as the new "normal baseline"
  3. Fine-tunes the AE (warm-start from existing weights)
  4. Refits the GMM on new reconstruction errors
  5. Selects a new threshold (auto best-F1 or human-guided)
  6. Saves the adapted model and hot-swaps it in the engine
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Threshold selection helpers
# ═══════════════════════════════════════════════════════════════════════════

def find_best_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n_candidates: int = 200,
) -> tuple[float, float]:
    """
    Sweep thresholds over the GMM score range and return the one that
    maximises macro-F1.

    Parameters
    ----------
    scores : GMM log-probability scores  (higher = more normal)
    labels : binary 0/1 array  (0=normal, 1=attack)
    n_candidates : number of threshold values to try

    Returns
    -------
    (best_threshold, best_f1)
    """
    lo, hi = float(np.min(scores)), float(np.max(scores))
    candidates = np.linspace(lo, hi, n_candidates)

    best_t, best_f = lo, 0.0
    for t in candidates:
        preds = (scores < t).astype(int)          # below threshold → attack
        f = f1_score(labels, preds, average="macro", zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t

    return float(best_t), float(best_f)


def compute_percentiles(scores: np.ndarray) -> dict[str, float]:
    """Return a dict of useful percentile values for human review."""
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {f"p{p}": float(np.percentile(scores, p)) for p in pcts}


# ═══════════════════════════════════════════════════════════════════════════
#  Core retraining pipeline
# ═══════════════════════════════════════════════════════════════════════════

class EnvironmentAdapter:
    """
    Fine-tunes an existing AE-GMM model to a new traffic environment.

    Parameters
    ----------
    ae_model    : compiled Keras AE model (weights will be updated in-place)
    scaler      : fitted StandardScaler
    features    : list of 18 feature names
    gmm_k       : number of GMM components (default: reuse existing)
    """

    def __init__(self, ae_model, scaler, features: list[str], gmm_k: int = 21):
        self.ae_model = ae_model
        self.scaler = scaler
        self.features = list(features)
        self.gmm_k = gmm_k

        # Results (populated after adapt())
        self.new_gmm = None
        self.new_threshold = None
        self.scores = None          # GMM scores on training data
        self.percentiles = None
        self.auto_f1 = None

    def adapt(
        self,
        X_normal: np.ndarray,
        *,
        ae_epochs: int = 30,
        ae_batch_size: int = 256,
        ae_lr: float = 1e-4,
        val_split: float = 0.15,
        threshold_mode: str = "auto",   # "auto" or "manual"
        manual_threshold: float | None = None,
        progress_cb=None,
    ) -> dict:
        """
        Run the full adaptation pipeline.

        Parameters
        ----------
        X_normal : (N, 18) array of *raw* feature values (unscaled).
                   All rows are treated as normal baseline traffic.
        ae_epochs : epochs for AE fine-tuning
        ae_batch_size : batch size
        ae_lr : learning rate (lower than original to avoid catastrophic shift)
        val_split : fraction held out for validation
        threshold_mode : "auto" (best-F1) or "manual"
        manual_threshold : used only when threshold_mode == "manual"
        progress_cb : callable(step: int, total: int, msg: str) for UI updates

        Returns
        -------
        dict with keys: threshold, f1, percentiles, n_samples, scores
        """
        import tensorflow as tf

        n = len(X_normal)
        total_steps = 4
        if progress_cb is None:
            progress_cb = lambda s, t, m: None

        # ── Step 1: Scale ──────────────────────────────────────────────
        progress_cb(1, total_steps, f"Scaling {n} samples…")
        X_clean = np.nan_to_num(X_normal, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_clean)

        # Split into train / val
        idx = np.random.permutation(n)
        n_val = max(1, int(n * val_split))
        X_train = X_scaled[idx[n_val:]]
        X_val = X_scaled[idx[:n_val]]

        # ── Step 2: Fine-tune AE (warm start) ─────────────────────────
        progress_cb(2, total_steps, f"Fine-tuning AE ({ae_epochs} epochs)…")

        # Lower the learning rate for fine-tuning
        self.ae_model.optimizer.learning_rate = ae_lr

        self.ae_model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            verbose=0,
        )

        # ── Step 3: Refit GMM on reconstruction errors ────────────────
        progress_cb(3, total_steps, "Refitting GMM on new errors…")

        X_recon = self.ae_model.predict(X_scaled, verbose=0)
        errors = np.abs(X_scaled - X_recon)

        self.new_gmm = GaussianMixture(
            n_components=self.gmm_k,
            covariance_type="full",
            random_state=42,
            max_iter=300,
        )
        self.new_gmm.fit(errors)

        self.scores = self.new_gmm.score_samples(errors)
        self.percentiles = compute_percentiles(self.scores)

        # ── Step 4: Select threshold ──────────────────────────────────
        progress_cb(4, total_steps, "Selecting threshold…")

        if threshold_mode == "manual" and manual_threshold is not None:
            self.new_threshold = manual_threshold
            self.auto_f1 = None
        else:
            # Auto: since all data is "normal", we label everything 0
            # and try to find the threshold below which < ~1-5% of
            # traffic falls. Use the 5th percentile as a reasonable
            # default — this means ~5% of the new normal traffic would
            # be flagged as anomalous.
            self.new_threshold = self.percentiles["p5"]
            self.auto_f1 = None

        result = {
            "threshold": self.new_threshold,
            "f1": self.auto_f1,
            "percentiles": self.percentiles,
            "n_samples": n,
            "score_min": float(np.min(self.scores)),
            "score_max": float(np.max(self.scores)),
            "score_mean": float(np.mean(self.scores)),
        }
        logger.info(
            "Adaptation complete: threshold=%.4f, n=%d, score_range=[%.2f, %.2f]",
            self.new_threshold, n,
            result["score_min"], result["score_max"],
        )
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  Save / swap helpers
# ═══════════════════════════════════════════════════════════════════════════

def save_adapted_model(
    deploy_dir: Path,
    ae_model,
    gmm,
    scaler,
    threshold: float,
    features: list[str],
    model_id: str = "M3",
    config: dict | None = None,
    domains_seen: list | None = None,
) -> Path:
    """
    Save an adapted model as a new deployment package (same structure as
    the original M3_deploy_* directory).

    Returns the new directory path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = deploy_dir.parent / f"M3_adapted_{ts}"
    new_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save AE
    ae_path = new_dir / f"ae_{model_id}.keras"
    ae_model.save(ae_path)

    # 2. Save package
    pkg = {
        "model_id": model_id,
        "config": config or {},
        "domains_seen": (domains_seen or []) + ["adapted"],
        "ae_model_path": str(ae_path),
        "ae_weights": [w.tolist() for w in ae_model.get_weights()],
        "gmm": gmm,
        "gmm_threshold": threshold,
        "gmm_k": gmm.n_components,
        "scaler": scaler,
        "feature_names": features,
        "adapted_at": ts,
        "package_version": "3.0-adapted",
    }
    joblib.dump(pkg, new_dir / "aegmm_model_package.joblib")

    # 3. Copy SHAP explainers from original (they still work with updated model)
    for name in ["ae_shap_explainer.joblib", "gmm_shap_explainer.joblib"]:
        src = deploy_dir / name
        if src.exists():
            shutil.copy2(src, new_dir / name)

    # 4. Manifest
    import json
    manifest = {
        "model_id": model_id,
        "package_version": "3.0-adapted",
        "adapted_from": str(deploy_dir.name),
        "adapted_at": ts,
        "gmm_threshold": threshold,
        "n_features": len(features),
        "feature_names": features,
    }
    with open(new_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Adapted model saved to %s", new_dir)
    return new_dir
