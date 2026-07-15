"""
M3 AE-GMM Inference Engine
──────────────────────────
Production inference engine for the M3 AE-GMM intrusion detection model.
Loads a deployment package produced by M3_Inference_Demo.ipynb and provides:

  • Batch prediction with confidence scoring
  • Two-level SHAP explanations (AE reconstruction + GMM anomaly)
  • Rotating CSV buffers for collecting benign / high-confidence attack samples
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import shap
from scipy.special import expit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  PicklableShapWrapper — matches the class pickled in notebook explainers
# ---------------------------------------------------------------------------

def _ae_reconstruction_fn(X, model_components):
    """AE reconstruction error for SHAP."""
    ae = model_components["autoencoder"]
    scaler = model_components["scaler"]
    features = model_components["feature_names"]
    X = X[list(features)]
    X_sc = scaler.transform(pd.DataFrame(X, columns=features) if not isinstance(X, pd.DataFrame) else X)
    recon = ae.predict(X_sc, verbose=0)
    return np.mean(np.abs(X_sc - recon), axis=1)


def _gmm_score_fn(X, model_components):
    """GMM anomaly score for SHAP."""
    ae = model_components["autoencoder"]
    gmm = model_components["gmm"]
    scaler = model_components["scaler"]
    features = model_components["feature_names"]
    X = X[list(features)]
    X_sc = scaler.transform(pd.DataFrame(X, columns=features) if not isinstance(X, pd.DataFrame) else X)
    recon = ae.predict(X_sc, verbose=0)
    errors = np.abs(X_sc - recon)
    return gmm.score_samples(errors)


class _PicklableShapWrapper:
    """Mirror of the notebook-defined PicklableShapWrapper so joblib can unpickle SHAP explainers."""

    def __init__(self, function_name, model_components):
        self.function_name = function_name
        self.model_components = model_components

    def __call__(self, X):
        if self.function_name == "ae_reconstruction":
            return _ae_reconstruction_fn(X, self.model_components)
        elif self.function_name == "gmm_score":
            return _gmm_score_fn(X, self.model_components)
        else:
            raise ValueError(f"Unknown function: {self.function_name}")


# Alias so pickle resolves the original class name
PicklableShapWrapper = _PicklableShapWrapper

# ---------------------------------------------------------------------------
#  Lazy TensorFlow import (heavy, only load once)
# ---------------------------------------------------------------------------
_tf = None


def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


# ---------------------------------------------------------------------------
#  Column standardisation helper
# ---------------------------------------------------------------------------
_standardize_fn = None


def _get_standardize():
    """Lazily import standardize_to_ids2018 from the project utils."""
    global _standardize_fn
    if _standardize_fn is not None:
        return _standardize_fn
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from utils.match_to_ids2018 import standardize_to_ids2018
        _standardize_fn = standardize_to_ids2018
    except ImportError:
        _standardize_fn = lambda df: df          # no-op fallback
        logger.warning("match_to_ids2018 not found — skipping column standardisation")
    return _standardize_fn


# ═══════════════════════════════════════════════════════════════════════════
#  SampleBuffer — CSV-backed rotating buffer for retraining data collection
# ═══════════════════════════════════════════════════════════════════════════

class SampleBuffer:
    """
    Append-only CSV buffer with rotation.

    Parameters
    ----------
    name : str          Buffer identifier (e.g. ``'benign'``, ``'attack'``).
    buffer_dir : str    Directory where the CSV file lives.
    max_samples : int   Maximum rows before oldest are dropped.
    """

    def __init__(self, name: str, buffer_dir: str, max_samples: int = 500_000):
        self.name = name
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.buffer_dir / f"{name}_buffer.csv"
        self.max_samples = max_samples

    # -- properties ----------------------------------------------------------

    @property
    def count(self) -> int:
        if not self.csv_path.exists():
            return 0
        # fast line count
        with open(self.csv_path, "r", encoding="utf-8") as fh:
            return max(0, sum(1 for _ in fh) - 1)

    # -- public API ----------------------------------------------------------

    def add(self, df: pd.DataFrame) -> None:
        """Append *df* rows to the CSV buffer, rotating if over capacity."""
        if df.empty:
            return
        header = not self.csv_path.exists()
        df.to_csv(self.csv_path, mode="a", header=header, index=False)
        if self.count > self.max_samples:
            self._rotate()

    def load(self) -> pd.DataFrame:
        """Return buffer contents as a DataFrame."""
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame()

    def clear(self) -> None:
        """Delete the buffer CSV."""
        if self.csv_path.exists():
            self.csv_path.unlink()

    # -- internals -----------------------------------------------------------

    def _rotate(self):
        full = pd.read_csv(self.csv_path)
        if len(full) > self.max_samples:
            full.tail(self.max_samples).to_csv(self.csv_path, index=False)
            logger.info("%s buffer rotated → kept last %d rows", self.name, self.max_samples)


# ═══════════════════════════════════════════════════════════════════════════
#  M3InferenceEngine
# ═══════════════════════════════════════════════════════════════════════════

class M3InferenceEngine:
    """
    End-to-end inference engine that wraps the M3 deployment package.

    Usage
    -----
    >>> from engine import M3InferenceEngine
    >>> engine = M3InferenceEngine("trained_models/M3_deploy_20260602_105838")
    >>> results = engine.predict(cicflowmeter_df)   # DataFrame of flows
    >>> shap_ex = engine.explain(results.iloc[0])    # per-sample SHAP
    """

    def __init__(
        self,
        deploy_dir: str,
        buffer_dir: str | None = None,
        buffer_max: int = 500_000,
        attack_conf_threshold: float = 0.60,
    ):
        self.deploy_dir = Path(deploy_dir)
        self.attack_conf_threshold = attack_conf_threshold

        self._load_model()
        self._load_shap()

        buf = Path(buffer_dir) if buffer_dir else self.deploy_dir.parent / "buffers"
        self.benign_buffer = SampleBuffer("benign", str(buf), buffer_max)
        self.attack_buffer = SampleBuffer("attack", str(buf), buffer_max)

    # ── model loading ──────────────────────────────────────────────────────

    def _load_model(self):
        tf = _get_tf()
        pkg = joblib.load(self.deploy_dir / "aegmm_model_package.joblib")
        self.ae_model = tf.keras.models.load_model(
            self.deploy_dir / f"ae_{pkg['model_id']}.keras"
        )
        self.gmm = pkg["gmm"]
        self.scaler = pkg["scaler"]
        self.threshold = pkg["gmm_threshold"]
        self.features = list(pkg["feature_names"])
        self.model_id = pkg["model_id"]
        self.config = pkg.get("config", {})
        self.domains_seen = pkg.get("domains_seen", [])
        logger.info(
            "Model %s loaded — %d features, threshold %.4f",
            self.model_id, len(self.features), self.threshold,
        )

    def _load_shap(self):
        # The SHAP explainers were pickled from a notebook where
        # PicklableShapWrapper lived in __main__.  Make the class
        # resolvable during unpickling by injecting it into __main__.
        import __main__ as _main
        if not hasattr(_main, "PicklableShapWrapper"):
            _main.PicklableShapWrapper = _PicklableShapWrapper

        # pre-computed 2 000-sample cache
        cache_path = self.deploy_dir / "shap_precomputed_2000.joblib"
        self.shap_cache = joblib.load(cache_path) if cache_path.exists() else None

        # on-demand explainers (for new unseen samples)
        ae_p = self.deploy_dir / "ae_shap_explainer.joblib"
        gmm_p = self.deploy_dir / "gmm_shap_explainer.joblib"
        self.ae_explainer = joblib.load(ae_p) if ae_p.exists() else None
        self.gmm_explainer = joblib.load(gmm_p) if gmm_p.exists() else None

    # ── prediction ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame, *, standardize: bool = True) -> pd.DataFrame:
        """
        Run AE-GMM inference on CICFlowMeter output.

        Parameters
        ----------
        df : DataFrame
            Raw CICFlowMeter CSV (≥18 model features).
        standardize : bool
            Run column-name canonicalisation (recommended).

        Returns
        -------
        DataFrame with columns:
            prediction, confidence, gmm_score, timestamp,
            [Src IP, Dst IP, …  if present in input],
            <18 model features>
        """
        work = df.copy()

        if standardize:
            work = _get_standardize()(work)

        missing = [f for f in self.features if f not in work.columns]
        if missing:
            raise ValueError(f"Missing features in input: {missing}")

        # optional metadata
        META_COLS = ["Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp"]
        meta = {c: work[c].values for c in META_COLS if c in work.columns}

        # feature matrix
        X = work[self.features].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # AE → reconstruction error → GMM score
        X_scaled = self.scaler.transform(X)
        X_recon = self.ae_model.predict(X_scaled, verbose=0)
        errors = np.abs(X_scaled - X_recon)
        scores = self.gmm.score_samples(errors)

        # classify
        predictions = np.where(scores < self.threshold, "Attack", "Normal")

        # confidence: sigmoid-mapped |distance from threshold| → [0, 1]
        distance = scores - self.threshold
        pctl95 = np.percentile(np.abs(distance), 95) or 1.0
        confidences = (expit(np.abs(distance / (pctl95 / 3))) - 0.5) * 2

        # assemble result
        results = pd.DataFrame(
            {
                "prediction": predictions,
                "confidence": np.round(confidences, 4),
                "gmm_score": np.round(scores, 4),
                "timestamp": datetime.now().isoformat(),
            }
        )
        for col, vals in meta.items():
            results[col] = vals

        feat_df = pd.DataFrame(X, columns=self.features)
        results = pd.concat([results, feat_df], axis=1)

        # route to buffers
        self._buffer(results)

        return results

    # ── buffering ──────────────────────────────────────────────────────────

    def _buffer(self, results: pd.DataFrame):
        benign = results[results["prediction"] == "Normal"]
        attack = results[
            (results["prediction"] == "Attack")
            & (results["confidence"] >= self.attack_conf_threshold)
        ]
        if not benign.empty:
            self.benign_buffer.add(benign)
        if not attack.empty:
            self.attack_buffer.add(attack)

    # ── human feedback ─────────────────────────────────────────────────────

    def submit_feedback(
        self,
        sample: pd.Series,
        human_label: str,
    ) -> str:
        """
        Route a sample to the correct buffer based on a human-provided label.

        Parameters
        ----------
        sample : Series
            A row from a ``predict()`` result DataFrame (must contain the
            18 model features plus ``prediction``, ``confidence``, etc.).
        human_label : str
            The corrected label from the human reviewer: ``'Normal'`` or
            ``'Attack'``.

        Returns
        -------
        str  Which buffer the sample was sent to (``'benign'`` or ``'attack'``).
        """
        row = sample.to_frame().T.copy()
        row["human_label"] = human_label
        row["feedback_ts"] = datetime.now().isoformat()

        # Log every feedback event
        self._log_feedback(row)

        # Route to buffer by the *human* label, not the model prediction
        if human_label == "Normal":
            self.benign_buffer.add(row)
            return "benign"
        else:
            self.attack_buffer.add(row)
            return "attack"

    def _log_feedback(self, row: pd.DataFrame):
        """Append to a persistent feedback-log CSV (for auditing)."""
        log_path = self.benign_buffer.buffer_dir / "feedback_log.csv"
        header = not log_path.exists()
        row.to_csv(log_path, mode="a", header=header, index=False)

    @property
    def feedback_count(self) -> int:
        log_path = self.benign_buffer.buffer_dir / "feedback_log.csv"
        if not log_path.exists():
            return 0
        with open(log_path, "r", encoding="utf-8") as fh:
            return max(0, sum(1 for _ in fh) - 1)

    # ── SHAP explanations ─────────────────────────────────────────────────

    def explain(self, feature_values, *, level: str = "both") -> dict:
        """
        Compute SHAP explanation for **one** sample.

        Parameters
        ----------
        feature_values : Series | dict | array
            Raw (unscaled) feature values.
        level : {'ae', 'gmm', 'both'}

        Returns
        -------
        dict  with ``'ae'`` and/or ``'gmm'`` → ``shap.Explanation``
        """
        if isinstance(feature_values, dict):
            x = pd.DataFrame([feature_values])[self.features]
        elif isinstance(feature_values, pd.Series):
            x = pd.DataFrame(
                [feature_values[self.features].values], columns=self.features
            )
        else:
            x = pd.DataFrame([feature_values[: len(self.features)]], columns=self.features)

        out = {}
        if level in ("ae", "both") and self.ae_explainer is not None:
            out["ae"] = self.ae_explainer(x)[0]
        if level in ("gmm", "both") and self.gmm_explainer is not None:
            out["gmm"] = self.gmm_explainer(x)[0]
        return out

    def get_cached_explanation(self, idx: int) -> dict | None:
        """Retrieve a pre-computed SHAP explanation from the 2 000-sample cache."""
        if self.shap_cache is None or idx >= self.shap_cache["n_samples"]:
            return None
        c = self.shap_cache

        def _base(arr, i):
            return arr[i] if np.ndim(arr) > 0 else arr

        return {
            "ae": shap.Explanation(
                values=c["ae_shap_values"][idx],
                base_values=_base(c["ae_shap_base_values"], idx),
                data=c["X_raw"].iloc[idx].values,
                feature_names=list(c["features"]),
            ),
            "gmm": shap.Explanation(
                values=c["gmm_shap_values"][idx],
                base_values=_base(c["gmm_shap_base_values"], idx),
                data=c["X_raw"].iloc[idx].values,
                feature_names=list(c["features"]),
            ),
            "meta": c["meta"].iloc[idx].to_dict() if "meta" in c else {},
        }

    # ── convenience ────────────────────────────────────────────────────────

    @property
    def buffer_stats(self) -> dict:
        return {
            "benign_count": self.benign_buffer.count,
            "attack_count": self.attack_buffer.count,
            "buffer_max": self.benign_buffer.max_samples,
            "attack_conf_threshold": self.attack_conf_threshold,
        }

    # ── environment adaptation ─────────────────────────────────────────────

    def check_attack_rate(self, df: pd.DataFrame) -> float:
        """Return the fraction of rows predicted as Attack."""
        if df.empty or "prediction" not in df.columns:
            return 0.0
        return float((df["prediction"] == "Attack").mean())

    def adapt_to_environment(
        self,
        flow_data: pd.DataFrame,
        *,
        ae_epochs: int = 30,
        ae_lr: float = 1e-4,
        threshold_mode: str = "auto",
        manual_threshold: float | None = None,
        progress_cb=None,
    ) -> dict:
        """
        Fine-tune the model to a new network environment.

        All flows in *flow_data* are treated as the normal baseline.
        The AE is warm-started from current weights, then GMM is refit.

        Parameters
        ----------
        flow_data : DataFrame with the 18 model features (raw values).
        ae_epochs : fine-tuning epochs (default 30).
        ae_lr : fine-tuning learning rate (default 1e-4).
        threshold_mode : "auto" (5th percentile) or "manual".
        manual_threshold : value for manual mode.
        progress_cb : callable(step, total, msg) for UI.

        Returns
        -------
        dict with threshold, percentiles, n_samples, score stats.
        """
        from retrain import EnvironmentAdapter, save_adapted_model

        # Extract the feature matrix
        missing = [f for f in self.features if f not in flow_data.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X = flow_data[self.features].values.astype(np.float64)

        adapter = EnvironmentAdapter(
            ae_model=self.ae_model,
            scaler=self.scaler,
            features=self.features,
            gmm_k=self.gmm.n_components,
        )

        result = adapter.adapt(
            X,
            ae_epochs=ae_epochs,
            ae_lr=ae_lr,
            threshold_mode=threshold_mode,
            manual_threshold=manual_threshold,
            progress_cb=progress_cb,
        )

        # Save the adapted model
        new_dir = save_adapted_model(
            deploy_dir=self.deploy_dir,
            ae_model=self.ae_model,
            gmm=adapter.new_gmm,
            scaler=self.scaler,
            threshold=adapter.new_threshold,
            features=self.features,
            model_id=self.model_id,
            config=self.config,
            domains_seen=self.domains_seen,
        )

        # Hot-swap: update engine state in-place
        self.gmm = adapter.new_gmm
        self.threshold = adapter.new_threshold
        self.deploy_dir = new_dir

        # Invalidate SHAP cache (model weights changed)
        self.shap_cache = None

        result["adapted_dir"] = str(new_dir)
        logger.info(
            "Engine hot-swapped to adapted model at %s (threshold=%.4f)",
            new_dir.name, self.threshold,
        )
        return result
