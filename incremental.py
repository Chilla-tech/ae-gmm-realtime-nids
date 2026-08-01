"""
Incremental Learning — MAS+Replay Hybrid for Deployment
────────────────────────────────────────────────────────
Implements the same MAS+Replay hybrid approach used in offline
training (D1→D2→D3→D4) for live drift adaptation in the deployed app.

Two adaptation modes:
  1. Initial Environment Adaptation (from-scratch):
     - New scaler fitted on environment data
     - AE retrained from scratch
     - GMM fitted from scratch
     - Human selects threshold

  2. Incremental Drift Adaptation (MAS+Replay):
     - Checkpoint current models
     - Compute MAS importance (Ω) on replay buffer
     - Fine-tune AE with MAS penalty + replay mixing
     - Refit GMM on mixed reconstruction errors
     - Human evaluates and accepts/rejects
"""

import logging
import copy
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Weight utilities
# ═══════════════════════════════════════════════════════════════════════════

def store_model_weights(model) -> dict:
    """Snapshot all trainable weights keyed by '{var.name}_{idx}'."""
    return {f"{v.name}_{i}": v.numpy().copy()
            for i, v in enumerate(model.trainable_variables)}


def restore_model_weights(model, weight_dict: dict):
    """Restore weights from a snapshot dict."""
    for i, var in enumerate(model.trainable_variables):
        key = f"{var.name}_{i}"
        if key in weight_dict:
            var.assign(weight_dict[key])


# ═══════════════════════════════════════════════════════════════════════════
#  MAS Importance Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_mas_importance(model, data: np.ndarray,
                           num_samples: int = 10000,
                           normalize: bool = True) -> dict:
    """
    Compute Memory-Aware Synapses (MAS) parameter importance Ω.

    Ω_k = (1/N) Σ_i (∂||f(x_i)||² / ∂θ_k)²

    Parameters
    ----------
    model : Keras model (autoencoder)
    data : scaled input data (N, D)
    num_samples : cap on samples used for computation
    normalize : if True, normalize Ω to [0, 1] via global max

    Returns
    -------
    omega_dict : dict keyed by '{var.name}_{idx}' → numpy array
    """
    if len(data) > num_samples:
        idx = np.random.choice(len(data), num_samples, replace=False)
        X_imp = data[idx]
    else:
        X_imp = data

    n = len(X_imp)
    X_tensor = tf.convert_to_tensor(X_imp, dtype=tf.float32)

    with tf.GradientTape() as tape:
        output = model(X_tensor, training=False)
        loss = tf.reduce_sum(tf.square(output))

    grads = tape.gradient(loss, model.trainable_variables)

    omega_raw = {}
    for i, (var, g) in enumerate(zip(model.trainable_variables, grads)):
        key = f"{var.name}_{i}"
        if g is not None:
            omega_raw[key] = (g.numpy() ** 2) / n
        else:
            omega_raw[key] = np.zeros(var.shape.as_list())

    if normalize:
        global_max = max(v.max() for v in omega_raw.values())
        if global_max > 0:
            omega_dict = {k: v / global_max for k, v in omega_raw.items()}
        else:
            omega_dict = omega_raw
    else:
        omega_dict = omega_raw

    logger.info("MAS Ω computed: %d tensors, normalize=%s", len(omega_dict), normalize)
    return omega_dict


# ═══════════════════════════════════════════════════════════════════════════
#  Replay Mixing
# ═══════════════════════════════════════════════════════════════════════════

def mix_replay(new_data: np.ndarray, replay_buffer: np.ndarray,
               ratio: float, random_state: int = 42) -> np.ndarray:
    """
    Mix new-domain data with replay buffer.

    Parameters
    ----------
    new_data : (N_new, D) array of new environment data (scaled)
    replay_buffer : (N_old, D) array of previous domain data (scaled)
    ratio : replay fraction of the final mixed batch (e.g. 0.4 → 40% old)

    Returns
    -------
    Mixed and shuffled array.
    """
    rng = np.random.default_rng(random_state)
    if ratio <= 0 or len(replay_buffer) == 0:
        return new_data.copy()

    n_replay = int(ratio * new_data.shape[0] / (1 - ratio)) if ratio < 1.0 \
        else replay_buffer.shape[0]
    n_replay = min(n_replay, replay_buffer.shape[0])

    idx = rng.choice(replay_buffer.shape[0], size=n_replay, replace=False)
    mixed = np.concatenate([new_data, replay_buffer[idx]], axis=0)
    return mixed[rng.permutation(mixed.shape[0])]


# ═══════════════════════════════════════════════════════════════════════════
#  MAS Trainer
# ═══════════════════════════════════════════════════════════════════════════

class MASTrainer:
    """
    Custom trainer with Memory-Aware Synapses regularization.

    Loss = MAE_recon + (λ/2) * Σ_i Ω_i * (θ_i − θ*_i)²

    Early stopping monitors val_recon_loss only (not total loss).
    """

    def __init__(self, model, omega: dict, theta_star: dict,
                 lambda_mas: float = 50.0, learning_rate: float = 1e-3):
        self.model = model
        self.omega = omega
        self.theta_star = theta_star
        self.lambda_mas = lambda_mas
        self.optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)

    def mas_loss(self, y_true, y_pred):
        """Compute total loss = reconstruction + MAS penalty."""
        recon_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        mas_penalty = tf.constant(0.0, dtype=tf.float32)

        for i, var in enumerate(self.model.trainable_variables):
            key = f"{var.name}_{i}"
            if key in self.omega and key in self.theta_star:
                omega_i = tf.convert_to_tensor(self.omega[key], dtype=tf.float32)
                theta_i = tf.convert_to_tensor(self.theta_star[key], dtype=tf.float32)
                mas_penalty += tf.reduce_sum(omega_i * tf.square(var - theta_i))

        total = recon_loss + (self.lambda_mas / 2.0) * mas_penalty
        return total, recon_loss, mas_penalty

    @tf.function
    def train_step(self, x_batch):
        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch, training=True)
            total, recon, penalty = self.mas_loss(x_batch, y_pred)
        grads = tape.gradient(total, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total, recon, penalty

    def fit(self, X_train: np.ndarray, X_val: np.ndarray,
            epochs: int = 1000, batch_size: int = 32,
            patience: int = 25,
            progress_cb: Optional[Callable] = None) -> dict:
        """
        Train with MAS penalty and early stopping on val_recon_loss.

        Returns history dict with per-epoch metrics.
        """
        history = {
            'loss': [], 'recon_loss': [], 'mas_penalty': [],
            'val_loss': [], 'val_recon_loss': [], 'val_mas_penalty': []
        }
        best_val_recon = float('inf')
        patience_ctr = 0
        best_weights = None

        for epoch in range(epochs):
            perm = np.random.permutation(len(X_train))
            X_shuf = X_train[perm]
            ep_loss, ep_recon, ep_mas = [], [], []

            for i in range(0, len(X_train), batch_size):
                batch = tf.convert_to_tensor(
                    X_shuf[i:i + batch_size], dtype=tf.float32)
                tl, rl, mp = self.train_step(batch)
                ep_loss.append(tl.numpy())
                ep_recon.append(rl.numpy())
                ep_mas.append(mp.numpy())

            # Validation
            val_t = tf.convert_to_tensor(X_val, dtype=tf.float32)
            vl, vr, vm = self.mas_loss(val_t, self.model(val_t, training=False))

            history['loss'].append(float(np.mean(ep_loss)))
            history['recon_loss'].append(float(np.mean(ep_recon)))
            history['mas_penalty'].append(float(np.mean(ep_mas)))
            history['val_loss'].append(float(vl.numpy()))
            history['val_recon_loss'].append(float(vr.numpy()))
            history['val_mas_penalty'].append(float(vm.numpy()))

            if vr.numpy() < best_val_recon:
                best_val_recon = float(vr.numpy())
                patience_ctr = 0
                best_weights = store_model_weights(self.model)
            else:
                patience_ctr += 1

            if progress_cb and epoch % 10 == 0:
                progress_cb(epoch, epochs,
                            f"Epoch {epoch}: recon={history['recon_loss'][-1]:.4f} "
                            f"val_recon={history['val_recon_loss'][-1]:.4f}")

            if patience_ctr >= patience:
                logger.info("Early stopping at epoch %d (val_recon=%.4f)",
                            epoch, best_val_recon)
                break

        if best_weights:
            restore_model_weights(self.model, best_weights)

        logger.info("MAS training complete: best val_recon=%.4f, epochs=%d",
                    best_val_recon, len(history['loss']))
        return history


# ═══════════════════════════════════════════════════════════════════════════
#  Initial Environment Adaptation (From Scratch)
# ═══════════════════════════════════════════════════════════════════════════

class InitialEnvironmentAdapter:
    """
    Phase 1: Adapt model to a completely new environment.

    All captured flows are treated as normal baseline.
    Retrains scaler, AE, and GMM from scratch.
    """

    def __init__(self, input_dim: int = 18,
                 encoder_layers: list = None,
                 latent_dim: int = 16,
                 gmm_k: int = 21):
        self.input_dim = input_dim
        self.encoder_layers = encoder_layers or [64, 48, 24, 18]
        self.latent_dim = latent_dim
        self.gmm_k = gmm_k

        # Outputs (populated after adapt())
        self.scaler = None
        self.ae_model = None
        self.gmm = None
        self.threshold = None
        self.scores = None
        self.history = None

    def adapt(self, X_raw: np.ndarray, *,
              ae_epochs: int = 200,
              ae_batch_size: int = 256,
              ae_lr: float = 1e-3,
              val_split: float = 0.15,
              progress_cb: Optional[Callable] = None) -> dict:
        """
        Full from-scratch adaptation.

        Parameters
        ----------
        X_raw : (N, 18) raw feature values (unscaled)
        ae_epochs : max training epochs
        ae_lr : learning rate
        val_split : validation fraction
        progress_cb : callable(step, total, msg)

        Returns
        -------
        dict with scores, percentiles, history for threshold selection UI.
        """
        from retrain import compute_percentiles

        n = len(X_raw)
        total_steps = 4
        if progress_cb is None:
            progress_cb = lambda s, t, m: None

        # ── Step 1: Fit new scaler ────────────────────────────────────
        progress_cb(1, total_steps, f"Fitting scaler on {n} samples…")
        X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)

        # Split train/val
        idx = np.random.permutation(n)
        n_val = max(1, int(n * val_split))
        X_train = X_scaled[idx[n_val:]]
        X_val = X_scaled[idx[:n_val]]

        # ── Step 2: Build and train AE from scratch ───────────────────
        progress_cb(2, total_steps, f"Training AE from scratch ({ae_epochs} epochs)…")
        self.ae_model = self._build_ae(ae_lr)

        self.history = self.ae_model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=25,
                    restore_best_weights=True
                )
            ]
        ).history

        # ── Step 3: Fit GMM on reconstruction errors ──────────────────
        progress_cb(3, total_steps, "Fitting GMM on reconstruction errors…")
        X_recon = self.ae_model.predict(X_scaled, verbose=0)
        errors = np.abs(X_scaled - X_recon)

        self.gmm = GaussianMixture(
            n_components=self.gmm_k,
            covariance_type="full",
            random_state=42,
            max_iter=300,
        )
        self.gmm.fit(errors)

        # ── Step 4: Score using VAL split only (unseen by AE) ─────────
        progress_cb(4, total_steps, "Computing score distribution (val split)…")
        val_recon = self.ae_model.predict(X_val, verbose=0)
        val_errors = np.abs(X_val - val_recon)
        self.scores = self.gmm.score_samples(val_errors)
        percentiles = compute_percentiles(self.scores)

        # Default threshold: 5th percentile (conservative)
        self.threshold = percentiles["p5"]

        return {
            "scores": self.scores,
            "percentiles": percentiles,
            "threshold": self.threshold,
            "n_samples": n,
            "history": self.history,
            "score_min": float(np.min(self.scores)),
            "score_max": float(np.max(self.scores)),
            "score_mean": float(np.mean(self.scores)),
        }

    def _build_ae(self, lr: float):
        """Build a fresh autoencoder matching the M3 architecture."""
        from tensorflow.keras.layers import (
            Input, Dense, BatchNormalization, Activation
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l1

        inp = Input(shape=(self.input_dim,))
        x = inp

        # Encoder
        for units in self.encoder_layers:
            x = Dense(units, kernel_regularizer=l1(1e-5))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        # Latent
        x = Dense(self.latent_dim, kernel_regularizer=l1(1e-5))(x)
        x = BatchNormalization()(x)
        latent = Activation('relu')(x)

        # Decoder (mirror)
        x = latent
        for units in reversed(self.encoder_layers):
            x = Dense(units, kernel_regularizer=l1(1e-5))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        output = Dense(self.input_dim, activation='linear')(x)

        model = Model(inp, output)
        model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0),
                      loss='mae')
        return model


# ═══════════════════════════════════════════════════════════════════════════
#  Incremental Drift Adaptation (MAS + Replay)
# ═══════════════════════════════════════════════════════════════════════════

class IncrementalAdapter:
    """
    Phase 2+: Adapt model to distribution drift using MAS+Replay.

    Preserves learned representations via:
    - MAS penalty (prevents catastrophic forgetting)
    - Replay buffer (rehearses old-domain patterns)

    Workflow:
    1. Checkpoint current model (θ*)
    2. Compute Ω on replay buffer
    3. Mix new data with replay buffer
    4. Train with MAS penalty
    5. Refit GMM on mixed errors
    6. Present to human for acceptance
    """

    def __init__(self, ae_model, scaler, gmm,
                 features: list,
                 lambda_mas: float = 50.0,
                 replay_ratio: float = 0.4,
                 gmm_k: int = 21):
        self.ae_model = ae_model
        self.scaler = scaler
        self.current_gmm = gmm
        self.features = features
        self.lambda_mas = lambda_mas
        self.replay_ratio = replay_ratio
        self.gmm_k = gmm_k

        # Outputs
        self.new_gmm = None
        self.new_threshold = None
        self.scores = None
        self.history = None
        self.theta_star = None  # checkpoint weights

    def adapt(self, X_new_scaled: np.ndarray,
              replay_buffer: np.ndarray, *,
              ae_epochs: int = 1000,
              ae_batch_size: int = 32,
              ae_lr: float = 1e-3,
              val_split: float = 0.15,
              mas_num_samples: int = 10000,
              progress_cb: Optional[Callable] = None) -> dict:
        """
        Run the MAS+Replay incremental adaptation.

        Parameters
        ----------
        X_new_scaled : (N, D) scaled new-environment data
        replay_buffer : (M, D) scaled data from previous states
        ae_epochs : max epochs (early stopping applies)
        ae_lr : learning rate
        val_split : validation fraction
        mas_num_samples : samples for Ω computation
        progress_cb : callable(step, total, msg)

        Returns
        -------
        dict with scores, percentiles, history for human review.
        """
        from retrain import compute_percentiles

        total_steps = 5
        if progress_cb is None:
            progress_cb = lambda s, t, m: None

        # ── Step 1: Checkpoint current weights (θ*) ───────────────────
        progress_cb(1, total_steps, "Checkpointing current model…")
        self.theta_star = store_model_weights(self.ae_model)

        # ── Step 2: Compute MAS importance (Ω) on replay buffer ───────
        progress_cb(2, total_steps,
                    f"Computing MAS importance ({len(replay_buffer)} samples)…")
        omega = compute_mas_importance(
            self.ae_model, replay_buffer,
            num_samples=mas_num_samples, normalize=True
        )

        # ── Step 3: Mix new data with replay ──────────────────────────
        progress_cb(3, total_steps,
                    f"Mixing data (replay ratio={self.replay_ratio:.0%})…")
        X_mixed = mix_replay(X_new_scaled, replay_buffer, self.replay_ratio)

        # Split into train/val
        n = len(X_mixed)
        idx = np.random.permutation(n)
        n_val = max(1, int(n * val_split))
        X_train = X_mixed[idx[n_val:]]
        X_val = X_mixed[idx[:n_val]]

        # ── Step 4: Train with MAS penalty ────────────────────────────
        progress_cb(4, total_steps,
                    f"MAS+Replay training (λ={self.lambda_mas}, up to {ae_epochs} epochs)…")
        trainer = MASTrainer(
            model=self.ae_model,
            omega=omega,
            theta_star=self.theta_star,
            lambda_mas=self.lambda_mas,
            learning_rate=ae_lr,
        )
        self.history = trainer.fit(
            X_train, X_val,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            patience=25,
            progress_cb=progress_cb,
        )

        # ── Step 5: Refit GMM on mixed reconstruction errors ──────────
        progress_cb(5, total_steps, "Refitting GMM on updated errors…")
        X_recon = self.ae_model.predict(X_mixed, verbose=0)
        errors = np.abs(X_mixed - X_recon)

        self.new_gmm = GaussianMixture(
            n_components=self.gmm_k,
            covariance_type="full",
            random_state=42,
            max_iter=300,
        )
        self.new_gmm.fit(errors)

        # Score using VAL split only (unseen by AE during this round)
        val_recon = self.ae_model.predict(X_val, verbose=0)
        val_errors = np.abs(X_val - val_recon)
        self.scores = self.new_gmm.score_samples(val_errors)
        percentiles = compute_percentiles(self.scores)
        self.new_threshold = percentiles["p5"]

        return {
            "scores": self.scores,
            "percentiles": percentiles,
            "threshold": self.new_threshold,
            "n_samples": n,
            "n_new": len(X_new_scaled),
            "n_replay": n - len(X_new_scaled),
            "history": self.history,
            "score_min": float(np.min(self.scores)),
            "score_max": float(np.max(self.scores)),
            "score_mean": float(np.mean(self.scores)),
        }

    def rollback(self):
        """Reject adaptation — restore θ* checkpoint."""
        if self.theta_star:
            restore_model_weights(self.ae_model, self.theta_star)
            logger.info("Rolled back to checkpoint weights")
        else:
            logger.warning("No checkpoint to rollback to")

    def accept(self):
        """Accept adaptation — update GMM, clear checkpoint."""
        if self.new_gmm is not None:
            self.current_gmm = self.new_gmm
            self.theta_star = None  # consumed
            logger.info("Adaptation accepted: new GMM + updated AE in place")
        return self.new_gmm, self.new_threshold
