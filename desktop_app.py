"""
M3 AE-GMM IDS — Standalone Desktop Application
────────────────────────────────────────────────
PySide6 (Qt) desktop app with CICFlowMeter-style flow table,
colour-coded predictions, and right-click human-feedback panel.

Run:
    python desktop_app.py
"""

import sys
import os
import logging
import threading
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure this directory is importable (engine, config, capture live here)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Ensure Qt can find its platform plugins
import PySide6
_qt_plugin_path = os.path.join(os.path.dirname(PySide6.__file__), "plugins")
os.environ.setdefault("QT_PLUGIN_PATH", _qt_plugin_path)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTableView, QToolBar, QStatusBar,
    QFileDialog, QMenu, QComboBox, QPushButton, QHeaderView,
    QAbstractItemView, QMessageBox, QLabel, QWidget, QHBoxLayout,
    QVBoxLayout, QSplitter, QFrame, QSizePolicy, QDialog,
    QDialogButtonBox, QRadioButton, QButtonGroup, QDoubleSpinBox,
    QGroupBox, QProgressBar,
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QTimer, QThread, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont


# ── Sequential-row-number proxy ──────────────────────────────────────────

class SeqRowProxyModel(QSortFilterProxyModel):
    """Proxy that always shows 1, 2, 3… as vertical header, regardless of sort."""

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return str(section + 1)
        return super().headerData(section, orientation, role)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for embedding in Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import shap

from engine import M3InferenceEngine
from capture import list_interfaces
from flow_extractor import RealtimeFlowExtractor
from incremental import (
    InitialEnvironmentAdapter, IncrementalAdapter,
    store_model_weights, restore_model_weights,
)
from config import (
    DEPLOY_DIR, BUFFER_DIR, BUFFER_MAX_SAMPLES, ATTACK_CONFIDENCE_THRESHOLD,
)

# ── Attack-rate threshold for triggering adaptation suggestion ───────────
ATTACK_RATE_TRIGGER = 0.50   # ≥50% attacks → suggest adaptation


# ═══════════════════════════════════════════════════════════════════════════
#  RetrainWorker — runs adaptation off the main thread
# ═══════════════════════════════════════════════════════════════════════════

class RetrainWorker(QThread):
    """Background worker for model adaptation (keeps UI responsive).

    Supports two modes:
      - 'initial': From-scratch adaptation (new scaler, AE, GMM)
      - 'incremental': MAS+Replay hybrid (preserves learned representations)
    """

    progress = Signal(int, int, str)       # step, total, message
    finished = Signal(dict)                # result dict
    failed = Signal(str)                   # error message

    def __init__(self, engine, flow_data, mode="initial",
                 replay_buffer=None, lambda_mas=50.0, replay_ratio=0.4):
        super().__init__()
        self.engine = engine
        self.flow_data = flow_data
        self.mode = mode                    # "initial" or "incremental"
        self.replay_buffer = replay_buffer  # scaled ndarray for incremental mode
        self.lambda_mas = lambda_mas
        self.replay_ratio = replay_ratio

    def run(self):
        try:
            if self.mode == "initial":
                result = self._run_initial()
            else:
                result = self._run_incremental()
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _run_initial(self):
        """Phase 1: From-scratch environment adaptation."""
        from engine import MAX_TRAINING_SIZE

        adapter = InitialEnvironmentAdapter(
            input_dim=len(self.engine.features),
            gmm_k=self.engine.gmm.n_components,
        )
        X_raw = self.flow_data[self.engine.features].values.astype(np.float64)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Cap at MAX_TRAINING_SIZE (288K)
        if len(X_raw) > MAX_TRAINING_SIZE:
            idx = np.random.choice(len(X_raw), MAX_TRAINING_SIZE, replace=False)
            X_raw = X_raw[idx]

        result = adapter.adapt(
            X_raw,
            ae_epochs=200,
            ae_lr=1e-3,
            progress_cb=lambda s, t, m: self.progress.emit(s, t, m),
        )

        # Attach adapter objects so the main thread can swap them in
        result["_adapter"] = adapter
        result["_mode"] = "initial"
        return result

    def _run_incremental(self):
        """Phase 2+: MAS+Replay incremental adaptation.

        Training composition:
          - 30% from replay buffer (max 86,400 samples)
          - 70% from current in-memory flows (new environment data)
          - Total capped at MAX_TRAINING_SIZE (288K)
        """
        from engine import MAX_TRAINING_SIZE, REPLAY_INCR_RATIO, REPLAY_POST_ADAPT_CAP

        # Scale new data
        X_raw = self.flow_data[self.engine.features].values.astype(np.float64)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_new_scaled = self.engine.scaler.transform(X_raw)

        # Determine training composition
        n_replay_available = len(self.replay_buffer) if self.replay_buffer is not None else 0
        n_replay = min(
            int(MAX_TRAINING_SIZE * REPLAY_INCR_RATIO),  # 30% of max
            REPLAY_POST_ADAPT_CAP,                         # hard cap 86,400
            n_replay_available,                            # what's available
        )
        n_new = min(
            len(X_new_scaled),
            MAX_TRAINING_SIZE - n_replay,                  # remainder goes to new data
        )

        # Subsample if needed
        if len(X_new_scaled) > n_new:
            idx = np.random.choice(len(X_new_scaled), n_new, replace=False)
            X_new_scaled = X_new_scaled[idx]

        # Sample from replay buffer
        if n_replay > 0 and self.replay_buffer is not None:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(self.replay_buffer), n_replay, replace=False)
            replay_subset = self.replay_buffer[idx]
        else:
            replay_subset = np.empty((0, len(self.engine.features)))

        adapter = IncrementalAdapter(
            ae_model=self.engine.ae_model,
            scaler=self.engine.scaler,
            gmm=self.engine.gmm,
            features=self.engine.features,
            lambda_mas=self.lambda_mas,
            replay_ratio=REPLAY_INCR_RATIO,
            gmm_k=self.engine.gmm.n_components,
        )

        result = adapter.adapt(
            X_new_scaled,
            replay_subset,
            ae_epochs=1000,
            ae_lr=1e-3,
            progress_cb=lambda s, t, m: self.progress.emit(s, t, m),
        )

        result["_adapter"] = adapter
        result["_mode"] = "incremental"
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  ThresholdPickerDialog — shows score distribution + percentile lines
# ═══════════════════════════════════════════════════════════════════════════

class ThresholdPickerDialog(QDialog):
    """
    Shows the GMM score distribution from the adaptation run and lets
    the user either accept the auto threshold or pick a custom one.
    """

    def __init__(self, scores, auto_threshold, percentiles, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Anomaly Threshold")
        self.setMinimumSize(800, 550)

        self.scores = scores
        self.auto_threshold = auto_threshold
        self.percentiles = percentiles
        self.chosen_threshold = auto_threshold

        layout = QVBoxLayout(self)

        # ── Info label ────────────────────────────────────────────────
        info = QLabel(
            f"<b>Adaptation complete.</b>  {len(scores)} flows scored.<br>"
            f"All captured traffic was treated as normal baseline.<br>"
            f"Choose a threshold below which traffic will be flagged as <b>Attack</b>."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── Chart ─────────────────────────────────────────────────────
        self.figure, self.ax = plt.subplots(figsize=(8, 3.5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)
        layout.addWidget(self.canvas)

        # ── Radio: auto vs manual ─────────────────────────────────────
        grp = QGroupBox("Threshold Selection")
        grp_layout = QVBoxLayout(grp)

        self.btn_group = QButtonGroup(self)
        self.radio_auto = QRadioButton(
            f"Auto (5th percentile = {auto_threshold:.4f})  —  "
            f"~5% of normal traffic flagged"
        )
        self.radio_auto.setChecked(True)
        self.btn_group.addButton(self.radio_auto, 0)
        grp_layout.addWidget(self.radio_auto)

        manual_row = QHBoxLayout()
        self.radio_manual = QRadioButton("Manual threshold:")
        self.btn_group.addButton(self.radio_manual, 1)
        manual_row.addWidget(self.radio_manual)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(4)
        self.spin.setRange(float(np.min(scores)) - 10, float(np.max(scores)) + 10)
        self.spin.setValue(auto_threshold)
        self.spin.setEnabled(False)
        self.spin.setSingleStep(0.5)
        manual_row.addWidget(self.spin)
        grp_layout.addLayout(manual_row)

        # Percentile quick-picks
        pct_row = QHBoxLayout()
        pct_row.addWidget(QLabel("Quick picks:"))
        for pname, pval in sorted(percentiles.items()):
            btn = QPushButton(f"{pname} ({pval:.2f})")
            btn.setFixedWidth(120)
            btn.clicked.connect(lambda checked, v=pval: self._set_manual(v))
            pct_row.addWidget(btn)
        pct_row.addStretch()
        grp_layout.addLayout(pct_row)

        layout.addWidget(grp)

        # ── Buttons ───────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # ── Wiring ────────────────────────────────────────────────────
        self.btn_group.idToggled.connect(self._mode_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)

        self._draw_chart()

    def _mode_changed(self, id_, checked):
        if not checked:
            return
        self.spin.setEnabled(id_ == 1)
        if id_ == 0:
            self.chosen_threshold = self.auto_threshold
        else:
            self.chosen_threshold = self.spin.value()
        self._draw_chart()

    def _set_manual(self, value):
        self.radio_manual.setChecked(True)
        self.spin.setValue(value)
        self.chosen_threshold = value
        self._draw_chart()

    def _on_spin_changed(self, value):
        if self.radio_manual.isChecked():
            self.chosen_threshold = value
            self._draw_chart()

    def _draw_chart(self):
        self.ax.clear()
        self.ax.hist(self.scores, bins=80, color="#4a90d9", alpha=0.7,
                     edgecolor="white", linewidth=0.3, label="Score distribution")

        # Percentile lines
        colors_pct = {
            "p1": "#d32f2f", "p5": "#e64a19", "p10": "#f57c00",
            "p25": "#fbc02d", "p50": "#388e3c", "p75": "#1976d2",
            "p90": "#7b1fa2", "p95": "#c2185b", "p99": "#455a64",
        }
        for pname, pval in sorted(self.percentiles.items()):
            c = colors_pct.get(pname, "gray")
            self.ax.axvline(pval, color=c, linestyle=":", linewidth=1, alpha=0.7)
            self.ax.text(pval, self.ax.get_ylim()[1] * 0.95, f" {pname}",
                         fontsize=7, color=c, rotation=90, va="top")

        # Chosen threshold
        self.ax.axvline(self.chosen_threshold, color="red", linewidth=2.5,
                        linestyle="--", label=f"Threshold = {self.chosen_threshold:.4f}")

        # Shade attack region
        xlim = self.ax.get_xlim()
        self.ax.axvspan(xlim[0], self.chosen_threshold, alpha=0.08, color="red")
        self.ax.text(
            self.chosen_threshold - (self.chosen_threshold - xlim[0]) * 0.3,
            self.ax.get_ylim()[1] * 0.5, "ATTACK\nzone",
            fontsize=10, color="red", alpha=0.5, ha="center", fontweight="bold",
        )

        self.ax.set_xlabel("GMM Log-Probability Score")
        self.ax.set_ylabel("Count")
        self.ax.set_title("Score Distribution of New-Environment Traffic")
        self.ax.legend(loc="upper right", fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw()

    def _accept(self):
        if self.radio_manual.isChecked():
            self.chosen_threshold = self.spin.value()
        else:
            self.chosen_threshold = self.auto_threshold
        self.accept()

    def get_threshold(self) -> float:
        return self.chosen_threshold


# ── Column layout ────────────────────────────────────────────────────────
META_COLS   = ["Timestamp", "Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port"]
LABEL_COLS  = ["prediction", "confidence", "gmm_score"]
RESULT_COLS = LABEL_COLS + ["human_label"]

# Colours
CLR_NORMAL       = QColor(210, 245, 210)   # light green
CLR_ATTACK       = QColor(255, 210, 210)   # light red
CLR_FB_NORMAL    = QColor(180, 230, 180)   # stronger green (human-confirmed)
CLR_FB_ATTACK    = QColor(255, 180, 180)   # stronger red   (human-confirmed)


# ═════════════════════════════════════════════════════════════════════════
#  FlowTableModel — QAbstractTableModel backed by a DataFrame
# ═════════════════════════════════════════════════════════════════════════

class FlowTableModel(QAbstractTableModel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame = pd.DataFrame()
        self._cols: list[str] = []

    # ── data mutation ─────────────────────────────────────────────────────

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self._cols = list(df.columns)
        self.endResetModel()

    def append_rows(self, new: pd.DataFrame):
        """Append rows (used by live capture)."""
        if new.empty:
            return
        # If table was empty, initialise columns first
        if self._df.empty or not self._cols:
            self.beginResetModel()
            self._df = new.reset_index(drop=True)
            self._cols = list(new.columns)
            self.endResetModel()
            return
        # Align new columns to existing columns
        for c in self._cols:
            if c not in new.columns:
                new[c] = ""
        new = new[self._cols]
        first = len(self._df)
        last  = first + len(new) - 1
        self.beginInsertRows(QModelIndex(), first, last)
        self._df = pd.concat([self._df, new], ignore_index=True)
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self._df = pd.DataFrame()
        self._cols = []
        self.endResetModel()

    # ── QAbstractTableModel interface ─────────────────────────────────────

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._cols)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        val = self._df.iat[r, c]

        if role == Qt.DisplayRole:
            if isinstance(val, float):
                return f"{val:.4f}" if abs(val) < 1e6 else f"{val:.2e}"
            return str(val) if pd.notna(val) else ""

        if role == Qt.BackgroundRole:
            return self._row_brush(r)

        if role == Qt.TextAlignmentRole:
            col_name = self._cols[c]
            if col_name in ("prediction", "confidence", "gmm_score", "human_label",
                            "Src Port", "Dst Port"):
                return int(Qt.AlignCenter)
            return int(Qt.AlignLeft | Qt.AlignVCenter)

        if role == Qt.FontRole:
            col_name = self._cols[c]
            if col_name in ("prediction", "human_label"):
                f = QFont()
                f.setBold(True)
                return f

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and section < len(self._cols):
            return self._cols[section]
        if orientation == Qt.Vertical:
            return str(section + 1)
        return None

    # ── helpers ───────────────────────────────────────────────────────────

    def _row_brush(self, row) -> QBrush | None:
        human = self._cell_str(row, "human_label")
        pred  = self._cell_str(row, "prediction")
        if human == "Attack":
            return QBrush(CLR_FB_ATTACK)
        if human == "Normal":
            return QBrush(CLR_FB_NORMAL)
        if pred == "Attack":
            return QBrush(CLR_ATTACK)
        if pred == "Normal":
            return QBrush(CLR_NORMAL)
        return None

    def _cell_str(self, row, col) -> str:
        if col not in self._cols:
            return ""
        v = self._df.iat[row, self._cols.index(col)]
        s = str(v).strip() if pd.notna(v) else ""
        return "" if s in ("", "—") else s

    def get_row(self, row: int) -> pd.Series:
        return self._df.iloc[row]

    def col_name(self, col: int) -> str:
        return self._cols[col] if 0 <= col < len(self._cols) else ""

    def update_cell(self, row: int, col_name: str, value):
        if col_name not in self._cols:
            return
        ci = self._cols.index(col_name)
        self._df.iat[row, ci] = value
        self.dataChanged.emit(
            self.index(row, 0),
            self.index(row, len(self._cols) - 1),
        )


# ═════════════════════════════════════════════════════════════════════════
#  MainWindow
# ═════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("M3 AE-GMM — Network Intrusion Detection System")
        self.setMinimumSize(1280, 720)

        # ── Caches ────────────────────────────────────────────────────────
        self._shap_cache: dict[int, dict] = {}   # row → SHAP explanation

        # ── Engine ────────────────────────────────────────────────────────
        self.engine = M3InferenceEngine(
            str(DEPLOY_DIR),
            buffer_dir=str(BUFFER_DIR),
            buffer_max=BUFFER_MAX_SAMPLES,
            attack_conf_threshold=ATTACK_CONFIDENCE_THRESHOLD,
        )

        # ── Live capture ─────────────────────────────────────────────────
        self._extractor: RealtimeFlowExtractor | None = None
        self._pending_flows: deque[dict] = deque()   # raw flow dicts from sniff thread
        self._lock = threading.Lock()  # protects _pending_flows from callback thread

        # Drip timer: feed batches of flows into the table
        self._drip_timer = QTimer(self)
        self._drip_timer.setInterval(100)
        self._drip_timer.timeout.connect(self._drip_batch)
        # Feature columns for display (Dst Port already in META_COLS)
        self._feat_cols = [f for f in self.engine.features if f not in META_COLS]
        # Timestamp first, then label cols, then meta, features, feedback last
        self._display_order = (
            ["Timestamp"] + LABEL_COLS +
            [c for c in META_COLS if c != "Timestamp"] +
            self._feat_cols + ["human_label"]
        )

        # ── Table ─────────────────────────────────────────────────────────
        self.table_model = FlowTableModel()

        self.proxy = SeqRowProxyModel()
        self.proxy.setSourceModel(self.table_model)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        self.table.doubleClicked.connect(self._on_double_click)
        self.table.verticalHeader().setDefaultSectionSize(24)

        # ── SHAP detail panel (hidden until double-click) ─────────────────
        self._expanded_row = -1  # which source row is currently expanded
        self._shap_panel = QFrame()
        self._shap_panel.setFrameStyle(QFrame.StyledPanel)
        self._shap_panel.setVisible(False)
        self._shap_layout = QVBoxLayout(self._shap_panel)
        self._shap_layout.setContentsMargins(4, 4, 4, 4)
        # header label
        self._shap_header = QLabel()
        self._shap_header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        self._shap_layout.addWidget(self._shap_header)
        # canvas container (side by side AE + GMM)
        self._shap_canvas_container = QHBoxLayout()
        self._shap_layout.addLayout(self._shap_canvas_container)

        # ── Attack-rate warning banner (hidden by default) ────────────────
        self._warn_banner = QLabel()
        self._warn_banner.setStyleSheet(
            "background-color: #FFF3CD; color: #856404; "
            "border: 1px solid #FFEEBA; border-radius: 4px; "
            "padding: 8px; font-size: 13px; font-weight: bold;"
        )
        self._warn_banner.setWordWrap(True)
        self._warn_banner.setVisible(False)

        # ── Splitter: table on top, SHAP panel on bottom ──────────────────
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(self._shap_panel)
        splitter.setStretchFactor(0, 3)  # table gets more space
        splitter.setStretchFactor(1, 1)

        # ── Central layout with banner above splitter ─────────────────────
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(self._warn_banner)
        central_layout.addWidget(splitter)
        self.setCentralWidget(central)

        # ── Chrome ────────────────────────────────────────────────────────
        self._build_menubar()
        self._build_toolbar()
        self._build_statusbar()

        # ── Populate network interfaces ───────────────────────────────────
        self._populate_interfaces()

    def closeEvent(self, event):
        """Ensure capture is stopped on exit."""
        if self._extractor and self._extractor.is_running:
            self._extractor.stop()
        self._drip_timer.stop()
        super().closeEvent(event)

    # ══════════════════════════════════════════════════════════════════════
    #  Menu bar
    # ══════════════════════════════════════════════════════════════════════

    def _build_menubar(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")

        act = QAction("&Open CSV…", self)
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self._open_csv)
        fm.addAction(act)

        act = QAction("&Export Results…", self)
        act.setShortcut("Ctrl+S")
        act.triggered.connect(self._export_csv)
        fm.addAction(act)

        fm.addSeparator()
        act = QAction("&Adapt Model to Environment…", self)
        act.triggered.connect(self._start_adaptation)
        fm.addAction(act)

        act = QAction("Adjust &Threshold…", self)
        act.setShortcut("Ctrl+T")
        act.triggered.connect(self._adjust_threshold)
        fm.addAction(act)

        fm.addSeparator()
        act = QAction("E&xit", self)
        act.setShortcut("Alt+F4")
        act.triggered.connect(self.close)
        fm.addAction(act)

    # ══════════════════════════════════════════════════════════════════════
    #  Toolbar
    # ══════════════════════════════════════════════════════════════════════

    def _build_toolbar(self):
        tb = self.addToolBar("Controls")
        tb.setMovable(False)
        tb.setFloatable(False)

        btn = QPushButton("  Open CSV  ")
        btn.clicked.connect(self._open_csv)
        tb.addWidget(btn)

        tb.addSeparator()

        tb.addWidget(QLabel("  Interface: "))
        self.iface_combo = QComboBox()
        self.iface_combo.setMinimumWidth(300)
        self.iface_combo.addItem("Select network interface…")
        self.iface_combo.currentIndexChanged.connect(self._on_iface_changed)
        tb.addWidget(self.iface_combo)

        # Subnet filter (for network-wide monitoring via SPAN / mirror port)
        tb.addWidget(QLabel("  Subnet: "))
        self.subnet_edit = QComboBox()
        self.subnet_edit.setEditable(True)
        self.subnet_edit.setMinimumWidth(160)
        self.subnet_edit.addItems(["(all traffic)", "10.10.10.0/24", "192.168.1.0/24"])
        self.subnet_edit.setToolTip(
            "Optional BPF subnet filter.\n"
            "Use with a SPAN/mirror port to monitor an entire network segment.\n"
            "Leave as '(all traffic)' to capture everything."
        )
        tb.addWidget(self.subnet_edit)

        self.btn_start = QPushButton("  Start  ")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._start_capture)
        tb.addWidget(self.btn_start)

        self.btn_stop = QPushButton("  Stop  ")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_capture)
        tb.addWidget(self.btn_stop)

        tb.addSeparator()

        self.btn_adapt = QPushButton("  Adapt Model  ")
        self.btn_adapt.setToolTip(
            "Fine-tune the model to this network's traffic.\n"
            "Use when ≥50% of flows are flagged as attacks."
        )
        self.btn_adapt.clicked.connect(self._start_adaptation)
        tb.addWidget(self.btn_adapt)

        tb.addSeparator()

        btn = QPushButton("  Clear  ")
        btn.clicked.connect(self._clear)
        tb.addWidget(btn)

    def _on_iface_changed(self, idx):
        self.btn_start.setEnabled(idx > 0 and (self._extractor is None or not self._extractor.is_running))

    # ══════════════════════════════════════════════════════════════════════
    #  Live capture
    # ══════════════════════════════════════════════════════════════════════

    def _populate_interfaces(self):
        """Scan and populate the interface combo box."""
        self.iface_combo.blockSignals(True)
        self.iface_combo.clear()
        self.iface_combo.addItem("Select network interface…")
        self._iface_map: list[dict] = []
        try:
            ifaces = list_interfaces(include_monitor=True)
            for iface in ifaces:
                if iface.get("is_monitor"):
                    label = f"[MONITOR] {iface['name']}  (no IP)"
                elif iface.get("is_link_local"):
                    label = f"{iface['name']}  (link-local)"
                else:
                    label = f"{iface['name']}  ({iface['ip']})"
                if iface.get('description'):
                    label += f"  — {iface['description']}"
                self.iface_combo.addItem(label)
                self._iface_map.append(iface)
        except Exception as exc:
            self.sbar.showMessage(f"Failed to list interfaces: {exc}", 5000)
        self.iface_combo.blockSignals(False)

    def _start_capture(self):
        idx = self.iface_combo.currentIndex()
        if idx < 1:
            return
        iface = self._iface_map[idx - 1]
        iface_name = iface.get("network_name") or iface["name"]

        # Build optional BPF filter from subnet combo
        bpf_filter = ""
        subnet_text = self.subnet_edit.currentText().strip()
        if subnet_text and subnet_text != "(all traffic)":
            bpf_filter = f"net {subnet_text}"

        self._extractor = RealtimeFlowExtractor(
            callback=self._on_flow_complete,
            activity_timeout=5.0,     # emit flow after 5s of no packets
            expire_interval=1.0,       # sweep every second
        )
        self._extractor.start(iface_name, bpf_filter=bpf_filter)
        with self._lock:
            self._pending_flows.clear()
        self._drip_timer.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.iface_combo.setEnabled(False)
        self.subnet_edit.setEnabled(False)

        mode = "[MONITOR] " if iface.get("is_monitor") else ""
        filter_info = f" | filter: {bpf_filter}" if bpf_filter else ""
        ip_info = iface.get('ip') or 'no IP'
        self.sbar.showMessage(f"{mode}Capturing on {iface['name']} ({ip_info}){filter_info}…")

    def _stop_capture(self):
        if self._extractor:
            self._extractor.stop()
            self._extractor = None

        # Don't flush synchronously — let drip timer drain remaining flows
        # Just update UI state immediately so the app stays responsive
        self.btn_start.setEnabled(self.iface_combo.currentIndex() > 0)
        self.btn_stop.setEnabled(False)
        self.iface_combo.setEnabled(True)
        self.subnet_edit.setEnabled(True)

        queued = 0
        with self._lock:
            queued = len(self._pending_flows)

        if queued > 0:
            self.sbar.showMessage(f"Capture stopped. Processing {queued} remaining flows…")
            # Keep drip timer running to drain the queue
        else:
            self._drip_timer.stop()
            self.sbar.showMessage("Capture stopped.", 5000)

    def _on_flow_complete(self, flow_features: dict):
        """
        Callback from RealtimeFlowExtractor (runs on sniff/expire thread).
        Only enqueue the raw dict — prediction happens on the main thread.
        """
        with self._lock:
            self._pending_flows.append(flow_features)

    def _predict_one(self, flow_features: dict) -> pd.Series | None:
        """Run prediction on a single flow dict (main thread only)."""
        try:
            flow_df = pd.DataFrame([flow_features])
            result = self.engine.predict(flow_df)
            if "human_label" not in result.columns:
                result["human_label"] = "—"
            result = self._reorder(result)
            return result.iloc[0]
        except Exception as exc:
            logger.warning("Prediction failed for flow: %s", exc)
            return None

    DRIP_BATCH_SIZE = 128  # flows per timer tick

    def _drip_batch(self):
        """Pop up to DRIP_BATCH_SIZE pending flows, batch-predict, append."""
        with self._lock:
            if not self._pending_flows:
                if self._extractor is None:
                    self._drip_timer.stop()
                    self.sbar.showMessage("Capture stopped.", 5000)
                return
            batch_size = min(len(self._pending_flows), self.DRIP_BATCH_SIZE)
            flows = [self._pending_flows.popleft() for _ in range(batch_size)]
            queued = len(self._pending_flows)

        try:
            batch_df = pd.DataFrame(flows)
            results = self.engine.predict(batch_df)
            if "human_label" not in results.columns:
                results["human_label"] = "—"
            results = self._reorder(results)
        except Exception as exc:
            logger.warning("Batch prediction failed: %s", exc)
            return

        self.table_model.append_rows(results)
        self._refresh_status()

        # Auto-scroll to latest
        last = self.table_model.rowCount() - 1
        if last >= 0:
            proxy_idx = self.proxy.index(last, 0)
            self.table.scrollTo(proxy_idx)

        if self._extractor:
            stats = self._extractor.stats
            self.sbar.showMessage(
                f"Live: {stats['flows_emitted']} flows "
                f"(TCP:{stats.get('flows_tcp',0)} UDP:{stats.get('flows_udp',0)})  |  "
                f"pkts: {stats['packets_processed']} "
                f"(v4:{stats.get('packets_ipv4',0)} v6:{stats.get('packets_ipv6',0)})  |  "
                f"active: {stats['active_flows']}  |  "
                f"queued: {queued}"
            )

    def _flush_pending(self):
        """Predict + append all pending flows at once (used on stop)."""
        with self._lock:
            if not self._pending_flows:
                return
            flows = list(self._pending_flows)
            self._pending_flows.clear()
        rows = []
        for flow in flows:
            row = self._predict_one(flow)
            if row is not None:
                rows.append(row.to_frame().T)
        if rows:
            batch = pd.concat(rows, ignore_index=True)
            self.table_model.append_rows(batch)
            self._auto_resize()
            self._refresh_status()

    # ══════════════════════════════════════════════════════════════════════
    #  Status bar
    # ══════════════════════════════════════════════════════════════════════

    def _build_statusbar(self):
        self.sbar = QStatusBar()
        self.setStatusBar(self.sbar)

        self.lbl_flows   = QLabel("Flows: 0")
        self.lbl_normal  = QLabel("Normal: 0")
        self.lbl_attack  = QLabel("Attack: 0")
        self.lbl_fb      = QLabel("Feedback: 0")
        self.lbl_buffers = QLabel("Buffers — benign: 0  attack: 0")

        for lbl in (self.lbl_flows, self.lbl_normal, self.lbl_attack,
                     self.lbl_fb, self.lbl_buffers):
            self.sbar.addPermanentWidget(lbl)

        self._refresh_status()

    def _refresh_status(self):
        n = self.table_model.rowCount()
        if n == 0:
            self.lbl_flows.setText("Flows: 0")
            self.lbl_normal.setText("Normal: 0")
            self.lbl_attack.setText("Attack: 0")
            self.lbl_fb.setText("Feedback: 0")
        else:
            df  = self.table_model._df
            atk = int((df.get("prediction", pd.Series(dtype=str)) == "Attack").sum())
            fb  = int(
                df.get("human_label", pd.Series(dtype=str))
                .fillna("").astype(str).str.strip()
                .apply(lambda x: x not in ("", "—")).sum()
            )
            self.lbl_flows.setText(f"Flows: {n}")
            self.lbl_normal.setText(f"Normal: {n - atk}")
            self.lbl_attack.setText(f"Attack: {atk}")
            self.lbl_fb.setText(f"Feedback: {fb}")

        self.lbl_buffers.setText(
            f"Replay Buffer: {self.engine.replay_buffer.count:,} samples"
        )

        self._check_attack_rate()

    # ══════════════════════════════════════════════════════════════════════
    #  CSV open / export
    # ══════════════════════════════════════════════════════════════════════

    def _open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CICFlowMeter CSV", "",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        self.setEnabled(False)
        self.sbar.showMessage("Running inference…")
        QApplication.processEvents()
        try:
            raw = pd.read_csv(path)
            results = self.engine.predict(raw)
            # Ensure human_label column exists and is visible
            if "human_label" not in results.columns:
                results.insert(len(results.columns), "human_label", "—")
            else:
                results["human_label"] = results["human_label"].fillna("—")
            ordered = self._reorder(results)
            self.table_model.set_dataframe(ordered)
            self._auto_resize()
            self._refresh_status()
            self.sbar.showMessage(
                f"Loaded {len(ordered)} flows from {Path(path).name}", 5000,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Prediction Error", str(exc))
        finally:
            self.setEnabled(True)

    def _export_csv(self):
        if self.table_model._df.empty:
            QMessageBox.information(self, "Export", "No data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "ids_results.csv", "CSV (*.csv)",
        )
        if path:
            self.table_model._df.to_csv(path, index=False)
            self.sbar.showMessage(f"Exported to {path}", 5000)

    # ══════════════════════════════════════════════════════════════════════
    #  Right-click feedback
    # ══════════════════════════════════════════════════════════════════════

    def _on_context_menu(self, pos):
        proxy_idx = self.table.indexAt(pos)
        if not proxy_idx.isValid():
            return
        source_idx = self.proxy.mapToSource(proxy_idx)
        row = source_idx.row()
        self._show_feedback_menu(row, self.table.viewport().mapToGlobal(pos))

    def _show_feedback_menu(self, row: int, global_pos=None):
        human = self.table_model._cell_str(row, "human_label")

        menu = QMenu(self)
        menu.setTitle("Human Feedback")

        a_normal = QAction("Mark as Normal", self)
        a_normal.triggered.connect(lambda: self._feedback(row, "Normal"))
        if human == "Normal":
            a_normal.setEnabled(False)
        menu.addAction(a_normal)

        a_attack = QAction("Mark as Attack", self)
        a_attack.triggered.connect(lambda: self._feedback(row, "Attack"))
        if human == "Attack":
            a_attack.setEnabled(False)
        menu.addAction(a_attack)

        menu.addSeparator()
        a_clear = QAction("Clear Feedback", self)
        a_clear.triggered.connect(lambda: self._clear_feedback(row))
        a_clear.setEnabled(bool(human))
        menu.addAction(a_clear)

        if global_pos is None:
            global_pos = self.cursor().pos()
        menu.exec(global_pos)

    def _feedback(self, row: int, label: str):
        sample = self.table_model.get_row(row)
        self.engine.submit_feedback(sample, label)
        self.table_model.update_cell(row, "human_label", label)
        self._refresh_status()

    def _clear_feedback(self, row: int):
        self.table_model.update_cell(row, "human_label", "—")
        self._refresh_status()

    # ══════════════════════════════════════════════════════════════════════
    #  SHAP inline panel (double-click to expand / collapse)
    # ══════════════════════════════════════════════════════════════════════

    def _on_double_click(self, proxy_idx):
        source_idx = self.proxy.mapToSource(proxy_idx)
        row = source_idx.row()

        # Double-click on human_label column → open feedback menu instead
        col_name = self.table_model.col_name(source_idx.column())
        if col_name == "human_label":
            self._show_feedback_menu(row)
            return

        # Toggle: double-click same row → collapse
        if self._expanded_row == row and self._shap_panel.isVisible():
            self._collapse_shap()
            return

        sample = self.table_model.get_row(row)
        pred = self.table_model._cell_str(row, "prediction")
        conf = self.table_model._cell_str(row, "confidence")

        self.sbar.showMessage("Computing SHAP explanations…")
        QApplication.processEvents()
        try:
            expl = self.engine.explain(sample)
        except Exception as exc:
            QMessageBox.warning(self, "SHAP Error", str(exc))
            self.sbar.clearMessage()
            return
        self.sbar.clearMessage()

        if not expl:
            QMessageBox.information(self, "SHAP", "No SHAP explainers loaded.")
            return

        self._show_shap(row, pred, conf, expl)

    def _show_shap(self, row, pred, conf, expl):
        # Clear previous canvases
        self._clear_shap_canvases()
        plt.close("all")

        self._expanded_row = row
        self._shap_header.setText(
            f"▼  SHAP Explanation — Row {row + 1}   "
            f"[{pred} @ {conf}]   (double-click row to collapse)"
        )

        for key, title in [("ae", "AE Reconstruction Error"), ("gmm", "GMM Anomaly Score")]:
            if key not in expl:
                continue
            fig = plt.figure(constrained_layout=True)
            shap.plots.waterfall(expl[key], max_display=15, show=False)
            fig.suptitle(title, fontsize=10)
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            canvas.setMinimumHeight(350)
            self._shap_canvas_container.addWidget(canvas)

        self._shap_panel.setVisible(True)

    def _collapse_shap(self):
        self._shap_panel.setVisible(False)
        self._clear_shap_canvases()
        self._expanded_row = -1
        plt.close("all")

    def _clear_shap_canvases(self):
        while self._shap_canvas_container.count():
            item = self._shap_canvas_container.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

    # ══════════════════════════════════════════════════════════════════════
    #  Environment Adaptation
    # ══════════════════════════════════════════════════════════════════════

    def _check_attack_rate(self):
        """Show/hide the warning banner based on current attack rate."""
        df = self.table_model._df
        if df.empty or "prediction" not in df.columns:
            self._warn_banner.setVisible(False)
            return

        rate = self.engine.check_attack_rate(df)
        if rate >= ATTACK_RATE_TRIGGER and len(df) >= 20:
            n_atk = int((df["prediction"] == "Attack").sum())
            self._warn_banner.setText(
                f"⚠  High attack rate detected: {rate:.0%} "
                f"({n_atk}/{len(df)} flows).  "
                f"The model may not understand this network's normal traffic.  "
                f"Click \"Adapt Model\" to fine-tune."
            )
            self._warn_banner.setVisible(True)
        else:
            self._warn_banner.setVisible(False)

    def _adjust_threshold(self):
        """Let the user adjust the anomaly threshold without retraining."""
        from retrain import compute_percentiles

        df = self.table_model._df
        if df.empty:
            QMessageBox.information(
                self, "Adjust Threshold",
                "No flow data loaded.\n"
                "Capture or load traffic first so the score distribution "
                "can be visualized."
            )
            return

        # Score current data with existing model
        X = df[self.engine.features].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.engine.scaler.transform(X)
        X_recon = self.engine.ae_model.predict(X_scaled, verbose=0)
        errors = np.abs(X_scaled - X_recon)
        scores = self.engine.gmm.score_samples(errors)
        percentiles = compute_percentiles(scores)

        dlg = ThresholdPickerDialog(
            scores=scores,
            auto_threshold=self.engine.threshold,
            percentiles=percentiles,
            parent=self,
        )
        # Relabel for this context
        dlg.setWindowTitle("Adjust Anomaly Threshold")
        dlg.radio_auto.setText(
            f"Current threshold = {self.engine.threshold:.4f}"
        )

        if dlg.exec() == QDialog.Accepted:
            chosen = dlg.get_threshold()
            old = self.engine.threshold
            if chosen != old:
                self.engine.threshold = chosen
                self._repredict_all()
                logger.info("Threshold changed: %.4f → %.4f", old, chosen)
                self.sbar.showMessage(
                    f"Threshold updated: {old:.4f} → {chosen:.4f}  |  "
                    f"All {len(df)} rows re-predicted.", 10000
                )

    def _start_adaptation(self):
        """Collect data and launch the adaptation workflow.

        Two modes:
          - Initial (first adaptation): retrain from scratch
          - Incremental (subsequent): MAS+Replay hybrid
        """
        df = self.table_model._df
        if df.empty:
            QMessageBox.information(
                self, "Adapt Model",
                "No flow data to adapt from.\n"
                "Capture or load traffic first, then adapt."
            )
            return

        n_total = len(df)
        n_atk = int((df.get("prediction", pd.Series(dtype=str)) == "Attack").sum())
        n_norm = n_total - n_atk

        # Determine mode: initial if replay buffer is empty, else incremental
        has_replay = not self.engine.replay_buffer.is_empty
        mode = "incremental" if has_replay else "initial"

        if mode == "initial":
            msg = (
                f"<b>Initial Environment Adaptation (from scratch)</b><br><br>"
                f"Current flows: <b>{n_total}</b> "
                f"(Normal: {n_norm}, Attack: {n_atk})<br><br>"
                f"This will:<br>"
                f"  1. Treat <b>all {n_total} flows</b> as normal baseline<br>"
                f"  2. Fit a <b>new scaler</b> to this environment<br>"
                f"  3. Train a <b>new AE</b> from scratch<br>"
                f"  4. Fit a <b>new GMM</b> on reconstruction errors<br>"
                f"  5. Let you pick an anomaly threshold<br><br>"
                f"<i>After this, subsequent adaptations will use MAS+Replay<br>"
                f"to preserve what the model learned here.</i><br><br>"
                f"Continue?"
            )
        else:
            n_replay = self.engine.replay_buffer.count
            msg = (
                f"<b>Incremental Drift Adaptation (MAS+Replay)</b><br><br>"
                f"New flows: <b>{n_total}</b> | "
                f"Replay buffer: <b>{n_replay:,}</b> samples<br><br>"
                f"This will:<br>"
                f"  1. Checkpoint current model<br>"
                f"  2. Compute MAS importance on replay buffer<br>"
                f"  3. Fine-tune AE with MAS penalty (λ=50) + replay mixing (30%)<br>"
                f"  4. Refit GMM on mixed reconstruction errors<br>"
                f"  5. Let you review and accept/reject<br><br>"
                f"<i>You can roll back if results are unsatisfactory.</i><br><br>"
                f"Continue?"
            )

        reply = QMessageBox.question(
            self, "Adapt Model",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        # Disable UI during retraining
        self.setEnabled(False)

        # Progress bar in status bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 5)
        self._progress.setTextVisible(True)
        self.sbar.addWidget(self._progress)

        # Load replay data from disk for incremental mode
        replay_data = self.engine.replay_buffer.load() if has_replay else None

        self._retrain_worker = RetrainWorker(
            engine=self.engine,
            flow_data=df,
            mode=mode,
            replay_buffer=replay_data,
        )
        self._retrain_worker.progress.connect(self._on_retrain_progress)
        self._retrain_worker.finished.connect(self._on_retrain_finished)
        self._retrain_worker.failed.connect(self._on_retrain_failed)
        self._retrain_worker.start()

    def _on_retrain_progress(self, step, total, msg):
        self._progress.setValue(step)
        self._progress.setFormat(f"Step {step}/{total}: {msg}")
        self.sbar.showMessage(msg)

    def _on_retrain_finished(self, result):
        self.sbar.removeWidget(self._progress)
        self._progress.deleteLater()

        from retrain import compute_percentiles, save_adapted_model

        mode = result.get("_mode", "initial")
        adapter = result.get("_adapter")
        scores = result.get("scores")
        auto_threshold = result["threshold"]
        percentiles = result.get("percentiles") or compute_percentiles(scores)

        # Show threshold picker dialog
        dlg = ThresholdPickerDialog(
            scores=scores,
            auto_threshold=auto_threshold,
            percentiles=percentiles,
            parent=self,
        )

        self.setEnabled(True)

        if dlg.exec() == QDialog.Accepted:
            chosen = dlg.get_threshold()

            from engine import REPLAY_POST_ADAPT_RATIO, REPLAY_POST_ADAPT_CAP, MAX_TRAINING_SIZE

            if mode == "initial":
                # ── Swap in the new from-scratch model ────────────────
                self.engine.scaler = adapter.scaler
                self.engine.ae_model = adapter.ae_model
                self.engine.gmm = adapter.gmm
                self.engine.threshold = chosen

                # Populate replay buffer: 30% of training samples (capped at 86,400)
                df = self.table_model._df
                X_raw = df[self.engine.features].values.astype(np.float64)
                X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
                X_scaled = adapter.scaler.transform(X_raw)

                n_to_add = min(
                    int(len(X_scaled) * REPLAY_POST_ADAPT_RATIO),
                    REPLAY_POST_ADAPT_CAP
                )
                if n_to_add > 0:
                    idx = np.random.choice(len(X_scaled), n_to_add, replace=False)
                    # Update the replay buffer's feature_names to match new scaler
                    self.engine.replay_buffer.feature_names = self.engine.features
                    self.engine.replay_buffer.clear()
                    self.engine.replay_buffer.add(X_scaled[idx])

                # Save adapted model
                new_dir = save_adapted_model(
                    deploy_dir=self.engine.deploy_dir,
                    ae_model=adapter.ae_model,
                    gmm=adapter.gmm,
                    scaler=adapter.scaler,
                    threshold=chosen,
                    features=self.engine.features,
                    model_id=self.engine.model_id,
                    config=self.engine.config,
                    domains_seen=self.engine.domains_seen,
                )
                self.engine.deploy_dir = new_dir
                self.engine.shap_cache = None

                logger.info("Initial adaptation accepted: threshold=%.4f, "
                            "replay buffer seeded with %d samples", chosen, n_to_add)

            else:
                # ── Incremental mode: accept or rollback ──────────────
                adapter.new_threshold = chosen
                new_gmm, _ = adapter.accept()
                self.engine.gmm = new_gmm
                self.engine.threshold = chosen

                # Add 30% of current in-memory flows to replay buffer (cap 86,400)
                df = self.table_model._df
                X_raw = df[self.engine.features].values.astype(np.float64)
                X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
                X_new_scaled = self.engine.scaler.transform(X_raw)

                n_to_add = min(
                    int(len(X_new_scaled) * REPLAY_POST_ADAPT_RATIO),
                    REPLAY_POST_ADAPT_CAP
                )
                if n_to_add > 0:
                    idx = np.random.choice(len(X_new_scaled), n_to_add, replace=False)
                    self.engine.replay_buffer.add(X_new_scaled[idx])

                # Save
                new_dir = save_adapted_model(
                    deploy_dir=self.engine.deploy_dir,
                    ae_model=self.engine.ae_model,
                    gmm=new_gmm,
                    scaler=self.engine.scaler,
                    threshold=chosen,
                    features=self.engine.features,
                    model_id=self.engine.model_id,
                    config=self.engine.config,
                    domains_seen=self.engine.domains_seen,
                )
                self.engine.deploy_dir = new_dir
                self.engine.shap_cache = None

                logger.info("Incremental adaptation accepted: threshold=%.4f, "
                            "added %d samples to replay", chosen, n_to_add)

            # Re-predict all rows with new model + threshold
            self._repredict_all()

            mode_label = "Initial (from scratch)" if mode == "initial" \
                else "Incremental (MAS+Replay)"
            replay_count = self.engine.replay_buffer.count
            QMessageBox.information(
                self, "Adaptation Complete",
                f"<b>Model adapted successfully!</b><br><br>"
                f"Mode: {mode_label}<br>"
                f"New threshold: <b>{self.engine.threshold:.4f}</b><br>"
                f"Replay buffer: {replay_count:,} samples<br><br>"
                f"All rows re-predicted with the updated model."
            )
        else:
            # User cancelled
            if mode == "incremental" and adapter:
                adapter.rollback()
                logger.info("Incremental adaptation rejected — rolled back")
                QMessageBox.information(
                    self, "Adaptation Cancelled",
                    "Adaptation rejected. Model rolled back to previous state."
                )
            else:
                QMessageBox.information(
                    self, "Adaptation Cancelled",
                    "Adaptation cancelled. No changes were made."
                )

        self._warn_banner.setVisible(False)

    def _on_retrain_failed(self, error_msg):
        self.sbar.removeWidget(self._progress)
        self._progress.deleteLater()
        self.setEnabled(True)
        QMessageBox.critical(
            self, "Adaptation Failed",
            f"Model adaptation failed:\n\n{error_msg}"
        )

    def _repredict_all(self):
        """Re-run prediction on all rows using the current (adapted) engine."""
        df = self.table_model._df
        if df.empty:
            return

        X = df[self.engine.features].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.engine.scaler.transform(X)
        X_recon = self.engine.ae_model.predict(X_scaled, verbose=0)
        errors = np.abs(X_scaled - X_recon)
        scores = self.engine.gmm.score_samples(errors)

        from scipy.special import expit
        predictions = np.where(scores < self.engine.threshold, "Attack", "Normal")
        distance = scores - self.engine.threshold
        pctl95 = np.percentile(np.abs(distance), 95) or 1.0
        confidences = (expit(np.abs(distance / (pctl95 / 3))) - 0.5) * 2

        # Update in place
        self.table_model.beginResetModel()
        self.table_model._df["prediction"] = predictions
        self.table_model._df["confidence"] = np.round(confidences, 4)
        self.table_model._df["gmm_score"] = np.round(scores, 4)
        self.table_model.endResetModel()

        self._refresh_status()
        self._check_attack_rate()

    # ══════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _reorder(self, df: pd.DataFrame) -> pd.DataFrame:
        """Re-order columns to: meta → features → results, deduplicate."""
        # Drop the engine's lowercase 'timestamp' — keep the CSV's 'Timestamp'
        if "Timestamp" in df.columns and "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])
        elif "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "Timestamp"})
        # Engine may produce duplicate Dst Port (meta + feature)
        df = df.loc[:, ~df.columns.duplicated()]
        cols  = [c for c in self._display_order if c in df.columns]
        extra = [c for c in df.columns if c not in cols]
        return df[cols + extra]

    def _clear(self):
        self.table_model.clear()
        with self._lock:
            self._pending_flows.clear()
        self._refresh_status()

    def _auto_resize(self):
        hdr = self.table.horizontalHeader()
        widths = {
            "Timestamp": 160, "Flow ID": 220,
            "Src IP": 130, "Dst IP": 130,
            "Src Port": 75,  "Dst Port": 75,
            "prediction": 90, "confidence": 85,
            "gmm_score": 95, "human_label": 100,
        }
        for i, col in enumerate(self.table_model._cols):
            hdr.resizeSection(i, widths.get(col, 120))


# ═════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Catch unhandled exceptions so crashes print a traceback
    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_tb)
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _excepthook

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
