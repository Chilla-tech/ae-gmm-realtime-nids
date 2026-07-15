"""
M3 AE-GMM IDS — Live Monitoring Dashboard
──────────────────────────────────────────
Launch:  streamlit run app.py

Features:
  • Upload CSV or watch directory for CICFlowMeter output
  • Tabular results with clickable rows → SHAP waterfall plots
  • Human-in-the-loop feedback: confirm or correct predictions
  • Feedback routes samples to the correct buffer for retraining
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEPLOY_DIR,
    BUFFER_DIR,
    BUFFER_MAX_SAMPLES,
    ATTACK_CONFIDENCE_THRESHOLD,
    WATCH_DIR,
    SHAP_MAX_DISPLAY,
    CICFLOWMETER_DIR,
)
from engine import M3InferenceEngine
from capture import LiveCapturePipeline, list_interfaces

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="M3 AE-GMM IDS", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_engine():
    return M3InferenceEngine(
        deploy_dir=str(DEPLOY_DIR),
        buffer_dir=str(BUFFER_DIR),
        buffer_max=BUFFER_MAX_SAMPLES,
        attack_conf_threshold=ATTACK_CONFIDENCE_THRESHOLD,
    )


engine = load_engine()

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🛡️ M3 AE-GMM Intrusion Detection System")
st.caption(
    f"Model **{engine.model_id}**  ·  "
    f"GMM threshold **{engine.threshold:.4f}**  ·  "
    f"**{len(engine.features)}** features  ·  "
    f"Domains: {', '.join(engine.domains_seen)}"
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input Mode")
    mode = st.radio(
        "Input Mode", ["🔴 Live Capture", "📂 Upload CSV", "👁️ Watch Directory"],
        label_visibility="collapsed",
    )

    st.divider()
    st.header("📊 Retraining Buffers")
    stats = engine.buffer_stats
    c1, c2 = st.columns(2)
    c1.metric("Benign", f"{stats['benign_count']:,}")
    c2.metric("Attack (≥60%)", f"{stats['attack_count']:,}")
    st.metric("Human Feedback", f"{engine.feedback_count:,}")
    st.caption(f"Max capacity: {stats['buffer_max']:,} per buffer")

    if st.button("🗑️ Clear Buffers"):
        engine.benign_buffer.clear()
        engine.attack_buffer.clear()
        st.rerun()

    st.divider()
    with st.expander("ℹ️ Model Details"):
        st.json({
            "model_id": engine.model_id,
            "config": engine.config,
            "threshold": engine.threshold,
            "features": engine.features,
        })


# ═══════════════════════════════════════════════════════════════════════════
#  Result viewer with SHAP + Human Feedback
# ═══════════════════════════════════════════════════════════════════════════

def show_results(results: pd.DataFrame, key_prefix: str = "main"):
    """Render results table, clickable SHAP panel, and human feedback UI."""

    n_total = len(results)
    n_normal = (results["prediction"] == "Normal").sum()
    n_attack = (results["prediction"] == "Attack").sum()
    avg_conf = results["confidence"].mean()

    # ── summary metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Flows", f"{n_total:,}")
    m2.metric("Normal", f"{n_normal:,}")
    m3.metric("Attack", f"{n_attack:,}", delta_color="inverse")
    m4.metric("Avg Confidence", f"{avg_conf:.1%}")

    # ── display columns ──
    disp = ["prediction", "confidence", "gmm_score"]
    for extra in ["Src IP", "Dst IP", "Dst Port"]:
        if extra in results.columns:
            disp.insert(0, extra)
    disp += engine.features[:5]

    st.subheader("Inference Results")
    st.caption("👆 Click a row to see its SHAP explanation and provide feedback")

    event = st.dataframe(
        results[disp],
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1, format="%.1f%%",
            ),
            "gmm_score": st.column_config.NumberColumn("GMM Score", format="%.2f"),
        },
        key=f"{key_prefix}_table",
    )

    # ── Selected row: SHAP + Feedback ──
    if not (event.selection and event.selection.rows):
        return

    idx = event.selection.rows[0]
    row = results.iloc[idx]

    st.divider()

    # ── prediction banner ──
    icon = "🟢" if row["prediction"] == "Normal" else "🔴"
    st.subheader(f"🔍 Flow #{idx}  —  {icon} {row['prediction']}")

    info_cols = st.columns(3)
    info_cols[0].metric("Confidence", f"{row['confidence']:.1%}")
    info_cols[1].metric("GMM Score", f"{row['gmm_score']:.2f}")
    info_cols[2].metric("Threshold", f"{engine.threshold:.2f}")

    # ── SHAP waterfall plots ──
    with st.spinner("Computing SHAP values …"):
        shap_result = engine.explain(row)

    col_ae, col_gmm = st.columns(2)

    with col_ae:
        st.markdown("**Level 1 — AE Reconstruction Error**")
        if "ae" in shap_result:
            shap.plots.waterfall(
                shap_result["ae"], max_display=SHAP_MAX_DISPLAY, show=False,
            )
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close("all")

    with col_gmm:
        st.markdown("**Level 2 — GMM Anomaly Score**")
        if "gmm" in shap_result:
            shap.plots.waterfall(
                shap_result["gmm"], max_display=SHAP_MAX_DISPLAY, show=False,
            )
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close("all")

    # ══════════════════════════════════════════════════════════════════════
    #  Human Feedback Panel
    # ══════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🧑‍💻 Human Feedback")
    st.markdown(
        "Review the SHAP explanation above and confirm or correct the prediction.  \n"
        "The sample will be placed in the **correct buffer** for next retraining."
    )

    fb_key = f"{key_prefix}_fb_{idx}"

    fb_col1, fb_col2, fb_col3 = st.columns([2, 2, 3])

    with fb_col1:
        st.markdown(f"**Model says:** {icon} **{row['prediction']}**")

    with fb_col2:
        human_label = st.radio(
            "Your label:",
            ["Normal", "Attack"],
            index=0 if row["prediction"] == "Normal" else 1,
            key=f"{fb_key}_radio",
            horizontal=True,
        )

    with fb_col3:
        agrees = human_label == row["prediction"]
        if agrees:
            st.success("✓ Agrees with model")
        else:
            st.warning("✗ Overrides model prediction")

    if st.button(
        f"✅ Submit → **{human_label}** buffer",
        key=f"{fb_key}_submit",
        type="primary",
    ):
        dest = engine.submit_feedback(row, human_label)
        st.toast(
            f"Flow #{idx} → **{dest}** buffer  "
            + ("(confirmed)" if agrees else "(corrected)"),
            icon="✅" if agrees else "🔄",
        )
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  Mode: Live Capture
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_pipeline():
    work_dir = Path(__file__).resolve().parent / "live_capture"
    return LiveCapturePipeline(
        engine=engine,
        cicflowmeter_dir=str(CICFLOWMETER_DIR),
        work_dir=str(work_dir),
    )


if mode == "🔴 Live Capture":
    st.subheader("🔴 Live Network Capture")
    st.markdown(
        "Continuously **captures** network traffic → extracts flow features "
        "via CICFlowMeter → runs AE-GMM inference in real time."
    )

    pipeline = get_pipeline()
    p_stats = pipeline.stats

    # ── interface selector + controls ──
    col_if, col_dur = st.columns([3, 1])
    with col_if:
        ifaces = list_interfaces()
        iface_labels = [
            f"{ifc['name']}  ({ifc['description']})  [{ifc['ip']}]"
            if ifc["description"]
            else f"{ifc['name']}  [{ifc['ip']}]"
            for ifc in ifaces
        ] if ifaces else ["No interfaces found"]

        selected_idx = st.selectbox(
            "Network Interface",
            range(len(iface_labels)),
            format_func=lambda i: iface_labels[i],
            disabled=pipeline.is_running,
        )
    with col_dur:
        capture_secs = st.number_input(
            "Window (s)", min_value=5, max_value=300, value=30, step=5,
            disabled=pipeline.is_running,
            help="Duration of each capture window before processing",
        )

    # ── start / stop ──
    col_start, col_stop, col_clear, _ = st.columns([1, 1, 1, 3])
    with col_start:
        if st.button(
            "▶️ Start", type="primary", disabled=pipeline.is_running or not ifaces,
        ):
            pipeline.capture_seconds = capture_secs
            # Use network_name (NPF device path) for scapy on Windows
            iface_id = ifaces[selected_idx].get("network_name") or ifaces[selected_idx]["name"]
            pipeline.start(iface_id)
            st.rerun()
    with col_stop:
        if st.button("⏹️ Stop", disabled=not pipeline.is_running):
            pipeline.stop()
            st.rerun()
    with col_clear:
        if st.button("🗑️ Clear Results"):
            pipeline.clear_results()
            st.rerun()

    # ── live status ──
    if pipeline.is_running:
        st.success(
            f"🟢 Capturing on **{p_stats['interface']}** "
            f"(window: {pipeline.capture_seconds}s) — "
            f"started {p_stats['started_at']}"
        )
    elif p_stats["status"] == "stopping":
        st.warning("⏳ Stopping after current capture window…")
    else:
        st.info("Pipeline is **stopped**. Select an interface and click Start.")

    # ── stats bar ──
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Captures", p_stats["captures"])
    s2.metric("Packets", f"{p_stats['total_packets']:,}")
    s3.metric("Flows", f"{p_stats['total_flows']:,}")
    s4.metric("Attacks", f"{p_stats['attacks_detected']:,}")
    s5.metric("Errors", len(p_stats.get("errors", [])))

    if p_stats.get("errors"):
        with st.expander("⚠️ Errors"):
            for err in p_stats["errors"][-10:]:
                st.code(err)

    # ── results ──
    live_results = pipeline.get_results()
    if not live_results.empty:
        show_results(live_results, key_prefix="live")
    elif pipeline.is_running:
        st.info("Waiting for first capture window to complete…")

    # ── auto-refresh while running ──
    if pipeline.is_running:
        import time
        time.sleep(5)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  Mode: Upload CSV
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "📂 Upload CSV":
    uploaded = st.file_uploader(
        "Upload a CICFlowMeter CSV",
        type=["csv"],
        help="CSV produced by CICFlowMeter from pcap capture",
    )
    if uploaded:
        df = pd.read_csv(uploaded)
        st.info(f"Loaded **{len(df):,}** flows from `{uploaded.name}`")
        with st.spinner("Running AE-GMM inference …"):
            results = engine.predict(df)
        st.session_state["upload_results"] = results

    if "upload_results" in st.session_state:
        show_results(st.session_state["upload_results"], key_prefix="upload")


# ═══════════════════════════════════════════════════════════════════════════
#  Mode: Watch Directory
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "👁️ Watch Directory":  # noqa: RUF001
    watch_path = st.text_input(
        "Directory to watch for CICFlowMeter CSVs", str(WATCH_DIR),
    )
    col_r, col_a = st.columns([1, 4])
    with col_r:
        refresh = st.button("🔄 Refresh")
    with col_a:
        auto = st.checkbox("Auto-refresh (every 10 s)", value=False)

    watch_dir = Path(watch_path)

    if not watch_dir.exists():
        st.warning(f"Directory does not exist: `{watch_path}`")
        if st.button("Create it"):
            watch_dir.mkdir(parents=True, exist_ok=True)
            st.rerun()
    else:
        csv_files = sorted(
            watch_dir.glob("*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not csv_files:
            st.info(
                "No CSV files yet.  "
                "Run CICFlowMeter and point its output here:\n\n"
                f"```\njava -jar CICFlowMeter.jar <pcap> {watch_dir}\n```"
            )
        else:
            need_run = refresh or auto or "watch_results" not in st.session_state
            if need_run:
                all_r = []
                bar = st.progress(0, text="Processing …")
                for i, fp in enumerate(csv_files):
                    try:
                        chunk = pd.read_csv(fp)
                        r = engine.predict(chunk)
                        r.insert(0, "source_file", fp.name)
                        all_r.append(r)
                    except Exception as exc:
                        st.warning(f"Skipped `{fp.name}`: {exc}")
                    bar.progress((i + 1) / len(csv_files))
                bar.empty()
                if all_r:
                    st.session_state["watch_results"] = pd.concat(
                        all_r, ignore_index=True,
                    )

            if "watch_results" in st.session_state:
                show_results(
                    st.session_state["watch_results"], key_prefix="watch",
                )

    if auto:
        import time
        time.sleep(10)
        st.rerun()
