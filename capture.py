"""
Live Capture Pipeline
─────────────────────
Continuously captures network traffic, extracts CICFlowMeter features,
and runs AE-GMM inference — all in a background thread.

Pipeline:
  [Network Interface]
        │  scapy sniff → rotating pcap files
        ▼
  [CICFlowMeter CLI]
        │  pcap → CSV (84 flow features)
        ▼
  [M3InferenceEngine]
        │  18 selected features → AE → GMM → classify
        ▼
  [Results queue → Streamlit dashboard]
"""

import logging
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  CICFlowMeter wrapper
# ═══════════════════════════════════════════════════════════════════════════

class CICFlowMeterRunner:
    """Run CICFlowMeter CLI (Cmd class) on pcap files to produce flow CSVs."""

    # Subset of JARs needed at runtime (resolved from Gradle cache)
    _DEP_PATTERNS = [
        "jnetpcap*.jar",
        "slf4j-api*.jar",
        "slf4j-log4j12*.jar",
        "log4j-1*.jar",
        "log4j-core*.jar",
        "log4j-api-2*.jar",
        "commons-io*.jar",
        "commons-lang3*.jar",
        "commons-math3*.jar",
        "guava*.jar",
        "weka-stable*.jar",
        "jfreechart*.jar",
        "tika-core*.jar",
        "java-cup*.jar",
    ]

    def __init__(self, cicflowmeter_dir: str):
        self.cfm_dir = Path(cicflowmeter_dir)
        self.classes_dir = self.cfm_dir / "build" / "classes" / "java" / "main"
        self.jnetpcap_dll_dir = self.cfm_dir / "jnetpcap" / "win" / "jnetpcap-1.4.r1425"
        self.jnetpcap_jar = self.jnetpcap_dll_dir / "jnetpcap.jar"

        if not self.classes_dir.exists():
            raise FileNotFoundError(
                f"CICFlowMeter classes not compiled.  "
                f"Run 'gradlew build' in {self.cfm_dir}"
            )

        self._classpath = self._build_classpath()
        logger.info("CICFlowMeter classpath: %d entries", len(self._classpath.split(";")))

    def _build_classpath(self) -> str:
        """Resolve JARs from the Gradle cache + compiled classes."""
        entries = [str(self.classes_dir), str(self.jnetpcap_jar)]

        # Search Gradle cache
        gradle_cache = Path.home() / ".gradle" / "caches" / "modules-2" / "files-2.1"
        if gradle_cache.exists():
            for pattern in self._DEP_PATTERNS:
                matches = list(gradle_cache.rglob(pattern))
                # skip sources/javadoc JARs
                matches = [
                    m for m in matches
                    if "sources" not in m.name and "javadoc" not in m.name
                ]
                if matches:
                    entries.append(str(matches[0]))

        return ";".join(entries)

    def extract_features(self, pcap_file: Path, output_dir: Path) -> Optional[Path]:
        """
        Run CICFlowMeter on a pcap file.

        Returns the path to the generated CSV, or None on failure.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "java",
            f"-Djava.library.path={self.jnetpcap_dll_dir}",
            "-cp", self._classpath,
            "cic.cs.unb.ca.ifm.Cmd",
            str(pcap_file),
            str(output_dir),
        ]

        logger.info("Running CICFlowMeter: %s → %s", pcap_file.name, output_dir)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.cfm_dir),
            )

            if result.returncode != 0:
                logger.warning("CICFlowMeter stderr: %s", result.stderr[-500:] if result.stderr else "")

            # CICFlowMeter writes: <pcap_name>_Flow.csv
            expected_csv = output_dir / f"{pcap_file.name}_Flow.csv"
            if expected_csv.exists() and expected_csv.stat().st_size > 0:
                logger.info("CICFlowMeter produced: %s (%d bytes)", expected_csv.name, expected_csv.stat().st_size)
                return expected_csv

            # Fallback: look for any new CSV
            csvs = sorted(output_dir.glob("*_Flow.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if csvs:
                return csvs[0]

            logger.warning("CICFlowMeter produced no CSV for %s", pcap_file.name)
            return None

        except subprocess.TimeoutExpired:
            logger.error("CICFlowMeter timed out on %s", pcap_file.name)
            return None
        except FileNotFoundError:
            logger.error("Java not found — is it on PATH?")
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  Packet capture using scapy
# ═══════════════════════════════════════════════════════════════════════════

def list_interfaces(include_monitor: bool = True) -> List[dict]:
    """List available network interfaces using scapy (Windows-compatible).

    Parameters
    ----------
    include_monitor : bool
        If True, also return interfaces without an IP address
        (useful for SPAN / port-mirror / monitor NICs).
    """
    try:
        from scapy.all import IFACES
        ifaces = []

        # Skip these virtual/system adapters that can't capture real traffic
        _SKIP_DESCS = {"WAN Miniport"}

        for _key, iface in IFACES.data.items():
            ip = getattr(iface, "ip", "") or ""
            mac = getattr(iface, "mac", "") or ""
            network_name = getattr(iface, "network_name", "") or ""
            name = getattr(iface, "name", "") or network_name
            description = getattr(iface, "description", "") or ""

            # Skip WAN Miniport adapters (system-level, cannot capture)
            if any(skip in description for skip in _SKIP_DESCS):
                continue

            # Determine interface category
            is_monitor = not ip          # no IP at all
            is_link_local = ip.startswith("169.254.")

            # For monitor interfaces, require a MAC to filter out truly
            # disconnected/virtual stubs
            if is_monitor and not include_monitor:
                continue
            if is_monitor and not mac:
                continue

            ifaces.append({
                "name": name,
                "description": description,
                "ip": ip,
                "mac": mac,
                "network_name": network_name,
                "is_monitor": is_monitor,
                "is_link_local": is_link_local,
            })
        return ifaces
    except Exception as e:
        logger.error("Failed to list interfaces: %s", e)
        return []


def _capture_to_pcap(
    interface: str,
    output_file: Path,
    duration: int,
    stop_event: threading.Event,
):
    """Capture packets for `duration` seconds and save to pcap."""
    from scapy.all import sniff, wrpcap

    logger.info("Capturing on '%s' for %ds → %s", interface, duration, output_file.name)

    packets = sniff(
        iface=interface,
        timeout=duration,
        store=True,
        stop_filter=lambda _: stop_event.is_set(),
    )

    if packets:
        wrpcap(str(output_file), packets)
        logger.info("Captured %d packets → %s", len(packets), output_file.name)
    else:
        logger.info("No packets captured in this window")

    return len(packets) if packets else 0


# ═══════════════════════════════════════════════════════════════════════════
#  LiveCapturePipeline — orchestrates capture → extract → infer
# ═══════════════════════════════════════════════════════════════════════════

class LiveCapturePipeline:
    """
    Continuous capture → CICFlowMeter → AE-GMM inference pipeline.

    Runs in a daemon thread.  Results accumulate in a thread-safe list
    that the Streamlit UI can poll.
    """

    def __init__(
        self,
        engine,
        cicflowmeter_dir: str,
        work_dir: str,
        capture_seconds: int = 30,
    ):
        self.engine = engine
        self.cfm = CICFlowMeterRunner(cicflowmeter_dir)
        self.work_dir = Path(work_dir)
        self.pcap_dir = self.work_dir / "pcaps"
        self.csv_dir = self.work_dir / "csvs"
        self.capture_seconds = capture_seconds

        self.pcap_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe state
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._all_results: List[pd.DataFrame] = []
        self._stats = {
            "status": "stopped",
            "interface": "",
            "captures": 0,
            "total_packets": 0,
            "total_flows": 0,
            "attacks_detected": 0,
            "started_at": None,
            "last_capture_at": None,
            "errors": [],
        }

    # ── public API ─────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> dict:
        with self._lock:
            s = dict(self._stats)
            s["is_running"] = self.is_running
            s["result_count"] = sum(len(r) for r in self._all_results)
            return s

    def get_results(self) -> pd.DataFrame:
        """Return all accumulated results as a single DataFrame."""
        with self._lock:
            if not self._all_results:
                return pd.DataFrame()
            return pd.concat(self._all_results, ignore_index=True)

    def start(self, interface: str):
        """Start the capture pipeline in a background thread."""
        if self.is_running:
            logger.warning("Pipeline already running")
            return

        self._stop_event.clear()
        with self._lock:
            self._stats["status"] = "starting"
            self._stats["interface"] = interface
            self._stats["started_at"] = datetime.now().isoformat()
            self._stats["errors"] = []

        self._thread = threading.Thread(
            target=self._pipeline_loop,
            args=(interface,),
            daemon=True,
            name="LiveCapturePipeline",
        )
        self._thread.start()
        logger.info("Pipeline started on interface '%s'", interface)

    def stop(self):
        """Signal the pipeline to stop after the current capture window."""
        if not self.is_running:
            return
        logger.info("Stopping pipeline…")
        self._stop_event.set()
        with self._lock:
            self._stats["status"] = "stopping"

    def clear_results(self):
        """Discard accumulated results."""
        with self._lock:
            self._all_results.clear()

    # ── pipeline loop ──────────────────────────────────────────────────

    def _pipeline_loop(self, interface: str):
        with self._lock:
            self._stats["status"] = "running"

        while not self._stop_event.is_set():
            try:
                self._one_cycle(interface)
            except Exception as exc:
                msg = f"Pipeline error: {exc}"
                logger.error(msg, exc_info=True)
                with self._lock:
                    self._stats["errors"].append(msg)
                # back-off before retry
                if not self._stop_event.wait(5):
                    continue
                else:
                    break

        with self._lock:
            self._stats["status"] = "stopped"
        logger.info("Pipeline stopped.")

    def _one_cycle(self, interface: str):
        """One capture → extract → infer cycle."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pcap_file = self.pcap_dir / f"capture_{ts}.pcap"

        # ── 1. Capture packets ──
        n_packets = _capture_to_pcap(
            interface=interface,
            output_file=pcap_file,
            duration=self.capture_seconds,
            stop_event=self._stop_event,
        )

        with self._lock:
            self._stats["total_packets"] += n_packets
            self._stats["last_capture_at"] = datetime.now().isoformat()

        if n_packets == 0 or not pcap_file.exists():
            return

        # ── 2. Extract features via CICFlowMeter ──
        csv_file = self.cfm.extract_features(pcap_file, self.csv_dir)

        # Clean up pcap (can be large)
        try:
            pcap_file.unlink(missing_ok=True)
        except OSError:
            pass

        if csv_file is None:
            return

        # ── 3. Run inference ──
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                return
            results = self.engine.predict(df)
            results.insert(0, "capture_time", ts)

            n_attacks = int((results["prediction"] == "Attack").sum())

            with self._lock:
                self._all_results.append(results)
                self._stats["captures"] += 1
                self._stats["total_flows"] += len(results)
                self._stats["attacks_detected"] += n_attacks

            logger.info(
                "Cycle %s: %d flows (%d attacks)",
                ts, len(results), n_attacks,
            )
        except Exception as exc:
            logger.warning("Inference failed for %s: %s", csv_file.name, exc)
