"""End-to-end test: capture 5s -> CICFlowMeter -> inference.

Note: requires the separate (Java-based) CICFlowMeter project as a
sibling directory — see config.CICFLOWMETER_DIR. This is the legacy
capture path; the default live-capture path (flow_extractor.py) does
not need CICFlowMeter.

Run from the repo root (or anywhere) with:
    python dev_scripts/test_e2e.py
"""
import sys, threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from capture import CICFlowMeterRunner, _capture_to_pcap
from config import CICFLOWMETER_DIR, DEPLOY_DIR, BUFFER_DIR, BUFFER_MAX_SAMPLES, ATTACK_CONFIDENCE_THRESHOLD
from engine import M3InferenceEngine
import pandas as pd

work = REPO_ROOT / "dev_scripts" / "_work" / "live_capture"
pcap_dir = work / "pcaps"
csv_dir = work / "csvs"
pcap_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)

# 1. Capture 5 seconds of Wi-Fi traffic
print("Step 1: Capturing 5 seconds of network traffic...")
pcap_file = pcap_dir / "e2e_test.pcap"
stop = threading.Event()
# NOTE: replace with an interface name from list_interfaces() (see test_capture.py)
n = _capture_to_pcap(
    interface=r"\Device\NPF_{REPLACE-WITH-YOUR-INTERFACE-GUID}",
    output_file=pcap_file,
    duration=5,
    stop_event=stop,
)
print(f"  Captured {n} packets → {pcap_file.name}")

if n == 0:
    print("  No packets captured. Is the interface active?")
    sys.exit(1)

# 2. Extract features via CICFlowMeter
print("\nStep 2: Extracting flow features via CICFlowMeter...")
cfm = CICFlowMeterRunner(str(CICFLOWMETER_DIR))
csv_file = cfm.extract_features(pcap_file, csv_dir)

if csv_file is None:
    print("  CICFlowMeter produced no output. Check Java logs.")
    sys.exit(1)

df = pd.read_csv(csv_file)
print(f"  {len(df)} flows extracted, {len(df.columns)} columns")
print(f"  Columns: {list(df.columns[:10])}...")

# 3. Run inference
print("\nStep 3: Running AE-GMM inference...")
engine = M3InferenceEngine(
    deploy_dir=str(DEPLOY_DIR),
    buffer_dir=str(BUFFER_DIR),
    buffer_max=BUFFER_MAX_SAMPLES,
    attack_conf_threshold=ATTACK_CONFIDENCE_THRESHOLD,
)
results = engine.predict(df)
n_normal = (results["prediction"] == "Normal").sum()
n_attack = (results["prediction"] == "Attack").sum()
print(f"  Results: {len(results)} flows — {n_normal} Normal, {n_attack} Attack")
print(f"  Avg confidence: {results['confidence'].mean():.1%}")
if not results.empty:
    print(f"\n  Sample:\n{results[['prediction','confidence','gmm_score']].head()}")

print("\n✓ End-to-end pipeline works!")
