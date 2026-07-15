"""Test capture pipeline components.

Note: the CICFlowMeterRunner test requires the separate (Java-based)
CICFlowMeter project as a sibling directory — see config.CICFLOWMETER_DIR.
It is optional; the default live-capture path (flow_extractor.py) does
not need it.

Run from the repo root (or anywhere) with:
    python dev_scripts/test_capture.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from capture import list_interfaces, CICFlowMeterRunner
from config import CICFLOWMETER_DIR

print("=== Network Interfaces ===")
ifaces = list_interfaces()
for ifc in ifaces:
    print(f"  {ifc['name']:30s} {ifc['ip']:20s} {ifc['network_name']}")
print(f"\nTotal: {len(ifaces)} active interfaces")

print("\n=== CICFlowMeter Runner ===")
try:
    cfm = CICFlowMeterRunner(str(CICFLOWMETER_DIR))
    print(f"Classes dir: {cfm.classes_dir}")
    print(f"jnetpcap DLL: {cfm.jnetpcap_dll_dir}")
    print(f"Classpath entries: {len(cfm._classpath.split(';'))}")
    print("CICFlowMeter runner: OK")
except Exception as e:
    print(f"CICFlowMeter runner error: {e}")

print("\n✓ All capture components ready.")
