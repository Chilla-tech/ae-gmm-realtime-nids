"""Debug scapy interface listing."""
from scapy.all import IFACES, get_if_list, conf
import scapy.interfaces

# Method 1: IFACES
print("=== IFACES.data ===")
for key, iface in list(IFACES.data.items())[:5]:
    attrs = {}
    for a in ["name", "description", "ip", "mac", "network_name"]:
        attrs[a] = getattr(iface, a, "N/A")
    print(f"  {attrs}")

# Method 2: get_if_list
print("\n=== get_if_list ===")
for name in get_if_list()[:5]:
    print(f"  {name}")

# Method 3: Windows specific
print("\n=== conf.iface ===")
print(f"  Default: {conf.iface}")

# Method 4: Try NetworkInterface
try:
    from scapy.arch.windows import get_windows_if_list
    print("\n=== get_windows_if_list ===")
    for iface in get_windows_if_list()[:5]:
        print(f"  name={iface.get('name','?')}, desc={iface.get('description','?')}, ips={iface.get('ips',[][:2])}")
except ImportError:
    print("get_windows_if_list not available")
