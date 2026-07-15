"""
Real-time Flow Feature Extractor
─────────────────────────────────
Pure-Python replacement for CICFlowMeter that computes the 18 features
needed by the M3 AE-GMM model directly from live packets (scapy).

Each flow is emitted the instant it terminates (FIN/RST or idle timeout),
eliminating the batch delay of pcap → CICFlowMeter → CSV.

Usage:
    extractor = RealtimeFlowExtractor(callback=on_flow)
    extractor.start("Ethernet")   # blocks, or call in a thread
    extractor.stop()
"""

import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Flow record — accumulates per-packet statistics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FlowRecord:
    """Tracks packet-level statistics for a single network flow."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int

    # Timestamps
    start_time: float = 0.0
    last_time: float = 0.0

    # Forward (src → dst) packets
    fwd_lengths: list = field(default_factory=list)
    fwd_times: list = field(default_factory=list)
    fwd_init_win: int = -1       # TCP window from first forward packet
    fwd_header_sizes: list = field(default_factory=list)  # transport header lengths

    # Backward (dst → src) packets
    bwd_lengths: list = field(default_factory=list)
    bwd_times: list = field(default_factory=list)

    # All packets
    all_lengths: list = field(default_factory=list)
    all_times: list = field(default_factory=list)

    # Flags
    fin_count: int = 0
    fin_seen_fwd: bool = False
    fin_seen_bwd: bool = False
    rst_seen: bool = False

    # Activity / idle tracking
    _active_threshold: float = 1.0  # seconds — gap > this is "idle"

    def add_packet(self, payload_len: int, timestamp: float, is_forward: bool,
                   tcp_flags: int = 0, tcp_window: int = 0,
                   header_size: int = 0):
        """Register one packet into the flow."""
        if self.start_time == 0:
            self.start_time = timestamp
        self.last_time = timestamp

        self.all_lengths.append(payload_len)
        self.all_times.append(timestamp)

        if is_forward:
            self.fwd_lengths.append(payload_len)
            self.fwd_times.append(timestamp)
            if self.fwd_init_win < 0:
                self.fwd_init_win = tcp_window   # capture from 1st fwd pkt
            self.fwd_header_sizes.append(header_size)
        else:
            self.bwd_lengths.append(payload_len)
            self.bwd_times.append(timestamp)

        # Flag counting
        if tcp_flags:
            if tcp_flags & 0x01:  # FIN
                self.fin_count += 1
                if is_forward:
                    self.fin_seen_fwd = True
                else:
                    self.fin_seen_bwd = True
            if tcp_flags & 0x04:  # RST
                self.rst_seen = True

    @property
    def is_terminated(self) -> bool:
        """Flow is done if FIN seen in both directions, or RST seen."""
        return self.rst_seen or (self.fin_seen_fwd and self.fin_seen_bwd)

    def to_features(self) -> dict:
        """Compute the 18 CICFlowMeter-compatible features."""
        fwd_len = np.array(self.fwd_lengths, dtype=np.float64) if self.fwd_lengths else np.array([0.0])
        bwd_len = np.array(self.bwd_lengths, dtype=np.float64) if self.bwd_lengths else np.array([0.0])
        all_len = np.array(self.all_lengths, dtype=np.float64) if self.all_lengths else np.array([0.0])

        # Inter-arrival times
        fwd_iat = np.diff(self.fwd_times) if len(self.fwd_times) > 1 else np.array([0.0])
        bwd_iat = np.diff(self.bwd_times) if len(self.bwd_times) > 1 else np.array([0.0])
        flow_iat = np.diff(self.all_times) if len(self.all_times) > 1 else np.array([0.0])

        # Convert IAT from seconds to microseconds (CICFlowMeter convention)
        fwd_iat_us = fwd_iat * 1e6
        bwd_iat_us = bwd_iat * 1e6
        flow_iat_us = flow_iat * 1e6

        # Idle times: gaps longer than activity threshold
        idle_times = flow_iat[flow_iat > self._active_threshold] if len(flow_iat) > 0 else np.array([0.0])
        idle_times_us = idle_times * 1e6

        # Fwd Seg Size Min: CICFlowMeter uses the *transport header* length
        fwd_hdr = np.array(self.fwd_header_sizes, dtype=np.float64) if self.fwd_header_sizes else np.array([0.0])

        return {
            # ── Meta (not model features, but useful for display) ──
            "Flow ID": f"{self.src_ip}-{self.dst_ip}-{self.src_port}-{self.dst_port}-{self.protocol}",
            "Src IP": self.src_ip,
            "Dst IP": self.dst_ip,
            "Src Port": self.src_port,
            "Timestamp": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),

            # ── The 18 model features ──
            "Dst Port": self.dst_port,
            "Protocol": self.protocol,
            "FIN Flag Count": self.fin_count,
            "Fwd Packet Length Mean": float(np.mean(fwd_len)),
            "Fwd Packet Length Std": float(np.std(fwd_len, ddof=1)) if len(fwd_len) > 1 else 0.0,
            "Bwd Packet Length Mean": float(np.mean(bwd_len)),
            "Bwd Packet Length Max": float(np.max(bwd_len)),
            "Bwd Packet Length Std": float(np.std(bwd_len, ddof=1)) if len(bwd_len) > 1 else 0.0,
            "Packet Length Mean": float(np.mean(all_len)),
            "Packet Length Max": float(np.max(all_len)),
            "Packet Length Min": float(np.min(all_len)),
            "Flow IAT Max": float(np.max(flow_iat_us)) if len(flow_iat_us) > 0 else 0.0,
            "Flow IAT Std": float(np.std(flow_iat_us, ddof=1)) if len(flow_iat_us) > 1 else 0.0,
            "Fwd IAT Total": float(np.sum(fwd_iat_us)),
            "Bwd IAT Std": float(np.std(bwd_iat_us, ddof=1)) if len(bwd_iat_us) > 1 else 0.0,
            "FWD Init Win Bytes": self.fwd_init_win if self.fwd_init_win >= 0 else 0,
            "Fwd Seg Size Min": float(np.min(fwd_hdr)) if len(fwd_hdr) > 0 else 0.0,
            "Idle Max": float(np.max(idle_times_us)) if len(idle_times_us) > 0 else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Flow table — tracks active flows by 5-tuple
# ═══════════════════════════════════════════════════════════════════════════

def _flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    """Canonical bidirectional flow key (sorted so A→B == B→A)."""
    forward = (src_ip, dst_ip, src_port, dst_port, proto)
    backward = (dst_ip, src_ip, dst_port, src_port, proto)
    return min(forward, backward)


class FlowTable:
    """Thread-safe table of active flows."""

    def __init__(self, activity_timeout: float = 5.0):
        self.activity_timeout = activity_timeout      # no-packet gap → emit
        self._flows: dict[tuple, FlowRecord] = {}
        self._lock = threading.Lock()

    def process_packet(self, src_ip, dst_ip, src_port, dst_port, proto,
                       payload_len, timestamp, tcp_flags=0, tcp_window=0,
                       header_size=0) -> Optional[dict]:
        """
        Register a packet.  Returns the flow's feature dict if the flow
        just terminated (FIN+FIN or RST), otherwise None.
        """
        key = _flow_key(src_ip, dst_ip, src_port, dst_port, proto)

        with self._lock:
            if key not in self._flows:
                # First packet defines forward direction (like CICFlowMeter)
                self._flows[key] = FlowRecord(
                    src_ip=src_ip, dst_ip=dst_ip,
                    src_port=src_port, dst_port=dst_port,
                    protocol=proto,
                )

            flow = self._flows[key]
            is_forward = (src_ip == flow.src_ip and src_port == flow.src_port)

            flow.add_packet(
                payload_len=payload_len,
                timestamp=timestamp,
                is_forward=is_forward,
                tcp_flags=tcp_flags,
                tcp_window=tcp_window,
                header_size=header_size,
            )

            if flow.is_terminated:
                features = flow.to_features()
                del self._flows[key]
                return features

        return None

    def expire_flows(self, now: float) -> list[dict]:
        """Flush flows that have had no packets for activity_timeout."""
        expired = []
        with self._lock:
            to_remove = []
            for key, flow in self._flows.items():
                if (now - flow.last_time) > self.activity_timeout:
                    expired.append(flow.to_features())
                    to_remove.append(key)
            for key in to_remove:
                del self._flows[key]
        return expired

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._flows)


# ═══════════════════════════════════════════════════════════════════════════
#  Real-time extractor — sniff + track + emit
# ═══════════════════════════════════════════════════════════════════════════

class RealtimeFlowExtractor:
    """
    Sniffs packets on a network interface and emits flow feature dicts
    in real-time via a callback.

    Parameters
    ----------
    callback : callable(dict)
        Called with a feature dict every time a flow completes.
    activity_timeout : float
        Seconds with no new packets before a flow is emitted (default 5).
    expire_interval : float
        How often to sweep for expired flows (default 1 second).
    """

    def __init__(
        self,
        callback: Callable[[dict], None],
        activity_timeout: float = 5.0,
        expire_interval: float = 1.0,
    ):
        self.callback = callback
        self.flow_table = FlowTable(activity_timeout=activity_timeout)
        self.expire_interval = expire_interval

        self._stop_event = threading.Event()
        self._sniffer = None              # AsyncSniffer instance
        self._expire_thread: Optional[threading.Thread] = None
        self._interface: str = ""
        self._stats = {
            "packets_processed": 0,
            "packets_tcp": 0,
            "packets_udp": 0,
            "packets_ipv4": 0,
            "packets_ipv6": 0,
            "packets_skipped": 0,
            "flows_emitted": 0,
            "flows_tcp": 0,
            "flows_udp": 0,
            "active_flows": 0,
        }

    @property
    def is_running(self) -> bool:
        return self._sniffer is not None and self._sniffer.running

    @property
    def stats(self) -> dict:
        s = dict(self._stats)
        s["active_flows"] = self.flow_table.active_count
        s["is_running"] = self.is_running
        return s

    def start(self, interface: str, bpf_filter: str = ""):
        """Start sniffing on the given interface.

        Parameters
        ----------
        interface : str
            Network interface name (e.g. NPF device on Windows).
        bpf_filter : str
            Optional BPF filter expression, e.g. ``'net 10.10.10.0/24'``
            to restrict capture to a specific subnet (useful for SPAN/
            mirror ports that see traffic from multiple segments).
        """
        if self.is_running:
            logger.warning("Extractor already running")
            return

        from scapy.all import AsyncSniffer

        self._stop_event.clear()
        self._interface = interface

        sniffer_kwargs = dict(
            iface=interface,
            prn=self._on_packet,
            store=False,
        )
        if bpf_filter:
            sniffer_kwargs["filter"] = bpf_filter

        self._sniffer = AsyncSniffer(**sniffer_kwargs)
        self._expire_thread = threading.Thread(
            target=self._expire_loop,
            daemon=True,
            name="FlowExtractor-expire",
        )
        self._sniffer.start()
        self._expire_thread.start()
        if bpf_filter:
            logger.info("RealtimeFlowExtractor started on '%s' (filter: %s)",
                        interface, bpf_filter)
        else:
            logger.info("RealtimeFlowExtractor started on '%s'", interface)

    def stop(self):
        """Stop sniffing and flush remaining flows (non-blocking)."""
        self._stop_event.set()

        # AsyncSniffer.stop() closes the socket immediately — no waiting
        if self._sniffer:
            try:
                self._sniffer.stop()
            except Exception:
                pass
            self._sniffer = None

        if self._expire_thread:
            self._expire_thread.join(timeout=1)
            self._expire_thread = None

        # Flush all remaining active flows
        expired = self.flow_table.expire_flows(float('inf'))
        for feat in expired:
            self._emit(feat)

        logger.info("RealtimeFlowExtractor stopped. Total flows: %d "
                     "(TCP: %d, UDP: %d) | Packets: %d "
                     "(IPv4: %d, IPv6: %d, TCP: %d, UDP: %d, skipped: %d)",
                     self._stats["flows_emitted"],
                     self._stats["flows_tcp"], self._stats["flows_udp"],
                     self._stats["packets_processed"],
                     self._stats["packets_ipv4"], self._stats["packets_ipv6"],
                     self._stats["packets_tcp"], self._stats["packets_udp"],
                     self._stats["packets_skipped"])

    def _emit(self, features: dict):
        """Send a completed flow to the callback."""
        self._stats["flows_emitted"] += 1
        proto = features.get("Protocol", 0)
        if proto == 6:
            self._stats["flows_tcp"] += 1
        elif proto == 17:
            self._stats["flows_udp"] += 1
        try:
            self.callback(features)
        except Exception as exc:
            logger.error("Callback error: %s", exc)

    # ── Packet handler ──────────────────────────────────────────────────

    def _on_packet(self, pkt):
        """Process a single packet from scapy (IPv4 + IPv6)."""
        from scapy.layers.inet import IP, TCP, UDP
        from scapy.layers.inet6 import IPv6

        # ── Determine IP layer (v4 or v6) ─────────────────────────────
        if pkt.haslayer(IP):
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto
            self._stats["packets_ipv4"] += 1
        elif pkt.haslayer(IPv6):
            src_ip = pkt[IPv6].src
            dst_ip = pkt[IPv6].dst
            proto = pkt[IPv6].nh       # may be ext-header; overridden below
            self._stats["packets_ipv6"] += 1
        else:
            return

        # ── Only process TCP and UDP ──────────────────────────────────
        if not (pkt.haslayer(TCP) or pkt.haslayer(UDP)):
            self._stats["packets_skipped"] += 1
            return

        timestamp = float(pkt.time)

        src_port = 0
        dst_port = 0
        tcp_flags = 0
        tcp_window = 0
        payload_len = 0
        header_size = 0

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            src_port = tcp.sport
            dst_port = tcp.dport
            tcp_flags = int(tcp.flags)
            tcp_window = tcp.window
            tcp_hdr_len = tcp.dataofs * 4 if tcp.dataofs else 20
            header_size = tcp_hdr_len        # for Fwd Seg Size Min
            proto = 6                        # correct even for IPv6 ext-hdrs

            # Payload = transport-layer data (no headers)
            if pkt.haslayer(IP):
                ip = pkt[IP]
                payload_len = max(0, ip.len - (ip.ihl * 4) - tcp_hdr_len) if ip.len else 0
            else:
                # IPv6: use TCP layer length directly (avoids ext-hdr math)
                payload_len = max(0, len(pkt[TCP]) - tcp_hdr_len)

            self._stats["packets_tcp"] += 1
            if self._stats["packets_tcp"] <= 5:
                logger.debug(
                    "TCP pkt: %s:%d → %s:%d  flags=0x%02x win=%d "
                    "payload=%d hdr=%d",
                    src_ip, src_port, dst_ip, dst_port,
                    tcp_flags, tcp_window, payload_len, header_size,
                )

        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            src_port = udp.sport
            dst_port = udp.dport
            header_size = 8                  # fixed UDP header
            proto = 17                       # correct even for IPv6 ext-hdrs
            payload_len = max(0, udp.len - 8) if udp.len else 0
            self._stats["packets_udp"] += 1

        self._stats["packets_processed"] += 1

        result = self.flow_table.process_packet(
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=src_port, dst_port=dst_port,
            proto=proto, payload_len=payload_len, timestamp=timestamp,
            tcp_flags=tcp_flags, tcp_window=tcp_window,
            header_size=header_size,
        )

        if result is not None:
            self._emit(result)

    # ── Idle-expiry loop ───────────────────────────────────────────────

    def _expire_loop(self):
        """Periodically sweep for idle/expired flows."""
        while not self._stop_event.wait(self.expire_interval):
            expired = self.flow_table.expire_flows(time.time())
            for feat in expired:
                self._emit(feat)
