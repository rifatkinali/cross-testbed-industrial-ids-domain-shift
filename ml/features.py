"""Shared Suricata EVE feature extraction for offline and live IDS paths."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import pandas as pd

# Pilot capture assigns each traffic origin a distinct static IP, so a write's
# source identity ("who did it") is recoverable.  The basic shared-gateway
# capture cannot separate sources, so everything maps to "unknown".  Overridable
# by the MODBUS_ACTOR_IP_MAP env var (same contract as the live IDS).
DEFAULT_ACTOR_MAP = {
    "172.20.0.10": "scenario-server",
    "172.20.0.11": "authorized-control",
    "172.20.0.12": "watchdog",
}


def _actor_map() -> dict:
    raw = os.environ.get("MODBUS_ACTOR_IP_MAP")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return DEFAULT_ACTOR_MAP


def actor_of(src_ip: Any) -> str:
    """Map a source IP to its pilot actor identity ('unknown' if unattributed)."""
    return _actor_map().get(str(src_ip or ""), "unknown")

# Lazy logging import to avoid circular imports
_log: Optional[logging.Logger] = None

def get_log():
    global _log
    if _log is None:
        try:
            from ml.logging_utils import get_logger
            _log = get_logger(__name__)
        except ImportError:
            _log = logging.getLogger(__name__)
    return _log


PROCESS_REGISTERS = {
    0: "engine_rpm",
    1: "ballast_level",
    2: "ballast_setpoint",
    3: "rpm_command",
    4: "heading_cmd",
    5: "heading",
    6: "rudder_angle",
    7: "gen_load",
    8: "bus_freq",
    9: "load_cmd",
}

NUMERIC_FEATURES = [
    "dest_port",
    "pkts_toserver",
    "pkts_toclient",
    "bytes_toserver",
    "bytes_toclient",
    "modbus_address",
    "modbus_quantity",
    "modbus_value",
    *PROCESS_REGISTERS.values(),
]

CATEGORICAL_FEATURES = [
    "proto",
    "app_proto",
    "flow_state",
    "modbus_function",
    "modbus_access",
    "src_port_role",
    "write_band",
    "read_band",
]

# Yazma fonksiyonlari (FC5/6/15/16) ve register basina guvenli deger bandi.
# Bant-disi bir yazma = guclu saldiri sinyali; bant-ici yazma = mesru/bakim.
WRITE_FUNCTIONS = {5, 6, 15, 16}
SAFE_BANDS = {
    2: (40, 60),      # ballast_setpoint
    3: (0, 1000),     # rpm_command
    4: (45, 135),     # heading_cmd (normal rota ~90)
    9: (0, 80),       # load_cmd
}


def write_band(address: Any, value: Any, function: Any) -> str:
    """
    Bir Modbus yazmasinin guvenli bant durumunu etiketler.
    
    Parameters
    ----------
    address : Any
        Modbus register address.
    value : Any
        Value being written.
    function : Any
        Modbus function code.
    
    Returns
    -------
    str
        Band status: "in_band", "out_of_band", "write_unknown", "write_other", or "na".
    """
    log = get_log()
    try:
        fn = int(function)
    except (TypeError, ValueError):
        log.debug(f"Invalid function code for write_band: {function}")
        return "na"
    if fn not in WRITE_FUNCTIONS:
        return "na"                      # okuma / yazma-disi
    try:
        addr = int(address)
        val = int(value)
    except (TypeError, ValueError) as e:
        log.debug(f"Failed to parse write_band params: address={address}, value={value} - {e}")
        return "write_unknown"
    
    band = SAFE_BANDS.get(addr)
    if band is None:
        log.debug(f"No safe band defined for address {addr}")
        return "write_other"             # bantsiz register (coil)
    
    in_band = band[0] <= val <= band[1]
    if not in_band:
        # DEBUG: bu fonksiyon toplu egitimde satir basina cagrilir; WARNING
        # seviyesi log seli yaratir. Alarm kararini realtime_ids.py loglar.
        log.debug(f"Out-of-band write: address={addr}, value={val}, band={band}")
    return "in_band" if in_band else "out_of_band"


# Okuma fonksiyonlari (FC1/2/3/4) ve MESRU HMI'nin okudugu tek desen.
# Bu desenin DISINDAKI okuma = kesif/tarama (recon) sinyali. Bu ozellik,
# model recon'u egitimde hic gormese bile "kapsam-disi okuma" isaretini verir
# -> gorulmemis kesif saldirisina genellemeyi guclendirir (LOSO).
READ_FUNCTIONS = {1, 2, 3, 4}
LEGIT_READS = {
    3: range(0, 10),   # holding registers 0..9
    1: range(0, 2),    # coils 0..1
}


def read_band(address: Any, function: Any) -> str:
    """
    Bir Modbus okumasinin kapsam durumunu etiketler: in_scope | scan | na.
    
    Parameters
    ----------
    address : Any
        Modbus register address.
    function : Any
        Modbus function code.
    
    Returns
    -------
    str
        Scope status: "in_scope", "scan", "scan_unknown", or "na".
    """
    log = get_log()
    try:
        fn = int(function)
    except (TypeError, ValueError):
        log.debug(f"Invalid function code for read_band: {function}")
        return "na"
    if fn not in READ_FUNCTIONS:
        return "na"                      # yazma / okuma-disi
    try:
        addr = int(address)
    except (TypeError, ValueError) as e:
        log.debug(f"Failed to parse read_band address: {address} - {e}")
        return "scan_unknown"
    
    legit = LEGIT_READS.get(fn)
    if legit is not None and addr in legit:
        return "in_scope"                # mesru poll okumasi
    
    log.debug(f"Scan/recon read: address={addr}, function={fn}")
    return "scan"                        # kapsam-disi -> recon/tarama


MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# NOT: 'anomaly' BILEREK cikarildi (bkz. BULGU-ariza-anomaly.md).
# Kanit: iyi huylu arizalarda yanlis alarmlarin %100'u event_type='anomaly'
# olaylarindan geliyordu (modbus/flow=0). 'anomaly' saldiri tespitine deger
# katmiyor (tum senaryolar modbus/flow ile zaten yakalaniyor) ama ariza
# yanlis-alarmini %0'dan %50'ye cikariyordu. Cikarilinca: ariza yanlis-alarm
# %50->%0, saldiri recall degismeden korunur.
MODEL_EVENT_TYPES = {"flow", "modbus"}

DATASET_COLUMNS = [
    "timestamp",
    "run_id",
    "label",
    "scenario",
    "sensor",
    "event_type",
    "proto",
    "app_proto",
    "src_ip",
    "actor",
    "src_port",
    "dest_ip",
    "dest_port",
    "pkts_toserver",
    "pkts_toclient",
    "bytes_toserver",
    "bytes_toclient",
    "flow_state",
    "modbus_function",
    "modbus_access",
    "modbus_address",
    "modbus_quantity",
    "modbus_value",
    *PROCESS_REGISTERS.values(),
    "alert_signature",
    "alert_sid",
]


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_dicts(nested)


def _first(mapping: Any, key: str, default: Any = "") -> Any:
    for item in _walk_dicts(mapping):
        if item.get(key) is not None:
            return item[key]
    return default


def _request(modbus: Any) -> dict[str, Any]:
    if not isinstance(modbus, dict):
        return {}
    request = modbus.get("request")
    return request if isinstance(request, dict) else modbus


def _operation(modbus: Any) -> dict[str, Any]:
    request = _request(modbus)
    for key in ("write", "read"):
        operation = request.get(key)
        if isinstance(operation, dict):
            return operation
    return {}


def _decode_registers(modbus: Any) -> dict[str, int]:
    """Decode FC3 response bytes into named OpenPLC holding-register values."""
    if not isinstance(modbus, dict):
        return {}
    request = _request(modbus)
    if request.get("function_raw") != 3:
        return {}
    read_request = request.get("read")
    response = modbus.get("response")
    if not isinstance(read_request, dict) or not isinstance(response, dict):
        return {}
    read_response = response.get("read")
    if not isinstance(read_response, dict):
        return {}
    raw = read_response.get("data")
    if not isinstance(raw, str):
        return {}
    payload = bytes(ord(char) & 0xFF for char in raw)
    if len(payload) < 2:
        return {}
    start = int(read_request.get("address", 0) or 0)
    quantity = int(read_request.get("quantity", len(payload) // 2) or 0)
    values: dict[str, int] = {}
    for index in range(min(quantity, len(payload) // 2)):
        address = start + index
        name = PROCESS_REGISTERS.get(address)
        if name:
            offset = index * 2
            values[name] = int.from_bytes(payload[offset:offset + 2], "big")
    return values


def port_role(value: Any) -> str:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if port < 1024:
        return "well_known"
    if port < 49152:
        return "registered"
    return "ephemeral"


def flatten_eve(
    event: dict[str, Any],
    *,
    run_id: Any = "",
    label: str = "",
    scenario: str = "",
    sensor: str = "ot",
) -> dict[str, Any]:
    """Convert one raw EVE event into the canonical dataset/live feature row."""
    flow = event.get("flow")
    flow = flow if isinstance(flow, dict) else {}
    alert = event.get("alert")
    alert = alert if isinstance(alert, dict) else {}
    modbus = event.get("modbus")
    operation = _operation(modbus)
    request = _request(modbus)
    function = request.get("function_raw")
    if function is None:
        function = _first(modbus, "function_raw", "")
    access = request.get("access_type") or _first(modbus, "access_type", "")
    address = operation.get("address", "")
    quantity = operation.get("quantity", 1 if "data" in operation else "")
    value = operation.get("data", "")

    row: dict[str, Any] = {
        "timestamp": event.get("timestamp", ""),
        "run_id": run_id,
        "label": label,
        "scenario": scenario,
        "sensor": sensor,
        "event_type": event.get("event_type", ""),
        "proto": event.get("proto", ""),
        "app_proto": event.get("app_proto", ""),
        "src_ip": event.get("src_ip", ""),
        "actor": actor_of(event.get("src_ip", "")),
        "src_port": event.get("src_port", ""),
        "dest_ip": event.get("dest_ip", ""),
        "dest_port": event.get("dest_port", ""),
        "pkts_toserver": flow.get("pkts_toserver", 0),
        "pkts_toclient": flow.get("pkts_toclient", 0),
        "bytes_toserver": flow.get("bytes_toserver", 0),
        "bytes_toclient": flow.get("bytes_toclient", 0),
        "flow_state": flow.get("state", ""),
        "modbus_function": function,
        "modbus_access": access,
        "modbus_address": address,
        "modbus_quantity": quantity,
        "modbus_value": value,
        "src_port_role": port_role(event.get("src_port")),
        "write_band": write_band(address, value, function),
        "read_band": read_band(address, function),
        "alert_signature": alert.get("signature", ""),
        "alert_sid": alert.get("signature_id", ""),
    }
    for name in PROCESS_REGISTERS.values():
        row[name] = ""
    row.update(_decode_registers(modbus))

    # A write command also carries a meaningful process value.
    try:
        register_name = PROCESS_REGISTERS.get(int(address))
        if register_name and value != "":
            row[register_name] = int(value)
    except (TypeError, ValueError):
        pass
    return row


def ensure_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable model input schema, filling absent legacy columns."""
    result = frame.copy()

    def _column(name: str) -> list:
        """Sutunu liste olarak dondur; yoksa satir sayisi kadar bos deger.

        NOT: burada `if name in result else ""` kisayolu KULLANILAMAZ --
        bos string'in de `__len__`'i vardir ve zip() sessizce 0 satir uretip
        DataFrame atamasini patlatir (Modbus icermeyen olaylarda canli IDS
        cokerdi).
        """
        if name in result.columns:
            return list(result[name])
        return [""] * len(result)

    # Eski veri setlerinde write_band/read_band yoksa mevcut sutunlardan turet.
    _addr = _column("modbus_address")
    _val = _column("modbus_value")
    _fn = _column("modbus_function")
    if "write_band" not in result.columns:
        result["write_band"] = [write_band(a, v, f) for a, v, f in zip(_addr, _val, _fn)]
    if "read_band" not in result.columns:
        result["read_band"] = [read_band(a, f) for a, f in zip(_addr, _fn)]
    for column in NUMERIC_FEATURES:
        if column not in result:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    for column in CATEGORICAL_FEATURES:
        if column not in result:
            result[column] = "unknown"
        result[column] = result[column].fillna("unknown").astype(str)
    return result[MODEL_FEATURES]


def filter_model_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude signature alerts and non-traffic bookkeeping from ML inputs."""
    if "event_type" not in frame:
        return frame.copy()
    return frame[frame["event_type"].astype(str).isin(MODEL_EVENT_TYPES)].copy()


def eve_model_frame(event: dict[str, Any]) -> pd.DataFrame:
    """Build the exact model input used by the live IDS from one raw event."""
    return ensure_model_frame(pd.DataFrame([flatten_eve(event)]))
