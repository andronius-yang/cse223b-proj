"""Unit conversion helpers.

CLI bandwidths are specified in Gbps. Internally, timings use bytes/us.
"""


def gbps_to_bytes_per_us(gbps: float) -> float:
    if gbps <= 0:
        raise ValueError(f"Bandwidth must be positive, got {gbps}")
    return gbps * 125.0


def bytes_per_us_to_gb_per_s(bytes_per_us: float) -> float:
    return bytes_per_us / 1000.0


def bytes_and_us_to_gb_per_s(num_bytes: float, time_us: float) -> float:
    if time_us <= 0:
        return 0.0
    return bytes_per_us_to_gb_per_s(num_bytes / time_us)
