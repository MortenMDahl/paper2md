"""
vmcheck.py — Windows virtual memory health checks.

Detects dangerously high commit charge and zombie Python processes that
hold large CUDA virtual address reservations.  Call check_system_health()
before loading PyTorch to avoid adding to an already-exhausted system.

All Windows API calls use ctypes (stdlib).  Zombie detection uses psutil
if available but degrades gracefully without it.
"""

import ctypes
import os
import sys

# ---------------------------------------------------------------------------
# Commit-charge check (Windows only, via kernel32)
# ---------------------------------------------------------------------------

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _get_commit_charge() -> tuple[int, int] | None:
    """Return (used_bytes, total_bytes) for the system commit charge.

    Returns None on non-Windows or if the API call fails.
    """
    if sys.platform != "win32":
        return None
    try:
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        total = stat.ullTotalPageFile
        avail = stat.ullAvailPageFile
        return (total - avail, total)
    except Exception:
        return None


def _fmt_gb(b: int) -> str:
    return f"{b / (1024 ** 3):.1f} GB"


# ---------------------------------------------------------------------------
# Zombie Python process detection (optional psutil)
# ---------------------------------------------------------------------------

_ZOMBIE_VM_THRESHOLD = 10 * (1024 ** 3)  # 10 GB virtual memory


def _find_zombie_pythons() -> list[dict]:
    """Find Python processes (other than us) with >10 GB virtual memory.

    Returns a list of dicts with pid, name, vms (bytes).
    Returns empty list if psutil is not installed.
    """
    try:
        import psutil
    except ImportError:
        return []

    zombies = []
    my_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = proc.info
            if info["pid"] == my_pid:
                continue
            name = (info["name"] or "").lower()
            if "python" not in name:
                continue
            mem = info.get("memory_info")
            if mem and mem.vms > _ZOMBIE_VM_THRESHOLD:
                zombies.append({
                    "pid": info["pid"],
                    "name": info["name"],
                    "vms": mem.vms,
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return zombies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

WARN_THRESHOLD = 0.80   # 80% commit charge → warning
ABORT_THRESHOLD = 0.90  # 90% commit charge → abort


def check_system_health(abort_on_critical: bool = True) -> bool:
    """Pre-flight check before loading PyTorch / CUDA.

    - Checks Windows commit charge (warn >=80%, abort >=90%)
    - Detects zombie Python processes with >10 GB virtual memory
    - Returns True if safe to proceed, False if critical

    Set abort_on_critical=False to warn without sys.exit().
    """
    if sys.platform != "win32":
        return True

    ok = True

    # --- Commit charge ---
    charge = _get_commit_charge()
    if charge is not None:
        used, total = charge
        ratio = used / total if total > 0 else 0
        print(
            f"  [vmcheck] Commit charge: {_fmt_gb(used)} / {_fmt_gb(total)} "
            f"({ratio:.0%})",
            file=sys.stderr,
        )
        if ratio >= ABORT_THRESHOLD:
            print(
                f"  [vmcheck] CRITICAL: commit charge at {ratio:.0%} — "
                f"loading PyTorch/CUDA will likely crash the system.",
                file=sys.stderr,
            )
            ok = False
        elif ratio >= WARN_THRESHOLD:
            print(
                f"  [vmcheck] WARNING: commit charge at {ratio:.0%} — "
                f"system is under memory pressure.",
                file=sys.stderr,
            )

    # --- Zombie Python processes ---
    zombies = _find_zombie_pythons()
    if zombies:
        print(
            f"  [vmcheck] WARNING: {len(zombies)} zombie Python process(es) "
            f"with >10 GB virtual memory:",
            file=sys.stderr,
        )
        for z in zombies:
            print(
                f"    PID {z['pid']} ({z['name']}): {_fmt_gb(z['vms'])} VM",
                file=sys.stderr,
            )
        print(
            "  [vmcheck] These may be holding CUDA reservations. "
            "Consider killing them with: taskkill /F /PID <pid>",
            file=sys.stderr,
        )

    if not ok and abort_on_critical:
        print(
            "  [vmcheck] Aborting to prevent system instability. "
            "Free memory or kill zombie processes, then retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    return ok


def check_memory_pressure() -> bool:
    """Lightweight runtime check — call between operations.

    Returns True if OK, False if commit charge exceeds warning threshold.
    """
    if sys.platform != "win32":
        return True

    charge = _get_commit_charge()
    if charge is None:
        return True

    used, total = charge
    ratio = used / total if total > 0 else 0

    if ratio >= ABORT_THRESHOLD:
        print(
            f"  [vmcheck] CRITICAL: commit charge at {ratio:.0%} during processing. "
            f"Consider stopping to prevent system instability.",
            file=sys.stderr,
        )
        return False
    elif ratio >= WARN_THRESHOLD:
        print(
            f"  [vmcheck] WARNING: commit charge at {ratio:.0%} during processing.",
            file=sys.stderr,
        )

    return True
