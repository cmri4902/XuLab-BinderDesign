#!/usr/bin/env python3
"""Xu Lab BindCraft2 - CUDA driver and JAX backend probe.

Answers one question with evidence: can this process reach a GPU through the
CUDA driver, and does JAX then select a GPU backend? It is meant to run *inside*
the real SLURM allocation (the launcher calls it), so no separate GPU test job
is needed. It allocates almost nothing and exits in a second or two.

    python lab/scripts/gpu_probe.py            # human-readable lines
    python lab/scripts/gpu_probe.py --json     # same findings as one JSON object

Every line is prefixed `xulab-gpu:` so a log can be searched and so
lab/scripts/validate_outputs.py can read the backend evidence back.

Exit status:
    0  a JAX GPU device is usable here
    2  libcuda.so.1 could not be loaded at all
    3  libcuda.so.1 loaded but cuInit() failed (the job 1014429 symptom)
    4  the CUDA driver works but JAX selected no GPU device
    5  JAX itself could not be imported

Why both halves matter: `nvidia-smi` is a separate binary that talks to the
driver through /dev/nvidiactl, so it can succeed while this process cannot
initialise CUDA - typically because a *stub* libcuda is ahead of the real driver
library on the search path, because /dev/nvidia-uvm is missing from the job's
device cgroup, or because CUDA_VISIBLE_DEVICES names a device this step was not
given.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import sys

PREFIX = "xulab-gpu:"
# Environment that decides, or explains, which devices a process can see.
INTERESTING_ENV = (
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
    "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_ID",
    "SLURM_STEP_ID", "SLURM_PROCID", "CUDA_DEVICE_ORDER", "CUDA_MPS_PIPE_DIRECTORY",
    "JAX_PLATFORMS", "JAX_PLATFORM_NAME", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "CONDA_PREFIX", "BINDCRAFT_HOME", "PYTHONPATH",
)
findings: dict = {"exit": None}


def say(line: str) -> None:
    print(f"{PREFIX} {line}", flush=True)


# --------------------------------------------------------------------------- #
# where libcuda comes from
# --------------------------------------------------------------------------- #

def mapped_libcuda() -> list[str]:
    """Paths of every libcuda the process has actually mapped."""
    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="replace") as maps:
            found = {line.split()[-1] for line in maps
                     if "libcuda.so" in line and line.split()[-1].startswith("/")}
        return sorted(found)
    except OSError:
        return []


def libcuda_candidates() -> list[str]:
    """Every libcuda.so* reachable through LD_LIBRARY_PATH, in search order.

    A path containing /stubs/ is the CUDA toolkit's link-time stub. It satisfies
    the linker and then fails at runtime with CUDA_ERROR_NO_DEVICE, because it
    talks to no driver at all.
    """
    candidates = []
    for directory in (os.environ.get("LD_LIBRARY_PATH") or "").split(os.pathsep):
        if not directory:
            continue
        for name in ("libcuda.so.1", "libcuda.so"):
            path = os.path.join(directory, name)
            if os.path.exists(path):
                candidates.append(path)
    return candidates


def stub_paths(paths) -> list[str]:
    return [path for path in paths if re.search(r"(^|/)stubs(/|$)", path)]


# --------------------------------------------------------------------------- #
# the CUDA driver itself
# --------------------------------------------------------------------------- #

def driver_probe() -> int:
    """Load libcuda, call cuInit, count devices. Returns an exit status."""
    say(f"python     {sys.executable}")
    say(f"platform   {platform.platform()}")

    candidates = libcuda_candidates()
    stubs = stub_paths(candidates)
    findings["ld_library_path_libcuda"] = candidates
    findings["stub_libcuda"] = stubs
    if candidates:
        say("libcuda on LD_LIBRARY_PATH: " + ", ".join(candidates))
    if stubs:
        say("WARNING stub libcuda ahead of the real driver: " + ", ".join(stubs))
        say("        a stub libcuda returns CUDA_ERROR_NO_DEVICE; remove that "
            "directory from LD_LIBRARY_PATH")

    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as failure:
        findings["libcuda_loaded"] = False
        findings["libcuda_error"] = str(failure)
        say(f"FAIL libcuda.so.1 could not be loaded: {failure}")
        say("     on a GPU node this means the NVIDIA driver libraries are not on "
            "the library path for this step")
        return 2
    findings["libcuda_loaded"] = True
    mapped = mapped_libcuda()
    findings["libcuda_mapped"] = mapped
    say("libcuda loaded from: " + (", ".join(mapped) or "path not reported by /proc"))

    def error_name(code: int) -> str:
        name = ctypes.c_char_p()
        try:
            driver.cuGetErrorName(ctypes.c_int(code), ctypes.byref(name))
            return (name.value or b"").decode() or f"code {code}"
        except Exception:                                      # noqa: BLE001
            return f"code {code}"

    version = ctypes.c_int()
    if driver.cuDriverGetVersion(ctypes.byref(version)) == 0:
        major, minor = divmod(version.value, 1000)
        findings["cuda_driver_version"] = f"{major}.{minor // 10}"
        say(f"cuDriverGetVersion -> CUDA {major}.{minor // 10}")

    status = driver.cuInit(0)
    findings["cuInit"] = status
    findings["cuInit_name"] = error_name(status)
    if status != 0:
        say(f"FAIL cuInit(0) -> {findings['cuInit_name']} ({status})")
        if status == 100:                                      # CUDA_ERROR_NO_DEVICE
            say("     CUDA_ERROR_NO_DEVICE: the driver is reachable but this "
                "process was given no usable device. Check, in this order:")
            say("     1. a stub libcuda on LD_LIBRARY_PATH (reported above)")
            say("     2. /dev/nvidia-uvm present in this step's device cgroup")
            say("     3. CUDA_VISIBLE_DEVICES naming a device this step was not "
                "allocated (compare with SLURM_JOB_GPUS / SLURM_STEP_GPUS)")
            say("     4. the work running in the batch step rather than under srun")
        return 3

    count = ctypes.c_int()
    if driver.cuDeviceGetCount(ctypes.byref(count)) != 0:
        say("FAIL cuDeviceGetCount failed after a successful cuInit")
        return 3
    findings["cuda_device_count"] = count.value
    say(f"cuInit ok, cuDeviceGetCount -> {count.value}")
    for ordinal in range(count.value):
        handle = ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        if driver.cuDeviceGet(ctypes.byref(handle), ctypes.c_int(ordinal)) == 0:
            driver.cuDeviceGetName(name, ctypes.c_int(256), handle)
            say(f"cuda device {ordinal}: {name.value.decode(errors='replace')}")
    return 0 if count.value else 3


# --------------------------------------------------------------------------- #
# what JAX then does with it
# --------------------------------------------------------------------------- #

def jax_probe() -> int:
    try:
        import jax
    except Exception as failure:                               # noqa: BLE001
        findings["jax_import_error"] = str(failure)
        say(f"FAIL jax could not be imported: {failure}")
        return 5
    findings["jax_version"] = jax.__version__
    try:
        import jaxlib
        findings["jaxlib_version"] = getattr(jaxlib, "__version__", "unknown")
    except Exception:                                          # noqa: BLE001
        findings["jaxlib_version"] = "unknown"
    for plugin in ("jax_plugins.xla_cuda12", "jax_plugins.xla_cuda13"):
        try:
            __import__(plugin)
            findings.setdefault("jax_plugins", []).append(plugin)
            say(f"jax plugin importable: {plugin}")
        except Exception:                                      # noqa: BLE001
            pass
    say(f"jax {jax.__version__} jaxlib {findings['jaxlib_version']}")

    try:
        devices = jax.devices()
    except Exception as failure:                               # noqa: BLE001
        findings["jax_devices_error"] = str(failure)
        say(f"FAIL jax.devices() raised: {failure}")
        return 4
    try:
        backend = jax.default_backend()
    except Exception:                                          # noqa: BLE001
        backend = "unknown"
    findings["jax_backend"] = backend
    findings["jax_devices"] = [f"{device.platform}:{device.id}" for device in devices]
    # This is the line validate_outputs.py reads as JAX GPU backend evidence.
    say(f"jax backend={backend}")
    for device in devices:
        say(f"jax device {device.platform}:{device.id} "
            f"{getattr(device, 'device_kind', '?')}")
    if any(device.platform in ("gpu", "cuda", "rocm") for device in devices):
        return 0
    say("FAIL jax selected no GPU device; designing here would run on the CPU")
    return 4


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    say("--- environment ---")
    for name in INTERESTING_ENV:
        if name in os.environ:
            value = os.environ[name]
            findings.setdefault("env", {})[name] = value
            say(f"env {name}={value if len(value) < 400 else value[:400] + ' ...'}")
    say("--- cuda driver ---")
    status = driver_probe()
    if status == 0:
        say("--- jax ---")
        status = jax_probe()
    findings["exit"] = status
    verdict = {0: "GPU USABLE", 2: "NO LIBCUDA", 3: "DRIVER INIT FAILED",
               4: "NO JAX GPU", 5: "NO JAX"}.get(status, "UNKNOWN")
    say(f"verdict {verdict} (exit {status})")
    if "--json" in arguments:
        print(json.dumps(findings, indent=2), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
