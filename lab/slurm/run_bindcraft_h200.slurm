#!/bin/bash
# ---------------------------------------------------------------------------
# Xu Lab BindCraft2 - H200 campaign launcher (Phase 1B)
#
# Runs the Xu Lab source copy at $XULAB_BC2_HOME on one H200 card, writing every
# artefact under $XULAB_PROJECT_ROOT. It installs nothing, downloads nothing, and
# refuses to start if JAX cannot see a GPU.
#
# Submit (the log directory must exist first, because Slurm opens --output
# before this script runs):
#
#   mkdir -p /project/yli4/ypan/bindcraft/logs
#   cd /home/ypan/BindCraft2/BindCraft2_deploy
#   sbatch lab/slurm/run_bindcraft_h200.slurm lab/configs/pdl1_smoke.json
#
# Optional overrides, all exported before sbatch or passed with --export:
#   XULAB_BC2_HOME        source tree to execute, in HOME (default below)
#   XULAB_PROJECT_ROOT    results/logs/cache/tmp root, in project storage
#   XULAB_PYTHON          interpreter of the bindcraft2 conda environment
#   BINDCRAFT_AF2_PARAMS  directory holding params_<model>.npz (never a file)
#   BINDCRAFT_WEIGHTS     cache root holding alphafold/ and the shared caches
#
# An account or a different partition is a site detail this file does not guess:
#   sbatch --account=<account> lab/slurm/run_bindcraft_h200.slurm <config>
# ---------------------------------------------------------------------------
#SBATCH --job-name=xulab_bc2_h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/project/yli4/ypan/bindcraft/logs/xulab_bc2_%j.out
#SBATCH --error=/project/yli4/ypan/bindcraft/logs/xulab_bc2_%j.err

set -euo pipefail

# --- Xu Lab paths (HPC only; no Windows path belongs in this file) ----------
# The source tree lives in HOME; every artefact the campaign writes lives in the
# project tree. They are set independently so one can move without the other.
XULAB_PROJECT_ROOT="${XULAB_PROJECT_ROOT:-/project/yli4/ypan/bindcraft}"
XULAB_BC2_HOME="${XULAB_BC2_HOME:-/home/ypan/BindCraft2/BindCraft2_deploy}"
XULAB_PYTHON="${XULAB_PYTHON:-/home/ypan/.conda/envs/bindcraft2/bin/python}"
XULAB_RESULTS="${XULAB_PROJECT_ROOT}/results"
XULAB_LOGS="${XULAB_PROJECT_ROOT}/logs"
XULAB_CACHE="${XULAB_PROJECT_ROOT}/cache"
XULAB_TMP="${XULAB_PROJECT_ROOT}/tmp"

CONFIG="${1:-lab/configs/pdl1_smoke.json}"
shift || true

fail() { echo "xulab-bc2: ERROR: $*" >&2; exit "${2:-1}"; }

echo "xulab-bc2: host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"

# --- Preconditions ---------------------------------------------------------
[[ -x "${XULAB_PYTHON}" ]] || fail "interpreter not executable: ${XULAB_PYTHON}" 2
[[ -d "${XULAB_BC2_HOME}" ]] || fail "source tree missing: ${XULAB_BC2_HOME}" 2
[[ -f "${XULAB_BC2_HOME}/bindcraft.py" ]] || fail "not a BindCraft2 tree: ${XULAB_BC2_HOME}/bindcraft.py" 2

cd "${XULAB_BC2_HOME}"
[[ -f "${CONFIG}" ]] || fail "settings file not found: ${CONFIG} (relative to ${XULAB_BC2_HOME})" 2

mkdir -p "${XULAB_RESULTS}" "${XULAB_LOGS}" "${XULAB_CACHE}/jax" "${XULAB_TMP}"

# --- Environment -----------------------------------------------------------
# PYTHONPATH and BINDCRAFT_HOME both point at the lab copy, so an editable
# install of the older source tree cannot win the import.
export BINDCRAFT_HOME="${XULAB_BC2_HOME}"
export PYTHONPATH="${XULAB_BC2_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
# Keeps bindcraft.py from re-executing itself under some other .venv interpreter.
export BINDCRAFT_LAUNCHER_SWITCHED=1
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${XULAB_CACHE}/jax}"
export TMPDIR="${TMPDIR:-${XULAB_TMP}}"
export BINDCRAFT_WEIGHTS="${BINDCRAFT_WEIGHTS:-${XULAB_CACHE}/bindcraft}"
mkdir -p "${BINDCRAFT_WEIGHTS}"

# --- AlphaFold parameters: reuse an existing copy, never download ----------
# BINDCRAFT_AF2_PARAMS set in the environment is used as given. Otherwise the
# candidates below are *searched*, not assumed; none of them is treated as known
# to exist. Verify the real location once and export it to skip this search.
if [[ -z "${BINDCRAFT_AF2_PARAMS:-}" ]]; then
  for candidate in \
      "${BINDCRAFT_WEIGHTS}/alphafold" \
      "${HOME}/.cache/bindcraft/alphafold" \
      "${XULAB_PROJECT_ROOT}/cache/bindcraft/alphafold" \
      "${XULAB_BC2_HOME}/bindcraft/weights/alphafold"; do
    if compgen -G "${candidate}/params_*.npz" > /dev/null || compgen -G "${candidate}/params/params_*.npz" > /dev/null; then
      export BINDCRAFT_AF2_PARAMS="${candidate}"
      break
    fi
  done
fi
[[ -n "${BINDCRAFT_AF2_PARAMS:-}" ]] || fail \
  "AlphaFold parameters not found. Locate the existing 5.3 GB set on this cluster and resubmit with
         sbatch --export=ALL,BINDCRAFT_AF2_PARAMS=/abs/path/to/dir lab/slurm/run_bindcraft_h200.slurm ${CONFIG}
       The path must be the DIRECTORY holding params_<model>.npz, not a single .npz file.
       This job does not download weights." 3
[[ -d "${BINDCRAFT_AF2_PARAMS}" ]] || fail "BINDCRAFT_AF2_PARAMS is not a directory: ${BINDCRAFT_AF2_PARAMS}" 3
echo "xulab-bc2: BINDCRAFT_AF2_PARAMS=${BINDCRAFT_AF2_PARAMS}"
echo "xulab-bc2: BINDCRAFT_WEIGHTS=${BINDCRAFT_WEIGHTS}"
echo "xulab-bc2: JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR}"

# --- Allocation evidence ---------------------------------------------------
echo "xulab-bc2: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 || \
  echo "xulab-bc2: nvidia-smi unavailable in this allocation"

# --- Gate: the lab copy must be the imported package, on a real JAX GPU ----
# Both facts are checked in one short process that exits before the campaign
# starts, and a failure here is a nonzero exit rather than a CPU fallback.
"${XULAB_PYTHON}" -s -c '
import os, sys, pathlib
home = pathlib.Path(os.environ["BINDCRAFT_HOME"]).resolve()
import bindcraft
module = pathlib.Path(bindcraft.__file__).resolve()
version = getattr(bindcraft, "__version__", "unknown")
print("xulab-bc2: python      " + sys.executable)
print("xulab-bc2: bindcraft   " + str(module))
print("xulab-bc2: version     " + str(version))
if home not in module.parents:
    sys.exit("xulab-bc2: ERROR: imported bindcraft from " + str(module) + ", outside the lab copy " + str(home))
import jax
try:
    devices = jax.devices()
except Exception as failure:
    sys.exit("xulab-bc2: ERROR: JAX reports no usable device: " + str(failure))
print("xulab-bc2: jax         " + jax.__version__ + " backend=" + jax.default_backend())
for device in devices:
    print("xulab-bc2: device      " + device.platform + ":" + str(device.id) + " " + str(getattr(device, "device_kind", "?")))
if not any(device.platform in ("gpu", "cuda", "rocm") for device in devices):
    sys.exit("xulab-bc2: ERROR: no JAX GPU in this allocation; refusing to design on CPU")
' || fail "preflight failed (import path or JAX GPU); see messages above" 4

# --- Campaign --------------------------------------------------------------
echo "xulab-bc2: launching: ${XULAB_PYTHON} bindcraft.py design ${CONFIG} $*"
set +e
"${XULAB_PYTHON}" -u bindcraft.py design "${CONFIG}" "$@"
status=$?
set -e
echo "xulab-bc2: campaign exit status ${status} finished=$(date -Is)"
exit "${status}"
