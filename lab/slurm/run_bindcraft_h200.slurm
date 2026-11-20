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
#   XULAB_SRUN            auto (default) | always | never - whether GPU work runs
#                         as an srun step, which is what carries the GRES binding
#                         on a cgroup-constrained cluster
#   XULAB_SRUN_FLAGS      flags for that step (default: --ntasks=1 --gres=gpu:h200:1)
#   XULAB_DIAGNOSTICS     1 (default) | 0 - print the in-allocation GPU report
#   XULAB_KEEP_LD_STUBS   1 keeps stub libcuda directories on LD_LIBRARY_PATH
#                         (default 0: they are removed, because a stub libcuda
#                         answers cuInit with CUDA_ERROR_NO_DEVICE)
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

XULAB_SRUN="${XULAB_SRUN:-auto}"
XULAB_SRUN_FLAGS="${XULAB_SRUN_FLAGS:---ntasks=1 --gres=gpu:h200:1}"
XULAB_DIAGNOSTICS="${XULAB_DIAGNOSTICS:-1}"
XULAB_KEEP_LD_STUBS="${XULAB_KEEP_LD_STUBS:-0}"

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

# --- GPU environment repair ------------------------------------------------
# Job 1014429 failed with cuInit(0) -> CUDA_ERROR_NO_DEVICE while nvidia-smi saw
# an H200 NVL. These three repairs address the causes that produce exactly that
# split; each one reports whether it changed anything, so the log says which
# applied.

# (a) A stub libcuda (the CUDA toolkit's link-time placeholder, usually under a
#     .../stubs directory and often dragged in by a conda prefix) loads fine and
#     then answers cuInit with CUDA_ERROR_NO_DEVICE. Drop those directories; the
#     real driver library comes from the node's ldconfig path.
if [[ ${XULAB_KEEP_LD_STUBS} != 1 && -n ${LD_LIBRARY_PATH:-} ]]; then
  cleaned="" removed=""
  IFS=':' read -r -a library_dirs <<< "${LD_LIBRARY_PATH}"
  for directory in "${library_dirs[@]}"; do
    [[ -z ${directory} ]] && continue
    if [[ ${directory} == *"/stubs" || ${directory} == *"/stubs/" ]] \
       && { [[ -e "${directory}/libcuda.so" ]] || [[ -e "${directory}/libcuda.so.1" ]]; }; then
      removed="${removed} ${directory}"
      continue
    fi
    cleaned="${cleaned:+${cleaned}:}${directory}"
  done
  if [[ -n ${removed} ]]; then
    export LD_LIBRARY_PATH="${cleaned}"
    echo "xulab-bc2: removed stub libcuda director(ies) from LD_LIBRARY_PATH:${removed}"
  fi
fi

# (b) --export=ALL carries the submitting shell's CUDA_VISIBLE_DEVICES into the
#     job, where it can name a device this allocation was never given (for
#     example a Jupyter node's T4). SLURM's own view of the allocation wins.
slurm_gpus="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
if [[ -n ${slurm_gpus} ]]; then
  if [[ ${CUDA_VISIBLE_DEVICES:-unset} != "${slurm_gpus}" ]]; then
    echo "xulab-bc2: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} did not match the"
    echo "xulab-bc2: allocation (SLURM says ${slurm_gpus}); using SLURM's value"
    export CUDA_VISIBLE_DEVICES="${slurm_gpus}"
  fi
elif [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  echo "xulab-bc2: WARNING CUDA_VISIBLE_DEVICES is empty and SLURM named no GPU;"
  echo "xulab-bc2: an empty value hides every device from CUDA"
fi
unset CUDA_DEVICE_ORDER 2>/dev/null || true

# (c) JAX_PLATFORMS=cpu (or JAX_PLATFORM_NAME=cpu) silently forces CPU execution.
for forced in JAX_PLATFORMS JAX_PLATFORM_NAME; do
  if [[ ${!forced-} == *cpu* ]]; then
    echo "xulab-bc2: unsetting ${forced}=${!forced} which would force CPU execution"
    unset "${forced}"
  fi
done

# --- Allocation evidence ---------------------------------------------------
echo "xulab-bc2: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 || \
  echo "xulab-bc2: nvidia-smi unavailable in this allocation"

# --- In-allocation diagnostics --------------------------------------------
# Runs in this very job, so it needs no separate GPU test job. It reports device
# nodes, the nvidia_uvm node CUDA needs but nvidia-smi does not, libcuda
# resolution, cuInit and the JAX backend.
GPU_RUN_MODE="direct"
DIAGNOSTICS="${XULAB_BC2_HOME}/lab/scripts/gpu_diagnostics.sh"
if [[ ${XULAB_DIAGNOSTICS} == 1 && -f ${DIAGNOSTICS} ]]; then
  echo "xulab-bc2: diagnostics in the batch step"
  # set -e is active: capture the status without letting a failure end the job,
  # because a failing diagnostic is information, not a reason to stop here.
  diagnostic_status=0
  bash "${DIAGNOSTICS}" || diagnostic_status=$?
  echo "xulab-bc2: batch-step diagnostics exit ${diagnostic_status}"
  # A failure in the batch step and a pass under srun is the signature of a
  # cluster that binds the GRES to the step, so collect both views once rather
  # than guessing which one is true.
  if [[ ${diagnostic_status} -ne 0 && ${XULAB_SRUN} != never ]] \
     && command -v srun > /dev/null 2>&1 && [[ -n ${SLURM_JOB_ID:-} ]]; then
    echo "xulab-bc2: repeating the diagnostics as an srun step for comparison"
    srun_status=0
    srun ${XULAB_SRUN_FLAGS} bash "${DIAGNOSTICS}" || srun_status=$?
    echo "xulab-bc2: srun-step diagnostics exit ${srun_status}"
    if [[ ${srun_status} -eq 0 ]]; then
      echo "xulab-bc2: the GPU is reachable from an srun step but not from the batch"
      echo "xulab-bc2: step: this cluster binds the GRES to the step, so the campaign"
      echo "xulab-bc2: will run under srun"
    fi
  fi
fi

# --- Gate: the lab copy must be the imported package, on a real JAX GPU ----
# Both facts are checked in one short process that exits before the campaign
# starts, and a failure here is a nonzero exit rather than a CPU fallback.
#
# The gate is attempted as an srun step first. On a cluster with cgroup device
# constraints the GRES binding is attached to the step, so a batch-step process
# can see /dev/nvidiactl (enough for nvidia-smi) while CUDA finds no device.
# Whichever mode passes the gate is the mode the campaign then runs in.
gate_script='
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
'

gate_passed=0
if [[ ${XULAB_SRUN} != never ]] && command -v srun > /dev/null 2>&1 && [[ -n ${SLURM_JOB_ID:-} ]]; then
  echo "xulab-bc2: gate attempt 1: srun ${XULAB_SRUN_FLAGS}"
  if srun ${XULAB_SRUN_FLAGS} "${XULAB_PYTHON}" -s -c "${gate_script}"; then
    GPU_RUN_MODE="srun"
    gate_passed=1
  else
    echo "xulab-bc2: the srun step did not pass the gate (exit $?)"
  fi
fi
if [[ ${gate_passed} -eq 0 && ${XULAB_SRUN} != always ]]; then
  echo "xulab-bc2: gate attempt 2: directly in the batch step"
  if "${XULAB_PYTHON}" -s -c "${gate_script}"; then
    GPU_RUN_MODE="direct"
    gate_passed=1
  else
    echo "xulab-bc2: the batch step did not pass the gate (exit $?)"
  fi
fi
[[ ${gate_passed} -eq 1 ]] || fail \
  "no usable JAX GPU in this allocation (import origin and GPU both checked).
       The diagnostics above name the cause. The three that produce
       'cuInit(0) failed: CUDA_ERROR_NO_DEVICE' while nvidia-smi works are:
         * a stub libcuda on LD_LIBRARY_PATH   (search the log for 'STUB')
         * /dev/nvidia-uvm missing from this step's cgroup (search for 'MISSING')
         * CUDA_VISIBLE_DEVICES naming a device the step was not allocated
       Rerun the diagnostics alone in an interactive allocation with
         salloc --partition=gpu --gres=gpu:h200:1 -c 8 --mem=16G -t 00:20:00
         bash lab/scripts/gpu_diagnostics.sh" 4
echo "xulab-bc2: gate passed in ${GPU_RUN_MODE} mode"

# --- Campaign --------------------------------------------------------------
echo "xulab-bc2: launching (${GPU_RUN_MODE}): ${XULAB_PYTHON} bindcraft.py design ${CONFIG} $*"
set +e
if [[ ${GPU_RUN_MODE} == srun ]]; then
  srun ${XULAB_SRUN_FLAGS} "${XULAB_PYTHON}" -u bindcraft.py design "${CONFIG}" "$@"
else
  "${XULAB_PYTHON}" -u bindcraft.py design "${CONFIG}" "$@"
fi
status=$?
set -e
echo "xulab-bc2: campaign exit status ${status} finished=$(date -Is)"
exit "${status}"
