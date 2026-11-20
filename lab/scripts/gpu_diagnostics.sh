#!/bin/bash
# ---------------------------------------------------------------------------
# Xu Lab BindCraft2 - GPU diagnostics inside a real allocation
#
# Collects, in one pass and in the order that actually discriminates between
# causes, the evidence needed when nvidia-smi sees a card but CUDA will not
# initialise (the job 1014429 symptom: cuInit(0) -> CUDA_ERROR_NO_DEVICE):
#
#   1. what SLURM gave this step          (GRES, CUDA_VISIBLE_DEVICES, bindings)
#   2. what the kernel exposes            (/dev/nvidia*, nvidia_uvm module)
#   3. what nvidia-smi reports            (card, driver version)
#   4. which libcuda would be loaded      (ldconfig, LD_LIBRARY_PATH, stubs)
#   5. whether cuInit and JAX succeed     (lab/scripts/gpu_probe.py)
#
# It is called by lab/slurm/run_bindcraft_h200.slurm inside the campaign job, so
# no separate GPU test job is needed. It can also be run by hand in an
# interactive allocation:
#
#   salloc --partition=gpu --gres=gpu:h200:1 --cpus-per-task=8 --mem=16G -t 00:20:00
#   bash lab/scripts/gpu_diagnostics.sh
#
# Exit status mirrors gpu_probe.py: 0 GPU usable, 2 no libcuda, 3 driver init
# failed, 4 no JAX GPU, 5 no JAX. It changes nothing and installs nothing.
# ---------------------------------------------------------------------------
set -uo pipefail

XULAB_BC2_HOME="${XULAB_BC2_HOME:-/home/ypan/BindCraft2/BindCraft2_deploy}"
XULAB_PYTHON="${XULAB_PYTHON:-/home/ypan/.conda/envs/bindcraft2/bin/python}"
PROBE="${XULAB_BC2_HOME}/lab/scripts/gpu_probe.py"

section() { printf '\nxulab-diag: === %s ===\n' "$1"; }
line()    { printf 'xulab-diag: %s\n' "$1"; }

line "host $(hostname) job ${SLURM_JOB_ID:-none} step ${SLURM_STEP_ID:-none} at $(date -Is)"

# --- 1. what SLURM gave this step ------------------------------------------
section "slurm allocation"
for name in SLURM_JOB_ID SLURM_STEP_ID SLURM_JOB_PARTITION SLURM_JOB_GPUS \
            SLURM_STEP_GPUS SLURM_GPUS_ON_NODE SLURM_GPUS SLURM_JOB_GRES \
            SLURM_GRES SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE \
            CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL NVIDIA_VISIBLE_DEVICES; do
  line "${name}=${!name-<unset>}"
done
if command -v scontrol > /dev/null 2>&1 && [[ -n ${SLURM_JOB_ID:-} ]]; then
  line "scontrol TRES: $(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
      | tr ' ' '\n' | grep -E '^(TRES|GRES|AllocTRES)' | paste -sd' ' -)"
fi

# --- 2. what the kernel exposes to this cgroup -----------------------------
section "device nodes and driver module"
# nvidia-smi only needs /dev/nvidiactl and /dev/nvidiaN; CUDA additionally needs
# /dev/nvidia-uvm. A missing uvm node is a classic CUDA_ERROR_NO_DEVICE cause.
if ls -l /dev/nvidia* 2>/dev/null; then :; else line "no /dev/nvidia* visible here"; fi
for node in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
  [[ -e ${node} ]] && line "present ${node}" || line "MISSING ${node}"
done
if [[ -r /proc/driver/nvidia/version ]]; then
  line "driver: $(tr '\n' ' ' < /proc/driver/nvidia/version)"
else
  line "/proc/driver/nvidia/version not readable"
fi
if command -v lsmod > /dev/null 2>&1; then
  line "modules: $(lsmod 2>/dev/null | awk '/^nvidia/ {printf "%s ", $1}')"
fi

# --- 3. nvidia-smi ---------------------------------------------------------
section "nvidia-smi"
if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi -L 2>&1 | sed 's/^/xulab-diag: /'
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_mode,persistence_mode \
    --format=csv 2>&1 | sed 's/^/xulab-diag: /'
else
  line "nvidia-smi not on PATH in this step"
fi

# --- 4. which libcuda would win --------------------------------------------
section "cuda driver library resolution"
if command -v ldconfig > /dev/null 2>&1; then
  ldconfig -p 2>/dev/null | grep -E 'libcuda\.so' | sed 's/^/xulab-diag: ldconfig /' \
    || line "ldconfig lists no libcuda.so"
fi
line "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
IFS=':' read -r -a library_dirs <<< "${LD_LIBRARY_PATH:-}"
for directory in "${library_dirs[@]}"; do
  [[ -z ${directory} ]] && continue
  for name in libcuda.so.1 libcuda.so; do
    if [[ -e "${directory}/${name}" ]]; then
      if [[ ${directory} == *"/stubs"* ]]; then
        line "STUB   ${directory}/${name}  <- returns CUDA_ERROR_NO_DEVICE if loaded"
      else
        line "driver ${directory}/${name}"
      fi
    fi
  done
done
line "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"

# --- 5. cuInit and JAX, in this very process tree --------------------------
section "cuInit and jax backend"
if [[ ! -f ${PROBE} ]]; then
  line "probe not found at ${PROBE}"
  exit 1
fi
if [[ ! -x ${XULAB_PYTHON} ]]; then
  line "interpreter not executable: ${XULAB_PYTHON}"
  exit 1
fi
"${XULAB_PYTHON}" -s "${PROBE}"
status=$?
section "verdict"
case ${status} in
  0) line "GPU usable in this step" ;;
  2) line "libcuda.so.1 unavailable - the driver libraries are not on this step's library path" ;;
  3) line "CUDA driver present but cuInit failed - see the ordered causes printed by the probe" ;;
  4) line "CUDA works but JAX chose no GPU - check JAX_PLATFORMS and the cuda plugin" ;;
  5) line "jax not importable with ${XULAB_PYTHON}" ;;
  *) line "probe returned ${status}" ;;
esac
exit ${status}
