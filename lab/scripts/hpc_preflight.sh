#!/bin/bash
# ---------------------------------------------------------------------------
# Xu Lab BindCraft2 - HPC preflight (login node, no GPU required)
#
# Verifies, without installing or downloading anything:
#   1. the project directories exist and are writable          BLOCKING
#   2. the conda interpreter is present and runnable            BLOCKING
#   3. python imports bindcraft FROM THE LAB COPY               BLOCKING
#   4. every module and ProteinMPNN checkpoint is present       BLOCKING
#   5. the existing AlphaFold parameters can be located         BLOCKING
#   6. Linux-executed scripts have LF line endings              BLOCKING
#   7. no stub libcuda / forced-CPU JAX in the environment    BLOCKING
#   8. free space, nvidia-smi, quota readings                   WARNING only
#
# Run from the lab source tree on the login node:
#   cd /home/ypan/BindCraft2/BindCraft2_deploy
#   bash lab/scripts/hpc_preflight.sh
#
# Exit status:
#   0  every blocking check passed (warnings may still be printed)
#   1  at least one BLOCKING check failed - do not submit the campaign
#   2  the script could not run its checks at all (no interpreter, no tree)
#
# A GPU check is deliberately NOT done here: the login node is not the H200
# allocation. The GPU gate lives in lab/slurm/run_bindcraft_h200.slurm.
# ---------------------------------------------------------------------------
set -uo pipefail

# Source tree in HOME, campaign artefacts in project storage. Independent, so
# one can be moved or replaced without the other.
XULAB_PROJECT_ROOT="${XULAB_PROJECT_ROOT:-/project/yli4/ypan/bindcraft}"
XULAB_BC2_HOME="${XULAB_BC2_HOME:-/home/ypan/BindCraft2/BindCraft2_deploy}"
XULAB_PYTHON="${XULAB_PYTHON:-/home/ypan/.conda/envs/bindcraft2/bin/python}"

# Directories the campaign writes into. Every one of them is required: a missing
# one is a BLOCKING failure, never a warning, because the batch job must not be
# the place where a path problem is discovered.
REQUIRED_DIRECTORIES=(
  "${XULAB_PROJECT_ROOT}"
  "${XULAB_PROJECT_ROOT}/results"
  "${XULAB_PROJECT_ROOT}/logs"
  "${XULAB_PROJECT_ROOT}/cache"
  "${XULAB_PROJECT_ROOT}/tmp"
)

# Scripts that are executed by bash on Linux and must not carry CRLF.
LINUX_SCRIPTS=(
  "lab/slurm/run_bindcraft_h200.slurm"
  "lab/scripts/hpc_preflight.sh"
  "bindcraft.slurm"
  "install.sh"
)

blocking=0
warnings=0

fail()  { printf 'FAIL     %s\n' "$1"; blocking=$((blocking + 1)); }
warn()  { printf 'WARN     %s\n' "$1"; warnings=$((warnings + 1)); }
pass()  { printf 'PASS     %s\n' "$1"; }
note()  { printf '         %s\n' "$1"; }

# CR detection without a regex: stripping CR changes a file only if CR is in it.
# A dollar-quoted CR pattern is not portable across shells and can degenerate
# into an empty pattern, which would then report CRLF in every file.
holds_cr() { ! tr -d '\015' < "$1" | cmp -s - "$1"; }

echo "xulab-bc2 preflight on $(hostname) at $(date -Is)"
echo "  project root : ${XULAB_PROJECT_ROOT}"
echo "  source tree  : ${XULAB_BC2_HOME}"
echo "  interpreter  : ${XULAB_PYTHON}"
echo

# --- 1. project directories (BLOCKING) -------------------------------------
echo "[1] project directories"
for directory in "${REQUIRED_DIRECTORIES[@]}"; do
  if [[ ! -d ${directory} ]]; then
    fail "required directory missing: ${directory}"
    note "create it with: mkdir -p ${directory}"
  elif [[ ! -w ${directory} ]]; then
    fail "required directory is not writable: ${directory}"
  else
    pass "writable: ${directory}"
  fi
done
if [[ -d ${XULAB_PROJECT_ROOT} ]]; then
  probe="${XULAB_PROJECT_ROOT}/.xulab_write_probe.$$"
  if ( : > "${probe}" ) 2>/dev/null; then
    rm -f "${probe}"
    pass "write probe created and removed a file in ${XULAB_PROJECT_ROOT}"
  else
    fail "cannot create a file in ${XULAB_PROJECT_ROOT} despite the mode bits"
  fi
  if space=$(df -h "${XULAB_PROJECT_ROOT}" 2>/dev/null | tail -1); then
    note "free space: ${space}"
  else
    warn "df could not read ${XULAB_PROJECT_ROOT} (informational only)"
  fi
fi
echo

# --- 2. interpreter (BLOCKING, and fatal for the checks that follow) -------
echo "[2] interpreter"
if [[ ! -x ${XULAB_PYTHON} ]]; then
  fail "interpreter not executable: ${XULAB_PYTHON}"
  echo
  echo "preflight ABORTED: ${blocking} blocking failure(s), ${warnings} warning(s)"
  echo "the remaining checks need a working interpreter"
  exit 2
fi
if version=$("${XULAB_PYTHON}" -V 2>&1); then
  pass "interpreter ${XULAB_PYTHON} (${version})"
else
  fail "interpreter present but not runnable: ${XULAB_PYTHON}"
  echo
  echo "preflight ABORTED: ${blocking} blocking failure(s), ${warnings} warning(s)"
  exit 2
fi
echo

# --- 3./4. import origin and dependency/checkpoint inventory (BLOCKING) ----
echo "[3] import origin and [4] modules / bundled checkpoints"
if [[ ! -f ${XULAB_BC2_HOME}/bindcraft/__init__.py ]]; then
  fail "lab source copy not found at ${XULAB_BC2_HOME}/bindcraft"
else
  pass "lab source copy present at ${XULAB_BC2_HOME}"
  if BINDCRAFT_HOME="${XULAB_BC2_HOME}" \
     PYTHONPATH="${XULAB_BC2_HOME}${PYTHONPATH:+:${PYTHONPATH}}" \
     "${XULAB_PYTHON}" -s -c '
import os, sys, pathlib
home = pathlib.Path(os.environ["BINDCRAFT_HOME"]).resolve()
problems = 0
try:
    import bindcraft
except Exception as failure:                      # noqa: BLE001 - reported, not raised
    print("FAIL     bindcraft could not be imported: " + str(failure))
    raise SystemExit(1)
module = pathlib.Path(bindcraft.__file__).resolve()
if home in module.parents:
    print("PASS     imported bindcraft from " + str(module))
else:
    print("FAIL     imported bindcraft from " + str(module) + ", OUTSIDE the lab copy")
    print("         the older installation is winning the import; keep PYTHONPATH="
          + str(home) + " in every job, or pip install -e the lab copy")
    problems += 1
print("         package version " + str(getattr(bindcraft, "__version__", "unknown")))
from bindcraft.selfcheck import report as selfcheck
status = selfcheck("", shipped_only=True)
if status == 0:
    print("PASS     all required modules and bundled ProteinMPNN checkpoints present")
else:
    print("FAIL     missing modules or bundled checkpoints - see the lines above")
    problems += 1
raise SystemExit(1 if problems else 0)
'; then
    : # the inline checks printed their own PASS lines
  else
    blocking=$((blocking + 1))
    note "blocking: import origin and/or dependency inventory failed"
  fi
fi
echo

# --- 5. existing AlphaFold parameters (BLOCKING; never downloaded) ---------
echo "[5] AlphaFold parameters"
found=""
for candidate in "${BINDCRAFT_AF2_PARAMS:-}" \
                 "${BINDCRAFT_WEIGHTS:+${BINDCRAFT_WEIGHTS}/alphafold}" \
                 "${HOME}/.cache/bindcraft/alphafold" \
                 "${XULAB_PROJECT_ROOT}/cache/bindcraft/alphafold" \
                 "${XULAB_BC2_HOME}/bindcraft/weights/alphafold"; do
  [[ -z ${candidate} ]] && continue
  if compgen -G "${candidate}/params_*.npz" > /dev/null \
     || compgen -G "${candidate}/params/params_*.npz" > /dev/null; then
    found="${candidate}"
    count=$(ls "${candidate}"/params_*.npz "${candidate}"/params/params_*.npz 2>/dev/null | wc -l)
    pass "AlphaFold parameters: ${count} params_*.npz in ${candidate}"
    note "size: $(du -sh "${candidate}" 2>/dev/null | cut -f1)"
    note "export BINDCRAFT_AF2_PARAMS=${candidate}"
    # A campaign needs seven specific models; report any that is absent.
    missing=""
    for model in model_1_multimer_v3 model_2_multimer_v3 model_3_multimer_v3 \
                 model_4_multimer_v3 model_5_multimer_v3 model_1_ptm model_2_ptm; do
      if [[ ! -f ${candidate}/params_${model}.npz && ! -f ${candidate}/params/params_${model}.npz ]]; then
        missing="${missing} ${model}"
      fi
    done
    if [[ -n ${missing} ]]; then
      fail "parameter set incomplete, missing:${missing}"
    else
      pass "all seven required model parameter files present"
    fi
    break
  fi
done
if [[ -z ${found} ]]; then
  fail "no params_*.npz found in any searched location"
  note "locate the existing 5.3 GB copy, for example:"
  note "  find \$HOME /project /scratch -maxdepth 6 -name 'params_model_1_multimer_v3.npz' 2>/dev/null"
  note "then re-run with: BINDCRAFT_AF2_PARAMS=<that directory> bash lab/scripts/hpc_preflight.sh"
  note "it must be a DIRECTORY holding params_<model>.npz, never a single file"
  note "do not re-download: the parameters are already on this cluster"
fi
echo

# --- 6. line endings of Linux-executed scripts (BLOCKING) ------------------
echo "[6] line endings"
for script in "${LINUX_SCRIPTS[@]}"; do
  path="${XULAB_BC2_HOME}/${script}"
  if [[ ! -f ${path} ]]; then
    warn "script not present, not checked: ${script}"
    continue
  fi
  if holds_cr "${path}"; then
    fail "CRLF line endings in ${script} - bash will fail with 'bad interpreter'"
    note "fix with: sed -i 's/\\r$//' ${path}   (or re-stage via lab/scripts/make_release.sh)"
  else
    pass "LF line endings: ${script}"
  fi
done
echo

# --- 7. CUDA driver library resolution (BLOCKING) --------------------------
# A stub libcuda - the CUDA toolkit's link-time placeholder, usually under a
# .../stubs directory and often inherited from a conda prefix - loads happily and
# then answers cuInit with CUDA_ERROR_NO_DEVICE. That is what stopped job
# 1014429, and it is visible from a login node, before a GPU is ever requested.
echo "[7] CUDA driver library resolution"
stub_found=""
if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
  IFS=':' read -r -a library_dirs <<< "${LD_LIBRARY_PATH}"
  for directory in "${library_dirs[@]}"; do
    [[ -z ${directory} ]] && continue
    for name in libcuda.so.1 libcuda.so; do
      if [[ -e "${directory}/${name}" ]]; then
        if [[ ${directory} == *"/stubs"* ]]; then
          stub_found="${stub_found} ${directory}/${name}"
        else
          pass "driver library on LD_LIBRARY_PATH: ${directory}/${name}"
        fi
      fi
    done
  done
fi
if [[ -n ${stub_found} ]]; then
  fail "stub libcuda on LD_LIBRARY_PATH:${stub_found}"
  note "a stub libcuda answers cuInit with CUDA_ERROR_NO_DEVICE, so a job would see"
  note "no GPU even though nvidia-smi works. Remove that directory from"
  note "LD_LIBRARY_PATH before submitting; the launcher also strips it, and"
  note "XULAB_KEEP_LD_STUBS=1 disables that removal."
else
  pass "no stub libcuda ahead of the real driver on LD_LIBRARY_PATH"
fi
for forced in JAX_PLATFORMS JAX_PLATFORM_NAME; do
  if [[ ${!forced-} == *cpu* ]]; then
    fail "${forced}=${!forced} would force JAX onto the CPU in every job"
    note "unset it, or let the launcher unset it for the job"
  fi
done
note "GPU reachability itself is tested inside the allocation by"
note "lab/scripts/gpu_diagnostics.sh, which the launcher runs for every campaign"
echo

# --- 8. informational (WARNING only) ---------------------------------------
echo "[8] informational"
if command -v nvidia-smi > /dev/null 2>&1; then
  note "nvidia-smi on this node: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | paste -sd', ' -)"
  note "this is the LOGIN/interactive node, not the batch H200 allocation"
else
  warn "nvidia-smi not available here (expected on a login node)"
fi
if command -v sbatch > /dev/null 2>&1; then
  pass "sbatch found: $(command -v sbatch)"
else
  warn "sbatch not on PATH from this shell; submit from a node that has it"
fi
echo

# --- summary ---------------------------------------------------------------
echo "preflight summary: ${blocking} blocking failure(s), ${warnings} warning(s)"
if [[ ${blocking} -gt 0 ]]; then
  echo "RESULT: BLOCKED - fix every FAIL above before submitting the campaign"
  exit 1
fi
echo "RESULT: READY - blocking checks passed; warnings above are informational"
exit 0
