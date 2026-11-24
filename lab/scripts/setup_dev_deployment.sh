#!/bin/bash
# ---------------------------------------------------------------------------
# Xu Lab BindCraft2 - create the isolated development deployment on the HPC
#
#   bash lab/scripts/setup_dev_deployment.sh            # check, then report
#   bash lab/scripts/setup_dev_deployment.sh --create    # also create and clone
#
# Without --create this only looks: it prints what exists, what is missing and
# what it would do. With --create it creates the development project directories
# and clones or updates the development source tree.
#
# It never writes to the production source tree or the production project tree,
# and it refuses outright if either development path is configured to be one of
# them. It copies no conda environment and no model weights: those are shared,
# read-only, and copying 5.3 GB of parameters would be waste.
#
# The proven SLURM launcher is not touched, patched or copied over. The
# development deployment runs the same launcher out of its own checkout; which
# tree and which project root a job uses is decided by the environment that
# lab/scripts/dev_env.sh exports, which the launcher already reads.
# ---------------------------------------------------------------------------

set -uo pipefail

CREATE=0
[[ "${1:-}" == "--create" ]] && CREATE=1

REPOSITORY="${XULAB_DEV_REPOSITORY:-https://github.com/cmri4902/XuLab-BinderDesign.git}"
BRANCH="${XULAB_DEV_BRANCH:-main}"

PRODUCTION_SOURCE="/home/ypan/BindCraft2/BindCraft2_deploy"
PRODUCTION_PROJECT="/project/yli4/ypan/bindcraft"

DEV_SOURCE="${XULAB_BC2_HOME:-/home/ypan/BindCraft2/BindCraft2_cd19_dev}"
DEV_PROJECT="${XULAB_PROJECT_ROOT:-/project/yli4/ypan/bindcraft_cd19_dev}"

problems=0
note() { printf '  %-10s %s\n' "$1" "$2"; }
fail() { note "REFUSED" "$1"; problems=$((problems + 1)); }

echo "Xu Lab BindCraft2 - development deployment"
echo "=========================================="
note "source" "${DEV_SOURCE}"
note "project" "${DEV_PROJECT}"
note "clone of" "${REPOSITORY} (${BRANCH})"
echo

# --- refuse to act on the production paths --------------------------------
echo "1. the development paths must not be the production ones"
for path in "${DEV_SOURCE}" "${DEV_PROJECT}"; do
  case "${path%/}" in
    "${PRODUCTION_SOURCE%/}"|"${PRODUCTION_PROJECT%/}")
      fail "${path} is a production path; set XULAB_BC2_HOME / XULAB_PROJECT_ROOT first"
      ;;
    *) note "ok" "${path} is not a production path" ;;
  esac
done
if [[ ${problems} -gt 0 ]]; then
  echo
  echo "RESULT: REFUSED - nothing was created."
  exit 2
fi

# --- the production deployment must be left alone --------------------------
echo
echo "2. the production deployment, which this script never writes to"
for path in "${PRODUCTION_SOURCE}" "${PRODUCTION_PROJECT}"; do
  if [[ -d ${path} ]]; then
    note "present" "${path}"
  else
    note "absent" "${path} (nothing to protect here)"
  fi
done
for campaign in "${PRODUCTION_PROJECT}/results"/*; do
  [[ -d ${campaign} ]] && note "existing" "campaign ${campaign##*/} (untouched)"
done 2>/dev/null

# --- shared, read-only resources -------------------------------------------
echo
echo "3. shared resources, reused and never copied"
PYTHON="${XULAB_PYTHON:-/home/ypan/.conda/envs/bindcraft2/bin/python}"
if [[ -x ${PYTHON} ]]; then
  note "ok" "interpreter ${PYTHON}"
else
  note "MISSING" "interpreter ${PYTHON} is not executable"
  problems=$((problems + 1))
fi
WEIGHTS="${BINDCRAFT_WEIGHTS:-${PRODUCTION_PROJECT}/cache/bindcraft}"
if [[ -d ${WEIGHTS} ]]; then
  count=$(find "${WEIGHTS}" -name 'params_*.npz' 2>/dev/null | wc -l)
  note "ok" "weights ${WEIGHTS} (${count} params_*.npz, read only)"
else
  note "note" "weights ${WEIGHTS} not found; the launcher searches for them at run time"
fi

# --- create ----------------------------------------------------------------
echo
echo "4. development directories"
DIRECTORIES=("${DEV_PROJECT}/results" "${DEV_PROJECT}/logs" "${DEV_PROJECT}/cache/jax"
             "${DEV_PROJECT}/tmp" "${DEV_PROJECT}/targets")
for directory in "${DIRECTORIES[@]}"; do
  if [[ -d ${directory} ]]; then
    note "exists" "${directory}"
  elif [[ ${CREATE} -eq 1 ]]; then
    if mkdir -p "${directory}" 2>/dev/null; then
      note "created" "${directory}"
    else
      note "FAILED" "could not create ${directory}"
      problems=$((problems + 1))
    fi
  else
    note "would" "mkdir -p ${directory}"
  fi
done

echo
echo "5. development source tree"
if [[ -d "${DEV_SOURCE}/.git" ]]; then
  if [[ ${CREATE} -eq 1 ]]; then
    if git -C "${DEV_SOURCE}" fetch origin "${BRANCH}" --quiet \
       && git -C "${DEV_SOURCE}" checkout --quiet "${BRANCH}" \
       && git -C "${DEV_SOURCE}" merge --ff-only --quiet "origin/${BRANCH}"; then
      note "updated" "$(git -C "${DEV_SOURCE}" rev-parse --short HEAD) on ${BRANCH}"
    else
      note "FAILED" "could not fast-forward ${DEV_SOURCE}; resolve it by hand"
      problems=$((problems + 1))
    fi
  else
    note "exists" "${DEV_SOURCE} at $(git -C "${DEV_SOURCE}" rev-parse --short HEAD 2>/dev/null)"
    note "would" "git -C ${DEV_SOURCE} pull --ff-only origin ${BRANCH}"
  fi
elif [[ -e ${DEV_SOURCE} ]]; then
  note "note" "${DEV_SOURCE} exists but is not a git clone; update it however it was made"
elif [[ ${CREATE} -eq 1 ]]; then
  if git clone --branch "${BRANCH}" "${REPOSITORY}" "${DEV_SOURCE}"; then
    note "cloned" "$(git -C "${DEV_SOURCE}" rev-parse --short HEAD) on ${BRANCH}"
  else
    note "FAILED" "clone failed; clone it by hand or copy the release archive instead"
    problems=$((problems + 1))
  fi
else
  note "would" "git clone --branch ${BRANCH} ${REPOSITORY} ${DEV_SOURCE}"
fi

# --- what the launcher will do ---------------------------------------------
echo
echo "6. what a submission from this deployment will use"
note "launcher" "${DEV_SOURCE}/lab/slurm/run_bindcraft_h200.slurm (unmodified)"
note "export" "XULAB_BC2_HOME=${DEV_SOURCE}"
note "export" "XULAB_PROJECT_ROOT=${DEV_PROJECT}"
note "logs" "--output/--error into ${DEV_PROJECT}/logs (an sbatch flag beats the directive)"
note "no srun" "the launcher runs the campaign directly in the batch step"
if [[ -f "${DEV_SOURCE}/lab/slurm/run_bindcraft_h200.slurm" ]] \
   && [[ -f "${PRODUCTION_SOURCE}/lab/slurm/run_bindcraft_h200.slurm" ]]; then
  if cmp -s "${DEV_SOURCE}/lab/slurm/run_bindcraft_h200.slurm" \
            "${PRODUCTION_SOURCE}/lab/slurm/run_bindcraft_h200.slurm"; then
    note "ok" "the dev launcher is byte-identical to the production one"
  else
    note "DIFFERS" "the dev launcher differs from the production one - check why"
    problems=$((problems + 1))
  fi
fi

echo
if [[ ${problems} -gt 0 ]]; then
  echo "RESULT: ${problems} problem(s) above. Fix them before submitting."
  exit 1
fi
if [[ ${CREATE} -eq 1 ]]; then
  cat <<NEXT
RESULT: development deployment ready.

Next, in a shell that drives the development deployment only:
  cd ${DEV_SOURCE}
  source lab/scripts/dev_env.sh
  bash lab/scripts/hpc_preflight.sh
  ${PYTHON} lab/scripts/test_scfv_annotation.py
  ${PYTHON} lab/scripts/test_target_tools.py
  ${PYTHON} lab/scripts/test_notebook_cells.py
NEXT
else
  echo "RESULT: checks only, nothing created. Re-run with --create to act."
fi
