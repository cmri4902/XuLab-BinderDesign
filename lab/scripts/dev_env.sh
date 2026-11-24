#!/bin/bash
# ---------------------------------------------------------------------------
# Xu Lab BindCraft2 - isolated DEVELOPMENT deployment
#
# Source this (do not execute it) before running anything that should touch the
# development tree and nothing else:
#
#   source lab/scripts/dev_env.sh
#   jupyter lab                       # the notebook then reads these paths
#   sbatch ... lab/slurm/run_bindcraft_h200.slurm lab/configs/<campaign>.json
#
# Everything a campaign reads or writes is addressed by one of the variables
# below, so this file is the only thing that decides which deployment a shell
# drives. No tracked file names a deployment: the proven SLURM launcher is not
# edited, and the notebook and its helper read these variables.
#
# What is isolated:
#   source tree, project root, results, logs, caches, tmp, prepared targets,
#   generated campaign JSON (written under the dev source tree), and the job
#   registry (written under the dev results root).
#
# What is deliberately SHARED, because copying it would waste 5.3 GB and a day:
#   the bindcraft2 conda environment and the AlphaFold / ProteinMPNN weights.
#   Both are read-only to a campaign.
#
# The production deployment is untouched by anything run under these variables:
#   source  /home/ypan/BindCraft2/BindCraft2_deploy
#   project /project/yli4/ypan/bindcraft
# Nothing here writes into either path.
# ---------------------------------------------------------------------------

# A label, printed by the notebook and recorded on every submission.
export XULAB_DEPLOYMENT="cd19_dev"

# The two roots. Everything else is derived from them by xulab_hpc.py and by the
# launcher, which both read these names and fall back to the production paths.
export XULAB_BC2_HOME="${XULAB_BC2_HOME_DEV:-/home/ypan/BindCraft2/BindCraft2_cd19_dev}"
export XULAB_PROJECT_ROOT="${XULAB_PROJECT_ROOT_DEV:-/project/yli4/ypan/bindcraft_cd19_dev}"

# Stated explicitly rather than left to defaulting, so a later change to a
# default cannot silently point one of them at the production tree.
export XULAB_RESULTS="${XULAB_PROJECT_ROOT}/results"
export XULAB_LOGS="${XULAB_PROJECT_ROOT}/logs"
export XULAB_CACHE="${XULAB_PROJECT_ROOT}/cache"
export XULAB_TMP="${XULAB_PROJECT_ROOT}/tmp"
export XULAB_TARGETS="${XULAB_PROJECT_ROOT}/targets"

# Shared, not copied: the same interpreter and the same model weights the
# production deployment uses. A campaign only reads them.
export XULAB_PYTHON="${XULAB_PYTHON:-/home/ypan/.conda/envs/bindcraft2/bin/python}"
export BINDCRAFT_WEIGHTS="${BINDCRAFT_WEIGHTS:-/project/yli4/ypan/bindcraft/cache/bindcraft}"
# Point this at the directory holding params_<model>.npz once it is confirmed, so
# neither the launcher nor the notebook has to search for it:
#   export BINDCRAFT_AF2_PARAMS=/abs/path/to/alphafold

# JAX compilation cache in the dev tree, so a dev run cannot invalidate or grow
# the production cache.
export JAX_COMPILATION_CACHE_DIR="${XULAB_CACHE}/jax"
export TMPDIR="${XULAB_TMP}"

cat <<SUMMARY
xulab: DEVELOPMENT deployment selected in this shell
  XULAB_DEPLOYMENT   ${XULAB_DEPLOYMENT}
  XULAB_BC2_HOME     ${XULAB_BC2_HOME}
  XULAB_PROJECT_ROOT ${XULAB_PROJECT_ROOT}
  results            ${XULAB_RESULTS}
  logs               ${XULAB_LOGS}
  targets            ${XULAB_TARGETS}
  shared (read only) ${XULAB_PYTHON}
                     ${BINDCRAFT_WEIGHTS}
  production tree    /home/ypan/BindCraft2/BindCraft2_deploy  (NOT touched)
                     /project/yli4/ypan/bindcraft             (NOT touched)
Open a new shell, or source nothing, to go back to the production deployment.
SUMMARY
