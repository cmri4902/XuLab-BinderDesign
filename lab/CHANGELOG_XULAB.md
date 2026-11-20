# Xu Lab change log

Upstream baseline: `https://github.com/PacesaLab/BindCraft2.git`, branch `main`,
commit `8c1a858fdc1bfc94c60846ff11a0d01c9aa44136` (`v1.0.4-4-g8c1a858`,
`pyproject.toml` version 1.0.4), fetched 2026-11-20.

The upstream license requires that modifications be marked. This file is that
record.

## 2026-11-20 (Phase 3a) - GPU initialisation fix after job 1014429

The first real H200 job failed in JAX GPU initialisation. From the job's own
logs: an H200 NVL visible to `nvidia-smi`, `CUDA_VISIBLE_DEVICES=0`, the correct
conda environment, imports resolving to the lab source copy, AlphaFold
parameters found, preflight READY, JAX 0.11.2 with the CUDA 12 plugin - and
`cuInit(0) failed: CUDA_ERROR_NO_DEVICE`, after which JAX fell back to the CPU
and the launcher's gate stopped the campaign (the gate behaved correctly).

`nvidia-smi` is a separate binary needing only `/dev/nvidiactl`, so it can
succeed where CUDA cannot initialise. The causes that produce exactly this split
are a stub `libcuda` on `LD_LIBRARY_PATH`, a missing `/dev/nvidia-uvm` in the
step's device cgroup, a `CUDA_VISIBLE_DEVICES` naming a device the step was not
granted, GRES bound to the step rather than the batch step, and
`JAX_PLATFORMS=cpu`. Rather than guess, the launcher now gathers the evidence
inside the real allocation and repairs what it can.

### Added

| File | What it does |
| --- | --- |
| `lab/scripts/gpu_probe.py` | Loads `libcuda.so.1`, reports where it came from and whether a stub is ahead of it, calls `cuDriverGetVersion`, `cuInit`, `cuDeviceGetCount` and names each device, then imports JAX and reports the backend and devices. Prints the ordered causes when `cuInit` returns `CUDA_ERROR_NO_DEVICE`. Exit 0 usable, 2 no libcuda, 3 driver init failed, 4 no JAX GPU, 5 no JAX; `--json` for machine reading. |
| `lab/scripts/gpu_diagnostics.sh` | The shell half: SLURM GRES and binding variables, `scontrol` TRES, `/dev/nvidia*` nodes with an explicit check for `/dev/nvidia-uvm` and `nvidia-uvm-tools`, `/proc/driver/nvidia/version`, loaded `nvidia*` modules, `nvidia-smi -L` and driver version, `ldconfig` and `LD_LIBRARY_PATH` resolution of `libcuda` with stub flagging, then the probe. Runs inside the campaign job; also usable under `salloc`. |

### Changed

| File | Change |
| --- | --- |
| `lab/slurm/run_bindcraft_h200.slurm` | (1) strips stub `libcuda` directories from `LD_LIBRARY_PATH` (`XULAB_KEEP_LD_STUBS=1` disables); (2) reconciles `CUDA_VISIBLE_DEVICES` with `SLURM_STEP_GPUS`/`SLURM_JOB_GPUS`, SLURM winning, and warns on an empty value; (3) unsets `JAX_PLATFORMS`/`JAX_PLATFORM_NAME` when they force CPU; (4) runs the diagnostics in the batch step and, if that fails, once more as an `srun` step, so the two views can be compared in one job; (5) attempts the import-origin + JAX-GPU gate as an `srun` step first and falls back to the batch step, then runs the campaign in whichever mode passed (`XULAB_SRUN`, `XULAB_SRUN_FLAGS`); (6) the failure message lists the ordered causes and the interactive command to re-run the diagnostics. The hard refusal to design on CPU is unchanged. |
| `lab/scripts/hpc_preflight.sh` | new blocking check 7: a stub `libcuda` on `LD_LIBRARY_PATH`, and `JAX_PLATFORMS`/`JAX_PLATFORM_NAME` forcing CPU. Both are visible from a login node, before a GPU is requested. |
| `lab/scripts/validate_outputs.py` | recognises `cuInit` failures, `CUDA_ERROR_NO_DEVICE`, `Unable to initialize backend`, and the probe's own verdict lines; reports a `CUDA driver initialisation` check and refuses Level A when the driver failed, stating that an allocated card and an `nvidia-smi` listing mean nothing in that case. |
| `README.md`, `README_XULAB.md`, `lab/README.md` | the diagnostics feature, and a troubleshooting section that documents the failure, its causes in discriminating order, and the log lines to read. |

### Not changed

No package was installed, no CUDA or JAX version changed, no AI algorithm
touched, no repository rebuild, and no notebook functionality altered. Source
stays at `/home/ypan/BindCraft2/BindCraft2_deploy`, outputs under
`/project/yli4/ypan/bindcraft/`.

### Local tests

39 assertions: `bash -n` on all four shell scripts; the probe's behaviour with
no CUDA present (exit 2, correct verdict, no GPU claimed) and with a simulated
allocation environment including `--json`; the launcher's stub-stripping,
`CUDA_VISIBLE_DEVICES` reconciliation (stale, matching, step-over-job, empty)
and `JAX_PLATFORMS` handling, exercised by sourcing those blocks with fixture
values; the launcher's structure (diagnostics, srun-first gate, batch fallback,
campaign in the passing mode, no `pip install`, unchanged paths); preflight
READY on a clean path and BLOCKED on a stub libcuda or `JAX_PLATFORMS=cpu`; and
the validator on both a reproduction of the 1014429 log (Level A FAIL, driver
check named) and a healthy log (Level A PASS). The embedded gate script was
parsed as Python to confirm the shell quoting is intact.

Still not verified: the fix itself against the cluster. Whether job 1014429's
cause was the stub library, the uvm node, the inherited variable or the batch
step will be stated by the next job's own diagnostics, not guessed here.

## 2026-11-20 (Phase 2c) - final HPC paths

The deployment location was fixed by the user: the source tree and the notebook
live in HOME, while every campaign artefact stays on project storage.

| Role | Path |
| --- | --- |
| Source code and Jupyter notebook | `/home/ypan/BindCraft2/BindCraft2_deploy` |
| Campaign output | `/project/yli4/ypan/bindcraft/results` |
| SLURM logs | `/project/yli4/ypan/bindcraft/logs` |
| Persistent caches (JAX compilation, weight cache) | `/project/yli4/ypan/bindcraft/cache` |
| Temporary files (`TMPDIR`) | `/project/yli4/ypan/bindcraft/tmp` |
| Conda environment (unchanged) | `/home/ypan/.conda/envs/bindcraft2` |

### Changed

| File | Change |
| --- | --- |
| `lab/slurm/run_bindcraft_h200.slurm` | `XULAB_BC2_HOME` now defaults to the HOME deployment directory instead of being derived from `XULAB_PROJECT_ROOT`; the two roots are independent. `--output`/`--error`, results, cache and `TMPDIR` still point at project storage. |
| `lab/scripts/hpc_preflight.sh` | same decoupling, so the write-permission and import-origin checks test the real layout. |
| `lab/notebooks/xulab_hpc.py` | `SOURCE_TREE` defaults to the HOME deployment directory; `RESULTS_ROOT`, `LOGS_ROOT`, `CACHE_ROOT` and `TMP_ROOT` stay under the project root. All still overridable through `XULAB_*`. |
| `lab/notebooks/xulab_bindcraft_h200.ipynb` | the `sys.path` fallback for a kernel started elsewhere, the Colab comparison table, the storage note and the first-campaign instructions. No cell logic or widget behaviour changed. |
| `lab/scripts/make_release.sh` | the archive now unpacks to `BindCraft2_deploy/` (overridable with `XULAB_DEPLOY_NAME`), so no rename is needed on the HPC, and the printed transfer commands target `/home/ypan/BindCraft2/`. |
| `README.md`, `README_XULAB.md`, `lab/README.md`, `lab/notebooks/README.md` | environment tables, directory trees, transfer, submission, import-verification and troubleshooting commands. |

`lab/configs/pdl1_smoke.json` needed no change: its `project_folder` was already
`/project/yli4/ypan/bindcraft/results/pdl1_h200_smoke`.

### Not changed

No AI algorithm, no dependency, no environment, and no notebook functionality.
Upstream file contents remain untouched, and the notebook's cells still do
exactly what they did before - only the default paths they print and fall back
to have moved.

### Regression tests after the move

`bash -n` on all three shell scripts; `xulab_hpc.py` suite (90 assertions) on
both the pandas and no-pandas paths; the notebook execution suite (all 10 code
cells plus every widget callback, including a dry-run submission, campaign load,
export with zip and a validator run); `nbformat` validation; a real
`make_release.sh` build verified to unpack as `BindCraft2_deploy` with the 12
bundled checkpoints and no CRLF.

## 2026-11-20 (Phase 2b) - repository homepage

The GitHub homepage renders the root `README.md`, which was upstream's project
page: header image, installation instructions and upstream-specific guidance
that do not describe this deployment.

### Changed

| Path | Change |
| --- | --- |
| `README.md` -> `README_UPSTREAM.md` | **Renamed, content unchanged** (identical Git blob `f55d8c9`). The upstream project documentation stays in the repository and is linked from the new README, the design guide and `docs/source/`. |
| `README.md` (new file) | Xu Lab AI Binder Design Platform README: overview, H200/SLURM environment table, feature table, the notebook path, usage, expected outputs, a validation-status table, and licensing/attribution. |

No upstream *content* was edited and nothing was deleted: the upstream README's
bytes are preserved under the new name, `LICENSE` and
`README_licensing_section.md` are untouched, `docs/` and `examples/` keep their
documentation, and the design algorithms, notebook and settings are unchanged.
`examples/README.md` links back to `../README.md`, which now reaches the lab
README; its deep links to upstream sections (for example
`#design-modalities`) live in `README_UPSTREAM.md`.

### Honesty constraints in the new README

The status banner, the per-item validation table and the outputs section all
state that no campaign has been run from this copy, that no binder has been
produced, that GPU execution is unproven, and that computational acceptance is
not experimental evidence.

## 2026-11-20 (Phase 2) - Jupyter interface for the CMRI HPC

Authorised by the user after the Phase 1 review. Upstream code is still
unmodified: `bindcraft.ipynb` is left exactly as shipped and the Xu Lab notebook
is a separate, adapted file.

### Added

| File | What it is |
| --- | --- |
| `lab/notebooks/xulab_bindcraft_h200.ipynb` | 22 cells (10 code), nbformat 4.5, kernel `python3`. Adapted from the official notebook's eight-step shape with the Colab machinery replaced. |
| `lab/notebooks/xulab_hpc.py` | The notebook's logic as an importable module, so it is reviewable and testable outside Jupyter. |

### Colab to Jupyter/SLURM changes

| Upstream Colab behaviour | Replacement |
| --- | --- |
| Google Drive mount | `/project/yli4/ypan/bindcraft/` paths, overridable through `XULAB_*` environment variables |
| `pip install` of ipywidgets/py3Dmol into the runtime | existing `bindcraft2` conda environment; a missing module is reported, never installed |
| `google.colab.output.enable_custom_widget_manager()` | plain `ipywidgets` (no `google.colab` import in any code cell) |
| campaign runs inside the notebook process | `sbatch lab/slurm/run_bindcraft_h200.slurm <settings.json>`; the kernel does no design work, so closing the notebook leaves the job running |
| no job records | `results/.xulab_jobs.json` registry, so a later session finds the job, its log and its results folder |
| `files.download(zip)` | export folder and zip under `results/exports/`, plus Jupyter `FileLink`s and the `scp` command |
| AlphaFold parameters downloaded on demand | existing cluster copy located and passed as `BINDCRAFT_AF2_PARAMS`; nothing is downloaded |
| `no_filters` demo shortcut | deliberately not offered: filtering off makes a campaign's numbers meaningless |

### What the notebook shows and exports

Ranked accepted designs; candidates with their `failed_filters` and a rejection
tally; attempts with their termination stage; per-design confidence
(`i_pDAE`, `i_pTM`, `i_pAE`, pLDDT) and interface readings (contacting residues,
buried area, hotspot contact, clashes); 3D complexes through `py3Dmol`; binder
FASTA (one record per binder chain, split on the `/` separator); the ranked,
candidate, attempt and summary CSVs; `campaign_metadata.json`; the accepted
`.cif` files; and the `validate_outputs.py` audit with its Level A / Level B
verdicts.

### Honesty constraints kept in the interface

- The environment cell reports `GPU / H200: NOT TESTED` and says why: a Jupyter
  node is not the batch allocation.
- A dry run never returns a job id, and a missing `sbatch` is reported instead
  of being worked around.
- Zero accepted designs is presented as an ordinary outcome with the diagnostic
  order from `docs/source/outputs.md`, not as a failure.
- Every table states that acceptance is computational and not evidence of
  binding.

### Testing

90 assertions against synthetic fixtures, run twice (with and without pandas):
paths and environment reporting, AlphaFold parameter search including an
incomplete set, preset and custom targets, settings building (name sanitising,
length ordering, scaffolded modalities, unknown modality rejection, every
written key checked against `settings/core/reference.json`), `sbatch` command
construction, the job registry, campaign loading, zero-accepted handling,
per-design reports, FASTA generation, mmCIF chain reading, export with zip, and
the validator integration. All 10 notebook code cells execute and every widget
callback fires. `nbformat` validates the notebook with no warnings.

Not tested: a real `sbatch`, `squeue`/`sacct`, an H200 allocation, a real
campaign, `py3Dmol` rendering in institutional Jupyter. No GPU execution is
claimed.

## 2026-11-20 (revision 2) - Phase 1 review corrections

Four issues raised in the Phase 1 review. Upstream code is still unmodified.

### 1. Line endings

The clone was created on Windows with `core.autocrlf=true`, so the entire
working tree - including the upstream `install.sh`, `bindcraft.slurm` and
`containers/build.sh` - was CRLF on disk. Transferring that tree would give
every shell script a `#!/bin/bash\r` shebang and `bad interpreter: /bin/bash^M`
on Linux.

- **Added** `.gitattributes` at the repository root (new file, no upstream file
  touched): `* text=auto eol=lf` plus explicit `eol=lf` for every Linux-read
  text type, `binary` for `*.npz`/`*.png`/archives, and `*.bat -text` because
  upstream committed `docs/make.bat` with CRLF and it must stay byte-identical.
- **Repository config** (not file content): `core.autocrlf=false`,
  `core.eol=lf`, set locally for this clone, then the working tree was
  re-checked out from the index and the index renormalised. `git diff HEAD` is
  empty afterwards, so upstream content is provably unchanged while the disk
  now holds LF.
- **Enforcement** in both directions: `lab/scripts/make_release.sh` refuses to
  build an archive whose `*.sh`/`*.slurm` files carry CR, and
  `lab/scripts/hpc_preflight.sh` treats CRLF in a Linux-executed script as a
  blocking failure.
- Both checks use `holds_cr() { ! tr -d '\015' < "$1" | cmp -s - "$1"; }` rather
  than a `grep` with a dollar-quoted CR pattern. Local testing showed that
  pattern degenerating into an empty pattern in some shell contexts, where it
  matches every line and would report CRLF in every file - a check that fails
  loudly on correct input is as bad as one that passes on broken input.

### 2. Preflight exit codes

`lab/scripts/hpc_preflight.sh` previously counted only `FAIL` lines, so a
missing project directory or absent AlphaFold parameters printed `MISSING` and
the script still exited 0. Rewritten with an explicit blocking/warning split:

- Blocking (exit 1, `RESULT: BLOCKED`): missing or unwritable required
  directory, failed write probe, interpreter present but not runnable, import
  resolved outside the lab copy, missing module or bundled checkpoint, no
  AlphaFold parameters found, an incomplete parameter set (the seven required
  models are checked individually), CRLF in a Linux-executed script.
- Exit 2: the checks could not run at all (no usable interpreter).
- Warning (exit 0, counted and printed): `df` unreadable, no `nvidia-smi` on
  the login node, `sbatch` not on PATH, a listed script absent.

### 3. Output validation

`lab/scripts/validate_outputs.py`:

- Level A now requires **explicit JAX GPU backend evidence**
  (`backend=gpu|cuda|rocm`, a `jax device gpu:` line, an XLA GPU client line).
  `CUDA_VISIBLE_DEVICES`, `nvidia-smi` card names, `SLURM_JOB_GPUS` and the
  driver-level "N GPUs visible" note moved to a separate *supporting evidence*
  class that can never grant Level A; allocation-only evidence is an explicit
  FAIL with that reason stated.
- The campaign **exit status** is parsed (`campaign exit status <n>` and generic
  exit lines) and SLURM termination markers are detected. A nonzero exit, a
  termination marker, or a missing exit line prevents a Level A PASS.
- **Per-candidate structure correspondence** is its own check: every accepted
  row without a complex structure file is named individually and the check
  fails. A row whose only file is a `_monomer` is reported as such. Level B now
  requires a sequence *and* a structure for **every** accepted design.
- Structure checking is split into a **lightweight text check** (nonempty,
  `data_` block, `_atom_site` loop, ATOM rows, chain labels; explicitly no
  chemistry or geometry validation) and a **full structure parse** through
  `biotite` or `gemmi`, reported `NOT TESTED` when neither is importable rather
  than being conflated with the text check. `--no-full-parse` skips it.

### 4. Deployment transfer

- **Added** `lab/scripts/make_release.sh`: exports tracked files from the Git
  object store (recorded revision, LF endings), adds the uncommitted lab files,
  removes `__pycache__`, converts stray CR, writes `lab/BUILD_REVISION.txt`
  (remote, branch, commit, describe, dirty state, file count, pyproject version,
  SHA-256 of every lab file, untracked-file list), then verifies the 12 bundled
  MPNN checkpoints, the required entry points and the absence of CR before
  writing the archive. Also emits a `*.gitbundle` with the full history.
- `README_XULAB.md` transfer section rewritten: three options (release archive,
  rsync, Git remote) with the shell for each one stated, because the previous
  version mixed `rsync` and `/c/...` paths into a PowerShell block and used
  brace expansion (`{summary.csv,...}`) that PowerShell does not support. Added
  the LF verification commands (`git ls-files --eol`, `grep -rlU $'\r'`), the
  `sed -i 's/\r$//'` repair, and a note that `tar -xzf` merges into an existing
  `BindCraft2/` so the old source directory should be moved aside first (results
  are siblings and are untouched).

### Files changed in revision 2

| File | Change |
| --- | --- |
| `.gitattributes` | added |
| `lab/scripts/hpc_preflight.sh` | rewritten (exit-code semantics, CRLF check, model completeness) |
| `lab/scripts/validate_outputs.py` | rewritten (JAX evidence, exit codes, per-candidate correspondence, parse levels) |
| `lab/scripts/make_release.sh` | added |
| `README_XULAB.md` | transfer, preflight, validation, line-ending and status sections updated |
| `lab/README.md` | new files and status vocabulary updated |
| `lab/CHANGELOG_XULAB.md` | this section |

No upstream file was modified in revision 2 either; `git diff HEAD` remains
empty against commit `8c1a858`.

## 2026-11-20 - Phase 1 preparation

### Upstream files modified

**None.** Every upstream tracked file is unchanged at commit `8c1a858`:

```bash
git -C C:\XuLab\BindCraft2 status --short      # only untracked lab/ additions
git -C C:\XuLab\BindCraft2 diff --stat HEAD    # empty
```

In particular `examples/pdl1.json`, `bindcraft.slurm`, `bindcraft.py`,
`bindcraft.ipynb`, `install.sh`, `pyproject.toml` and `LICENSE` are untouched.
No AlphaFold, ProteinMPNN or design-algorithm code was rewritten.

### Files added (all under `lab/`, plus one root README)

| File | Purpose | Why it is additive rather than a modification |
| --- | --- | --- |
| `lab/slurm/run_bindcraft_h200.slurm` | H200/SLURM launcher for this site | `bindcraft.slurm` is generic (it autosizes from `sinfo` and cannot know this site's partition, GPU type, project tree or conda environment). Keeping the upstream template unchanged preserves it as a reference and keeps upstream merges clean. |
| `lab/configs/pdl1_smoke.json` | PD-L1 smoke campaign for the H200 | `examples/pdl1.json` must stay as shipped; a campaign needs an absolute `project_folder` under `/project/yli4/ypan/bindcraft/results/`, which is site-specific. |
| `lab/scripts/hpc_preflight.sh` | Login-node permission / import-origin / weight checks | No upstream equivalent. `python -m bindcraft.selfcheck` covers modules and checkpoints but not the project tree, the import origin against an older installation, or the shared AlphaFold copy. |
| `lab/scripts/validate_outputs.py` | Read-only campaign output validator and report | No upstream equivalent. Uses only the output contract documented in `docs/source/outputs.md`. |
| `lab/notebooks/README.md` | Phase 2 placeholder | Keeps the directory in the tree without implementing Phase 2. |
| `lab/README.md`, `lab/SOURCE_INVENTORY.md`, `lab/CHANGELOG_XULAB.md` | Lab records | Provenance and change marking required by the license. |
| `README_XULAB.md` | Deployment and handoff documentation | Upstream `README.md` is left as shipped. |
| `XULAB_BindCraft2_Deployment_Spec.md` (pre-existing, untracked) | The specification this work follows | Written by the lab, not upstream. |

### Deliberate non-changes

- No dependency, CUDA or JAX version was changed. The environment stays at
  Python 3.12.16 / JAX 0.11.2 / CUDA 12 wheels, as recorded for this site.
- Nothing was vendored into the repository: no conda packages, no AlphaFold
  parameters, no copies of external repositories.
- `bindcraft/weights/proteinmpnn/*` (the 12 bundled checkpoints, 77 MB) is kept
  exactly as distributed, including the neutral / negative / positive variants.
- No upstream file was deleted, including files the specification's illustrative
  tree did not list (`README_licensing_section.md`, `containers/`, `scaffolds/`,
  `.github/`).
- No `install.sh` run, no `.venv` created in the source tree: the existing conda
  environment is used instead, and the launcher sets
  `BINDCRAFT_LAUNCHER_SWITCHED=1` so `bindcraft.py` cannot silently re-execute
  itself under some other interpreter.

### Third-party attribution preserved

`LICENSE` (BindCraft2, source-available with hosting restrictions),
`README_licensing_section.md`, and the attribution inside
`bindcraft/af/`, `bindcraft/mpnn/` and `bindcraft/weights/README.md`
(AlphaFold, ColabDesign, ProteinMPNN / HyperMPNN) are all unmodified.
