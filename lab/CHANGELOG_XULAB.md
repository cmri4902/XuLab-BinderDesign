# Xu Lab change log

Upstream baseline: `https://github.com/PacesaLab/BindCraft2.git`, branch `main`,
commit `8c1a858fdc1bfc94c60846ff11a0d01c9aa44136` (`v1.0.4-4-g8c1a858`,
`pyproject.toml` version 1.0.4), fetched 2026-11-20.

The upstream license requires that modifications be marked. This file is that
record.

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
