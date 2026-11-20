# `lab/` - Xu Lab additions

Everything in this directory was written by the Xu Lab. Upstream file *contents*
at commit `8c1a858` are untouched; outside this directory the repository adds
`.gitattributes`, `README_XULAB.md` and a lab `README.md`, and renames the
upstream `README.md` to `README_UPSTREAM.md` (same Git blob).

| Path | Status | What it is |
| --- | --- | --- |
| `slurm/run_bindcraft_h200.slurm` | PREPARED, NOT TESTED | H200 batch launcher: partition `gpu`, `--gres=gpu:h200:1`, 8 CPUs, 64 GB host RAM, 08:00:00. Refuses to run without a JAX GPU or without the existing AlphaFold parameters. |
| `configs/pdl1_smoke.json` | PREPARED, NOT RUN | PD-L1 smoke campaign, 20 trajectories, 1 final design, filters left at campaign defaults. |
| `scripts/hpc_preflight.sh` | PREPARED, NOT TESTED on HPC | Login-node checks with **blocking vs warning separation**: missing/unwritable project directory, unusable interpreter, wrong import origin, missing module or bundled checkpoint, missing AlphaFold parameters (all seven models), or CRLF in a Linux script each make it exit nonzero. Exit 0 READY / 1 BLOCKED / 2 could not run. |
| `scripts/make_release.sh` | PREPARED | Builds the HPC transfer archive from the Git object store (LF, recorded revision) plus the lab additions, writes `BUILD_REVISION.txt`, and verifies content and line endings before writing the archive. |
| `scripts/validate_outputs.py` | PREPARED, exercised on synthetic fixtures only | Read-only campaign validator producing a PASS/FAIL/BLOCKED/NOT TESTED markdown report. Level A needs an explicit **JAX GPU backend** line plus exit status 0; `CUDA_VISIBLE_DEVICES`/`nvidia-smi` count as supporting evidence only. Lightweight mmCIF text check and full `biotite`/`gemmi` parse are reported separately. |
| `notebooks/xulab_bindcraft_h200.ipynb` | IMPLEMENTED, tested on fixtures only | Jupyter interface for the CMRI HPC, adapted from the official `bindcraft.ipynb`: environment check, target and campaign forms, SLURM submission to an H200, job status, result tables, interface and confidence readings, 3D complexes, export, audit. |
| `notebooks/xulab_hpc.py` | IMPLEMENTED, 90 local assertions | The notebook's non-widget logic: paths, settings building, `sbatch` submission, job registry, campaign reading, FASTA and file export, validator integration. Importable and testable without Jupyter, SLURM or a GPU. |
| `SOURCE_INVENTORY.md` | RECORD | Upstream remote, commit SHA, tree SHA, file counts, component evidence. |
| `CHANGELOG_XULAB.md` | RECORD | Every Xu Lab change, with the reason. |
| `BUILD_REVISION.txt` | GENERATED, untracked | Written by `make_release.sh`: commit, tag, dirty state, SHA-256 of each lab file. Present only after a release build. |

One file sits outside this directory: `../.gitattributes`, a Xu Lab addition
that pins LF for every Linux-executed file (upstream ships no `.gitattributes`,
and a Windows checkout would otherwise hand the HPC CRLF shell scripts).

Full deployment instructions, including the Windows-to-HPC transfer and the
submission and validation commands, are in `../README_XULAB.md`.

## Status vocabulary used throughout this project

| Word | Meaning |
| --- | --- |
| PREPARED | The artefact exists and has been reviewed, but has not been executed in its target environment. |
| NOT TESTED | No execution evidence exists yet. Not a synonym for PASS. |
| PASS / FAIL | An actual run produced evidence, with a path and a line reference. |
| BLOCKED | The check could not run (missing allocation, unreadable folder, absent weights). |

## Conventions

- Results, logs, caches and temporary files go under `/project/yli4/ypan/bindcraft/`,
  never into the source tree and never into `$HOME` (≈300 GB quota).
- No Windows path appears in any file intended to run on the HPC.
- No script in `lab/` downloads weights, installs packages, or writes outside
  the project tree (`make_release.sh` writes one archive into a directory you
  name, plus `lab/BUILD_REVISION.txt`).
- Every Linux-executed file is LF. `make_release.sh` refuses to ship CR, and
  `hpc_preflight.sh` blocks on it.
- Lab scripts take paths from `XULAB_*` environment variables so a second user
  or a second project root needs no edit to a tracked file.
