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
| `scripts/gpu_diagnostics.sh` | PREPARED, exercised locally | In-allocation GPU report: SLURM GRES and bindings, `/dev/nvidia*` nodes (including the `nvidia-uvm` node CUDA needs but `nvidia-smi` does not), driver version, which `libcuda` would be loaded and whether it is a stub, then `cuInit` and the JAX backend. Run by the launcher inside every campaign job, so no separate GPU test job is needed. |
| `scripts/gpu_probe.py` | PREPARED, exercised locally | The `cuInit`/JAX half of the above, as a standalone probe. Exit 0 GPU usable, 2 no libcuda, 3 driver init failed, 4 no JAX GPU, 5 no JAX. |
| `scripts/make_release.sh` | PREPARED | Builds the HPC transfer archive from the Git object store (LF, recorded revision) plus the lab additions, writes `BUILD_REVISION.txt`, and verifies content and line endings before writing the archive. |
| `scripts/validate_outputs.py` | PREPARED, exercised on synthetic fixtures only | Read-only campaign validator producing a PASS/FAIL/BLOCKED/NOT TESTED markdown report. Level A needs an explicit **JAX GPU backend** line plus exit status 0; `CUDA_VISIBLE_DEVICES`/`nvidia-smi` count as supporting evidence only. Lightweight mmCIF text check and full `biotite`/`gemmi` parse are reported separately. |
| `notebooks/xulab_bindcraft_h200.ipynb` | IMPLEMENTED, tested on fixtures only | Jupyter interface for the CMRI HPC, adapted from the official `bindcraft.ipynb`: environment check, target and campaign forms, SLURM submission to an H200, job status, result tables, interface and confidence readings, 3D complexes, export, audit. |
| `notebooks/xulab_hpc.py` | IMPLEMENTED, 90 local assertions + 127 for the scFv layer | The notebook's non-widget logic: paths, settings building, `sbatch` submission, job registry, campaign reading, FASTA and file export, validator integration, and the scFv VH/VL and CDR annotation behind the Top 10 report and the individual 3D viewer. Importable and testable without Jupyter, SLURM or a GPU. |
| `targets/CD19_ECD.pdb` | BUILT and CHECKED locally, NOT ON THE HPC | The CD19 campaign target: RCSB 6AL5 chain A, residues 21-277, 227 residues. Fab chains, glycans, waters and the C-terminal His tag removed; deposited numbering and the three unplaced segments left exactly as deposited. Provenance is in the file's `REMARK 999` lines. |
| `targets/CD19_ECD.md` | RECORD | Source entry, construct differences, the two-antibody epitope consensus, the four hotspots and the membrane / CD81 / glycan accessibility checks, with the distances each was decided on. |
| `scripts/prepare_cd19_target.py` | IMPLEMENTED, run locally | Builds `targets/CD19_ECD.pdb` from a downloaded RCSB entry and can print the antibody epitope the source entry contains. Standard library only. |
| `configs/cd19_scfv_discovery.json` | PREPARED, NOT RUN | The CD19 scFv campaign: `modality: scFv`, the prepared target with hotspots `A162,A219,A220,A222`, 10 final designs, 500 trajectories, `resume: true`. Every filter and CDR span is the shipped default. |
| `scripts/test_scfv_annotation.py` | IMPLEMENTED, 127 assertions PASS | Local tests for the scFv annotation, ranking and export helpers, including every refusal path. Needs no SLURM, GPU, pandas or py3Dmol. |
| `scripts/test_notebook_cells.py` | IMPLEMENTED, PASS | Executes notebook cells 7-9 and every widget callback against four fixture campaigns, so a cell cannot first fail inside an H200 allocation. |
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

- The source copy and this notebook directory live in HOME at
  `/home/ypan/BindCraft2/BindCraft2_deploy` (about 100 MB, including the bundled
  ProteinMPNN checkpoints).
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
