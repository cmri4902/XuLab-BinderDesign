# Xu Lab BindCraft2 - upstream source inventory

Generated during Phase 1A from the local clone at `C:\XuLab\BindCraft2`.
Regenerate after any upstream fetch with the commands at the end of this file.

## Upstream revision

| Field | Value |
| --- | --- |
| Remote | `https://github.com/PacesaLab/BindCraft2.git` |
| Branch | `main` (tracking `origin/main`) |
| Commit SHA | `8c1a858fdc1bfc94c60846ff11a0d01c9aa44136` |
| Short SHA | `8c1a858` |
| `git describe --tags` | `v1.0.4-4-g8c1a858` |
| Commit date | 2026-11-05T10:34:29+01:00 |
| Commit subject | Merge pull request #50 from PacesaLab/fix/biotite-v2 |
| Tree SHA | `2af66313e10bc8306ebe86ed633d71ca73280798` |
| `pyproject.toml` version | 1.0.4 |
| Tracked files | 186 |
| Upstream tags fetched | v1.0.0 v1.0.1 v1.0.2 v1.0.3 v1.0.4  |

The clone was created with `git init` + `git remote add` + `git fetch` (rather than
`git clone`) because the target directory already held the deployment
specification. All upstream refs, tags and history are present: `git log`,
`git tag` and `git branch -r` work normally.

## Top-level tracked content

| Path | Tracked files | Purpose (upstream) |
| --- | --- | --- |
| `.github/` | 1 | CI workflow |
| `bindcraft/` | 83 | complete Python package: AlphaFold2/ColabDesign code, ProteinMPNN, losses, filters, campaign engine, bundled MPNN weights |
| `containers/` | 4 | Dockerfile and build script (not used in Phase 1) |
| `docs/` | 20 | Sphinx documentation incl. installation.md, outputs.md, reference.md |
| `examples/` | 28 | example campaign settings files, incl. pdl1.json (unmodified) |
| `scaffolds/` | 4 | binder scaffold inputs |
| `settings/` | 37 | core/modality/property/target presets and target structures |
| root files | 9 | CLI, notebook, slurm template, install.sh, packaging, license, README (the upstream `README.md` is now `README_UPSTREAM.md`, same blob) |

## Critical upstream components confirmed present

| Component | Evidence |
| --- | --- |
| Python package | `bindcraft/` with 70 tracked `.py` modules |
| AlphaFold / ColabDesign-derived code | `bindcraft/af/` (35 files), `bindcraft/af2.py` |
| ProteinMPNN code | `bindcraft/mpnn/` (5 files), `bindcraft/proteinmpnn.py`, `bindcraft/MPNN_stage.py` |
| Bundled MPNN checkpoints | 12 `.npz` files in weights_neutral / weights_negative / weights_positive (77M total) |
| CLI entry points | `bindcraft.py`, `bindcraft/cli.py` (commands: design, archive, unarchive, fetch-weights, filter, rank, score, campaign_output) |
| Official notebook | `bindcraft.ipynb` (69304 bytes) |
| Official SLURM template | `bindcraft.slurm` (kept unchanged; the lab script is a separate file) |
| Presets | `settings/core/`, `settings/modality/` (12 modalities), `settings/property/` (9), `settings/target/` (6 targets) |
| Target structures | `settings/target/structures/` (6 files, incl. hPDL1.pdb) |
| Scaffolds | `scaffolds/` (4 files) |
| Example campaigns | `examples/` (27 JSON campaigns + README) |
| Documentation | `docs/source/` (installation.md, outputs.md, reference.md, examples.md, design-guide) |
| Containers | `containers/` (4 files) |
| License | `LICENSE` (6372 bytes) + `README_licensing_section.md` - source-available with hosting restrictions, preserved unmodified |

## What a complete clone still does not contain

| Item | Size | How this deployment obtains it |
| --- | --- | --- |
| AlphaFold2 parameters (`params_*.npz`, 7 models used) | ~5.3 GB | Reuse the copy already on the HPC; point `BINDCRAFT_AF2_PARAMS` at its directory. Never re-downloaded by the lab SLURM script. |
| Python / JAX / CUDA runtime | - | Existing conda environment `/home/ypan/.conda/envs/bindcraft2` (Python 3.12.16, JAX/jaxlib 0.11.2, CUDA 12 wheels). Not vendored into this repository. |
| Rosetta | - | Not part of BindCraft2 and not required for Phase 1. |

## Regeneration commands

```bash
git -C /path/to/BindCraft2 fetch origin
git -C /path/to/BindCraft2 log -1 --format='%H %cI %s'
git -C /path/to/BindCraft2 ls-files | wc -l
git -C /path/to/BindCraft2 status --short      # expect only lab/ additions and this file
```
