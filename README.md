# Xu Lab AI Binder Design Platform

Internal protein-binder design platform of the Xu Lab (CMRI), built on
[BindCraft2](https://github.com/PacesaLab/BindCraft2) and adapted to run on the lab's NVIDIA H200
nodes through SLURM, with a Jupyter interface for lab members.

The design engine is upstream BindCraft2, unmodified: AlphaFold 2 sequence optimisation,
ProteinMPNN redesign, and validation of candidates against separate AlphaFold models and
structural filters. The Xu Lab layer adds the cluster plumbing - batch submission, environment
and weight reuse, output auditing, and the notebook - without touching the design algorithms.

> **Status.** The platform's code is complete and tested locally against synthetic fixtures.
> **No design campaign has been run on the HPC yet from this copy**, so no binder has been
> produced and no GPU execution is claimed. See [Validation status](#validation-status).

- Upstream documentation (unchanged): **[`README_UPSTREAM.md`](README_UPSTREAM.md)**, plus
  [`docs/source/`](docs/source) for settings, outputs and the design guide.
- Xu Lab deployment manual: **[`README_XULAB.md`](README_XULAB.md)**.
- Every Xu Lab change, itemised: [`lab/CHANGELOG_XULAB.md`](lab/CHANGELOG_XULAB.md).

---

## Environment

| Component | This deployment |
| --- | --- |
| GPU | NVIDIA H200, partition `gpu`, `--gres=gpu:h200:1` |
| Scheduler | SLURM; `lab/slurm/run_bindcraft_h200.slurm` (8 CPUs, 64 GB host RAM, 08:00:00) |
| Python | existing `bindcraft2` conda environment, `/home/ypan/.conda/envs/bindcraft2` (Python 3.12, JAX 0.11.2, CUDA 12 wheels) |
| Source copy and notebook | `/home/ypan/BindCraft2/BindCraft2_deploy` (HOME storage) |
| Campaign output | `/project/yli4/ypan/bindcraft/results/<campaign>/` (project storage) |
| Logs / caches | `/project/yli4/ypan/bindcraft/logs/`, `/project/yli4/ypan/bindcraft/cache/` |
| AlphaFold parameters | the existing ~5.3 GB cluster copy, located at run time through `BINDCRAFT_AF2_PARAMS`; never re-downloaded and never committed |
| ProteinMPNN / HyperMPNN weights | the 12 checkpoints shipped with the source (`bindcraft/weights/proteinmpnn/`) |

The source copy sits in HOME and every campaign artefact in project storage, so replacing or
updating the source cannot delete a completed campaign, and results never consume the HOME
quota.

## Features

| Feature | What it does |
| --- | --- |
| **Jupyter interface** | `lab/notebooks/xulab_bindcraft_h200.ipynb` - ten steps from environment check to campaign audit, for regular Jupyter on the HPC (no Colab, no Google Drive, no `pip install`). |
| **H200 batch submission** | `lab/slurm/run_bindcraft_h200.slurm` - asserts that `bindcraft` is imported from this source copy and that JAX holds a GPU, and exits nonzero rather than silently computing on CPU. Installs nothing, downloads nothing. |
| **Jobs outlive the notebook** | Every campaign is a batch job; job ids are recorded in `results/.xulab_jobs.json`, so a later notebook session finds a running campaign, its log and its results again. |
| **Design configuration** | Shipped targets or your own PDB/mmCIF/FASTA, with chains, hotspots and coldspots; 10 binder modalities; property presets; length, oligomer, design and trajectory budgets. Written to a settings file validated against the upstream setting names. |
| **Result reading** | Ranked designs, scored candidates with the filters that rejected them, and attempts with their termination stage - plus interface and confidence readings per accepted design. |
| **Output audit** | `lab/scripts/validate_outputs.py` - read-only PASS / FAIL / BLOCKED / NOT TESTED report. Compute-smoke acceptance requires explicit JAX GPU evidence from the job log, not merely an allocated card. |
| **GPU diagnostics in the allocation** | `lab/scripts/gpu_diagnostics.sh` + `lab/scripts/gpu_probe.py` - SLURM bindings, `/dev/nvidia*` nodes, driver version, `libcuda` resolution (including stub detection), `cuInit` and the JAX backend, printed by every campaign job. No separate GPU test job. |
| **Login-node preflight** | `lab/scripts/hpc_preflight.sh` - project directories, interpreter, import origin, module and checkpoint inventory, AlphaFold parameter completeness, CRLF check. Blocking failures exit nonzero. |
| **Reproducible deployment** | `lab/scripts/make_release.sh` - transfer archive built from the Git object store with a `BUILD_REVISION.txt` provenance record and LF line endings. |

Acceptance filters are always on. The Colab demo's `no_filters` shortcut is deliberately not
exposed: a campaign run without filtering produces numbers that mean nothing.

## Usage

```bash
# 1. once, on a login node
mkdir -p /project/yli4/ypan/bindcraft/{results,logs,cache/jax,tmp}
cd /home/ypan/BindCraft2/BindCraft2_deploy
conda activate bindcraft2
bash lab/scripts/hpc_preflight.sh            # must print RESULT: READY
export BINDCRAFT_AF2_PARAMS=<directory holding params_<model>.npz>

# 2a. the notebook (recommended)
jupyter lab                                  # or the institutional Jupyter launcher
#   open lab/notebooks/xulab_bindcraft_h200.ipynb with the bindcraft2 kernel
#   cell 1 environment -> cell 2 target -> cell 3 settings -> cell 4 submit
#   cell 5 status and log -> cells 6-10 results, structures, export, audit

# 2b. or the command line, identical result
sbatch --export=ALL,BINDCRAFT_AF2_PARAMS="$BINDCRAFT_AF2_PARAMS" \
  lab/slurm/run_bindcraft_h200.slurm lab/configs/pdl1_smoke.json

# 3. confirm the job really ran on a GPU, then audit it
grep -n 'backend=gpu\|device      gpu:' /project/yli4/ypan/bindcraft/logs/xulab_bc2_<jobid>.out
python lab/scripts/validate_outputs.py /project/yli4/ypan/bindcraft/results/<campaign> \
  --slurm-log '/project/yli4/ypan/bindcraft/logs/xulab_bc2_*.out'
```

Full deployment, transfer and troubleshooting instructions: [`README_XULAB.md`](README_XULAB.md).

## Expected outputs

Written under `/project/yli4/ypan/bindcraft/results/<campaign>/`:

| Output | File | Content |
| --- | --- | --- |
| **Binder sequences** | `<campaign>_binders.fasta` (notebook export) | One FASTA record per binder chain, taken from `Binder_Sequence`; multi-chain binders are split on the `/` separator. |
| **Predicted complexes** | `3_Ranked/<design>_seq<n>.cif` | Accepted binder-target complexes as mmCIF, one file per target state, with per-residue pLDDT in the B-factor column (0-100). Free-binder monomers appear as `_monomer.cif` where predicted. |
| **Interface information** | `3_Ranked/!_Ranked.csv` columns | Contacting binder and target residues, interface residue count, buried area and fraction, hotspot contact fraction, surface hydrophobicity, backbone and all-atom clashes. |
| **Confidence scores** | same table | `i_pDAE` (ranking score, 0-1), `i_pTM`, `i_pAE` (divided by 31 A), `pLDDT`, `pTM`, target and unbound-binder pLDDT. |
| **Ranked results** | `3_Ranked/!_Ranked.csv` | Every accepted design, best first by `i_pDAE`. |
| **Rejected candidates** | `2_Refolded/!_Refolded.csv` | Every scored candidate with its outcome and `failed_filters` - the first place to look when nothing is accepted. |
| **Attempts** | `1_Trajectories/!_Trajectories.csv` | One row per design attempt with the stage it terminated at. |
| **Provenance / summary** | `campaign_metadata.json`, `summary.csv` | Resolved settings, source revision, checkpoint hashes; aggregated counts and metric summaries. |

The notebook's export step collects the FASTA, the CSVs, the metadata and the accepted `.cif`
files into `results/exports/<campaign>/` plus a zip. Column meanings in full:
[`docs/source/outputs.md`](docs/source/outputs.md).

**Zero accepted designs is an ordinary outcome** for a short campaign, not a software failure.
Read `2_Refolded/!_Refolded.csv` for the filters that rejected candidates before changing
anything.

## Validation status

| Item | Status |
| --- | --- |
| Upstream source, presets, scaffolds, notebook, bundled checkpoints | **Complete**, byte-identical to upstream commit `8c1a858` (`v1.0.4-4-g8c1a858`) |
| SLURM launcher, smoke config, preflight, validator, release tooling | **Implemented**; `bash -n` clean, 59 assertions against synthetic fixtures including negative cases |
| Jupyter notebook and its helper module | **Implemented**; `nbformat`-valid, all 10 code cells and every widget callback exercised locally, 90 assertions on both the pandas and no-pandas paths |
| HPC directories, write permission, AlphaFold parameter path | **Not verified** - no HPC access from the development machine |
| Real `sbatch` / `squeue` / `sacct`, H200 allocation, JAX GPU backend | **Not tested** |
| A design campaign, accepted designs, any binder | **Not run.** No sequence, structure or score in this repository was produced by this deployment |
| `py3Dmol` rendering inside institutional Jupyter | **Not tested** (needs internet access for 3Dmol.js) |

Nothing here is experimental evidence. A computationally accepted design means the configured
filters passed; binding, affinity, specificity, stability and function all require laboratory
validation.

## Licensing and attribution

BindCraft2 is **source-available with hosting restrictions**, not MIT.
Copyright (c) 2026 Martin Pacesa, University of Zurich, and the BindCraft2 contributors.
[`LICENSE`](LICENSE) and [`README_licensing_section.md`](README_licensing_section.md) are
preserved verbatim, and the upstream Git history and release tags are intact in this repository.

This copy is **modified**: the Xu Lab additions are `lab/`, `.gitattributes`,
`README_XULAB.md` and this `README.md`, and the upstream `README.md` was renamed to
`README_UPSTREAM.md` with its content unchanged. No upstream source file was edited or deleted;
[`lab/CHANGELOG_XULAB.md`](lab/CHANGELOG_XULAB.md) is the required change notice.

Third-party terms also apply to the AlphaFold 2 model parameters, ColabDesign and
ProteinMPNN / HyperMPNN. This is an internal lab deployment for Xu Lab members; it is not a
hosted service, and the software's functionality is not offered to third parties.
