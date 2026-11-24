# Xu Lab change log

Upstream baseline: `https://github.com/PacesaLab/BindCraft2.git`, branch `main`,
commit `8c1a858fdc1bfc94c60846ff11a0d01c9aa44136` (`v1.0.4-4-g8c1a858`,
`pyproject.toml` version 1.0.4), fetched 2026-11-20.

The upstream license requires that modifications be marked. This file is that
record.

## 2026-11-24 (Phase 4) - CD19 scFv discovery campaign, Top 10 report, individual 3D viewer

A second campaign on the deployment the PD-L1 smoke test proved, plus the two
PI-facing readouts it needs. Nothing in the execution path changed:
`lab/slurm/run_bindcraft_h200.slurm` is byte-identical, the H200 and `sbatch`
settings are untouched, no dependency was added or reinstalled, no AlphaFold or
ProteinMPNN model or weight was touched, and no upstream file was edited. The
PD-L1 campaign in `results/pdl1_h200_smoke` and its backup are not read, written
or referenced by any of this except to be left alone.

The campaign has **not** been submitted: there is no HPC access from the
development machine, so there is no job ID and no result to report. The exact
deployment and submission commands are `README_XULAB.md` §8.3.

### Added

| File | What it does |
| --- | --- |
| `lab/targets/CD19_ECD.pdb` | The target: human CD19 extracellular domain from RCSB **6AL5** (X-ray, 3.0 Å), chain **A**, residues **21-277**, 227 residues, 1656 atoms. B43 Fab chains, N-glycans, waters and the C-terminal GSHHHHHH expression tag removed; deposited author numbering (= UniProt P15391) kept unchanged; the three segments the depositors could not place (40-47, 138-152, 177-183) left absent rather than modelled. Provenance is written into the file as `REMARK 999` lines. Committed because the compute nodes have no internet access. |
| `lab/targets/CD19_ECD.md` | The target's record: source entry, construct differences (N138Q lies inside the unplaced 138-152 segment, so no mutated residue is present), the B43 epitope computed from 6AL5, the coltuximab epitope computed from 7JIC, their consensus, the four hotspots chosen from it, and the membrane / CD81 / glycan accessibility checks with the distances they were decided on. |
| `lab/scripts/prepare_cd19_target.py` | Builds that file from a downloaded RCSB entry: chain and residue-range selection, tag and heteroatom removal, provenance header, and a `--report` mode that prints the antibody epitope contained in the source entry. Standard library only; `--fetch` downloads the entry where there is internet access. |
| `lab/configs/cd19_scfv_discovery.json` | The campaign: `modality: scFv`, that target with hotspots `A162,A219,A220,A222`, `number_of_final_designs: 10`, `max_trajectories: 500`, `workers_per_gpu: 1`, its own `project_folder`, `resume: true`. Nothing else is set, so every loss, filter, CDR span and acceptance threshold is the shipped default. In particular `min_iptm_final` is left at the core default 0.7 and **not** relaxed to the 0.6 that upstream's own `examples/pdl1_scfv.json` uses. |
| `lab/scripts/test_scfv_annotation.py` | 127 assertions over the new helper functions: the preset's paratope spans and flag meanings, the scaffold reference, annotation of the scaffold against itself, annotation of a synthetic design with edited loops and a changed HCDR3 length, the refusals (truncated domain, missing W-G-x-G motif, non-antibody sequence, mismatched paratope label), the paratope split, and four fixture campaigns end to end including the export. Standard library plus the helper; no SLURM, GPU, pandas or py3Dmol needed. |
| `lab/scripts/test_notebook_cells.py` | Executes notebook cells 7, 8 and 9 out of the `.ipynb` in a plain Python process and then invokes every widget callback (show top candidates, refresh, show structure, Next, Previous, export, Compare View) against four fixture campaigns: scFv with a design whose structure file is missing, de novo binder, nothing accepted, and no campaign loaded. |

### Changed

| File | Change |
| --- | --- |
| `lab/notebooks/xulab_hpc.py` | One new section added (`4b`), existing functions left alone except `export_campaign`. New: `cif_chain_residues`, `paratope_spans`, `designed_positions`, `merged_loops`, `scaffold_reference`, `domain_anchors`, `annotate_domain`, `kabat_hcdr3`, `assign_domains`, `split_paratope`, `is_antibody_campaign`, `top_designs`, `domain_fasta`, `cdr_table`, `cdr_table_view`, `accepted_designs`, `structure_domains`, `show_structure`. Loop spans are read from the shipped modality preset's `mutate_positions` in the scaffold's own numbering and located in each design by exact framework anchors; a domain is called VH or VL only when its framework matches that scaffold chain; the quotable HCDR3 is the Kabat one, measured between the conserved Cys stem and the W-G-x-G motif found in the design itself. Anything that cannot be established that way is returned as unavailable with a reason, never guessed. `export_campaign` gains a `<campaign>_VH_VL.fasta`, a `<campaign>_CDRs.csv` and a `<campaign>_CDR_numbering.txt` **for antibody-format campaigns only**, so a de novo binder export is unchanged. `show_structures` is untouched. |
| `lab/notebooks/xulab_bindcraft_h200.ipynb` | Cells 7, 8 and 9 only; cells 1-6 and 10 are unchanged. Cell 7 becomes the Top 10 accepted-candidate report (ranked order straight from `3_Ranked/!_Ranked.csv`, VH/VL sequences, annotated loops with their convention, Kabat HCDR3, confidence and interface readings, contacted target residues, CDR-versus-framework contacts, and the rejected candidates with their `failed_filters` underneath). Cell 8 defaults to **Individual View**: a dropdown built from the accepted designs that actually have a complex file, labelled by rank and ID, one complex at a time, with Previous/Next, `colour by chain` / `colour by pLDDT` kept, an optional HCDR3 or epitope highlight that is skipped with a reason when it cannot be established, and the rank, ID, filename, path and metrics printed beside the viewer; Compare View is retained with its limit raised from 8 to 10. Cell 9's text notes the extra antibody exports. |
| `README_XULAB.md` | New §8, the CD19 campaign: target, settings, exact deployment and submission commands, what the output is and is not. The old §8 becomes §9. New status rows, all of which say NOT SUBMITTED for the job itself. |
| `lab/README.md` | Rows for the new files and their status. |

### Deliberately not done

* No linker between VH and VL. The upstream `scFv` preset models them as two
  chains without one; inventing a linker would be inventing a construct.
* `min_iptm_final` not relaxed. Ten accepted designs is a goal, not a target to
  be reached by moving the bar; the report says plainly when fewer passed and
  never promotes a rejected candidate.
* No change to the CDR spans, the filter set, the losses, the models, the
  launcher, the GPU settings or the environment.
* No overlay in Compare View. The limit was raised to 10, which is what was
  asked for; a structural overlay would have been a fragile addition.

## 2026-11-23 (Phase 3b) - the launcher runs on the proven sbatch path, no srun

Job 1015050 settled the open question from Phase 3a. On `cmri-gpu05`, submitted
with `sbatch`, the job COMPLETED with exit `0:0`: JAX 0.11.2 detected the NVIDIA
H200 NVL with `backend=gpu` and a real GPU matrix computation passed. `srun`, by
contrast, does not work in this CMRI setup at all - it fails with SLURM GRES
errors - so the Phase 3a hypothesis that the GRES might be bound to the step is
not merely unneeded here, it is unusable. The launcher now takes the one path
that is proven, and `srun` is gone from the execution path entirely.

### Added

| File | What it does |
| --- | --- |
| `lab/configs/pdl1_campaign.json` | The real PD-L1 campaign, as opposed to the smoke test: upstream's own `examples/pdl1.json` request (`modality: binder`, `target: hPDL1`, 10 final designs) with the Xu Lab output path, `workers_per_gpu: 1` and `resume: true`, so a resubmission after the 08:00:00 wall clock continues the same `project_folder`. Every key was checked against `CAMPAIGN_SETTING_NAMES` in `bindcraft/settings.py` and the target against `settings/target/hPDL1.json`. Not yet executed. |

### Changed

| File | Change |
| --- | --- |
| `lab/slurm/run_bindcraft_h200.slurm` | (1) BindCraft2 is executed directly by the batch script: `"${XULAB_PYTHON}" -u bindcraft.py design "${CONFIG}" "$@"`, with no `srun` wrapper and no run-mode branch; (2) the import-origin + JAX-GPU gate runs in the batch step only - the `srun`-first attempt, the `XULAB_SRUN` / `XULAB_SRUN_FLAGS` knobs and the `GPU_RUN_MODE` plumbing are removed; (3) the diagnostics no longer repeat themselves as an `srun` step for comparison; (4) the proven allocation is pinned as directives: `--partition=gpu --nodelist=cmri-gpu05 --gres=gpu:h200:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=08:00:00`, and a command-line `--nodelist` still overrides it; (5) the gate's failure message cites job 1015050 and says explicitly not to retry under `srun`. The hard refusal to design on CPU, the stub-`libcuda` strip, the `CUDA_VISIBLE_DEVICES` reconciliation, the `JAX_PLATFORMS` repair, the AlphaFold-parameter search and every output path are unchanged. |
| `lab/notebooks/xulab_hpc.py` | `submit_campaign()` takes an optional `nodelist`, emitted as `--nodelist=<node>` before the script path, so the notebook can move a campaign off the pinned node without editing the launcher. Everything else about submission is unchanged: still `sbatch --parsable --export=ALL[,BINDCRAFT_AF2_PARAMS=...]`, still no `srun`, still no design work in the kernel. |
| `lab/notebooks/xulab_bindcraft_h200.ipynb` | a *node (optional)* field in the submission widget, wired to that parameter; section 4 states the verified configuration and that nothing runs under `srun`; section 5 drops the retired "GRES bound to the srun step" cause from the gate-failure list. All ten code cells, the widget layout and every output path are otherwise untouched. |
| `lab/scripts/gpu_probe.py` | the fourth ordered cause for `CUDA_ERROR_NO_DEVICE` no longer suggests `srun`; it now names an allocation carrying no GPU and gives the verified `--gres`/`--nodelist` resubmission. |
| `README_XULAB.md` | the launcher is recorded as VERIFIED on the HPC with job 1015050; Level A - compute smoke is **PASS** with its evidence; `sbatch` is recorded as working and `srun` as not working in this setup; the troubleshooting tables drop the batch-step-GRES row and the "run the gate under `srun`" advice. |

### Not changed

No dependency was installed or upgraded, no completed environment test was
repeated, no CUDA or JAX version changed, and no AI algorithm, scoring function
or upstream file was touched. The JAX GPU safety gate still refuses to design on
CPU. Source stays at `/home/ypan/BindCraft2/BindCraft2_deploy`, outputs under
`/project/yli4/ypan/bindcraft/`.

### Local tests

73 assertions, all passing: `bash -n` on all four shell scripts and the
launcher; a parser pass over every non-comment launcher line proving no `srun`
invocation survives anywhere, including the diagnostics and the gate, and that
`XULAB_SRUN` and `GPU_RUN_MODE` are gone; every `#SBATCH` directive compared
against the proven configuration; the campaign line asserted to be the direct
interpreter call and not inside a mode branch; the safety gate asserted present,
hard-exiting, checking the import origin and requiring a GPU platform, with the
embedded gate script parsed as Python to confirm the shell quoting is intact;
the preserved repairs, the no-install/no-download property, the three HPC paths
and LF-only line endings; the notebook re-validated as `nbformat` 4 with all ten
code cells compiling and the node field wired through; and `submit_campaign()`
dry-run commands checked end to end - `sbatch --parsable` first, no `--nodelist`
by default, `--nodelist` and `--account` passed through before the script path,
the settings file last, and no `srun` in the command. Both settings files were
validated key by key against `CAMPAIGN_SETTING_NAMES`, their target preset
confirmed to exist, their `project_folder` confirmed to sit under the project
results root, and `workers_per_gpu` confirmed to be 1.

Still not verified: a full design campaign (Level B). Level A is PASS on the
evidence of job 1015050.

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
