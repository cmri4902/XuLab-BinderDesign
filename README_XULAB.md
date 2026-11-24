# Xu Lab BindCraft2 deployment (Phase 1)

> **Modification notice (required by the upstream license).** This repository
> contains a copy of BindCraft2, Copyright (c) 2026 Martin Pacesa, University of
> Zurich, and the BindCraft2 contributors, distributed under the BindCraft2
> Source-Available License (Hosting-Restricted); see [`LICENSE`](LICENSE),
> preserved verbatim. The copy is taken from upstream commit `8c1a858`.
> **No upstream file has been edited or deleted.** The Xu Lab changes are
> additions - `lab/`, `.gitattributes`, this file and the root `README.md` - plus
> one rename: the upstream `README.md` is preserved unchanged as
> [`README_UPSTREAM.md`](README_UPSTREAM.md) so the repository homepage can carry
> the lab's own README. Every change is itemised in
> [`lab/CHANGELOG_XULAB.md`](lab/CHANGELOG_XULAB.md). The software
> is not offered to third parties as a hosted service, and the name
> "BindCraft2" is not used to market any service or derivative.

Internal Xu Lab source copy of [PacesaLab/BindCraft2](https://github.com/PacesaLab/BindCraft2),
prepared for H200 execution through SLURM. Upstream `README_UPSTREAM.md`,
`docs/source/*` and `LICENSE` remain the authority on the software itself; this
file documents only what the Xu Lab added and how to run it here.

**Current state: Phase 1 artefacts are PREPARED. Nothing has been executed on
the HPC from this copy. No campaign has been run, no GPU job submitted, and no
binder has been produced.** See [Status](#status) for the per-item breakdown.

---

## 1. Provenance

| Field | Value |
| --- | --- |
| Upstream remote | `https://github.com/PacesaLab/BindCraft2.git` |
| Branch | `main` (tracking `origin/main`) |
| Commit SHA | `8c1a858fdc1bfc94c60846ff11a0d01c9aa44136` |
| `git describe --tags` | `v1.0.4-4-g8c1a858` |
| `pyproject.toml` version | 1.0.4 |
| Tracked upstream files | 186, contents unmodified (`README.md` renamed to `README_UPSTREAM.md`, same blob) |
| Fetched | 2026-11-20 |

Full component inventory: [`lab/SOURCE_INVENTORY.md`](lab/SOURCE_INVENTORY.md).
Every Xu Lab change, with its reason: [`lab/CHANGELOG_XULAB.md`](lab/CHANGELOG_XULAB.md).

### What is copied from upstream

The complete tracked repository with its Git history and tags: the `bindcraft/`
package (AlphaFold2/ColabDesign-derived code, ProteinMPNN, losses, filters,
campaign engine), the 12 bundled ProteinMPNN/HyperMPNN checkpoints
(`bindcraft/weights/proteinmpnn/weights_{neutral,negative,positive}`, 77 MB),
all presets (`settings/core`, 12 modalities, 9 properties, 6 targets and their
structures), `scaffolds/`, the 27 `examples/*.json` campaigns, `docs/`,
`containers/`, `.github/`, `bindcraft.py`, `bindcraft.ipynb`, `bindcraft.slurm`,
`install.sh`, `pyproject.toml`, `LICENSE` and `README_licensing_section.md`.

### What the Xu Lab changed

No upstream file was edited or deleted. One upstream file was renamed with its
content untouched (`README.md` -> `README_UPSTREAM.md`, identical Git blob), so
that the repository homepage shows the lab's README. Everything else is an
addition:

```
.gitattributes                           LF enforcement for every Linux-executed file
lab/slurm/run_bindcraft_h200.slurm       H200 batch launcher
lab/configs/pdl1_smoke.json              PD-L1 smoke campaign
lab/scripts/hpc_preflight.sh             login-node checks (dirs, imports, weights, CRLF)
lab/scripts/make_release.sh              reproducible transfer archive + BUILD_REVISION.txt
lab/scripts/validate_outputs.py          read-only output validator and report
lab/notebooks/xulab_bindcraft_h200.ipynb Jupyter interface for the HPC (Phase 2)
lab/notebooks/xulab_hpc.py               the notebook's logic, importable and testable
lab/README.md  lab/SOURCE_INVENTORY.md  lab/CHANGELOG_XULAB.md  lab/notebooks/README.md
README_XULAB.md                          this file
README.md                                Xu Lab platform README (homepage)
README_UPSTREAM.md                       the upstream README, renamed, content unchanged
```

---

## 2. Directory layout

### Windows development tree

```text
C:\XuLab\BindCraft2\            <- the Git clone itself, no nested second clone
├── .git\                       full upstream history and tags
├── .github\  bindcraft\  containers\  docs\  examples\  scaffolds\  settings\
├── .gitattributes              LF for every Linux-executed file (Xu Lab addition)
├── lab\
│   ├── slurm\run_bindcraft_h200.slurm
│   ├── configs\pdl1_smoke.json
│   ├── scripts\hpc_preflight.sh
│   ├── scripts\make_release.sh
│   ├── scripts\validate_outputs.py
│   ├── notebooks\xulab_bindcraft_h200.ipynb   the Jupyter interface
│   ├── notebooks\xulab_hpc.py                 its logic, as a module
│   ├── README.md  SOURCE_INVENTORY.md  CHANGELOG_XULAB.md
│   └── BUILD_REVISION.txt      (written by make_release.sh, untracked)
├── bindcraft.py  bindcraft.ipynb  bindcraft.slurm  install.sh
├── pyproject.toml  LICENSE  README.md  README_licensing_section.md
├── README_XULAB.md
└── XuLab_BindCraft2_Deployment_Spec.md   (lab document, untracked)
```

### HPC tree (to be created; see §3)

```text
/home/ypan/BindCraft2/BindCraft2_deploy/      the source copy and the notebook (HOME storage)
├── bindcraft/               upstream package, replaceable at any time
├── lab/                     Xu Lab launcher, configs, scripts, notebooks
└── ...                      the rest of the upstream tree

/project/yli4/ypan/bindcraft/        campaign data (project storage)
├── results/                 campaign output, NEVER deleted by a source update
│   ├── pdl1_h200_smoke/
│   ├── exports/             notebook exports (FASTA, CSV, CIF, zip)
│   └── .xulab_jobs.json     submitted job registry
├── logs/                    SLURM stdout/stderr (must exist before sbatch)
├── cache/                   jax/ compilation cache, bindcraft/ weight cache
└── tmp/                     TMPDIR for the job
```

Source and data are in different trees: replacing the deployment directory
cannot touch `results/`. Only the source copy (about 100 MB including the
bundled ProteinMPNN checkpoints) lives in HOME; campaign output, logs, JAX
caches and temporary files all go to project storage, so the HOME quota is not
consumed by campaigns, and no second copy of the 5.3 GB AlphaFold parameters is
ever made.

### Reused, not recreated

| Resource | Location | Policy |
| --- | --- | --- |
| Conda environment | `/home/ypan/.conda/envs/bindcraft2` (Python 3.12.16, JAX/jaxlib 0.11.2, CUDA 12 wheels) | Used as it is. `install.sh` is **not** run; no `.venv` is built in the source tree. |
| AlphaFold parameters (~5.3 GB) | Already on the cluster. Typically `~/.cache/bindcraft/alphafold`; confirm the real path once. | Pointed at with `BINDCRAFT_AF2_PARAMS` (**a directory holding `params_*.npz`, never a single file**). Never re-downloaded by a batch job. |
| ProteinMPNN / HyperMPNN weights | `bindcraft/weights/proteinmpnn/` inside this repository | Shipped with the source; nothing to fetch. `bindcraft` selects the variant with `mpnn_variant`. |
| JAX compilation cache | `/project/yli4/ypan/bindcraft/cache/jax` | `JAX_COMPILATION_CACHE_DIR`, so repeat campaigns do not recompile. |

Resolution order upstream implements: `BINDCRAFT_AF2_PARAMS` >
`BINDCRAFT_WEIGHTS` > `XDG_CACHE_HOME` > `~/.cache/bindcraft`
(`bindcraft/model_weights.py`).

### Line endings (Windows development, Linux execution)

A Windows clone with `core.autocrlf=true` writes CRLF into the working tree.
Shipping that tree to the HPC gives every shell script a `#!/bin/bash\r`
shebang, and bash then fails with `bad interpreter: /bin/bash^M` - including for
the **upstream** `install.sh`, `bindcraft.slurm` and `containers/build.sh`.

Three measures, none of which edits an upstream file:

1. **`.gitattributes` at the repository root (Xu Lab addition).** `* text=auto
   eol=lf` plus explicit `eol=lf` for `*.sh`, `*.slurm`, `*.py`, `*.json`,
   `*.md`, `*.ipynb`, `*.cif`, `*.pdb`, `*.fasta`, `Dockerfile`, `LICENSE`;
   `*.npz`, `*.png` and the archive formats are marked `binary`;
   `docs/make.bat` is marked `-text` because upstream committed it with CRLF and
   it must stay byte-identical. Git blobs are LF either way, so this changes no
   upstream content - only what is written to disk and what `git archive`
   exports.
2. **Repository config:** `core.autocrlf=false`, `core.eol=lf` (set locally,
   already applied in `C:\XuLab\BindCraft2`). After setting them the working
   tree was re-checked out from the index, and `git diff HEAD` is empty, proving
   upstream content is untouched.
3. **Two checks:** `lab/scripts/make_release.sh` refuses to build an archive
   whose `*.sh`/`*.slurm` files carry CR, and `lab/scripts/hpc_preflight.sh`
   treats CRLF in a Linux-executed script as a blocking failure on the HPC side.

Verify at any time:

```bash
git config core.autocrlf                                   # false
git ls-files --eol install.sh bindcraft.slurm lab/slurm/run_bindcraft_h200.slurm
#   expect: i/lf  w/lf   (index LF, working tree LF)
```

---

## 3. Deployment and execution

Commands below are exact except where a value is a placeholder written
`<like-this>`: SSH/transfer details and the real AlphaFold parameter path have
not been confirmed and are not invented here.

### 3.1 Create the project tree (after confirming write permission)

```bash
# on a login node
ls -ld /project/yli4/ypan                       # confirm the parent is writable
mkdir -p /home/ypan/BindCraft2                  # HOME parent for the source copy
mkdir -p /project/yli4/ypan/bindcraft/{results,logs,cache/jax,tmp}
touch /project/yli4/ypan/bindcraft/.write_probe && rm /project/yli4/ypan/bindcraft/.write_probe
df -h /project/yli4/ypan/bindcraft              # confirm free space
```

### 3.2 Stage the source from Windows

The transfer has to carry four things: the complete upstream tree, the Xu Lab
additions (which may not be committed yet), LF line endings, and the Git
revision the copy was built from. `lab/scripts/make_release.sh` does all four
and refuses to produce an archive if any of them is wrong.

`<user>` and `<hpc-host>` below are placeholders; no hostname is assumed here.

#### Option A - release archive (recommended, works with scp alone)

Run this in **Git Bash** (not PowerShell: it is a POSIX script and uses
`/c/...` style paths internally through Git's own shell):

```bash
cd /c/XuLab/BindCraft2
bash lab/scripts/make_release.sh                 # writes into C:\XuLab\bc2_release
# or choose the output directory:
bash lab/scripts/make_release.sh /c/XuLab/transfers
```

The script exports the tracked files from the Git object store (so their content
is exactly commit `8c1a858` with LF endings, not whatever an editor left on
disk), adds `lab/`, `README_XULAB.md` and `.gitattributes` from the working
tree, strips `__pycache__`, converts any stray CRLF, writes
`lab/BUILD_REVISION.txt` (commit, tag, dirty state, SHA-256 of every lab file,
and the list of untracked files at build time), then verifies that the 12
bundled ProteinMPNN checkpoints and every entry point are present. It prints the
archive path, its SHA-256, and the unpacking commands. It also writes a
`*.gitbundle` holding the full history, in case the HPC copy should become a
real Git clone later.

Then, from **PowerShell or Git Bash** (plain `scp`, no brace expansion, no
rsync needed):

```powershell
scp C:\XuLab\bc2_release\bindcraft2_xulab_<short>_<state>_<stamp>.tar.gz <user>@<hpc-host>:/home/ypan/BindCraft2/
```

On the HPC:

```bash
mkdir -p /home/ypan/BindCraft2
cd /home/ypan/BindCraft2
# The archive unpacks to BindCraft2_deploy/, the deployment directory name, so no
# rename is needed. tar merges into an existing directory, so move the old copy
# aside for a clean replacement. Nothing under /project/yli4/ypan/bindcraft/ is
# touched: results, logs, caches and tmp live there, not here.
[ -d BindCraft2_deploy ] && mv BindCraft2_deploy BindCraft2_deploy.$(date +%Y%m%d%H%M%S).bak
tar -xzf bindcraft2_xulab_<short>_<state>_<stamp>.tar.gz     # unpacks ./BindCraft2_deploy
cat BindCraft2_deploy/lab/BUILD_REVISION.txt                 # provenance of this copy
```

#### Option B - rsync of the working tree (for incremental re-syncs)

Run in **Git Bash or WSL**, where `rsync` and `/c/...` paths both exist. A
trailing slash on the source is required, and `--delete` is pointed at the
source directory only, never at the project root:

```bash
rsync -av --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' \
  --exclude '.ipynb_checkpoints/' \
  /c/XuLab/BindCraft2/ <user>@<hpc-host>:/home/ypan/BindCraft2/BindCraft2_deploy/
```

rsync copies bytes, so this only ships LF if the Windows working tree is LF.
The repository's `.gitattributes` plus `core.autocrlf=false` keep it that way;
confirm before syncing, and after syncing on the HPC:

```bash
# Windows (Git Bash), in the repository
git config core.autocrlf        # expect: false
git ls-files --eol install.sh lab/slurm/run_bindcraft_h200.slurm   # expect w/lf

# HPC, after the transfer
cd /home/ypan/BindCraft2/BindCraft2_deploy
grep -rlU $'\r' --include='*.sh' --include='*.slurm' . || echo "no CRLF, good"
```

If a script ever does arrive with CRLF, bash fails with
`bad interpreter: /bin/bash^M`. Repair it in place with
`sed -i 's/\r$//' <file>`; `lab/scripts/hpc_preflight.sh` also checks this and
treats CRLF as a blocking failure.

#### Option C - Git remote (best long-term)

Push this repository, including a committed `lab/`, to a lab-internal Git
remote and `git clone` it on the HPC. The HPC copy then carries
`.git`, so `git rev-parse HEAD` and `git status` answer the provenance question
directly and `.gitattributes` guarantees LF on checkout. The lab files are not
committed yet, so until then Option A is the reproducible route.

### 3.3 Preflight on the login node

```bash
cd /home/ypan/BindCraft2/BindCraft2_deploy
bash lab/scripts/hpc_preflight.sh
```

Blocking checks (any one of them failing makes the script exit nonzero, so it
cannot be mistaken for a green light):

| # | Blocking check |
| --- | --- |
| 1 | all five project directories exist **and** are writable, plus a real write probe |
| 2 | the conda interpreter exists and runs (exit 2 and stop if not) |
| 3 | `bindcraft` imports from this tree, not from the older installation |
| 4 | every required module and bundled ProteinMPNN checkpoint is present |
| 5 | the AlphaFold parameters are located **and** all seven required models exist |
| 6 | the Linux-executed scripts have LF line endings |

Warnings (printed, counted, but never blocking): `df` unreadable, `nvidia-smi`
absent on the login node, `sbatch` not on this shell's PATH, a listed script not
present.

Exit status: **0** every blocking check passed (`RESULT: READY`), **1** at least
one blocking failure (`RESULT: BLOCKED`), **2** the checks could not run at all.
It installs and downloads nothing and does not test the GPU, because a login
node is not the H200 allocation. Export the parameter directory it reports:

```bash
export BINDCRAFT_AF2_PARAMS=<absolute dir holding params_*.npz>
```

### 3.4 Submit the smoke campaign

```bash
mkdir -p /project/yli4/ypan/bindcraft/logs          # SLURM opens --output before the job runs
cd /home/ypan/BindCraft2/BindCraft2_deploy
sbatch --export=ALL,BINDCRAFT_AF2_PARAMS="$BINDCRAFT_AF2_PARAMS" \
  lab/slurm/run_bindcraft_h200.slurm lab/configs/pdl1_smoke.json
# add --account=<account> if this site charges work to one
```

Declared resources: partition `gpu`, `--gres=gpu:h200:1`, 8 CPUs, 64 GB **host**
RAM (not GPU VRAM), 08:00:00. The script `cd`s into the lab copy, exports
`BINDCRAFT_HOME` and `PYTHONPATH` to it, sets `JAX_COMPILATION_CACHE_DIR` and
`TMPDIR` into the project tree, prints `nvidia-smi`, then runs a short gate
process that **exits nonzero** if `bindcraft` was imported from outside the lab
copy or if JAX sees no GPU. It never falls back to CPU, never downloads weights
and never installs anything. The campaign's exit status is propagated as the
job's exit status.

### 3.5 Watch the job

```bash
squeue -u "$USER"
tail -f /project/yli4/ypan/bindcraft/logs/xulab_bc2_<jobid>.out
grep -n 'xulab-bc2:' /project/yli4/ypan/bindcraft/logs/xulab_bc2_<jobid>.out
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,ReqTRES%40
# per-worker detail, once the campaign has started:
ls /project/yli4/ypan/bindcraft/results/pdl1_h200_smoke/workers/
```

Expected early lines: the host and job id, the `nvidia-smi` table naming an
H200, `xulab-bc2: bindcraft /home/ypan/BindCraft2/BindCraft2_deploy/bindcraft/__init__.py`,
and `xulab-bc2: jax 0.11.2 backend=gpu`.

### 3.6 Validate the output

```bash
python lab/scripts/validate_outputs.py \
  /project/yli4/ypan/bindcraft/results/pdl1_h200_smoke \
  --slurm-log '/project/yli4/ypan/bindcraft/logs/xulab_bc2_*.out' \
  --output /project/yli4/ypan/bindcraft/logs/pdl1_h200_smoke_validation.md
```

Read-only, standard library only, so it also runs on Windows after a transfer.
It reports per-file PASS / FAIL / BLOCKED / NOT TESTED with evidence paths, and
two independent verdicts:

| Level | Granted when |
| --- | --- |
| **A - compute smoke** | An explicit **JAX GPU backend** line in a supplied log (`backend=gpu`, a `jax device gpu:` line, an XLA GPU client line) **and** at least one trajectory or candidate record **and** campaign exit status 0. No log ⇒ `NOT TESTED`. CPU-fallback line, nonzero exit, or job-termination marker ⇒ FAIL. |
| **B - end-to-end binder output** | **Every** accepted design has a `Binder_Sequence` **and** a complex structure file of its own. Zero accepted ⇒ `NOT ACHIEVED`, which is not a FAIL. Accepted rows with a missing structure are named individually and are a FAIL. |

`CUDA_VISIBLE_DEVICES`, an `nvidia-smi` card listing and the driver-level
"N GPUs visible" note prove only that a card was *allocated*. They are reported
as **supporting evidence** and can never grant Level A on their own.

Structure checking is reported at two clearly separated levels:

| Check | What it proves |
| --- | --- |
| lightweight text check | the file is nonempty, has a `data_` block, an `_atom_site` loop and ATOM rows; chain labels are read by whitespace splitting. No chemistry, geometry or occupancy validation. |
| full structure parse | `biotite` (an upstream dependency) or `gemmi` accepted the file and produced atoms and chains. Reported `NOT TESTED` when neither is importable, e.g. under a bare Windows Python - run it under the `bindcraft2` environment for a real parse, or pass `--no-full-parse` to skip it. |

Exit status: 0 nothing failed, 1 at least one FAIL, 2 folder unreadable.

### 3.7 Retrieve results

PowerShell has no brace expansion, so each file is named separately (one
`scp` per line; `New-Item -Force` on a directory is safe, it does not truncate):

```powershell
New-Item -ItemType Directory -Force C:\XuLab\results\pdl1_h200_smoke | Out-Null
scp -r "<user>@<hpc-host>:/project/yli4/ypan/bindcraft/results/pdl1_h200_smoke/3_Ranked" C:\XuLab\results\pdl1_h200_smoke\
scp "<user>@<hpc-host>:/project/yli4/ypan/bindcraft/results/pdl1_h200_smoke/summary.csv" C:\XuLab\results\pdl1_h200_smoke\
scp "<user>@<hpc-host>:/project/yli4/ypan/bindcraft/results/pdl1_h200_smoke/campaign_metadata.json" C:\XuLab\results\pdl1_h200_smoke\
scp "<user>@<hpc-host>:/project/yli4/ypan/bindcraft/logs/xulab_bc2_<jobid>.out" C:\XuLab\results\pdl1_h200_smoke\
```

In Git Bash the brace form works and the whole set is one command:

```bash
mkdir -p /c/XuLab/results/pdl1_h200_smoke
scp -r "<user>@<hpc-host>:/project/yli4/ypan/bindcraft/results/pdl1_h200_smoke/{3_Ranked,summary.csv,campaign_metadata.json}" \
  /c/XuLab/results/pdl1_h200_smoke/
```

Keep `campaign_metadata.json` with any results you move: it holds the resolved
settings, source revision and checkpoint hashes. Keep the whole folder
(including `.campaign_state.json`) if the campaign should ever be resumed.
Retrieve results into `C:\XuLab\results\`, outside the Git clone.

---

## 4. How Python is made to use *this* copy

Three independent mechanisms, because the existing conda environment may hold an
**editable** install pointing at the older source directory:

1. the job `cd`s into `/home/ypan/BindCraft2/BindCraft2_deploy`, and
   `bindcraft.py` inserts its own directory at the front of `sys.path`;
2. `PYTHONPATH` is prefixed with the lab tree and `BINDCRAFT_HOME` names it;
3. the gate process asserts it, comparing the resolved `bindcraft.__file__`
   against `BINDCRAFT_HOME` and exiting nonzero on a mismatch.

Verify by hand at any time (the printed path, not the version string, is the
evidence):

```bash
cd /home/ypan/BindCraft2/BindCraft2_deploy
BINDCRAFT_HOME=$PWD PYTHONPATH=$PWD /home/ypan/.conda/envs/bindcraft2/bin/python -s \
  -c 'import bindcraft, pathlib; print(pathlib.Path(bindcraft.__file__).resolve())'
# must print /home/ypan/BindCraft2/BindCraft2_deploy/bindcraft/__init__.py
```

If it prints a path under the older installation, that installation is winning
the import; `pip install -e /home/ypan/BindCraft2/BindCraft2_deploy` inside the
environment is the permanent fix (it replaces the old editable pointer, and the
`lab/` scripts keep working either way).

---

## 5. Interpreting results and common failures

### Zero accepted designs

`3_Ranked/` absent or empty after 20 trajectories is an **ordinary outcome**, not
a software failure. Read, in order:

1. `2_Refolded/!_Refolded.csv` - `outcome` and `failed_filters` per candidate.
   `not measured` is a missing reading, not a low score.
2. `1_Trajectories/!_Trajectories.csv` - `terminated` names the stage an attempt
   stopped at; blank means gradient design completed, not that a sequence was
   accepted. `autotuned` entries mean the attempt ran against an easier task
   than the campaign asked for.
3. Raise `max_trajectories` for a real campaign. Do **not** enable the Colab
   notebook's `no_filters` shortcut to manufacture acceptances.

An accepted design means the configured computational checks passed. It is
**not** evidence of binding, affinity, specificity, stability or function.

### GPU initialisation: what stopped job 1014429

The first real H200 job failed with `cuInit(0) failed: CUDA_ERROR_NO_DEVICE`
even though `nvidia-smi` listed an H200 NVL, `CUDA_VISIBLE_DEVICES=0`, the conda
environment and source imports were correct, the AlphaFold parameters were found
and preflight reported READY. JAX fell back to the CPU and the launcher's gate
stopped the campaign, which is the behaviour it was built for.

That exact split - `nvidia-smi` works, `cuInit` does not - has a short list of
causes, because `nvidia-smi` is a separate binary that only needs
`/dev/nvidiactl`, while CUDA needs a real driver library, the `nvidia-uvm`
device and a device this process was actually granted:

| Cause | Evidence in the log | What the launcher now does |
| --- | --- | --- |
| A **stub `libcuda`** on `LD_LIBRARY_PATH` (the CUDA toolkit's link-time placeholder, often inherited from a conda prefix). It loads, then answers `cuInit` with `CUDA_ERROR_NO_DEVICE`. | `STUB .../stubs/libcuda.so.1` from the diagnostics, or `stub libcuda ahead of the real driver` from the probe | strips those directories from `LD_LIBRARY_PATH` before anything imports JAX (`XULAB_KEEP_LD_STUBS=1` disables this), and `hpc_preflight.sh` now blocks on it from the login node |
| **`/dev/nvidia-uvm` missing** from the step's device cgroup | `MISSING /dev/nvidia-uvm` | reported explicitly; it needs a site fix (the module must be loaded on the node), and the diagnostics say so rather than letting the campaign fail obscurely |
| **`CUDA_VISIBLE_DEVICES` naming a device this step was not given**, typically carried in by `--export=ALL` from the submitting shell (for example a Jupyter node's T4) | `CUDA_VISIBLE_DEVICES=... did not match the allocation` | reconciles it against `SLURM_STEP_GPUS` / `SLURM_JOB_GPUS`, with SLURM's view winning |
| **An allocation that carries no GPU at all** | `SLURM_JOB_GPUS=unset` with no `/dev/nvidia*` nodes | the launcher pins the verified `--gres=gpu:h200:1 --nodelist=cmri-gpu05` configuration and names it again in the failure message. It does **not** retry under `srun`: `srun` fails on this cluster with SLURM GRES errors, and job 1015050 proved the batch step alone reaches the H200 |
| **`JAX_PLATFORMS=cpu`** forcing CPU execution | `env JAX_PLATFORMS=cpu` | unset for the job, and blocked by preflight |

Every campaign job now prints this report before designing anything:

```bash
# inside the job, automatically; or by hand in an interactive allocation
salloc --partition=gpu --gres=gpu:h200:1 -c 8 --mem=16G -t 00:20:00
bash lab/scripts/gpu_diagnostics.sh      # exit 0 usable, 2 no libcuda, 3 cuInit failed,
                                         #      4 no JAX GPU, 5 no JAX
```

Read the log in this order: `xulab-diag: === slurm allocation ===` (what the
step was given), `=== device nodes and driver module ===` (`MISSING` lines),
`=== cuda driver library resolution ===` (`STUB` lines), then
`xulab-gpu: verdict`. `validate_outputs.py` reads the same lines and refuses
Level A when the driver failed, whatever `nvidia-smi` reported.

Nothing was reinstalled, upgraded or rebuilt to produce this fix: no CUDA, JAX
or dependency version changed, and no AI algorithm was touched.

### Common execution failures

| Symptom in the log | Cause | Fix |
| --- | --- | --- |
| `ERROR: no JAX GPU in this allocation` | No GPU requested, `CUDA_VISIBLE_DEVICES` empty, or the CUDA driver failed to initialise | Read the diagnostics block above it in the log; see [GPU initialisation](#gpu-initialisation-what-stopped-job-1014429). Check `sacct -j <id> --format=ReqTRES%40`. |
| `cuInit(0) failed: CUDA_ERROR_NO_DEVICE` while `nvidia-smi` works | stub `libcuda`, missing `/dev/nvidia-uvm`, or a `CUDA_VISIBLE_DEVICES` the job was not given | the launcher repairs the first and third and names the second. Full order of causes in the section above. |
| `ERROR: imported bindcraft from ...` outside the lab copy | Editable install of the older source wins | See §4. |
| `AlphaFold parameters not found` | `BINDCRAFT_AF2_PARAMS` unset or wrong | Locate the existing copy (`find $HOME /project -maxdepth 6 -name 'params_model_1_multimer_v3.npz'`) and export the **directory**. Do not download. |
| `RESOURCE_EXHAUSTED` / out of memory | VRAM or host RAM too small for the target | Keep `workers_per_gpu: 1`; raise `--mem` above 64G; shorten `binder_lengths`. |
| `DUE TO TIME LIMIT` | 08:00:00 too short | Resubmit; a campaign resumes into the same `project_folder` by default. |
| `nvidia-smi unavailable` but JAX reports a GPU | Allocation without the CLI tool | Not fatal; the JAX device line is the evidence that matters. |
| A T4 appears instead of an H200 | The Jupyter node (`cmri-gpu01`) is not the batch allocation | Judge the GPU only from the batch job's own log. |

Every campaign recompiles AlphaFold graphs on its first run; with
`JAX_COMPILATION_CACHE_DIR` in the project tree, later campaigns on the same
card reuse them.

### Protecting completed campaigns

Results live in `/project/yli4/ypan/bindcraft/results/` on project storage,
while the source copy lives in `/home/ypan/BindCraft2/BindCraft2_deploy`, so
updating or replacing the code cannot delete them. Keep `--delete` in any
`rsync` pointed only at `/home/ypan/BindCraft2/BindCraft2_deploy/`, never at
`/home/ypan/BindCraft2/` or at the project tree. Deleting
a `.cif` from `3_Ranked/` is upstream's documented way to reject a design and
drops its row when the campaign closes - do it deliberately, never as cleanup.

---

## 6. Status

| Phase 1 item | Status |
| --- | --- |
| Complete upstream clone at `C:\XuLab\BindCraft2`, history and tags | **DONE** (186 tracked files, commit `8c1a858`) |
| Upstream commit SHA and source inventory | **DONE** (`lab/SOURCE_INVENTORY.md`) |
| All upstream modalities, presets, scaffolds, notebook, checkpoints preserved | **DONE** (no upstream file edited or deleted; `README.md` renamed to `README_UPSTREAM.md`, same blob) |
| `lab/slurm/run_bindcraft_h200.slurm` | **VERIFIED on the HPC**: job 1015050 COMPLETED (exit `0:0`) on `cmri-gpu05`, JAX 0.11.2 `backend=gpu` on the NVIDIA H200 NVL, GPU matrix computation PASS. Executes BindCraft2 directly in the batch step; no `srun` anywhere in the execution path |
| `lab/configs/pdl1_smoke.json` | **PREPARED**, NOT RUN (keys checked against `bindcraft/settings.py`, not executed) |
| `lab/configs/pdl1_campaign.json` | **PREPARED**, NOT RUN - the real PD-L1 campaign (upstream's `examples/pdl1.json` request, 10 final designs, `resume: true`, Xu Lab output path); keys checked against `bindcraft/settings.py` and the target against `settings/target/hPDL1.json`, not executed |
| `lab/scripts/validate_outputs.py` | **PREPARED**; syntax-checked and exercised on synthetic fixtures outside the repository, including negative fixtures (allocation-only GPU evidence, nonzero exit, missing structure per candidate); never run against a real campaign |
| `lab/scripts/hpc_preflight.sh` | **PREPARED**, NOT TESTED on the HPC; `bash -n` clean and its exit-code logic exercised locally against missing directories, a missing interpreter, missing AlphaFold parameters and a CRLF script |
| `lab/scripts/make_release.sh` | **PREPARED**; `bash -n` clean and run locally in Git Bash to build a verified archive (content, 12 checkpoints, no CRLF). Never transferred to the HPC |
| `.gitattributes` + `core.autocrlf=false` | **DONE and VERIFIED locally**: all Linux-executed scripts, upstream ones included, are LF on disk, and `git diff HEAD` against commit `8c1a858` is empty |
| `README_XULAB.md` + change log | **DONE** |
| HPC deployment plan and commands | **DONE** as documentation; **NOT EXECUTED** |
| HPC directories (`/home/ypan/BindCraft2/BindCraft2_deploy`, `/project/yli4/ypan/bindcraft/{results,logs,cache,tmp}`) | **NOT CREATED** (no HPC access from here) |
| Write permission on `/project/yli4/ypan/` | **NOT VERIFIED** (`lab/scripts/hpc_preflight.sh` verifies it) |
| AlphaFold parameter path on the HPC | **NOT VERIFIED**; searched at run time, never hardcoded, never downloaded |
| Python imports the new source copy | **MECHANISMS PREPARED + CHECKS WRITTEN**; not yet demonstrated on the HPC |
| Level A - compute smoke | **PASS** - job 1015050, COMPLETED exit `0:0`, JAX 0.11.2 `backend=gpu`, H200 NVL detected, GPU matrix computation PASS |
| Level B - end-to-end binder output | **NOT ACHIEVED** |
| Jupyter notebook (`lab/notebooks/xulab_bindcraft_h200.ipynb`) | **IMPLEMENTED**; `nbformat`-valid, every code cell parses, and all of cells 2, 3, 7, 8 and 9's widget callbacks exercised locally against fixtures and a stubbed RCSB; **never run on the HPC** |
| Phase 2 helper module (`lab/notebooks/xulab_hpc.py`) | **IMPLEMENTED**; 90 local assertions pass on both the pandas and no-pandas paths |
| Phase 5 example target (`lab/targets/CD19_ECD.pdb`, `lab/targets/CD19_ECD.md`) | **BUILT and CHECKED locally** from RCSB 6AL5: chain A, residues 21-277, 227 residues, 1656 atoms, provenance in its own `REMARK 999` header; epitope numbers computed from 6AL5 and 7JIC coordinates. **Not yet copied to the HPC** |
| Phase 5 target builder (`lab/scripts/prepare_target.py`) | **IMPLEMENTED**; general purpose, run locally against RCSB 6AL5 (PDB) and 7JIC (mmCIF). Replaces the earlier CD19-specific script and reproduces its output |
| Phase 5 RCSB search / entry lookup / download | **VERIFIED against the live API** on 2026-11-24: `CD19` returned `7URV, 7URX, 6AL5, 7JIC, 6AL4`; `P15391` returned `6AL5, 7JIC, 7URV, 7URX`; the 6AL5 entry report returned the title, X-RAY 3.0 A, three entities with chains and the SIFTS mapping `entity 1-257 -> UniProt P15391 21-277`. Offline behaviour tested with a stub, not against a real outage |
| Phase 5 example settings (`lab/configs/cd19_scfv_discovery.json`) | **PREPARED**, NOT RUN - keys and target-path resolution checked against `bindcraft/settings.py` (`read_settings` resolves `targets[].target_path`; the scFv `binder_scaffold` resolves from the preset against the repository root), not executed |
| Phase 5 scFv annotation, Top 10 report and individual 3D viewer | **IMPLEMENTED**; 128 local assertions in `lab/scripts/test_scfv_annotation.py` and every cell-2/3/7/8/9 widget callback exercised by `lab/scripts/test_notebook_cells.py`. **Never run on real campaign output** |
| Phase 5 target step, generic reporting and chain labelling | **IMPLEMENTED**; 155 local assertions in `lab/scripts/test_target_tools.py` (PDB and mmCIF parsing, declared mutations and unplaced residues, chain-ambiguity refusal, preparation, FASTA and upload handling, RCSB routing and offline behaviour) |
| Phase 5 isolated development deployment | **IMPLEMENTED and TESTED LOCALLY**; 118 assertions in `lab/scripts/test_deployment_isolation.py` prove all thirteen derived paths and a real `dry_run` `sbatch` command name neither production tree. `lab/scripts/setup_dev_deployment.sh` is `bash -n` clean and was dry-run locally, including its refusal of the production paths. **The HPC directories do not exist yet and nothing has been cloned** |
| Phase 5a CD19 campaign validation | **36 static checks PASS, 0 FAIL** against the rules in `bindcraft/settings.py`, `bindcraft/protein_preparation.py` and `settings/modality/scFv.json`: every campaign and target key recognised, the target file single-chain and clean, all four hotspots present in it, the scFv scaffold's two chains and all 23 `mutate_positions` spans in range, and all six CDR loops anchored on both sides. One blocker was found and fixed - the settings file named the production project root |
| Any campaign on the H200 since the PD-L1 smoke test | **NOT SUBMITTED** - no HPC access from the development machine. The exact commands are §9.2 and §9.3; no job ID, no H200 evidence and no CD19 result exists |
| Real `sbatch` behaviour on the HPC | **VERIFIED**: `sbatch` works on `cmri-gpu05`. `srun` does **not** work in this CMRI setup (SLURM GRES errors), so nothing in the execution path uses it |
| `squeue` / `sacct` polling from the notebook kernel | **NOT TESTED** (no SLURM on the development machine) |
| `py3Dmol` rendering inside institutional Jupyter | **NOT TESTED** (needs internet access for 3Dmol.js) |

Level A is PASS on the evidence of job 1015050. A full design campaign (Level B)
has not been run. Every PASS in this project must cite a log path and line.

---

## 7. The Jupyter notebook on CMRI HPC (Phase 2)

`lab/notebooks/xulab_bindcraft_h200.ipynb` is the official `bindcraft.ipynb`
adapted for regular Jupyter here; `bindcraft.ipynb` itself is untouched. All of
its non-widget logic is in `lab/notebooks/xulab_hpc.py`, which imports and runs
without Jupyter, SLURM or a GPU, so it can be tested and reviewed on its own.

```bash
# on the HPC, with the bindcraft2 environment active
cd /home/ypan/BindCraft2/BindCraft2_deploy
jupyter lab                      # or the institutional Jupyter/Workbench launcher
#   open lab/notebooks/xulab_bindcraft_h200.ipynb
```

The ten steps: environment report → target → binder and campaign → submit to
SLURM → job status and log → load results → interface and confidence readings →
3D complexes → export → audit with `validate_outputs.py`. Details and the full
Colab-to-Jupyter mapping: [`lab/notebooks/README.md`](lab/notebooks/README.md).

What it does **not** do: design anything in the kernel (every campaign is a
batch job, so closing the notebook leaves it running), install packages,
download weights, write into the source checkout, offer the `no_filters`
shortcut, or make any claim about the GPU - the H200 is evidenced only by the
job's own log.

Paths come from `XULAB_PROJECT_ROOT`, `XULAB_BC2_HOME`, `XULAB_RESULTS`,
`XULAB_LOGS`, `XULAB_CACHE`, `XULAB_TMP` and `XULAB_PYTHON` where set, so a
second user or a second project root needs no edit to a tracked file. Submitted
jobs are recorded in `results/.xulab_jobs.json`, which is how a later notebook
session finds a running campaign, its log and its results folder again.

If `ipywidgets` or `py3Dmol` is missing from the kernel, the environment cell
says so and the rest still works: `py3Dmol` only affects step 8, and the CIF
files can always be downloaded and opened locally.

## 8. The binder discovery platform (Phase 5)

The deployment is general purpose: any protein target, any binder format
BindCraft2 supports. Human CD19 and the scFv format are the **default example**
the platform was first driven with end to end, not a restriction built into it.

Nothing in `lab/slurm/run_bindcraft_h200.slurm`, in the H200 or `sbatch`
settings, in the environment, in the AlphaFold/ProteinMPNN models or in any
upstream algorithm was changed for any of this.

### 8.1 Naming a target

Notebook step 2 takes a target six ways, and whichever is used the result is the
same: a file under the deployment's **targets root**, plus a settings entry the
notebook writes into the campaign JSON. No JSON is edited by hand.

| Source | Needs | Notes |
| --- | --- | --- |
| search RCSB | a protein name, a UniProt accession or a PDB ID | needs internet from the kernel |
| a PDB ID | e.g. `6AL5` | downloads the entry, then the chain is chosen |
| upload a file | `.pdb`, `.cif`, `.fasta` | stored unchanged under `targets/uploads/` |
| paste a sequence | FASTA or a bare sequence | written as a FASTA target |
| a file on the cluster | an absolute path | used where it is; nothing is copied |
| a shipped preset | one of upstream's own targets | unchanged |

A name goes to RCSB's full-text search; a UniProt accession goes to the
accession attribute, so it returns entries that actually reference that protein
rather than entries whose text happens to contain the string. Both were verified
against the live API: `CD19` → `7URV, 7URX, 6AL5, 7JIC, 6AL4`; `P15391` →
`6AL5, 7JIC, 7URV, 7URX`.

**Chain selection is never done for you.** When an entry holds more than one
polymer chain the form lists each chain with its size and the entity description
RCSB gives for it, and waits. Nothing is pre-selected, because the longest chain
of an antibody complex is usually the antibody. A single-chain file needs no
choice and is pre-selected.

**Reported before anything is committed:** entry title, method, resolution,
release date and citation; per entity the description, author chain, length,
organism and UniProt accession; the residue range present; the ordered segments
and the gaps; the engineered mutations and expression tags the file itself
declares (`SEQADV`, or mmCIF `_struct_ref_seq_dif`); the residues the depositors
declare unplaced (`REMARK 465`, or `_pdbx_unobs_or_zero_occ_residues`); and the
SIFTS mapping from the entity's numbering to its reference sequence. The target
is written with the **deposited author numbering unchanged**, which is the
numbering hotspots have to be given in.

**Hotspots and coldspots are optional.** Left empty, BindCraft2 picks its own
interface on the target.

**On accessibility.** The geometry report measures the site's widest CA-to-CA
span, its distance to each end of the modelled chain, and the nearest
N-glycosylation sequon present. It then states in as many words that whether the
site is reachable on a cell is **not established** by those coordinates -
membrane proximity, partner-protein occlusion and glycan shielding are not
properties of a soluble fragment. No code path in this platform claims
accessibility from coordinates.

**With no network**, the search and PDB-ID routes say so, name the host and give
the alternative. Prepare the file where there is internet access and copy it
across:

```bash
python lab/scripts/prepare_target.py --search "CD19"
python lab/scripts/prepare_target.py --entry 6AL5 --describe
python lab/scripts/prepare_target.py --entry 6AL5 --chain A --first 21 --last 277 \
    --name hCD19_ECD --contacts
python lab/scripts/prepare_target.py --source /path/to/already_downloaded.cif --describe
```

### 8.2 The default example target

| Item | Value |
| --- | --- |
| Structure | RCSB **6AL5**, X-ray 3.0 Å, "COMPLEX BETWEEN CD19 (N138Q MUTANT) AND B43 FAB" |
| CD19 chain | **A**, residues **21-277**, deposited author numbering (= UniProt P15391) unchanged |
| Prepared file | `lab/targets/CD19_ECD.pdb` (227 residues, 1656 atoms; Fab chains, glycans, waters and the C-terminal `GSHHHHHH` tag removed) |
| Built by | `lab/scripts/prepare_target.py` (standard library only) |
| Hotspots | `A162,A219,A220,A222` - each contacted by **both** B43 in 6AL5 and coltuximab in 7JIC |
| Full record | [`lab/targets/CD19_ECD.md`](lab/targets/CD19_ECD.md) |

The prepared `.pdb` is committed because the compute nodes have no internet
access, and its own `REMARK 999` header carries the full provenance including
the SIFTS mapping.

### 8.3 Campaign settings

Notebook step 3 opens on **scFv**, 10 final designs and 500 trajectories, with
resume on. Every other format is one dropdown away; scaffolded formats (VHH,
ARP, scFv, Fab) hide the length boxes because the scaffold decides the domain
lengths. Every acceptance filter, loss and CDR span comes from the shipped
presets and stays enabled.

`lab/configs/cd19_scfv_discovery.json` is the committed example of what step 3
writes. Note for comparison: upstream's own `examples/pdl1_scfv.json` relaxes
`min_iptm_final` from the core default 0.7 to 0.6. This example does **not**
copy that, so its acceptance bar is the stricter core default. If a campaign
finishes its budget with few or no accepted designs, that one threshold - not
the CDR spans, and not the filter set - is the first thing to reconsider, and
changing it is a deliberate, recorded decision.

### 8.4 Reading the results

Step 7 reports up to ten **accepted** designs in the pipeline's own `i_pDAE`
order, with the predicted target-contact residues taken from
`Interface_Target_Residues` as the pipeline recorded them. For an antibody
format each design also gets its VH and VL named from the scaffold framework and
its loops annotated; the quotable HCDR3 length is the Kabat one, measured
between the conserved Cys stem and the `W-G-x-G` motif located in the design
itself. Rejected candidates and their `failed_filters` stay available underneath.

Step 8 defaults to **Individual View**: one complex at a time from a dropdown of
the accepted designs that actually have a file, with Previous/Next, colour by
chain or pLDDT, and an optional HCDR3 or epitope highlight. Chains are labelled
on evidence, in this order: a chain whose sequence matches the campaign's own
target file is `target (<name>)`; a chain carrying the scaffold framework
anchors is `VH` or `VL`; anything else is `binder chain`, or - when the target
file is not readable from the notebook - `target or binder (not determined)`.
Each label carries the evidence it was given on. Compare View is retained, up to
10 complexes.

A highlight that cannot be mapped safely onto the structure's numbering is
skipped with a reason rather than drawn in the wrong place.

### 8.5 What the output is, and is not

The upstream `scFv` preset models **VH and VL as two separate chains with no
designed linker**. An accepted design is therefore not a complete, expressible
single-chain scFv construct, and this project does not supply a linker for it.

`i_pDAE`, `i_pTM`, `i_pAE`, `pLDDT` and `pTM` are prediction confidence
readings. They are not affinity, not specificity, not stability, not
developability and not probabilities of binding. The contacted target residues
are a property of the complex the pipeline predicted, not a mapped epitope.
Acceptance means the configured computational filters passed.

---

## 9. The isolated development deployment

Development happens in a second deployment that shares the cluster's conda
environment and the existing AlphaFold parameter set, and **nothing else**,
with production.

| | Production | Development |
| --- | --- | --- |
| Source tree | `/home/ypan/BindCraft2/BindCraft2_deploy` | `/home/ypan/BindCraft2/BindCraft2_cd19_dev` |
| Project root | `/project/yli4/ypan/bindcraft` | `/project/yli4/ypan/bindcraft_cd19_dev` |
| Results | `<project>/results` | `<project>/results` |
| Logs | `<project>/logs` | `<project>/logs` |
| Prepared targets | `<project>/targets` | `<project>/targets` |
| Caches, tmp, JAX cache | `<project>/cache`, `/tmp` | `<project>/cache`, `/tmp` |
| Generated campaign JSON | `<source>/lab/configs/generated` | `<source>/lab/configs/generated` |
| Job registry | `<project>/results/.xulab_jobs.json` | `<project>/results/.xulab_jobs.json` |
| Writable weight cache (`BINDCRAFT_WEIGHTS`) | `<project>/cache/bindcraft` | `<project>/cache/bindcraft` |
| conda environment | `/home/ypan/.conda/envs/bindcraft2` | **the same, shared, executed only** |
| AlphaFold parameters (`BINDCRAFT_AF2_PARAMS`) | wherever the 5.3 GB set already is | **the same path, read only, never copied and never downloaded** |

### 9.1 How the separation is enforced

No tracked file names a deployment, and the proven SLURM launcher is not edited.
Two mechanisms do the work:

1. **`lab/scripts/dev_env.sh`** exports `XULAB_DEPLOYMENT`, `XULAB_BC2_HOME`,
   `XULAB_PROJECT_ROOT` and every derived root. `xulab_hpc.py` reads them, so
   results, logs, targets, caches, generated JSON and the job registry all move
   together. Sourcing nothing leaves a shell on production.
2. **`submit_campaign` states every path on the `sbatch` command line.** The
   launcher reads `XULAB_BC2_HOME` and `XULAB_PROJECT_ROOT` from the environment
   and falls back to the production paths, and its `--output`/`--error`
   directives name the production log directory. So every submission now passes
   `--export=ALL,XULAB_BC2_HOME=...,XULAB_PROJECT_ROOT=...` and
   `--output=/--error=` into its own logs root, where an `sbatch` flag beats a
   directive in the script. For production the values are identical to the
   launcher's own; for development nothing can fall back into the production
   tree. `BINDCRAFT_AF2_PARAMS` and `BINDCRAFT_WEIGHTS` are passed through, so
   the existing parameter set is read in place and never copied. Note that
   `BINDCRAFT_WEIGHTS` - the one weight path the launcher creates - is the dev
   cache, not the production one, so a dev run cannot write into production
   even if a future change tried to fetch a weight.

`lab/scripts/test_deployment_isolation.py` asserts this: it loads the helper
under each deployment's variables, checks all thirteen derived paths, builds a
real `sbatch` command with `dry_run=True`, and fails if the production source
tree or project tree is named anywhere in it, and asserts that the committed
CD19 campaign settings file writes into the development tree. 118 assertions,
passing.

### 9.2 Setting it up (exact commands)

```bash
# 1. on an HPC login node, inspect what would happen - this creates nothing
cd /home/ypan/BindCraft2/BindCraft2_deploy      # or any checkout of this repo
export XULAB_BC2_HOME=/home/ypan/BindCraft2/BindCraft2_cd19_dev
export XULAB_PROJECT_ROOT=/project/yli4/ypan/bindcraft_cd19_dev
bash lab/scripts/setup_dev_deployment.sh

# 2. create the project directories and clone the source tree
bash lab/scripts/setup_dev_deployment.sh --create
#    equivalently, by hand:
#      mkdir -p /project/yli4/ypan/bindcraft_cd19_dev/{results,logs,cache/jax,tmp,targets}
#      git clone --branch main https://github.com/cmri4902/XuLab-BinderDesign.git \
#          /home/ypan/BindCraft2/BindCraft2_cd19_dev

# 3. from now on, every development shell starts with this one line
cd /home/ypan/BindCraft2/BindCraft2_cd19_dev
source lab/scripts/dev_env.sh                   # prints the deployment it selected

# 4. the checks, none of which needs a GPU or an allocation
bash lab/scripts/hpc_preflight.sh               # must print RESULT: READY
PY=/home/ypan/.conda/envs/bindcraft2/bin/python
$PY lab/scripts/test_scfv_annotation.py         # expect 128/128, RESULT: PASS
$PY lab/scripts/test_target_tools.py            # expect 155/155, RESULT: PASS
$PY lab/scripts/test_deployment_isolation.py    # expect 103/103, RESULT: PASS
$PY lab/scripts/test_notebook_cells.py          # expect RESULT: PASS

# 5. confirm production is untouched before and after anything is submitted
ls -ld /project/yli4/ypan/bindcraft/results/pdl1_h200_smoke \
       /project/yli4/ypan/bindcraft_backups/pdl1_20261123_172646
ls /project/yli4/ypan/bindcraft/results
```

### 9.3 Running the CD19-scFv campaign (exact commands)

`lab/configs/cd19_scfv_discovery.json` is committed with the **development**
paths, so no settings file has to be written: steps 1-5 below are the whole
sequence. Run them in order; each one is checkable before the next.

#### Step 1 - deploy the source tree

```bash
# on an HPC login node, from any existing checkout of this repository
cd /home/ypan/BindCraft2/BindCraft2_deploy
export XULAB_BC2_HOME=/home/ypan/BindCraft2/BindCraft2_cd19_dev
export XULAB_PROJECT_ROOT=/project/yli4/ypan/bindcraft_cd19_dev

bash lab/scripts/setup_dev_deployment.sh            # look first; creates nothing
bash lab/scripts/setup_dev_deployment.sh --create   # mkdir + git clone
```

If the clone cannot reach GitHub from the HPC, copy the tree instead and skip
only the clone:

```bash
mkdir -p /project/yli4/ypan/bindcraft_cd19_dev/{results,logs,cache/jax,tmp,targets}
rsync -a --exclude '.git/' --exclude '__pycache__/' \
  /home/ypan/BindCraft2/BindCraft2_deploy/ /home/ypan/BindCraft2/BindCraft2_cd19_dev/
# then bring that copy up to this revision however the production tree was updated
```

#### Step 2 - preflight

```bash
cd /home/ypan/BindCraft2/BindCraft2_cd19_dev
source lab/scripts/dev_env.sh          # prints every dev path and the AF2 parameters it found

# if it printed "AF2 parameters NOT FOUND", locate the existing 5.3 GB set and export it.
# It is never downloaded and never copied:
#   export BINDCRAFT_AF2_PARAMS=/abs/path/to/dir/holding/params_model_1_multimer_v3.npz

bash lab/scripts/hpc_preflight.sh      # must print RESULT: READY

PY=/home/ypan/.conda/envs/bindcraft2/bin/python
$PY lab/scripts/test_scfv_annotation.py       # expect 128/128, RESULT: PASS
$PY lab/scripts/test_target_tools.py          # expect 155/155, RESULT: PASS
$PY lab/scripts/test_deployment_isolation.py  # expect 118/118, RESULT: PASS
$PY lab/scripts/test_notebook_cells.py        # expect RESULT: PASS
```

#### Step 3 - place the target, and confirm production is untouched

```bash
cd /home/ypan/BindCraft2/BindCraft2_cd19_dev
source lab/scripts/dev_env.sh

cp -n lab/targets/CD19_ECD.pdb "$XULAB_TARGETS/CD19_ECD.pdb"
grep -c '^ATOM' "$XULAB_TARGETS/CD19_ECD.pdb"      # expect 1656
grep -m1 'Residues present' "$XULAB_TARGETS/CD19_ECD.pdb"   # expect 227

# the settings file must name that copy and the dev results folder, and nothing else
python -c "import json;s=json.load(open('lab/configs/cd19_scfv_discovery.json'));print(s['targets'][0]['target_path']);print(s['project_folder'])"
# expect /project/yli4/ypan/bindcraft_cd19_dev/targets/CD19_ECD.pdb
#        /project/yli4/ypan/bindcraft_cd19_dev/results/cd19_scfv_discovery

# production, which nothing above or below writes to
ls -ld /project/yli4/ypan/bindcraft/results/pdl1_h200_smoke \
       /project/yli4/ypan/bindcraft_backups/pdl1_20261123_172646
ls /project/yli4/ypan/bindcraft/results
```

#### Step 4 - submit

The launcher is the unmodified, proven one. Every path is stated on the command
line, because an `sbatch` flag beats a directive inside the script, and there is
no `srun` anywhere.

```bash
cd /home/ypan/BindCraft2/BindCraft2_cd19_dev
source lab/scripts/dev_env.sh
mkdir -p "$XULAB_LOGS"

sbatch --parsable \
  --export=ALL,XULAB_BC2_HOME="$XULAB_BC2_HOME",XULAB_PROJECT_ROOT="$XULAB_PROJECT_ROOT",BINDCRAFT_AF2_PARAMS="$BINDCRAFT_AF2_PARAMS",BINDCRAFT_WEIGHTS="$BINDCRAFT_WEIGHTS" \
  --output="$XULAB_LOGS/xulab_bc2_%j.out" \
  --error="$XULAB_LOGS/xulab_bc2_%j.err" \
  lab/slurm/run_bindcraft_h200.slurm lab/configs/cd19_scfv_discovery.json
# prints the job id, and nothing else
# add --account=<account> if this site charges work to one
# the launcher pins cmri-gpu05; another H200 node needs only --nodelist=<node>
```

From the notebook instead, which builds exactly that command:

```bash
cd /home/ypan/BindCraft2/BindCraft2_cd19_dev && source lab/scripts/dev_env.sh && jupyter lab
#   cell 1 must print   deployment : cd19_dev
#   cell 2  -> a file on the cluster: $XULAB_TARGETS/CD19_ECD.pdb, chain A,
#             hotspots A162,A219,A220,A222
#   cell 3  -> scFv, cd19_scfv_discovery, 10 designs, 500 trajectories, resume on
#   cell 4  -> submit
```

#### Step 5 - monitor

```bash
JOB=<the job id sbatch printed>
squeue -j "$JOB" -o '%T %M %L %R %N'
tail -f "$XULAB_LOGS/xulab_bc2_$JOB.out"

# the four lines that prove the allocation, in the launcher's own output
grep -E 'xulab-bc2: (host|jax|device|gate passed)' "$XULAB_LOGS/xulab_bc2_$JOB.out"

# progress
ls "$XULAB_PROJECT_ROOT/results/cd19_scfv_discovery/1_Trajectories/" 2>/dev/null | head
wc -l "$XULAB_PROJECT_ROOT/results/cd19_scfv_discovery/2_Refolded/!_Refolded.csv" 2>/dev/null
ls "$XULAB_PROJECT_ROOT/results/cd19_scfv_discovery/3_Ranked/" 2>/dev/null

# when it ends
sacct -j "$JOB" -X --format=JobID,State,ExitCode,Elapsed,NodeList,ReqTRES
```

#### Step 6 - validate and read the results

```bash
cd /home/ypan/BindCraft2/BindCraft2_cd19_dev
source lab/scripts/dev_env.sh

$PY lab/scripts/validate_outputs.py \
  "$XULAB_PROJECT_ROOT/results/cd19_scfv_discovery" \
  --slurm-log "$XULAB_LOGS/xulab_bc2_$JOB.out" \
  --output "$XULAB_PROJECT_ROOT/results/cd19_scfv_discovery/validation.md"
# exit 0 nothing failed, 1 at least one FAIL, 2 the folder was unreadable
```

Then in the notebook: cell 6 loads
`/project/yli4/ypan/bindcraft_cd19_dev/results/cd19_scfv_discovery`, cell 7
prints up to ten accepted candidates with VH/VL sequences, annotated loops and
the Kabat HCDR3, cell 8 reviews them one at a time in 3D, cell 9 exports to
`/project/yli4/ypan/bindcraft_cd19_dev/results/exports/cd19_scfv_discovery/`
plus a zip, and cell 10 runs the audit above.

#### Budget and expectations

`max_trajectories: 500` with `resume: true` means **the submit command in step 4
can be run again unchanged** and the campaign continues in the same folder. The
launcher declares an 08:00:00 wall clock, so a 500-trajectory budget may need
more than one submission; resuming is the supported way to do that.

Ten accepted candidates is the **goal, not a guarantee**. A campaign that
accepts fewer, or none, is an ordinary outcome and is reported as such: cell 7
says how many were accepted and states that nothing was substituted to reach
ten, and cell 6 shows the rejected candidates with the filters that rejected
them. Nothing in this project relaxes a filter to reach a count.

---

## 10. Licensing and safety

BindCraft2 is **source-available with hosting restrictions**, not MIT. Internal
organizational use and source modification are permitted subject to attribution
and marking changes; third-party hosted access to the software's substantial
functionality is restricted. `LICENSE` and `README_licensing_section.md` are
preserved verbatim, and `lab/CHANGELOG_XULAB.md` is the change notice.

Third-party terms also apply to the AlphaFold model parameters, ColabDesign and
ProteinMPNN/HyperMPNN. This is an internal lab deployment; it is not permission
to expose unpublished CMRI research externally, and any public read-only viewer
would need institutional and licensing review first.

Phase 1 tests software behaviour, not biology. A computationally accepted design
is a hypothesis for experimental testing.
