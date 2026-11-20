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
/project/yli4/ypan/bindcraft/
├── BindCraft2/              the source copy, replaceable at any time
├── results/                 campaign output, NEVER deleted by a source update
│   └── pdl1_h200_smoke/
├── logs/                    SLURM stdout/stderr (must exist before sbatch)
├── cache/                   jax/ compilation cache, bindcraft/ weight cache
└── tmp/                     TMPDIR for the job
```

Source and data are deliberately siblings: `rm -rf BindCraft2 && rsync` again
cannot touch `results/`. Nothing is written to `$HOME` (≈300 GB quota) and no
second copy of the 5.3 GB AlphaFold parameters is ever made.

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
scp C:\XuLab\bc2_release\bindcraft2_xulab_<short>_<state>_<stamp>.tar.gz <user>@<hpc-host>:/project/yli4/ypan/bindcraft/
```

On the HPC:

```bash
cd /project/yli4/ypan/bindcraft
# tar merges into an existing BindCraft2/; move the old source aside for a clean
# replacement. results/, logs/, cache/ and tmp/ are siblings and are untouched.
[ -d BindCraft2 ] && mv BindCraft2 BindCraft2.$(date +%Y%m%d%H%M%S).bak
tar -xzf bindcraft2_xulab_<short>_<state>_<stamp>.tar.gz     # unpacks ./BindCraft2
cat BindCraft2/lab/BUILD_REVISION.txt                        # provenance of this copy
```

#### Option B - rsync of the working tree (for incremental re-syncs)

Run in **Git Bash or WSL**, where `rsync` and `/c/...` paths both exist. A
trailing slash on the source is required, and `--delete` is pointed at the
source directory only, never at the project root:

```bash
rsync -av --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' \
  --exclude '.ipynb_checkpoints/' \
  /c/XuLab/BindCraft2/ <user>@<hpc-host>:/project/yli4/ypan/bindcraft/BindCraft2/
```

rsync copies bytes, so this only ships LF if the Windows working tree is LF.
The repository's `.gitattributes` plus `core.autocrlf=false` keep it that way;
confirm before syncing, and after syncing on the HPC:

```bash
# Windows (Git Bash), in the repository
git config core.autocrlf        # expect: false
git ls-files --eol install.sh lab/slurm/run_bindcraft_h200.slurm   # expect w/lf

# HPC, after the transfer
cd /project/yli4/ypan/bindcraft/BindCraft2
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
cd /project/yli4/ypan/bindcraft/BindCraft2
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
cd /project/yli4/ypan/bindcraft/BindCraft2
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
H200, `xulab-bc2: bindcraft /project/yli4/ypan/bindcraft/BindCraft2/bindcraft/__init__.py`,
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

1. the job `cd`s into `/project/yli4/ypan/bindcraft/BindCraft2`, and
   `bindcraft.py` inserts its own directory at the front of `sys.path`;
2. `PYTHONPATH` is prefixed with the lab tree and `BINDCRAFT_HOME` names it;
3. the gate process asserts it, comparing the resolved `bindcraft.__file__`
   against `BINDCRAFT_HOME` and exiting nonzero on a mismatch.

Verify by hand at any time (the printed path, not the version string, is the
evidence):

```bash
cd /project/yli4/ypan/bindcraft/BindCraft2
BINDCRAFT_HOME=$PWD PYTHONPATH=$PWD /home/ypan/.conda/envs/bindcraft2/bin/python -s \
  -c 'import bindcraft, pathlib; print(pathlib.Path(bindcraft.__file__).resolve())'
# must print /project/yli4/ypan/bindcraft/BindCraft2/bindcraft/__init__.py
```

If it prints a path under the older installation, that installation is winning
the import; `pip install -e /project/yli4/ypan/bindcraft/BindCraft2` inside the
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

### Common execution failures

| Symptom in the log | Cause | Fix |
| --- | --- | --- |
| `ERROR: no JAX GPU in this allocation` | No GPU requested, or `CUDA_VISIBLE_DEVICES` empty | Submit through the lab script and keep `--gres=gpu:h200:1`; check `sacct -j <id> --format=ReqTRES%40`. |
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

Results live in `/project/yli4/ypan/bindcraft/results/`, outside the source
tree, so updating the code cannot delete them. Keep `--delete` in any `rsync`
pointed only at `.../bindcraft/BindCraft2/`, never at `.../bindcraft/`. Deleting
a `.cif` from `3_Ranked/` is upstream's documented way to reject a design and
drops its row when the campaign closes - do it deliberately, never as cleanup.

---

## 6. Status

| Phase 1 item | Status |
| --- | --- |
| Complete upstream clone at `C:\XuLab\BindCraft2`, history and tags | **DONE** (186 tracked files, commit `8c1a858`) |
| Upstream commit SHA and source inventory | **DONE** (`lab/SOURCE_INVENTORY.md`) |
| All upstream modalities, presets, scaffolds, notebook, checkpoints preserved | **DONE** (no upstream file edited or deleted; `README.md` renamed to `README_UPSTREAM.md`, same blob) |
| `lab/slurm/run_bindcraft_h200.slurm` | **PREPARED**, NOT TESTED (never submitted) |
| `lab/configs/pdl1_smoke.json` | **PREPARED**, NOT RUN (keys checked against `bindcraft/settings.py`, not executed) |
| `lab/scripts/validate_outputs.py` | **PREPARED**; syntax-checked and exercised on synthetic fixtures outside the repository, including negative fixtures (allocation-only GPU evidence, nonzero exit, missing structure per candidate); never run against a real campaign |
| `lab/scripts/hpc_preflight.sh` | **PREPARED**, NOT TESTED on the HPC; `bash -n` clean and its exit-code logic exercised locally against missing directories, a missing interpreter, missing AlphaFold parameters and a CRLF script |
| `lab/scripts/make_release.sh` | **PREPARED**; `bash -n` clean and run locally in Git Bash to build a verified archive (content, 12 checkpoints, no CRLF). Never transferred to the HPC |
| `.gitattributes` + `core.autocrlf=false` | **DONE and VERIFIED locally**: all Linux-executed scripts, upstream ones included, are LF on disk, and `git diff HEAD` against commit `8c1a858` is empty |
| `README_XULAB.md` + change log | **DONE** |
| HPC deployment plan and commands | **DONE** as documentation; **NOT EXECUTED** |
| HPC directories under `/project/yli4/ypan/bindcraft/` | **NOT CREATED** (no HPC access from here) |
| Write permission on `/project/yli4/ypan/` | **NOT VERIFIED** (`lab/scripts/hpc_preflight.sh` verifies it) |
| AlphaFold parameter path on the HPC | **NOT VERIFIED**; searched at run time, never hardcoded, never downloaded |
| Python imports the new source copy | **MECHANISMS PREPARED + CHECKS WRITTEN**; not yet demonstrated on the HPC |
| Level A - compute smoke | **NOT TESTED** |
| Level B - end-to-end binder output | **NOT ACHIEVED** |
| Phase 2 Jupyter notebook (`lab/notebooks/xulab_bindcraft_h200.ipynb`) | **IMPLEMENTED**; `nbformat`-valid, all 10 code cells and every widget callback exercised locally against synthetic fixtures; **never run on the HPC** |
| Phase 2 helper module (`lab/notebooks/xulab_hpc.py`) | **IMPLEMENTED**; 90 local assertions pass on both the pandas and no-pandas paths |
| Real `sbatch` / `squeue` / `sacct` behaviour from the notebook | **NOT TESTED** (no SLURM on the development machine) |
| `py3Dmol` rendering inside institutional Jupyter | **NOT TESTED** (needs internet access for 3Dmol.js) |

No GPU job, campaign, `sbatch`, `srun` or HPC command has been run from this
copy. Every PASS in this project must cite a log path and line.

---

## 7. The Jupyter notebook on CMRI HPC (Phase 2)

`lab/notebooks/xulab_bindcraft_h200.ipynb` is the official `bindcraft.ipynb`
adapted for regular Jupyter here; `bindcraft.ipynb` itself is untouched. All of
its non-widget logic is in `lab/notebooks/xulab_hpc.py`, which imports and runs
without Jupyter, SLURM or a GPU, so it can be tested and reviewed on its own.

```bash
# on the HPC, with the bindcraft2 environment active
cd /project/yli4/ypan/bindcraft/BindCraft2
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

## 8. Licensing and safety

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
