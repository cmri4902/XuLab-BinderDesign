"""Xu Lab BindCraft2 - Jupyter/SLURM helper layer (Phase 2).

Everything the notebook does that is not a widget lives here, so it can be read,
reviewed and tested outside a notebook. The module is import-safe on any
machine: it never requires SLURM, a GPU, pandas or py3Dmol at import time, and
every optional dependency is reported rather than installed.

Division of labour:
  * this module          paths, settings, submission, job tracking, result reading
  * the .slurm script    the actual H200 allocation and its GPU/import gates
  * validate_outputs.py  the auditable PASS/FAIL report over a campaign folder

Nothing here runs a design campaign in the notebook process. A campaign is
always a batch job, so closing the notebook cannot kill it.

Paths come from XULAB_* environment variables where set, so one notebook serves
several users without editing the file.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# locations
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(os.environ.get("XULAB_PROJECT_ROOT", "/project/yli4/ypan/bindcraft"))
SOURCE_TREE = Path(os.environ.get("XULAB_BC2_HOME", str(PROJECT_ROOT / "BindCraft2")))
RESULTS_ROOT = Path(os.environ.get("XULAB_RESULTS", str(PROJECT_ROOT / "results")))
LOGS_ROOT = Path(os.environ.get("XULAB_LOGS", str(PROJECT_ROOT / "logs")))
CACHE_ROOT = Path(os.environ.get("XULAB_CACHE", str(PROJECT_ROOT / "cache")))
TMP_ROOT = Path(os.environ.get("XULAB_TMP", str(PROJECT_ROOT / "tmp")))
CONDA_PYTHON = Path(os.environ.get("XULAB_PYTHON", "/home/ypan/.conda/envs/bindcraft2/bin/python"))

SLURM_SCRIPT = SOURCE_TREE / "lab" / "slurm" / "run_bindcraft_h200.slurm"
VALIDATOR = SOURCE_TREE / "lab" / "scripts" / "validate_outputs.py"
GENERATED_CONFIGS = SOURCE_TREE / "lab" / "configs" / "generated"
PRESETS = SOURCE_TREE / "settings"
# Job records live with the results, not with the source, so a source update or a
# new notebook session cannot lose them.
JOB_REGISTRY = RESULTS_ROOT / ".xulab_jobs.json"

REQUIRED_MODELS = ("model_1_multimer_v3", "model_2_multimer_v3", "model_3_multimer_v3",
                   "model_4_multimer_v3", "model_5_multimer_v3", "model_1_ptm", "model_2_ptm")

# Form label -> modality preset name, as settings/modality/*.json names them.
BINDER_TYPES = {
    "de novo binder": "binder",
    "large binder": "large_binder",
    "linear peptide": "peptide",
    "cyclic peptide": "cyclic_peptide",
    "homo-oligomer": "homo_oligomer",
    "multidomain binder": "multidomain",
    "VHH (single-domain antibody)": "VHH",
    "ankyrin repeat protein": "ARP",
    "scFv": "scFv",
    "Fab": "Fab",
}
SCAFFOLDED_TYPES = ("VHH", "ARP", "scFv", "Fab")
PROPERTY_OPTIONS = {
    "bind only the named residues": "forced_targeting",
    "human-like sequence": "humanize",
    "protease resistant": "protease_stable",
    "add a disulfide staple": "disulfide_staple",
    "more beta sheet": "mixed_topology",
    "termini close together": "termini_together",
    "termini point away from the target": "termini_accessible",
    "hard target": "hard_target",
}

# Columns worth showing; missing ones are skipped rather than invented.
RANKED_COLUMNS = ["rank", "design", "Binder_Length", "i_pDAE", "i_pTM", "i_pAE", "pLDDT",
                  "Interface_Residues", "Interface_BuriedArea", "Surface_Hydrophobicity",
                  "Binder_Mass_kDa", "Binder_pI", "Binder_Net_Charge", "Binder_Sequence"]
CANDIDATE_COLUMNS = ["design", "outcome", "failed_filters", "i_pDAE", "i_pTM", "pLDDT",
                     "i_pAE", "Interface_Residues", "Binder_Sequence"]
ATTEMPT_COLUMNS = ["design", "trajectory", "terminated", "autotuned", "i_pTM", "pLDDT"]
CONFIDENCE_COLUMNS = ["i_pDAE", "i_pTM", "i_pAE", "pLDDT", "pTM", "Target_pLDDT",
                      "Unbound_Binder_pLDDT", "SS_pLDDT"]
INTERFACE_COLUMNS = ["Interface_Residues", "Interface_BuriedArea", "Interface_BuriedArea_Fraction",
                     "Hotspot_Contact_Fraction", "Off_Epitope_Contact_Fraction",
                     "Surface_Hydrophobicity", "Backbone_Clashes", "All_Atom_Clashes",
                     "Interface_Binder_Residues", "Interface_Target_Residues"]


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #

def _run(command, cwd=None, timeout=120, env=None):
    """Run a command and return (returncode, stdout, stderr); never raises."""
    try:
        finished = subprocess.run([str(part) for part in command], cwd=str(cwd) if cwd else None,
                                  capture_output=True, text=True, timeout=timeout,
                                  env={**os.environ, **(env or {})})
        return finished.returncode, finished.stdout, finished.stderr
    except FileNotFoundError as missing:
        return 127, "", f"command not found: {missing}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except OSError as failure:                                  # noqa: BLE001
        return 1, "", str(failure)


def have_command(name: str) -> bool:
    return shutil.which(name) is not None


def read_csv_rows(path: Path):
    """(columns, rows) from a CSV, or ([], []) when it cannot be read."""
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return (reader.fieldnames or []), list(reader)
    except (OSError, UnicodeDecodeError, csv.Error):
        return [], []


def as_frame(columns, rows, keep=None):
    """A pandas DataFrame when pandas is importable, else the list of dicts."""
    selected = [column for column in (keep or columns) if column in columns]
    try:
        import pandas
    except ImportError:
        return [{column: row.get(column) for column in selected} for row in rows]
    frame = pandas.DataFrame(rows, columns=columns)
    return frame[selected] if selected else frame


def alphafold_parameters() -> tuple[Path | None, str]:
    """Where the existing 5.3 GB parameter set is, searched and never downloaded."""
    candidates = []
    configured = os.environ.get("BINDCRAFT_AF2_PARAMS")
    if configured:
        candidates.append(Path(configured))
    weights = os.environ.get("BINDCRAFT_WEIGHTS")
    if weights:
        candidates.append(Path(weights) / "alphafold")
    candidates += [Path.home() / ".cache" / "bindcraft" / "alphafold",
                   CACHE_ROOT / "bindcraft" / "alphafold",
                   SOURCE_TREE / "bindcraft" / "weights" / "alphafold"]
    for candidate in candidates:
        try:
            files = list(candidate.glob("params_*.npz")) + list((candidate / "params").glob("params_*.npz"))
        except OSError:
            continue
        if files:
            names = {path.name for path in files}
            missing = [model for model in REQUIRED_MODELS
                       if f"params_{model}.npz" not in names]
            if missing:
                return candidate, f"{len(files)} file(s) but missing: {', '.join(missing)}"
            return candidate, f"{len(files)} params_*.npz, all {len(REQUIRED_MODELS)} required models present"
    return None, "not found in any searched location; set BINDCRAFT_AF2_PARAMS"


# --------------------------------------------------------------------------- #
# 1. environment
# --------------------------------------------------------------------------- #

def environment_report() -> list[tuple[str, str, str]]:
    """(check, status, detail) rows. Status is PASS / FAIL / WARN / NOT TESTED.

    This runs on the Jupyter node, which is not the H200 allocation, so it makes
    no GPU claim at all: the GPU is proven only by the batch job's own log.
    """
    rows: list[tuple[str, str, str]] = []

    rows.append(("source tree", "PASS" if (SOURCE_TREE / "bindcraft" / "__init__.py").is_file()
                 else "FAIL", str(SOURCE_TREE)))
    rows.append(("slurm launcher", "PASS" if SLURM_SCRIPT.is_file() else "FAIL", str(SLURM_SCRIPT)))
    rows.append(("validator", "PASS" if VALIDATOR.is_file() else "FAIL", str(VALIDATOR)))

    for label, directory in (("results root", RESULTS_ROOT), ("logs root", LOGS_ROOT),
                             ("cache root", CACHE_ROOT), ("tmp root", TMP_ROOT)):
        if not directory.is_dir():
            rows.append((label, "FAIL", f"missing: {directory} (mkdir -p {directory})"))
        elif not os.access(directory, os.W_OK):
            rows.append((label, "FAIL", f"not writable: {directory}"))
        else:
            rows.append((label, "PASS", str(directory)))

    rows.append(("conda interpreter", "PASS" if os.access(CONDA_PYTHON, os.X_OK) else "FAIL",
                 str(CONDA_PYTHON)))
    rows.append(("sbatch", "PASS" if have_command("sbatch") else "FAIL",
                 shutil.which("sbatch") or "not on PATH; submission is impossible from here"))
    rows.append(("squeue/sacct", "PASS" if have_command("squeue") else "WARN",
                 "job status needs squeue/sacct"))

    parameters, detail = alphafold_parameters()
    rows.append(("AlphaFold parameters", "PASS" if parameters and "missing" not in detail else "FAIL",
                 f"{parameters}: {detail}" if parameters else detail))

    checkpoints = list((SOURCE_TREE / "bindcraft" / "weights" / "proteinmpnn").glob("*/*.npz"))
    rows.append(("bundled MPNN checkpoints", "PASS" if len(checkpoints) == 12 else "FAIL",
                 f"{len(checkpoints)} .npz (expected 12), shipped with the source"))

    for module in ("ipywidgets", "pandas", "py3Dmol"):
        try:
            __import__(module)
            rows.append((f"python: {module}", "PASS", "importable in this kernel"))
        except ImportError:
            note = ("3D views are skipped; tables and downloads still work"
                    if module == "py3Dmol" else "install it in the kernel environment")
            rows.append((f"python: {module}", "WARN", f"not importable - {note}"))

    rows.append(("GPU / H200", "NOT TESTED",
                 "a Jupyter node is not the batch allocation; GPU use is proven only by the "
                 "job log (the launcher exits nonzero when JAX has no GPU)"))
    return rows


def imported_bindcraft_path() -> tuple[str, str]:
    """Which bindcraft the conda interpreter imports, checked out of process."""
    if not os.access(CONDA_PYTHON, os.X_OK):
        return "NOT TESTED", f"interpreter not executable: {CONDA_PYTHON}"
    code = ("import pathlib, bindcraft;"
            "print(pathlib.Path(bindcraft.__file__).resolve())")
    status, out, err = _run([CONDA_PYTHON, "-s", "-c", code], cwd=SOURCE_TREE,
                            env={"PYTHONPATH": str(SOURCE_TREE)})
    if status != 0:
        return "FAIL", (err or out).strip()[:400]
    module = out.strip()
    inside = str(SOURCE_TREE.resolve()) in module
    return ("PASS" if inside else "FAIL"),  module + ("" if inside else
                                                      f"  <- OUTSIDE {SOURCE_TREE}")


# --------------------------------------------------------------------------- #
# 2. targets and settings
# --------------------------------------------------------------------------- #

def available_targets() -> list[str]:
    return sorted(path.stem for path in (PRESETS / "target").glob("*.json"))


def available_modalities() -> list[str]:
    return sorted(path.stem for path in (PRESETS / "modality").glob("*.json"))


def available_properties() -> list[str]:
    return sorted(path.stem for path in (PRESETS / "property").glob("*.json"))


def preset_target(name: str) -> dict:
    """A shipped target expanded into a settings entry with an absolute path."""
    preset = json.loads((PRESETS / "target" / f"{name}.json").read_text(encoding="utf-8"))
    entry = dict(preset["targets"][0])
    entry["target_path"] = str((PRESETS / "target" / entry["target_path"]).resolve())
    return entry


def custom_target(path: str, name: str = "", chains: str = "", hotspots: str = "",
                  coldspots: str = "") -> dict:
    """A target of your own: a PDB, mmCIF or FASTA file already on the cluster."""
    located = Path(path).expanduser()
    if not located.is_file():
        raise FileNotFoundError(f"target file not found: {located}")
    entry = {"name": name.strip() or located.stem, "target_path": str(located.resolve())}
    for key, value in (("chains", chains), ("hotspots", hotspots), ("coldspots", coldspots)):
        if str(value).strip():
            entry[key] = str(value).strip()
    return entry


def modality_lengths(modality: str):
    """What a binder type already decides about length, or None."""
    try:
        preset = json.loads((PRESETS / "modality" / f"{modality}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return preset.get("binder_lengths")


def build_campaign_settings(campaign_name: str, targets: list[dict], modality: str = "binder",
                            properties=(), binder_lengths=None, number_of_final_designs: int = 5,
                            max_trajectories: int = 100, workers_per_gpu: int = 1,
                            copies: int | None = None, extra: dict | None = None) -> dict:
    """The settings dict BindCraft2 reads, with an absolute HPC project_folder.

    Acceptance filters are left exactly as the chosen presets define them. The
    notebook deliberately offers no "turn the filters off" control: that is the
    Colab demo shortcut and it would make a campaign's output meaningless.
    """
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", campaign_name.strip()) or "campaign"
    if not targets:
        raise ValueError("a campaign needs at least one target")
    if modality not in available_modalities():
        raise ValueError(f"unknown modality {modality!r}; one of {available_modalities()}")
    unknown = [item for item in properties if item not in available_properties()]
    if unknown:
        raise ValueError(f"unknown property preset(s): {unknown}")

    settings: dict = {
        "campaign_name": name,
        "modality": [modality] if isinstance(modality, str) else list(modality),
        "targets": [dict(target) for target in targets],
        "number_of_final_designs": int(number_of_final_designs),
        "max_trajectories": int(max_trajectories),
        "workers_per_gpu": int(workers_per_gpu),
        "project_folder": str(RESULTS_ROOT / name),
    }
    for item in properties:
        settings[item] = True
    if binder_lengths and modality not in SCAFFOLDED_TYPES:
        shortest, longest = int(binder_lengths[0]), int(binder_lengths[1])
        settings["binder_lengths"] = [min(shortest, longest), max(shortest, longest)]
    if copies and copies > 1:
        settings["copies"] = int(copies)
    settings.update(extra or {})
    return settings


def write_campaign_config(settings: dict, directory: Path | None = None) -> Path:
    """Write the settings next to the source copy and return the file path."""
    folder = Path(directory) if directory else GENERATED_CONFIGS
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{settings['campaign_name']}.json"
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


# --------------------------------------------------------------------------- #
# 3. submission and job tracking
# --------------------------------------------------------------------------- #

@dataclass
class Submission:
    job_id: str = ""
    campaign: str = ""
    config: str = ""
    project_folder: str = ""
    submitted_at: str = ""
    command: list = field(default_factory=list)
    log_out: str = ""
    log_err: str = ""
    message: str = ""

    def as_dict(self) -> dict:
        return dict(job_id=self.job_id, campaign=self.campaign, config=self.config,
                    project_folder=self.project_folder, submitted_at=self.submitted_at,
                    command=self.command, log_out=self.log_out, log_err=self.log_err,
                    message=self.message)


def read_registry() -> list[dict]:
    try:
        records = json.loads(JOB_REGISTRY.read_text(encoding="utf-8"))
        return records if isinstance(records, list) else []
    except (OSError, ValueError):
        return []


def record_job(submission: Submission) -> None:
    records = read_registry()
    records.append(submission.as_dict())
    try:
        JOB_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        JOB_REGISTRY.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8",
                                newline="\n")
    except OSError as failure:
        print(f"xulab: job recorded in memory only, registry unwritable: {failure}",
              file=sys.stderr)


def submit_campaign(config_path: Path, account: str = "", af2_parameters: str = "",
                    extra_sbatch=(), dry_run: bool = False) -> Submission:
    """sbatch the H200 launcher for one settings file.

    The campaign runs entirely inside the batch job: this notebook kernel does no
    design work, so closing the notebook leaves the job running.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"settings file not found: {config_path}")
    if not SLURM_SCRIPT.is_file():
        raise FileNotFoundError(f"launcher not found: {SLURM_SCRIPT}")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    campaign = settings.get("campaign_name", config_path.stem)
    project_folder = settings.get("project_folder", str(RESULTS_ROOT / campaign))

    parameters = af2_parameters or os.environ.get("BINDCRAFT_AF2_PARAMS") or ""
    if not parameters:
        located, _ = alphafold_parameters()
        parameters = str(located) if located else ""
    exports = ["ALL"]
    if parameters:
        exports.append(f"BINDCRAFT_AF2_PARAMS={parameters}")

    # SLURM opens --output before the job starts, so the directory must exist now.
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)

    command = ["sbatch", "--parsable", f"--export={','.join(exports)}"]
    if account.strip():
        command.append(f"--account={account.strip()}")
    command += [str(part) for part in extra_sbatch]
    command += [str(SLURM_SCRIPT), str(config_path)]

    submission = Submission(campaign=campaign, config=str(config_path),
                            project_folder=project_folder,
                            submitted_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            command=command)
    if dry_run:
        submission.message = "dry run: nothing was submitted"
        return submission
    if not have_command("sbatch"):
        submission.message = ("sbatch is not on PATH in this kernel; run the printed command "
                              "from a login shell instead")
        return submission

    status, out, err = _run(command, cwd=SOURCE_TREE, timeout=180)
    if status != 0:
        submission.message = f"sbatch failed (exit {status}): {(err or out).strip()[:500]}"
        return submission
    submission.job_id = (out.strip().split(";")[0] or "").strip()
    submission.log_out = str(LOGS_ROOT / f"xulab_bc2_{submission.job_id}.out")
    submission.log_err = str(LOGS_ROOT / f"xulab_bc2_{submission.job_id}.err")
    submission.message = f"submitted as job {submission.job_id}"
    record_job(submission)
    return submission


def job_status(job_id: str) -> dict:
    """Live state from squeue, falling back to sacct once the job has left it."""
    job_id = str(job_id).strip()
    if not job_id:
        return {"job_id": job_id, "state": "UNKNOWN", "detail": "no job id"}
    if have_command("squeue"):
        status, out, _ = _run(["squeue", "-j", job_id, "--noheader",
                               "--format=%T|%M|%L|%R|%N"], timeout=60)
        line = out.strip().splitlines()[0] if status == 0 and out.strip() else ""
        if line:
            state, elapsed, left, reason, node = (line.split("|") + [""] * 5)[:5]
            return {"job_id": job_id, "state": state, "elapsed": elapsed, "time_left": left,
                    "reason": reason, "node": node, "source": "squeue"}
    if have_command("sacct"):
        status, out, _ = _run(["sacct", "-j", job_id, "--noheader", "-X", "-P",
                               "--format=State,ExitCode,Elapsed,NodeList,ReqTRES"], timeout=60)
        line = out.strip().splitlines()[0] if status == 0 and out.strip() else ""
        if line:
            state, exit_code, elapsed, node, tres = (line.split("|") + [""] * 5)[:5]
            return {"job_id": job_id, "state": state, "exit_code": exit_code, "elapsed": elapsed,
                    "node": node, "requested": tres, "source": "sacct"}
    return {"job_id": job_id, "state": "UNKNOWN",
            "detail": "neither squeue nor sacct could report this job from this kernel"}


def tracked_jobs(limit: int = 20) -> list[dict]:
    """Recorded jobs, newest first, each with its current state."""
    records = read_registry()[-limit:][::-1]
    for record in records:
        record = record
        record["status"] = job_status(record.get("job_id", ""))
    return records


def log_tail(job_id: str = "", path: str = "", lines: int = 40) -> str:
    candidates = [Path(path)] if path else []
    if job_id:
        candidates += [LOGS_ROOT / f"xulab_bc2_{job_id}.out",
                       LOGS_ROOT / f"xulab_bc2_{job_id}.err"]
    pieces = []
    for candidate in candidates:
        if candidate.is_file():
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as failure:
                pieces.append(f"--- {candidate}: {failure}")
                continue
            pieces.append(f"--- {candidate} (last {min(lines, len(text))} of {len(text)} lines)")
            pieces += text[-lines:]
    return "\n".join(pieces) or f"no log file yet for job {job_id or path}"


def cancel_job(job_id: str) -> str:
    """scancel one job. Only ever called from the notebook's explicit button."""
    if not have_command("scancel"):
        return "scancel is not available from this kernel"
    status, out, err = _run(["scancel", str(job_id)], timeout=60)
    return f"scancel exit {status} {(out or err).strip()}".strip()


# --------------------------------------------------------------------------- #
# 4. reading finished or partial campaigns
# --------------------------------------------------------------------------- #

def campaign_folders(root: Path | None = None) -> list[Path]:
    """Every campaign folder under the results root, most recently written first."""
    base = Path(root) if root else RESULTS_ROOT
    if not base.is_dir():
        return []
    found = {path.parent.parent for path in base.glob("*/1_Trajectories/!_Trajectories.csv")}
    found |= {path.parent.parent for path in base.glob("*/3_Ranked/!_Ranked.csv")}
    found |= {path.parent for path in base.glob("*/campaign_metadata*.json")}
    found |= {path.parent for path in base.glob("*/.campaign_state.json")}
    return sorted(found, key=lambda folder: folder.stat().st_mtime, reverse=True)


@dataclass
class Campaign:
    folder: Path
    ranked_columns: list = field(default_factory=list)
    ranked: list = field(default_factory=list)
    candidate_columns: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    attempt_columns: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    structures: list = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.ranked)

    def summary(self) -> str:
        parts = [f"campaign folder : {self.folder}",
                 f"attempts        : {len(self.attempts)}",
                 f"candidates      : {len(self.candidates)}",
                 f"accepted designs: {len(self.ranked)}",
                 f"complex files   : {len(self.structures)}"]
        if not self.ranked:
            parts.append("")
            parts.append("Nothing accepted. For a short campaign this is an ordinary outcome, "
                         "not a failure: read the candidates' failed_filters below, then the "
                         "attempts' termination stages.")
        else:
            parts.append("")
            parts.append("Acceptance means the configured computational filters passed. It is "
                         "not evidence of binding, affinity, specificity or stability.")
        return "\n".join(parts)


def _first_existing(folder: Path, *relatives):
    for relative in relatives:
        candidate = folder / relative
        if candidate.is_file():
            return candidate
    return None


def load_campaign(folder) -> Campaign:
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"campaign folder not found: {folder}")
    campaign = Campaign(folder=folder)

    ranked = _first_existing(folder, "3_Ranked/!_Ranked.csv", "ranked.csv", "accepted.csv")
    if ranked:
        campaign.ranked_columns, campaign.ranked = read_csv_rows(ranked)
    candidates = _first_existing(folder, "2_Refolded/!_Refolded.csv", "candidates.csv")
    if candidates:
        campaign.candidate_columns, campaign.candidates = read_csv_rows(candidates)
    attempts = _first_existing(folder, "1_Trajectories/!_Trajectories.csv", "trajectories.csv")
    if attempts:
        campaign.attempt_columns, campaign.attempts = read_csv_rows(attempts)

    for path in sorted(folder.glob("campaign_metadata*.json")):
        try:
            campaign.metadata = json.loads(path.read_text(encoding="utf-8"))
            break
        except (OSError, ValueError):
            continue

    ranked_dir = folder / "3_Ranked" if (folder / "3_Ranked").is_dir() else folder / "accepted"
    if ranked_dir.is_dir():
        campaign.structures = [path for path in sorted(ranked_dir.glob("*.cif"))
                               if "_monomer" not in path.name]
    return campaign


def ranked_view(campaign: Campaign, limit: int = 20):
    return as_frame(campaign.ranked_columns, campaign.ranked[:limit], RANKED_COLUMNS)


def candidate_view(campaign: Campaign, limit: int = 20):
    return as_frame(campaign.candidate_columns, campaign.candidates[:limit], CANDIDATE_COLUMNS)


def attempt_view(campaign: Campaign, limit: int = 20):
    return as_frame(campaign.attempt_columns, campaign.attempts[:limit], ATTEMPT_COLUMNS)


def rejection_counts(campaign: Campaign) -> dict:
    """How often each filter rejected a candidate."""
    counts: dict = {}
    for row in campaign.candidates:
        for reason in (row.get("failed_filters") or "").split(","):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def design_report(campaign: Campaign, limit: int = 10) -> list[dict]:
    """Per accepted design: confidence, interface readings and its structure files."""
    report = []
    for row in campaign.ranked[:limit]:
        design = (row.get("design") or "").strip()
        files = [path.name for path in campaign.structures
                 if path.stem == design or path.stem.startswith(design + "_")]
        entry = {"design": design,
                 "rank": row.get("rank", ""),
                 "sequence": row.get("Binder_Sequence", ""),
                 "chains": len([part for part in str(row.get("Binder_Sequence", "")).split("/")
                                if part]),
                 "structures": files or ["(none found)"],
                 "confidence": {key: row[key] for key in CONFIDENCE_COLUMNS
                                if key in row and str(row[key]).strip()},
                 "interface": {key: row[key] for key in INTERFACE_COLUMNS
                               if key in row and str(row[key]).strip()}}
        report.append(entry)
    return report


def binder_fasta(campaign: Campaign, limit: int = 20) -> str:
    """FASTA of the accepted binders. Multi-chain binders become one record each."""
    records = []
    for row in campaign.ranked[:limit]:
        design = (row.get("design") or "design").strip()
        notes = " ".join(f"{key}={row[key]}" for key in ("rank", "i_pDAE", "i_pTM", "pLDDT")
                         if key in row and str(row[key]).strip())
        chains = [chain.strip() for chain in str(row.get("Binder_Sequence", "")).split("/")
                  if chain.strip()]
        for number, sequence in enumerate(chains, start=1):
            label = design if len(chains) == 1 else f"{design}_chain{number}"
            records.append(f">{label} {notes}".rstrip() + "\n" + sequence)
    return "\n".join(records) + ("\n" if records else "")


def chain_names(path: Path) -> list[str]:
    """Chain labels in an mmCIF file, read as text (no parser dependency)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    headers, chains = [], []
    in_loop = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            headers.append(stripped.split(".", 1)[1])
            in_loop = True
            continue
        if in_loop:
            if not stripped or stripped.startswith(("#", "loop_", "_", "data_")):
                in_loop = False
                continue
            if stripped.startswith(("ATOM", "HETATM")):
                fields = stripped.split()
                for name in ("auth_asym_id", "label_asym_id"):
                    if name in headers and len(fields) > headers.index(name):
                        chain = fields[headers.index(name)]
                        if chain not in chains:
                            chains.append(chain)
                        break
    return chains


def show_structures(campaign: Campaign, how_many: int = 2, colour_by: str = "chain"):
    """A py3Dmol grid of the accepted complexes, best first.

    Returns the view (call .show() on it) or raises RuntimeError with a readable
    reason. py3Dmol needs internet access for 3Dmol.js: on a closed network the
    tables and the CIF downloads remain the way to inspect results.
    """
    if not campaign.structures:
        raise RuntimeError(f"no accepted complex files in {campaign.folder / '3_Ranked'}")
    try:
        import py3Dmol
    except ImportError as missing:
        raise RuntimeError(
            "py3Dmol is not importable in this kernel. Install it in the bindcraft2 environment "
            "(python -m pip install py3Dmol) or download the .cif files and open them locally."
        ) from missing

    order = [(row.get("design") or "").strip() for row in campaign.ranked]
    by_rank = [path for design in order for path in campaign.structures
               if path.stem == design or path.stem.startswith(design + "_")]
    shown = (by_rank or campaign.structures)[:max(1, how_many)]

    columns = min(2, len(shown))
    rows = -(-len(shown) // columns)
    view = py3Dmol.view(viewergrid=(rows, columns), width=460 * columns, height=360 * rows)
    palette = ["#3c5b6f", "#B76E79", "#8FA98F", "#C9A227"]
    for index, path in enumerate(shown):
        where = (index // columns, index % columns)
        view.addModel(path.read_text(encoding="utf-8", errors="replace"), "cif", viewer=where)
        if colour_by == "pLDDT":
            # BC2 writes per-residue pLDDT on a 0-100 scale into the B-factor column.
            view.setStyle({}, {"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb",
                                                           "min": 50, "max": 90}}}, viewer=where)
        else:
            for number, chain in enumerate(chain_names(path) or ["A", "B"]):
                view.setStyle({"chain": chain},
                              {"cartoon": {"color": palette[number % len(palette)]}}, viewer=where)
        view.zoomTo(viewer=where)
    view.setBackgroundColor("white")
    return view, shown


# --------------------------------------------------------------------------- #
# 5. export and validation
# --------------------------------------------------------------------------- #

def export_campaign(campaign: Campaign, destination: Path | None = None, limit: int = 20,
                    make_zip: bool = True) -> dict:
    """Collect the result files worth keeping into one folder (and a zip).

    Written under the results tree by default, never into the source checkout,
    so a source update cannot delete an export.
    """
    folder = Path(destination) if destination else campaign.folder.parent / "exports" / campaign.folder.name
    folder.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    fasta = binder_fasta(campaign, limit=limit)
    if fasta:
        target = folder / f"{campaign.folder.name}_binders.fasta"
        # FASTA goes to Linux tools: write LF regardless of the host platform.
        target.write_text(fasta, encoding="utf-8", newline="\n")
        written.append(target.name)

    for relative in ("3_Ranked/!_Ranked.csv", "2_Refolded/!_Refolded.csv",
                     "1_Trajectories/!_Trajectories.csv", "summary.csv"):
        source = campaign.folder / relative
        if source.is_file():
            target = folder / (Path(relative).name.lstrip("!_") or Path(relative).name)
            shutil.copy2(source, target)
            written.append(target.name)
    for metadata in sorted(campaign.folder.glob("campaign_metadata*.json")):
        shutil.copy2(metadata, folder / metadata.name)
        written.append(metadata.name)

    accepted_designs = [(row.get("design") or "").strip() for row in campaign.ranked[:limit]]
    ranked_dir = campaign.folder / "3_Ranked"
    if ranked_dir.is_dir():
        for path in sorted(ranked_dir.glob("*.cif")) + sorted(ranked_dir.glob("*.html")):
            if not accepted_designs or any(path.stem == design or path.stem.startswith(design + "_")
                                           for design in accepted_designs):
                shutil.copy2(path, folder / path.name)
                written.append(path.name)

    result = {"folder": str(folder), "files": written,
              "sequences": fasta.count(">"), "zip": ""}
    if make_zip and written:
        archive = folder.parent / f"{campaign.folder.name}_export.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name in sorted(set(written)):
                bundle.write(folder / name, arcname=f"{campaign.folder.name}/{name}")
        result["zip"] = str(archive)
        result["zip_bytes"] = archive.stat().st_size
    return result


def download_links(paths):
    """Clickable links in Jupyter for files under the notebook's working tree."""
    from IPython.display import FileLink, FileLinks, display  # noqa: F401

    shown = []
    for path in paths:
        path = Path(path)
        try:
            relative = path.relative_to(Path.cwd())
        except ValueError:
            print(f"{path}  (outside the notebook directory: use the Jupyter file browser, "
                  "or scp it)")
            continue
        display(FileLink(str(relative)))
        shown.append(str(relative))
    return shown


def validate_campaign(campaign_folder, job_id: str = "", log_glob: str = "",
                      report_path: str = "") -> tuple[int, str]:
    """Run lab/scripts/validate_outputs.py and return (exit code, report text).

    Exit 0 nothing failed, 1 at least one FAIL, 2 the folder was unreadable.
    Level A needs explicit JAX GPU evidence from a job log, so pass a log.
    """
    interpreter = str(CONDA_PYTHON) if os.access(CONDA_PYTHON, os.X_OK) else sys.executable
    command = [interpreter, str(VALIDATOR), str(campaign_folder)]
    pattern = log_glob or (str(LOGS_ROOT / f"xulab_bc2_{job_id}.out") if job_id else "")
    if pattern:
        command += ["--slurm-log", pattern]
    if report_path:
        command += ["--output", str(report_path)]
    status, out, err = _run(command, cwd=SOURCE_TREE, timeout=600)
    return status, (out or "") + (("\n" + err) if err else "")
