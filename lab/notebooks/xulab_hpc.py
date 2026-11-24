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

# The source copy and the notebook live in HOME; everything a campaign writes
# lives in project storage. The two roots are independent, so the source tree can
# be replaced without touching results, and results can be moved without
# reinstalling the source.
PROJECT_ROOT = Path(os.environ.get("XULAB_PROJECT_ROOT", "/project/yli4/ypan/bindcraft"))
SOURCE_TREE = Path(os.environ.get("XULAB_BC2_HOME", "/home/ypan/BindCraft2/BindCraft2_deploy"))
RESULTS_ROOT = Path(os.environ.get("XULAB_RESULTS", str(PROJECT_ROOT / "results")))
LOGS_ROOT = Path(os.environ.get("XULAB_LOGS", str(PROJECT_ROOT / "logs")))
CACHE_ROOT = Path(os.environ.get("XULAB_CACHE", str(PROJECT_ROOT / "cache")))
TMP_ROOT = Path(os.environ.get("XULAB_TMP", str(PROJECT_ROOT / "tmp")))
CONDA_PYTHON = Path(os.environ.get("XULAB_PYTHON", "/home/ypan/.conda/envs/bindcraft2/bin/python"))
# Prepared target files, kept in project storage beside the results that cite
# them, so a target survives a source replacement and is reusable by later
# campaigns. Uploads and downloads land in subdirectories of it.
TARGETS_ROOT = Path(os.environ.get("XULAB_TARGETS", str(PROJECT_ROOT / "targets")))

# A free-text label for which deployment this process is driving. It is printed
# everywhere a path is printed, because the one mistake that costs real data is
# running a development campaign against the production project root. It is only
# a label: the paths above are what actually decide, and they come from the
# XULAB_* variables, never from this string.
DEPLOYMENT = os.environ.get("XULAB_DEPLOYMENT", "").strip() or "default"

SLURM_SCRIPT = SOURCE_TREE / "lab" / "slurm" / "run_bindcraft_h200.slurm"
VALIDATOR = SOURCE_TREE / "lab" / "scripts" / "validate_outputs.py"
GENERATED_CONFIGS = SOURCE_TREE / "lab" / "configs" / "generated"
PRESETS = SOURCE_TREE / "settings"
SCAFFOLDS = SOURCE_TREE / "scaffolds"
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

    rows.append(("deployment", "PASS", f"{DEPLOYMENT}  (set XULAB_DEPLOYMENT to label it)"))
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

    # The targets root is created on demand by the target step, so its absence is
    # not a reason to refuse a submission: an existing target elsewhere still works.
    if TARGETS_ROOT.is_dir():
        rows.append(("targets root", "PASS" if os.access(TARGETS_ROOT, os.W_OK) else "WARN",
                     str(TARGETS_ROOT)))
    else:
        rows.append(("targets root", "WARN", f"{TARGETS_ROOT} does not exist yet; the target "
                     "step creates it when it prepares a target"))

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
# 1b. structure files: reading, describing and preparing any target
#
# A binder campaign needs one thing from a structure: the antigen's own chain,
# with nothing else in the file. This section does that for any protein, not for
# one particular antigen. It reads PDB and mmCIF, reports what the file actually
# contains - identity, coverage, engineered differences, unplaced residues and
# how its numbering maps to a reference sequence - and writes the selection out
# with a provenance header.
#
# Three rules are enforced here rather than left to the caller:
#   * a chain is never picked for the user when the choice is ambiguous;
#   * nothing is modelled, mutated, renumbered or filled in;
#   * no claim is made about whether an epitope is reachable on a cell. The
#     coordinates of a soluble fragment cannot answer that, so this code reports
#     geometry and names what the geometry does not establish.
# --------------------------------------------------------------------------- #

RCSB_FILES = "https://files.rcsb.org/download"
RCSB_DATA = "https://data.rcsb.org/rest/v1/core"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
NETWORK_TIMEOUT = 30

# Residues that are polymer, so a selection keeps them, as three-letter codes.
# Everything else in a coordinate file - ligands, glycans, ions, water, modified
# residues - is a heteroatom and is reported and dropped rather than guessed at.
STANDARD_RESIDUES = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
                     "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                     "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
                     "TYR": "Y", "VAL": "V"}
ONE_TO_THREE = {code: name for name, code in STANDARD_RESIDUES.items()}


def structure_format(path) -> str:
    """"pdb", "cif", "fasta" or "" - decided by extension, then by content.

    A path that is not a readable file has no format: an extension alone is not
    taken as evidence about a file that is not there.
    """
    if not Path(path).is_file():
        return ""
    suffix = Path(path).suffix.lower()
    if suffix in (".pdb", ".ent"):
        return "pdb"
    if suffix in (".cif", ".mmcif"):
        return "cif"
    if suffix in (".fasta", ".fa", ".faa", ".seq"):
        return "fasta"
    try:
        head = Path(path).read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return ""
    if head.lstrip().startswith(">"):
        return "fasta"
    if "_atom_site." in head or head.lstrip().startswith("data_"):
        return "cif"
    if head.startswith(("HEADER", "ATOM  ", "HETATM", "REMARK", "TITLE")):
        return "pdb"
    return ""


def _pdb_atom(line: str) -> dict | None:
    try:
        return {"record": line[:6].strip(), "name": line[12:16].strip(), "altloc": line[16],
                "residue": line[17:20].strip(), "chain": line[21].strip() or " ",
                "number": int(line[22:26]), "insertion": line[26].strip(),
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
                "occupancy": line[54:60].strip(), "bfactor": line[60:66].strip(),
                "element": line[76:78].strip()}
    except (ValueError, IndexError):
        return None


def _cif_atoms(text: str) -> list:
    headers, atoms, in_loop = [], [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            headers.append(stripped.split(".", 1)[1])
            in_loop = True
            continue
        if not in_loop:
            continue
        if stripped.startswith(("ATOM", "HETATM")):
            fields = stripped.split()
            if len(fields) < len(headers):
                continue
            field = dict(zip(headers, fields))

            def value(*names, default=""):
                for name in names:
                    if field.get(name) not in (None, "", "?", "."):
                        return field[name]
                return default

            try:
                number = int(value("auth_seq_id", "label_seq_id", default="0"))
            except ValueError:
                continue
            try:
                coordinates = (float(field["Cartn_x"]), float(field["Cartn_y"]),
                               float(field["Cartn_z"]))
            except (KeyError, ValueError):
                continue
            atoms.append({"record": fields[0],
                          "name": value("label_atom_id", "auth_atom_id"),
                          "altloc": value("label_alt_id", default=" "),
                          "residue": value("auth_comp_id", "label_comp_id"),
                          "chain": value("auth_asym_id", "label_asym_id", default=" "),
                          "number": number,
                          "insertion": value("pdbx_PDB_ins_code", default=""),
                          "x": coordinates[0], "y": coordinates[1], "z": coordinates[2],
                          "occupancy": value("occupancy", default="1.00"),
                          "bfactor": value("B_iso_or_equiv", default="0.00"),
                          "element": value("type_symbol")})
        elif stripped and not stripped.startswith(("#", "loop_", "_", "data_")):
            in_loop = False
    return atoms


def structure_atoms(path) -> list:
    """Every atom of the first model, from a PDB or mmCIF file.

    Returns [] for an unreadable file or a format this cannot parse, so callers
    report "could not read" rather than treating emptiness as "no atoms".
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    kind = structure_format(path)
    if kind == "cif":
        return _cif_atoms(text)
    if kind != "pdb":
        return []
    atoms = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM  ", "HETATM")):
            atom = _pdb_atom(line)
            if atom:
                atoms.append(atom)
    return atoms


def residue_spans(numbers) -> list:
    """Contiguous runs in a sorted iterable of residue numbers."""
    spans: list = []
    for number in sorted(numbers):
        if spans and number == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], number)
        else:
            spans.append((number, number))
    return spans


def span_text(spans) -> str:
    return ", ".join(f"{start}-{stop}" if start != stop else str(start)
                     for start, stop in spans) or "none"


def sequence_differences(path) -> list:
    """Engineered mutations and expression tags the file itself declares.

    PDB `SEQADV` records and mmCIF `_struct_ref_seq_dif` both say where the
    deposited construct differs from its reference sequence. This reads them as
    written; it never infers a mutation by comparing sequences.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found: list = []
    if structure_format(path) == "pdb":
        for line in text.splitlines():
            if not line.startswith("SEQADV"):
                continue
            found.append({"chain": line[16:17].strip(), "residue": line[12:15].strip(),
                          "number": line[18:22].strip(),
                          "database": line[24:28].strip(), "accession": line[29:38].strip(),
                          "reference_residue": line[39:42].strip(),
                          "reference_number": line[43:48].strip(),
                          "note": line[49:].strip()})
        return found
    # mmCIF: the loop is short, so it is read by header name like the atom loop.
    headers, in_loop = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_struct_ref_seq_dif."):
            headers.append(stripped.split(".", 1)[1])
            in_loop = True
            continue
        if not in_loop:
            continue
        if stripped.startswith(("#", "loop_", "_", "data_")) or not stripped:
            if headers:
                in_loop = False
            continue
        fields = _cif_row(stripped)
        if len(fields) < len(headers):
            continue
        field = dict(zip(headers, fields))
        found.append({"chain": field.get("pdbx_pdb_strand_id", field.get("pdbx_PDB_strand_id", "")),
                      "residue": field.get("mon_id", ""),
                      "number": field.get("pdbx_auth_seq_num", field.get("seq_num", "")),
                      "database": field.get("pdbx_seq_db_name", ""),
                      "accession": field.get("pdbx_seq_db_accession_code", ""),
                      "reference_residue": field.get("db_mon_id", ""),
                      "reference_number": field.get("pdbx_seq_db_seq_num", ""),
                      "note": field.get("details", "").strip("'\"")})
    return found


def _cif_row(line: str) -> list:
    """Split one mmCIF loop row, honouring single-quoted fields."""
    fields, current, quote = [], "", ""
    for character in line:
        if quote:
            if character == quote:
                quote = ""
            else:
                current += character
        elif character in "'\"":
            quote = character
        elif character.isspace():
            if current:
                fields.append(current)
                current = ""
        else:
            current += character
    if current:
        fields.append(current)
    return fields


def declared_unobserved(path) -> dict:
    """{chain: [residue numbers]} the file declares as not located.

    PDB `REMARK 465` and mmCIF `_pdbx_unobs_or_zero_occ_residues`. This is what
    the depositors said; `chain_inventory` separately reports the gaps actually
    present in the coordinates, and the two are shown side by side rather than
    merged, because a disagreement between them is information.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    missing: dict = {}
    if structure_format(path) == "pdb":
        for line in text.splitlines():
            if not line.startswith("REMARK 465"):
                continue
            fields = line[10:].split()
            if len(fields) == 3 and len(fields[0]) == 3 and fields[0].isalpha():
                try:
                    missing.setdefault(fields[1], []).append(int(fields[2]))
                except ValueError:
                    continue
        return missing
    headers, in_loop = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_pdbx_unobs_or_zero_occ_residues."):
            headers.append(stripped.split(".", 1)[1])
            in_loop = True
            continue
        if not in_loop:
            continue
        if stripped.startswith(("#", "loop_", "_", "data_")) or not stripped:
            if headers:
                in_loop = False
            continue
        field = dict(zip(headers, _cif_row(stripped)))
        chain = field.get("auth_asym_id") or field.get("label_asym_id") or ""
        try:
            missing.setdefault(chain, []).append(int(field.get("auth_seq_id", "")))
        except ValueError:
            continue
    return missing


def chain_inventory(path) -> list:
    """What each polymer chain of a structure file actually contains.

    One entry per chain: residue count, observed range, the contiguous segments,
    the gaps inside the range, the one-letter sequence of what is present, the
    heteroatom residues sitting on that chain, and any insertion codes. This is
    the evidence a chain choice is made on.
    """
    atoms = structure_atoms(path)
    chains: dict = {}
    for atom in atoms:
        entry = chains.setdefault(atom["chain"], {"residues": {}, "hetero": {}, "insertions": set(),
                                                  "atoms": 0})
        entry["atoms"] += 1
        if atom["record"] == "ATOM" and atom["residue"] in STANDARD_RESIDUES:
            entry["residues"].setdefault(atom["number"], atom["residue"])
            if atom["insertion"]:
                entry["insertions"].add(f"{atom['number']}{atom['insertion']}")
        else:
            entry["hetero"][atom["residue"]] = entry["hetero"].get(atom["residue"], 0) + 1

    inventory = []
    for chain, entry in sorted(chains.items()):
        numbers = sorted(entry["residues"])
        if not numbers:
            inventory.append({"chain": chain, "residue_count": 0, "first": None, "last": None,
                              "segments": [], "gaps": [], "sequence": "",
                              "hetero": entry["hetero"], "insertion_codes": [],
                              "atoms": entry["atoms"], "polymer": False})
            continue
        segments = residue_spans(numbers)
        present = set(numbers)
        gaps = residue_spans([number for number in range(numbers[0], numbers[-1] + 1)
                              if number not in present])
        inventory.append({"chain": chain, "residue_count": len(numbers),
                          "first": numbers[0], "last": numbers[-1], "segments": segments,
                          "gaps": gaps,
                          "sequence": "".join(STANDARD_RESIDUES.get(entry["residues"][number], "X")
                                              for number in numbers),
                          "hetero": entry["hetero"],
                          "insertion_codes": sorted(entry["insertions"]),
                          "atoms": entry["atoms"], "polymer": True})
    return inventory


def polymer_chains(path) -> list:
    """The chain letters of a file that hold a real polymer, longest first."""
    return [entry["chain"] for entry in
            sorted((entry for entry in chain_inventory(path) if entry["residue_count"] >= 2),
                   key=lambda entry: -entry["residue_count"])]


def resolve_chain(path, chain: str = "") -> tuple:
    """(chain, note) - the chain to use, or a refusal to choose one.

    A file with exactly one polymer chain needs no choice. A file with several
    does, and this returns ("", reason) rather than taking the longest, because
    the longest chain of an antibody complex is usually the antibody.
    """
    chain = (chain or "").strip()
    inventory = {entry["chain"]: entry for entry in chain_inventory(path)}
    candidates = polymer_chains(path)
    if chain:
        if chain not in inventory:
            return "", (f"chain {chain!r} is not in this file; it holds "
                        f"{', '.join(candidates) or 'no polymer chain'}")
        if not inventory[chain]["polymer"]:
            return "", f"chain {chain!r} holds no polymer residues"
        return chain, (f"chain {chain} as requested, of "
                       f"{', '.join(sorted(candidates)) or 'no polymer chain'}")
    if not candidates:
        return "", "this file holds no polymer chain"
    if len(candidates) == 1:
        return candidates[0], f"chain {candidates[0]} is the only polymer chain in the file"
    described = "; ".join(f"{name} ({inventory[name]['residue_count']} residues)"
                          for name in sorted(candidates))
    return "", ("this file holds several polymer chains, so the antigen chain has to be named: "
                f"{described}. The longest chain is not assumed to be the target - in an "
                "antibody complex it usually is not.")


def geometry_notes(path, chain: str, residues) -> list:
    """What the coordinates do and do not say about a set of residues.

    Returns readable lines: the span of the selection, its distance from each
    end of the modelled chain, and the nearest N-glycosylation sequon present in
    the chain. Every line is a measurement on this file. None of them is an
    accessibility claim: a soluble fragment's coordinates cannot establish that a
    site is reachable on a cell, and the last line says so explicitly.
    """
    wanted = sorted({int(number) for number in residues})
    alpha = {}
    for atom in structure_atoms(path):
        if atom["chain"] == chain and atom["name"] == "CA" and atom["record"] == "ATOM":
            alpha.setdefault(atom["number"], (atom["x"], atom["y"], atom["z"]))
    placed = [number for number in wanted if number in alpha]
    notes: list[str] = []
    if not placed:
        return ["none of the named residues has a modelled CA atom in this chain, so no "
                "geometry can be reported for them"]
    absent = [number for number in wanted if number not in alpha]
    if absent:
        notes.append(f"not modelled, so excluded from these measurements: {span_text(residue_spans(absent))}")

    def distance(first, second):
        return sum((first[axis] - second[axis]) ** 2 for axis in range(3)) ** 0.5

    widest = max((distance(alpha[one], alpha[two]) for one in placed for two in placed),
                 default=0.0)
    notes.append(f"the {len(placed)} named residues span {widest:.0f} A at their widest "
                 f"CA-to-CA separation"
                 + ("" if widest <= 25 else ", which is wide for one contiguous site"))
    ordered = sorted(alpha)
    for label, end in (("first", ordered[0]), ("last", ordered[-1])):
        reach = [distance(alpha[number], alpha[end]) for number in placed]
        notes.append(f"CA distance to the {label} modelled residue of chain {chain} "
                     f"({end}): {min(reach):.0f}-{max(reach):.0f} A")
    inventory = {entry["chain"]: entry for entry in chain_inventory(path)}
    entry = inventory.get(chain, {})
    sequence, numbers = entry.get("sequence", ""), None
    if sequence:
        numbers = [number for number, _ in sorted(alpha.items())]
    sequons = []
    if sequence and numbers and len(sequence) == len(numbers):
        for index in range(len(sequence) - 2):
            if sequence[index] == "N" and sequence[index + 1] != "P" \
                    and sequence[index + 2] in "ST":
                sequons.append(numbers[index])
    if sequons:
        nearest = min(((min(distance(alpha[site], alpha[number]) for number in placed), site)
                       for site in sequons if site in alpha), default=None)
        if nearest:
            notes.append(f"nearest N-glycosylation sequon present in the modelled chain: "
                         f"N{nearest[1]}, {nearest[0]:.0f} A from the nearest named residue "
                         f"({len(sequons)} sequons in the chain)")
    else:
        notes.append("no N-X-S/T sequon in the modelled part of this chain")
    notes.append("NOT ESTABLISHED by these coordinates: whether this site is reachable on a "
                 "cell. Membrane proximity, partner-protein occlusion, glycan shielding and "
                 "expression context are not properties of a soluble fragment's coordinates. "
                 "Decide that from the biology and from structures of the full assembly.")
    return notes


def describe_structure(path, chain: str = "") -> dict:
    """Everything this code can say about a structure file, as data.

    {"path", "format", "chains": [...], "chain": resolved or "",
     "chain_note", "differences": [...], "declared_unobserved": {...},
     "problems": [...]}. Nothing here writes a file or chooses for the user.
    """
    path = Path(path)
    report = {"path": str(path), "format": structure_format(path), "chains": [],
              "chain": "", "chain_note": "", "differences": [], "declared_unobserved": {},
              "problems": []}
    if not path.is_file():
        report["problems"].append(f"not a file: {path}")
        return report
    if report["format"] == "fasta":
        records = read_fasta(path)
        report["chains"] = [{"chain": name, "residue_count": len(sequence),
                             "sequence": sequence, "first": 1, "last": len(sequence),
                             "segments": [(1, len(sequence))], "gaps": [], "hetero": {},
                             "insertion_codes": [], "polymer": True}
                            for name, sequence in records]
        report["chain_note"] = ("a FASTA target carries no coordinates, so there is no chain, "
                                "no numbering and no coverage to report: the campaign predicts "
                                "the target's own structure as part of the run")
        if not records:
            report["problems"].append("no FASTA record read from this file")
        return report
    if report["format"] not in ("pdb", "cif"):
        report["problems"].append(f"unrecognised structure format for {path.name}; this reads "
                                  ".pdb, .cif and .fasta")
        return report
    report["chains"] = chain_inventory(path)
    if not report["chains"]:
        report["problems"].append("no atoms read from this file")
        return report
    report["chain"], report["chain_note"] = resolve_chain(path, chain)
    report["differences"] = sequence_differences(path)
    report["declared_unobserved"] = declared_unobserved(path)
    return report


def read_fasta(path_or_text) -> list:
    """[(name, sequence)] from a FASTA file or a pasted FASTA/bare sequence."""
    text = path_or_text
    try:
        candidate = Path(str(path_or_text))
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass
    text = str(text or "")
    if not text.strip():
        return []
    if ">" not in text:
        sequence = re.sub(r"[^A-Za-z]", "", text).upper()
        return [("sequence", sequence)] if sequence else []
    records, name, parts = [], "", []
    for line in text.splitlines():
        if line.startswith(">"):
            if name or parts:
                records.append((name, re.sub(r"[^A-Za-z]", "", "".join(parts)).upper()))
            name, parts = line[1:].strip().split()[0] if line[1:].strip() else "sequence", []
        else:
            parts.append(line)
    if name or parts:
        records.append((name, re.sub(r"[^A-Za-z]", "", "".join(parts)).upper()))
    return [(name or "sequence", sequence) for name, sequence in records if sequence]


def _pdb_line(serial: int, atom: dict) -> str:
    name = atom["name"]
    formatted = f" {name:<3}" if len(name) < 4 else name[:4]
    element = atom.get("element") or (name[0] if name else "")
    try:
        occupancy = f"{float(atom.get('occupancy') or 1.0):6.2f}"
    except (TypeError, ValueError):
        occupancy = "  1.00"
    try:
        bfactor = f"{float(atom.get('bfactor') or 0.0):6.2f}"
    except (TypeError, ValueError):
        bfactor = "  0.00"
    return (f"ATOM  {serial:>5} {formatted}{' '}{atom['residue']:>3} {atom['chain']:>1}"
            f"{atom['number']:>4}{(atom['insertion'] or ' '):>1}   "
            f"{atom['x']:>8.3f}{atom['y']:>8.3f}{atom['z']:>8.3f}"
            f"{occupancy}{bfactor}          {element:>2}")


def prepare_target(source, chain: str = "", output=None, name: str = "",
                   first: int | None = None, last: int | None = None,
                   provenance: list | None = None) -> dict:
    """Write one chain of a structure file out as a clean PDB target.

    Keeps that chain's standard polymer residues over the requested residue range
    and nothing else: no other chain, no ligand, no glycan, no water, no
    expression tag beyond `last`. The deposited author numbering is preserved,
    residues the depositors did not place stay absent, and a provenance header
    records all of it inside the file.

    Returns a report dict with "output", "chain", "residue_count", "segments",
    "absent", "removed", "sequence", "differences", "problems". A refusal - an
    ambiguous chain, an unreadable file, data that does not fit PDB columns -
    comes back in "problems" with nothing written.
    """
    source = Path(source)
    report: dict = {"source": str(source), "output": "", "chain": "", "name": name,
                    "residue_count": 0, "segments": [], "absent": [], "removed": {},
                    "sequence": "", "differences": [], "problems": []}
    described = describe_structure(source, chain)
    report["problems"] += described["problems"]
    report["format"] = described["format"]
    if described["format"] == "fasta":
        report["problems"].append("this is a FASTA file: use write_fasta_target instead, which "
                                  "copies the sequence without pretending it has coordinates")
        return report
    if report["problems"]:
        return report
    if not described["chain"]:
        report["problems"].append(described["chain_note"])
        return report

    chain = described["chain"]
    report["chain"] = chain
    report["chain_note"] = described["chain_note"]
    report["differences"] = [difference for difference in described["differences"]
                            if not difference["chain"] or difference["chain"] == chain]

    atoms = structure_atoms(source)
    selected, removed = [], {}
    for atom in atoms:
        if atom["chain"] != chain:
            removed[f"chain {atom['chain']}"] = removed.get(f"chain {atom['chain']}", 0) + 1
            continue
        if atom["record"] != "ATOM" or atom["residue"] not in STANDARD_RESIDUES:
            label = f"heteroatom {atom['residue']}"
            removed[label] = removed.get(label, 0) + 1
            continue
        if atom["altloc"] not in ("", " ", ".", "A"):
            removed["alternate location"] = removed.get("alternate location", 0) + 1
            continue
        if first is not None and atom["number"] < int(first):
            label = f"chain {chain} residue < {first}"
            removed[label] = removed.get(label, 0) + 1
            continue
        if last is not None and atom["number"] > int(last):
            label = f"chain {chain} residue > {last}"
            removed[label] = removed.get(label, 0) + 1
            continue
        selected.append(atom)

    if not selected:
        report["problems"].append(f"no standard polymer atoms left in chain {chain} over the "
                                  f"requested range; nothing was written")
        report["removed"] = removed
        return report

    unfit = [atom for atom in selected
             if len(atom["chain"]) != 1 or not -999 <= atom["number"] <= 9999]
    if unfit:
        report["problems"].append(
            f"{len(unfit)} atom(s) of chain {chain!r} do not fit the fixed PDB columns "
            "(a chain id longer than one character, or a residue number outside -999..9999). "
            "Nothing was written: use the original mmCIF as the target_path instead of a "
            "converted copy.")
        return report
    if len(selected) > 99999:
        report["problems"].append(f"{len(selected)} atoms exceed the PDB serial-number column; "
                                  "use the original mmCIF as the target_path instead")
        return report

    numbers = sorted({atom["number"] for atom in selected})
    present = set(numbers)
    report["residue_count"] = len(numbers)
    report["segments"] = residue_spans(numbers)
    report["absent"] = residue_spans([number for number in range(numbers[0], numbers[-1] + 1)
                                      if number not in present])
    residues = {}
    for atom in selected:
        residues.setdefault(atom["number"], atom["residue"])
    report["sequence"] = "".join(STANDARD_RESIDUES.get(residues[number], "X")
                                 for number in numbers)
    report["removed"] = removed

    label = name.strip() or f"{source.stem}_{chain}"
    destination = Path(output) if output else TARGETS_ROOT / f"{label}.pdb"
    header = [f"REMARK 999 Target prepared by xulab_hpc.prepare_target from {source.name}.",
              f"REMARK 999 Selection: chain {chain}, residues "
              f"{first if first is not None else numbers[0]}-"
              f"{last if last is not None else numbers[-1]} of the source model.",
              "REMARK 999 The source file's author numbering is preserved unchanged. No",
              "REMARK 999 coordinate was modelled, mutated, renumbered, minimised or added.",
              f"REMARK 999 Residues present : {len(numbers)}",
              f"REMARK 999 Ordered segments : {span_text(report['segments'])}",
              f"REMARK 999 Absent in source : {span_text(report['absent'])}"]
    for item, count in sorted(removed.items()):
        header.append(f"REMARK 999 Removed          : {item} ({count} atoms)")
    for difference in report["differences"][:20]:
        header.append(f"REMARK 999 Source SEQADV    : {difference['residue']}"
                      f"{difference['number']} vs {difference['reference_residue']}"
                      f"{difference['reference_number']} "
                      f"{difference['database']} {difference['accession']} "
                      f"{difference['note']}".rstrip())
    header += [f"REMARK 999 {line}" for line in (provenance or [])]

    body = list(header)
    for serial, atom in enumerate(selected, start=1):
        body.append(_pdb_line(serial, atom))
    final = selected[-1]
    body.append(f"TER   {len(selected) + 1:>5}      {final['residue']:>3} {final['chain']:>1}"
                f"{final['number']:>4}")
    body.append("END")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # The target is read on Linux: LF regardless of the host platform.
        destination.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
    except OSError as failure:
        report["problems"].append(f"could not write {destination}: {failure}")
        return report
    report["output"] = str(destination)
    report["name"] = label
    return report


def write_fasta_target(name: str, sequence_or_fasta: str, directory=None) -> dict:
    """Write a pasted or uploaded sequence out as a FASTA target.

    BindCraft2 accepts a FASTA target and predicts the target's own structure as
    part of the campaign. Nothing is validated beyond the alphabet: a sequence
    that is not protein is reported, not silently accepted.
    """
    records = read_fasta(sequence_or_fasta)
    report = {"output": "", "name": name.strip(), "records": records, "problems": []}
    if not records:
        report["problems"].append("no sequence found in that text")
        return report
    unknown = sorted({letter for _, sequence in records for letter in sequence
                      if letter not in ONE_TO_THREE})
    if unknown:
        report["problems"].append("not a protein sequence: it carries the non-amino-acid "
                                  f"letter(s) {', '.join(unknown)}")
        return report
    label = name.strip() or records[0][0] or "target"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    destination = Path(directory or TARGETS_ROOT) / f"{label}.fasta"
    text = "\n".join(f">{record_name}\n{sequence}" for record_name, sequence in records) + "\n"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
    except OSError as failure:
        report["problems"].append(f"could not write {destination}: {failure}")
        return report
    report["output"] = str(destination)
    report["name"] = label
    report["residue_count"] = sum(len(sequence) for _, sequence in records)
    return report


def save_upload(filename: str, content: bytes, directory=None) -> dict:
    """Store an uploaded structure or FASTA under the targets root.

    The uploaded bytes are written unchanged into `targets/uploads/`, so the
    original is always still there to re-derive a target from. Nothing is
    executed and nothing is unpacked: only these extensions are accepted.
    """
    allowed = {".pdb", ".ent", ".cif", ".mmcif", ".fasta", ".fa", ".faa", ".seq", ".txt"}
    report = {"output": "", "problems": []}
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(filename)).name) or "upload"
    if Path(safe).suffix.lower() not in allowed:
        report["problems"].append(f"{safe}: this step accepts {', '.join(sorted(allowed))}")
        return report
    destination = Path(directory or (TARGETS_ROOT / "uploads")) / safe
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    except OSError as failure:
        report["problems"].append(f"could not write {destination}: {failure}")
        return report
    report["output"] = str(destination)
    return report


# --------------------------------------------------------------------------- #
# 1c. RCSB PDB lookup
#
# Every function here returns (data, error) and never raises: the HPC compute
# nodes have no internet access and institutional networks block outbound HTTP
# often enough that "no network" has to be an ordinary, readable outcome rather
# than a traceback. `opener` exists so the tests can run without a network.
# --------------------------------------------------------------------------- #

def _fetch(url: str, payload: dict | None = None, opener=None) -> tuple:
    """(text, error). A POST when `payload` is given, otherwise a GET."""
    import urllib.error
    import urllib.request

    opener = opener or urllib.request.urlopen
    try:
        if payload is None:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
        else:
            request = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "application/json"})
        with opener(request, timeout=NETWORK_TIMEOUT) as response:
            return response.read().decode("utf-8", errors="replace"), ""
    except urllib.error.HTTPError as failure:
        return "", f"{url}: HTTP {failure.code} {failure.reason}"
    except urllib.error.URLError as failure:
        return "", (f"no network access to {url.split('/')[2]}: {failure.reason}. "
                    "Prepare the target on a machine with internet access and copy the file "
                    "across, or point this step at a file already on the cluster.")
    except (OSError, ValueError) as failure:                         # noqa: BLE001
        return "", f"{url}: {failure}"


def _fetch_json(url: str, payload: dict | None = None, opener=None) -> tuple:
    text, error = _fetch(url, payload, opener)
    if error:
        return {}, error
    try:
        return json.loads(text), ""
    except ValueError as failure:
        return {}, f"{url}: response was not JSON ({failure})"


def looks_like_pdb_id(text: str) -> bool:
    return bool(re.fullmatch(r"[0-9][A-Za-z0-9]{3}", (text or "").strip()))


def looks_like_uniprot(text: str) -> bool:
    text = (text or "").strip().upper()
    return bool(re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}",
                             text))


def rcsb_search(text: str, limit: int = 10, opener=None) -> tuple:
    """([pdb id, ...], error) for a protein name, a UniProt accession or a PDB ID.

    A UniProt accession is searched as an accession, so it returns the entries
    that actually reference that protein rather than entries whose text happens
    to contain the string. Anything else is a full-text search.
    """
    text = (text or "").strip()
    if not text:
        return [], "type a protein name, a UniProt accession or a PDB ID"
    if looks_like_pdb_id(text):
        return [text.upper()], ""
    if looks_like_uniprot(text):
        query = {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity_container_identifiers."
                         "reference_sequence_identifiers.database_accession",
            "operator": "exact_match", "value": text.upper()}}
    else:
        query = {"type": "terminal", "service": "full_text", "parameters": {"value": text}}
    payload = {"query": query, "return_type": "entry",
               "request_options": {"paginate": {"start": 0, "rows": max(1, int(limit))},
                                   "results_content_type": ["experimental"],
                                   "sort": [{"sort_by": "score", "direction": "desc"}]}}
    data, error = _fetch_json(RCSB_SEARCH, payload, opener)
    if error:
        return [], error
    found = [str(item.get("identifier", "")).upper()
             for item in data.get("result_set", []) if item.get("identifier")]
    if not found:
        return [], f"RCSB found no experimental structure for {text!r}"
    return found, ""


def rcsb_entry_report(pdb_id: str, opener=None) -> tuple:
    """(report, error) describing one entry and each of its polymer entities.

    The report carries, per entity: description, author chain ids, length,
    source organism, the UniProt accession SIFTS maps it to, and the alignment
    offset between the entity's own numbering and that reference sequence. That
    offset is the residue-number mapping the campaign has to be read with.
    """
    pdb_id = (pdb_id or "").strip().upper()
    if not looks_like_pdb_id(pdb_id):
        return {}, f"{pdb_id!r} is not a 4-character PDB ID"
    entry, error = _fetch_json(f"{RCSB_DATA}/entry/{pdb_id}", opener=opener)
    if error:
        return {}, error
    info = entry.get("rcsb_entry_info", {})
    resolution = info.get("resolution_combined") or []
    report = {"pdb_id": pdb_id,
              "title": (entry.get("struct") or {}).get("title", ""),
              "methods": [item.get("method", "") for item in entry.get("exptl", [])],
              "resolution": resolution[0] if resolution else None,
              "released": (entry.get("rcsb_accession_info") or {}).get("initial_release_date", ""),
              "citation": "", "entities": [], "problems": []}
    citations = entry.get("citation") or []
    if citations:
        first = citations[0]
        report["citation"] = " ".join(str(part) for part in
                                      (first.get("title"), first.get("journal_abbrev"),
                                       first.get("year")) if part)
    for entity_id in (entry.get("rcsb_entry_container_identifiers") or {}).get(
            "polymer_entity_ids", []):
        data, entity_error = _fetch_json(f"{RCSB_DATA}/polymer_entity/{pdb_id}/{entity_id}",
                                         opener=opener)
        if entity_error:
            report["problems"].append(f"entity {entity_id}: {entity_error}")
            continue
        identifiers = data.get("rcsb_polymer_entity_container_identifiers") or {}
        polymer = data.get("entity_poly") or {}
        organisms = [item.get("scientific_name", "") for item
                     in data.get("rcsb_entity_source_organism") or []]
        alignment = []
        for align in data.get("rcsb_polymer_entity_align") or []:
            for region in align.get("aligned_regions") or []:
                alignment.append({"database": align.get("reference_database_name", ""),
                                  "accession": align.get("reference_database_accession", ""),
                                  "entity_start": region.get("entity_beg_seq_id"),
                                  "reference_start": region.get("ref_beg_seq_id"),
                                  "length": region.get("length")})
        report["entities"].append({
            "entity_id": str(entity_id),
            "description": (data.get("rcsb_polymer_entity") or {}).get("pdbx_description", ""),
            "chains": identifiers.get("auth_asym_ids") or identifiers.get("asym_ids") or [],
            "length": polymer.get("rcsb_sample_sequence_length"),
            "sequence": polymer.get("pdbx_seq_one_letter_code_can", ""),
            "organisms": [name for name in organisms if name],
            "uniprot": identifiers.get("uniprot_ids") or [],
            "alignment": alignment})
    if not report["entities"]:
        report["problems"].append("the entry reports no polymer entity")
    return report, ""


def rcsb_download(pdb_id: str, directory=None, opener=None) -> tuple:
    """(path, error). Downloads the PDB file, falling back to mmCIF.

    Large entries have no PDB-format file at all, which is why the fallback
    exists rather than being treated as a failure.
    """
    pdb_id = (pdb_id or "").strip().upper()
    if not looks_like_pdb_id(pdb_id):
        return "", f"{pdb_id!r} is not a 4-character PDB ID"
    folder = Path(directory or (TARGETS_ROOT / "rcsb"))
    errors = []
    for suffix in (".pdb", ".cif"):
        text, error = _fetch(f"{RCSB_FILES}/{pdb_id}{suffix}", opener=opener)
        if error:
            errors.append(error)
            continue
        destination = folder / f"{pdb_id}{suffix}"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8", newline="\n")
        except OSError as failure:
            return "", f"could not write {destination}: {failure}"
        return str(destination), ""
    return "", "; ".join(errors)


def entity_of_chain(report: dict, chain: str) -> dict:
    """The entity of an `rcsb_entry_report` that owns a given author chain."""
    for entity in report.get("entities", []):
        if chain in (entity.get("chains") or []):
            return entity
    return {}


def numbering_mapping(entity: dict) -> list:
    """Readable lines mapping an entity's own numbering to its reference sequence.

    SIFTS gives the alignment as (entity start, reference start, length). The
    offset between them is what turns a residue number in the coordinate file
    into a residue number in, say, UniProt - which is the only way a hotspot
    list can be compared with the literature.
    """
    lines = []
    for align in entity.get("alignment", []):
        entity_start = align.get("entity_start")
        reference_start = align.get("reference_start")
        length = align.get("length")
        if not all(isinstance(value, int) for value in (entity_start, reference_start, length)):
            continue
        offset = reference_start - entity_start
        lines.append(
            f"entity residues {entity_start}-{entity_start + length - 1} map to "
            f"{align.get('database', '?')} {align.get('accession', '?')} "
            f"{reference_start}-{reference_start + length - 1} "
            f"(offset {offset:+d}), per SIFTS")
    if not lines:
        lines.append("RCSB reports no reference-sequence alignment for this entity, so no "
                     "mapping to a database numbering can be stated")
    lines.append("The coordinate file's own author numbering is what this platform writes and "
                 "what hotspots must be given in; it is not always the same as the entity "
                 "numbering above. chain_inventory() prints the author numbering actually "
                 "present in the file.")
    return lines


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
    # Which deployment submitted this, so a registry entry can never be mistaken
    # for one belonging to the other tree.
    deployment: str = ""
    source_tree: str = ""

    def as_dict(self) -> dict:
        return dict(job_id=self.job_id, campaign=self.campaign, config=self.config,
                    project_folder=self.project_folder, submitted_at=self.submitted_at,
                    command=self.command, log_out=self.log_out, log_err=self.log_err,
                    message=self.message, deployment=self.deployment,
                    source_tree=self.source_tree)


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
                    nodelist: str = "", extra_sbatch=(), dry_run: bool = False) -> Submission:
    """sbatch the H200 launcher for one settings file.

    The campaign runs entirely inside the batch job, executed directly by the
    batch script with no srun step (srun fails on this cluster with GRES errors).
    This notebook kernel does no design work, so closing the notebook leaves the
    job running.

    `nodelist` overrides the node the launcher pins by default (cmri-gpu05, the
    node job 1015050 completed on); a command-line flag beats the directive in
    the script. Leave it empty to use that default.

    The launcher is never edited, including for an isolated deployment. It reads
    `XULAB_BC2_HOME` and `XULAB_PROJECT_ROOT` from the environment and falls back
    to the production paths, and its `--output`/`--error` directives name the
    production log directory. So this function states all three explicitly on
    every submission: the source tree and project root as `--export`, and the log
    paths as `--output`/`--error`, where an sbatch flag beats a directive in the
    script. For the production deployment the values are identical to the
    launcher's own, and for a development deployment nothing can leak into the
    production tree by default.
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
    # XULAB_BC2_HOME decides which source tree the job executes; XULAB_PROJECT_ROOT
    # decides where results, caches and tmp go. Both are stated, never inherited.
    exports = ["ALL", f"XULAB_BC2_HOME={SOURCE_TREE}", f"XULAB_PROJECT_ROOT={PROJECT_ROOT}"]
    if parameters:
        exports.append(f"BINDCRAFT_AF2_PARAMS={parameters}")
    # The shared weight cache is reused where one is already set, rather than a
    # second copy being created under a development project root.
    weights = os.environ.get("BINDCRAFT_WEIGHTS", "")
    if weights:
        exports.append(f"BINDCRAFT_WEIGHTS={weights}")

    # SLURM opens --output before the job starts, so the directory must exist now.
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)

    command = ["sbatch", "--parsable", f"--export={','.join(exports)}",
               f"--output={LOGS_ROOT / 'xulab_bc2_%j.out'}",
               f"--error={LOGS_ROOT / 'xulab_bc2_%j.err'}"]
    if account.strip():
        command.append(f"--account={account.strip()}")
    if nodelist.strip():
        command.append(f"--nodelist={nodelist.strip()}")
    command += [str(part) for part in extra_sbatch]
    command += [str(SLURM_SCRIPT), str(config_path)]

    submission = Submission(campaign=campaign, config=str(config_path),
                            project_folder=project_folder,
                            submitted_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            command=command, deployment=DEPLOYMENT,
                            source_tree=str(SOURCE_TREE))
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
# 4b. antibody-format designs: VH / VL identity and loop annotation
#
# Everything here is derived from two files that ship with the source tree - the
# scaffold the campaign designed on (scaffolds/scFv.cif) and the modality preset
# that declares its paratope (settings/modality/scFv.json) - and from the design's
# own output row. Nothing is inferred from a bare sequence: a domain is called VH
# or VL only when its framework matches the scaffold chain, and a loop is reported
# only when the framework anchors on both sides of it are found. Anything that
# cannot be established that way is reported as unavailable.
# --------------------------------------------------------------------------- #

AMINO_ACID_CODES = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
                    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
                    "TYR": "Y", "VAL": "V"}

# chain letter in the antibody scaffold -> variable domain it is
ANTIBODY_DOMAINS = {"H": "VH", "L": "VL"}
ANTIBODY_MODALITIES = ("scFv", "Fab", "VHH")

# One span in a mutate_positions string: "L30-32", "H52-55(4-6)", "H72*", "L24-29!".
_SPAN_PATTERN = re.compile(r"(?P<chain>[A-Za-z]+)(?P<start>\d+)(?:-(?P<end>\d+))?"
                           r"(?:\((?P<lengths>[^)]+)\))?(?P<flags>(?:[+-][A-Za-z]+|[*!])*)")

# How many framework residues on each side of a loop are used to locate it.
ANCHOR_LENGTH = 6

NUMBERING_CONVENTION = """\
Loop spans are NOT inferred from the designed sequence. They are the paratope
spans the shipped modality preset declares in `mutate_positions`, written in the
residue numbering of the scaffold file the campaign designed on
(`scaffolds/<modality>.cif`, chains H and L, numbered from 1), and they are
located inside each design by exact framework anchors of up to six residues on
each side - framework positions the preset never redesigns. A loop whose anchors
are not both found exactly once is reported as unavailable rather than guessed.

For the shipped scFv preset those spans are L24-34, L50-56, L89-97 and H26-35,
H50-65, H95-106. L1/L2/L3 and H2 coincide with the Kabat CDR definitions; H1
(26-35) is the Chothia-Kabat union; H95-106 begins at the conserved Cys-1 stem,
so it carries four held stem residues (`YCAR` in the scaffold) ahead of the loop
proper. The separately reported `HCDR3 (Kabat)` is the classical definition -
the residues between the conserved Cys stem and the W-G-x-G motif, both located
in the design itself - so it is the number to quote as the HCDR3 length."""


def cif_chain_residues(path) -> dict:
    """{chain: [(residue number, one-letter code), ...]} from an mmCIF, as text.

    Reads the _atom_site loop by header name, so it does not depend on column
    order, and needs no parser library.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    headers: list[str] = []
    chains: dict = {}
    in_loop = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            headers.append(stripped.split(".", 1)[1])
            in_loop = True
            continue
        if not in_loop:
            continue
        if stripped.startswith(("ATOM", "HETATM")):
            fields = stripped.split()
            if len(fields) < len(headers):
                continue
            chain = number = residue = None
            for name in ("auth_asym_id", "label_asym_id"):
                if name in headers:
                    chain = fields[headers.index(name)]
                    break
            for name in ("auth_seq_id", "label_seq_id"):
                if name in headers:
                    try:
                        number = int(fields[headers.index(name)])
                    except ValueError:
                        number = None
                    if number is not None:
                        break
            for name in ("auth_comp_id", "label_comp_id"):
                if name in headers:
                    residue = fields[headers.index(name)]
                    break
            if chain is None or number is None or residue is None:
                continue
            seen = chains.setdefault(chain, {})
            seen.setdefault(number, AMINO_ACID_CODES.get(residue.upper(), "X"))
        elif stripped and not stripped.startswith(("#", "loop_", "_", "data_")):
            in_loop = False
    return {chain: [(number, seen[number]) for number in sorted(seen)]
            for chain, seen in chains.items()}


def paratope_spans(edit_string: str) -> dict:
    """{chain letter: [(start, end, editable), ...]} from a mutate_positions string.

    `editable` is False for a span written with `!` (held unredesigned) and for a
    span written with `*` (a framework position, not paratope at all). The `*`
    spans are dropped: they are framework, so they belong in the anchors.
    """
    spans: dict = {}
    for match in _SPAN_PATTERN.finditer(edit_string or ""):
        flags = match.group("flags") or ""
        if "*" in flags:
            continue
        start = int(match.group("start"))
        end = int(match.group("end")) if match.group("end") else start
        spans.setdefault(match.group("chain"), []).append(
            (min(start, end), max(start, end), "!" not in flags))
    return {chain: sorted(found) for chain, found in spans.items()}


def designed_positions(edit_string: str) -> dict:
    """{chain letter: {positions the preset redesigns}}, `*` framework included.

    A position the preset may change cannot serve as an anchor, so this is what
    the anchor builder avoids.
    """
    positions: dict = {}
    for match in _SPAN_PATTERN.finditer(edit_string or ""):
        flags = match.group("flags") or ""
        if "!" in flags:
            continue
        start = int(match.group("start"))
        end = int(match.group("end")) if match.group("end") else start
        positions.setdefault(match.group("chain"), set()).update(
            range(min(start, end), max(start, end) + 1))
    return positions


def merged_loops(spans: list) -> list:
    """Adjacent or overlapping paratope spans joined into one loop each."""
    loops: list = []
    for start, end, _editable in spans:
        if loops and start <= loops[-1][1] + 1:
            loops[-1] = (loops[-1][0], max(loops[-1][1], end))
        else:
            loops.append((start, end))
    return loops


def scaffold_reference(modality: str = "scFv") -> dict:
    """What a scaffolded modality's loops are, read off the preset and scaffold.

    {chain letter: {"domain", "sequence", "first", "loops": [...], "anchors": ...}}
    with one loop entry per declared paratope loop, named LCDR1..HCDR3 in the
    order they occur. Raises RuntimeError naming the missing file, because an
    annotation built on a guessed scaffold would be worthless.
    """
    preset_path = PRESETS / "modality" / f"{modality}.json"
    try:
        preset = json.loads(preset_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as unreadable:
        raise RuntimeError(f"cannot read the {modality} preset at {preset_path}: "
                           f"{unreadable}") from unreadable
    scaffold_name = preset.get("binder_scaffold")
    if not scaffold_name:
        raise RuntimeError(f"the {modality} preset declares no binder_scaffold, so it has no "
                           "framework to annotate against")
    scaffold = SOURCE_TREE / scaffold_name if not Path(scaffold_name).is_absolute() \
        else Path(scaffold_name)
    if not scaffold.is_file():
        scaffold = SCAFFOLDS / Path(scaffold_name).name
    residues = cif_chain_residues(scaffold)
    if not residues:
        raise RuntimeError(f"no residues read from the {modality} scaffold {scaffold}")

    spans = paratope_spans(str(preset.get("mutate_positions") or ""))
    editable = designed_positions(str(preset.get("mutate_positions") or ""))
    reference: dict = {}
    for chain, numbered in residues.items():
        if chain not in spans:
            continue
        numbers = [number for number, _ in numbered]
        sequence = "".join(code for _, code in numbered)
        at = {number: index for index, number in enumerate(numbers)}
        domain = ANTIBODY_DOMAINS.get(chain, chain)
        loops = []
        for order, (start, end) in enumerate(merged_loops(spans[chain]), start=1):
            if start not in at or end not in at:
                continue
            loops.append({"name": f"{domain[-1]}CDR{order}", "chain": chain,
                          "span": f"{chain}{start}-{end}", "start": start, "end": end,
                          "scaffold_sequence": sequence[at[start]:at[end] + 1],
                          "editable_positions": sorted(number for number
                                                       in editable.get(chain, set())
                                                       if start <= number <= end)})
        reference[chain] = {"domain": domain, "chain": chain, "sequence": sequence,
                            "numbers": numbers, "loops": loops,
                            "scaffold": str(scaffold),
                            "fixed": [number for number in numbers
                                      if number not in editable.get(chain, set())]}
    if not reference:
        raise RuntimeError(f"the {modality} preset names chains {sorted(spans)} but the scaffold "
                           f"{scaffold} holds {sorted(residues)}")
    return reference


def _fixed_runs(reference: dict, start: int, end: int) -> list:
    """Runs of consecutive never-redesigned residues over [start, end], in order."""
    at = {number: index for index, number in enumerate(reference["numbers"])}
    fixed = set(reference["fixed"])
    runs: list[list[str]] = []
    for number in range(start, end + 1):
        if number in at and number in fixed:
            if not runs or runs[-1] is None:
                runs.append([])
            runs[-1].append(reference["sequence"][at[number]])
        elif runs and runs[-1] is not None:
            runs.append(None)
    return ["".join(run) for run in runs if run]


def _anchor_before(reference: dict, start: int, end: int) -> str:
    """The last ANCHOR_LENGTH framework residues immediately before a loop."""
    runs = _fixed_runs(reference, start, end)
    return runs[-1][-ANCHOR_LENGTH:] if runs else ""


def _anchor_after(reference: dict, start: int, end: int) -> str:
    """The first ANCHOR_LENGTH framework residues immediately after a loop."""
    runs = _fixed_runs(reference, start, end)
    return runs[0][:ANCHOR_LENGTH] if runs else ""


def domain_anchors(reference: dict) -> list:
    """[(before, loop, after), ...]: the framework anchors flanking each loop."""
    numbers = reference["numbers"]
    anchors = []
    for index, loop in enumerate(reference["loops"]):
        previous_end = reference["loops"][index - 1]["end"] if index else numbers[0] - 1
        next_start = (reference["loops"][index + 1]["start"]
                      if index + 1 < len(reference["loops"]) else numbers[-1] + 1)
        anchors.append((_anchor_before(reference, previous_end + 1, loop["start"] - 1), loop,
                        _anchor_after(reference, loop["end"] + 1, next_start - 1)))
    return anchors


def annotate_domain(sequence: str, reference: dict) -> dict:
    """Locate a reference's loops inside one designed domain sequence.

    Returns {"domain", "sequence", "length", "loops": [...], "notes": [...]}.
    A loop whose anchors are not both found, exactly once each and in order, is
    returned with sequence None and a reason, never with a guessed span.
    """
    sequence = (sequence or "").strip().upper()
    result = {"domain": reference["domain"], "sequence": sequence, "length": len(sequence),
              "loops": [], "notes": []}
    cursor = 0
    for before, loop, after in domain_anchors(reference):
        entry = {"name": loop["name"], "scaffold_span": loop["span"],
                 "scaffold_sequence": loop["scaffold_sequence"],
                 "editable_positions": loop["editable_positions"],
                 "sequence": None, "length": None, "start": None, "end": None,
                 "unavailable": ""}
        if not before or not after:
            entry["unavailable"] = "the preset leaves no framework anchor on one side"
        elif sequence.count(before) != 1 or sequence.count(after) != 1:
            entry["unavailable"] = (f"framework anchor {before!r} appears "
                                    f"{sequence.count(before)}x and {after!r} "
                                    f"{sequence.count(after)}x; one each is required")
        else:
            opening = sequence.find(before, cursor)
            closing = sequence.find(after, opening + len(before) if opening >= 0 else cursor)
            if opening < 0 or closing < 0:
                entry["unavailable"] = "framework anchors are not in the expected order"
            else:
                start = opening + len(before)
                entry.update(sequence=sequence[start:closing], length=closing - start,
                             start=start + 1, end=closing)
                cursor = closing
        result["loops"].append(entry)
    located = [loop for loop in result["loops"] if loop["sequence"] is not None]
    if len(located) != len(result["loops"]):
        result["notes"].append(f"{len(located)} of {len(result['loops'])} loops located by "
                               "framework anchors; the rest are reported as unavailable")
    return result


def kabat_hcdr3(sequence: str) -> dict:
    """CDR-H3 by the classical definition, located in the sequence itself.

    The loop runs from three residues after the conserved Cys of the `C-x-x`
    stem to the residue before the `W-G-x-G` motif. Both anchors are found in
    this sequence; if either is absent the result says so and carries no span.
    """
    sequence = (sequence or "").strip().upper()
    motif = None
    for match in re.finditer(r"W[GA].G", sequence):
        motif = match
    if motif is None:
        return {"sequence": None, "length": None,
                "unavailable": "no W-G-x-G framework motif in this sequence"}
    cysteine = sequence.rfind("C", 0, motif.start())
    if cysteine < 0:
        return {"sequence": None, "length": None,
                "unavailable": "no Cys stem before the W-G-x-G motif"}
    start, end = cysteine + 3, motif.start()
    if end <= start:
        return {"sequence": None, "length": None,
                "unavailable": "the Cys stem and the W-G-x-G motif leave no loop between them"}
    return {"sequence": sequence[start:end], "length": end - start,
            "start": start + 1, "end": end, "stem": sequence[cysteine:cysteine + 3],
            "motif": motif.group(0), "unavailable": ""}


def _similarity(sequence: str, reference: dict) -> int:
    """How many of a reference's framework anchors occur in a sequence."""
    sequence = (sequence or "").upper()
    anchors = {text for before, _loop, after in domain_anchors(reference)
               for text in (before, after) if text}
    return sum(1 for text in anchors if text in sequence)


def assign_domains(chain_sequences: list, reference: dict) -> list:
    """Match designed binder chains to scaffold domains by their frameworks.

    `chain_sequences` is the Binder_Sequence chains in output order. Each is
    assigned to the reference domain whose framework anchors it carries most of,
    so VH and VL are identified by evidence rather than by output order. A chain
    that matches nothing is returned as an unidentified domain.
    """
    unclaimed = dict(reference)
    assigned: list = []
    for position, sequence in enumerate(chain_sequences):
        best_chain, best_score = "", 0
        for chain, entry in unclaimed.items():
            score = _similarity(sequence, entry)
            if score > best_score:
                best_chain, best_score = chain, score
        if not best_chain:
            assigned.append({"domain": f"chain {position + 1} (unidentified)",
                             "sequence": sequence, "length": len(sequence), "loops": [],
                             "notes": ["no scaffold framework anchor found in this chain, so it "
                                       "is not called VH or VL"],
                             "chain_position": position + 1})
            continue
        entry = unclaimed.pop(best_chain)
        annotated = annotate_domain(sequence, entry)
        annotated["chain_position"] = position + 1
        annotated["scaffold_chain"] = best_chain
        annotated["anchors_matched"] = f"{best_score} framework anchors matched"
        assigned.append(annotated)
    return assigned


def split_paratope(paratope: str, domain: dict) -> dict:
    """Which contacting residues of one designed domain fall in a loop.

    `paratope` is one chain's part of Interface_Binder_Residues, as `K58,Y59`.
    Each label is checked against the domain sequence at that position before it
    is used: a label that does not match is listed as unmapped rather than
    assigned to a loop, because a silent off-by-one would mislabel the paratope.
    """
    sequence = domain.get("sequence") or ""
    in_loop: dict = {}
    framework, unmapped = [], []
    for label in (paratope or "").split(","):
        label = label.strip()
        match = re.fullmatch(r"([A-Z])(\d+)", label)
        if not match:
            if label:
                unmapped.append(label)
            continue
        code, position = match.group(1), int(match.group(2))
        if not (1 <= position <= len(sequence)) or sequence[position - 1] != code:
            unmapped.append(label)
            continue
        for loop in domain.get("loops", []):
            if loop.get("start") and loop["start"] <= position <= loop["end"]:
                in_loop.setdefault(loop["name"], []).append(label)
                break
        else:
            framework.append(label)
    return {"loops": in_loop, "framework": framework, "unmapped": unmapped,
            "contacts": sum(len(found) for found in in_loop.values()) + len(framework)}


def is_antibody_campaign(campaign: Campaign) -> str:
    """The antibody modality a campaign ran, or "" - read, never assumed.

    BindCraft2 records the modality in three independent places: the resolved
    settings inside `campaign_metadata.json`, that file's provenance block, and a
    `settings_modality` column on every output row. All three are consulted, so a
    campaign whose metadata file is missing is still identified from its CSV.
    """
    metadata = campaign.metadata or {}
    settings = metadata.get("settings") if isinstance(metadata.get("settings"), dict) else {}
    provenance = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    sources = [settings.get("modality"), provenance.get("settings_modality"),
               metadata.get("modality")]
    for row in (campaign.ranked[:1] + campaign.candidates[:1]):
        sources.append(row.get("settings_modality"))
    for declared in sources:
        if not declared:
            continue
        names = (re.split(r"[;,]", declared) if isinstance(declared, str) else list(declared))
        for name in names:
            if str(name).strip() in ANTIBODY_MODALITIES:
                return str(name).strip()
    return ""


def top_designs(campaign: Campaign, limit: int = 10, modality: str = "") -> dict:
    """The accepted designs, best first, with antibody annotation where possible.

    {"modality", "convention", "designs": [...], "notes": [...]}. The order is the
    order of `3_Ranked/!_Ranked.csv`, which the pipeline writes by its i_pDAE
    ranking; this function never re-sorts and never pads the list to `limit`.
    A campaign of another modality still gets its designs back, without loop
    annotation, so the PD-L1 path is unaffected.
    """
    modality = modality or is_antibody_campaign(campaign)
    report = {"modality": modality, "convention": NUMBERING_CONVENTION if modality else "",
              "designs": [], "notes": []}
    reference = {}
    if modality:
        try:
            reference = scaffold_reference(modality)
        except RuntimeError as failure:
            report["notes"].append(f"loop annotation unavailable: {failure}")

    for row in campaign.ranked[:max(0, limit)]:
        design = (row.get("design") or "").strip()
        chains = [chain.strip() for chain in str(row.get("Binder_Sequence") or "").split("/")
                  if chain.strip()]
        paratopes = [part.strip() for part in
                     str(row.get("Interface_Binder_Residues") or "").split("/")]
        files = [path for path in campaign.structures
                 if path.stem == design or path.stem.startswith(design + "_")]
        entry = {"rank": row.get("rank", ""), "design": design,
                 "structure": files[0].name if files else "",
                 "structure_path": str(files[0]) if files else "",
                 "chain_count": len(chains),
                 "confidence": {key: row[key] for key in CONFIDENCE_COLUMNS
                                if key in row and str(row[key]).strip()},
                 "interface": {key: row[key] for key in INTERFACE_COLUMNS
                               if key in row and str(row[key]).strip()},
                 "target_residues": (row.get("Interface_Target_Residues") or "").strip(),
                 "domains": [], "hcdr3": {}}
        if reference:
            entry["domains"] = assign_domains(chains, reference)
            for position, domain in enumerate(entry["domains"]):
                domain["paratope"] = split_paratope(
                    paratopes[position] if position < len(paratopes) else "", domain)
                if domain.get("domain") == "VH":
                    entry["hcdr3"] = kabat_hcdr3(domain.get("sequence") or "")
        else:
            entry["domains"] = [{"domain": f"chain {position + 1}", "sequence": sequence,
                                 "length": len(sequence), "loops": [],
                                 "paratope": {"loops": {}, "framework": [], "unmapped": [],
                                              "contacts": 0},
                                 "chain_position": position + 1, "notes": []}
                                for position, sequence in enumerate(chains)]
        report["designs"].append(entry)

    if len(report["designs"]) < limit:
        report["notes"].append(f"{len(report['designs'])} accepted design(s) available; "
                               f"{limit} were asked for. Nothing was substituted to reach "
                               f"{limit}: a rejected candidate is not a candidate.")
    return report


def domain_fasta(campaign: Campaign, limit: int = 10, modality: str = "") -> str:
    """FASTA of the accepted designs with each variable domain named as a domain.

    One record per chain, headed `<design>_VH` / `<design>_VL` where the chain's
    framework identified it, so the PI can tell which sequence is which without
    counting chains. Falls back to `binder_fasta`'s chain numbering otherwise.
    """
    report = top_designs(campaign, limit=limit, modality=modality)
    records = []
    for entry in report["designs"]:
        notes = " ".join(f"{key}={entry['confidence'][key]}"
                         for key in ("i_pDAE", "i_pTM", "i_pAE", "pLDDT")
                         if key in entry["confidence"])
        for domain in entry["domains"]:
            sequence = domain.get("sequence") or ""
            if not sequence:
                continue
            label = domain.get("domain", "")
            suffix = label if label in ("VH", "VL") else f"chain{domain.get('chain_position', 1)}"
            header = f">{entry['design']}_{suffix} rank={entry['rank']}"
            records.append(f"{header} {notes}".rstrip() + "\n" + sequence)
    return "\n".join(records) + ("\n" if records else "")


CDR_TABLE_COLUMNS = ["rank", "design", "domain", "loop", "scaffold_span", "sequence", "length",
                     "contacting_residues", "unavailable"]


def cdr_table(campaign: Campaign, limit: int = 10, modality: str = "") -> tuple:
    """(columns, rows) of every located loop of the accepted designs.

    One row per loop per domain. A loop that could not be located keeps its row,
    with an empty sequence and the reason in `unavailable`, so the table never
    silently drops a loop.
    """
    report = top_designs(campaign, limit=limit, modality=modality)
    rows = []
    for entry in report["designs"]:
        for domain in entry["domains"]:
            contacts = (domain.get("paratope") or {}).get("loops", {})
            for loop in domain.get("loops", []):
                rows.append({"rank": entry["rank"], "design": entry["design"],
                             "domain": domain.get("domain", ""), "loop": loop["name"],
                             "scaffold_span": loop["scaffold_span"],
                             "sequence": loop["sequence"] or "",
                             "length": "" if loop["length"] is None else loop["length"],
                             "contacting_residues": ",".join(contacts.get(loop["name"], [])),
                             "unavailable": loop["unavailable"]})
            if domain.get("domain") == "VH" and entry.get("hcdr3"):
                hcdr3 = entry["hcdr3"]
                rows.append({"rank": entry["rank"], "design": entry["design"],
                             "domain": "VH", "loop": "HCDR3 (Kabat)",
                             "scaffold_span": "located in the design (C-x-x .. W-G-x-G)",
                             "sequence": hcdr3.get("sequence") or "",
                             "length": "" if hcdr3.get("length") is None else hcdr3["length"],
                             "contacting_residues": "",
                             "unavailable": hcdr3.get("unavailable", "")})
    return CDR_TABLE_COLUMNS, rows


def cdr_table_view(campaign: Campaign, limit: int = 10, modality: str = ""):
    columns, rows = cdr_table(campaign, limit=limit, modality=modality)
    return as_frame(columns, rows, columns)


def accepted_designs(campaign: Campaign, limit: int = 10) -> list:
    """[{rank, design, path, label}] for the accepted designs that have a file.

    In ranked order, one entry per accepted design, never the same file twice.
    This is what populates the individual 3D viewer, so a campaign with three
    accepted designs offers exactly three.
    """
    entries, used = [], set()
    for row in campaign.ranked[:max(0, limit)]:
        design = (row.get("design") or "").strip()
        for path in campaign.structures:
            if path in used:
                continue
            if path.stem == design or path.stem.startswith(design + "_"):
                rank = str(row.get("rank", "") or len(entries) + 1)
                entries.append({"rank": rank, "design": design, "path": path,
                                "label": f"rank {rank} - {design or path.stem}"})
                used.add(path)
                break
    return entries


def campaign_target_sequences(campaign: Campaign) -> dict:
    """{name: sequence} for each target the campaign's metadata names, where readable.

    Read from the target files the recorded settings point at. A target file that
    has moved or was never on this machine simply does not appear, so the viewer
    falls back to calling a chain "target" by elimination rather than wrongly.
    """
    settings = (campaign.metadata or {}).get("settings")
    sequences: dict = {}
    for target in (settings or {}).get("targets", []) if isinstance(settings, dict) else []:
        located = Path(str(target.get("target_path") or ""))
        if not located.is_file():
            continue
        name = str(target.get("name") or located.stem)
        wanted = {part.strip() for part in str(target.get("chains") or "").split(",")
                  if part.strip()}
        for entry in chain_inventory(located):
            if entry["sequence"] and (not wanted or entry["chain"] in wanted):
                sequences[f"{name}:{entry['chain']}"] = entry["sequence"]
        for record_name, sequence in (read_fasta(located)
                                      if structure_format(located) == "fasta" else []):
            sequences[f"{name}:{record_name}"] = sequence
    return sequences


def structure_domains(path, reference: dict | None = None, targets: dict | None = None) -> list:
    """[{chain, length, domain, evidence}] for one complex file.

    A chain is labelled on evidence, in this order: it matches a chain of the
    campaign's own target file ("target (<name>)"); it carries the scaffold
    framework anchors of a variable domain ("VH"/"VL"); or it is a binder chain
    that this code cannot name more precisely ("binder chain"). Nothing is
    labelled by position in the file.
    """
    labelled = []
    for chain, numbered in cif_chain_residues(path).items():
        sequence = "".join(code for _, code in numbered)
        domain, evidence = "unidentified chain", "no sequence evidence for this chain"
        for name, target_sequence in (targets or {}).items():
            if sequence and (sequence == target_sequence
                             or sequence in target_sequence or target_sequence in sequence):
                domain = f"target ({name.split(':')[0]})"
                evidence = f"sequence matches the campaign's target file, chain {name.split(':')[-1]}"
                break
        else:
            best, score = "", 0
            for entry in (reference or {}).values():
                matched = _similarity(sequence, entry)
                if matched > score:
                    best, score = entry["domain"], matched
            if best:
                domain = best
                evidence = f"{score} scaffold framework anchors matched"
            elif targets:
                domain = "binder chain"
                evidence = ("no target match and no scaffold framework match, so it is "
                            "reported as a binder chain without a domain name")
        labelled.append({"chain": chain, "length": len(numbered), "domain": domain,
                         "evidence": evidence})
    # With no target sequences to compare against, a chain that is not a variable
    # domain cannot be told apart from the target, and is named as the open
    # question it is rather than asserted to be either.
    if not targets:
        for entry in labelled:
            if entry["domain"] == "unidentified chain":
                entry["domain"] = "target or binder (not determined)"
                entry["evidence"] = ("the campaign's target file was not readable from here, so "
                                     "this chain is not claimed to be either")
    return labelled


CHAIN_PALETTE = ["#3c5b6f", "#B76E79", "#8FA98F", "#C9A227"]


def show_structure(campaign: Campaign, index: int = 0, colour_by: str = "chain",
                   modality: str = "", highlight: str = "none"):
    """One accepted complex in py3Dmol, chosen by position in the ranked order.

    Returns (view, entry, chains) where `entry` is the `accepted_designs` record
    and `chains` the `structure_domains` labelling. `highlight` may be "none",
    "HCDR3" or "epitope"; a highlight that cannot be established from the
    recorded metadata is skipped and reported in the returned chain list rather
    than drawn in the wrong place.
    """
    entries = accepted_designs(campaign, limit=10)
    if not entries:
        raise RuntimeError(f"no accepted complex files in {campaign.folder / '3_Ranked'}")
    try:
        import py3Dmol
    except ImportError as missing:
        raise RuntimeError(
            "py3Dmol is not importable in this kernel. Install it in the bindcraft2 environment "
            "(python -m pip install py3Dmol) or download the .cif files and open them locally."
        ) from missing

    entry = entries[max(0, min(int(index), len(entries) - 1))]
    modality = modality or is_antibody_campaign(campaign)
    reference = {}
    if modality:
        try:
            reference = scaffold_reference(modality)
        except RuntimeError:
            reference = {}
    chains = structure_domains(entry["path"], reference or None,
                               campaign_target_sequences(campaign) or None)

    view = py3Dmol.view(width=720, height=520)
    view.addModel(entry["path"].read_text(encoding="utf-8", errors="replace"), "cif")
    if colour_by == "pLDDT":
        # BC2 writes per-residue pLDDT on a 0-100 scale into the B-factor column.
        view.setStyle({}, {"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb",
                                                       "min": 50, "max": 90}}})
    else:
        for number, chain in enumerate(chains or [{"chain": "A"}]):
            view.setStyle({"chain": chain["chain"]},
                          {"cartoon": {"color": CHAIN_PALETTE[number % len(CHAIN_PALETTE)]}})

    if highlight and highlight != "none":
        drawn = _highlight_residues(view, campaign, entry, chains, reference, highlight)
        for chain in chains:
            chain["highlight"] = drawn
    view.zoomTo()
    view.setBackgroundColor("white")
    return view, entry, chains


def _highlight_residues(view, campaign: Campaign, entry: dict, chains: list,
                        reference: dict, highlight: str) -> str:
    """Draw sticks on the HCDR3 or the contacted target residues; say what happened."""
    row = next((item for item in campaign.ranked
                if (item.get("design") or "").strip() == entry["design"]), {})
    if highlight == "epitope":
        labels = [(match.group(1), int(match.group(2))) for match in
                  re.finditer(r"([A-Z])(\d+)", row.get("Interface_Target_Residues") or "")]
        # Any chain this code has not positively called a variable domain is a
        # candidate target chain; the residue-letter check below decides which
        # one actually carries the recorded epitope.
        targets = [chain["chain"] for chain in chains
                   if chain["domain"] not in ("VH", "VL", "binder chain")]
        if not labels:
            return "epitope highlight skipped: no contacted target residues recorded for this design"
        if not targets:
            return "epitope highlight skipped: no target chain identified in this structure"
        # Each recorded label carries its residue letter. Only labels that match the
        # structure at that number are drawn, so a numbering difference between the
        # table and the file shows up as a count rather than as sticks in the wrong place.
        residues_by_chain = cif_chain_residues(entry["path"])
        drawn, mismatched = [], []
        for letter in targets:
            present = dict(residues_by_chain.get(letter, []))
            for code, number in labels:
                if present.get(number) == code:
                    drawn.append((letter, number))
                else:
                    mismatched.append(f"{code}{number}")
        if not drawn:
            return (f"epitope highlight skipped: none of {len(labels)} recorded target residue(s) "
                    "matches the target chain in this structure file by residue and number")
        for letter in targets:
            numbers = [number for chain, number in drawn if chain == letter]
            if numbers:
                view.addStyle({"chain": letter, "resi": numbers},
                              {"stick": {"colorscheme": "yellowCarbon", "radius": 0.2}})
        return (f"{len(drawn)} contacted target residue(s) shown as sticks"
                + (f"; {len(set(mismatched))} recorded residue(s) did not match the file and "
                   "were not drawn" if mismatched else ""))

    if highlight == "HCDR3":
        if not reference:
            return "HCDR3 highlight skipped: no scaffold reference for this campaign"
        sequences = [part.strip() for part in
                     str(row.get("Binder_Sequence") or "").split("/") if part.strip()]
        heavy = next((domain for domain in assign_domains(sequences, reference)
                      if domain.get("domain") == "VH"), None)
        if heavy is None:
            return "HCDR3 highlight skipped: no chain identified as VH"
        located = kabat_hcdr3(heavy.get("sequence") or "")
        if not located.get("sequence"):
            return f"HCDR3 highlight skipped: {located.get('unavailable', 'not located')}"
        letter = next((chain["chain"] for chain in chains if chain["domain"] == "VH"), "")
        if not letter:
            return "HCDR3 highlight skipped: the VH chain is not identifiable in the structure"
        # The located span counts positions along the sequence. Drawing it needs the
        # structure to number that chain the same way, so that is checked rather
        # than assumed: a highlight on the wrong residues would be worse than none.
        numbered = cif_chain_residues(entry["path"]).get(letter, [])
        structure_numbers = [number for number, _ in numbered]
        structure_sequence = "".join(code for _, code in numbered)
        if structure_sequence != (heavy.get("sequence") or ""):
            return ("HCDR3 highlight skipped: the VH chain in the structure file does not "
                    "match the VH sequence in the ranked table")
        if structure_numbers != list(range(1, len(structure_numbers) + 1)):
            return (f"HCDR3 highlight skipped: chain {letter} is numbered "
                    f"{structure_numbers[0]}-{structure_numbers[-1]}, not from 1, so the "
                    "located span cannot be mapped onto it safely")
        view.addStyle({"chain": letter,
                       "resi": list(range(located["start"], located["end"] + 1))},
                      {"stick": {"colorscheme": "magentaCarbon", "radius": 0.2}})
        return (f"HCDR3 {located['sequence']} ({located['length']} aa) shown as sticks on "
                f"chain {letter}, residues {located['start']}-{located['end']}")
    return f"unknown highlight {highlight!r}"


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

    # Antibody-format campaigns additionally get a chain-named FASTA and the loop
    # table. Both are skipped silently for a de novo binder campaign, so the
    # PD-L1 export is byte-for-byte what it was.
    if is_antibody_campaign(campaign):
        domains = domain_fasta(campaign, limit=limit)
        if domains:
            target = folder / f"{campaign.folder.name}_VH_VL.fasta"
            target.write_text(domains, encoding="utf-8", newline="\n")
            written.append(target.name)
        columns, rows = cdr_table(campaign, limit=limit)
        if rows:
            target = folder / f"{campaign.folder.name}_CDRs.csv"
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            written.append(target.name)
            note = folder / f"{campaign.folder.name}_CDR_numbering.txt"
            note.write_text(NUMBERING_CONVENTION + "\n", encoding="utf-8", newline="\n")
            written.append(note.name)

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
