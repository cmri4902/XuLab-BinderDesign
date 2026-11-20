#!/usr/bin/env python3
"""Xu Lab BindCraft2 output validator (Phase 1D).

Reads a completed *or partial* campaign folder and reports what the campaign
actually produced, as an auditable markdown report. It only ever reads: it
creates no result file, writes no score, and invents no design.

    python lab/scripts/validate_outputs.py /project/yli4/ypan/bindcraft/results/pdl1_h200_smoke \
        --slurm-log '/project/yli4/ypan/bindcraft/logs/xulab_bc2_*.out' \
        --output /project/yli4/ypan/bindcraft/logs/pdl1_h200_smoke_validation.md

The validator itself needs only the standard library, so the same file runs
under the HPC conda interpreter and under any Python 3.9+ on Windows after a
transfer. If `biotite` or `gemmi` happens to be importable it additionally runs
a full structure parse; otherwise that check reports NOT TESTED and the
lightweight text check stands on its own.

Two independent acceptance levels are reported:

  Level A  compute smoke  - the lab source copy ran with a JAX GPU backend and
                            performed genuine design computation, with at least
                            one trajectory/candidate record to show for it, and
                            the campaign process exited 0.
  Level B  end-to-end     - at least one redesigned sequence passed the normal
                            filters and has a ranking row plus a predicted
                            complex file of its own.

Level A deliberately requires *explicit JAX GPU backend evidence*: a
`CUDA_VISIBLE_DEVICES` assignment, an `nvidia-smi` listing or a driver-level
"N GPUs visible" note proves a card was allocated, not that JAX computed on it,
so those are recorded as supporting evidence only and can never grant Level A.

Level A can pass while Level B stays NOT ACHIEVED. Zero accepted designs out of
a short run is an ordinary outcome, not evidence of a broken deployment, and an
accepted design is a computational result, not experimental binding evidence.

Exit status: 0 nothing failed, 1 at least one FAIL, 2 the campaign folder could
not be read at all (BLOCKED).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

PASS, FAIL, BLOCKED, NOT_TESTED, INFO, NOT_PRODUCED, WARN = (
    "PASS", "FAIL", "BLOCKED", "NOT TESTED", "INFO", "NOT PRODUCED", "WARN")

# Stage tables in the current layout, with the pre-1.0 flat layout as fallback.
TABLES = {
    "trajectories": ("1_Trajectories/!_Trajectories.csv", "trajectories.csv"),
    "refolded": ("2_Refolded/!_Refolded.csv", "candidates.csv"),
    "ranked": ("3_Ranked/!_Ranked.csv", "ranked.csv", "accepted.csv"),
}

# Explicit proof that JAX itself selected a GPU/ROCm backend. Only these can
# grant Level A.
JAX_GPU_EVIDENCE = (
    ("jax backend line", re.compile(r"\bbackend\s*=\s*(gpu|cuda|rocm)\b", re.I)),
    ("jax default_backend", re.compile(r"default_backend\(\)\s*[=:]\s*'?(gpu|cuda|rocm)", re.I)),
    ("jax device line", re.compile(r"xulab-bc2:\s*device\s+(gpu|cuda|rocm):", re.I)),
    ("jax device repr", re.compile(r"(cuda|rocm)Device\(id=\d+|StreamExecutorGpuDevice|"
                                   r"TFRT_GPU|gpu:\d+\s+(NVIDIA|AMD)", re.I)),
    ("xla gpu client", re.compile(r"Created TensorFlow device|XLA service.*device.*GPU|"
                                  r"StreamExecutor device \(\d+\)", re.I)),
)

# A card was allocated or visible to the driver. Necessary, never sufficient.
SUPPORTING_GPU_EVIDENCE = (
    ("allocated devices", re.compile(r"CUDA_VISIBLE_DEVICES=(?!unset\b)\S+")),
    ("nvidia-smi card", re.compile(r"\b(H200|H100|A100|L40S|GH200|B200|T4|V100)\b")),
    ("driver gpu count", re.compile(r"GPUs? visible", re.I)),
    ("slurm gpu binding", re.compile(r"SLURM_JOB_GPUS=(?!unset\b)\S+")),
)

CPU_WARNING = re.compile(r"no GPU visible|leaves AlphaFold on the CPU|"
                         r"\bbackend\s*=\s*cpu\b|refusing to design on CPU", re.I)
# A driver that will not initialise is the job-1014429 failure: nvidia-smi sees
# the card, cuInit does not, and JAX quietly falls back to the CPU. It is called
# out separately because the supporting GPU evidence all looks healthy.
DRIVER_FAILURE = re.compile(
    r"cuInit\(0\)[^\n]*fail|CUDA_ERROR_NO_DEVICE|CUDA_ERROR_NO_CUDA_CAPABLE_DEVICE|"
    r"no CUDA-capable device is detected|Unable to initialize backend ['\"]?(cuda|gpu|rocm)|"
    r"xulab-gpu: verdict (NO LIBCUDA|DRIVER INIT FAILED|NO JAX GPU|NO JAX)", re.I)
# "campaign exit status 0" from the lab launcher, and generic exit reports.
EXIT_STATUS = re.compile(r"campaign exit status\s+(\d+)", re.I)
GENERIC_EXIT = re.compile(r"\bexit(?:ed with|\s+status|\s+code)[:= ]\s*(\d+)", re.I)
SLURM_TERMINATION = re.compile(r"DUE TO TIME LIMIT|CANCELLED AT|"
                               r"Killed|oom-kill|slurmstepd: error", re.I)
ERROR_EVIDENCE = re.compile(
    r"Traceback \(most recent call last\)|CUDA_ERROR|CUDA error|RESOURCE_EXHAUSTED|"
    r"out of memory|XlaRuntimeError|slurmstepd: error|srun: error|DUE TO TIME LIMIT|"
    r"CANCELLED|xulab-bc2: ERROR", re.I)
MONOMER_SUFFIX = re.compile(r"_monomer$")


class Check:
    """One reportable observation."""

    def __init__(self, name: str, status: str, detail: str = "", evidence=()):
        self.name, self.status, self.detail = name, status, detail
        self.evidence = list(evidence)

    def as_dict(self) -> dict:
        return {"check": self.name, "status": self.status,
                "detail": self.detail, "evidence": self.evidence}


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #

def read_table(path: Path):
    """Return (fieldnames, rows, error). Never raises on malformed input."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader]
            return (reader.fieldnames or []), rows, None
    except (OSError, UnicodeDecodeError, csv.Error) as failure:
        return [], [], str(failure)


def locate_table(campaign: Path, kind: str):
    """First existing candidate path for a stage table, else None."""
    for relative in TABLES[kind]:
        candidate = campaign / relative
        if candidate.is_file():
            return candidate
    return None


def label(campaign: Path, path: Path) -> str:
    """Campaign-relative name for the report, falling back to the full path."""
    try:
        return str(path.relative_to(campaign)).replace(os.sep, "/")
    except ValueError:
        return str(path)


def filled(row: dict, column: str) -> bool:
    value = (row.get(column) or "").strip()
    return bool(value) and value.lower() not in ("nan", "none", "na")


def tally(rows, column: str) -> dict:
    counts: dict = {}
    for row in rows:
        value = (row.get(column) or "").strip() or "(blank)"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


# --------------------------------------------------------------------------- #
# structure inspection: two clearly separated levels
# --------------------------------------------------------------------------- #

def mmcif_text_check(path: Path):
    """LIGHTWEIGHT check: text only, no chemistry, no geometry, no validation.

    Confirms the file is nonempty, carries an mmCIF `data_` block and an
    `_atom_site` loop, and counts ATOM/HETATM rows and chain labels by splitting
    whitespace. It does not verify residue chemistry, atom naming, bond
    geometry, occupancies or that coordinates are physically meaningful.
    """
    try:
        if path.stat().st_size == 0:
            return 0, [], "file is empty"
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as failure:
        return 0, [], str(failure)
    if not re.search(r"^data_", text, re.M):
        return 0, [], "no mmCIF data_ block"
    if "_atom_site." not in text:
        return 0, [], "no _atom_site loop"
    headers, atoms, chains = [], 0, []
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
                atoms += 1
                fields = stripped.split()
                for name in ("auth_asym_id", "label_asym_id"):
                    if name in headers and len(fields) > headers.index(name):
                        chain = fields[headers.index(name)]
                        if chain not in chains:
                            chains.append(chain)
                        break
    if atoms == 0:
        return 0, chains, "_atom_site loop holds no ATOM/HETATM rows"
    return atoms, chains, None


def full_parser():
    """Return (name, parse) for a real mmCIF parser, or (None, None)."""
    try:
        from biotite.structure.io.pdbx import CIFFile, get_structure  # type: ignore

        def parse_with_biotite(path: Path):
            structure = get_structure(CIFFile.read(str(path)), model=1)
            chains = sorted({str(chain) for chain in structure.chain_id})
            return len(structure), chains

        return "biotite", parse_with_biotite
    except Exception:                                      # noqa: BLE001
        pass
    try:
        import gemmi  # type: ignore

        def parse_with_gemmi(path: Path):
            structure = gemmi.read_structure(str(path))
            structure.setup_entities()
            model = structure[0]
            atoms = sum(len(residue) for chain in model for residue in chain)
            return atoms, sorted({chain.name for chain in model})

        return "gemmi", parse_with_gemmi
    except Exception:                                      # noqa: BLE001
        return None, None


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #

def check_trajectories(campaign: Path, checks: list) -> int:
    path = locate_table(campaign, "trajectories")
    if path is None:
        checks.append(Check(
            "1_Trajectories/!_Trajectories.csv", NOT_PRODUCED,
            "No trajectory table. A stage folder appears when its first result is "
            "written, so this means no gradient-design attempt has been recorded yet."))
        return 0
    columns, rows, failure = read_table(path)
    if failure:
        checks.append(Check(label(campaign, path), FAIL,
                            f"table unreadable: {failure}", [str(path)]))
        return 0
    if not rows:
        checks.append(Check(label(campaign, path), INFO,
                            "table present with a header but no attempt rows yet "
                            "(campaign started, nothing completed)", [str(path)]))
        return 0
    detail = [f"{len(rows)} attempt row(s); {len(columns)} column(s)"]
    if "terminated" in columns:
        stages = tally(rows, "terminated")
        completed = stages.get("(blank)", 0)
        ended_early = {stage: count for stage, count in stages.items() if stage != "(blank)"}
        detail.append(f"terminated stages: {ended_early or 'none'}; "
                      f"{completed} attempt(s) completed gradient design "
                      f"(blank means completed, not that a sequence was accepted)")
    else:
        detail.append("no 'terminated' column, so stage interpretation is unavailable "
                      "(an older or rewritten table)")
    for column in ("design", "hash", "trajectory"):
        if column not in columns:
            detail.append(f"identity column '{column}' absent")
    confidence = [column for column in ("i_pTM", "i_pDAE", "pLDDT", "plddt", "iptm")
                  if column in columns]
    detail.append("confidence columns present: " + (", ".join(confidence) or "none"))
    checks.append(Check(label(campaign, path), PASS, "; ".join(detail), [str(path)]))
    return len(rows)


def check_refolded(campaign: Path, checks: list) -> int:
    path = locate_table(campaign, "refolded")
    if path is None:
        checks.append(Check(
            "2_Refolded/!_Refolded.csv", NOT_PRODUCED,
            "No refolded table: ProteinMPNN redesign/validation has not written a "
            "candidate yet. Expected while only trajectories have run."))
        return 0
    columns, rows, failure = read_table(path)
    if failure:
        checks.append(Check(label(campaign, path), FAIL,
                            f"table unreadable: {failure}", [str(path)]))
        return 0
    if not rows:
        checks.append(Check(label(campaign, path), INFO,
                            "header only, no scored candidate rows", [str(path)]))
        return 0
    detail = [f"{len(rows)} scored candidate row(s)"]
    if "outcome" in columns:
        detail.append(f"outcomes: {tally(rows, 'outcome')} "
                      "(a 'passed' candidate can still fall outside kept_sequences)")
    else:
        detail.append("no 'outcome' column; pass/reject cannot be read from this table")
    if "failed_filters" in columns:
        reasons: dict = {}
        for row in rows:
            for reason in (row.get("failed_filters") or "").split(","):
                reason = reason.strip()
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
        top = dict(sorted(reasons.items(), key=lambda item: -item[1])[:10])
        detail.append(f"most frequent failed filters: {top or 'none recorded'} "
                      "('not measured' is a missing reading, not a low score)")
    identities = sum(1 for row in rows if filled(row, "design"))
    detail.append(f"{identities}/{len(rows)} row(s) carry a 'design' identity")
    complexes = sorted((campaign / "2_Refolded" / "Complexes").glob("*.cif"))
    detail.append(f"{len(complexes)} candidate complex file(s) under 2_Refolded/Complexes "
                  "(rejected predictions are kept unless save_failed_refolds is false)")
    checks.append(Check(label(campaign, path), PASS, "; ".join(detail), [str(path)]))
    return len(rows)


def check_ranked(campaign: Path, checks: list, full_parse: bool):
    """Return (accepted_rows, rows_with_sequence, rows_with_complex)."""
    path = locate_table(campaign, "ranked")
    rows: list = []
    if path is None:
        checks.append(Check(
            "3_Ranked/!_Ranked.csv", NOT_PRODUCED,
            "No ranking table: no design has been accepted. For a 20-trajectory smoke "
            "run this is an ordinary outcome and not by itself a deployment failure."))
    else:
        columns, rows, failure = read_table(path)
        if failure:
            checks.append(Check(label(campaign, path), FAIL,
                                f"table unreadable: {failure}", [str(path)]))
            rows = []
        elif not rows:
            checks.append(Check(label(campaign, path), INFO,
                                "ranking table present but empty: nothing accepted so far",
                                [str(path)]))
        else:
            with_sequence = sum(1 for row in rows if filled(row, "Binder_Sequence"))
            detail = [f"{len(rows)} accepted design row(s)",
                      f"{with_sequence} row(s) carry Binder_Sequence "
                      "('/' separates binder chains)"]
            if "rank" in columns:
                detail.append("rank column present (position in this ranking only, "
                              "not a probability of experimental success)")
            scores = [column for column in ("i_pDAE", "i_pTM", "i_pAE", "pLDDT")
                      if column in columns]
            detail.append("ranking metrics present: " + (", ".join(scores) or "none"))
            status = PASS
            if with_sequence < len(rows):
                status = FAIL
                missing = [(row.get("design") or "(row without a design value)")
                           for row in rows if not filled(row, "Binder_Sequence")]
                detail.append("accepted row(s) without a sequence: " + ", ".join(missing[:10]))
            checks.append(Check(label(campaign, path), status, "; ".join(detail), [str(path)]))

    with_sequence = sum(1 for row in rows if filled(row, "Binder_Sequence"))
    with_complex = check_ranked_structures(campaign, rows, checks, full_parse)
    return len(rows), with_sequence, with_complex


def check_ranked_structures(campaign: Path, ranked_rows, checks: list, full_parse: bool) -> int:
    ranked_dir = campaign / "3_Ranked"
    files = sorted(ranked_dir.glob("*.cif"))
    relaxed = sorted((ranked_dir / "relaxed").glob("*.cif"))
    if not files and not relaxed:
        checks.append(Check("3_Ranked/*.cif", FAIL if ranked_rows else NOT_PRODUCED,
                            "no accepted complex files in 3_Ranked"
                            + (f" although the ranking table holds {len(ranked_rows)} row(s)"
                               if ranked_rows else "")))
        if ranked_rows:
            checks.append(Check("accepted design -> structure correspondence", FAIL,
                                "every accepted row is missing its structure: "
                                + ", ".join((row.get("design") or "(unnamed row)")
                                            for row in ranked_rows[:10])))
        return 0

    # --- lightweight text check ------------------------------------------- #
    parsed, broken = [], []
    for path in files + relaxed:
        atoms, chains, problem = mmcif_text_check(path)
        if problem:
            broken.append(f"{path.name}: {problem}")
        else:
            parsed.append((path, atoms, chains))
    detail = [f"{len(files)} file(s) in 3_Ranked plus {len(relaxed)} relaxed file(s)",
              f"{len(parsed)} nonempty file(s) with a data_ block, an _atom_site loop "
              "and at least one ATOM/HETATM row"]
    if parsed:
        atom_counts = [atoms for _, atoms, _ in parsed]
        detail.append(f"atom rows per file: min {min(atom_counts)}, max {max(atom_counts)}")
        chain_sets = sorted({",".join(chains) for _, _, chains in parsed if chains})
        detail.append("chain labels seen: " + (" | ".join(chain_sets[:5]) or "unreadable"))
    detail.append("SCOPE: text-level check only - no chemistry, geometry, bond or "
                  "occupancy validation is performed here")
    if broken:
        detail.append("unparseable or empty: " + "; ".join(broken[:10]))
    checks.append(Check("3_Ranked/*.cif (lightweight text check)",
                        FAIL if broken else PASS, "; ".join(detail),
                        [str(path) for path, _, _ in parsed[:5]]))

    # --- full structure parse, when a real parser is importable ------------ #
    if not full_parse:
        checks.append(Check("3_Ranked/*.cif (full structure parse)", NOT_TESTED,
                            "skipped on request (--no-full-parse)"))
    else:
        parser_name, parse = full_parser()
        if parse is None:
            checks.append(Check("3_Ranked/*.cif (full structure parse)", NOT_TESTED,
                                "neither biotite nor gemmi is importable in this "
                                "interpreter, so no full mmCIF parse was attempted; the "
                                "lightweight text check above is all that was verified. "
                                "Re-run under the bindcraft2 conda environment (biotite is "
                                "an upstream dependency) for a real parse."))
        else:
            failures, summaries = [], []
            for path in files + relaxed:
                try:
                    atoms, chains = parse(path)
                except Exception as failure:               # noqa: BLE001
                    failures.append(f"{path.name}: {type(failure).__name__}: {failure}")
                    continue
                if atoms == 0:
                    failures.append(f"{path.name}: parsed but holds no atoms")
                else:
                    summaries.append(f"{path.name}: {atoms} atoms, chains {','.join(chains)}")
            full_detail = [f"parser: {parser_name}",
                           f"{len(summaries)}/{len(files) + len(relaxed)} file(s) parsed into "
                           "a structure with atoms"]
            full_detail += summaries[:5]
            if failures:
                full_detail.append("parse failures: " + "; ".join(failures[:10]))
            full_detail.append("SCOPE: the parser accepted the file and produced atoms and "
                               "chains; this is still not a validation of design quality")
            checks.append(Check("3_Ranked/*.cif (full structure parse)",
                                FAIL if failures else PASS, "; ".join(full_detail)))

    # --- per-candidate correspondence ------------------------------------- #
    # A design row may own several files (one per target state, plus a free-binder
    # monomer), so correspondence is counted per row, never per file.
    if not ranked_rows:
        orphan_names = [path.name for path, _, _ in parsed]
        checks.append(Check("accepted design -> structure correspondence", INFO,
                            f"no accepted rows to link; {len(orphan_names)} structure "
                            "file(s) present without a ranking table: "
                            + ", ".join(orphan_names[:5])))
        return 0

    matched, missing_rows = {}, []
    identities = [(row.get("design") or "").strip() for row in ranked_rows]
    for identity in identities:
        if not identity:
            missing_rows.append("(row without a 'design' value)")
            continue
        owned = [path.name for path, _, _ in parsed
                 if path.stem == identity or path.stem.startswith(identity + "_")]
        complexes = [name for name in owned if not MONOMER_SUFFIX.search(Path(name).stem)]
        if complexes:
            matched[identity] = complexes
        else:
            monomer_only = " (a monomer file exists but no complex)" if owned else ""
            missing_rows.append(identity + monomer_only)
    orphans = [path.name for path, _, _ in parsed
               if not any(path.stem == identity or path.stem.startswith(identity + "_")
                          for identity in identities if identity)]
    correspondence = [f"{len(matched)}/{len(identities)} accepted row(s) have at least one "
                      "complex structure file"]
    if missing_rows:
        correspondence.append(f"{len(missing_rows)} accepted design(s) WITHOUT a complex "
                              "structure: " + ", ".join(missing_rows[:15]))
    if orphans:
        correspondence.append(f"{len(orphans)} structure file(s) match no ranking row "
                              "(deleting a .cif is the documented way to reject a design, so "
                              "this alone is not an error): " + ", ".join(orphans[:5]))
    checks.append(Check("accepted design -> structure correspondence",
                        FAIL if missing_rows else PASS, "; ".join(correspondence)))
    return len(matched)


def check_summary(campaign: Path, checks: list) -> None:
    path = campaign / "summary.csv"
    if not path.is_file():
        checks.append(Check("summary.csv", NOT_PRODUCED,
                            "not written yet; it is rebuilt by 'bindcraft campaign_output "
                            "<folder>' and by a campaign that closes normally"))
        return
    columns, rows, failure = read_table(path)
    if failure:
        checks.append(Check("summary.csv", FAIL, f"unreadable: {failure}", [str(path)]))
        return
    scopes = tally(rows, "scope") if "scope" in columns else {}
    checks.append(Check("summary.csv", PASS,
                        f"{len(rows)} summary row(s); scopes: {scopes or 'no scope column'} "
                        "('samples' counts available readings, not attempted designs)",
                        [str(path)]))


def check_metadata(campaign: Path, checks: list) -> None:
    paths = sorted(campaign.glob("campaign_metadata*.json"))
    if not paths:
        checks.append(Check("campaign_metadata.json", NOT_PRODUCED,
                            "no provenance file written yet"))
        return
    for path in paths:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as failure:
            checks.append(Check(path.name, FAIL, f"not valid JSON: {failure}", [str(path)]))
            continue
        keys = list(metadata)[:12] if isinstance(metadata, dict) else []
        revision = ""
        if isinstance(metadata, dict):
            for name in ("source_revision", "revision", "bindcraft_revision", "version"):
                if name in metadata:
                    revision = f"{name}={metadata[name]}"
                    break
        checks.append(Check(path.name, PASS,
                            f"valid JSON, {len(metadata) if isinstance(metadata, dict) else '?'} "
                            f"top-level key(s): {keys}"
                            + (f"; {revision}" if revision else "")
                            + " ('-dirty' in a revision marks a modified source tree)",
                            [str(path)]))


def check_worker_logs(campaign: Path, checks: list) -> None:
    logs = sorted((campaign / "workers").glob("worker_*.log"))
    settings = campaign / "workers" / "campaign_settings.json"
    if not logs and not settings.is_file():
        checks.append(Check("workers/", NOT_PRODUCED, "no per-worker records"))
        return
    detail = [f"{len(logs)} worker log(s)"]
    if settings.is_file():
        try:
            resolved = json.loads(settings.read_text(encoding="utf-8"))
            detail.append("workers/campaign_settings.json is valid JSON with "
                          f"{len(resolved) if isinstance(resolved, dict) else '?'} setting(s)")
        except (OSError, ValueError) as failure:
            detail.append(f"workers/campaign_settings.json unreadable: {failure}")
    for log in logs[:4]:
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as failure:
            detail.append(f"{log.name}: {failure}")
            continue
        errors = [number for number, line in enumerate(lines, 1) if ERROR_EVIDENCE.search(line)]
        detail.append(f"{log.name}: {len(lines)} line(s)"
                      + (f", error-like lines at {errors[:5]}" if errors else ", no error markers"))
    checks.append(Check("workers/", INFO, "; ".join(detail), [str(log) for log in logs[:5]]))


def check_slurm_logs(patterns, checks: list) -> dict:
    """Scan the supplied logs. Returns the evidence dictionary used for Level A."""
    evidence = {"jax": {}, "supporting": {}, "exit_codes": [], "terminations": [],
                "paths": [], "cpu_warned": False, "driver_failures": []}
    paths = []
    for pattern in patterns:
        expanded = [Path(match) for match in sorted(glob.glob(os.path.expanduser(pattern)))]
        paths.extend(expanded or ([Path(pattern)] if Path(pattern).is_file() else []))
    if not paths:
        checks.append(Check("SLURM logs", NOT_TESTED,
                            "no log file supplied or matched; pass --slurm-log to let this "
                            "report judge GPU execution and the campaign exit code. Without "
                            "a log, Level A cannot be granted."))
        return evidence

    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as failure:
            checks.append(Check(f"SLURM log {path.name}", FAIL,
                                f"unreadable: {failure}", [str(path)]))
            continue
        evidence["paths"].append(path)
        detail, errors = [f"{len(lines)} line(s)"], []
        for number, line in enumerate(lines, 1):
            where = f"{path}:{number}"
            excerpt = line.strip()[:160]
            for name, pattern in JAX_GPU_EVIDENCE:
                if pattern.search(line):
                    evidence["jax"].setdefault(name, f"{where}: {excerpt}")
            for name, pattern in SUPPORTING_GPU_EVIDENCE:
                if pattern.search(line):
                    evidence["supporting"].setdefault(name, f"{where}: {excerpt}")
            if CPU_WARNING.search(line):
                evidence["cpu_warned"] = True
            if DRIVER_FAILURE.search(line):
                evidence["driver_failures"].append(f"{where}: {excerpt}")
            found_exit = EXIT_STATUS.search(line) or GENERIC_EXIT.search(line)
            if found_exit:
                evidence["exit_codes"].append((int(found_exit.group(1)), where, excerpt))
            if SLURM_TERMINATION.search(line):
                evidence["terminations"].append(f"{where}: {excerpt}")
            if ERROR_EVIDENCE.search(line):
                errors.append(f"{number}: {excerpt}")
        detail.append(f"{len(errors)} error-like line(s)"
                      + (": " + " || ".join(errors[:5]) if errors else ""))
        checks.append(Check(f"SLURM log {path.name}", FAIL if errors else INFO,
                            "; ".join(detail), [str(path)]))

    # --- JAX GPU backend: the only evidence that can grant Level A --------- #
    sources = [str(path) for path in evidence["paths"]]
    if evidence["jax"]:
        checks.append(Check("JAX GPU backend evidence", PASS,
                            "; ".join(f"{name} at {where}"
                                      for name, where in evidence["jax"].items()), sources))
    elif evidence["supporting"]:
        checks.append(Check("JAX GPU backend evidence", FAIL,
                            "a GPU was allocated or visible, but no line shows JAX selecting "
                            "a GPU backend. An allocation alone does not prove the "
                            "computation ran on the card, so this cannot grant Level A. "
                            "Supporting evidence only: "
                            + "; ".join(f"{name} at {where}"
                                        for name, where in evidence["supporting"].items()),
                            sources))
    else:
        checks.append(Check("JAX GPU backend evidence", FAIL,
                            "logs were read and carry neither a JAX GPU backend line nor any "
                            "GPU allocation marker", sources))
    if evidence["supporting"]:
        checks.append(Check("Supporting GPU evidence (not sufficient alone)", INFO,
                            "; ".join(f"{name} at {where}"
                                      for name, where in evidence["supporting"].items()),
                            sources))

    # --- campaign exit status --------------------------------------------- #
    if evidence["exit_codes"]:
        nonzero = [item for item in evidence["exit_codes"] if item[0] != 0]
        if nonzero:
            checks.append(Check("campaign exit status", FAIL,
                                "nonzero exit reported: "
                                + "; ".join(f"status {code} at {where} ({excerpt})"
                                            for code, where, excerpt in nonzero[:5]),
                                sources))
        else:
            checks.append(Check("campaign exit status", PASS,
                                "; ".join(f"status {code} at {where}"
                                          for code, where, _ in evidence["exit_codes"][:5]),
                                sources))
    else:
        checks.append(Check("campaign exit status", NOT_TESTED,
                            "no exit-status line found in the supplied logs; the lab launcher "
                            "prints 'xulab-bc2: campaign exit status <n>' at the end of a run, "
                            "so its absence usually means the job was killed before finishing",
                            sources))
    if evidence["terminations"]:
        checks.append(Check("job termination markers", FAIL,
                            "; ".join(evidence["terminations"][:5]), sources))
    if evidence["cpu_warned"]:
        checks.append(Check("CPU fallback warning", FAIL,
                            "a log line reports no visible GPU or CPU execution; such a run "
                            "must not be scored as an H200 campaign", sources))
    if evidence["driver_failures"]:
        checks.append(Check("CUDA driver initialisation", FAIL,
                            "the CUDA driver or the JAX GPU backend failed to initialise, so a "
                            "card being allocated and listed by nvidia-smi means nothing here: "
                            + "; ".join(evidence["driver_failures"][:5])
                            + ". lab/scripts/gpu_diagnostics.sh in the allocation names the "
                              "cause (stub libcuda, missing /dev/nvidia-uvm, or a "
                              "CUDA_VISIBLE_DEVICES the step was not given)", sources))
    return evidence


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #

def verdicts(counts: dict, evidence: dict):
    trajectories, candidates = counts["trajectories"], counts["refolded"]
    accepted, sequences, complexes = counts["ranked"], counts["sequences"], counts["complexes"]
    computed = trajectories > 0 or candidates > 0
    records = (f"{trajectories} trajectory record(s), {candidates} candidate record(s)")
    nonzero = [item for item in evidence["exit_codes"] if item[0] != 0]

    if evidence["driver_failures"]:
        level_a = (FAIL, "the CUDA driver or the JAX GPU backend failed to initialise "
                         f"({evidence['driver_failures'][0]}); the campaign cannot have run on "
                         "a GPU regardless of what nvidia-smi reported")
    elif evidence["cpu_warned"]:
        level_a = (FAIL, "a log reports CPU execution or no visible GPU")
    elif not evidence["paths"]:
        level_a = (NOT_TESTED, "no SLURM log was supplied, so neither the JAX GPU backend nor "
                               "the campaign exit code could be verified"
                               + (f" (although {records} exist)" if computed else ""))
    elif not evidence["jax"]:
        if evidence["supporting"]:
            level_a = (FAIL, "a GPU was allocated or listed by nvidia-smi, but no log line "
                             "shows JAX selecting a GPU backend; allocation is not execution")
        else:
            level_a = (FAIL, "no JAX GPU backend evidence in the supplied logs")
    elif not computed:
        level_a = (FAIL, "JAX ran on a GPU but no trajectory or candidate record was written, "
                         "so no genuine design computation is demonstrated")
    elif nonzero:
        level_a = (FAIL, f"JAX GPU backend confirmed and {records} written, but the campaign "
                         f"exited nonzero (status {nonzero[0][0]} at {nonzero[0][1]}); the run "
                         "did not complete cleanly and is not a compute smoke PASS")
    elif evidence["terminations"]:
        level_a = (FAIL, f"JAX GPU backend confirmed and {records} written, but the job carries "
                         f"a termination marker ({evidence['terminations'][0]})")
    elif not evidence["exit_codes"]:
        level_a = (NOT_TESTED, f"JAX GPU backend confirmed and {records} written, but no exit "
                               "status was found in the logs, so a clean completion is "
                               "unproven")
    else:
        level_a = (PASS, f"JAX GPU backend evidenced in the logs, campaign exit status 0, and "
                         f"{records} written by the campaign")

    if accepted == 0:
        level_b = (NOT_TESTED if level_a[0] in (PASS, NOT_TESTED) else BLOCKED,
                   "NOT ACHIEVED: no design passed the campaign filters. For a short smoke "
                   "run this is an ordinary result; inspect 2_Refolded/!_Refolded.csv for "
                   "the filters that rejected candidates")
    elif sequences == accepted and complexes == accepted:
        level_b = (PASS, f"all {accepted} accepted design(s) carry a sequence and a predicted "
                         "complex file. This is a computational acceptance only - no binding, "
                         "affinity, specificity or stability is established")
    else:
        level_b = (FAIL, f"{accepted} accepted row(s) but {sequences} with a sequence and "
                         f"{complexes} with a complex structure; the record set is incomplete "
                         "(see the correspondence check for the designs concerned)")
    return level_a, level_b


def render(campaign: Path, checks, counts, level_a, level_b, patterns) -> str:
    rows = ["# Xu Lab BindCraft2 output validation",
            "",
            f"- Campaign folder: `{campaign}`",
            f"- Folder exists: {campaign.is_dir()}",
            f"- SLURM log patterns: {', '.join(f'`{p}`' for p in patterns) or '(none supplied)'}",
            "- Validator: `lab/scripts/validate_outputs.py` (read-only)",
            "",
            "## Acceptance levels",
            "",
            "| Level | Status | Basis |",
            "| --- | --- | --- |",
            f"| A - compute smoke | **{level_a[0]}** | {level_a[1]} |",
            f"| B - end-to-end binder output | **{level_b[0]}** | {level_b[1]} |",
            "",
            "Level A requires an explicit JAX GPU backend line plus a clean campaign exit.",
            "`CUDA_VISIBLE_DEVICES`, `nvidia-smi` output and driver-level GPU counts are",
            "recorded as supporting evidence and never grant Level A on their own.",
            "",
            "## Counts",
            "",
            "| Record | Count |",
            "| --- | --- |",
            f"| trajectory attempts | {counts['trajectories']} |",
            f"| scored candidates | {counts['refolded']} |",
            f"| accepted designs | {counts['ranked']} |",
            f"| accepted designs with a sequence | {counts['sequences']} |",
            f"| accepted designs with a complex structure | {counts['complexes']} |",
            "",
            "## Checks",
            "",
            "| Check | Status | Detail |",
            "| --- | --- | --- |"]
    for check in checks:
        detail = (check.detail or "").replace("|", "\\|").replace("\n", " ")
        rows.append(f"| `{check.name}` | **{check.status}** | {detail} |")
    evidenced = [check for check in checks if check.evidence]
    if evidenced:
        rows += ["", "## Evidence paths", ""]
        for check in evidenced:
            for path in check.evidence:
                rows.append(f"- `{check.name}` -> `{path}`")
    rows += ["", "## Reading this report", "",
             "- `NOT PRODUCED` means the stage wrote nothing; a stage folder only appears "
             "with its first result.",
             "- Zero accepted designs is not a failure of the deployment and is not "
             "end-to-end success either.",
             "- The lightweight mmCIF check is a text-level check only. The full structure "
             "parse is a separate check and reports `NOT TESTED` where no parser is available.",
             "- No file in this report was created by the validator, and no score in it was "
             "computed by the validator; every number is read back from the campaign's own "
             "output.",
             "- Computational acceptance is not experimental evidence of binding."]
    return "\n".join(rows) + "\n"


def emit(report: str, checks, arguments) -> None:
    sys.stdout.write(report)
    if arguments.output:
        Path(arguments.output).expanduser().write_text(report, encoding="utf-8")
        sys.stderr.write(f"validate_outputs: report written to {arguments.output}\n")
    if arguments.json:
        payload = [check.as_dict() for check in checks]
        Path(arguments.json).expanduser().write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        sys.stderr.write(f"validate_outputs: JSON written to {arguments.json}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a BindCraft2 campaign folder (read-only).")
    parser.add_argument("campaign", help="campaign project_folder to inspect")
    parser.add_argument("--slurm-log", action="append", default=[], metavar="PATH_OR_GLOB",
                        help="SLURM/stdout log to scan for JAX GPU evidence and the exit "
                             "code; repeatable")
    parser.add_argument("--output", metavar="FILE", help="write the markdown report here")
    parser.add_argument("--json", metavar="FILE", help="also write the checks as JSON")
    parser.add_argument("--no-full-parse", action="store_true",
                        help="skip the full mmCIF parse even if biotite/gemmi is importable")
    arguments = parser.parse_args(argv)

    campaign = Path(arguments.campaign).expanduser()
    checks: list = []
    if not campaign.is_dir():
        checks.append(Check(str(campaign), BLOCKED,
                            "campaign folder does not exist or is not readable from here; "
                            "nothing could be validated"))
        counts = dict(trajectories=0, refolded=0, ranked=0, sequences=0, complexes=0)
        blocked = (BLOCKED, "campaign folder unreadable")
        emit(render(campaign, checks, counts, blocked, blocked, arguments.slurm_log),
             checks, arguments)
        return 2

    trajectories = check_trajectories(campaign, checks)
    candidates = check_refolded(campaign, checks)
    accepted, sequences, complexes = check_ranked(campaign, checks,
                                                  not arguments.no_full_parse)
    check_summary(campaign, checks)
    check_metadata(campaign, checks)
    check_worker_logs(campaign, checks)
    evidence = check_slurm_logs(arguments.slurm_log, checks)

    counts = dict(trajectories=trajectories, refolded=candidates, ranked=accepted,
                  sequences=sequences, complexes=complexes)
    level_a, level_b = verdicts(counts, evidence)
    emit(render(campaign, checks, counts, level_a, level_b, arguments.slurm_log),
         checks, arguments)
    failed = any(check.status == FAIL for check in checks) \
        or level_a[0] == FAIL or level_b[0] == FAIL
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
