#!/usr/bin/env python3
"""Build the human CD19 extracellular-domain target file from an RCSB entry.

The CD19 campaign needs one chain of real, experimentally determined coordinates
with nothing else in the file: no Fab, no CD81, no glycan, no water and no
expression tag. This script performs exactly that selection on a downloaded PDB
entry and writes a provenance header into the result, so the target a campaign
ran against can always be traced back to a deposited structure.

Nothing is modelled, mutated, renumbered or filled in. Residues the depositors
could not place stay absent, and the file keeps the deposited author numbering
(UniProt P15391 numbering for 6AL5 chain A), because the hotspot and epitope
numbers used elsewhere in this project are that numbering.

    python lab/scripts/prepare_cd19_target.py --source 6AL5.pdb \
        --output lab/targets/CD19_ECD.pdb

Add --report to also print the antibody epitope contained in the source entry;
--fetch downloads the entry first and needs internet access, which the HPC
compute nodes do not have, so the prepared file is committed to the repository
instead.

Standard library only: this runs on a login node, a laptop or in the notebook
kernel without biotite, numpy or a GPU.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# 6AL5: CD19 (N138Q) in complex with the B43 Fab, X-ray, 3.0 A (Teplyakov et al.,
# Proteins 2018). Chain A is CD19; the construct carries a C-terminal GSHHHHHH
# expression tag modelled from residue 278, which is not part of CD19 and is
# therefore cut. Residue 277 is the last native CD19 residue present.
DEFAULT_ENTRY = "6AL5"
DEFAULT_CHAIN = "A"
DEFAULT_FIRST = 21
DEFAULT_LAST = 277

THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
                "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
                "TYR": "Y", "VAL": "V"}


def parse_atom_line(line: str) -> dict:
    return {"record": line[:6].strip(), "serial": line[6:11], "name": line[12:16].strip(),
            "altloc": line[16], "residue": line[17:20].strip(), "chain": line[21],
            "number": int(line[22:26]), "insertion": line[26],
            "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
            "line": line.rstrip("\n")}


def read_entry(path: Path) -> list[dict]:
    atoms = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                atoms.append(parse_atom_line(line))
            except ValueError:
                continue                       # an unparsable line is never guessed at
        elif line.startswith("ENDMDL"):
            break                              # first model only, if there are several
    return atoms


def fetch_entry(entry: str, destination: Path) -> Path:
    url = f"https://files.rcsb.org/download/{entry.upper()}.pdb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:   # noqa: S310 (fixed host)
        destination.write_bytes(response.read())
    return destination


def residue_spans(numbers) -> list[tuple[int, int]]:
    """Contiguous runs in a sorted list of residue numbers."""
    spans: list[tuple[int, int]] = []
    for number in numbers:
        if spans and number == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], number)
        else:
            spans.append((number, number))
    return spans


def one_letter(atoms: list[dict]) -> str:
    seen: dict[int, str] = {}
    for atom in atoms:
        seen.setdefault(atom["number"], THREE_TO_ONE.get(atom["residue"], "X"))
    return "".join(seen[number] for number in sorted(seen))


def contacts(selection: list[dict], partner: list[dict], cutoff: float = 4.0) -> dict:
    """{(number, residue): contact count} for heavy atoms within the cutoff."""
    squared = cutoff * cutoff
    found: dict[tuple[int, str], int] = defaultdict(int)
    for atom in selection:
        for other in partner:
            if ((atom["x"] - other["x"]) ** 2 + (atom["y"] - other["y"]) ** 2
                    + (atom["z"] - other["z"]) ** 2) <= squared:
                found[(atom["number"], atom["residue"])] += 1
    return dict(found)


def select_target(atoms: list[dict], chain: str, first: int, last: int) -> list[dict]:
    return [atom for atom in atoms
            if atom["record"] == "ATOM" and atom["chain"] == chain
            and first <= atom["number"] <= last
            and atom["residue"] in THREE_TO_ONE
            and atom["altloc"] in (" ", "A")]


def provenance(entry: str, chain: str, first: int, last: int, kept: list[dict],
               dropped: dict) -> list[str]:
    numbers = sorted({atom["number"] for atom in kept})
    spans = residue_spans(numbers)
    missing = [number for number in range(first, last + 1) if number not in set(numbers)]
    lines = [
        f"REMARK 999 Human CD19 extracellular domain, prepared from RCSB entry {entry}.",
        f"REMARK 999 Source chain {chain}, residues {first}-{last} of the deposited model,",
        "REMARK 999 written with the deposited author numbering unchanged (UniProt P15391",
        "REMARK 999 full-length numbering, signal peptide included in the count).",
        "REMARK 999 Prepared by lab/scripts/prepare_cd19_target.py. No coordinate was",
        "REMARK 999 modelled, mutated, renumbered, minimised or added.",
        f"REMARK 999 Residues present : {len(numbers)}",
        "REMARK 999 Ordered segments : " + ", ".join(f"{start}-{stop}" for start, stop in spans),
        "REMARK 999 Absent in source : " + (", ".join(f"{start}-{stop}" for start, stop
                                                      in residue_spans(missing)) or "none"),
    ]
    for label, count in sorted(dropped.items()):
        lines.append(f"REMARK 999 Removed          : {label} ({count} atoms)")
    lines.append("REMARK 999 Known construct differences are recorded in lab/targets/CD19_ECD.md.")
    return lines


def write_target(path: Path, header: list[str], kept: list[dict]) -> None:
    body = [*header]
    for serial, atom in enumerate(kept, start=1):
        body.append(f"{atom['line'][:6]}{serial:>5}{atom['line'][11:]}")
    last = kept[-1]
    body.append(f"TER   {len(kept) + 1:>5}      {last['residue']:>3} {last['chain']}"
                f"{last['number']:>4}{last['insertion']}")
    body.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    # The file is read on Linux: write LF whatever the host platform is.
    path.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")


def report_epitope(atoms: list[dict], kept: list[dict], chain: str) -> None:
    partners = sorted({atom["chain"] for atom in atoms
                       if atom["record"] == "ATOM" and atom["chain"] != chain})
    if not partners:
        print("no other polymer chain in the source entry, so no epitope is computed")
        return
    partner_atoms = [atom for atom in atoms
                     if atom["record"] == "ATOM" and atom["chain"] in partners]
    found = contacts(kept, partner_atoms)
    print(f"\npartner chains in the source entry : {', '.join(partners)}")
    print(f"epitope on chain {chain} within 4.0 A ({len(found)} residues):")
    for (number, residue), count in sorted(found.items()):
        print(f"  {THREE_TO_ONE.get(residue, 'X')}{number:<5} {residue}  contact atoms {count}")
    print("hotspot-style list: " + ",".join(str(number) for number, _ in sorted(found)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="a downloaded <entry>.pdb file")
    parser.add_argument("--fetch", action="store_true",
                        help="download the entry from RCSB first (needs internet access)")
    parser.add_argument("--entry", default=DEFAULT_ENTRY)
    parser.add_argument("--chain", default=DEFAULT_CHAIN)
    parser.add_argument("--first", type=int, default=DEFAULT_FIRST)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST,
                        help="last native residue to keep; anything after it is the tag")
    parser.add_argument("--output", default="lab/targets/CD19_ECD.pdb")
    parser.add_argument("--report", action="store_true",
                        help="also print the antibody epitope present in the source entry")
    arguments = parser.parse_args(argv)

    if arguments.fetch:
        source = Path(arguments.source or f"{arguments.entry.upper()}.pdb")
        print(f"downloading {arguments.entry.upper()} to {source}")
        fetch_entry(arguments.entry, source)
    elif arguments.source:
        source = Path(arguments.source)
    else:
        parser.error("give --source <entry>.pdb, or --fetch to download it")

    if not source.is_file():
        print(f"source not found: {source}", file=sys.stderr)
        return 2

    atoms = read_entry(source)
    if not atoms:
        print(f"no coordinates read from {source}", file=sys.stderr)
        return 2

    kept = select_target(atoms, arguments.chain, arguments.first, arguments.last)
    if not kept:
        print(f"no chain {arguments.chain} residues {arguments.first}-{arguments.last} "
              f"in {source}", file=sys.stderr)
        return 2

    dropped: dict[str, int] = defaultdict(int)
    selected = {id(atom) for atom in kept}
    for atom in atoms:
        if id(atom) in selected:
            continue
        if atom["record"] == "HETATM":
            dropped[f"heteroatom {atom['residue']}"] += 1
        elif atom["chain"] != arguments.chain:
            dropped[f"chain {atom['chain']}"] += 1
        elif atom["number"] > arguments.last:
            dropped[f"chain {atom['chain']} residue > {arguments.last} (expression tag)"] += 1
        elif atom["number"] < arguments.first:
            dropped[f"chain {atom['chain']} residue < {arguments.first}"] += 1

    output = Path(arguments.output)
    write_target(output, provenance(arguments.entry.upper(), arguments.chain, arguments.first,
                                    arguments.last, kept, dict(dropped)), kept)

    numbers = sorted({atom["number"] for atom in kept})
    print(f"source      : {source}  ({len(atoms)} atom records read)")
    print(f"target      : {output}  ({len(kept)} atoms, {len(numbers)} residues)")
    print(f"chain       : {arguments.chain}, residues {numbers[0]}-{numbers[-1]} "
          "(deposited author numbering, unchanged)")
    print("segments    : " + ", ".join(f"{start}-{stop}"
                                       for start, stop in residue_spans(numbers)))
    print("sequence    : " + one_letter(kept))
    for label, count in sorted(dropped.items()):
        print(f"removed     : {label} ({count} atoms)")
    if arguments.report:
        report_epitope(atoms, kept, arguments.chain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
