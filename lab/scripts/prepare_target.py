#!/usr/bin/env python3
"""Prepare a binder-design target from any experimental structure.

A binder campaign needs one chain of real coordinates with nothing else in the
file: no other protein, no antibody, no ligand, no glycan, no water and no
expression tag. This script does that selection for any antigen, from a local
file or straight from RCSB, and writes a provenance header into the result so
the target a campaign ran against can always be traced back to a deposition.

Nothing is modelled, mutated, renumbered, minimised or filled in. Residues the
depositors could not place stay absent, and the file keeps the deposited author
numbering, because that is the numbering hotspots have to be given in.

It is the command-line face of the same functions the notebook's target step
uses (`xulab_hpc.describe_structure`, `prepare_target`, `rcsb_*`), so both
routes produce the same file.

Look before selecting - this never picks a chain for you when the file holds
more than one polymer chain:

    python lab/scripts/prepare_target.py --entry 6AL5 --describe

Then prepare it:

    python lab/scripts/prepare_target.py --entry 6AL5 --chain A \
        --first 21 --last 277 --name hCD19_ECD \
        --output lab/targets/CD19_ECD.pdb --contacts

The example above is the human CD19 extracellular domain this platform ships as
its default example target; `lab/targets/CD19_ECD.md` records the result.
`--entry` needs internet access, which the HPC compute nodes do not have; use
`--source` with an already-downloaded file there.

Standard library only: this runs on a login node, a laptop or in the notebook
kernel without biotite, numpy or a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import xulab_hpc as hpc                                              # noqa: E402


def print_description(described: dict, entry: dict | None = None) -> None:
    print(f"file        : {described['path']}")
    print(f"format      : {described['format'] or 'unrecognised'}")
    if entry:
        print(f"RCSB entry  : {entry['pdb_id']}  {entry['title']}")
        print(f"method      : {', '.join(entry['methods']) or 'unknown'}"
              + (f", {entry['resolution']} A" if entry.get("resolution") else ""))
        if entry.get("citation"):
            print(f"citation    : {entry['citation']}")
        for item in entry.get("entities", []):
            print(f"  entity {item['entity_id']}  chain(s) {','.join(item['chains']) or '?'}"
                  f"  {item.get('length') or '?'} aa"
                  f"  {'/'.join(item.get('uniprot') or []) or 'no UniProt reference'}"
                  f"  {'; '.join(item.get('organisms') or []) or 'organism not stated'}")
            print(f"            {item.get('description') or '(no description)'}")

    print("\nchains present in the coordinates:")
    for chain in described["chains"]:
        if not chain.get("polymer"):
            print(f"  {chain['chain']}  no polymer residues "
                  f"({chain.get('atoms', 0)} atoms, heteroatoms only)")
            continue
        print(f"  {chain['chain']}  {chain['residue_count']:>4} residues  "
              f"{chain['first']}-{chain['last']}")
        print(f"       segments : {hpc.span_text(chain['segments'])}")
        if chain["gaps"]:
            print(f"       gaps     : {hpc.span_text(chain['gaps'])}")
        if chain["hetero"]:
            print("       hetero   : " + ", ".join(f"{name} x{count}" for name, count
                                                   in sorted(chain["hetero"].items())))
        if chain["insertion_codes"]:
            print(f"       insertion codes : {', '.join(chain['insertion_codes'])}")

    differences = described.get("differences") or []
    print(f"\nengineered differences the file declares: {len(differences)}")
    for difference in differences[:30]:
        print(f"  chain {difference['chain'] or '?'}  {difference['residue']}"
              f"{difference['number']} vs {difference['database']} "
              f"{difference['accession']} {difference['reference_residue']}"
              f"{difference['reference_number']}  {difference['note']}")
    if len(differences) > 30:
        print(f"  ... and {len(differences) - 30} more")

    declared = described.get("declared_unobserved") or {}
    if declared:
        print("\nresidues the depositors declare as not located:")
        for chain, numbers in sorted(declared.items()):
            print(f"  chain {chain}: {hpc.span_text(hpc.residue_spans(numbers))}")

    if described.get("chain"):
        print(f"\nchain to use : {described['chain']}  ({described['chain_note']})")
    else:
        print(f"\nchain to use : NOT CHOSEN - {described['chain_note']}")
        print("Name it with --chain <letter>. This script does not guess.")


def report_contacts(source, chain: str, kept_range) -> None:
    """The epitope the source entry itself contains, if it has partner chains."""
    atoms = hpc.structure_atoms(source)
    first, last = kept_range
    target = [atom for atom in atoms
              if atom["record"] == "ATOM" and atom["chain"] == chain
              and atom["residue"] in hpc.STANDARD_RESIDUES
              and (first is None or atom["number"] >= first)
              and (last is None or atom["number"] <= last)]
    partners = sorted({atom["chain"] for atom in atoms
                       if atom["record"] == "ATOM" and atom["chain"] != chain
                       and atom["residue"] in hpc.STANDARD_RESIDUES})
    if not partners:
        print("\nno other polymer chain in the source file, so it contains no epitope to report")
        return
    partner_atoms = [atom for atom in atoms
                     if atom["record"] == "ATOM" and atom["chain"] in partners]
    counts: dict = {}
    for atom in target:
        for other in partner_atoms:
            if ((atom["x"] - other["x"]) ** 2 + (atom["y"] - other["y"]) ** 2
                    + (atom["z"] - other["z"]) ** 2) <= 16.0:
                key = (atom["number"], atom["residue"])
                counts[key] = counts.get(key, 0) + 1
    print(f"\npartner chains in the source file : {', '.join(partners)}")
    print(f"residues of chain {chain} within 4.0 A of them ({len(counts)}):")
    for (number, residue), count in sorted(counts.items()):
        print(f"  {hpc.STANDARD_RESIDUES.get(residue, 'X')}{number:<5} {residue}"
              f"  contact atoms {count}")
    print("hotspot-style list: " + ",".join(f"{chain}{number}" for number, _ in sorted(counts)))
    print("This is the interface present in THIS deposition. It is evidence that the site is "
          "antibody-addressable in this construct, not that it is reachable on a cell.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="a local .pdb, .cif or .fasta file")
    source.add_argument("--entry", help="a 4-character RCSB PDB ID to download (needs internet)")
    parser.add_argument("--search", help="print RCSB hits for a protein name or UniProt "
                                         "accession and stop")
    parser.add_argument("--describe", action="store_true",
                        help="print what the file contains and stop, without writing anything")
    parser.add_argument("--chain", default="", help="the antigen chain; required when the file "
                                                    "holds more than one polymer chain")
    parser.add_argument("--first", type=int, default=None, help="first residue to keep")
    parser.add_argument("--last", type=int, default=None,
                        help="last residue to keep; use it to cut a C-terminal expression tag")
    parser.add_argument("--name", default="", help="target name, also the output file stem")
    parser.add_argument("--output", default="", help="where to write; default is "
                                                     "<targets root>/<name>.pdb")
    parser.add_argument("--download-dir", default="",
                        help="where --entry puts the downloaded file")
    parser.add_argument("--contacts", action="store_true",
                        help="also report the interface the source file itself contains")
    arguments = parser.parse_args(argv)

    if arguments.search:
        found, error = hpc.rcsb_search(arguments.search, limit=10)
        if error:
            print(error, file=sys.stderr)
            return 3
        print(f"RCSB entries for {arguments.search!r}: {', '.join(found)}")
        print("Inspect one with:  --entry <id> --describe")
        return 0

    entry_report = None
    if arguments.entry:
        entry_report, error = hpc.rcsb_entry_report(arguments.entry)
        if error:
            print(error, file=sys.stderr)
            return 3
        path, error = hpc.rcsb_download(arguments.entry,
                                        arguments.download_dir or None)
        if error:
            print(error, file=sys.stderr)
            return 3
        print(f"downloaded {arguments.entry.upper()} to {path}\n")
        located = Path(path)
    else:
        located = Path(arguments.source)

    described = hpc.describe_structure(located, arguments.chain)
    if arguments.describe:
        print_description(described, entry_report)
        for problem in described["problems"]:
            print(f"\nproblem: {problem}", file=sys.stderr)
        return 1 if described["problems"] else 0

    if described["problems"]:
        print_description(described, entry_report)
        for problem in described["problems"]:
            print(f"problem: {problem}", file=sys.stderr)
        return 2
    if not described["chain"]:
        print_description(described, entry_report)
        return 2

    provenance = []
    if entry_report:
        provenance.append(f"RCSB entry {entry_report['pdb_id']}: {entry_report['title']}")
        provenance.append(f"Method: {', '.join(entry_report['methods'])}"
                          + (f", {entry_report['resolution']} A"
                             if entry_report.get("resolution") else ""))
        entity = hpc.entity_of_chain(entry_report, described["chain"])
        if entity:
            provenance.append(f"Entity: {entity.get('description', '')}".rstrip())
            for line in hpc.numbering_mapping(entity):
                provenance.append(line)

    prepared = hpc.prepare_target(located, chain=described["chain"],
                                  output=arguments.output or None, name=arguments.name,
                                  first=arguments.first, last=arguments.last,
                                  provenance=provenance)
    if prepared["problems"]:
        for problem in prepared["problems"]:
            print(f"problem: {problem}", file=sys.stderr)
        return 2

    print(f"source      : {located}")
    print(f"target      : {prepared['output']}")
    print(f"name        : {prepared['name']}")
    print(f"chain       : {prepared['chain']}  ({prepared['chain_note']})")
    print(f"residues    : {prepared['residue_count']}")
    print(f"segments    : {hpc.span_text(prepared['segments'])}")
    print(f"absent      : {hpc.span_text(prepared['absent'])}")
    print(f"sequence    : {prepared['sequence']}")
    for item, count in sorted(prepared["removed"].items()):
        print(f"removed     : {item} ({count} atoms)")
    for difference in prepared["differences"]:
        print(f"declared    : {difference['residue']}{difference['number']} vs "
              f"{difference['reference_residue']}{difference['reference_number']} "
              f"{difference['note']}")
    print("\nUse it in a campaign settings file as:")
    # json.dumps, so a Windows path is escaped into valid JSON rather than
    # printed with bare backslashes that would not parse.
    print('  "targets": [' + json.dumps({"name": prepared["name"],
                                         "target_path": prepared["output"],
                                         "chains": prepared["chain"]}) + "]")
    print("hotspots and coldspots are optional; give them in this file's own numbering.")

    if arguments.contacts:
        report_contacts(located, prepared["chain"], (arguments.first, arguments.last))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
