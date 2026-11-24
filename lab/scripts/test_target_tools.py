#!/usr/bin/env python3
"""Local tests for the target-preparation and RCSB lookup helpers.

No network: every RCSB call is driven through a stub opener that replays recorded
response shapes, so these tests behave the same on an air-gapped compute node as
on a laptop. The structure parsing is tested against synthetic PDB and mmCIF
fixtures written here, plus the repository's own committed example target.

    python lab/scripts/test_target_tools.py

Exit 0 all assertions passed, 1 at least one failed.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import traceback
import urllib.error
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "lab" / "notebooks"))
os.environ["XULAB_BC2_HOME"] = str(REPOSITORY)

import xulab_hpc as hpc                                               # noqa: E402

CHECKS = {"passed": 0, "failed": 0}
EXAMPLE_TARGET = REPOSITORY / "lab" / "targets" / "CD19_ECD.pdb"


def check(condition, description: str) -> None:
    if condition:
        CHECKS["passed"] += 1
    else:
        CHECKS["failed"] += 1
        print(f"FAIL  {description}")


def equal(produced, expected, description: str) -> None:
    check(produced == expected, f"{description}: got {produced!r}, expected {expected!r}")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def pdb_fixture(path: Path) -> Path:
    """Two polymer chains, a gap, a ligand, a water, a tag and a declared mutation."""
    lines = [
        "HEADER    TEST STRUCTURE                          01-JAN-26   0TST",
        "TITLE     A TWO CHAIN TEST COMPLEX",
        "SEQADV 0TST GLN A    5  UNP  P00000    ASN     5 ENGINEERED MUTATION",
        "SEQADV 0TST HIS A   11  UNP  P00000              EXPRESSION TAG",
        "REMARK 465                                                                      ",
        "REMARK 465   M RES C SSSEQI                                                     ",
        "REMARK 465     GLY A     4                                                      ",
    ]
    # Chain A: residues 1-3, 5-10 present (4 missing), 11 is the tag. Chain B: 1-4.
    serial = 0
    for chain, numbers in (("A", [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]), ("B", [1, 2, 3, 4])):
        for number in numbers:
            serial += 1
            residue = "HIS" if (chain == "A" and number == 11) else "ALA"
            lines.append(f"ATOM  {serial:>5}  CA  {residue} {chain}{number:>4}    "
                         f"{float(number):>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00 30.00           C")
    serial += 1
    lines.append(f"HETATM{serial:>5}  C1  NAG A 901    "
                 f"{1.0:>8.3f}{1.0:>8.3f}{1.0:>8.3f}  1.00 40.00           C")
    serial += 1
    lines.append(f"HETATM{serial:>5}  O   HOH A 902    "
                 f"{2.0:>8.3f}{2.0:>8.3f}{2.0:>8.3f}  1.00 50.00           O")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def cif_fixture(path: Path) -> Path:
    """The same shape as the PDB fixture, in mmCIF, with a quoted details field."""
    lines = ["data_0TST", "#", "loop_",
             "_struct_ref_seq_dif.align_id", "_struct_ref_seq_dif.pdbx_pdb_strand_id",
             "_struct_ref_seq_dif.mon_id", "_struct_ref_seq_dif.pdbx_auth_seq_num",
             "_struct_ref_seq_dif.pdbx_seq_db_name",
             "_struct_ref_seq_dif.pdbx_seq_db_accession_code",
             "_struct_ref_seq_dif.db_mon_id", "_struct_ref_seq_dif.pdbx_seq_db_seq_num",
             "_struct_ref_seq_dif.details",
             "1 A GLN 5 UNP P00000 ASN 5 'engineered mutation'",
             "#", "loop_",
             "_pdbx_unobs_or_zero_occ_residues.id",
             "_pdbx_unobs_or_zero_occ_residues.auth_asym_id",
             "_pdbx_unobs_or_zero_occ_residues.auth_comp_id",
             "_pdbx_unobs_or_zero_occ_residues.auth_seq_id",
             "1 A GLY 4",
             "#", "loop_",
             "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
             "_atom_site.label_atom_id", "_atom_site.label_comp_id",
             "_atom_site.auth_asym_id", "_atom_site.auth_seq_id",
             "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
             "_atom_site.occupancy", "_atom_site.B_iso_or_equiv"]
    serial = 0
    for chain, numbers in (("A", [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]), ("B", [1, 2, 3, 4])):
        for number in numbers:
            serial += 1
            residue = "HIS" if (chain == "A" and number == 11) else "ALA"
            lines.append(f"ATOM {serial} C CA {residue} {chain} {number} "
                         f"{float(number)} 0.0 0.0 1.00 30.00")
    serial += 1
    lines.append(f"HETATM {serial} C C1 NAG A 901 1.0 1.0 1.0 1.00 40.00")
    lines.append("#")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


class StubOpener:
    """Replays canned RCSB responses, and refuses anything not registered."""

    def __init__(self, responses: dict, offline: bool = False):
        self.responses = responses
        self.offline = offline
        self.urls: list = []

    def __call__(self, request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        self.urls.append(url)
        if self.offline:
            raise urllib.error.URLError("Network is unreachable")
        for fragment, body in self.responses.items():
            if fragment in url:
                payload = body if isinstance(body, str) else json.dumps(body)
                return _Response(payload)
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)


class _Response(io.BytesIO):
    def __init__(self, text: str):
        super().__init__(text.encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


SEARCH_RESPONSE = {"result_set": [{"identifier": "6AL5"}, {"identifier": "7JIC"}]}
ENTRY_RESPONSE = {
    "struct": {"title": "COMPLEX BETWEEN CD19 (N138Q MUTANT) AND B43 FAB"},
    "exptl": [{"method": "X-RAY DIFFRACTION"}],
    "rcsb_entry_info": {"resolution_combined": [3.0]},
    "rcsb_accession_info": {"initial_release_date": "2018-03-28T00:00:00Z"},
    "citation": [{"title": "Crystal structure of CD19", "journal_abbrev": "Proteins",
                  "year": 2018}],
    "rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1", "2"]}}
ENTITY_RESPONSES = {
    "polymer_entity/6AL5/1": {
        "rcsb_polymer_entity": {"pdbx_description": "B-lymphocyte antigen CD19"},
        "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["A"],
                                                      "uniprot_ids": ["P15391"]},
        "entity_poly": {"rcsb_sample_sequence_length": 265,
                        "pdbx_seq_one_letter_code_can": "EEPLVVK"},
        "rcsb_entity_source_organism": [{"scientific_name": "Homo sapiens"}],
        "rcsb_polymer_entity_align": [{"reference_database_name": "UniProt",
                                       "reference_database_accession": "P15391",
                                       "aligned_regions": [{"entity_beg_seq_id": 1,
                                                            "ref_beg_seq_id": 21,
                                                            "length": 257}]}]},
    "polymer_entity/6AL5/2": {
        "rcsb_polymer_entity": {"pdbx_description": "B43 HEAVY CHAIN"},
        "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["H"]},
        "entity_poly": {"rcsb_sample_sequence_length": 233},
        "rcsb_entity_source_organism": [{"scientific_name": "Homo sapiens"}]}}


# --------------------------------------------------------------------------- #
# structure reading
# --------------------------------------------------------------------------- #

def test_format_detection() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        equal(hpc.structure_format(pdb_fixture(root / "a.pdb")), "pdb", "a .pdb is pdb")
        equal(hpc.structure_format(cif_fixture(root / "a.cif")), "cif", "a .cif is cif")
        # Content decides when the extension does not.
        mislabelled = root / "a.dat"
        mislabelled.write_text((root / "a.cif").read_text(encoding="utf-8"), encoding="utf-8")
        equal(hpc.structure_format(mislabelled), "cif", "an mmCIF with an odd extension")
        fasta = root / "a.txt"
        fasta.write_text(">x\nACDEFG\n", encoding="utf-8")
        equal(hpc.structure_format(fasta), "fasta", "a FASTA with a .txt extension")
        equal(hpc.structure_format(root / "missing.pdb"), "", "a missing file has no format")


def test_chain_inventory_both_formats() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for kind, path in (("pdb", pdb_fixture(root / "a.pdb")),
                           ("cif", cif_fixture(root / "a.cif"))):
            inventory = {entry["chain"]: entry for entry in hpc.chain_inventory(path)}
            equal(sorted(inventory), ["A", "B"], f"{kind}: both chains found")
            equal(inventory["A"]["residue_count"], 10, f"{kind}: chain A residue count")
            equal(inventory["A"]["first"], 1, f"{kind}: chain A first residue")
            equal(inventory["A"]["last"], 11, f"{kind}: chain A last residue")
            equal(inventory["A"]["segments"], [(1, 3), (5, 11)], f"{kind}: chain A segments")
            equal(inventory["A"]["gaps"], [(4, 4)], f"{kind}: the gap is reported, not filled")
            equal(inventory["A"]["sequence"], "AAAAAAAAAH", f"{kind}: chain A sequence")
            check("NAG" in inventory["A"]["hetero"], f"{kind}: the ligand is reported")
            equal(hpc.polymer_chains(path), ["A", "B"], f"{kind}: polymer chains, longest first")


def test_declared_records() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        differences = hpc.sequence_differences(pdb_fixture(root / "a.pdb"))
        equal(len(differences), 2, "both SEQADV rows are read")
        equal(differences[0]["note"], "ENGINEERED MUTATION", "the mutation is read as written")
        equal((differences[0]["residue"], differences[0]["reference_residue"]), ("GLN", "ASN"),
              "the mutation names both residues")
        equal(differences[1]["note"], "EXPRESSION TAG", "the tag row is read")
        equal(hpc.declared_unobserved(root / "a.pdb"), {"A": [4]},
              "REMARK 465 is read as the depositors wrote it")

        cif_differences = hpc.sequence_differences(cif_fixture(root / "a.cif"))
        equal(len(cif_differences), 1, "the mmCIF difference loop is read")
        equal(cif_differences[0]["note"], "engineered mutation",
              "a quoted mmCIF details field is unquoted correctly")
        equal(hpc.declared_unobserved(root / "a.cif"), {"A": [4]},
              "the mmCIF unobserved-residue loop is read")


def test_chain_is_never_guessed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        two = pdb_fixture(root / "a.pdb")
        chain, note = hpc.resolve_chain(two)
        equal(chain, "", "a file with two polymer chains refuses to choose")
        check("has to be named" in note, "the refusal says what is needed")
        check("A (10 residues)" in note and "B (4 residues)" in note,
              "the refusal lists the candidates with their sizes")
        check("longest chain is not assumed" in note,
              "the refusal says why the longest chain is not taken")
        equal(hpc.resolve_chain(two, "B")[0], "B", "a named chain is used")
        equal(hpc.resolve_chain(two, "Z")[0], "", "a chain that is not there is refused")
        check("not in this file" in hpc.resolve_chain(two, "Z")[1],
              "the refusal says the chain is absent")

        single = root / "one.pdb"
        single.write_text("\n".join(
            f"ATOM  {number:>5}  CA  ALA A{number:>4}    "
            f"{float(number):>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00 30.00           C"
            for number in range(1, 6)) + "\nEND\n", encoding="utf-8", newline="\n")
        chain, note = hpc.resolve_chain(single)
        equal(chain, "A", "a single-chain file needs no choice")
        check("only polymer chain" in note, "and says so")


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #

def test_prepare_target() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = pdb_fixture(root / "a.pdb")

        refused = hpc.prepare_target(source, output=root / "no.pdb")
        equal(refused["output"], "", "an ambiguous chain writes nothing")
        check(refused["problems"], "and reports why")
        check(not (root / "no.pdb").exists(), "no file is left behind by a refusal")

        report = hpc.prepare_target(source, chain="A", output=root / "t.pdb",
                                    name="test_target", first=1, last=10)
        equal(report["problems"], [], "a named chain prepares cleanly")
        equal(report["chain"], "A", "the chain is recorded")
        equal(report["residue_count"], 9, "the tag residue 11 is excluded by --last 10")
        equal(report["segments"], [(1, 3), (5, 10)], "the segments are recorded")
        equal(report["absent"], [(4, 4)], "the unplaced residue stays absent")
        equal(report["sequence"], "AAAAAAAAA", "the sequence is what is present")
        check("chain B" in report["removed"], "the other chain is removed and reported")
        check("heteroatom NAG" in report["removed"], "the ligand is removed and reported")
        check("heteroatom HOH" in report["removed"], "water is removed and reported")
        check("chain A residue > 10" in report["removed"], "the tag removal is reported")
        equal(len(report["differences"]), 2, "the chain's declared differences are carried")

        written = (root / "t.pdb").read_text(encoding="utf-8")
        check("REMARK 999" in written, "a provenance header is written into the file")
        check("Absent in source : 4" in written, "the header records the unplaced residue")
        check("ENGINEERED MUTATION" in written, "the header records the declared mutation")
        equal(written.count("\nATOM  "), 9, "one ATOM record per kept atom")
        check(written.rstrip().endswith("END"), "the file is terminated")
        check("\r" not in written, "the target is written with LF endings")
        # The written file must re-read as the same selection.
        again = hpc.chain_inventory(root / "t.pdb")
        equal(len(again), 1, "the prepared file holds exactly one chain")
        equal(again[0]["sequence"], report["sequence"], "the file round-trips")

        # mmCIF in, PDB out, same selection.
        from_cif = hpc.prepare_target(cif_fixture(root / "a.cif"), chain="A",
                                      output=root / "c.pdb", first=1, last=10)
        equal(from_cif["problems"], [], "an mmCIF source prepares cleanly")
        equal(from_cif["sequence"], report["sequence"], "mmCIF and PDB give the same selection")
        equal(from_cif["absent"], report["absent"], "and the same unplaced residues")

        empty = hpc.prepare_target(source, chain="A", output=root / "e.pdb",
                                   first=900, last=999)
        equal(empty["output"], "", "a range with nothing in it writes nothing")
        check(empty["problems"], "and reports why")


def test_prepare_refuses_unfit_data() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        wide = root / "wide.cif"
        wide.write_text("\n".join([
            "data_x", "loop_", "_atom_site.group_PDB", "_atom_site.label_atom_id",
            "_atom_site.label_comp_id", "_atom_site.auth_asym_id",
            "_atom_site.auth_seq_id", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "ATOM CA ALA AAA 1 1.0 0.0 0.0",
            "ATOM CA ALA AAA 2 2.0 0.0 0.0"]) + "\n", encoding="utf-8", newline="\n")
        report = hpc.prepare_target(wide, chain="AAA", output=root / "w.pdb")
        equal(report["output"], "", "a multi-character chain id is not written as PDB")
        check(any("fixed PDB columns" in problem for problem in report["problems"]),
              "the refusal explains the PDB column limit")
        check(any("mmCIF" in problem for problem in report["problems"]),
              "and names the alternative")


def test_example_target_is_what_it_claims() -> None:
    """The committed example target must still be the file its record describes."""
    check(EXAMPLE_TARGET.is_file(), f"the example target exists at {EXAMPLE_TARGET}")
    if not EXAMPLE_TARGET.is_file():
        return
    inventory = hpc.chain_inventory(EXAMPLE_TARGET)
    equal(len(inventory), 1, "the example target holds exactly one chain")
    entry = inventory[0]
    equal(entry["chain"], "A", "the example target's chain")
    equal(entry["residue_count"], 227, "the example target's residue count")
    equal((entry["first"], entry["last"]), (21, 277), "the example target's residue range")
    equal(entry["segments"], [(21, 39), (48, 137), (153, 176), (184, 277)],
          "the example target's ordered segments")
    equal(entry["gaps"], [(40, 47), (138, 152), (177, 183)],
          "the example target's unplaced segments")
    equal(entry["hetero"], {}, "no heteroatom survived preparation")
    header = EXAMPLE_TARGET.read_text(encoding="utf-8")[:4000]
    check("Residues present : 227" in header, "the header's residue count matches the file")
    check("Absent in source : 40-47, 138-152, 177-183" in header,
          "the header's absent segments match the file")
    check("UniProt P15391 21-277" in header, "the header records the reference mapping")
    chain, _note = hpc.resolve_chain(EXAMPLE_TARGET)
    equal(chain, "A", "a single-chain target needs no chain argument")


def test_geometry_notes_make_no_accessibility_claim() -> None:
    if not EXAMPLE_TARGET.is_file():
        return
    notes = hpc.geometry_notes(EXAMPLE_TARGET, "A", [162, 219, 220, 222])
    text = " ".join(notes)
    check("NOT ESTABLISHED" in text, "the notes refuse to claim accessibility")
    check("reachable on a cell" in text, "and say exactly what is not established")
    check(any("widest" in note for note in notes), "the span of the site is measured")
    check(any("first modelled residue" in note for note in notes),
          "the distance to the chain ends is measured")
    check(any("sequon" in note for note in notes), "glycosylation sequons are reported")
    check(not any("accessible" in note.lower() and "NOT ESTABLISHED" not in note
                  for note in notes),
          "no note asserts accessibility")
    # Residues 140-141 fall inside the example target's unplaced 138-152 segment.
    missing = hpc.geometry_notes(EXAMPLE_TARGET, "A", [140, 141])
    check("has a modelled CA atom" in " ".join(missing),
          "residues that are not modelled are reported, not measured")
    partly = hpc.geometry_notes(EXAMPLE_TARGET, "A", [140, 162, 220])
    check(any("not modelled, so excluded" in note for note in partly),
          "a mix of modelled and unplaced residues names the ones left out")


def test_fasta_and_upload() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        equal(hpc.read_fasta(">a\nACD\nEF\n>b\nGHI\n"), [("a", "ACDEF"), ("b", "GHI")],
              "a two-record FASTA is parsed")
        equal(hpc.read_fasta("acd ef\n"), [("sequence", "ACDEF")],
              "a bare pasted sequence is accepted and upper-cased")
        equal(hpc.read_fasta("   "), [], "blank text yields no record")

        report = hpc.write_fasta_target("my target", ">x\nACDEFGHIKL\n", root)
        equal(report["problems"], [], "a pasted sequence writes a FASTA target")
        equal(Path(report["output"]).name, "my_target.fasta", "the name is made filesystem-safe")
        equal(report["residue_count"], 10, "the length is reported")
        check("\r" not in Path(report["output"]).read_text(encoding="utf-8"),
              "the FASTA is written with LF endings")
        described = hpc.describe_structure(report["output"])
        equal(described["format"], "fasta", "the written file reads back as FASTA")
        check("no coordinates" in described["chain_note"],
              "a FASTA target says it has no coordinates, numbering or coverage")

        bad = hpc.write_fasta_target("x", ">y\nACDEFZZZ123\n", root)
        equal(bad["output"], "", "a non-protein sequence is refused")
        check(any("not a protein sequence" in problem for problem in bad["problems"]),
              "and says why")

        stored = hpc.save_upload("my file.pdb", b"ATOM\n", root / "uploads")
        equal(Path(stored["output"]).name, "my_file.pdb", "an upload name is made safe")
        equal(Path(stored["output"]).read_bytes(), b"ATOM\n", "the bytes are stored unchanged")
        refused = hpc.save_upload("payload.sh", b"#!/bin/sh\n", root / "uploads")
        equal(refused["output"], "", "an unexpected extension is refused")
        check(not (root / "uploads" / "payload.sh").exists(), "and nothing is written")
        traversal = hpc.save_upload("../../escape.pdb", b"ATOM\n", root / "uploads")
        equal(Path(traversal["output"]).parent, root / "uploads",
              "a path in the upload name cannot escape the upload directory")


# --------------------------------------------------------------------------- #
# RCSB lookup, driven through a stub
# --------------------------------------------------------------------------- #

def test_search_routing() -> None:
    opener = StubOpener({"rcsbsearch": SEARCH_RESPONSE})
    check(hpc.looks_like_pdb_id("6AL5") and not hpc.looks_like_pdb_id("CD19"),
          "a PDB ID is recognised and a protein name is not")
    check(hpc.looks_like_uniprot("P15391") and not hpc.looks_like_uniprot("CD19"),
          "a UniProt accession is recognised and a protein name is not")

    equal(hpc.rcsb_search("6al5", opener=opener), (["6AL5"], ""),
          "a PDB ID needs no search at all")
    equal(opener.urls, [], "and makes no network call")

    found, error = hpc.rcsb_search("CD19", opener=opener)
    equal((found, error), (["6AL5", "7JIC"], ""), "a name search returns the hits")
    payload = json.loads(_last_body(opener))
    equal(payload["query"]["service"], "full_text", "a name goes to the full-text service")

    hpc.rcsb_search("P15391", opener=opener)
    payload = json.loads(_last_body(opener))
    equal(payload["query"]["service"], "text",
          "a UniProt accession goes to the attribute service")
    check("database_accession" in payload["query"]["parameters"]["attribute"],
          "and is searched as an accession, not as free text")

    empty, error = hpc.rcsb_search("", opener=opener)
    equal(empty, [], "an empty query searches nothing")
    check(error, "and says what to type")
    nothing, error = hpc.rcsb_search("zzzz no such protein",
                                     opener=StubOpener({"rcsbsearch": {"result_set": []}}))
    equal(nothing, [], "a search with no hits returns nothing")
    check("found no experimental structure" in error, "and says so plainly")


_BODIES: list = []


def _last_body(opener) -> str:
    return _BODIES[-1] if _BODIES else "{}"


def test_entry_report() -> None:
    responses = {"entry/6AL5": ENTRY_RESPONSE, **ENTITY_RESPONSES}
    report, error = hpc.rcsb_entry_report("6al5", opener=StubOpener(responses))
    equal(error, "", "the entry report succeeds")
    equal(report["pdb_id"], "6AL5", "the id is normalised to upper case")
    equal(report["title"], "COMPLEX BETWEEN CD19 (N138Q MUTANT) AND B43 FAB", "the title")
    equal(report["methods"], ["X-RAY DIFFRACTION"], "the experimental method")
    equal(report["resolution"], 3.0, "the resolution")
    check("Proteins" in report["citation"], "the citation")
    equal(len(report["entities"]), 2, "both polymer entities are reported")
    first = report["entities"][0]
    equal(first["chains"], ["A"], "the entity's author chain ids")
    equal(first["uniprot"], ["P15391"], "the entity's UniProt accession")
    equal(first["organisms"], ["Homo sapiens"], "the source organism")
    equal(first["description"], "B-lymphocyte antigen CD19", "the entity description")
    equal(hpc.entity_of_chain(report, "A")["entity_id"], "1", "chain A maps to entity 1")
    equal(hpc.entity_of_chain(report, "H")["entity_id"], "2", "chain H maps to entity 2")
    equal(hpc.entity_of_chain(report, "Z"), {}, "an absent chain maps to no entity")

    mapping = hpc.numbering_mapping(hpc.entity_of_chain(report, "A"))
    check(any("UniProt P15391 21-277" in line for line in mapping),
          "the residue-number mapping to the reference sequence is stated")
    check(any("offset +20" in line for line in mapping), "with the offset")
    check(any("author numbering" in line for line in mapping),
          "and a note that the file's own numbering is what hotspots use")
    unmapped = hpc.numbering_mapping(hpc.entity_of_chain(report, "H"))
    check(any("no reference-sequence alignment" in line for line in unmapped),
          "an entity with no alignment says so rather than inventing a mapping")

    bad, error = hpc.rcsb_entry_report("nope", opener=StubOpener(responses))
    equal(bad, {}, "a malformed id is refused")
    check("not a 4-character PDB ID" in error, "and says why")


def test_offline_is_an_ordinary_outcome() -> None:
    offline = StubOpener({}, offline=True)
    found, error = hpc.rcsb_search("CD19", opener=offline)
    equal(found, [], "a search with no network returns nothing")
    check("no network access to search.rcsb.org" in error, "and names the host")
    check("copy the file across" in error, "and says what to do instead")

    report, error = hpc.rcsb_entry_report("6AL5", opener=offline)
    equal(report, {}, "an entry report with no network returns nothing")
    check("no network access" in error, "and says so")

    with tempfile.TemporaryDirectory() as temporary:
        path, error = hpc.rcsb_download("6AL5", Path(temporary), opener=offline)
        equal(path, "", "a download with no network writes nothing")
        check("no network access" in error, "and says so")
        equal(list(Path(temporary).iterdir()), [], "and leaves no partial file")

    missing = StubOpener({"entry/6AL5": ENTRY_RESPONSE})
    path, error = hpc.rcsb_download("6AL5", opener=missing)
    equal(path, "", "an entry with neither a .pdb nor a .cif file reports both failures")
    equal(error.count("HTTP 404"), 2, "naming each attempt")

    broken = StubOpener({"rcsbsearch": "not json at all"})
    found, error = hpc.rcsb_search("CD19", opener=broken)
    equal(found, [], "a non-JSON response yields no hits")
    check("not JSON" in error, "and is reported as such")


def test_download_writes_where_told() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        opener = StubOpener({"6AL5.pdb": "HEADER    TEST\nEND\n"})
        path, error = hpc.rcsb_download("6AL5", root, opener=opener)
        equal(error, "", "the download succeeds")
        equal(Path(path), root / "6AL5.pdb", "the file lands where it was told to")
        check("\r" not in Path(path).read_text(encoding="utf-8"), "written with LF endings")

        # No .pdb file for this entry: the .cif fallback is used, not treated as failure.
        cif_only = StubOpener({"6AL5.cif": "data_6AL5\n"})
        path, error = hpc.rcsb_download("6AL5", root, opener=cif_only)
        equal(error, "", "the mmCIF fallback succeeds")
        equal(Path(path).suffix, ".cif", "and returns the mmCIF")


def _patch_fetch_recorder() -> None:
    """Record POST bodies so the search tests can assert which service was used."""
    original = hpc._fetch

    def recording(url, payload=None, opener=None):
        if payload is not None:
            _BODIES.append(json.dumps(payload))
        return original(url, payload, opener)

    hpc._fetch = recording


def main() -> int:
    _patch_fetch_recorder()
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        try:
            test()
        except Exception:                                             # noqa: BLE001
            CHECKS["failed"] += 1
            print(f"ERROR in {test.__name__}:")
            traceback.print_exc()
    total = CHECKS["passed"] + CHECKS["failed"]
    print(f"\n{CHECKS['passed']}/{total} assertions passed across {len(tests)} tests")
    if CHECKS["failed"]:
        print(f"RESULT: FAIL ({CHECKS['failed']} assertion(s) failed)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
