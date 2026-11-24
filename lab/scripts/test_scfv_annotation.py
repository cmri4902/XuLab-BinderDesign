#!/usr/bin/env python3
"""Local tests for the scFv annotation, ranking and viewer helpers.

Runs anywhere the repository is checked out: standard library only, no SLURM, no
GPU, no pandas, no py3Dmol, no campaign. Every campaign it reads is a fixture
this file writes into a temporary directory, so running it cannot touch a real
results folder.

    python lab/scripts/test_scfv_annotation.py

Exit 0 all assertions passed, 1 at least one failed.

What is deliberately *not* tested here: anything that needs the HPC. These tests
prove the annotation logic, not that a campaign ran.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import traceback
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "lab" / "notebooks"))

# The helper reads the scaffold and the presets out of SOURCE_TREE, which
# defaults to the HPC path. Point it at this checkout before importing.
import os                                                             # noqa: E402

os.environ["XULAB_BC2_HOME"] = str(REPOSITORY)

import xulab_hpc as hpc                                               # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(condition, description: str) -> None:
    if condition:
        CHECKS["passed"] += 1
    else:
        CHECKS["failed"] += 1
        print(f"FAIL  {description}")


def equal(produced, expected, description: str) -> None:
    check(produced == expected, f"{description}: got {produced!r}, expected {expected!r}")


# --------------------------------------------------------------------------- #
# the shipped scFv preset and scaffold
# --------------------------------------------------------------------------- #

def test_paratope_spans() -> None:
    spans = hpc.paratope_spans("L24-29!,L30-32,L50-56!,H99-106(5-18),H72*")
    equal(sorted(spans), ["H", "L"], "mutate_positions names both chains")
    equal(spans["L"], [(24, 29, False), (30, 32, True), (50, 56, False)],
          "a span written with ! is paratope but not editable")
    equal(spans["H"], [(99, 106, True)], "a span written with * is framework, not paratope")
    equal(hpc.designed_positions("H72*,H99-101(2-4),L24-29!")["H"], {72, 99, 100, 101},
          "a * span is still a redesigned position, so it cannot be an anchor")
    check("L" not in hpc.designed_positions("L24-29!"),
          "a ! span contributes no redesigned position")
    equal(hpc.merged_loops([(24, 29, False), (30, 32, True), (33, 34, False), (50, 56, False)]),
          [(24, 34), (50, 56)], "adjacent paratope spans merge into one loop")


def test_scaffold_reference() -> None:
    reference = hpc.scaffold_reference("scFv")
    equal(sorted(reference), ["H", "L"], "the scFv scaffold has an H and an L chain")
    equal(reference["H"]["domain"], "VH", "scaffold chain H is the heavy variable domain")
    equal(reference["L"]["domain"], "VL", "scaffold chain L is the light variable domain")
    equal(len(reference["H"]["sequence"]), 119, "scaffold VH length")
    equal(len(reference["L"]["sequence"]), 107, "scaffold VL length")
    equal([loop["span"] for loop in reference["H"]["loops"]], ["H26-35", "H50-65", "H95-106"],
          "VH loop spans as the shipped preset declares them")
    equal([loop["span"] for loop in reference["L"]["loops"]], ["L24-34", "L50-56", "L89-97"],
          "VL loop spans as the shipped preset declares them")
    equal([loop["name"] for loop in reference["H"]["loops"]], ["HCDR1", "HCDR2", "HCDR3"],
          "VH loops are named HCDR1..3 in order")
    equal(reference["H"]["loops"][0]["scaffold_sequence"], "GFTFSSYAMS",
          "HCDR1 of the scaffold, by the preset span")
    equal(reference["H"]["loops"][2]["scaffold_sequence"], "YCARYDGIYGEL",
          "the H95-106 span carries the held YCAR stem")

    anchors = {loop["name"]: (before, after)
               for before, loop, after in hpc.domain_anchors(reference["H"])}
    equal(anchors["HCDR1"], ("LSCAAS", "WVRQAP"), "HCDR1 framework anchors")
    equal(anchors["HCDR2"], ("GLEWVS", "GRFTIS"), "HCDR2 framework anchors")
    equal(anchors["HCDR3"], ("EDTAVY", "DFWGQG"), "HCDR3 framework anchors")
    # H72 is a framework position the preset redesigns, so it must not appear in
    # an anchor: an anchor containing it would stop matching the moment it changed.
    check("GRFTIS" == anchors["HCDR2"][1] and len(anchors["HCDR2"][1]) == 6,
          "the anchor after HCDR2 stops before the redesigned framework position H72")
    light = {loop["name"]: (before, after)
             for before, loop, after in hpc.domain_anchors(reference["L"])}
    equal(light["LCDR1"], ("RVTITC", "WYQQKP"), "LCDR1 framework anchors")
    equal(light["LCDR3"], ("FATYYC", "FGQGTK"), "LCDR3 framework anchors")


def test_annotate_scaffold_itself() -> None:
    """Annotating the scaffold sequence must reproduce the preset's own spans."""
    reference = hpc.scaffold_reference("scFv")
    for chain in ("H", "L"):
        annotated = hpc.annotate_domain(reference[chain]["sequence"], reference[chain])
        equal(annotated["domain"], hpc.ANTIBODY_DOMAINS[chain], f"{chain}: domain name")
        for located, declared in zip(annotated["loops"], reference[chain]["loops"]):
            equal(located["sequence"], declared["scaffold_sequence"],
                  f"{declared['name']} located in the scaffold itself")
            equal(located["start"], declared["start"] - reference[chain]["numbers"][0] + 1,
                  f"{declared['name']} start position in the scaffold")
        equal(annotated["notes"], [], f"{chain}: nothing unavailable in the scaffold")


def designed_scfv(hcdr3: str = "WYRGPFDAYMDV", lcdr1_tail: str = "SSY",
                  hcdr1_middle: str = "GHW") -> tuple[str, str]:
    """A plausible designed VH/VL pair: scaffold frameworks, edited loops.

    Built by substituting into the scaffold sequences at the preset's editable
    positions, so the result is what a real design looks like to the annotator:
    identical framework, different loops, different HCDR3 length.
    """
    reference = hpc.scaffold_reference("scFv")
    heavy = list(reference["H"]["sequence"])
    # H31-33 are editable inside HCDR1 (scaffold numbering starts at 1).
    heavy[30:33] = list(hcdr1_middle)
    # H99-106 is the editable, length-variable HCDR3 loop.
    heavy = heavy[:98] + list(hcdr3) + heavy[106:]
    light = list(reference["L"]["sequence"])
    light[29:32] = list(lcdr1_tail)
    return "".join(heavy), "".join(light)


def test_annotate_designed_pair() -> None:
    reference = hpc.scaffold_reference("scFv")
    heavy_sequence, light_sequence = designed_scfv()

    # Chain order deliberately reversed: the domain must be identified by its
    # framework, not by the order the chains arrive in.
    domains = hpc.assign_domains([light_sequence, heavy_sequence], reference)
    equal([domain["domain"] for domain in domains], ["VL", "VH"],
          "VH/VL identified from the framework, with the chains supplied VL first")
    equal([domain["chain_position"] for domain in domains], [1, 2],
          "the output chain position is kept, so it can be matched to the structure")

    heavy = domains[1]
    loops = {loop["name"]: loop for loop in heavy["loops"]}
    equal(loops["HCDR3"]["sequence"], "YCAR" + "WYRGPFDAYMDV",
          "the H95-106 span of the design, stem included")
    # Scaffold HCDR1 is GFTFSSYAMS over H26-35; positions H31-33 (S,Y,A) are the
    # editable ones, so substituting GHW there leaves GFTFS + GHW + MS.
    equal(loops["HCDR1"]["sequence"], "GFTFS" + "GHW" + "MS", "HCDR1 of the design")
    equal(loops["HCDR2"]["sequence"], reference["H"]["loops"][1]["scaffold_sequence"],
          "HCDR2 was not edited in this fixture, so it matches the scaffold")
    equal(heavy["notes"], [], "every VH loop was located")

    kabat = hpc.kabat_hcdr3(heavy_sequence)
    equal(kabat["sequence"], "WYRGPFDAYMDVDF",
          "HCDR3 by the Kabat definition runs to the residue before W-G-x-G")
    equal(kabat["length"], 14, "HCDR3 length by the Kabat definition")
    equal(kabat["stem"], "CAR", "the Cys stem the HCDR3 was measured from")
    equal(kabat["motif"], "WGQG", "the framework motif the HCDR3 was measured to")
    equal(heavy_sequence[kabat["start"] - 1:kabat["end"]], kabat["sequence"],
          "the reported HCDR3 span indexes back to the reported sequence")

    # A longer loop must give a longer HCDR3, not a shifted framework.
    long_heavy, _ = designed_scfv(hcdr3="WYRGPFDAYMDVKTGGYS")
    equal(hpc.kabat_hcdr3(long_heavy)["length"], 20, "an 18-residue edit gives a 20 aa HCDR3")


def test_annotation_refuses_to_guess() -> None:
    reference = hpc.scaffold_reference("scFv")
    truncated = hpc.annotate_domain(reference["H"]["sequence"][:60], reference["H"])
    unavailable = [loop for loop in truncated["loops"] if loop["sequence"] is None]
    check(len(unavailable) >= 1, "a truncated domain reports loops as unavailable")
    check(all(loop["unavailable"] for loop in unavailable),
          "every unavailable loop carries a reason")
    check(truncated["notes"], "the domain records that not every loop was located")

    equal(hpc.kabat_hcdr3("PEPTIDEWITHOUTAMOTIF")["sequence"], None,
          "no W-G-x-G motif means no HCDR3 is reported")
    check(hpc.kabat_hcdr3("PEPTIDEWITHOUTAMOTIF")["unavailable"],
          "the missing HCDR3 says why")
    equal(hpc.kabat_hcdr3("")["sequence"], None, "an empty sequence yields no HCDR3")

    nonsense = hpc.assign_domains(["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"], reference)
    equal(nonsense[0]["domain"], "chain 1 (unidentified)",
          "a sequence with no antibody framework is not called VH or VL")
    equal(nonsense[0]["loops"], [], "an unidentified chain gets no loop annotation")


def test_paratope_split() -> None:
    reference = hpc.scaffold_reference("scFv")
    heavy_sequence, _ = designed_scfv()
    heavy = hpc.annotate_domain(heavy_sequence, reference["H"])
    loops = {loop["name"]: loop for loop in heavy["loops"]}

    inside = loops["HCDR3"]["start"] + 5
    framework_position = 10
    labels = (f"{heavy_sequence[inside - 1]}{inside},"
              f"{heavy_sequence[framework_position - 1]}{framework_position}")
    split = hpc.split_paratope(labels, heavy)
    equal(list(split["loops"]), ["HCDR3"], "a contact inside HCDR3 is attributed to HCDR3")
    equal(len(split["framework"]), 1, "a contact outside every loop is called framework")
    equal(split["unmapped"], [], "both labels mapped")
    equal(split["contacts"], 2, "every contact is counted exactly once")

    # A label whose residue letter does not match the sequence is a numbering
    # mismatch. It must be reported, never silently placed in a loop.
    wrong = "Z" if heavy_sequence[inside - 1] != "Z" else "A"
    mismatched = hpc.split_paratope(f"{wrong}{inside}", heavy)
    equal(mismatched["loops"], {}, "a mismatched label is not attributed to a loop")
    equal(mismatched["unmapped"], [f"{wrong}{inside}"], "a mismatched label is reported unmapped")
    equal(hpc.split_paratope(f"{heavy_sequence[0]}99999", heavy)["unmapped"], [f"{heavy_sequence[0]}99999"],
          "a position past the end of the sequence is unmapped")
    equal(hpc.split_paratope("", heavy)["contacts"], 0, "no paratope means no contacts")


# --------------------------------------------------------------------------- #
# campaign fixtures
# --------------------------------------------------------------------------- #

def write_fixture(root: Path, name: str, modality: str, designs: list) -> Path:
    """A minimal but realistically shaped campaign folder."""
    folder = root / name
    (folder / "3_Ranked").mkdir(parents=True, exist_ok=True)
    (folder / "2_Refolded").mkdir(parents=True, exist_ok=True)

    # Written with the csv module, because the real tables quote the fields that
    # hold commas (the paratope and epitope lists, and failed_filters).
    columns = ["rank", "design", "settings_modality", "Binder_Sequence",
               "Interface_Binder_Residues", "Interface_Target_Residues",
               "i_pDAE", "i_pTM", "i_pAE", "pLDDT", "pTM", "Interface_Residues",
               "Interface_BuriedArea", "Backbone_Clashes"]
    with (folder / "3_Ranked" / "!_Ranked.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, design in enumerate(designs, start=1):
            writer.writerow({"rank": index, "design": design["name"],
                             "settings_modality": modality,
                             "Binder_Sequence": "/".join(design["chains"]),
                             "Interface_Binder_Residues":
                                 "/".join(design.get("paratope", ["", ""])),
                             "Interface_Target_Residues": design.get("epitope", "D162,K220"),
                             "i_pDAE": "0.21", "i_pTM": "0.84", "i_pAE": "0.26",
                             "pLDDT": "0.88", "pTM": "0.81", "Interface_Residues": "14",
                             "Interface_BuriedArea": "812", "Backbone_Clashes": "0"})
    with (folder / "2_Refolded" / "!_Refolded.csv").open("w", encoding="utf-8",
                                                         newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["design", "settings_modality", "outcome",
                                                    "failed_filters", "i_pTM", "pLDDT"])
        writer.writeheader()
        writer.writerow({"design": "rejected_1", "settings_modality": modality,
                         "outcome": "rejected",
                         "failed_filters": "i_pTM,Interface_Residues",
                         "i_pTM": "0.41", "pLDDT": "0.62"})
        writer.writerow({"design": "rejected_2", "settings_modality": modality,
                         "outcome": "rejected", "failed_filters": "Backbone_Clashes",
                         "i_pTM": "0.55", "pLDDT": "0.71"})
    (folder / "campaign_metadata.json").write_text(
        json.dumps({"version": "test", "settings": {"modality": modality,
                                                    "campaign_name": name},
                    "metadata": {"settings_modality": modality}}, indent=2),
        encoding="utf-8", newline="\n")

    for design in designs:
        if not design.get("structure", True):
            continue
        atoms = ["data_test", "#", "loop_", "_atom_site.group_PDB",
                 "_atom_site.label_atom_id", "_atom_site.label_comp_id",
                 "_atom_site.auth_asym_id", "_atom_site.auth_seq_id",
                 "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
                 "_atom_site.B_iso_or_equiv"]
        letters = {"A": design.get("target_sequence", "EEPLVVK")}
        for letter, sequence in zip(("B", "C"), design["chains"]):
            letters[letter] = sequence
        reverse = {code: name for name, code in hpc.AMINO_ACID_CODES.items()}
        for letter, sequence in letters.items():
            for number, code in enumerate(sequence, start=1):
                atoms.append(f"ATOM CA {reverse.get(code, 'GLY')} {letter} {number} "
                             f"{number}.0 0.0 0.0 85.0")
        (folder / "3_Ranked" / f"{design['name']}.cif").write_text(
            "\n".join(atoms) + "\n", encoding="utf-8", newline="\n")
    return folder


def test_top_designs_on_fixture() -> None:
    heavy_sequence, light_sequence = designed_scfv()
    other_heavy, other_light = designed_scfv(hcdr3="WYRGPFDAYMDVKTGGYS", hcdr1_middle="NTW")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        folder = write_fixture(root, "cd19_scfv_discovery", "scFv", [
            {"name": "design_a", "chains": [heavy_sequence, light_sequence],
             "paratope": [f"{heavy_sequence[0]}1", f"{light_sequence[0]}1"]},
            {"name": "design_b", "chains": [other_heavy, other_light]},
        ])
        campaign = hpc.load_campaign(folder)
        equal(hpc.is_antibody_campaign(campaign), "scFv", "the modality is read from the campaign")
        equal(campaign.accepted_count, 2, "two accepted designs in the fixture")
        equal(len(campaign.candidates), 2, "the rejected candidates are still readable")
        equal(len(hpc.rejection_counts(campaign)), 3, "the rejection reasons are counted")

        report = hpc.top_designs(campaign, limit=10)
        equal(len(report["designs"]), 2, "ten were asked for and two exist")
        check(any("Nothing was substituted" in note for note in report["notes"]),
              "the report says it did not pad the list to ten")
        check(report["convention"], "the numbering convention is carried with the report")

        first = report["designs"][0]
        equal([domain["domain"] for domain in first["domains"]], ["VH", "VL"],
              "the first design's chains are identified as VH then VL")
        equal(first["structure"], "design_a.cif", "the design's structure file is found")
        equal(first["hcdr3"]["length"], 14, "the HCDR3 length of the first design")
        equal(sorted(first["confidence"]), ["i_pAE", "i_pDAE", "i_pTM", "pLDDT", "pTM"],
              "every recorded confidence reading is carried through")
        check("Interface_BuriedArea" in first["interface"], "buried area is carried through")
        equal(first["target_residues"], "D162,K220", "the contacted CD19 residues are carried")
        equal(report["designs"][1]["hcdr3"]["length"], 20, "the second design's HCDR3 length")

        columns, rows = hpc.cdr_table(campaign, limit=10)
        equal(columns, hpc.CDR_TABLE_COLUMNS, "the CDR table columns")
        equal(len(rows), 2 * (6 + 1), "six loops per design plus the Kabat HCDR3 row")
        kabat_rows = [row for row in rows if row["loop"] == "HCDR3 (Kabat)"]
        equal(len(kabat_rows), 2, "one Kabat HCDR3 row per design")
        equal(kabat_rows[0]["length"], 14, "the Kabat HCDR3 length reaches the table")
        check(all(row["sequence"] for row in rows), "every loop in the fixture was located")

        fasta = hpc.domain_fasta(campaign, limit=10)
        check(">design_a_VH " in fasta and ">design_a_VL " in fasta,
              "the FASTA names each record by its domain")
        equal(fasta.count(">"), 4, "two designs give four domain records")
        check(heavy_sequence in fasta, "the VH sequence is written whole")

        accepted = hpc.accepted_designs(campaign, limit=10)
        equal([entry["label"] for entry in accepted],
              ["rank 1 - design_a", "rank 2 - design_b"],
              "the viewer dropdown is built from the designs that exist")
        equal(len({entry["path"] for entry in accepted}), 2,
              "no structure file is offered twice under two ranks")

        reference = hpc.scaffold_reference("scFv")
        labelled = hpc.structure_domains(accepted[0]["path"], reference)
        equal(sorted((chain["chain"], chain["domain"]) for chain in labelled),
              [("A", "target / other"), ("B", "VH"), ("C", "VL")],
              "the viewer labels the structure chains CD19 / VH / VL")

        export = hpc.export_campaign(campaign, destination=root / "export", limit=10)
        for expected in ("cd19_scfv_discovery_VH_VL.fasta", "cd19_scfv_discovery_CDRs.csv",
                         "cd19_scfv_discovery_CDR_numbering.txt", "Ranked.csv",
                         "design_a.cif", "design_b.cif"):
            check(expected in export["files"], f"the export contains {expected}")
        check(export["zip"] and Path(export["zip"]).is_file(), "the export zip was written")


class RecordingView:
    """Stands in for a py3Dmol view, so the highlight logic is testable without it."""

    def __init__(self):
        self.styles: list = []

    def addStyle(self, selection, style):                             # noqa: N802 (py3Dmol API)
        self.styles.append((selection, style))


def test_highlights_only_draw_what_matches() -> None:
    heavy_sequence, light_sequence = designed_scfv()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        folder = write_fixture(root, "cd19_scfv_discovery", "scFv", [
            # The fixture target chain is EEPLVVK numbered 1-7, so E1 and P3 match
            # the file and D162 does not: only the two matching ones may be drawn.
            {"name": "design_a", "chains": [heavy_sequence, light_sequence],
             "epitope": "E1,P3,D162"},
        ])
        campaign = hpc.load_campaign(folder)
        reference = hpc.scaffold_reference("scFv")
        entry = hpc.accepted_designs(campaign, limit=10)[0]
        chains = hpc.structure_domains(entry["path"], reference)

        view = RecordingView()
        message = hpc._highlight_residues(view, campaign, entry, chains, reference, "epitope")
        check("2 contacted target residue(s)" in message,
              f"only the matching epitope residues are drawn: {message}")
        check("did not match the file" in message,
              "the unmatched epitope residue is reported, not drawn")
        equal([selection["resi"] for selection, _ in view.styles], [[1, 3]],
              "the drawn residue numbers are the ones that matched")

        view = RecordingView()
        message = hpc._highlight_residues(view, campaign, entry, chains, reference, "HCDR3")
        check(message.startswith("HCDR3 WYRGPFDAYMDVDF (14 aa)"),
              f"the HCDR3 highlight names the loop it drew: {message}")
        equal(len(view.styles), 1, "the HCDR3 highlight draws one selection")
        equal(view.styles[0][0]["resi"], list(range(99, 113)),
              "the HCDR3 highlight draws the located span")

        # Renumber the VH chain in the file: the highlight must refuse, not shift.
        structure = entry["path"]
        lines = []
        for line in structure.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if line.startswith("ATOM") and fields[3] == "B":
                fields[4] = str(int(fields[4]) + 500)
                line = " ".join(fields)
            lines.append(line)
        structure.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        campaign = hpc.load_campaign(folder)
        entry = hpc.accepted_designs(campaign, limit=10)[0]
        chains = hpc.structure_domains(entry["path"], reference)
        view = RecordingView()
        message = hpc._highlight_residues(view, campaign, entry, chains, reference, "HCDR3")
        check("not from 1" in message, f"a renumbered VH chain refuses the highlight: {message}")
        equal(view.styles, [], "nothing is drawn when the numbering cannot be mapped")

        equal(hpc._highlight_residues(RecordingView(), campaign, entry, chains, {}, "HCDR3"),
              "HCDR3 highlight skipped: no scaffold reference for this campaign",
              "without a scaffold reference the HCDR3 highlight is skipped")
        check("unknown highlight" in hpc._highlight_residues(RecordingView(), campaign, entry,
                                                             chains, reference, "nonsense"),
              "an unknown highlight name is reported")


def test_binder_campaign_is_untouched() -> None:
    """A de novo binder campaign must behave exactly as it did before."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        folder = write_fixture(root, "pdl1_h200_smoke", "binder", [
            {"name": "pdl1_design_1", "chains": ["SPEELLRKAREWAKKHGLSEEEAKKLLEELRK"]},
        ])
        campaign = hpc.load_campaign(folder)
        equal(hpc.is_antibody_campaign(campaign), "",
              "a de novo binder campaign is not an antibody campaign")

        report = hpc.top_designs(campaign, limit=10)
        equal(len(report["designs"]), 1, "the single accepted design is reported")
        equal(report["convention"], "", "no antibody numbering convention is claimed")
        equal([domain["domain"] for domain in report["designs"][0]["domains"]], ["chain 1"],
              "a de novo binder chain is not labelled VH or VL")
        equal(report["designs"][0]["domains"][0]["loops"], [], "no loops are invented")

        equal(hpc.binder_fasta(campaign).count(">"), 1, "the original FASTA still works")
        equal(hpc.cdr_table(campaign)[1], [], "no CDR table for a de novo campaign")
        accepted = hpc.accepted_designs(campaign, limit=10)
        equal(len(accepted), 1, "the viewer offers the one accepted design")
        labelled = hpc.structure_domains(accepted[0]["path"], None)
        equal([chain["domain"] for chain in labelled], ["target / other", "target / other"],
              "without a reference no chain is called a variable domain")

        export = hpc.export_campaign(campaign, destination=root / "export", limit=10)
        check(not any(name.endswith(("_VH_VL.fasta", "_CDRs.csv")) for name in export["files"]),
              "a de novo binder export gains no antibody files")
        check("pdl1_h200_smoke_binders.fasta" in export["files"],
              "the de novo binder FASTA is still exported")


def test_empty_and_partial_campaigns() -> None:
    heavy_sequence, light_sequence = designed_scfv()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        empty = write_fixture(root, "nothing_accepted", "scFv", [])
        campaign = hpc.load_campaign(empty)
        equal(campaign.accepted_count, 0, "no accepted designs")
        equal(hpc.top_designs(campaign, limit=10)["designs"], [], "no designs are invented")
        equal(hpc.accepted_designs(campaign, limit=10), [], "the viewer offers nothing")
        equal(hpc.domain_fasta(campaign), "", "no FASTA from nothing")
        try:
            hpc.show_structure(campaign, 0)
        except RuntimeError as refused:
            check("no accepted complex files" in str(refused),
                  "the viewer refuses an empty campaign with a readable reason")
        except ImportError:                              # py3Dmol absent is also fine
            check(True, "the viewer refused before needing py3Dmol")
        else:
            check(False, "the viewer must refuse an empty campaign")

        partial = write_fixture(root, "one_file_missing", "scFv", [
            {"name": "design_a", "chains": [heavy_sequence, light_sequence]},
            {"name": "design_b", "chains": [heavy_sequence, light_sequence],
             "structure": False},
        ])
        campaign = hpc.load_campaign(partial)
        equal(campaign.accepted_count, 2, "both designs are in the ranked table")
        accepted = hpc.accepted_designs(campaign, limit=10)
        equal([entry["design"] for entry in accepted], ["design_a"],
              "a design with no structure file is not offered to the viewer")
        report = hpc.top_designs(campaign, limit=10)
        equal(report["designs"][1]["structure"], "",
              "the design without a file is still reported, with no file named")


def main() -> int:
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
