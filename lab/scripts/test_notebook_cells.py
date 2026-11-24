#!/usr/bin/env python3
"""Execute the notebook's widget cells and drive every callback, offline.

A notebook cell that raises on the HPC wastes an allocation and a day, so the
widget cells are executed here in a plain Python process: the cell source is read
straight out of the `.ipynb` by cell id, run with `ipywidgets` and `pandas`
present, and the button callbacks are then invoked by hand.

Nothing here touches the network or a real deployment. RCSB lookups are replaced
by stubs that return recorded shapes and a local fixture file, and `XULAB_*` is
pointed at a temporary directory, so this is safe to run on a login node.

Covered: cell 2 (target: all six sources), cell 3 (binder and campaign settings,
including the written JSON), cell 7 (Top 10 report), cell 8 (Individual and
Compare view), cell 9 (export).

What it cannot prove: the rendering, which needs a browser and `py3Dmol`; a
missing `py3Dmol` is reported by the cell as a readable message and counts as a
pass here.

    python lab/scripts/test_notebook_cells.py

Exit 0 every cell and callback ran, 1 otherwise.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
NOTEBOOK = REPOSITORY / "lab" / "notebooks" / "xulab_bindcraft_h200.ipynb"
sys.path.insert(0, str(REPOSITORY / "lab" / "notebooks"))
sys.path.insert(0, str(REPOSITORY / "lab" / "scripts"))
os.environ["XULAB_BC2_HOME"] = str(REPOSITORY)
os.environ["XULAB_DEPLOYMENT"] = "test"

SCRATCH = tempfile.mkdtemp(prefix="xulab_notebook_test_")
os.environ["XULAB_TARGETS"] = str(Path(SCRATCH) / "targets")

import xulab_hpc as hpc                                               # noqa: E402
import test_scfv_annotation as fixtures                               # noqa: E402
import test_target_tools as target_tests                              # noqa: E402

# Reload so the module picks up XULAB_TARGETS set above, whatever import order
# the other test modules used.
import importlib                                                      # noqa: E402

importlib.reload(hpc)

# Cells by their notebook id, so inserting a cell cannot silently shift the test.
TARGET_CELL, CAMPAIGN_CELL = "xulab-04", "xulab-06"
RESULT_CELLS = ("xulab-14", "xulab-16", "xulab-18")
ALL_CELLS = (TARGET_CELL, CAMPAIGN_CELL) + RESULT_CELLS

FAILURES: list[str] = []
EXAMPLE_TARGET = REPOSITORY / "lab" / "targets" / "CD19_ECD.pdb"


def report(description: str, failure: BaseException | None = None) -> None:
    if failure is None:
        print(f"  ok    {description}")
    else:
        FAILURES.append(description)
        print(f"  FAIL  {description}: {type(failure).__name__}: {failure}")
        traceback.print_exception(failure)


def check(description: str, condition) -> None:
    report(description, None if condition else AssertionError("condition was false"))


def cell_sources() -> dict:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    found = {cell.get("id"): "".join(cell["source"]) for cell in notebook["cells"]
             if cell["cell_type"] == "code"}
    missing = [name for name in ALL_CELLS if name not in found]
    if missing:
        raise RuntimeError(f"the notebook has no code cell with id {missing}")
    return found


def stub_rcsb() -> None:
    """Replace the three network functions with offline stubs."""
    opener = target_tests.StubOpener({"entry/6AL5": target_tests.ENTRY_RESPONSE,
                                      **target_tests.ENTITY_RESPONSES,
                                      "rcsbsearch": target_tests.SEARCH_RESPONSE})
    # The real functions, called with a stub opener bound in: the routing,
    # parsing and error handling under test are the real ones, only the socket
    # is replaced.
    real_search, real_report = hpc.rcsb_search, hpc.rcsb_entry_report
    hpc.rcsb_search = lambda text, limit=10: real_search(text, limit, opener=opener)
    hpc.rcsb_entry_report = lambda pdb_id: real_report(pdb_id, opener=opener)
    # A download returns the repository's committed example target, a real
    # single-chain prepared file, so the cell gets genuine coordinates to describe.
    hpc.rcsb_download = lambda pdb_id, directory=None: (str(EXAMPLE_TARGET), "")


def notebook_namespace(campaign) -> dict:
    """The names the later cells expect from the cells before them."""
    import ipywidgets as widgets
    from IPython.display import display

    return {"widgets": widgets, "display": display, "hpc": hpc, "json": json, "Path": Path,
            "os": os, "sys": sys, "importlib": importlib, "CAMPAIGN": campaign,
            "STYLE": {"description_width": "180px"},
            "wide": lambda width="560px": widgets.Layout(width=width)}


def run_target_and_campaign_cells() -> None:
    """Cell 2 and cell 3: every target source, then the settings file."""
    print("\ncell 2 (target) and cell 3 (binder and campaign settings)")
    sources = cell_sources()
    namespace = notebook_namespace(None)
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            exec(compile(sources[TARGET_CELL], f"<{TARGET_CELL}>", "exec"), namespace)   # noqa: S102
            exec(compile(sources[CAMPAIGN_CELL], f"<{CAMPAIGN_CELL}>", "exec"), namespace)  # noqa: S102
    except Exception as failure:                                      # noqa: BLE001
        report("cells 2 and 3 execute", failure)
        print(captured.getvalue())
        return
    report("cells 2 and 3 execute")

    check("cell 3 defaults to the scFv binder type",
          namespace["binder_type"].value == "scFv")
    check("cell 3 defaults to 10 final designs and 500 trajectories",
          (namespace["final_designs"].value, namespace["trajectories"].value) == (10, 500))
    check("cell 2 defaults to the RCSB search route",
          namespace["source"].value.startswith("search RCSB"))
    check("cell 2 offers all six target sources", len(namespace["source"].options) == 6)

    # Every source option must lay the form out without raising.
    for option in namespace["source"].options:
        try:
            namespace["source"].value = option
            with redirect_stdout(io.StringIO()):
                namespace["on_source"]()
            report(f"cell 2 source {option!r} lays out")
        except Exception as failure:                                  # noqa: BLE001
            report(f"cell 2 source {option!r} lays out", failure)

    # --- RCSB search, stubbed ------------------------------------------------
    namespace["source"].value = "search RCSB by name / UniProt / PDB ID"
    namespace["on_source"]()
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            namespace["on_search"]()
        report("cell 2 'search RCSB' runs")
        text = output.getvalue()
        check("the search prints the entries it found", "6AL5" in text)
        check("the search prints each entry's title",
              "COMPLEX BETWEEN CD19" in text)
        check("the search prints each entity's chain and description",
              "B-lymphocyte antigen CD19" in text)
        check("the search populated the entry dropdown", len(namespace["hits"].options) >= 1)
    except Exception as failure:                                      # noqa: BLE001
        report("cell 2 'search RCSB' runs", failure)

    output = io.StringIO()
    try:
        with redirect_stdout(output):
            namespace["on_inspect"]()
        report("cell 2 'fetch and inspect' runs")
        text = output.getvalue()
        check("inspect prints the method and resolution", "X-RAY DIFFRACTION" in text)
        check("inspect prints the residue range present", "21-277" in text)
        check("inspect prints the unplaced segments", "40-47, 138-152, 177-183" in text)
        check("inspect prints the residue-number mapping", "UniProt P15391 21-277" in text)
        check("inspect prints the chain inventory", "227 residues" in text)
        check("a single-chain file is pre-selected", namespace["chain"].value == "A")
    except Exception as failure:                                      # noqa: BLE001
        report("cell 2 'fetch and inspect' runs", failure)

    # --- prepare and add from the inspected file -----------------------------
    namespace["name"].value = "test_target"
    namespace["hotspots"].value = "A162,A219,A220,A222"
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            namespace["on_add"]()
        report("cell 2 'prepare and add' runs")
        text = output.getvalue()
        check("the prepared target is reported with its chain and residue count",
              "residues    : 227" in text)
        check("the prepared target reports what was removed or that nothing was",
              "numbering   :" in text)
        check("the geometry report runs for the hotspots", "widest" in text)
        check("the geometry report refuses to claim accessibility",
              "NOT ESTABLISHED" in text)
        check("one target entry was added", len(namespace["TARGETS"]) == 1)
        entry = namespace["TARGETS"][0]
        check("the entry carries name, path, chain and hotspots",
              set(entry) == {"name", "target_path", "chains", "hotspots"})
        check("the entry's target file was written under the targets root",
              str(hpc.TARGETS_ROOT) in entry["target_path"])
    except Exception as failure:                                      # noqa: BLE001
        report("cell 2 'prepare and add' runs", failure)

    # --- an ambiguous chain must be refused, not guessed ---------------------
    two_chain = Path(SCRATCH) / "two_chain.pdb"
    target_tests.pdb_fixture(two_chain)
    namespace["source"].value = "a file on the cluster"
    namespace["on_source"]()
    namespace["file_path"].value = str(two_chain)
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            namespace["on_read_path"]()
        text = output.getvalue()
        check("a two-chain file lists both chains", "A  " in text and "B  " in text)
        check("a two-chain file pre-selects nothing",
              not namespace["chain"].value and "Nothing is " in text)
        before = len(namespace["TARGETS"])
        with redirect_stdout(output):
            namespace["on_add"]()
        check("adding without a chain is refused",
              len(namespace["TARGETS"]) == before
              and "does not guess" in output.getvalue())
        report("cell 2 refuses an ambiguous chain")
    except Exception as failure:                                      # noqa: BLE001
        report("cell 2 refuses an ambiguous chain", failure)

    # --- paste a sequence ----------------------------------------------------
    namespace["source"].value = "paste a sequence"
    namespace["on_source"]()
    namespace["pasted"].value = ">pasted\nACDEFGHIKLMNPQRSTVWY"
    namespace["name"].value = "pasted_example"
    namespace["hotspots"].value = ""
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            namespace["on_add"]()
        report("cell 2 'paste a sequence' adds a FASTA target")
        check("the FASTA target says hotspots are optional",
              "none given, which is allowed" in output.getvalue())
        check("two target entries are now queued", len(namespace["TARGETS"]) == 2)
        check("the FASTA entry carries no chain",
              "chains" not in namespace["TARGETS"][1])
    except Exception as failure:                                      # noqa: BLE001
        report("cell 2 'paste a sequence' adds a FASTA target", failure)

    # --- a shipped preset ----------------------------------------------------
    namespace["source"].value = "a shipped preset"
    namespace["on_source"]()
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            namespace["on_add"]()
        report("cell 2 'a shipped preset' adds an upstream target")
        check("three target entries are now queued", len(namespace["TARGETS"]) == 3)
    except Exception as failure:                                      # noqa: BLE001
        report("cell 2 'a shipped preset' adds an upstream target", failure)

    # --- upload with nothing chosen ------------------------------------------
    namespace["source"].value = "upload a file"
    namespace["on_source"]()
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            namespace["on_read_upload"]()
        check("an empty upload asks for a file", "choose a file first" in output.getvalue())
        report("cell 2 'upload a file' handles an empty upload")
    except Exception as failure:                                      # noqa: BLE001
        report("cell 2 'upload a file' handles an empty upload", failure)

    # --- the settings file ---------------------------------------------------
    while len(namespace["TARGETS"]) > 1:
        namespace["TARGETS"].pop()
    namespace["campaign_name"].value = "notebook_test_campaign"
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            namespace["on_write"]()
        report("cell 3 'write settings file' runs")
        path = namespace["CONFIG_PATH"]
        check("a settings file was written", path and Path(path).is_file())
        settings = json.loads(Path(path).read_text(encoding="utf-8"))
        check("the settings name the scFv modality", settings["modality"] == ["scFv"])
        check("the settings carry the prepared target",
              settings["targets"][0]["target_path"].endswith(".pdb"))
        check("the settings carry the hotspots as given",
              settings["targets"][0]["hotspots"] == "A162,A219,A220,A222")
        check("the settings ask for 10 designs and 500 trajectories",
              (settings["number_of_final_designs"], settings["max_trajectories"]) == (10, 500))
        check("the settings resume into the same folder", settings["resume"] is True)
        check("the project folder is under this deployment's results root",
              settings["project_folder"].startswith(str(hpc.RESULTS_ROOT)))
        check("no binder_lengths is written for a scaffolded format",
              "binder_lengths" not in settings)
        check("the deployment is printed with the settings",
              "deployment      : test" in output.getvalue())
    except Exception as failure:                                      # noqa: BLE001
        report("cell 3 'write settings file' runs", failure)

    # A de novo binder campaign must still write its lengths.
    namespace["binder_type"].value = "de novo binder"
    namespace["on_type"]()
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            namespace["on_write"]()
        settings = json.loads(Path(namespace["CONFIG_PATH"]).read_text(encoding="utf-8"))
        check("a de novo binder campaign writes binder_lengths",
              isinstance(settings.get("binder_lengths"), list))
        report("cell 3 still serves a de novo binder campaign")
    except Exception as failure:                                      # noqa: BLE001
        report("cell 3 still serves a de novo binder campaign", failure)

    # No target queued at all must be refused, not written.
    namespace["TARGETS"].clear()
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            namespace["on_write"]()
        check("writing with no target is refused", "add a target" in output.getvalue())
        report("cell 3 refuses to write without a target")
    except Exception as failure:                                      # noqa: BLE001
        report("cell 3 refuses to write without a target", failure)


def run_result_cells(campaign, label: str) -> None:
    """Cells 7, 8 and 9 against one campaign."""
    print(f"\n{label}")
    sources = cell_sources()
    namespace = notebook_namespace(campaign)
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            for name in (TARGET_CELL, *RESULT_CELLS):
                exec(compile(sources[name], f"<{name}>", "exec"), namespace)   # noqa: S102
    except Exception as failure:                                      # noqa: BLE001
        report(f"{label}: cells execute", failure)
        print(captured.getvalue())
        return
    report(f"{label}: cells execute")

    callbacks = [("cell 7 'show top candidates'", "on_top"),
                 ("cell 8 'refresh design list'", "on_refresh"),
                 ("cell 8 'show structure'", "on_draw"),
                 ("cell 8 Next", lambda: namespace["_step"](1)),
                 ("cell 8 Previous", lambda: namespace["_step"](-1)),
                 ("cell 9 'export results'", "on_export")]
    for description, callback in callbacks:
        function = namespace[callback] if isinstance(callback, str) else callback
        try:
            with redirect_stdout(io.StringIO()):
                function()
            report(f"{label}: {description}")
        except Exception as failure:                                  # noqa: BLE001
            report(f"{label}: {description}", failure)

    try:
        namespace["mode"].value = "Compare View"
        with redirect_stdout(io.StringIO()):
            namespace["on_mode"]()
            namespace["on_draw"]()
        report(f"{label}: cell 8 Compare View")
    except Exception as failure:                                      # noqa: BLE001
        report(f"{label}: cell 8 Compare View", failure)

    if campaign is not None and campaign.ranked:
        check(f"{label}: Compare View limit is 10", namespace["how_many"].max == 10)
        check(f"{label}: the viewer offered the designs that have a file",
              len(namespace["AVAILABLE"]) == len(hpc.accepted_designs(campaign, limit=10)))


def check_reporting_is_generic(campaign) -> None:
    """Step 7's report and step 8's chain labels must not be target-specific."""
    print("\ngeneric reporting")
    report_text = json.dumps(hpc.top_designs(campaign, limit=10), default=str)
    check("the Top 10 report names no particular antigen", "CD19" not in report_text)
    entries = hpc.accepted_designs(campaign, limit=10)
    labelled = hpc.structure_domains(entries[0]["path"],
                                     hpc.scaffold_reference("scFv"),
                                     hpc.campaign_target_sequences(campaign))
    labels = {entry["chain"]: entry["domain"] for entry in labelled}
    check("the target chain is identified from the campaign's own target file",
          labels.get("A", "").startswith("target ("))
    check("the VH chain is labelled VH", labels.get("B") == "VH")
    check("the VL chain is labelled VL", labels.get("C") == "VL")
    check("every chain label carries its evidence",
          all(entry["evidence"] for entry in labelled))
    first = hpc.top_designs(campaign, limit=1)["designs"][0]
    check("the report carries the predicted target contacts",
          first["target_residues"] != "")


def main() -> int:
    stub_rcsb()
    heavy, light = fixtures.designed_scfv()
    other_heavy, other_light = fixtures.designed_scfv(hcdr3="WYRGPFDAYMDVKTGGYS")
    root = Path(SCRATCH)

    run_target_and_campaign_cells()

    antibody = fixtures.write_fixture(root, "scfv_fixture", "scFv", [
        {"name": "design_a", "chains": [heavy, light], "epitope": "E1,P3"},
        {"name": "design_b", "chains": [other_heavy, other_light]},
        {"name": "design_c", "chains": [heavy, light], "structure": False},
    ], with_target_file=True)
    binder = fixtures.write_fixture(root, "binder_fixture", "binder", [
        {"name": "binder_design_1", "chains": ["SPEELLRKAREWAKKHGLSEEEAKKLLEELRK"]},
    ])
    nothing = fixtures.write_fixture(root, "nothing_accepted", "scFv", [])

    loaded = hpc.load_campaign(antibody)
    run_result_cells(loaded, "scFv campaign with 2 of 3 designs on disk")
    run_result_cells(hpc.load_campaign(binder), "de novo binder campaign (the PD-L1 path)")
    run_result_cells(hpc.load_campaign(nothing), "campaign with nothing accepted")
    run_result_cells(None, "no campaign loaded")
    check_reporting_is_generic(loaded)

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} step(s) failed)")
        return 1
    print("RESULT: PASS - every cell and callback ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
