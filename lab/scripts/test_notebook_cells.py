#!/usr/bin/env python3
"""Execute the notebook's Top 10 and 3D-review cells against a fixture campaign.

A notebook cell that raises on the HPC wastes an allocation and a day, so the
widget cells are executed here in a plain Python process: the cell source is read
straight out of the `.ipynb`, run with `ipywidgets` and `pandas` present, and the
button callbacks are then invoked by hand against a fixture campaign folder.

What this proves: cells 7, 8 and 9 import, build their widgets, and their
callbacks run to completion on a campaign with accepted designs, on one with
nothing accepted, and on a de novo binder campaign. What it cannot prove: the
rendering, which needs a browser and `py3Dmol`; a missing `py3Dmol` is reported
by the cell as a readable message and counts as a pass here.

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

import xulab_hpc as hpc                                               # noqa: E402
import test_scfv_annotation as fixtures                               # noqa: E402

# Cells by their notebook id, so inserting a cell cannot silently shift the test.
CELL_IDS = ("xulab-04", "xulab-14", "xulab-16", "xulab-18")

FAILURES: list[str] = []


def report(description: str, failure: BaseException | None = None) -> None:
    if failure is None:
        print(f"  ok    {description}")
    else:
        FAILURES.append(description)
        print(f"  FAIL  {description}: {type(failure).__name__}: {failure}")
        traceback.print_exception(failure)


def cell_sources() -> dict:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    found = {cell.get("id"): "".join(cell["source"]) for cell in notebook["cells"]
             if cell["cell_type"] == "code"}
    missing = [name for name in CELL_IDS if name not in found]
    if missing:
        raise RuntimeError(f"the notebook has no code cell with id {missing}")
    return found


def notebook_namespace(campaign) -> dict:
    """The names cells 7-9 expect from the cells before them, with the campaign set."""
    import ipywidgets as widgets
    from IPython.display import display

    return {"widgets": widgets, "display": display, "hpc": hpc, "json": json, "Path": Path,
            "os": os, "sys": sys, "CAMPAIGN": campaign,
            "STYLE": {"description_width": "180px"},
            "wide": lambda width="560px": widgets.Layout(width=width)}


def run_cells(campaign, label: str) -> None:
    print(f"\n{label}")
    sources = cell_sources()
    namespace = notebook_namespace(campaign)
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            for name in CELL_IDS:
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

    # Compare View shares the cell with Individual View, so exercise it too.
    try:
        namespace["mode"].value = "Compare View"
        with redirect_stdout(io.StringIO()):
            namespace["on_mode"]()
            namespace["on_draw"]()
        report(f"{label}: cell 8 Compare View")
    except Exception as failure:                                      # noqa: BLE001
        report(f"{label}: cell 8 Compare View", failure)

    if campaign is not None and campaign.ranked:
        designs = namespace["AVAILABLE"]
        report(f"{label}: the viewer offered {len(designs)} design(s)")
        if namespace["how_many"].max != 10:
            report(f"{label}: Compare View limit is 10",
                   AssertionError(f"limit is {namespace['how_many'].max}"))
        else:
            report(f"{label}: Compare View limit is 10")


def main() -> int:
    heavy, light = fixtures.designed_scfv()
    other_heavy, other_light = fixtures.designed_scfv(hcdr3="WYRGPFDAYMDVKTGGYS")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        antibody = fixtures.write_fixture(root, "cd19_scfv_discovery", "scFv", [
            {"name": "design_a", "chains": [heavy, light]},
            {"name": "design_b", "chains": [other_heavy, other_light]},
            {"name": "design_c", "chains": [heavy, light], "structure": False},
        ])
        binder = fixtures.write_fixture(root, "pdl1_h200_smoke", "binder", [
            {"name": "pdl1_design_1", "chains": ["SPEELLRKAREWAKKHGLSEEEAKKLLEELRK"]},
        ])
        nothing = fixtures.write_fixture(root, "nothing_accepted", "scFv", [])

        run_cells(hpc.load_campaign(antibody), "scFv campaign with 2 of 3 designs on disk")
        run_cells(hpc.load_campaign(binder), "de novo binder campaign (the PD-L1 path)")
        run_cells(hpc.load_campaign(nothing), "campaign with nothing accepted")
        run_cells(None, "no campaign loaded")

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} step(s) failed)")
        return 1
    print("RESULT: PASS - every cell and callback ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
