# `lab/notebooks/` - Jupyter interface (Phase 2)

| File | What it is |
| --- | --- |
| `xulab_bindcraft_h200.ipynb` | The notebook to open. Adapted from the official `../../bindcraft.ipynb` for **regular Jupyter on the CMRI HPC**: 10 steps, from environment check to campaign audit. |
| `xulab_hpc.py` | Everything the notebook does that is not a widget: paths, settings, submission, job tracking, result reading, export, validation. Plain Python, so it can be reviewed and tested outside a notebook. |

Open the notebook with a kernel from the `bindcraft2` conda environment:

```bash
# on the HPC, from the source copy
cd /home/ypan/BindCraft2/BindCraft2_deploy
jupyter lab            # or the institutional Jupyter/Workbench launcher
#   then open lab/notebooks/xulab_bindcraft_h200.ipynb
```

## What was changed from the Colab notebook

| Upstream Colab notebook | This notebook |
| --- | --- |
| mounts Google Drive | reads and writes `/project/yli4/ypan/bindcraft/` |
| `pip install`s into the runtime | uses the existing `bindcraft2` environment; installs nothing, and reports a missing module instead |
| `google.colab.output.enable_custom_widget_manager()` | plain `ipywidgets`; no `google.colab` import anywhere in the code |
| runs the campaign in the notebook process | `sbatch lab/slurm/run_bindcraft_h200.slurm <settings.json>` on an H200 |
| `files.download(zip)` | export folder plus a zip under `results/exports/`, with Jupyter `FileLink`s and the `scp` command |
| downloads AlphaFold parameters on demand | reuses the existing cluster copy through `BINDCRAFT_AF2_PARAMS` |
| one in-process campaign at a time | job ids recorded in `results/.xulab_jobs.json`, so a later session finds them again |

## The ten steps

1. environment report (paths, interpreter, `sbatch`, AlphaFold parameters, bundled checkpoints, which `bindcraft` the interpreter imports). Makes **no GPU claim**: a Jupyter node is not the batch allocation.
2. target, six ways: search RCSB by protein name, UniProt accession or PDB ID; enter a PDB ID; upload a .pdb/.cif/.fasta; paste a sequence; point at a file on the cluster; or pick a shipped preset. The form reports the entry's title, method and resolution, every entity's chain, length, organism and UniProt accession, the residue range present, the ordered and absent segments, the engineered mutations and expression tags the file declares, and the SIFTS mapping to the reference sequence - then waits for you to name the antigen chain. Nothing is pre-selected when a file holds more than one polymer chain. Hotspots and coldspots are optional. The prepared target and the settings entry are written for you. Several targets make it a multi-target campaign.
3. binder and campaign: binder type (10 modalities, opening on scFv as the default example), lengths, oligomer copies, property presets, designs wanted, trajectory budget, workers per GPU, resume. Writes a settings file to `lab/configs/generated/` of the source tree in use.
4. submit to SLURM, with an optional account, the parameter directory, and a dry-run switch.
5. job status (`squeue`, then `sacct`), log tail, cancel.
6. load any campaign - finished or still running - and read the ranked, candidate and attempt tables, with the filters that rejected candidates.
7. up to ten **accepted** candidates, best first, in the pipeline's own `i_pDAE` order: sequence, confidence readings, interface readings, contacted target residues and the structure file for each. For an antibody-format campaign (`scFv`, `Fab`, `VHH`) each design also gets its VH and VL named from the scaffold framework, its loops annotated with the convention they were read with, and its Kabat HCDR3. Fewer than ten are shown when fewer passed; nothing is substituted. The rejected candidates and their `failed_filters` stay underneath.
8. 3D review, **one complex at a time by default**: a dropdown of the accepted designs that actually have a file, labelled by rank and ID, with Previous/Next, `colour by chain` or `colour by pLDDT`, an optional HCDR3 or epitope highlight, and the rank, ID, filename and metrics beside the viewer. Compare View still draws up to 10 complexes in a grid. Needs `py3Dmol`.
9. export: binder FASTA, the CSVs, `campaign_metadata.json`, the accepted `.cif` complexes, and a zip - plus, for an antibody-format campaign, a VH/VL FASTA, the annotated loop table and the numbering record.
10. audit: runs `lab/scripts/validate_outputs.py` for the PASS/FAIL/BLOCKED/NOT TESTED report and the Level A / Level B verdicts.

## Deliberate omissions

- **No `no_filters` control.** That Colab shortcut turns off acceptance filtering; a campaign run that way produces numbers that mean nothing.
- **No design work in the kernel.** Closing the notebook never stops a campaign, and a notebook session can never be mistaken for GPU evidence.
- **No weight downloads, no package installs, no writes into the source checkout.**
- **No hosted/public viewer.** The upstream license restricts third-party hosted access; that needs institutional and licensing review first.

## Testing status

`xulab_hpc.py` and the notebook's code cells were exercised locally against synthetic fixtures
(90 assertions over both the pandas and the no-pandas table paths; all 10 code cells executed and
every widget callback fired, including a dry-run submission, a campaign load, an export with zip,
and a validator run). `nbformat` validates the notebook.

The target step and the scFv layer have their own runnable test scripts, all passing, none of
which needs a network, a GPU or an allocation:

```bash
python lab/scripts/test_scfv_annotation.py      # 128 assertions, 10 tests
python lab/scripts/test_target_tools.py         # 155 assertions, 13 tests
python lab/scripts/test_deployment_isolation.py # 118 assertions, 7 tests
python lab/scripts/test_notebook_cells.py       # cells 2, 3, 7, 8, 9 and every callback
```

They cover a campaign with a design whose structure file is missing, a de novo binder campaign
(to prove the PD-L1 path is unaffected), a campaign with nothing accepted, no campaign loaded at
all, every path where annotation or chain selection refuses to guess, both structure formats, an
RCSB outage, and the isolation of the development deployment.

**Not tested:** anything that needs the cluster - a real `sbatch`, `squeue`/`sacct` output, an H200
allocation, a real campaign, or `py3Dmol` rendering inside institutional Jupyter. No GPU run is
claimed anywhere.
