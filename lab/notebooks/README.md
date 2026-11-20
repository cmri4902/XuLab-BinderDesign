# `lab/notebooks/` - Jupyter interface (Phase 2)

| File | What it is |
| --- | --- |
| `xulab_bindcraft_h200.ipynb` | The notebook to open. Adapted from the official `../../bindcraft.ipynb` for **regular Jupyter on the CMRI HPC**: 10 steps, from environment check to campaign audit. |
| `xulab_hpc.py` | Everything the notebook does that is not a widget: paths, settings, submission, job tracking, result reading, export, validation. Plain Python, so it can be reviewed and tested outside a notebook. |

Open the notebook with a kernel from the `bindcraft2` conda environment:

```bash
# on the HPC, from the source copy
cd /project/yli4/ypan/bindcraft/BindCraft2
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
2. target: a shipped preset or your own PDB/mmCIF/FASTA on the cluster, with chains, hotspots and coldspots. Several targets make it a multi-target campaign.
3. binder and campaign: binder type (10 modalities), lengths, oligomer copies, property presets, designs wanted, trajectory budget, workers per GPU. Writes a settings file to `lab/configs/generated/`.
4. submit to SLURM, with an optional account, the parameter directory, and a dry-run switch.
5. job status (`squeue`, then `sacct`), log tail, cancel.
6. load any campaign - finished or still running - and read the ranked, candidate and attempt tables, with the filters that rejected candidates.
7. per accepted design: sequence, confidence readings, interface readings, and the structure files that belong to it.
8. 3D view of the predicted complexes with `py3Dmol`, coloured by chain or pLDDT.
9. export: binder FASTA, the CSVs, `campaign_metadata.json`, the accepted `.cif` complexes, and a zip.
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

**Not tested:** anything that needs the cluster - a real `sbatch`, `squeue`/`sacct` output, an H200
allocation, a real campaign, or `py3Dmol` rendering inside institutional Jupyter. No GPU run is
claimed anywhere.
