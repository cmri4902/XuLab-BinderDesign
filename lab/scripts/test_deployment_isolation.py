#!/usr/bin/env python3
"""Prove that a development deployment cannot write into the production tree.

The whole point of the isolated deployment is that nothing it does reaches
`/home/ypan/BindCraft2/BindCraft2_deploy` or `/project/yli4/ypan/bindcraft`. The
SLURM launcher is not edited for it, so the separation rests on exactly two
things: the `XULAB_*` variables the helper reads, and the `--export` /
`--output` / `--error` flags the submission states explicitly. This file tests
both, by loading the helper under each deployment's variables and checking every
path it would use.

Nothing is submitted: `submit_campaign` is called with `dry_run=True`, which
builds the real command and runs nothing.

    python lab/scripts/test_deployment_isolation.py

Exit 0 all assertions passed, 1 at least one failed.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "lab" / "notebooks"))

PRODUCTION_SOURCE = "/home/ypan/BindCraft2/BindCraft2_deploy"
PRODUCTION_PROJECT = "/project/yli4/ypan/bindcraft"
DEV_SOURCE = "/home/ypan/BindCraft2/BindCraft2_cd19_dev"
DEV_PROJECT = "/project/yli4/ypan/bindcraft_cd19_dev"

CHECKS = {"passed": 0, "failed": 0}


def check(condition, description: str) -> None:
    if condition:
        CHECKS["passed"] += 1
    else:
        CHECKS["failed"] += 1
        print(f"FAIL  {description}")


def equal(produced, expected, description: str) -> None:
    check(produced == expected, f"{description}: got {produced!r}, expected {expected!r}")


def load_helper(**environment):
    """Import xulab_hpc with a given XULAB_* environment, and restore it after."""
    keys = ["XULAB_DEPLOYMENT", "XULAB_BC2_HOME", "XULAB_PROJECT_ROOT", "XULAB_RESULTS",
            "XULAB_LOGS", "XULAB_CACHE", "XULAB_TMP", "XULAB_TARGETS", "XULAB_PYTHON",
            "BINDCRAFT_WEIGHTS", "BINDCRAFT_AF2_PARAMS"]
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    os.environ.update({key: value for key, value in environment.items() if value is not None})
    try:
        import xulab_hpc
        return importlib.reload(xulab_hpc), saved, keys
    except Exception:
        _restore(saved, keys)
        raise


def _restore(saved, keys) -> None:
    for key in keys:
        os.environ.pop(key, None)
        if saved.get(key) is not None:
            os.environ[key] = saved[key]


def dev_environment() -> dict:
    """Exactly what lab/scripts/dev_env.sh exports, read out of that file.

    Reading the shell file rather than restating its values is the point: if the
    env file and this test ever disagree, the test fails instead of passing
    against a copy of the values.
    """
    text = (REPOSITORY / "lab" / "scripts" / "dev_env.sh").read_text(encoding="utf-8")
    values: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        name, _, value = line[len("export "):].partition("=")
        value = value.strip().strip('"')
        # Resolve ${VAR:-default} against what has been read so far.
        while "${" in value:
            start = value.index("${")
            end = value.index("}", start)
            inner = value[start + 2:end]
            reference, _, default = inner.partition(":-")
            value = value[:start] + (values.get(reference) or default) + value[end + 1:]
        values[name] = value
    return values


def test_env_file_is_internally_consistent() -> None:
    values = dev_environment()
    equal(values.get("XULAB_DEPLOYMENT"), "cd19_dev", "dev_env.sh labels the deployment")
    equal(values.get("XULAB_BC2_HOME"), DEV_SOURCE, "dev_env.sh sets the dev source tree")
    equal(values.get("XULAB_PROJECT_ROOT"), DEV_PROJECT, "dev_env.sh sets the dev project root")
    for name, expected in (("XULAB_RESULTS", f"{DEV_PROJECT}/results"),
                           ("XULAB_LOGS", f"{DEV_PROJECT}/logs"),
                           ("XULAB_CACHE", f"{DEV_PROJECT}/cache"),
                           ("XULAB_TMP", f"{DEV_PROJECT}/tmp"),
                           ("XULAB_TARGETS", f"{DEV_PROJECT}/targets")):
        equal(values.get(name), expected, f"dev_env.sh sets {name} inside the dev project root")
    equal(values.get("JAX_COMPILATION_CACHE_DIR"), f"{DEV_PROJECT}/cache/jax",
          "the JAX cache is inside the dev project root")
    # Shared on purpose, and shared is not the same as copied.
    check(values.get("XULAB_PYTHON", "").startswith("/home/ypan/.conda/envs/bindcraft2"),
          "the conda interpreter is shared, not duplicated")
    check(PRODUCTION_PROJECT in values.get("BINDCRAFT_WEIGHTS", ""),
          "the model weights are read from the shared cache, not copied")
    for name, value in values.items():
        if name in ("XULAB_PYTHON", "BINDCRAFT_WEIGHTS", "XULAB_DEPLOYMENT"):
            continue
        check(not value.startswith(f"{PRODUCTION_PROJECT}/"),
              f"{name}={value} does not point inside the production project tree")
        check(value != PRODUCTION_SOURCE,
              f"{name}={value} is not the production source tree")


def test_helper_paths_follow_the_deployment() -> None:
    values = dev_environment()
    helper, saved, keys = load_helper(
        XULAB_DEPLOYMENT=values["XULAB_DEPLOYMENT"], XULAB_BC2_HOME=values["XULAB_BC2_HOME"],
        XULAB_PROJECT_ROOT=values["XULAB_PROJECT_ROOT"], XULAB_RESULTS=values["XULAB_RESULTS"],
        XULAB_LOGS=values["XULAB_LOGS"], XULAB_CACHE=values["XULAB_CACHE"],
        XULAB_TMP=values["XULAB_TMP"], XULAB_TARGETS=values["XULAB_TARGETS"])
    try:
        equal(helper.DEPLOYMENT, "cd19_dev", "the helper reports the dev deployment")
        paths = {"SOURCE_TREE": helper.SOURCE_TREE, "PROJECT_ROOT": helper.PROJECT_ROOT,
                 "RESULTS_ROOT": helper.RESULTS_ROOT, "LOGS_ROOT": helper.LOGS_ROOT,
                 "CACHE_ROOT": helper.CACHE_ROOT, "TMP_ROOT": helper.TMP_ROOT,
                 "TARGETS_ROOT": helper.TARGETS_ROOT, "SLURM_SCRIPT": helper.SLURM_SCRIPT,
                 "GENERATED_CONFIGS": helper.GENERATED_CONFIGS,
                 "JOB_REGISTRY": helper.JOB_REGISTRY, "PRESETS": helper.PRESETS,
                 "SCAFFOLDS": helper.SCAFFOLDS, "VALIDATOR": helper.VALIDATOR}
        for name, path in paths.items():
            text = str(path).replace("\\", "/")
            check(not text.startswith(f"{PRODUCTION_PROJECT}/") and text != PRODUCTION_PROJECT,
                  f"{name} ({text}) is outside the production project tree")
            check(not text.startswith(f"{PRODUCTION_SOURCE}/") and text != PRODUCTION_SOURCE,
                  f"{name} ({text}) is outside the production source tree")
        # The things that must specifically live in the dev tree.
        check(str(helper.GENERATED_CONFIGS).replace("\\", "/").startswith(DEV_SOURCE),
              "generated campaign JSON is written under the dev source tree")
        check(str(helper.JOB_REGISTRY).replace("\\", "/").startswith(DEV_PROJECT),
              "the job registry is written under the dev project root")
        check(str(helper.TARGETS_ROOT).replace("\\", "/").startswith(DEV_PROJECT),
              "prepared targets are written under the dev project root")
        check(str(helper.SLURM_SCRIPT).replace("\\", "/").startswith(DEV_SOURCE),
              "the launcher is the dev checkout's own copy")

    finally:
        _restore(saved, keys)

    # The presets live in the source tree, so exercising build_campaign_settings
    # needs a real checkout. This repository stands in for the dev clone, with the
    # dev project root: the combination a dev deployment actually runs with.
    helper, saved, keys = load_helper(XULAB_DEPLOYMENT="cd19_dev",
                                      XULAB_BC2_HOME=str(REPOSITORY),
                                      XULAB_PROJECT_ROOT=DEV_PROJECT)
    try:
        settings = helper.build_campaign_settings(
            campaign_name="isolation_check",
            targets=[{"name": "t", "target_path": "/some/target.pdb", "chains": "A"}],
            modality="scFv", number_of_final_designs=10, max_trajectories=500)
        # Separators are normalised because these tests also run on Windows,
        # where pathlib joins with a backslash; the notebook itself runs on Linux.
        equal(settings["project_folder"].replace("\\", "/"),
              f"{DEV_PROJECT}/results/isolation_check",
              "a generated campaign writes its results into the dev project root")
        # Careful: the dev project root has the production root as a string
        # prefix (`bindcraft_cd19_dev` vs `bindcraft`), so a bare substring test
        # would always "find" it. Only the production path followed by a
        # separator, or standing alone, counts.
        text = json.dumps(settings).replace("\\\\", "/")
        check(f"{PRODUCTION_PROJECT}/" not in text and f'"{PRODUCTION_PROJECT}"' not in text,
              "the generated settings name the production project tree nowhere")
        written = helper.GENERATED_CONFIGS
        check(str(written).replace("\\", "/").startswith(str(REPOSITORY).replace("\\", "/")),
              "generated settings files are written beside the source tree in use")
    finally:
        _restore(saved, keys)


def test_submission_states_every_path() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        # A fake dev tree, so the submission does not need the real HPC layout.
        source = root / "dev_source"
        (source / "lab" / "slurm").mkdir(parents=True)
        (source / "lab" / "slurm" / "run_bindcraft_h200.slurm").write_text(
            "#!/bin/bash\n", encoding="utf-8")
        project = root / "dev_project"
        config = root / "campaign.json"
        config.write_text(json.dumps({"campaign_name": "isolation_check", "modality": "scFv",
                                      "project_folder": str(project / "results"
                                                            / "isolation_check")}),
                          encoding="utf-8")

        helper, saved, keys = load_helper(
            XULAB_DEPLOYMENT="cd19_dev", XULAB_BC2_HOME=str(source),
            XULAB_PROJECT_ROOT=str(project), BINDCRAFT_WEIGHTS="/shared/weights",
            BINDCRAFT_AF2_PARAMS="/shared/weights/alphafold")
        try:
            submission = helper.submit_campaign(config, dry_run=True)
            command = " ".join(submission.command)
            equal(submission.command[0], "sbatch", "the submission is an sbatch command")
            check("--parsable" in command, "it asks sbatch for a parsable job id")
            check(f"XULAB_BC2_HOME={source}" in command,
                  "it states the source tree the job must execute")
            check(f"XULAB_PROJECT_ROOT={project}" in command,
                  "it states the project root the job must write to")
            check("BINDCRAFT_AF2_PARAMS=/shared/weights/alphafold" in command,
                  "it reuses the existing AlphaFold parameters")
            check("BINDCRAFT_WEIGHTS=/shared/weights" in command,
                  "it reuses the shared weight cache rather than making a new one")
            check("--output=" in command and "--error=" in command,
                  "it states the log paths, which overrides the launcher's own directives")
            check(str(project) in command.split("--output=")[1].split()[0],
                  "the stdout log goes into the dev project root")
            check(str(project) in command.split("--error=")[1].split()[0],
                  "the stderr log goes into the dev project root")
            check("srun" not in command, "no srun anywhere in the submission")
            # The production root is a string prefix of the dev root, so test for
            # it followed by a separator or standing at the end of a token.
            check(f"{PRODUCTION_PROJECT}/" not in command
                  and not command.endswith(PRODUCTION_PROJECT)
                  and f"{PRODUCTION_PROJECT} " not in command,
                  "the production project tree is named nowhere in the command")
            check(f"{PRODUCTION_SOURCE}/" not in command
                  and not command.endswith(PRODUCTION_SOURCE)
                  and f"{PRODUCTION_SOURCE} " not in command,
                  "the production source tree is named nowhere in the command")
            equal(submission.deployment, "cd19_dev", "the submission records its deployment")
            equal(submission.source_tree, str(source),
                  "the submission records the source tree it used")
            check((project / "logs").is_dir(),
                  "the log directory is created before the job, because SLURM opens it first")
            equal(submission.job_id, "", "a dry run submits nothing")
            check("nothing was submitted" in submission.message, "and says so")
        finally:
            _restore(saved, keys)


def test_production_submission_is_unchanged_in_substance() -> None:
    """With no XULAB_* set, every path must still be the production one."""
    helper, saved, keys = load_helper()
    try:
        equal(helper.DEPLOYMENT, "default", "the unconfigured deployment is the default one")
        equal(str(helper.SOURCE_TREE).replace("\\", "/"), PRODUCTION_SOURCE,
              "the default source tree")
        equal(str(helper.PROJECT_ROOT).replace("\\", "/"), PRODUCTION_PROJECT,
              "the default project root")
        equal(str(helper.RESULTS_ROOT).replace("\\", "/"), f"{PRODUCTION_PROJECT}/results",
              "the default results root")
        equal(str(helper.LOGS_ROOT).replace("\\", "/"), f"{PRODUCTION_PROJECT}/logs",
              "the default logs root")
        equal(str(helper.TARGETS_ROOT).replace("\\", "/"), f"{PRODUCTION_PROJECT}/targets",
              "the default targets root")
        # The launcher's own directives name exactly these log paths, so the flags
        # the submission adds are a restatement for production, not a change.
        launcher = (REPOSITORY / "lab" / "slurm" / "run_bindcraft_h200.slurm").read_text(
            encoding="utf-8")
        check(f"#SBATCH --output={PRODUCTION_PROJECT}/logs/xulab_bc2_%j.out" in launcher,
              "the launcher's --output directive is the production log path")
        check(f"#SBATCH --error={PRODUCTION_PROJECT}/logs/xulab_bc2_%j.err" in launcher,
              "the launcher's --error directive is the production log path")
    finally:
        _restore(saved, keys)


def test_launcher_is_untouched_and_srun_free() -> None:
    launcher = REPOSITORY / "lab" / "slurm" / "run_bindcraft_h200.slurm"
    text = launcher.read_text(encoding="utf-8")
    executed = [line for line in text.splitlines()
                if line.strip().startswith("srun") or " srun " in line
                and not line.strip().startswith("#")]
    equal(executed, [], "no line of the launcher invokes srun")
    check("#SBATCH --gres=gpu:h200:1" in text, "the H200 GRES request is unchanged")
    check("#SBATCH --nodelist=cmri-gpu05" in text, "the proven node pin is unchanged")
    check("#SBATCH --time=08:00:00" in text, "the wall clock is unchanged")
    check('XULAB_PROJECT_ROOT="${XULAB_PROJECT_ROOT:-' in text,
          "the launcher already reads XULAB_PROJECT_ROOT from the environment")
    check('XULAB_BC2_HOME="${XULAB_BC2_HOME:-' in text,
          "the launcher already reads XULAB_BC2_HOME from the environment")

    # The launcher must be identical to what is committed: this project's rule is
    # that the proven execution path is never edited.
    finished = subprocess.run(["git", "diff", "--quiet", "HEAD", "--",
                               "lab/slurm/run_bindcraft_h200.slurm"],
                              cwd=REPOSITORY, capture_output=True)
    if finished.returncode in (0, 1):
        equal(finished.returncode, 0,
              "lab/slurm/run_bindcraft_h200.slurm has no uncommitted change")
    else:
        print("  (git not available here, so the launcher's diff was not checked)")


def test_setup_script_refuses_production_paths() -> None:
    script = REPOSITORY / "lab" / "scripts" / "setup_dev_deployment.sh"
    check(script.is_file(), "the setup script exists")
    if not script.is_file():
        return
    environment = {**os.environ, "XULAB_BC2_HOME": PRODUCTION_SOURCE,
                   "XULAB_PROJECT_ROOT": PRODUCTION_PROJECT}
    finished = subprocess.run(["bash", str(script)], cwd=REPOSITORY, env=environment,
                              capture_output=True, text=True)
    if finished.returncode == 127 or "bash" in (finished.stderr or "")[:40].lower() \
            and "not found" in (finished.stderr or "").lower():
        print("  (bash not available here, so the setup script was not run)")
        return
    equal(finished.returncode, 2, "the setup script refuses the production paths")
    check("REFUSED" in finished.stdout, "and says it refused")
    check("nothing was created" in finished.stdout, "and that it created nothing")

    # With development paths it reports without creating anything.
    with tempfile.TemporaryDirectory() as temporary:
        environment = {**os.environ, "XULAB_BC2_HOME": f"{temporary}/src",
                       "XULAB_PROJECT_ROOT": f"{temporary}/project"}
        finished = subprocess.run(["bash", str(script)], cwd=REPOSITORY, env=environment,
                                  capture_output=True, text=True)
        check("would      mkdir -p" in finished.stdout,
              "without --create it only reports what it would do")
        check("never writes to" in finished.stdout,
              "and states that it does not write to the production deployment")
        equal(list(Path(temporary).iterdir()), [],
              "and creates nothing without --create")


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
