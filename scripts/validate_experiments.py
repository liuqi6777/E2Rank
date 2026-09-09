"""Dry-run every experiment script and parse each emitted command through the real
argument pipeline, so a config or flag typo fails here instead of on a GPU node.

    uv run python scripts/validate_experiments.py

Launches nothing and touches no checkpoints. bf16/tf32/deepspeed are forced off because
they assert on hardware this check does not need.
"""
import shlex, subprocess, sys, os, re, pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))
from transformers import HfArgumentParser
from config import (BaselineArguments, DataArguments, LoraArguments, ModelArguments,
                    MTEBEvalArguments, RLArguments, TrainingArguments)
import train as train_mod
import train_baseline as tb_mod
from utils import parse_config_from_base_overrides, BASE_CONFIG_SLOTS, BASELINE_CONFIG_SLOTS

SCRIPTS = ["stage1", "phase1_pilot", "reward_probe", "phaseA_mixture", "phaseA_reward", "phaseA_components",
           "phaseB_recipe", "phaseC_estimator", "phaseD_appendix", "phaseE_scale"]

# Values that make the optional blocks emit their commands instead of skipping. They are only
# here so the parse covers every row; none of them changes what a real run does.
SCRIPT_ENV = dict(
    DRY_RUN="1", SEED="42", A5_GROUP_SIZE="16", PROBE_STEPS="50",
    STAGE1_MODE="all", AX_REWARD="configs/reward/ndcg_in_batch_all.yaml", AX_GRPO="configs/grpo/factorized.yaml",
)

failures: list[str] = []


def collect():
    cmds = []
    for s in SCRIPTS:
        env = dict(os.environ, **SCRIPT_ENV)
        # phaseE_scale is a no-op at 0.6B, so it is also parsed at 4B.
        variants = [env] if s != "phaseE_scale" else [dict(env, SCALE="4b")]
        out = ""
        for e in variants:
            proc = subprocess.run(["bash", f"scripts/experiments/{s}.sh"],
                                  capture_output=True, text=True, env=e)
            # A script that dies mid-way would otherwise look like a script with fewer rows,
            # which is exactly the failure this check exists to catch.
            if proc.returncode != 0:
                print(f"  FAIL [{s}] script exited {proc.returncode}\n"
                      f"       {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}")
                failures.append(s)
            out += proc.stdout
        for line in out.splitlines():
            m = re.match(r"^\x1b\[1m>>> (.*)\x1b\[0m$", line)
            if m and ("run.sh" in m.group(1) or "run_baseline.sh" in m.group(1)):
                cmds.append((s, m.group(1)))
    return cmds

grpo_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments,
                                LoraArguments, RLArguments, MTEBEvalArguments))
base_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments,
                                LoraArguments, BaselineArguments, MTEBEvalArguments))

ok = 0
bad = len(failures)
seen = set()
for script, cmd in collect():
    toks = shlex.split(cmd)
    idx = next(i for i, t in enumerate(toks) if t.endswith(".sh"))
    is_rl = toks[idx].endswith("/run.sh")
    args = toks[idx + 1:]
    key = tuple(args)
    if key in seen: continue
    seen.add(key)
    parser = grpo_parser if is_rl else base_parser
    slots = BASE_CONFIG_SLOTS if is_rl else BASELINE_CONFIG_SLOTS
    try:
        # Both entrypoints route through train.parse_arguments, which owns the splitting.
        overrides, cli = train_mod.split_launcher_args(args, base_slots=slots)
        # No CUDA on this box; bf16 is a hardware assertion, not a config error.
        cli = cli + ["--bf16", "false", "--tf32", "false", "--deepspeed", ""]
        dcs = parse_config_from_base_overrides(parser=parser, base_overrides=overrides,
                                               cli_args=cli, base_slots=slots)
        run_name = dcs[2].run_name
        ok += 1
        print(f"  OK   [{script}] {run_name}")
    except Exception as e:
        bad += 1
        print(f"  FAIL [{script}] {' '.join(args[:6])}...\n       {type(e).__name__}: {e}")
print(f"\n{ok} parsed, {bad} failed")
sys.exit(1 if bad else 0)
