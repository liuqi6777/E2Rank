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
from config import (DataArguments, LoraArguments, ModelArguments, MTEBEvalArguments,
                    RLArguments, TrainingArguments)
from baselines.config import BaselineArguments
import train as train_mod
import train_baseline as tb_mod
from utils import parse_config_from_base_overrides, BASE_CONFIG_SLOTS, BASELINE_CONFIG_SLOTS

SCRIPTS = ["table1_main", "table2_recipe", "table3_components", "table4_reward",
           "table5_estimator", "table6_group_kappa", "tableD_surrogates", "tableX_secondary"]

def collect():
    cmds = []
    for s in SCRIPTS:
        env = dict(os.environ, DRY_RUN="1", SEEDS="42", STAGE1_MAX_STEPS="100",
                   SWEEP="0", A5_GROUP_SIZE="16")
        out = subprocess.run(["bash", f"scripts/experiments/{s}.sh"],
                             capture_output=True, text=True, env=env).stdout
        for line in out.splitlines():
            m = re.match(r"^\x1b\[1m>>> (.*)\x1b\[0m$", line)
            if m and ("run.sh" in m.group(1) or "run_baseline.sh" in m.group(1)):
                cmds.append((s, m.group(1)))
    return cmds

grpo_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments,
                                LoraArguments, RLArguments, MTEBEvalArguments))
base_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments,
                                LoraArguments, BaselineArguments))

ok = bad = 0
seen = set()
for script, cmd in collect():
    toks = shlex.split(cmd)
    idx = next(i for i, t in enumerate(toks) if t.endswith(".sh"))
    is_rl = toks[idx].endswith("/run.sh")
    args = toks[idx + 1:]
    key = tuple(args)
    if key in seen: continue
    seen.add(key)
    mod = train_mod if is_rl else tb_mod
    parser = grpo_parser if is_rl else base_parser
    try:
        overrides, cli = mod.split_launcher_args(args)
        # No CUDA on this box; bf16 is a hardware assertion, not a config error.
        cli = cli + ["--bf16", "false", "--tf32", "false", "--deepspeed", ""]
        slots = BASE_CONFIG_SLOTS if is_rl else BASELINE_CONFIG_SLOTS
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
