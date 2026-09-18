#!/usr/bin/env python3
"""Rebuild the shortlist results note and audit from local BRIGHT CSV exports."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

import yaml

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
SOURCE = PAPER / '_summary/g1_r2_bright'
SEEDS = (42, 3407, 2026)
PREFIX = 'G1-R2-RL-GradedNDCG64-CP-'
SUITES = ('shortlists', 'shortlist_sweep', 'shortlist_distribution', 'shortlist_alignment')
FOLLOWUP_SUITES = ('shortlist_mixed_rewards', 'shortlist_pairwise')
DOMAINS = ('biology', 'earth_science', 'economics', 'psychology', 'robotics',
           'stackoverflow', 'sustainable_living', 'pony', 'leetcode', 'aops',
           'theoremqa_theorems', 'theoremqa_questions')


def run_name(recipe, seed):
    return recipe + (f'-Seed{seed}' if seed != 42 else '') + f'-s{seed}'


def main():
    paths = [SOURCE / f'{name}_summary.csv' for name in ('run', 'subset')]
    rows = list(csv.DictReader(paths[0].open()))
    runs = {r['run']: r for r in rows}
    if len(runs) != len(rows):
        raise ValueError('Duplicate run rows')
    subsets = {}
    for row in csv.DictReader(paths[1].open()):
        key = (row['run'], row['subset'])
        if key in subsets:
            raise ValueError(f'Duplicate subset: {key}')
        subsets[key] = row
    expected = set()
    followups = set()
    suite_counts = []
    for suffix in SUITES:
        path = ROOT / f'configs/experiments/iclr2027/suite_g1_{suffix}.yaml'
        suite = yaml.safe_load(path.read_text())
        names = set()
        for name, cfg in suite['runs'].items():
            target = followups if cfg.get('analysis_group') == 'shortlist_followup' else names
            target.add(f"{name}-s{cfg['seed']}")
        suite_counts.append((suffix, len(names), len(names - expected)))
        expected |= names
    for suffix in FOLLOWUP_SUITES:
        path = ROOT / f'configs/experiments/iclr2027/suite_g1_{suffix}.yaml'
        suite = yaml.safe_load(path.read_text())
        followups.update(f"{name}-s{cfg['seed']}" for name, cfg in suite['runs'].items())
    actual = {name for name in runs if '-Shortlist' in name} - followups
    if expected != actual:
        raise ValueError(f'Suite/result mismatch: missing={expected-actual}, extra={actual-expected}')
    references = [PREFIX + 'Align080', PREFIX.rstrip('-'), PREFIX + 'Align090-LargePool', 'G1-R2-CL-Strong']
    selected = expected | {run_name(recipe, seed) for recipe in references for seed in SEEDS}
    max_gap = 0.
    for name in selected:
        row = runs[name]
        if any(int(row[k]) != v for k, v in [('tasks_found', 1), ('tasks_missing', 0), ('types_found', 1), ('errors', 0)]):
            raise ValueError(f'Incomplete run: {name}')
        if 'tokens-v2__pool-fp32' not in row['result_dir']:
            raise ValueError(f'Unexpected embedding protocol: {name}')
        found = {domain for run, domain in subsets if run == name}
        if found != set(DOMAINS):
            raise ValueError(f'Incomplete domains: {name}')
        scores = [float(subsets[name, d]['score']) for d in DOMAINS]
        score = float(row['mean_task_score'])
        if not all(math.isfinite(v) for v in [score, *scores]):
            raise ValueError(f'Nonfinite score: {name}')
        if any(subsets[name, d]['task'] != 'BrightRetrieval' for d in DOMAINS):
            raise ValueError(f'Unexpected task: {name}')
        gap = abs(stats.mean(scores) - score)
        if gap > 0.011:
            raise ValueError(f'Task/subset mean mismatch: {name}: {gap}')
        max_gap = max(max_gap, gap)

    def values(recipe):
        return [float(runs[run_name(recipe, seed)]['mean_task_score']) for seed in SEEDS]

    def mean(recipe):
        return stats.mean(values(recipe))

    def table(entries, baseline=None):
        lines = ['| 配方 | 42 | 3407 | 2026 | 均值 ± 样本 SD |' + (' 相对基准 |' if baseline else ''),
                 '|---|---:|---:|---:|---:|' + ('---:|' if baseline else '')]
        for label, recipe in entries:
            scores = values(recipe)
            line = f'| {label} | ' + ' | '.join(f'{v:.2f}' for v in scores)
            line += f' | {stats.mean(scores):.2f} ± {stats.stdev(scores):.2f} |'
            if baseline:
                line += f' {mean(recipe)-mean(baseline):+.2f} |'
            lines.append(line)
        return '\n'.join(lines)

    def short(suffix, alignment='080'):
        return PREFIX + f'Align{alignment}-Shortlist' + suffix

    uniform = short('Uniform-K15-T1')
    best = short('Uniform-K15-T1', '070')
    base = references[0]
    cl = references[-1]
    size_entries = [('Uniform K15/T1', uniform),
                    ('Uniform K15/T4', short('Uniform-K15-T4')),
                    ('Uniform K15/T8', short('Uniform-T8')),
                    ('Uniform K15/T16', short('Uniform-T16')),
                    ('Uniform K30/T4', short('Uniform-K30-T4')),
                    ('Uniform K30/T8', short('Uniform-K30-T8')),
                    ('Uniform K60/T4', short('Uniform-K60-T4'))]
    hard_entries = [('Uniform / Hard0 / T8', short('Uniform-T8')),
                    ('Mixed / Hard2 / T8', short('Mixed-K15-T8-Hard2')),
                    ('Mixed / Hard4 / T8', short('Mixed-K15-T8-Hard4')),
                    ('Mixed / Hard8 / T8', short('Mixed-T8')),
                    ('Mixed / Hard8 / T16', short('Mixed-T16'))]
    local = short('Uniform-LocalAll-K15-T1')
    representatives = short('Uniform-CrossDeviceRepresentatives-K15-T1')
    distribution_entries = [('同卡代表正例（原 CP）', base), ('同卡全部候选', local),
                            ('跨卡代表正例', representatives), ('跨卡全部候选', uniform)]
    alignment_entries = [(f'alignment {int(a)/100:.2f}', short('Uniform-K15-T1', a))
                         for a in ('065', '070', '080', '090', '095')]
    summary = table([('原 CP / alignment 0.80', base), ('Uniform K15/T1 / alignment 0.80', uniform),
                     ('Uniform K15/T1 / alignment 0.65', short('Uniform-K15-T1', '065')),
                     ('Uniform K15/T1 / alignment 0.70', best), ('CL-Strong', cl)])
    domain_lines = ['| 子集 | 原 CP 0.80 | Uniform 0.80 | Uniform 0.70 | CL-Strong | 0.70−0.80 | 0.70−CL |',
                    '|---|---:|---:|---:|---:|---:|---:|']
    for domain in DOMAINS:
        v = [stats.mean(float(subsets[run_name(recipe, seed), domain]['score']) for seed in SEEDS)
             for recipe in (base, uniform, best, cl)]
        domain_lines.append(f'| {domain} | ' + ' | '.join(f'{x:.2f}' for x in v)
                            + f' | {v[2]-v[1]:+.2f} | {v[2]-v[3]:+.2f} |')
    paired = ['| 改变 | 平均差 | Δ42 | Δ3407 | Δ2026 |', '|---|---:|---:|---:|---:|']
    for label, a, b in [('代表正例：同卡→跨卡', representatives, base),
                         ('全部候选：同卡→跨卡', uniform, local),
                         ('同卡：代表正例→全部候选', local, base),
                         ('跨卡：代表正例→全部候选', uniform, representatives)]:
        delta = [a-b for a, b in zip(values(a), values(b))]
        paired.append(f'| {label} | {stats.mean(delta):+.2f} | ' + ' | '.join(f'{x:+.2f}' for x in delta) + ' |')
    report = f'''# G1-R2 Shortlist 实验结果总结

结果快照：2026-09-19。覆盖四组 suite 的 **17 个独立 shortlist 配方、51 次训练、612 条领域分数**；alignment 0.80 的三个运行复用已有 K/T sweep，不重复计数。另列 4 组三 seed 参考配置。本文由 `python scripts/analyze_g1_shortlist_results.py` 从本地 CSV 生成。

## 1. 结论与当前配方

**当前已测 shortlist 配方中，跨卡全部候选、Uniform K15/T1、alignment 0.70 的 BRIGHT 均值最高：{mean(best):.2f}。0.65 为 {mean(short('Uniform-K15-T1', '065')):.2f}，两者仅差 0.03，视为接近的较优区间。**

- 相对原 CP/0.80，当前均值提升 **{mean(best)-mean(base):.2f} 分**；其中切换到 Uniform K15/T1/0.80 提升 **{mean(uniform)-mean(base):.2f}**，随后调 alignment 到 0.70 再提升 **{mean(best)-mean(uniform):.2f}**。这是沿已测配置路径的差值，不是独立因果效应分解。
- 与 CL-Strong 的差距由 **{mean(cl)-mean(base):.2f} → {mean(cl)-mean(uniform):.2f} → {mean(cl)-mean(best):.2f} 分**，仍未超过 CL，三个配对 seed 均落后。
- 增加 T 未带来稳定收益；扩大 K 和增加 hard 比例整体下降。结果支持小规模均匀 shortlist，尚不支持 multi-shortlist 的额外质量收益。
- 固定 K15/T1 时，跨卡来源相对同卡约提升 1 分；纳入全部候选而非仅代表正例约提升 0.2 分。跨卡可能引入异源负例，但具体原因未由分数证明。
- 当前以提点为目标，使用 Uniform K15/T1/0.70 作为工作配方，保留 0.65 作为接近的参照；不自动修改原论文主结果决策，不扩大 K/T/hard，也不延长 RL 训练步数。

{summary}

## 2. 统计口径与实验条件

输入：[总分](_summary/g1_r2_bright/run_summary.csv)、[领域分数](_summary/g1_r2_bright/subset_summary.csv)。总分采用 `mean_task_score`，三 seed 顺序为 42/3407/2026，SD 为样本标准差（ddof=1）。差值用未再次舍入的 CSV 数值计算；领域先跨 seed 平均。分数为 0–100 尺度，差值单位为分。

按 suite 去重计数：初始 Mixed/Uniform T8/T16 **12** 次，K/T 和轻量 hard 扩展 **21** 次，负例来源 **6** 次，alignment 新增 **12** 次，共 **51** 次。均有 12 个唯一 BRIGHT subset，任务无缺项、errors=0；加上 12 次参考运行共核验 63 次。总分与领域宏平均最大舍入差为 **{max_gap:.5f}**。

Shortlist 配置共同条件：Qwen3-Embedding-0.6B、graded nDCG@10、CP/G64、joint full FT、LR 5e-6、8 卡 × microbatch 16、**113 steps**、最终 checkpoint、无辅助 InfoNCE。RL 与 CL 保持相同步数；这不意味着计算量或墙钟耗时相同。除 alignment sweep 外 shortlist 均固定 alignment 0.80。

K 只计额外跨 query 候选，自有候选全部保留。T 个榜单复用 actions，分别计算 reward/LOO/CP，再平均 loss；不是把全部候选并成一张榜单。Mixed 的 hard 区为均值相似度前 128 个候选。来源池经过身份去重、已知正例排除和 route 过滤；有效池不足时实际数量可能小于 K。

本次核验结果导出和配置，不据此认定逐运行训练合同、实际成本或训练日志已审计。三个 seed 的 SD 不是置信区间；这里不作统计显著性主张。配方选择来自反复查看同一 BRIGHT 结果，尚无独立选参评测证明泛化最优。

## 3. Shortlist 大小 K 与组数 T

固定跨卡全部候选、Uniform、alignment 0.80；表中差值以 K15/T1 为基准。

{table(size_entries, uniform)}

固定 K15，T=1/4/8/16 均值处于 21.93–21.98，没有稳定增益。T1 均值最高，但差值仅 0.02–0.05，不能据此宣称质量显著更好；它所需的小榜单 reward/CP 计算较少，实际加速未测。

固定 T4，K15→30→60 在三个 seed 上都下降。同为最多覆盖 240 个额外负例，K15/T16、K30/T8、K60/T4 均值依次为 21.93、21.85、21.51，三个 seed 排序一致。候选覆盖量不是当前结果的充分解释；K 同时改变排序难度与 CP 投影空间，不能仅由最终分数断定具体机制。

## 4. Hard 采样比例

固定 K15、alignment 0.80；前四行固定 T8，差值以 Uniform T8 为基准。

{table(hard_entries, short('Uniform-T8'))}

Hard0→2→4→8 的三 seed 均值整体下降。Hard2 基本持平、没有稳定增益；Hard4 和 Hard8 在每个 seed 上都低于 Hard0。Hard8 增加到 T16 也未改善均值。高相似度并不保证适合当前 RL，但 false negatives、LOO 退化或投影变化等原因仍需日志证据。

## 5. 负例来源与组成

三种 shortlist 均为 Uniform K15/T1/0.80；原普通 CP 不是 shortlist 路径，而是使用过滤后同卡其他 query 各一个代表正例，最多 15 个。

{table(distribution_entries)}

{chr(10).join(paired)}

每卡 microbatch 按数据来源分组；同一步各卡取得不同 microbatch，不保证同来源。当前 G1 跨卡候选可包含异源文档。因此“跨卡”同时扩大候选范围并可能改变来源组成；不能简单等同于纯粹增大同源池，也不能把异源负例更易学当作已证实结论。

额外负例数量上限近似相同，但过滤后的实际数量未由本 CSV 给出。原 CP 与 shortlist 也并非逐动作严格等价。同卡全部候选与跨卡全部候选使用相同 shortlist 路径，是来源范围更直接的对照。四个配对比较均为三个 seed 同向，跨卡效应约 1 分，组成效应约 0.2 分；当前不继续拆分同源/异源机制。

## 6. Alignment

固定跨卡全部候选、Uniform K15/T1 和 113 steps；差值以 0.80 为基准。alignment 越低，探索越强。

{table(alignment_entries, uniform)}

0.65 在三个 seed 均优于 0.80；0.70 的 seed42/2026 提升，3407 基本持平（−0.02）。0.90/0.95 在三个 seed 均下降。0.65 与 0.70 尚未拉开差距，0.65 均值略低；已有曲线不支持把继续降低 alignment 视为确定的提点方向，也不能据这五点量化继续探索的成功概率。

## 7. 领域收益与剩余缺口

下表均为三 seed 的领域均值。

{chr(10).join(domain_lines)}

Uniform/0.80 相对原 CP 的提升主要来自 biology（+7.44）与 earth_science（+3.78）；0.70 相对 0.80 又在这两项提升 +2.16、+1.76，同时 theoremqa_theorems 下降 1.04。因此均分提升并非所有领域同步改善。

当前 0.70 配方相对 CL-Strong，biology +3.27、pony +1.15、earth_science +0.79；主要缺口是 robotics −3.30、sustainable_living −3.29、theoremqa_theorems −2.04、stackoverflow −1.59。

## 8. 全池参考与结论边界

{table([('普通 CP / alignment 0.90', references[1]), ('全池 LargePool / alignment 0.90', references[2]), ('Uniform K15/T1 / alignment 0.90', short('Uniform-K15-T1', '090'))])}

全池 LargePool 将过滤后的跨卡候选一起加入单个排序 reward，和 shortlist 目标不同。同 alignment 0.90 下，Uniform K15/T1 均值高于普通 CP 和全池 LargePool；不使用全池 0.90 与 shortlist 0.80 的差值作为纯 shortlist 效应。当前没有匹配 shortlist 配方的 SF 结果，不能把这些新增收益全部归于 CP 相对 SF 的优势。

适合结果陈述：**在固定 113-step 预算下，小规模均匀 shortlist 改善了当前纯 RL 配方；收益主要伴随候选分布改变，扩大榜单规模或增加组数没有进一步改善，适度增强探索可继续提点。最佳已测三 seed 均值为 22.25，仍低于 CL-Strong 的 22.64。**

## 9. 复算与追溯

```bash
python scripts/analyze_g1_shortlist_results.py
```

脚本核验 suite 与结果一一对应、三 seed、任务和领域完整性，再生成本文与 [审计记录](g1_shortlist_results/audit.json)。审计包含输入 SHA256、51 次 shortlist 的完整 run ID、配方参数与逐领域分数，以及参考运行。本文不改写原始结果，也不自动更新论文 LaTeX/PDF。

配置与运行细节见 [multi_shortlist_rl](../docs/multi_shortlist_rl.md)，原结果综述见 [G1_R2_RESULTS](G1_R2_RESULTS.md)。
'''
    output = PAPER / 'g1_shortlist_results'
    output.mkdir(exist_ok=True)
    configuration = {}
    suite_hashes = {}
    for suffix in SUITES:
        path = ROOT / f'configs/experiments/iclr2027/suite_g1_{suffix}.yaml'
        suite_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        suite = yaml.safe_load(path.read_text())
        for name, cfg in suite['runs'].items():
            configuration[f"{name}-s{cfg['seed']}"] = cfg
    audit = dict(snapshot='2026-09-19',
                 inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                 suites=[dict(name=n, runs=total, new_runs=new) for n, total, new in suite_counts],
                 suite_sha256=suite_hashes,
                 shortlist_runs=len(expected), reference_runs=len(selected-expected),
                 max_task_subset_rounding_gap=max_gap,
                 runs=[dict(**runs[n], shortlist=n in expected, configuration=configuration.get(n),
                            subsets={d:float(subsets[n,d]['score']) for d in DOMAINS}) for n in sorted(selected)])
    (output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    (PAPER / 'G1_SHORTLIST_RESULTS.md').write_text(report)
    print(f'Validated {len(expected)} shortlist runs + {len(selected-expected)} references; wrote result note and audit.')


if __name__ == '__main__':
    main()
