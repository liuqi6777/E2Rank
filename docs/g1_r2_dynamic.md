# G1-R2 动态全库检索

用户入口为 `scripts/run_g1_r2_dynamic.py`，支持 `encode / check / run / summary`。
默认三 seed：42/3407/2026，独立运行 `G1-R2-DR-GradedNDCG64-SF` 及 seed 后缀版本。
这是固定文档索引下的 query-only 动态检索实验，不是 joint 静态候选 CP 的单因素消融。

## 配方

- 固定 revision 的 E0 初始化，token protocol v2，FP32 pooling/评分。
- 只更新 query encoder；文档侧始终使用固定 E0 向量。
- 按 source 路由至 canonical ReasonRank `id_doc` 文档全集，而非训练候选文档并集。
- 每个 query 采样 G64 个 vMF 动作，alignment 0.90；每个动作实际全库检索 top-20。
- 使用训练记录中全部已知 graded qrels 计算 nDCG@10；未标注文档按 zero gain，
  IDCG 基于已知 qrels，训练候选 ID 仅承载这些标签，不限制检索返回范围。
- SF、LOO、无 advantage normalization、无 KL/辅助损失；动态检索不启用 CP。
- 113 optimizer steps，LR 5e-6，8 卡 × microbatch 16，global batch 128，无 dev/裁剪，
  checkpoint 保存和最终选模沿用 R2。
- 训练/data/独立 rollout seed 同值。独立训练适配层复用 `RolloutRNG`，按 rank/step 管理采样；
  未修改共享静态训练源码，也不绕过 CP 对动态检索的限制。

这里只有 query 动作轴，每条 query 是 64 次检索，不是静态双侧 product 的 64×64 个 reward cell。
训练索引编码的文档长度上限为 1024；BRIGHT 评测上限 8192，沿用 R2 的训练/评测长度差异。
离线文档编码使用现有 index encoder 的 FP16 backbone、FP32 pooling，索引存储为 FP16。
训练 query backbone 为 BF16，外部评测为 FP16。

## 在 GPU 任务中运行

从项目根目录、已激活训练环境执行。读取 `configs/experiments_r2.yaml`，也可传 `--config`。
默认检索后端为 FAISS，需要训练环境已安装可导入的 `faiss`；`--index-backend torch`
可改用已有 torch 检索后端，所有 seed 应使用相同后端并如实记录成本。

### 1. 准备 canonical 文档（已有则跳过）

```bash
python scripts/download_reasonrank_audit.py --include-reasonrank-documents
```

默认文档目录为 `data/audit_reasonrank_bright/reasonrank_documents/id_doc`。
下载器按项目固定的数据 revision 取得各 source 文档；建索引时会验证训练候选的 ID/文本能映射到它们。
自定义文档位置用 `encode --documents-dir /path/to/id_doc`。

### 2. 建一次 R2 索引，供所有 seed 复用

```bash
nohup python -u scripts/run_g1_r2_dynamic.py encode > g1_r2_dynamic_encode.log 2>&1 &
```

默认使用 8 卡、每卡 encoding batch 16；可用 `--encode-gpus`、`--encode-batch-size` 调整编码阶段。
模型和 tokenizer 明确固定 revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。
新索引默认位于训练数据目录下的 `reasonrank_frozen_document_indices_r2/`，不复用旧协议索引目录。
用 `--index-root` 修改时，后续 check/run 也必须传相同参数。

完整 source 索引可校验并跳过；中断的 `.building` 或不完整目录不会覆盖，
应先检查现场，再换新 `--index-root` 重建。索引建完会生成 `index_router_manifest.json`。
索引位置有独立构建锁；不同输出目录共享同一个索引时也不会并发写入。

### 3. 索引完成后预检并训练

```bash
python scripts/run_g1_r2_dynamic.py check
nohup python -u scripts/run_g1_r2_dynamic.py > g1_r2_dynamic.log 2>&1 &
```

默认串行完成三次训练及最终 BRIGHT。也可以分别在独立 8 卡任务中运行：

```bash
python -u scripts/run_g1_r2_dynamic.py --seeds 42
python -u scripts/run_g1_r2_dynamic.py --seeds 3407
python -u scripts/run_g1_r2_dynamic.py --seeds 2026
```

共享存储时三个任务复用同一份冻结训练索引。锁按 run ID 隔离，拒绝同一 seed 重复运行。
独立机器需要各自可访问配套的完整索引；不要只复制 router JSON。

## 最终评测与对照解释

BRIGHT 使用**训练后的 query encoder + 冻结 E0 document encoder**。
首次评测编码完整 BRIGHT corpus，缓存于输出根目录的 `.r2_dynamic/bright_e0_index`。
三个 seed 共用该缓存；由于现有评测缓存没有内部构建锁，入口串行保护整个评测阶段，
各 seed 的训练仍可并行。评测不能默认改成训练后的 joint encoder，否则部署问题发生变化。

新 E0 可以作为效果参照；与现有 joint SF/CP 的差异同时包含 policy 范围、
候选访问和部署文档编码方式，不能单独归因于动态候选。
本次不自动额外启动 fixed-index CL/static RL 对照；若后续要声称动态检索优于静态候选，
需补同冻结索引、同 query-only 部署的匹配对照。

## 输出和恢复

默认输出根目录沿用配置中的 `checkpoints/iclr2027-r2`：

```text
G1-R2-DR-GradedNDCG64-SF-s42/
G1-R2-DR-GradedNDCG64-SF-Seed3407-s3407/
G1-R2-DR-GradedNDCG64-SF-Seed2026-s2026/
.r2_dynamic/
  encode/<source>.log
  <run ID>/{contract,state,config,runtime}.json
  <run ID>/{train,eval}.log
  <run ID>/{events,timings}.jsonl
  queues/seeds-<seeds>/summary.{json,md}
  bright_e0_index/
```

合同保存数据/源码哈希、router 哈希、实际训练配置与冻结文档评测命令。
预检验证 route 协议、模型 revision、文件哈希与训练数据身份；保留模型 checkpoint。
训练成功而评测失败时重启只补评测；完整且身份一致的结果跳过；
失败训练不自动从不完整 checkpoint 续训，不覆盖现场，其他独立 seed 继续。
需重训时把运行配置的 `output_dir` 改为新根目录，并只选择失败 seed；冻结索引仍可复用。

训练/评测耗时分别记录；评测时间可能包含等待共享评测锁的时间，不能当作纯计算时间。
初始化全库索引的编码成本需另计，不从静态训练的耗时直接推断动态检索效率。

```bash
python scripts/run_g1_r2_dynamic.py summary
```

汇总检查三个 seed 的合同、训练数据/源码一致性和 12 个 BRIGHT subset，报告均值、样本 SD、
最差值和逐领域分数。缺失结果返回非零；只汇集分数时需带上各 run 的合同/状态，
不必复制模型权重。源路径可迁移，合同内容保持原样。

## 验证状态

本地已通过真实小型 Qwen3 + 可搜索冻结向量库的前向/反向检查，验证 top-20 动态搜索、
graded reward、query 参数梯度、独立 rollout RNG 与冻结文档评测配置；相关测试 62 项通过。
本地没有 CUDA、canonical 文档及 R2 冻结索引，尚未完成真实索引校验或启动 GPU 训练。
