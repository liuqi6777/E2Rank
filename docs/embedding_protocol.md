# 文本 embedding 协议 v2

## 行为

统一入口为 `src/embedding_protocol.py::tokenize_embedding_texts`。Query/document 模板仍在
tokenization 之前应用，pooling 与向量归一化规则保持由模型配置决定。

| 配置 | 正文 tokenization | 截断与末尾 |
|---|---|---|
| `append_token: pad` | `add_special_tokens=False` | 正文最多 L−1 个 token，再追加 `pad_token_id` |
| `append_token: eos` | `add_special_tokens=False` | 正文最多 L−1 个 token，再追加 `eos_token_id` |
| `append_token: none` | `add_special_tokens=True` | 使用原生 tokenizer 的 special tokens 与截断 |

显式末尾路径在 padding 前生成 attention mask；读出 token 的 mask 为 1，补齐位置为 0。
因此 Qwen 的末尾 token 即使与 padding token 共用 ID，也不会被当作 padding 丢弃。
左、右 padding 均支持。正文末尾已有同一特殊 token 时合并成一个边界；正文内部的特殊 token
不移除。空文本和 `max_length=1` 在显式末尾路径中产生单个有效末尾 token。

当前两个 0.6B Qwen 配置仍用 `append_token: pad`，实际读出 ID 均为 151643：

```text
Qwen3-0.6B:           test → [1944, 151643]
Qwen3-Embedding-0.6B: test → [1944, 151643]
```

两者 tokenizer 自动后处理不同，但新入口通过显式 ID 追加得到一致的边界语义。
不要将 Qwen 配置改成 `eos`：其 `eos_token_id=151645` 与 embedding 读出 ID 不同。
BGE/E5 的 `append_token: none` 继续保留原生 CLS/SEP 等结构。

## 覆盖的入口

- GRPO、InfoNCE/RankNet/LambdaLoss 共用的 `EmbeddingDataCollator`。
- MTEB 的 `TransformersTextEmbedder`，包括训练中的评测 callback。
- RAG query 训练、候选挖掘、离线评测，以及固定语料 shard 编码。
- embedding 几何分析与 seed 表示差异分析。

显式末尾的旧字符串拼接函数已移除，没有历史输入行为开关。

## Checkpoint 与缓存

每个训练 checkpoint（包括关闭 MTEB 的中间 checkpoint）和最终输出都保存 tokenizer 及
`embedding_protocol.json`。该 sidecar 除 pooling、padding、模板和最大长度外，还记录：

```json
{
  "tokenization_version": 2,
  "pooling_compute_dtype": "float32",
  "add_special_tokens": false,
  "terminal_token_id": 151643,
  "terminal_after_truncation": true
}
```

读取旧版本或没有版本的协议文件会报错；不提供旧行为兼容或自动迁移。
原始 Hugging Face 模型没有项目 sidecar 时，使用模型配置和新的统一 tokenization 规则。
现有旧权重和结果文件不删除、不重写。

归一化与 mean pooling 的计算精度固定为 FP32；backbone 可继续用混合精度。
项目 checkpoint 和固定索引必须包含 `pooling_compute_dtype: float32`，否则拒绝读取。
评测及几何分析的缓存 identity 同样记录该字段，避免与旧精度的向量缓存混用。

固定语料 manifest 记录新 token 元数据，消费旧索引会因版本不符而停止。
MTEB 结果模型名增加 `__tokens-v2__pool-fp32`；MTEB 固定语料 identity 和几何分析 identity 也包含协议版本。
新旧输入协议的 embedding 与结果缓存不能混用，旧语料索引需要重新编码到新目录。

## 验证

```bash
.venv/bin/python -m pytest -q tests/test_embedding_protocol.py
```

测试使用真实本地 fast tokenizer 与微型 Qwen3 backbone，无需下载模型。覆盖自动追加有/无、
短文本、截断、空文本、长度 1、重复末尾、左右 padding、原生 encoder special tokens、
训练/MTEB/RAG 输入一致性、实际 forward 输出一致性、checkpoint 保存重载与旧索引拒绝。
此外已用缓存的官方 `Qwen3-0.6B` 和 `Qwen3-Embedding-0.6B` tokenizer 检查实际 token IDs。

本次定向回归共通过 74 项测试、42 个子测试，另有 1 项索引预检通过。旧本地套件存在独立的
过期用例：runner 测试仍引用 `G1-Q-RL` 并期待 26 条训练行，而未修改的 HEAD 已没有该 run、
实际为 91 条；旧 frozen-corpus 测试导入了已移除的 `validate_training_against_bright_documents`。
这些不属于本次协议修改；当前索引接口的编码结果和协议拒绝行为已在新增测试中验证。
