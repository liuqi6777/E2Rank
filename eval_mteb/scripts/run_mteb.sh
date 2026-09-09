export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS='8'

model_path=$1
benchmark=$2
model_config=${3:-}
extra_args=()
if [ -n "${model_config}" ]; then
  extra_args+=(--model_config "${model_config}")
fi

if [ -f "${model_path}/embedding_protocol.json" ]; then
  model_kwargs='{"max_length": 8192, "instruction_dict_path": "eval_mteb/scripts/task_prompts.json"}'
else
  # Public Qwen initialization has no local sidecar. A third model-config argument
  # overrides this fallback for non-Qwen initializations.
  model_kwargs='{"max_length": 8192, "attn_type": "causal", "pooler_type": "last", "padding_side": "left", "append_token": "pad", "do_norm": true, "use_instruction": true, "query_prompt_template": "Instruct: {task_description}\nQuery:{text}", "document_prompt_template": "{text}", "instruction_dict_path": "eval_mteb/scripts/task_prompts.json"}'
fi

python eval_mteb/run_mteb.py \
  --model "${model_path}" \
  --precision fp16 \
  --model_kwargs "${model_kwargs}" \
  --output_dir results/mteb \
  --batch_size 16 \
  --langs "eng" \
  --benchmark "${benchmark}" \
  "${extra_args[@]}"
