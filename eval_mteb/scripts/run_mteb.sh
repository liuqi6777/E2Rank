export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS='8'

model_path=$1

eval_subset="SciFact,ArguAna,NFCorpus,StackOverflowDupQuestions,SciDocsRR,BiorxivClusteringS2S,MedrxivClusteringS2S,TwentyNewsgroupsClustering,SprintDuplicateQuestions,Banking77Classification,EmotionClassification,MassiveIntentClassification,STS17,SICK-R,STSBenchmark,SummEval"

python eval_mteb/run_mteb.py \
  --model ${model_path} \
  --precision fp16 \
  --model_kwargs "{\"max_length\": 8192, \"attn_type\": \"causal\", \"pooler_type\": \"last\", \"do_norm\": true, \"use_instruction\": true, \"instruction_template\": \"Instruct: {}\nQuery:\", \"instruction_dict_path\": \"eval_mteb/scripts/task_prompts.json\"}" \
  --output_dir results/mteb \
  --batch_size 16 \
  --langs "eng" \
  --tasks $eval_subset
