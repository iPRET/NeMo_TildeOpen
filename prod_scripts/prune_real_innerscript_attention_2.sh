#!/bin/bash
export HF_HUB_CACHE=/e/project1/jureap133/ingus/nemo/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TRITON_CACHE_DIR=/e/project1/jureap133/ingus/nemo/triton_cache
export TORCH_HOME=/e/project1/jureap133/ingus/nemo/torch_cache
export TORCH_EXTENSIONS_DIR=/e/project1/jureap133/ingus/nemo/torch_extensions

torchrun --nproc_per_node 4 NeMo_TildeOpen/scripts/llm/gpt_prune.py \
  --devices 4 \
  --tp_size 1 \
  --pp_size 4 \
  --seq_length 65536 \
  --restore_path 15B_Attention_Distilled \
  --save_path ./TLM64K_Real_Attention_2 \
  --data_paths ./ext_nemo/ext \
  --mbs 1 \
  --num_layers_in_last_pipeline_stage 3 \
  --num_train_samples 1024 \
  --target_ffn_hidden_size 10752 \
  --target_hidden_size 3072 \
  --target_num_attention_heads 24 \
  --target_num_query_groups 8
