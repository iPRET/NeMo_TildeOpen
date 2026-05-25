#!/bin/bash

export NCCL_BUFFSIZE=2097152

export HF_HUB_CACHE=/e/project1/jureap133/ingus/nemo/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TRITON_CACHE_DIR=/e/project1/jureap133/ingus/nemo/triton_cache
export TORCH_HOME=/e/project1/jureap133/ingus/nemo/torch_cache
export TORCH_EXTENSIONS_DIR=/e/project1/jureap133/ingus/nemo/torch_extensions/$SLURM_JOB_ID

torchrun \
  --nnodes 256 \
  --nproc_per_node 4 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --node_rank=$SLURM_NODEID \
    NeMo_TildeOpen/scripts/llm/gpt_train.py \
      --devices 4 \
      --num_nodes 256 \
      --tp_size 4 \
      --pp_size 1 \
      --model_path TLM64K_Real_15B_Attention \
      --teacher_path TLM64KNeMo \
      --recompute_granularity full \
      --recompute_method uniform \
      --recompute_num_layers 6 \
      --max_steps 18000 \
      --warmup_steps 200 \
      --gbs 256 \
      --mbs 1 \
      --lr 1.6e-4 \
      --min_lr 8e-6 \
      --clip_grad 0.4 \
      --seq_length 65536 \
      --legacy_ckpt \
      --log_dir "/e/scratch/jureap133/ingus/Distil_Real" \
      --name "Attention_v3" \
      --log_interval 1 \
      --val_check_interval 64 \
      --limit_val_batches 0 \
      --save_top_k -1 \
      --sync_checkpoints \
      --max_checkpoints 8 \
      --data_paths ./ext_nemo/ext
