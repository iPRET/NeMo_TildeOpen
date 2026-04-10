#!/bin/bash
# Script automagically replaces code installed in the 25.11 NeMo docker container.

# Modifications to Megatron-LM
cp ./gpt_dataset.py /opt/megatron-lm/megatron/core/datasets/gpt_dataset.py

# Modifications to NeMo
cp ./llama.py /opt/NeMo/nemo/collections/llm/gpt/model/llama.py

# Modifying tensorRT
cp ./megatron.py /opt/TensorRT-Model-Optimizer/modelopt/torch/nas/plugins/megatron.py

# Modifying tensor_engine The
#   The flash attention override.
cp ./dot_product_attention.py /opt/venv/lib/python3.12/site-packages/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py


# Fix: pass num_layers_in_first/last_pipeline_stage to MegatronStrategy
#   so strategy.connect() doesn't clobber model.config with None.
cp ./model_utils.py /opt/NeMo/nemo/collections/llm/modelopt/model_utils.py

# Debug prints
#cp ./attention.py /opt/megatron-lm/megatron/core/transformer/attention.py

