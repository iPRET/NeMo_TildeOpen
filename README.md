# NeMo for purpouses of Pruning and Distilling TildeOpen.

This fork is intended for pruning and distilling TildeOpen30B via the code conglomeration know as the NVIDIA NeMo Framework.

- This version of the NeMo repo will assume your model uses YaRN for positional embeddings when converting Llama models from Huggingface to NeMo format.

# Instructions
Pro tip: Whenever a comment in a code block is in all caps, that means you have to modify the code before running it.
Following these instructions does increase the chance you will succeed, but it's far from a guarantee of success. NeMo holds together on scotch tape and bubblegum. Also I didn't test most of these commands.
## How to setup running stuff from this NeMo fork on our servers?
### 1. Clone NeMo
```
git clone https://github.com/iPRET/NeMo_TildeOpen.git
```
I'll refer to `./NeMo_TileOpen/` as the "repo root".
### 2. Launch the NeMo Container
```
# ADJUST THIS WITH YOUR PATH
docker run \
    --gpus all \
    -it \
    --rm \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /local_data/ingus/nemo_test:/local_data/ingus/nemo_test \
    nvcr.io/nvidia/nemo:25.11
```
In docker the `-v` flag binds folders from your host system to the filesystem inside the container. That is to say, you will be able to modify the files in `-v` from inside the container. Change this flag to wherever you have your NeMo-relevant files - code, training data, checkpoints.
### 3. Apply monkey patches
Once inside the container apply the monkey patches. Assuming you're in the root of the repository run:
```
# ADJUST THIS WITH YOU PATH
cd /local_data/ingus/nemo_test/NeMo_TildeOpen/monkey_patches
./replace.sh
```
#### Note on masking and eod tokens.
The `monkey_patches/gpt_datset.py` file's loss masking and eod token index is hardcoded in the ltor function to correspond to TildeOpen30B.
If training a different model, make sure to modify the loss masking and eod as your model expects.
### 4. Running stuff
As long as you're inside the docker container you should be able to just run the rest of the commands. Well done.
You have to repeat these steps every time you relaunch the docker container.
## How to setup this NeMo fork on supercomputers?
Disclaimer: NeMo will only work on NVIDIA GPU supercomputers. So it won't work on LUMI. (Or at least I gave up after trying to run it on LUMI for a couple days)
### 1. Clone Nemo
```
git clone https://github.com/iPRET/NeMo_TildeOpen.git
```
I'll refer to `./NeMo_TileOpen/` as the "repo root".
### 2. Change singularity cache folder locations
In later commands you might run into problems with your home directory being too small to download or work with singularity containers. So change the location of the cache folders to somewhere in the project or scratch directory.
```
# ADJUST SCRATCH PATH TO WHERE YOUR SUPERCOMPUTER'S PROJECT'S SCRATCH FOLDER IS.
export SCRATCH=/e/project1/jureap133/ingus/nemo

export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
export APPTAINER_TMPDIR=$SCRATCH/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
```
### 3. Make a sandboxed singularity container

```
# Download the container.
singularity pull nemo_25.11.sif docker://nvcr.io/nvidia/nemo:25.11

# Convert container to a sandboxed container.
singularity build --sandbox nemo_sandbox/ nemo_25.11.sif
```
### 4. Create bind folders
```
# Create folders that you'll later bind when running the container.
# ADJUST THESE TO SUPERCOMPUTER'S FOLDER NAMES.
# Example command is with folders used on Jupiter.

mkdir -p nemo_sandbox/e nemo_sandbox/p
```
If you won't do this, you'll later get errors that look something like:
```
Error:
  WARNING: Skipping mount /e [binds]: /e doesn't exist in container
  WARNING: Skipping mount /p [binds]: /p doesn't exist in container
  WARNING: Skipping mount /etc/FZJ [binds]: /etc/FZJ doesn't exist in container
  WARNING: By using --writable, Apptainer can't create /e destination automatically without overlay or underlay
  FATAL:   container creation failed: mount hook function failure: mount /var/lib/apptainer/mnt/session/e->/e error: while mounting /var/lib/apptainer/mnt/session/e: destination /e doesn't exist in container
```
So you can just come back to this step later when you realized what you need.
### 5. Apply monkey patches
```
# Run singularity container with bindings (-B) so repo is accessable.
# ADJUST THIS TO HAVE CORRECT BINDINGS FOR YOUR SUPERCOMPUTER
# Example command is if you were on Jupiter in the /e disk.
singularity shell --nv --writable -B /e:/e nemo_sandbox

# Apply monkey patches.
cd NeMo_TildeOpen/monkey_patches
./replace.sh
exit
```
### 6. Rebuild container into a .sif file
```
singularity build nemo_patched.sif
```
Congratulations! You now have a singularity image that you can run stuff with. You will never have to repeat the last 6 steps again.
### 7. Running stuff
#### 7.1. Lightweight scripts
If a script it lightweight, you can probably get away with just running the script on a login node.
To launch a singularity container on a login node, run:
```
# ADJUST THIS TO YOUR nemo_patched.sif LOCATION
# ADJUST FILESYSTEM BINDINGS (-B) to SUPERCOMPUTER FILESYSTEM FOLDERS
# Example command is written to work on the JUPITER supercomputer.

singularity shell --nv -B /e:/e nemo_patched.sif
```
After running this, you're inside a singularity container. You can run nemo stuff from here. To exit write `exit`.
This might work for some data preprocessing scripts or conversion scripts.
#### 7.2. Computationally heavy scripts
Alternatively if the script is too heavy to run on a login node, you can run an interactive slurm job:
```
# ADJUST --account --partition --gres --cpus-per-task FOR YOUR SUPERCOMPTUER.
# ADJUST THIS TO YOUR nemo_patched.sif LOCATION
# ADJUST FILESYSTEM BINDINGS (-B) to SUPERCOMPUTER FILESYSTEM FOLDERS
# Example command is written to work on the JUPITER supercomputer.

srun \
  --account=jureap133 \
  --partition=booster \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gres=gpu:4 \
  --cpus-per-task=288 \
  --time=0:30:00 \
  --mem=0 \
  singularity shell --nv -B /e:/e nemo_patched.sif
```
#### 7.3. Parallel computational heavy stuff.
Pruning and distillation is parallelized and requires you write sbatch scripts.
## How to convert a Llama Model from HuggingFace to NeMo?
### 1. Move outside of the NeMo repository folder if you're in it
You have to create the convert script outside the NeMo repository folder, so that the container's envionment's NeMo is used for this.
### 2. Create conversion script
Create a python script (OUTSIDE OF REPOSITORY ROOT) with contents like this:
```
from nemo.collections import llm

# ADJUST source TO LOCATION OF YOUR HUGGINGFACE LLAMAFORCAUSALLM CHECKPOINT
# ADJUST output_path TO WHERE YOU WANT THE NEMO CHECKPOINT CREATED

if __name__ == "__main__":
  llm.import_ckpt(
    llm.LlamaModel(
      llm.LlamaConfig(
        seq_length=2048,
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=11008,
        num_attention_heads=64
      )
    ),
    source="hf://./TLM64KHF",
    output_path="./TLM64KNeMo")
```
What you write inside the llm.LlamaConfig doesn't really matter. Because the config gets discarded and then later loaded from the HF checkpoint's config.json. However you are required to write something hypothetically valid in there, because it does get validated during the config object's creation.

`source` is `"hf://"` + location of your HuggingFace LlamaForCausalLM checkpoint.

`output_path` is folder where NeMo checkpoint will be created.
### 3. Run conversion script
After that, just run it:
```
# YOU HAVE TO RUN THIS OUTSIDE THE REPO ROOT
# ON LOCAL INFRASTRUCTURE RUN THIS INSIDE DOCKER CONTAINER.
# ON SUPERCOMPUTERS TO RUN SEE 7.2 Computationally heavy scripts

python convert.py
```
If all goes successfully, you sould see something like this:
```
...
  [NeMo I 2026-04-08 09:52:02 nemo_logging:393] Successfully saved checkpoint from iteration       0 to TLM64KNeMo
  [NeMo I 2026-04-08 09:52:03 nemo_logging:393] Async finalization time took 33.940 s
  Converted Llama model to Nemo, model saved to TLM64KNeMo in torch.bfloat16.
  ✓ Checkpoint imported to TLM64KNeMo
  Imported Checkpoint
  ├── context/
  │   ├── artifacts/
  │   │   └── generation_config.json
  │   ├── nemo_tokenizer/
  │   │   ├── special_tokens_map.json
  │   │   ├── tokenizer.json
  │   │   ├── tokenizer.model
  │   │   └── tokenizer_config.json
  │   ├── io.json
  │   └── model.yaml
  └── weights/
      ├── .metadata
      ├── __0_0.distcp
      ├── __0_1.distcp
      ├── common.pt
      └── metadata.json
```
#### ### 4. Potential Problems
- The `if __name__ == "__main__":` part of the convert script is mandatory.

- If you see an error like this:
`ValueError: torch_dtype is not of type str/torch.dtype`
You have to rename your `dtype` to `torch_dtype` in your HuggingFace model's config.json

- If you applied my monkey patches, then you cannot convert models that use RoPE positional embeddings, the model has to use YaRN.

# How do I get Data?
## How do I Tokenize New Data?
You have to run this command from inside the NeMo repo folder. So that you'd use the files from this repository not the NeMo installed in the docker container.
```
# YOU HAVE TO RUN THIS FROM THE NEMO REPO ROOT.
# ON LOCAL INFRASTRUCTURE RUN THIS INSIDE DOCKER CONTAINER.
# ON SUPERCOMPUTERS TO RUN SEE 7.2 Computationally heavy scripts. 
#   Though you might get away with 7.1 Lightweight scripts for small datasets.
# ADJUST --input --output-prefix and --tokenizer-model TO YOUR STUFF.

python scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
  --input ../eng.jsonl \
  --output-prefix ../eng2 \
  --json-keys text \
  --tokenizer-library sentencepiece \
  --tokenizer-model ../TLM/tokenizer.model \
  --append-eod \
  --tilde-open-eod
```

`--input` should be a classic `.jsonl` file with dicts that look like `{"text": "<sample text>"}`.

`--output-prefix` is the place where *_text_document.idx and *_text_document.bin files will be placed.

`--tilde-open-eod` is a flag that overrides the EOD to be 48 as in TildeOpen30B.
## How do I Convert Old GPT NeoX Data to NeMo Format?
I wrote two scripts:
- One script `tilde-nlp/llm-gpt-neox/tools/datasets/geox_2_picklenp.py` (different repo) for converting GPT NeoX datasets to pickle files
- The other script NeMo_TildeOpen/picklenp_2_nemo.py for converting from those pickle files to NeMo datasets.
### 1. Convert data from GPT NeoX format to intermediate format
Launch environment that can run GPT NeoX stuff.
```
# ADJUST FOR HOW YOU LAUNCH YOUR GPT NEOX ENVIRONMENT STUFF ON YOUR SERVER.
# Example provided is how I allocated resources on LUMI when converting our extension phase datapacks to intermediate format.

module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems
srun --job-name=test --account=project_465002038 --partition=dev-g --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=7 --mem-per-gpu=60G --time=2:00:00 --nodes=1 singularity shell /scratch/project_465002038/environment/containers/rocm603_flash.sif
$WITH_CONDA
```
Run converter inside environment.
```
# ADJUST LOCATION WHERE YOU HAVE THE llm-gpt-neox REPOSITORY. 
# ADJUST FOR WHERE YOU HAVE YOUR DATA.

python /project/project_465002038/IP/masking-llm-gpt-neox/llm-gpt-neox/tools/datasets/geox_2_picklenp.py \
  --input-prefix /scratch/project_465002038/fictional_data \
  --output /scratch/project_465002038/fictional_data.pkl
```
`--input-prefix` is the part of the GPT NeoX dataset names without the .idx/.bin
### 2. Convert data from intermediate format to NeMo format
```
# YOU MUST RUN THIS FROM THE NEMO REPO ROOT.
# ON LOCAL INFRASTRUCTURE RUN THIS INSIDE DOCKER CONTAINER.
# ON SUPERCOMPUTERS TO RUN SEE 7.2 Computationally heavy scripts. 
#   Though you might get away with 7.1 Lightweight scripts for small datasets.
# ADJUST FOR LOCATION OF YOUR ACTUAL DATA.

python picklenp_2_nemo.py \
  --input ../fictional_data.pkl \
  --output-prefix ../fictional_data
```
`--output-prefix` is the file path without the .bin/.idx that NeMo adds to the end of datasets.
## Note on Shuffling NeMo Datasets
When running `gpt_train.py` the validation set is chosen by taking samples from the end of your provided `.idx/.bin` file, meanwhile the training data is taken from the beginning of that file. This can lead to biased validation numbers if your data was not shuffled before tokenization. So you probably want to shuffle your data at some point.
## How do I Prune a Model (Local Infrastructure)
```
# COMMANDS MUST BE RUN INSIDE DOCKER CONTAINER.

# ADJUST TO GPUS YOU WANT TO USE.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# COMMAND MUST BE RUN OUTSIDE REPO ROOT.
# ADJUST --nproc_per_node, --devices, --pp_size TO NUMBER OF GPUS.
# ADJUST ALL THE OTHER PARAMETERS TOO. PARAMETER MEANINGS AFTER THIS SCRIPT.
torchrun --nproc_per_node 4 NeMo_TildeOpen/scripts/llm/gpt_prune.py \
  --devices 4 \
  --tp_size 1 \
  --pp_size 4 \
  --seq_length 65536 \
  --restore_path ./TLM64KNeMo \
  --save_path ./TLM64K_Real_15B_Optimal \
  --data_paths ./ext_nemo/ext \
  --mbs 1 \
  --num_layers_in_last_pipeline_stage 3 \
  --num_train_samples 1024 \
  --target_ffn_hidden_size 14336 \
  --target_hidden_size 4096
```
`--tp_size` - Tensor parallelism. It doesn't work for pruning as far as I know, and has to be 1.

`--restore_path` - Location of source model that you wish to prune.

`--data_paths` - Path to NeMo `.idx/.bin` dataset without the extension.

`--mbs` - Batch size.

`--num_train_samples` - Number of samples, not number of batches/train steps.

`--num_layers_in_last_pipeline_stage` You can choose how many layers are in the last pipeline stage. You had to do some poking around here because of some divisibility issues.

`--target_ffn_head_size --target_hidden_size --target_num_attention_heads --target_query_groups --target_num_layers` These parameters decide what shape the pruned model will be. I might be wrong but I think the `--target_query_groups` one didn't work.
# How do I Prune a Model? (Supercomputer)
The scripts we used for doing pruning on JUPITER are available under `prod_scripts/`. 
I'll follow the prune_real_xxx_optimal.sh scripts as an example. They were used to perform the 30B->15B pruning on JUPITER.
The example scripts are written to work from the parent directory of the repo root.
## 1. Write an sbatch script
contents of `prod_scripts/prune_real_sbatchscript_optimal.sh`
```
#!/bin/bash
#SBATCH --account=jureap133 \
#SBATCH --partition=booster \
#SBATCH --nodes=1 \
#SBATCH --ntasks-per-node=1 \
#SBATCH --gres=gpu:4 \
#SBATCH --cpus-per-task=244 \
#SBATCH --time=8:30:00 \
#SBATCH --mem=0 \

# ADJUST --account TO YOUR PROJECT NAME.
# ADJUST --partition TO WHATEVER PARTITION YOU HAVE TO USE ON THE SUPERCOMPUTER.
# ADJUST --gres TO WHATEVER GPUS YOUR SUPERCOMPUTER HAS IN A NODE.
# ADJUST --cpus-per-task TO HOWEVER MANY CPUS GPU NODES ON YOUR SUPERCOMPUTER HAVE.
# ADJUST --time TO HOW LONG YOU THINK THE PRUNING WILL TAKE.

srun prune_real_singularityscript_optimal.sh
```
This script just tells slurm the computing resources necessary for the job and launches `prune_real_singularity_optimal.sh` once per node when the resources are allocated.
## 2. Write singularity script
contents of `prod_scripts/prune_real_singularityscript_optimal.sh`
```
#!/bin/bash

# ADJUST TO YOUR SINGULARITY CONTAINER LOCATION.
# ADJUST -B BINDINGS TO YOUR SUPERCOMPUTERS FILESYSTEM BINDINGS.

singularity exec --nv -B /e:/e nemo_tildeopen_3.sif ./prune_real_innerscript_optimal.sh
```
The script just launches the next script inside a singularity container. My and your brain would probably start to hurt if this was not a separate script.
## 3. Write script to be run in containers
contents of `prod_scripts/prune_real_innerscript_optimal.sh`
```
#!/bin/bash

# ADJUST THESE TO LOCATIONS ON YOUR SCRATCH FOLDER.
# These exports are necessary so your home folder doesn't overflow.
export HF_HUB_CACHE=/e/project1/jureap133/ingus/nemo/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/e/project1/jureap133/ingus/nemo/triton_cache
export TORCH_HOME=/e/project1/jureap133/ingus/nemo/torch_cache
export TORCH_EXTENSIONS_DIR=/e/project1/jureap133/ingus/nemo/torch_extensions

# ADJUST --nproc_per_node, --devices, --pp_size TO NUMBER OF GPUS.
# ADJUST ALL THE OTHER PARAMETERS TOO. PARAMETER MEANINGS AFTER THIS SCRIPT.
torchrun --nproc_per_node 4 NeMo_TildeOpen/scripts/llm/gpt_prune.py \
  --devices 4 \
  --tp_size 1 \
  --pp_size 4 \
  --seq_length 65536 \
  --restore_path ./TLM64KNeMo \
  --save_path ./TLM64K_Real_15B_Optimal \
  --data_paths ./ext_nemo/ext \
  --mbs 1 \
  --num_layers_in_last_pipeline_stage 3 \
  --num_train_samples 1024 \
  --target_ffn_hidden_size 14336 \
  --target_hidden_size 4096
```
`--tp_size` - Tensor parallelism. It doesn't work for pruning as far as I know, and has to be 1.

`--restore_path` - Location of source model that you wish to prune.

`--data_paths` - Path to NeMo `.idx/.bin` dataset without the extension.

`--mbs` - Batch size.

`--num_train_samples` - Number of samples, not number of batches/train steps.

`--num_layers_in_last_pipeline_stage` You can choose how many layers are in the last pipeline stage. You had to do some poking around here because of some divisibility issues.

`--target_ffn_head_size --target_hidden_size --target_num_attention_heads --target_query_groups --target_num_layers` These parameters decide what shape the pruned model will be. I might be wrong but I think the `--target_query_groups` one didn't work.
## 4. Run it
```
sbatch prune_real_sbatchcript_optimal.sh
```
# How do I Distil a Model? (Local Infrastructure)
```
# RUN THIS INSIDE THE DOCKER CONTAINER.

export CUDA_VISIBLE_DEVICES=0,1,2,3

# ADJUST PRETTY MUCH ALL COMMAND ARGUMENTS.
# You can see nonobvious argument explanations bellow in the (Supercomputer) section.

torchrun --nproc_per_node 4 \
    NeMo_TildeOpen/scripts/llm/gpt_train.py \
      --devices 4 \
      --num_nodes 256 \
      --tp_size 4 \
      --pp_size 1 \
      --model_path TLM64K_Real_15B_Optimal \
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
      --name "Optimal_v3" \
      --log_interval 1 \
      --val_check_interval 64 \
      --limit_val_batches 5 \
      --save_top_k -1 \
      --sync_checkpoints \
      --max_checkpoints 6 \
      --data_paths ./ext_nemo/ext
```
# How do I Distill a Model? (Supercomputer)
I'll follow the distil_real_xxx_optimal.sh scripts as an example. They were used to perform the 30B->15B pruning on JUPITER.
The example scripts are written to work from the parent directory of the repo root.
## 1. Write sbatch script
contents of `prod_scripts/distil_real_sbatchscript_optimal.sh`
```
#!/bin/bash
#SBATCH --account=jureap133 \
#SBATCH --partition=booster \
#SBATCH --nodes=256 \
#SBATCH --ntasks-per-node=1 \
#SBATCH --gres=gpu:4 \
#SBATCH --cpus-per-task=288 \
#SBATCH --time=3:00:00 \
#SBATCH --mem=0 \

# ADJUST --account TO YOUR PROJECT NAME.
# ADJUST --partition TO WHATEVER PARTITION YOU HAVE TO USE ON THE SUPERCOMPUTER.
# ADJUST --gres TO WHATEVER GPUS YOUR SUPERCOMPUTER HAS IN A NODE.
# ADJUST --cpus-per-task TO HOWEVER MANY CPUS GPU NODES ON YOUR SUPERCOMPUTER HAVE.
# ADJUST --time TO HOW LONG YOU THINK THE PRUNING WILL TAKE.

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

# This part prevents coredumps from spamming up your project folder.
ulimit -c 0
rm ./core.jpbo*

# This is here just for logging purpouses.
scontrol show hostname $SLURM_NODELIST > nodelist_$SLURM_JOB_ID.txt

srun distil_real_singularityscript_optimal.sh
```
## 2. Write a singularity script
contents of `prod_scripts/distil_real_singularityscript_optimal.sh`
```
#!/bin/bash

# ADJUST TO YOUR SINGULARITY CONTAINER LOCATION.
# ADJUST -B BINDINGS TO YOUR SUPERCOMPUTERS FILESYSTEM BINDINGS.

singularity exec --nv -B /e:/e nemo_tildeopen_3.sif ./distil_real_innerscript_optimal.sh
```
## 3. Write script to be run in containers
contents of `prod_scripts/distil_real_innerscript_optimal.sh`
```
#!/bin/bash

# This is a magic line that help with VRAM on JUPITER. 
export NCCL_BUFFSIZE=2097152

# ADJUST THESE TO LOCATIONS ON YOUR SCRATCH FOLDER.
# These exports are necessary so your home folder doesn't overflow.
export HF_HUB_CACHE=/e/project1/jureap133/ingus/nemo/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/e/project1/jureap133/ingus/nemo/triton_cache
export TORCH_HOME=/e/project1/jureap133/ingus/nemo/torch_cache
export TORCH_EXTENSIONS_DIR=/e/project1/jureap133/ingus/nemo/torch_extensions/$SLURM_JOB_ID

# ADJUST PRETTY MUCH ALL OF THE PARAMETERS BELLOW. EXPLANATIONS FOR TRICKY ONES AFTER THIS CODE BLOCK.

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
      --model_path TLM64K_Real_15B_Optimal \
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
      --name "Optimal_v3" \
      --log_interval 1 \
      --val_check_interval 64 \
      --limit_val_batches 5 \
      --save_top_k -1 \
      --sync_checkpoints \
      --max_checkpoints 6 \
      --data_paths ./ext_nemo/ext
```
`--nproc_per_node` - Number of GPUs per node.

`--devices` - Number of GPUs per node.

`--tp_size` - Tensor parallelism. Does actually work during distillation.

`--pp_size` - Pipeline parallelism.

`--model_path` - Location where pruned but untrained NeMo checkpoint is located.

`--recompute_granularity --recompute_method` - activation checkpointing settings. Leave as is.

`--recompute_num_layers` - Either meant the interval of activation checkpoint, or the amount of layers that are checkpointed. Either way you might need to adjust this for divisibility.

`--max_steps` - Number of training steps in distillation.

`--gbs` - Global batch size.

`--legacy_ckpt` - Necessary if you get errors that look like `[rank0]: RuntimeError: Missing key in checkpoint state_dict: module.decoder.final_layernorm._extra_state/shard_0_1.`

`--log_dir` and `--name` - Checkpoints, TensorBoard logs and other random stuff that NeMo logs will be saved under the path `--log_dir/--name/`

`--val_check_interval` - Interval for running validation and checkpointing.

`--limit_val_batches` - Number of batches done for validation.

`--save_tok_k` - If this is set to -1, then it will save all checkpoints. If it's an integer, then saves top k checkpoints by validation loss and autodeletes ones that have less. I remember this was buggy.

`--sync_checkpoitns` - Flag that makes NeMo do synchronous checkpointing rather than asynchronous checkpointing. There's a tradeoff that async checkpoints had some bug I don't remember, but it was serious enough that I switched to sync checkpoints. Meanwhile sync checkpoints had a bug where they don't save model config.

`--max_checkpoints` - Training exists after making this many checkpoints.

`--data_paths` - Path to training data `.idx/.bin`/.
## 4. Run it.
```
sbatch distil_real_sbatchscript_optimal.sh
```
# How do You convert NeMo Llama Model to HuggingFace?
## 1. Create Convert Script
You have to create the convert script outside the NeMo repository folder, so that the container's envionment's NeMo is used for this.
Create a convert `convert_nemo_hf.py` script by analogy with this and run it: 
```
if __name__ == "__main__":
  from nemo.collections import llm

  smth = llm.export_ckpt(
    path="/local_data/ingus/nemo/ol3b_prune_logs_2/OL3B_test_18/checkpoints/OL3B_test_18--val_loss=2.0893-epoch=0-consumed_samples=800.0-last",
    target="hf",
    output_path="./ol3b_prune_2",
    overwrite=True
  )
```
`--path` - Path to NeMo checkpoint.
`--output_path` - Path where huggingface checkpoint will be generated.
## 2. Run Convert Script
```
# RUN THIS INSIDE DOCKER CONTAINER IF ON LOCAL INFRASTRUCTURE.
# IF ON SUPER COMPUTER RUN THIS IN A SINGULARITY CONTAINER, see section 7.2 Computationally heavy scripts waaay up.

python convert_nemo_hf.py
```
### Note
You might run into conversion issues, because when training with synchronous checkpointing it doesn't save the model config in the checkpoints. 
I don't remember precisely but I think to fix was to copy `model.yaml` and `io.json` from pruned student model to the analogous checkpoint location.
## 3. Adjust YaRN Settings.
I didn't bother fixing the setting for YaRN config export in NeMo.
So you have to manually copy in the same `rope_scaling` setting into the output model's `config.json` from the original model's `config.json`.
# What follows is the previous README.md:

[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# **NVIDIA NeMo Framework**

## Latest News

<!-- markdownlint-disable -->
<details open>
  <summary><b>Pretrain and finetune :hugs:Hugging Face models via AutoModel</b></summary>
      Nemo Framework's latest feature AutoModel enables broad support for :hugs:Hugging Face models, with 25.04 focusing on

  
- <a href=https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#automodelforcausallm>AutoModelForCausalLM<a> in the <a href="https://huggingface.co/models?pipeline_tag=text-generation&sort=trending">Text Generation<a> category
- <a href=https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForImageTextToText>AutoModelForImageTextToText<a> in the <a href="https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending">Image-Text-to-Text<a> category

More Details in Blog: <a href=https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework>Run Hugging Face Models Instantly with Day-0 Support from NVIDIA NeMo Framework<a>. Future releases will enable support for more model families such as Video Generation models.(2025-05-19)
</details>

<details open>
  <summary><b>Training on Blackwell using Nemo</b></summary>
      NeMo Framework has added Blackwell support, with <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html>performance benchmarks on GB200 & B200<a>. More optimizations to come in the upcoming releases.(2025-05-19)
</details>

<details open>
  <summary><b>Training Performance on GPU Tuning Guide</b></summary>
      NeMo Framework has published <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html>a comprehensive guide for performance tuning to achieve optimal throughput<a>! (2025-05-19)
</details>

<details open>
  <summary><b>New Models Support</b></summary>
      NeMo Framework has added support for latest community models - <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html>Llama 4<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/vision/diffusionmodels/flux.html>Flux<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama_nemotron.html>Llama Nemotron<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/hyena.html#>Hyena & Evo2<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/qwen2vl.html>Qwen2-VL<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/qwen2.html>Qwen2.5<a>, Gemma3, Qwen3-30B&32B.(2025-05-19)
</details>


<details open>
  <summary><b>NeMo Framework 2.0</b></summary>
      We've released NeMo 2.0, an update on the NeMo Framework which prioritizes modularity and ease-of-use. Please refer to the <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html>NeMo Framework User Guide</a> to get started.
</details>
<details open>
  <summary><b>New Cosmos World Foundation Models Support</b></summary>
    <details> 
      <summary> <a href="https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform">Advancing Physical AI with NVIDIA Cosmos World Foundation Model Platform </a> (2025-01-09) 
      </summary> 
        The end-to-end NVIDIA Cosmos platform accelerates world model development for physical AI systems. Built on CUDA, Cosmos combines state-of-the-art world foundation models, video tokenizers, and AI-accelerated data processing pipelines. Developers can accelerate world model development by fine-tuning Cosmos world foundation models or building new ones from the ground up. These models create realistic synthetic videos of environments and interactions, providing a scalable foundation for training complex systems, from simulating humanoid robots performing advanced actions to developing end-to-end autonomous driving models. 
        <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/accelerate-custom-video-foundation-model-pipelines-with-new-nvidia-nemo-framework-capabilities/">
          Accelerate Custom Video Foundation Model Pipelines with New NVIDIA NeMo Framework Capabilities
        </a> (2025-01-07)
      </summary>
        The NeMo Framework now supports training and customizing the <a href="https://github.com/NVIDIA/Cosmos">NVIDIA Cosmos</a> collection of world foundation models. Cosmos leverages advanced text-to-world generation techniques to create fluid, coherent video content from natural language prompts.
        <br><br>
        You can also now accelerate your video processing step using the <a href="https://developer.nvidia.com/nemo-curator-video-processing-early-access">NeMo Curator</a> library, which provides optimized video processing and captioning features that can deliver up to 89x faster video processing when compared to an unoptimized CPU pipeline.
      <br><br>
    </details>
</details>
<details open>
  <summary><b>Large Language Models and Multimodal Models</b></summary>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/state-of-the-art-multimodal-generative-ai-model-development-with-nvidia-nemo/">
          State-of-the-Art Multimodal Generative AI Model Development with NVIDIA NeMo
        </a> (2024-11-06)
      </summary>
        NVIDIA recently announced significant enhancements to the NeMo platform, focusing on multimodal generative AI models. The update includes NeMo Curator and the Cosmos tokenizer, which streamline the data curation process and enhance the quality of visual data. These tools are designed to handle large-scale data efficiently, making it easier to develop high-quality AI models for various applications, including robotics and autonomous driving. The Cosmos tokenizers, in particular, efficiently map visual data into compact, semantic tokens, which is crucial for training large-scale generative models. The tokenizer is available now on the <a href=http://github.com/NVIDIA/cosmos-tokenizer/NVIDIA/cosmos-tokenizer>NVIDIA/cosmos-tokenizer</a> GitHub repo and on <a href=https://huggingface.co/nvidia/Cosmos-Tokenizer-CV8x8x8>Hugging Face</a>.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama/index.html#new-llama-3-1-support for more information/">
        New Llama 3.1 Support
        </a> (2024-07-23)
      </summary>
        The NeMo Framework now supports training and customizing the Llama 3.1 collection of LLMs from Meta.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://aws.amazon.com/blogs/machine-learning/accelerate-your-generative-ai-distributed-training-workloads-with-the-nvidia-nemo-framework-on-amazon-eks/">
          Accelerate your Generative AI Distributed Training Workloads with the NVIDIA NeMo Framework on Amazon EKS
        </a> (2024-07-16)
      </summary>
     NVIDIA NeMo Framework now runs distributed training workloads on an Amazon Elastic Kubernetes Service (Amazon EKS) cluster. For step-by-step instructions on creating an EKS cluster and running distributed training workloads with NeMo, see the GitHub repository <a href="https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/2.nemo-launcher/EKS/"> here.</a>
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/nvidia-nemo-accelerates-llm-innovation-with-hybrid-state-space-model-support/">
          NVIDIA NeMo Accelerates LLM Innovation with Hybrid State Space Model Support
        </a> (2024/06/17)
      </summary>
     NVIDIA NeMo and Megatron Core now support pre-training and fine-tuning of state space models (SSMs). NeMo also supports training models based on the Griffin architecture as described by Google DeepMind. 
      <br><br>
    </details>
      <details>
      <summary>
        <a href="https://huggingface.co/models?sort=trending&search=nvidia%2Fnemotron-4-340B">
          NVIDIA releases 340B base, instruct, and reward models pretrained on a total of 9T tokens.
        </a> (2024-06-18)
      </summary>
      See documentation and tutorials for SFT, PEFT, and PTQ with 
      <a href="https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/nemotron/index.html">
        Nemotron 340B 
      </a>
      in the NeMo Framework User Guide.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/nvidia-sets-new-generative-ai-performance-and-scale-records-in-mlperf-training-v4-0/">
          NVIDIA sets new generative AI performance and scale records in MLPerf Training v4.0
        </a> (2024/06/12)
      </summary>
      Using NVIDIA NeMo Framework and NVIDIA Hopper GPUs NVIDIA was able to scale to 11,616 H100 GPUs and achieve near-linear performance scaling on LLM pretraining. 
      NVIDIA also achieved the highest LLM fine-tuning performance and raised the bar for text-to-image training.
      <br><br>
    </details>
    <details>
        <summary>
          <a href="https://cloud.google.com/blog/products/compute/gke-and-nvidia-nemo-framework-to-train-generative-ai-models">
            Accelerate your generative AI journey with NVIDIA NeMo Framework on GKE
          </a> (2024/03/16)
        </summary>
        An end-to-end walkthrough to train generative AI models on the Google Kubernetes Engine (GKE) using the NVIDIA NeMo Framework is available at https://github.com/GoogleCloudPlatform/nvidia-nemo-on-gke. 
        The walkthrough includes detailed instructions on how to set up a Google Cloud Project and pre-train a GPT model using the NeMo Framework.
        <br><br>
      </details>
</details>
<details open>
  <summary><b>Speech Recognition</b></summary>
  <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/">
          Accelerating Leaderboard-Topping ASR Models 10x with NVIDIA NeMo
        </a> (2024/09/24)
      </summary>
      NVIDIA NeMo team released a number of inference optimizations for CTC, RNN-T, and TDT models that resulted in up to 10x inference speed-up. 
      These models now exceed an inverse real-time factor (RTFx) of 2,000, with some reaching RTFx of even 6,000.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/">
          New Standard for Speech Recognition and Translation from the NVIDIA NeMo Canary Model
        </a> (2024/04/18)
      </summary>
      The NeMo team just released Canary, a multilingual model that transcribes speech in English, Spanish, German, and French with punctuation and capitalization. 
      Canary also provides bi-directional translation, between English and the three other supported languages.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/">
          Pushing the Boundaries of Speech Recognition with NVIDIA NeMo Parakeet ASR Models
        </a> (2024/04/18)
      </summary>
      NVIDIA NeMo, an end-to-end platform for the development of multimodal generative AI models at scale anywhere—on any cloud and on-premises—released the Parakeet family of automatic speech recognition (ASR) models. 
      These state-of-the-art ASR models, developed in collaboration with Suno.ai, transcribe spoken English with exceptional accuracy.
      <br><br>
    </details>
  <details>
    <summary>
      <a href="https://developer.nvidia.com/blog/turbocharge-asr-accuracy-and-speed-with-nvidia-nemo-parakeet-tdt/">
        Turbocharge ASR Accuracy and Speed with NVIDIA NeMo Parakeet-TDT
      </a> (2024/04/18)
    </summary>
    NVIDIA NeMo, an end-to-end platform for developing multimodal generative AI models at scale anywhere—on any cloud and on-premises—recently released Parakeet-TDT. 
    This new addition to the  NeMo ASR Parakeet model family boasts better accuracy and 64% greater speed over the previously best model, Parakeet-RNNT-1.1B.
    <br><br>
  </details>
</details>
<!-- markdownlint-enable -->

## Introduction

NVIDIA NeMo Framework is a scalable and cloud-native generative AI
framework built for researchers and PyTorch developers working on Large
Language Models (LLMs), Multimodal Models (MMs), Automatic Speech
Recognition (ASR), Text to Speech (TTS), and Computer Vision (CV)
domains. It is designed to help you efficiently create, customize, and
deploy new generative AI models by leveraging existing code and
pre-trained model checkpoints.

For technical documentation, please see the [NeMo Framework User
Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## What's New in NeMo 2.0

NVIDIA NeMo 2.0 introduces several significant improvements over its predecessor, NeMo 1.0, enhancing flexibility, performance, and scalability.

- **Python-Based Configuration** - NeMo 2.0 transitions from YAML files to a Python-based configuration, providing more flexibility and control. This shift makes it easier to extend and customize configurations programmatically.

- **Modular Abstractions** - By adopting PyTorch Lightning’s modular abstractions, NeMo 2.0 simplifies adaptation and experimentation. This modular approach allows developers to more easily modify and experiment with different components of their models.

- **Scalability** - NeMo 2.0 seamlessly scaling large-scale experiments across thousands of GPUs using [NeMo-Run](https://github.com/NVIDIA/NeMo-Run), a powerful tool designed to streamline the configuration, execution, and management of machine learning experiments across computing environments.

Overall, these enhancements make NeMo 2.0 a powerful, scalable, and user-friendly framework for AI model development.

> [!IMPORTANT]  
> NeMo 2.0 is currently supported by the LLM (large language model) and VLM (vision language model) collections.

### Get Started with NeMo 2.0

- Refer to the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for examples of using NeMo-Run to launch NeMo 2.0 experiments locally and on a slurm cluster.
- For more information about NeMo 2.0, see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
- [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) contains additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run.
- For an in-depth exploration of the main features of NeMo 2.0, see the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
- To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.

### Get Started with Cosmos

NeMo Curator and NeMo Framework support video curation and post-training of the Cosmos World Foundation Models, which are open and available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos) and [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6). For more information on video datasets, refer to [NeMo Curator](https://developer.nvidia.com/nemo-curator). To post-train World Foundation Models using the NeMo Framework for your custom physical AI tasks, see the [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## LLMs and MMs Training, Alignment, and Customization

All NeMo models are trained with
[Lightning](https://github.com/Lightning-AI/lightning). Training is
automatically scalable to 1000s of GPUs. You can check the performance benchmarks using the
latest NeMo Framework container [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

When applicable, NeMo models leverage cutting-edge distributed training
techniques, incorporating [parallelism
strategies](https://docs.nvidia.com/nemo-framework/user-guide/latest/modeloverview.html)
to enable efficient training of very large models. These techniques
include Tensor Parallelism (TP), Pipeline Parallelism (PP), Fully
Sharded Data Parallelism (FSDP), Mixture-of-Experts (MoE), and Mixed
Precision Training with BFloat16 and FP8, as well as others.

NeMo Transformer-based LLMs and MMs utilize [NVIDIA Transformer
Engine](https://github.com/NVIDIA/TransformerEngine) for FP8 training on
NVIDIA Hopper GPUs, while leveraging [NVIDIA Megatron
Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) for
scaling Transformer model training.

NeMo LLMs can be aligned with state-of-the-art methods such as SteerLM,
Direct Preference Optimization (DPO), and Reinforcement Learning from
Human Feedback (RLHF). See [NVIDIA NeMo
Aligner](https://github.com/NVIDIA/NeMo-Aligner) for more information.

In addition to supervised fine-tuning (SFT), NeMo also supports the
latest parameter efficient fine-tuning (PEFT) techniques such as LoRA,
P-Tuning, Adapters, and IA3. Refer to the [NeMo Framework User
Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html)
for the full list of supported models and techniques.

## LLMs and MMs Deployment and Optimization

NeMo LLMs and MMs can be deployed and optimized with [NVIDIA NeMo
Microservices](https://developer.nvidia.com/nemo-microservices-early-access).

## Speech AI

NeMo ASR and TTS models can be optimized for inference and deployed for
production use cases with [NVIDIA Riva](https://developer.nvidia.com/riva).

## NeMo Framework Launcher

> [!IMPORTANT]  
> NeMo Framework Launcher is compatible with NeMo version 1.0 only. [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) is recommended for launching experiments using NeMo 2.0.

[NeMo Framework
Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher) is a
cloud-native tool that streamlines the NeMo Framework experience. It is
used for launching end-to-end NeMo Framework training jobs on CSPs and
Slurm clusters.

The NeMo Framework Launcher includes extensive recipes, scripts,
utilities, and documentation for training NeMo LLMs. It also includes
the NeMo Framework [Autoconfigurator](https://github.com/NVIDIA/NeMo-Megatron-Launcher#53-using-autoconfigurator-to-find-the-optimal-configuration),
which is designed to find the optimal model parallel configuration for
training on a specific cluster.

To get started quickly with the NeMo Framework Launcher, please see the
[NeMo Framework
Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).
The NeMo Framework Launcher does not currently support ASR and TTS
training, but it will soon.

## Get Started with NeMo Framework

Getting started with NeMo Framework is easy. State-of-the-art pretrained
NeMo models are freely available on [Hugging Face
Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia)
and [NVIDIA
NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
These models can be used to generate text or images, transcribe audio,
and synthesize speech in just a few lines of code.

We have extensive
[tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html)
that can be run on [Google Colab](https://colab.research.google.com) or
with our [NGC NeMo Framework
Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
We also have
[playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html)
for users who want to train NeMo models with the NeMo Framework
Launcher.

For advanced users who want to train NeMo models from scratch or
fine-tune existing NeMo models, we have a full suite of [example
scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) that support
multi-GPU/multi-node training.

## Key Features

- [Large Language Models](nemo/collections/nlp/README.md)
- [Multimodal](nemo/collections/multimodal/README.md)
- [Automatic Speech Recognition](nemo/collections/asr/README.md)
- [Text to Speech](nemo/collections/tts/README.md)
- [Computer Vision](nemo/collections/vision/README.md)

## Requirements

- Python 3.10 or above
- Pytorch 2.5 or above
- NVIDIA GPU (if you intend to do model training)

## Developer Documentation

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Install NeMo Framework

The NeMo Framework can be installed in a variety of ways, depending on
your needs. Depending on the domain, you may find one of the following
installation methods more suitable.

- [Conda / Pip](#conda--pip): Install NeMo-Framework with native Pip into a virtual environment.
  - Used to explore NeMo on any supported platform.
  - This is the recommended method for ASR and TTS domains.
  - Limited feature-completeness for other domains.
- [NGC PyTorch container](#ngc-pytorch-container): Install NeMo-Framework from source with feature-completeness into a highly optimized container.
  - For users that want to install from source in a highly optimized container.
- [NGC NeMo container](#ngc-nemo-container): Ready-to-go solution of NeMo-Framework
  - For users that seek highest performance.
  - Contains all dependencies installed and tested for performance and convergence.

### Support matrix

NeMo-Framework provides tiers of support based on OS / Platform and mode of installation. Please refer the following overview of support levels:

- Fully supported: Max performance and feature-completeness.
- Limited supported: Used to explore NeMo.
- No support yet: In development.
- Deprecated: Support has reached end of life.

Please refer to the following table for current support levels:

| OS / Platform              | Install from PyPi | Source into NGC container |
|----------------------------|-------------------|---------------------------|
| `linux` - `amd64/x84_64`   | Limited support   | Full support              |
| `linux` - `arm64`          | Limited support   | Limited support           |
| `darwin` - `amd64/x64_64`  | Deprecated        | Deprecated                |
| `darwin` - `arm64`         | Limited support   | Limited support           |
| `windows` - `amd64/x64_64` | No support yet    | No support yet            |
| `windows` - `arm64`        | No support yet    | No support yet            |

### Conda / Pip

Install NeMo in a fresh Conda environment:

```bash
conda create --name nemo python==3.10.12
conda activate nemo
```

#### Pick the right version

NeMo-Framework publishes pre-built wheels with each release.
To install nemo_toolkit from such a wheel, use the following installation method:

```bash
pip install "nemo_toolkit[all]"
```

If a more specific version is desired, we recommend a Pip-VCS install. From [NVIDIA/NeMo](github.com/NVIDIA/NeMo), fetch the commit, branch, or tag that you would like to install.  
To install nemo_toolkit from this Git reference `$REF`, use the following installation method:

```bash
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout @${REF:-'main'}
pip install '.[all]'
```

#### Install a specific Domain

To install a specific domain of NeMo, you must first install the
nemo_toolkit using the instructions listed above. Then, you run the
following domain-specific commands:

```bash
pip install nemo_toolkit['all'] # or pip install "nemo_toolkit['all']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['asr'] # or pip install "nemo_toolkit['asr']@git+https://github.com/NVIDIA/NeMo@$REF:-'main'}"
pip install nemo_toolkit['nlp'] # or pip install "nemo_toolkit['nlp']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['tts'] # or pip install "nemo_toolkit['tts']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['vision'] # or pip install "nemo_toolkit['vision']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['multimodal'] # or pip install "nemo_toolkit['multimodal']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
```

### NGC PyTorch container

**NOTE: The following steps are supported beginning with 24.04 (NeMo-Toolkit 2.3.0)**

We recommended that you start with a base NVIDIA PyTorch container:
nvcr.io/nvidia/pytorch:25.01-py3.

If starting with a base NVIDIA PyTorch container, you must first launch
the container:

```bash
docker run \
  --gpus all \
  -it \
  --rm \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/pytorch:25.01-py3'}
```

From [NVIDIA/NeMo](github.com/NVIDIA/NeMo), fetch the commit/branch/tag that you want to install.  
To install nemo_toolkit including all of its dependencies from this Git reference `$REF`, use the following installation method:

```bash
cd /opt
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout ${REF:-'main'}
bash docker/common/install_dep.sh --library all
pip install ".[all]"
```

## NGC NeMo container

NeMo containers are launched concurrently with NeMo version updates.
NeMo Framework now supports LLMs, MMs, ASR, and TTS in a single
consolidated Docker container. You can find additional information about
released containers on the [NeMo releases
page](https://github.com/NVIDIA/NeMo/releases).

To use a pre-built container, run the following code:

```bash
docker run \
  --gpus all \
  -it \
  --rm \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/nemo:25.02'}
```

## Future Work

The NeMo Framework Launcher does not currently support ASR and TTS
training, but it will soon.

## Discussions Board

FAQ can be found on the NeMo [Discussions
board](https://github.com/NVIDIA/NeMo/discussions). You are welcome to
ask questions or start discussions on the board.

## Contribute to NeMo

We welcome community contributions! Please refer to
[CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md)
for the process.

## Publications

We provide an ever-growing list of
[publications](https://nvidia.github.io/NeMo/publications/) that utilize
the NeMo Framework.

To contribute an article to the collection, please submit a pull request
to the `gh-pages-src` branch of this repository. For detailed
information, please consult the README located at the [gh-pages-src
branch](https://github.com/NVIDIA/NeMo/tree/gh-pages-src#readme).

## Blogs

<!-- markdownlint-disable -->
<details open>
  <summary><b>Large Language Models and Multimodal Models</b></summary>
    <details>
      <summary>
        <a href="https://blogs.nvidia.com/blog/bria-builds-responsible-generative-ai-using-nemo-picasso/">
          Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso
        </a> (2024/03/06)
      </summary>
      Bria, a Tel Aviv startup at the forefront of visual generative AI for enterprises now leverages the NVIDIA NeMo Framework. 
      The Bria.ai platform uses reference implementations from the NeMo Multimodal collection, trained on NVIDIA Tensor Core GPUs, to enable high-throughput and low-latency image generation. 
      Bria has also adopted NVIDIA Picasso, a foundry for visual generative AI models, to run inference.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/">
          New NVIDIA NeMo Framework Features and NVIDIA H200
        </a> (2023/12/06)
      </summary>
      NVIDIA NeMo Framework now includes several optimizations and enhancements, 
      including: 
      1) Fully Sharded Data Parallelism (FSDP) to improve the efficiency of training large-scale AI models, 
      2) Mix of Experts (MoE)-based LLM architectures with expert parallelism for efficient LLM training at scale, 
      3) Reinforcement Learning from Human Feedback (RLHF) with TensorRT-LLM for inference stage acceleration, and 
      4) up to 4.2x speedups for Llama 2 pre-training on NVIDIA H200 Tensor Core GPUs.
      <br><br>
      <a href="https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility">
      <img src="https://github.com/sbhavani/TransformerEngine/blob/main/docs/examples/H200-NeMo-performance.png" alt="H200-NeMo-performance" style="width: 600px;"></a>
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://blogs.nvidia.com/blog/nemo-amazon-titan/">
          NVIDIA now powers training for Amazon Titan Foundation models
        </a> (2023/11/28)
      </summary>
      NVIDIA NeMo Framework now empowers the Amazon Titan foundation models (FM) with efficient training of large language models (LLMs). 
      The Titan FMs form the basis of Amazon’s generative AI service, Amazon Bedrock. 
      The NeMo Framework provides a versatile framework for building, customizing, and running LLMs.
      <br><br>
    </details>
</details>
<!-- markdownlint-enable -->

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).
