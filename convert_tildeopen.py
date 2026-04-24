"""Convert a NeMo checkpoint to HuggingFace format and inject TildeOpen YaRN config.

Usage (run OUTSIDE the NeMo repo folder, so the container's NeMo is used):
    python convert_tildeopen.py --nemo-cp /path/to/nemo/checkpoint --output /path/to/hf/output

The script:
  1. Calls llm.export_ckpt to convert NeMo → HF
  2. Injects the correct rope_scaling (YaRN) config into the output config.json

Author: Claude Opus 4.6 (vibecoded)
"""
import argparse
import json
import os
import sys


# TildeOpen YaRN rope_scaling config.
# No attention_factor — HF5 interprets it literally (wrong), HF4 needs a custom loader anyway.
# truncate: true works for both HF4 (with custom loader) and HF5.
TILDEOPEN_ROPE_SCALING = {
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 10.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "yarn",
    "type": "yarn",
    "truncate": True,
}


def convert(nemo_path: str, output_path: str, overwrite: bool):
    from nemo.collections import llm

    print(f"Converting {nemo_path} -> {output_path}")
    llm.export_ckpt(
        path=nemo_path,
        target="hf",
        output_path=output_path,
        overwrite=overwrite,
    )
    print(f"NeMo -> HF conversion done.")

    # Inject rope_scaling into config.json
    config_path = os.path.join(output_path, "config.json")
    if not os.path.exists(config_path):
        print(f"WARNING: {config_path} not found, cannot inject rope_scaling.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    config["rope_scaling"] = TILDEOPEN_ROPE_SCALING
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Injected YaRN rope_scaling into {config_path}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NeMo checkpoint to HF with TildeOpen YaRN config.")
    parser.add_argument("--nemo-cp", type=str, required=True, help="Path to NeMo checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path for HF output")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    args = parser.parse_args()

    convert(args.nemo_cp, args.output, args.overwrite)
