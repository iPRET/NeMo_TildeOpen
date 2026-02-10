import argparse
import pickle as pkl
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset

def parse_args():
  parser = argparse.ArgumentParser("Turns a pickle file of numpy tensors into a NeMo .bin/.idx dataset.")

  parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to pickle file of numpy arrays."
    )

  parser.add_argument(
    "--output-prefix",
    type=str,
    required=True,
    help="Filename where .bin/.idx files will be generated (supply it without the .bin/.idx)."
  )

  return parser.parse_args()


def main(args):
  output_ds = indexed_dataset.make_builder(
    args.output_prefix + ".bin",
    impl="mmap"
  )

  input_ds = open(args.input, "rb")

  c = 0
  while True:
    try:
      tokens = pkl.load(input_ds)
    except Exception as e:
      #print(e)
      break
    c+=1
    if c % 1000000 == 0:
      print("Copied", c, "docs")
    output_ds.add_item(tokens)
    output_ds.end_document()
  output_ds.finalize(args.output_prefix + ".idx")


if __name__ == "__main__":
  main(parse_args())
