import argparse
import pickle as pkl
import random

def parse_args():
  parser = argparse.ArgumentParser("Reads a pickle file into memory, shuffles it, and prints out.")

  parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to pickle file of numpy arrays."
    )

  parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to output shuffled pickle file."
  )

  return parser.parse_args()


def main(args):
  input_ds = open(args.input, "rb")

  data = []
  print("reading data")
  while True:
    try:
      data.append(pkl.load(input_ds))
    except EOFError as e:
      #print(e)
      break
  input_ds.close()

  print("shuffling")
  random.shuffle(data)

  print("outputting data")
  output_ds = open(args.output, "wb")
  for sample in data:
    pkl.dump(sample, output_ds)
  output_ds.close()


if __name__ == "__main__":
  main(parse_args())
