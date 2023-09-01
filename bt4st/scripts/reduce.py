import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-file")
parser.add_argument("--output-file")
args = parser.parse_args()

data = open(args.input_file).read().splitlines()
with open(args.output_file, "w") as f:
    for line in data:
        src_tokens = torch.tensor(list(map(int, line.split(" ")))).unique_consecutive().numpy().tolist()
        f.write(" ".join(str(p) for p in src_tokens) + "\n")
