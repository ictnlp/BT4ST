import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input-file")
parser.add_argument("--output-cnt", type=int)
parser.add_argument("--output-path")
parser.add_argument("--seed", type=int)
args = parser.parse_args()

np.random.seed(args.seed)

def main():
    cdd_dvectors = np.load(args.input_file)
    num_samples = len(cdd_dvectors)
    out_ids = np.random.randint(0, num_samples, (args.output_cnt))
    out_dvectors = cdd_dvectors[out_ids]
    np.save(args.output_path, out_dvectors)

if __name__ == "__main__":
    main()