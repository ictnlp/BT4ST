import os
import argparse
import tqdm
import csv
import pandas as pd

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from yaml import parse

parser = argparse.ArgumentParser()
parser.add_argument("--input-file")
parser.add_argument("--bleu-path")
parser.add_argument("--output-file")
parser.add_argument("--retain-percent", type=float)
args = parser.parse_args()

def load_tsv_to_dicts(path: Union[str, Path]) -> List[dict]:
    with open(path, "r") as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        rows = [dict(e) for e in reader]
    return rows

def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        # escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def main():
    rows = load_tsv_to_dicts(args.input_file)
    with open(args.bleu_path) as f:
        all_bleu = list(map(float, f.read().splitlines()))
    all_pairs = []
    for item in rows:
        idx = int(item["id"].split("_")[-1])
        bleu = all_bleu[idx]
        all_pairs.append((item, bleu))
    all_pairs = sorted(all_pairs, key=lambda x: x[1], reverse=True)
    num_retain = int(len(rows) * args.retain_percent)
    filter_rows = [x for x, _ in all_pairs[:num_retain]]
    df = pd.DataFrame.from_dict(filter_rows)
    save_df_to_tsv(df, args.output_file)
    print(f"After filtering: {num_retain} samples.")

if __name__ == "__main__":
    main()