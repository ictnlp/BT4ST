import os
import argparse
import tqdm
import csv
import pandas as pd

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from yaml import parse

parser = argparse.ArgumentParser()
parser.add_argument("--input-files")
parser.add_argument("--output-file")
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
    filepaths = args.input_files.split(",")
    all_rows = []
    for path in filepaths:
        rows = load_tsv_to_dicts(path)
        all_rows.extend(rows)
    df = pd.DataFrame.from_dict(all_rows)
    save_df_to_tsv(df, args.output_file)

if __name__ == "__main__":
    main()