import os
import tqdm
import argparse
from pathlib import Path
from examples.speech_to_text.data_utils import (
    load_df_from_tsv
)

SPLITS = ["train", "dev", "tst-COMMON"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    parser.add_argument("--tgt-lang")
    parser.add_argument("--output-root")
    args = parser.parse_args()

    output_root = Path(args.output_root) / f"en-{args.tgt_lang}"
    output_root.mkdir(exist_ok=True)

    for split in SPLITS:
        df = load_df_from_tsv(
            Path(args.data_root) / f"en-{args.tgt_lang}" / "tgt_only" / f"{split}.tsv"
        )
        data = list(df.T.to_dict().values())
        output_file = open(output_root / f"{split}.{args.tgt_lang}", "w")

        for item in tqdm.tqdm(data):
            output_file.write(f"{item['tgt_text']}\n")


if __name__ == "__main__":
    main()