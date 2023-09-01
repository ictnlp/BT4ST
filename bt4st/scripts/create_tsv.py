import os
import argparse
import tqdm
import sacrebleu
import torchaudio
import sentencepiece as spm
import csv
import pandas as pd
import soundfile as sf

from functools import reduce
from typing import Any, Dict, List, Optional, Union
from sacrebleu.metrics import BLEU
from pathlib import Path


def filter_manifest_df(
    df, is_train_split=False, extra_filters=None, min_n_frames=5, max_n_frames=3000
):
    filters = {
        "no speech": df["audio"] == "",
        f"short speech (<{min_n_frames} frames)": df["n_frames"] < min_n_frames,
        "empty sentence": df["tgt_text"] == "",
    }
    if is_train_split:
        filters[f"long speech (>{max_n_frames} frames)"] = df["n_frames"] > max_n_frames
    if extra_filters is not None:
        filters.update(extra_filters)
    invalid = reduce(lambda x, y: x | y, filters.values())
    valid = ~invalid
    print(
        "| "
        + ", ".join(f"{n}: {f.sum()}" for n, f in filters.items())
        + f", total {invalid.sum()} filtered, {valid.sum()} remained."
    )
    return df[valid]

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

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]

parser = argparse.ArgumentParser()
parser.add_argument("--audio-dir")
parser.add_argument("--tgt-path")
parser.add_argument("--id-path")
parser.add_argument("--output-path")
parser.add_argument("--name")
args = parser.parse_args()


def main():
    with open(args.id_path) as f:
        all_id = list(map(int, f.read().splitlines()))
    num_samples = len(all_id)
    with open(args.tgt_path) as f:
        raw_tgt_texts = f.read().splitlines()

    manifest = {c: [] for c in MANIFEST_COLUMNS}
    for idx in tqdm.tqdm(range(num_samples)):
        manifest["id"].append(f"{args.name}_{idx}")
        audio_path = os.path.join(args.audio_dir, f"{idx}_pred.wav")
        waveform, sample_rate = sf.read(audio_path)
        manifest["audio"].append(f"{audio_path}:0:{len(waveform)}")
        manifest["n_frames"].append(len(waveform))
        manifest["src_text"].append("None")
        manifest["tgt_text"].append(raw_tgt_texts[all_id[idx]])
        manifest["speaker"].append("None")
    df = pd.DataFrame.from_dict(manifest)
    df = filter_manifest_df(df, is_train_split=True, min_n_frames=1000, max_n_frames=480000)
    save_df_to_tsv(df, args.output_path)

if __name__ == "__main__":
    main()