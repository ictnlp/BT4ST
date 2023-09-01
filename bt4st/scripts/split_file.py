import tqdm
import random
import argparse
import sentencepiece as spm
from pathlib import Path
from cleaner.cleaners import transliteration_cleaners


def preprocess(args, raw_texts):
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    char_texts = []
    for x in tqdm.tqdm(raw_texts):
        x = transliteration_cleaners(x)
        x = sp.encode(x, out_type=str)
        x = " ".join(x)
        char_texts.append(x)
    return char_texts


def main(args):
    with open(args.input_file) as f:
        data = f.read().splitlines()
    random.shuffle(data)
    n_chunks = args.num_chunks
    chunk_size = len(data) // n_chunks + 1
    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True)
    for chunk_id in range(n_chunks):
        begin = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, len(data))
        processed = preprocess(args, data[begin:end])
        with open(output_root / f"wmt.shard{chunk_id}.{args.tgt_lang}", "w") as f:
            f.write("\n".join(data[begin:end]))
        with open(output_root / f"wmt.char.shard{chunk_id}.{args.tgt_lang}", "w") as f:
            f.write("\n".join(processed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file")
    parser.add_argument("--num-chunks", type=int)
    parser.add_argument("--tgt-lang")
    parser.add_argument("--spm-model")
    parser.add_argument("--output-root")
    args = parser.parse_args()

    main(args)