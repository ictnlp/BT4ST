import os
import tqdm
import argparse
import sentencepiece as spm
from cleaner.cleaners import transliteration_cleaners
from pathlib import Path
from tempfile import NamedTemporaryFile
from examples.speech_to_text.data_utils import gen_vocab


def learn_spm_vocab(args):
    with open(os.path.join(args.data_dir, f"train.{args.tgt_lang}")) as f:
        train_text = f.read().splitlines()
    clean_train_text = [transliteration_cleaners(x) for x in train_text]
    with NamedTemporaryFile(mode="w") as f:
        for t in clean_train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            Path(args.dest_dir).absolute() / f"spm_char_{args.tgt_lang}",
            "char",
    )


def apply_spm(args):
    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.dest_dir, f"spm_char_{args.tgt_lang}.model"))
    for split in ["train", "dev", "tst-COMMON"]:
        with open(os.path.join(args.data_dir, f"{split}.{args.tgt_lang}")) as f:
            raw_texts = f.read().splitlines()
        char_texts = []
        for x in tqdm.tqdm(raw_texts):
            x = transliteration_cleaners(x)
            x = sp.encode(x, out_type=str)
            x = " ".join(x)
            char_texts.append(x)
        with open(os.path.join(args.dest_dir, f"{split}.{args.tgt_lang}"), "w") as f:
            f.write("\n".join(char_texts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--dest-dir")
    parser.add_argument("--tgt-lang")
    parser.add_argument("--learn-vocab", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.dest_dir):
        os.mkdir(args.dest_dir)
    if args.learn_vocab:
        learn_spm_vocab(args)
    apply_spm(args)


if __name__ == "__main__":
    main()