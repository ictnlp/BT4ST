import os
import argparse
import tqdm
import sacrebleu
import sentencepiece as spm

from sacrebleu.metrics import BLEU


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]

metric = BLEU(smooth_method="exp",effective_order=True)


def calc_bleu(args, idx, pred, ref):
    sp = spm.SentencePieceProcessor(args.spm_model)
    pred = sp.decode(pred.split(" "))
    try:
        bleu = metric.sentence_score(pred, [ref.lower()]).score
        return bleu
    except:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt-path")
    parser.add_argument("--re-trans-path")
    parser.add_argument("--id-path")
    parser.add_argument("--output-path")
    parser.add_argument("--spm-model")
    args = parser.parse_args()

    with open(args.id_path) as f:
        all_id = list(map(int, f.read().splitlines()))
    num_samples = len(all_id)
    with open(args.tgt_path) as f:
        raw_tgt_texts = f.read().splitlines()
    with open(args.re_trans_path) as f:
        re_trans_texts = f.read().splitlines()
    assert len(re_trans_texts) == num_samples
    all_bleu = []
    for idx in tqdm.tqdm(range(num_samples)):
        bleu = calc_bleu(args, idx, re_trans_texts[idx], raw_tgt_texts[all_id[idx]])
        all_bleu.append(bleu)
    with open(args.output_path, "w") as f:
        f.write("\n".join([str(x) for x in all_bleu]))


if __name__ == "__main__":
    main()