# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
from pathlib import Path
import random
import soundfile as sf
import torch
import numpy as np

from tqdm import tqdm

from fairseq import utils
from fairseq.models.text_to_speech.vocoder import CodeDvecHiFiGANVocoder


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_result(args, sample_id, pred_wav, suffix=""):
    sf.write(
        f"{args.results_path}/{sample_id}{suffix}_pred.wav",
        pred_wav.detach().cpu().numpy(),
        16000,
    )


def load_code(in_file):
    with open(in_file) as f:
        out = [list(map(int, line.strip().split())) for line in f]
    return out


def main(args):
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    with open(args.vocoder_cfg) as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeDvecHiFiGANVocoder(args.vocoder, vocoder_cfg)
    if use_cuda:
        vocoder = vocoder.cuda()

    multispkr = vocoder.model.multispkr
    if multispkr:
        logger.info("multi-speaker vocoder")

    data = load_code(args.in_code_file)
    dvectors = np.load(args.dvectors_path)
    Path(args.results_path).mkdir(exist_ok=True, parents=True)
    for i, d in tqdm(enumerate(data), total=len(data)):
        x = {
            "code": torch.LongTensor(d).view(1, -1),
        }
        suffix = ""
        if multispkr:
            x["spkr"] = torch.tensor(dvectors[i]).unsqueeze(0)

        x = utils.move_to_cuda(x) if use_cuda else x
        wav = vocoder(x, args.dur_prediction)
        dump_result(args, i, wav, suffix=suffix)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-code-file", type=str, required=True, help="one unit sequence per line"
    )
    parser.add_argument(
        "--vocoder", type=str, required=True, help="path to the CodeHiFiGAN vocoder"
    )
    parser.add_argument(
        "--vocoder-cfg",
        type=str,
        required=True,
        help="path to the CodeHiFiGAN vocoder config",
    )
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument(
        "--dur-prediction",
        action="store_true",
        help="enable duration prediction (for reduced/unique code sequences)",
    )
    parser.add_argument(
        "--dvectors-path",
        type=str,
        help="path to dvectors",
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()
