import os
import tqdm
import argparse
import torch
import torchaudio
import numpy as np
from examples.speech_to_text.data_utils import load_tsv_to_dicts
from fairseq.data.audio.audio_utils import (
    parse_path,
)


def read_audio(path, ref_len=None):
    _path, slice_ptr = parse_path(path)
    assert len(slice_ptr) == 2
    waveform, sample_rate = torchaudio.load(_path, frame_offset=slice_ptr[0], num_frames=slice_ptr[1])
    return waveform, sample_rate


def get_spkr_embs(args):
    rows = load_tsv_to_dicts(args.input_tsv)
    all_spkr = [x["speaker"] for x in rows]
    embs_path = os.path.join(args.dest_dir, "spkr_emb.npy")
    if not os.path.exists(embs_path):
        wav2mel = torch.jit.load(args.wav2mel_path)
        dvector = torch.jit.load(args.dvector_path)
        all_embs = []
        for item in tqdm.tqdm(rows):
            path = item["audio"]
            wav_tensor, sample_rate = read_audio(path)
            try:
                mel_tensor = wav2mel(wav_tensor, sample_rate)
                emb_tensor = dvector.embed_utterance(mel_tensor)
                all_embs.append(emb_tensor.detach().numpy())
            except:
                print(wav_tensor.shape)
                all_embs.append(all_embs[-1])
        all_embs = np.array(all_embs)
        np.save(embs_path, all_embs)
    else:
        all_embs = np.load(embs_path)
    assert len(all_embs) == len(all_spkr), (len(all_embs), len(all_spkr))
    return all_embs, all_spkr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv")
    parser.add_argument("--wav2mel-path")
    parser.add_argument("--dvector-path")
    parser.add_argument("--dest-dir")
    args = parser.parse_args()
    if not os.path.exists(args.dest_dir):
        os.mkdir(args.dest_dir)
    get_spkr_embs(args)


if __name__ == "__main__":
    main()