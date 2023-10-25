# Back Translation for Speech-to-text Translation Without Transcripts

**Qingkai Fang, Yang Feng\* | Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS)**

This is a PyTorch implementation of the **ACL 2023 main conference paper** [Back Translation for Speech-to-text Translation Without Transcripts](https://arxiv.org/abs/2305.08709).

## Environment Configuration

1. Clone this repository:

```
git clone git@github.com:ictnlp/BT4ST.git
cd BT4ST/
```

2. Install `fairseq`:

```
cd fairseq/
pip install --editable ./
python setup.py build develop
```

3. We put our scripts related to this work in the `bt4st/` directory. We also borrow the fairseq plug-ins from [CRESS](https://github.com/ictnlp/CRESS/tree/main) to support ST training. (You can also use your commonly used training framework, as the focus of our work is on synthesizing ST data.)

## Getting Started

**In the following, we will show how to synthesize pseudo ST data using the En-De direction as an example.**

1. Download [MuST-C v1.0 dataset](https://ict.fbk.eu/must-c/) to the `data/mustc/` directory and uncompress it. Run the following script to preprocess the dataset.

```
python bt4st/scripts/prep_mustc_data.py \
    --data-root bt4st/data/mustc/ \
    --vocab-type unigram \
    --vocab-size 8000 \
    --tgt-lang de
```

2. Create manifest for quantization.

```
python bt4st/scripts/create_manifest.py \
    --data-root bt4st/data/mustc \
    --tgt-lang de \
    --output-root bt4st/data/bt/
```

3. Quantize source speech with pretrained kmeans model. Please first download the [acoustic model](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) and [kmeans model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin).

```
KM_MODEL_PATH=path/to/km.bin
CKPT_PATH=path/to/hubert_base_ls960.pt
python bt4st/scripts/quantize_with_kmeans.py \
    --feature_type hubert \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer 6 \
    --manifest_path bt4st/data/bt/en-de/${train,dev,tst-COMMON}.txt \
    --out_quantized_file_path bt4st/data/bt/en-de/${train,dev,tst-COMMON}.unit
```

4. Merge consecutive same units and rename the split.

```
python bt4st/scripts/reduce.py \
    --input-file bt4st/data/bt/en-de/${train,dev,tst-COMMON}.unit
    --output-file bt4st/data/bt/en-de/${train,dev,tst-COMMON}.unit
```

5. Make unit dictionary.

```
for x in $(seq 0 $((100 - 1))); do
  echo "$x 1"
done >> bt4st/data/bt/en-de/dict.km100.txt
```

6. Extract target sentences.

```
python bt4st/scripts/extract_target_from_tsv.py \
    --data-root bt4st/data/mustc \
    --tgt-lang de \
    --output-root bt4st/data/bt/
```

7. Learn character-level vocabulary and convert target text into characters.

```
python bt4st/scripts/convert_to_char.py \
    --data-dir bt4st/data/bt/en-de \
    --dest-dir bt4st/data/bt/en-de \
    --tgt-lang de --learn-vocab
```

8. Process into fairseq binary format.

```
python fairseq/fairseq_cli/preprocess.py \
    --source-lang de --target-lang unit \
    --trainpref bt4st/data/bt/en-de/train \
    --validpref bt4st/data/bt/en-de/dev \
    --testpref bt4st/data/bt/en-de/tst-COMMON \
    --destdir bt4st/data/bt/en-de/binary_de_unit \
    --srcdict bt4st/data/bt/en-de/spm_char_de.txt \
    --tgtdict bt4st/data/bt/en-de/dict.km100.txt --workers=20

python fairseq/fairseq_cli/preprocess.py \
    --source-lang unit --target-lang de \
    --trainpref bt4st/data/bt/en-de/train \
    --validpref bt4st/data/bt/en-de/dev \
    --testpref bt4st/data/bt/en-de/tst-COMMON \
    --destdir bt4st/data/bt/en-de/binary_unit_de \
    --srcdict bt4st/data/bt/en-de/dict.km100.txt \
    --tgtdict bt4st/data/bt/en-de/spm_char_de.txt --workers=20
```

9. Train **target-to-unit** and **unit-to-target** models with 4 GPUs.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python fairseq/fairseq_cli/train.py bt4st/data/bt/en-de/binary_de_unit \
    --task translation --fp16 \
    --arch transformer --encoder-layers 6 --decoder-layers 6 --dropout 0.3 \
    --share-decoder-input-output-embed \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --max-source-positions 2048 --max-target-positions 2048 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --log-format 'simple' --log-interval 100 --save-dir checkpoints/de_to_unit --save-interval-updates 1000 \
    --batch-size 50 --max-update 100000 --update-freq 2

CUDA_VISIBLE_DEVICES=0,1,2,3 python fairseq/fairseq_cli/train.py bt4st/data/bt/en-de/binary_unit_de \
    --task translation --fp16 \
    --arch transformer --encoder-layers 6 --decoder-layers 6 --dropout 0.3 \
    --share-decoder-input-output-embed \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --max-source-positions 2048 --max-target-positions 2048 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --log-format 'simple' --log-interval 100 --save-dir checkpoints/unit_to_de --save-interval-updates 1000 \
    --batch-size 50 --max-update 100000 --update-freq 2
```

10. Average last 10 checkpoints.

```
NAME=checkpoints/{de_to_unit,unit_to_de}
python fairseq/scripts/average_checkpoints.py \
    --inputs checkpoints/$NAME \
    --num-update-checkpoints 10 \
    --output checkpoints/$NAME/avg_last_10_update.pt
```

11. Download the monolingual target corpus to `bt4st/data/bt/de/wmt.de`. Since the monolingual corpus may be very large, we split the corpus into several shards and convert text into characters.

```
python bt4st/scripts/split_file.py \
    --input-file bt4st/data/bt/de/wmt.de \
    --num-chunks 4 \
    --tgt-lang de \
    --spm-model bt4st/data/bt/en-de/spm_char_de.model \
    --output-root bt4st/data/bt/de/
```

12. Process monolingual corpus into fairseq binary format.

```
for SHARD in $(seq 0 3) ;
do
    python fairseq/fairseq_cli/preprocess.py \
        --only-source \
        --source-lang de --target-lang unit \
        --srcdict bt4st/data/bt/en-de/spm_char_de.txt \
        --testpref bt4st/data/bt/de/wmt.char.shard$SHARD \
        --destdir bt4st/data/bt/de/binary_wmt_char.shard$SHARD \
        --workers 20
    cp bt4st/data/bt/en-de/dict.km100.txt bt4st/data/bt/de/binary_wmt_char.shard$SHARD/dict.unit.txt
done
```

13. Target-to-unit back translation.

```
SHARD={0,1,2,3}
CKPT_PATH=checkpoints/de_to_unit/avg_last_10_update.pt
RESULTS_PATH=bt4st/data/bt/unit

mkdir -p $RESULTS_PATH
python fairseq/fairseq_cli/generate.py \
    bt4st/data/bt/de/binary_wmt_char.shard$SHARD \
    --path $CKPT_PATH \
    --skip-invalid-size-inputs-valid-test \
    --max-source-positions 1024 --max-target-positions 2048 \
    --max-tokens 4096 \
    --beam 8 --remove-bpe --lenpen 1 --max-len-a 1 \
    --fp16 > $RESULTS_PATH/wmt.shard$SHARD.out
```

14. Extract predicted units and preprocess them.

```
grep "^D\-" $RESULTS_PATH/wmt.shard$SHARD.out | \
    sed 's/^D-//ig' | sort -nk1 | cut -f3 \
    > $RESULTS_PATH/wmt.shard$SHARD.unit

grep "^D\-" $RESULTS_PATH/wmt.shard$SHARD.out | \
    sed 's/^D-//ig' | sort -nk1 | cut -f1 \
    > $RESULTS_PATH/wmt.shard$SHARD.id

python fairseq/fairseq_cli/preprocess.py \
    --only-source \
    --source-lang unit --target-lang de \
    --srcdict bt4st/data/bt/en-de/dict.km100.txt \
    --testpref $RESULTS_PATH/wmt.shard$SHARD \
    --destdir $RESULTS_PATH/binary_bt.shard$SHARD \
    --workers 20
cp bt4st/data/bt/en-de/spm_char_de.txt $RESULTS_PATH/binary_bt.shard$SHARD/dict.de.txt
```

15. Translate the predicted units back into target texts for data selection.

```
CKPT_PATH=checkpoints/unit_to_de/avg_last_10_update.pt
python fairseq/fairseq_cli/generate.py \
    $RESULTS_PATH/binary_bt.shard$SHARD \
    --path $CKPT_PATH \
    --skip-invalid-size-inputs-valid-test \
    --max-source-positions 2048 --max-target-positions 1024 \
    --max-tokens 65536 \
    --beam 1 --remove-bpe --lenpen 1 --max-len-a 1 \
    --fp16 > $RESULTS_PATH/bt.shard$SHARD.out
```

16. Extract speaker embedding for the source speech. Please download `wav2mel.pt` and `dvector-step250000.pt` from [the official repository](https://github.com/yistLin/dvector/releases/tag/v1.1.1) to the `bt4st/data/bt/dvector/` directory.

```
python bt4st/scripts/extract_dvector.py \
    --input-tsv data/mustc/en-de/tgt_only/train.tsv \
    --wav2mel-path bt4st/data/bt/dvector/wav2mel.pt \
    --dvector-path bt4st/data/bt/dvector/dvector-step250000.pt \
    --dest-dir bt4st/data/bt/dvector/train/
```

17. Random select speaker embedding for speech synthesis.

```
num_samples=$(wc -l $RESULTS_PATH/wmt.shard$SHARD.unit | cut -d' ' -f1)

python bt4st/scripts/random_choose_dvector.py \
    --input-file bt4st/data/bt/dvector/train/spkr_emb.npy \
    --output-cnt $num_samples \
    --output-path $RESULTS_PATH/dvector.shard$SHARD.seed$SEED.npy \
    --seed $SEED
```

18. Generate waveform using the unit-to-speech model. Please download the `checkpoint` and `config` from [this link](https://drive.google.com/drive/folders/1icP9zh07Ll1IX6hLDZlD-AlO3g7teApK?usp=sharing).

```
VOCODER_CKPT=path/to/vocoder/g_00100000
VOCODER_CFG=path/to/vocoder/config.json

mkdir -p $RESULTS_PATH/wave.multi.shard$SHARD.seed$SEED
python bt4st/scripts/generate_waveform_from_code_dvec.py \
    --in-code-file $RESULTS_PATH/wmt.shard$SHARD.unit \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
    --dvectors-path $RESULTS_PATH/dvector.shard$SHARD.seed$SEED.npy \
    --results-path $RESULTS_PATH/wave.shard$SHARD.seed$SEED --dur-prediction
```

19. Create manifest for synthesized data.

```
mkdir -p bt4st/data/st
AUDIO_DIR=wave.shard$SHARD.seed$SEED
python bt4st/scripts/create_tsv.py \
    --audio-dir $RESULTS_PATH/wave.shard$SHARD.seed$SEED \
    --tgt-path bt4st/data/bt/de/wmt.shard$SHARD.de \
    --id-path $RESULTS_PATH/wmt.shard$SHARD.id \
    --output-path bt4st/data/st/train.$AUDIO_DIR.tsv --name $AUDIO_DIR
```

20. Data selection based on BLEU scores (Please install sacrebleu==2.0.0).

```
grep "^D\-" $RESULTS_PATH/bt.shard$SHARD.out | \
    sed 's/^D-//ig' | sort -nk1 | cut -f3 \
    > $RESULTS_PATH/bt.shard$SHARD.de

python bt4st/scripts/calc_bleu.py \
    --tgt-path bt4st/data/bt/de/wmt.shard$SHARD.de \
    --re-trans-path $RESULTS_PATH/bt.shard$SHARD.de \
    --id-path $RESULTS_PATH/wmt.shard$SHARD.id \
    --output-path $RESULTS_PATH/bt.shard$SHARD.bleu \
    --spm-model bt4st/data/bt/en-de/spm_char_de.model

python bt4st/scripts/filter_by_bleu.py \
    --input-file bt4st/data/st/train.$AUDIO_DIR.tsv \
    --bleu-path $RESULTS_PATH/bt.shard$SHARD.bleu \
    --output-file bt4st/data/st/train.$AUDIO_DIR.top75.tsv \
    --retain-percent 0.75
```

21. Merge manifests of several shards.

```
python bt4st/scripts/merge_tsv.py \
    --input-files {path/to/shard1.tsv,path/to/shard2.tsv,path/to/shard3.tsv,path/to/shard4.tsv} \
    --output-file bt4st/data/st/train.tsv
```

22. Copy the config and vocabulary files.

```
MUSTC_DIR=bt4st/data/mustc/en-de/tgt_only/
DEST_DIR=bt4st/data/st/
cp $MUSTC_DIR/config.yaml $DEST_DIR/
cp $MUSTC_DIR/spm_unigram8000.* $DEST_DIR/
cp $MUSTC_DIR/dev.tsv $DEST_DIR/
```

23. Pretrain the ST model with synthesized data.

```
python fairseq/fairseq_cli/train.py bt4st/data/st/ \
    --user-dir cress \
    --tgt-lang de \
    --config-yaml config.yaml --train-subset train --valid-subset dev \
    --save-dir checkpoints/en_de_pretrain --num-workers 4 --max-tokens 2000000 --batch-size 32 --max-tokens-text 4096 --max-update 300000 \
    --task speech_and_text_translation --criterion speech_and_text_translation --label-smoothing 0.1 \
    --arch hubert_transformer_postln --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --no-progress-bar --log-format json --log-interval 100 \
    --ddp-backend=legacy_ddp \
    --warmup-updates 4000 --clip-norm 0.0 --seed 1 --update-freq 2 --save-interval-updates 5000 \
    --layernorm-embedding \
    --fp16 \
    --st-training \
    --hubert-model-path checkpoints/hubert_base_ls960.pt
```

24. Finetune the ST model with real data.

```
python fairseq/scripts/average_checkpoints.py \
    --inputs checkpoints/en_de_pretrain \
    --num-update-checkpoints 10 \
    --output checkpoints/en_de_pretrain/avg_last_10_epoch.pt

python fairseq/fairseq_cli/train.py data/mustc/en-de/tgt_only \
    --user-dir cress \
    --tgt-lang de \
    --config-yaml config.yaml --train-subset train --valid-subset dev \
    --save-dir checkpoints/en_de_finetune --num-workers 4 --max-tokens 2000000 --batch-size 32 --max-tokens-text 4096 --max-update 100000 \
    --task speech_and_text_translation --criterion speech_and_text_translation --label-smoothing 0.1 \
    --arch hubert_transformer_postln --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --no-progress-bar --log-format json --log-interval 100 \
    --ddp-backend=legacy_ddp \
    --warmup-updates 4000 --clip-norm 0.0 --seed 1 --update-freq 2 \
    --layernorm-embedding \
    --fp16 \
    --st-training \
    --patience 10 \
    --hubert-model-path checkpoints/hubert_base_ls960.pt \
    --finetune-from-model checkpoints/en_de_pretrain/avg_last_10_epoch.pt
```

25. Evaluate the final model.

```
python fairseq/scripts/average_checkpoints.py \
    --inputs checkpoints/en_de_finetune \
    --num-epoch-checkpoints 10 \
    --output checkpoints/en_de_finetune/avg_last_10_epoch.pt

python fairseq/fairseq_cli/generate.py data/mustc/en-de/tgt_only \
    --user-dir cress \
    --config-yaml config.yaml --gen-subset tst-COMMON --task speech_to_text_modified \
    --path checkpoints/en_de_finetune/avg_last_10_epoch.pt \
    --max-source-positions 900000 \
    --max-tokens 2000000 --beam 8 --lenpen 1.2 --scoring sacrebleu
```

## Citation

If this repository is useful for you, please cite as:

```
@inproceedings{fang-and-feng-2023-back,
	title = {Back Translation for Speech-to-text Translation Without Transcripts},
	author = {Fang, Qingkai and Feng, Yang},
	booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
	year = {2023},
}
```
