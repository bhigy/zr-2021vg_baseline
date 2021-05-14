# Prerequisites

First, we'll need to download the checkpoints of the audio-only baseline as we'll be using pretrained CPC models. 
You can do that by running :

```bash
curl -L -o exps.zip https://www.dropbox.com/sh/xcdtml8a3go3fk8/AAAfymU80lS6ZKMr3y9okwApa?dl=0
unzip exps.zip -d exps
```

We'll assume that these checkpoints are stored under `~/zr2021vg_baseline/zr2021_models`.
At this point, if you have followed the instructions in the [Datasets](./docs/DATASETS.md) instructions, you should have all the required datasets and models under :

* `~/corpora/spokencoco`
* `~/corpora/librispeech`
* `~/corpora/zerospeech2021`
* `~/zr2021vg_baseline/zr2021_models`

Let's warm up the GPUs then !

## 1) Train the visually grounded model 

First, we must extract CPC representations of SpokenCOCO :

```bash
python -m platalea.utils.preprocessing spokencoco --spokencoco_root ~/corpora/spokencoco \
  --cpc_feature_size 256 --audio_features_fn cpc_small.pt \
  --cpc_model_path ~/zr2021vg_baseline/zr2021_models/checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt
```

This will create a file `~/corpora/spokencoco/cpc_small.pt` containing the CPC features of the audio files, as well as another file `~/corpora/spokencoco/resnet_features.pt` containing the visual features.
Then, we can train the visually grounded model with :

```bash
mkdir -p exps/vg/cpc_small_vg-spokencoco
cd exps/vg/cpc_small_vg-spokencoco 
python -m scripts.train_cpc_vg spokencoco --epochs 12 --spokencoco_root ~/corpora/spokencoco --cpc_feature_size 256 --audio_features_fn cpc_small.pt
```

## 2) Cluster with K-means from the VG representations

Similarly, we first extract CPC+VG representations of librispeech 100h : 

 ```bash
python -m scripts.extract_activations exps/vg/cpc_small_vg-spokencoco/net.best.pt \
  ~/corpora/librispeech/train-clean-100 \
  data/activations/cpc_small_vg-spokencoco/librispeech/train-clean-100 \
  --batch_size 8 --layer rnn0 --output_file_extension '.pt' \
  --file_extension '.flac' --recursionLevel 2 --audio_features_fn 'cpc_small.pt' \
  --cpc_model_path ~/zr2021vg_baseline/zr2021_models/checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt
```

The first parameter is the path to the VG model whose representations need to be extracted.\
The second parameter is the path to the librispeech 100h set of audio files.\
The third parameter is the output folder where representations will be extracted.\
This script will first extract CPC representations. Then it will feed the latter to the VG model whose representations of the first recurrent layer will be extracted.
We then train the K-means model :

```bash
python -m scripts.clustering data/activations/cpc_small_vg-spokencoco/librispeech/train-clean-100 \
  exps/kmeans/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50 \
  --nClusters 50 --MAX_ITER 150 --batchSizeGPU 500 --recursionLevel 2 --save
```

## 3) Train the language model

We must extract CPC+VG representations of `~/corpora/librispeech/{train-full-960,dev-clean,test-clean}`. We can do so by typing :

```bash
# train-full-960
python -m scripts.extract_activations exps/vg/cpc_small_vg-spokencoco/net.best.pt \
  ~/corpora/librispeech/train-full-960 \
  data/activations/cpc_small_vg-spokencoco/librispeech/train-full-960 \
  --batch_size 8 --layer rnn0 --output_file_extension '.pt' \
  --file_extension '.flac' --recursionLevel 2 --audio_features_fn 'cpc_small.pt' \
  --cpc_model_path ~/zr2021vg_baseline/zr2021_models/checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt
# dev-clean
python -m scripts.extract_activations exps/vg/cpc_small_vg-spokencoco/net.best.pt \
  ~/corpora/librispeech/dev-clean \
  data/activations/cpc_small_vg-spokencoco/librispeech/dev-clean \
  --batch_size 8 --layer rnn0 --output_file_extension '.pt' \
  --file_extension '.flac' --recursionLevel 2 --audio_features_fn 'cpc_small.pt' \
  --cpc_model_path ~/zr2021vg_baseline/zr2021_models/checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt
# test-clean
python -m scripts.extract_activations exps/vg/cpc_small_vg-spokencoco/net.best.pt \
  ~/corpora/librispeech/test-clean \
  data/activations/cpc_small_vg-spokencoco/librispeech/test-clean \
  --batch_size 8 --layer rnn0 --output_file_extension '.pt' \
  --file_extension '.flac' --recursionLevel 2 --audio_features_fn 'cpc_small.pt' \
  --cpc_model_path ~/zr2021vg_baseline/zr2021_models/checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt
```

Once these representations are extracted, we can quantize them :

```bash
# train-full-960
python -m scripts.quantize_activations exps/kmeans/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50 \
  data/activations/cpc_small_vg-spokencoco/librispeech/train-full-960/rnn0 \
  data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960
# dev-clean
python -m scripts.quantize_activations exps/kmeans/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50 \
  data/activations/cpc_small_vg-spokencoco/librispeech/dev-clean/rnn0 \
  data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/dev-clean
# test-clean
python -m scripts.quantize_activations exps/kmeans/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50 \
  data/activations/cpc_small_vg-spokencoco/librispeech/test-clean/rnn0 \
  data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/test-clean
```

The first parameter of `python -m scripts.quantize_activations` is the path to the folder containing the K-means model.\
The second parameter is the path to input representations : the ones that have been extracted from CPC+VG.\
The third parameter is the path to the output representations : the quantized ones.

Next step is to convert the quantized representations to the format needed by fairseq

```bash
# Fairseq format conversion
python -m scripts.convert_for_fairseq \
    data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960/quantized_outputs.txt \
    data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960/fairseq.txt
python -m scripts.convert_for_fairseq \
    data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/dev-clean/quantized_outputs.txt \
    data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/dev-clean/fairseq.txt
python -m scripts.convert_for_fairseq \
    data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/test-clean/quantized_outputs.txt \
    data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/test-clean/fairseq.txt
# Preprocessing
fairseq.preprocess --only-source \
    --trainpref data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960/fairseq.txt \
    --validpref data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/dev-clean/fairseq.txt \
    --testpref data/quantized/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/test-clean/fairseq.txt \
    --destdir data/fairseq-bin-data/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960 \
    --workers 20
```

We can finally train the language model. There exist 3 versions of the LM. We'll go through each of them.

1) LSTM

```bash
fairseq-train --fp16 \
    data/fairseq-bin-data/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960 \
    --task language_modeling \
    --save-dir exps/lm/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-lstm-librispeech960 \
    --keep-last-epochs 2 \
    --tensorboard-logdir tensorboard \
    --arch lstm_lm \
    --decoder-embed-dim 200 \
    --decoder-hidden-size 1024 \
    --decoder-layers 3 \
    --decoder-out-embed-dim 200 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --lr 0.0005 \
    --warmup-updates 1000 \
    --warmup-init-lr 1e-07 \
    --dropout 0.1 \
    --weight-decay 0.01 \
    --sample-break-mode none \
    --tokens-per-sample 2048 \
    --max-tokens 131072 \
    --update-freq 1 \
    --max-update 100000
```

2) BERT base :

```bash
SPAN_SIZE = 5 # equivalent to 100 ms
MAX_TOKENS = 4096
fairseq-train --fp16 \
  data/fairseq-bin-data/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960 \
  --save-dir exps/lm/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert_small-librispeech960\
  --task masked_lm \
  --keep-last-epochs 1 \
  --tensorboard-logdir tensorboard \
  --train-subset train \
  --num-workers 4 \
  --criterion masked_lm \
  --arch roberta_base \
  --sample-break-mode eos --tokens-per-sample 3072 --max-positions 6144 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr 0.0005 --total-num-update 250000 --warmup-updates 10000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --mask-multiple-length $SPAN_SIZE --mask-prob 0.5 --mask-stdev $SPAN_SIZE \
  --max-tokens $MAX_TOKENS --max-update 5000000 --encoder-embed-dim 512 --encoder-ffn-embed-dim 2048 --encoder-attention-heads 8 --encoder-layers 8 \
  --seed 5 --log-format simple --log-interval 10 --skip-invalid-size-inputs-valid-test
```

3) BERT large (16 to 32 GPUs needed) :

```bash
SPAN_SIZE=5 # equivalent to 100 ms
MAX_TOKENS=4096
GPU_PER_TASK=8
CPU_PER_TASK=64
TASKS_PER_NODE=1
NODES=4
TOTAL_GPU=$((GPU_PER_TASK * TASKS_PER_NODE * NODES))
DISTRIBUTED_PORT=52663 
UPDATE_FREQ=$((128 / TOTAL_GPU))

fairseq-train --fp16 data/fairseq-bin-data/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960 \
  --save-dir exps/lm/cpc_small_vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert_large-librispeech960 \
  --task masked_lm \
  --keep-last-epochs 1 \
  --tensorboard-logdir tensorboard \
  --train-subset train \
  --num-workers 4 \
  --criterion masked_lm \
  --arch roberta_base \
  --sample-break-mode eos --tokens-per-sample 3072 --max-positions 6144 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr 0.0005 --total-num-update 250000 --warmup-updates 10000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --mask-multiple-length $SPAN_SIZE --mask-prob 0.5 --mask-stdev $SPAN_SIZE \
  --max-tokens $MAX_TOKENS --max-update 250000 \
  --seed 5 --log-format simple --log-interval 10 --skip-invalid-size-inputs-valid-test \
  --distributed-world-size $TOTAL_GPU --distributed-port $DISTRIBUTED_PORT
```
