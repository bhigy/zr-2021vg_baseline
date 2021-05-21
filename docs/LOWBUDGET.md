## Description of the baseline

Similarly to the baselines for the [ZeroSpeech 2021 challenge - audio-only track](https://github.com/bootphon/zerospeech2021_baseline) [[1]](LOWBUDGET.md#references), the low-budget baseline is mainly composed of two main elements:
* A visually-grounded (VG) model which replaces the CPC-based acoustic model of the audio-only baselines. This model is similar to the *speech-image* model described in [[2-3]](LOWBUDGET.md#references).
* A language model (LM) which is trained on activations extracted from the VG model and quantized through K-means clustering. We re-use here BERT-small which was introduced in the audio-only track.

## Pretrained models

Pretrained models can be downloaded from [the ZeroSpeech challenge website](https://download.zerospeech.com). Simply unzip the archive under the repository root directory.

## Training procedure

If alternatively you want to retrain the baseline from scratch, you will need to go through following steps. We assume that you already have the datasets stored under `~/corpora` (otherwise, you will first need to follow the instructions provided [here](DATASETS.md).

### Training the VG model.

To train the visually-grounded model, you will first need to preprocess the dataset to extract visual and audio features.
This can conveniantly be done by running:

```bash
python -m platalea.utils.preprocessing spokencoco --spokencoco_root ~/corpora/spokencoco
```

The model can then be trained with:

```bash
mkdir -p exps/vg/vg-spokencoco
cd exps/vg/vg-spokencoco
python -m scripts.train_vg spokencoco --epochs 12 --spokencoco_root ~/corpora/spokencoco
python -m platalea.utils.copy_best
cd ../../..
```

After running these commands, the folder `exps/vg/vg-spokencoco` should contain, among other files:
* `result.json`, which contains the performance of the model after each epoch.
* `net.best.pt`, corresponding to the best checkpoint of the model (according to the R@10 metric).

### Training K-means clustering

We train K-means clustering on LibriSpeech-100, using the activations of the first recurrent layer of the VG model.
We use 50 clusters.

#### Extracting activations

To extract the relevant activations run (the GRU layers are named `rnn0` to `rnn3`):

```bash
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/LibriSpeech/train-clean-100 \
    data/activations/vg-spokencoco/librispeech/train-clean-100 \
    --batch_size 8 --layer rnn0 --output_file_extension '.pt' \
    --file_extension '.flac' --recursionLevel 2
```

#### Training the model

We can now train K-means clustering:

```bash
python -m scripts.clustering data/activations/vg-spokencoco/librispeech/train-clean-100 \
    exps/kmeans/vg-spokencoco-rnn0_kmeans-librispeech100-50 \
    --nClusters 50 --MAX_ITER 150 --batchSizeGPU 500 --recursionLevel 2 --save
```

### Training the language model

We finally need to train the BERT-small language model on LibriSpeech 960.
To do that, we need to extract activations for LibriSpeech and quantize them using K-means.

#### Extracting activations

```bash
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/LibriSpeech/train-full-960 \
    data/activations/vg-spokencoco/librispeech/train-full-960 \
    --batch_size 8 --layer rnn0 --output_file_extension '.pt' \
    --file_extension '.flac' --recursionLevel 2
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/LibriSpeech/dev-clean \
    data/activations/vg-spokencoco/librispeech/dev-clean \
    --batch_size 8 --layer rnn0 --output_file_extension '.pt' \
    --file_extension '.flac' --recursionLevel 2
python -m scripts.extract_activations exps/vg/vg-spokencoco/net.best.pt \
    ~/corpora/LibriSpeech/test-clean \
    data/activations/vg-spokencoco/librispeech/test-clean \
    --batch_size 8 --layer rnn0 --output_file_extension '.pt' \
    --file_extension '.flac' --recursionLevel 2
```

#### Quantizing the activations

``` bash
python -m scripts.quantize_activations exps/kmeans/vg-spokencoco-rnn0_kmeans-librispeech100-50/checkpoint_last.pt \
    data/activations/vg-spokencoco/librispeech/train-full-960/rnn0 \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960 \
    --recursionLevel 2
python -m scripts.quantize_activations exps/kmeans/vg-spokencoco-rnn0_kmeans-librispeech100-50/checkpoint_last.pt \
    data/activations/vg-spokencoco/librispeech/dev-clean/rnn0 \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/dev-clean \
    --recursionLevel 2
python -m scripts.quantize_activations exps/kmeans/vg-spokencoco-rnn0_kmeans-librispeech100-50/checkpoint_last.pt \
    data/activations/vg-spokencoco/librispeech/test-clean/rnn0 \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/test-clean \
    --recursionLevel 2
```

#### Training the model

We can now train the model:

``` bash
# Converting quantized output for fairseq
python -m scripts.convert_for_fairseq \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960/quantized_outputs.txt \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960/fairseq.txt
python -m scripts.convert_for_fairseq \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/dev-clean/quantized_outputs.txt \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/dev-clean/fairseq.txt
python -m scripts.convert_for_fairseq \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/test-clean/quantized_outputs.txt \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/test-clean/fairseq.txt

# Preprocessing of the data
fairseq-preprocess --only-source \
    --trainpref data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960/fairseq.txt \
    --validpref data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/dev-clean/fairseq.txt \
    --testpref data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/test-clean/fairseq.txt \
    --destdir data/fairseq-bin-data/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960 \
    --workers 20

# Training
fairseq-train --fp16 \
    data/fairseq-bin-data/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960 \
    --save-dir exps/lm/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-bert-small-librispeech960 \
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
    --mask-multiple-length 5 --mask-prob 0.5 --mask-stdev 5 \
    --max-tokens 4096 --max-update 5000000 --encoder-embed-dim 512 --encoder-ffn-embed-dim 2048 --encoder-attention-heads 8 --encoder-layers 8 \
    --seed 5 --log-format simple --log-interval 10 --skip-invalid-size-inputs-valid-test
```

For examples of alternative language models (LSTM or BERT-big), please see [the instructions](HIGHBUDGET.md) for the high-budget baseline.
