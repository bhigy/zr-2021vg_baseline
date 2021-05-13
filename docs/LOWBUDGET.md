## Description of the baseline

Similarly to the baselines for the [ZeroSpeech 2021 challenge - audio-only track](https://github.com/bootphon/zerospeech2021_baseline) [[1]](LOWBUDGET.md#references), the low-budget baseline is mainly composed of two main elements:
* A visually-grounded (VG) model which replaces the CPC-based acoustic model of the audio-only baselines. This model is similar to the *speech-image* model described in [[2-3]](LOWBUDGET.md#references).
* A language model (LM) which is trained on activations extracted from the VG model and quantized through K-means clustering. We re-use here BERT-small which was introduced in the audio-only track.

## Pretrained models

Pretrained models can be downloaded from [the ZeroSpeech challenge website](https://download.zerospeech.com). Simply unzip the archive under the repository root directory.

## Training procedure

If alternatively you want to retrain the baseline from scratch, you will need to go through following steps. We assume that you already have the datasets stored under `~/corpora` (otherwise, you will first need to follow the instructions provided [here](DATASETS.md).

### Training the VG model.

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
python -m scripts.quantize_activations exps/kmeans/vg-spokencoco-rnn0_kmeans-librispeech100-50 \
    data/activations/vg-spokencoco/librispeech/train-full-960/rnn0 \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960 \
    --recursionLevel 2
python -m scripts.quantize_activations exps/kmeans/vg-spokencoco-rnn0_kmeans-librispeech100-50 \
    data/activations/vg-spokencoco/librispeech/dev-clean/rnn0 \
    data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/dev-clean \
    --recursionLevel 2
python -m scripts.quantize_activations exps/kmeans/vg-spokencoco-rnn0_kmeans-librispeech100-50 \
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
fairseq.preprocess --only-source \
    --trainpref data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960/fairseq.txt \
    --validpref data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/dev-clean/fairseq.txt \
    --testpref data/quantized/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/test-clean/fairseq.txt \
    --destdir data/fairseq-bin-data/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960 \
    --workers 20

# Training
fairseq-train --fp16 \
    data/fairseq-bin-data/vg-spokencoco-rnn0_kmeans-librispeech100-50/librispeech/train-full-960 \
    --task language_modeling \
    --save-dir exps/lm/vg-spokencoco-rnn0_kmeans-librispeech100-50_lm-lstm-librispeech100 \
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