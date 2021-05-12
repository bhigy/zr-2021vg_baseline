# ZeroSpeech2021-VG &mdash; Baselines

This repository contains the code to run the baselines for the Zero-Resource Speech Challenge using Visually-Grounded Models of Spoken Language, 2021 edition.


## Installation

The package can be installed using following commands:

```bash
conda create --name zrvg python=3.8 & conda activate zrvg
python -m pip install -r requirements.txt
```


## Description of the baselines


## Instructions for running the baselines

The baselines are based on the baselines for the [Zerospeech 2021 challenge](https://github.com/bootphon/zerospeech2021_baseline) [[1]](README.md#reference), with the CPC-based acoustic model replaced by or complemented with a visually-grounded (VG) model similar to the *speech-image* model described in [[2-3]](README.md#references).

### Datasets

We trained our baselines with SpokenCOCO (for the visually-grounded model) and LibriSpeech. To evaluate the models, you will additionally need the [ZeroSpeech 2021 dataset](https://download.zerospeech.com).

#### SpokenCOCO

You need to download:
* [COCO images](https://cocodataset.org/#download). SpokenCOCO is based on the 2014 train/val/test sets.
* [SpokenCOCO](https://groups.csail.mit.edu/sls/downloads/placesaudio/index.cgi)

Create a folder to store the dataset (we will assume here that the folder is `~/corpora/spokencoco`) and extract the content of the different archives under this folder.

To train the visually-grounded model, you will need to preprocess the dataset to extract visual and audio features. This can conveniantly be done by running:

```bash
python -m platalea.utils.preprocessing spokencoco
```

#### LibriSpeech

LibriSpeech can be downloaded from [here](http://www.openslr.org/12/).

The low-budget baseline is trained on the *train-clean-100* subset. The high budget baseline uses in addition the *train-clean-300* and *train-other-500* subsets. In that later case, create a folder called *train-960* and symlink/copy the files from the other 3 subsets under this folder. In all cases, you will also need the *dev-clean* and *test-clean* subsets.

As previously, we will assume here that the data is stored under `~/corpora/LibriSpeech`.

#### ZeroSpeech 2021

To evaluate models, you will finally need to download the [ZeroSpeech 2021 dataset](https://download.zerospeech.com). We assume here that the dataset is stored under `~/corpora/zerospeech2021`.

### Training and evaluation

The two baselines are complex pipelines. Training and evaluating them requires to follow many steps in a specific order, alternating the training of the different components with the extraction of the information necessary for each step. To simplify the reproduction of our results, we provide two scripts, `run_lowbudget.py` and `run_highbudget.py`, that can take care of the complete process automatically.

More details about the different options they provide can be obtained using the parameter `-h`. We also provide more details on the different steps [below](README.md#steps).

Finally, pretrained models can be found [here](https://download.zerospeech.com). Simply unzip the archive under the repository root directory. The scripts `run_lowbudget.py` and `run_highbudget.py` will automatically detect the presence of a model's checkpoint and skip the training of that component.

## Steps

We present now in more details the different steps necessary to train the full baseline systems and evaluate them.

* **train_vg.py**: trains the VG model.

```bash
python -m scripts.train_vg spokencoco
```

### Training of the VG model

The VG model can be trained by running:

```bash
mkdir -p exps/vg
cd exps/vg
cp ../../scripts/train_vg.py .
python train_vg.py flickr8k --flickr8k_root ~/corpora/flickr8k
```

### Extracting activations

In order to compute the ABX score or train the k-means clustering, the activations of one of the GRU layers need to be extracted.
This can be done with the script `scripts/extract_activations.py`; e.g., for the first GRU layer (`rnn0`), run:

```bash
python -m extract_activations exps/vgslu/<net.best.pt> ~/corpora/flickr8k/flickr_audio/wavs data/activations/flickr8k/train \
    --batch_size 8 --layer rnn0 --output_file_extension '.pt' \
    --seqList data/datasets/flicrk8k/flickr8k_train.txt --recursionLevel 0
```

Where net.best.pt should be replaced with the checkpoint corresponding to the best epoch (see `exps/vg/results.json`).
The GRU layers are named `rnn0` to `rnn3`.
See `python -m scripts.extract_activations --help` for more options.

### Computing ABX scores

As explained in previous section, you will first need to extract activations for the zerospeech2021 dataset using the script `scripts/extract_activations.py`.

```bash
python -m script.extract_activations exps/vg/<net.best.pt> ~/corpora/zerospeech2021/phonetic/dev-clean/ data/activations/zerospeech2021 \
  --batch_size 8 --layer rnn0 \
  --output_file_extension '.pt' --file_extension '.wav'
```

There are then two main ways to compute the ABX scores:

* using the [utility scripts from ZeroSpeech 2021](https://github.com/bootphon/zerospeech2021) to validate and evaluate a submission.

```bash
zerospeech2021-validate ~/corpora/zerospeech2021 data/submission/vg-rnn0 --no-lexical --no-syntactic --no-semantic --only-dev
zerospeech2021-evaluate ~/corpora/zerospeech2021 data/submission/vg-rnn0 --no-lexical --no-syntactic --no-semantic --force-cpu -o results/zerospeech2021/rnn0
```

* using [libri-light's evaluation script](https://github.com/facebookresearch/libri-light/tree/master/eval).

```bash
python <path_to_libri-light_eval>/eval_ABX.py data/activations/zerospeech2021/rnn0/  ~/corpora/zerospeech2021/phonetic/dev-clean/dev-clean.item --file_extension '.pt' --out results/abx/rnn0 --feature_size 0.02 --distance_mode 'cosine'
```

### Training clustering

To train the k-means clustering on Flickr8K train set, first extract activations as explained [above](#extracting-activations) and then run:

```bash
python clustering.py --recursionLevel 0 --nClusters 50 --MAX_ITER 150 --save --batchSizeGPU 500 data/activations/flickr8k/train/rnn0 exps/kmeans/flickr8k/rnn0
```

To train the k-means clustering on LibriSpeech train-clean-100 set, run:

```bash
python -m scripts.extract_activations exps/vgslu/net.best.pt ~/corpora/LibriSpeech/train-clean-100 data/activations/librispeech/train-clean-100 --batch_size 8 --layer rnn0 --output_file_extension '.pt' --file_extension '.flac'
python clustering.py --recursionLevel 1 --nClusters 50 --MAX_ITER 150 --save --batchSizeGPU 500 data/activations/librispeech/train-clean-100/rnn0 exps/kmeans/librispeech/rnn0
```

## Full evaluation pipeline

```
# Extract activations for ABX task
python extract_activations.py exps/vg/net.15.pt /private/home/marvinlvn/DATA/CPC_data/test/zerospeech2021/phonetic/dev-clean vg_net_15/phonetic/dev-clean --batch_size 8 --layer rnn0 --output_file_extension 'txt' --file_extension wav
python extract_activations.py exps/vg/net.15.pt /private/home/marvinlvn/DATA/CPC_data/test/zerospeech2021/phonetic/dev-other vg_net_15/phonetic/dev-other --batch_size 8 --layer rnn0 --output_file_extension 'txt' --file_extension wav

# Extract activations for sSIMI task
python extract_activations.py exps/vg/net.15.pt /private/home/marvinlvn/DATA/CPC_data/test/zerospeech2021/semantic/dev/librispeech vg_net_15/semantic/dev/librispeech --batch_size 8 --layer rnn0 --output_file_extension 'txt' --file_extension wav
python extract_activations.py exps/vg/net.15.pt /private/home/marvinlvn/DATA/CPC_data/test/zerospeech2021/semantic/dev/synthetic vg_net_15/semantic/dev/synthetic --batch_size 8 --layer rnn0 --output_file_extension 'txt' --file_extension wav

# Checking format is OK
zerospeech2021-validate vg_net_15 --no-lexical --no-syntactic --only-dev

# Running the evaluation
zerospeech2021-evaluate /private/home/marvinlvn/DATA/CPC_data/test/zerospeech2021 submission_test/ --no-syntactic --no-lexical
```

## Instructions for VG model trained from CPC representations

First, let's get the baseline of ZeroSpeech 2021 :

```bash
mkdir zr2021_models && cd zr2021_models
curl https://download.zerospeech.com/2021/baseline_checkpoints.tar.gz | tar xz
echo "{}" > checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_logs.json
echo "{}" > checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_logs.json

cd ..
```

This folder contains pretrained CPC models from which the representations will be extracted.
We can now extract audio and visual features by typing : 

```bash
python -m platalea.utils.preprocessing flickr8k --flicrk8k_root ~/corpora/flickr8k \
  --cpc_model_path zr2021_models/checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt \
  --audio_features_fn cpc_small.pt
```

Similary, you can choose to train from CPC representations by typing :

```bash
python -m platalea.utils.preprocessing flickr8k --flicrk8k_root ~/corpora/flickr8k \
  --cpc_model_path zr2021_models/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt \
  --audio_features_fn cpc_big_2nd_layer.pt --cpc_gru_level 2
```

And train the visually ground model :

```bash
mkdir -p exps/cpc_vg
cd exps/cpc_vg
cp ../../scripts/train_cpc_vg.py .
python train_cpc_vg.py flickr8k --flickr8k_root ~/corpora/flickr8k
```

## Instructions for training CPC (comparison with the audio-only baseline)

### Data Preparation

Please download [Flickr Audio](https://groups.csail.mit.edu/sls/downloads/flickraudio/) and [SpokenCOCO](https://groups.csail.mit.edu/sls/downloads/placesaudio/index.cgi).
Once the corpora have been downloaded, you can run the python scripts to convert the original format to the format needed by CPC.

```bash
python utils/spoken_coco_to_cpc_format.py --audio /path/to/SpokenCOCO/wavs \
  --output /path/to/SpokenCOCO_CPC

python utils/flickr_audio_to_cpc_format.py --flickr_audio /path/to/FLICKR8K/flickr_audio/wavs \
  --flickr_wav2spkr /path/to/FLICKR8K/wav2spk.txt \
  --output /path/to/FLICKR_CPC
```

Original files won't be modified. The scripts will create symbolic links with the following structure :

```bash
PATH_AUDIO_FILES
│
└───speaker1
│        │   seq_11.wav
│        │   seq_12.wav
│        │   ...
│
└───speaker2
        │   seq_21.wav
        │   seq_22.wav
```

### CPC training

To train the CPC model, follow the instructions at https://github.com/facebookresearch/CPC_audio.

Example command:

```bash
python /path/to/CPC_audio/cpc/train.py \
    --pathDB /path/to/SpokenCOCO_CPC \
    --pathCheckpoint /path/to/checkpoints/CPC_small_SpokenCOCO \
    --file_extension .wav --nLevelsGRU 2
```

## References

[1] Nguyen, T. A., de Seyssel, M., Rozé, P., Rivière, M., Kharitonov, E., Baevski, A., Dunbar, E., & Dupoux, E. (2020). The Zero Resource Speech Benchmark 2021: Metrics and baselines for unsupervised spoken language modeling. http://arxiv.org/abs/2011.11588

[2] Chrupała, G. (2019). Symbolic Inductive Bias for Visually Grounded Learning of Spoken Language. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 6452–6462. https://doi.org/10.18653/v1/P19-1647

[3] Higy, B., Elliott, D., & Chrupała, G. (2020). Textual Supervision for Visually Grounded Spoken Language Understanding. Findings of the Association for Computational Linguistics: EMNLP 2020, 2698–2709. https://doi.org/10.18653/v1/2020.findings-emnlp.244

[4] Hsu, W.-N., Harwath, D., Song, C., & Glass, J. (2020). Text-Free Image-to-Speech Synthesis Using Learned Segmental Units. http://arxiv.org/abs/2012.15454

