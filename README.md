# ZeroSpeech2021-VG &mdash; Baselines

Baselines for the Zero-Resources Speech Challenge using Visually-Grounded Models of Spoken Language, 2021 edition.

## Installation

```bash
conda create --name zrmm python=3.8 & conda activate zrmm
python -m pip install -r requirements.txt
```

## Instructions for running the baselines

The baselines are based on the baselines for [Zerospeech 2021 challenge](https://github.com/bootphon/zerospeech2021_baseline), with the CPC-based acoustic model replaced by a visually-grounded (VG) model similar to the *speech-image* model described in [[1-2]](README.md#references).

### Datasets

We trained our baselines with Flickr8K. To reproduce our results, you need to download:
* [Flickr8K](http://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html) [1].
  Note that downloading from the official website seems broken at the moment.
  Alternatively, the dataset can be obtained from [here](https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names).
* The [Flickr Audio Caption Corpus](https://groups.csail.mit.edu/sls/downloads/flickraudio/) [2].
* Some additional [metadata files](https://surfdrive.surf.nl/files/index.php/s/EF1bA9YYfhiBxoN).

Create a folder to store the dataset (we will assume here that the folder is `~/corpora/flickr8k`)  and move all the files you downloaded there, then extract the content of the archives.

To evaluate models, you will additionally need to download the [ZeroSpeech 2021 dataset](https://download.zerospeech.com). We assume here that the dataset is stored under `~/corpora/zerospeech2021`.

### Preprocessing

Run the preprocessing script to extract input features from Flickr8K:

```bash
python -m platalea.utils.preprocessing flickr8k --flicrk8k_root ~/corpora/flickr8k
```

### Training of the VG model

The VG model can be trained by running:

```bash
mkdir -p exps/vg
cd exps/vg
cp ../../train_vg.py .
python train_vg.py flickr8k --flickr8k_root ~/corpora/flickr8k
```

### Extracting activations

In order to compute the ABX score or train the k-means clustering, the activations of one of the GRU layers need to be extracted.
This can be done with the script `extract_activations.py`; e.g., for the first GRU layer (`rnn0`), run:

```bash
python extract_activations.py exps/vg/<net.best.pt> ~/corpora/zerospeech2021/phonetic/dev-clean/ data/activations/rnn0 \
  --batch_size 8 --layer rnn0 \
  --output_file_extension 'pt' --file_extension wav
```

Where net.best.pt should be replaced with the checkpoint corresponding to the best epoch (see `exps/vg/results.json`).
The GRU layers are named `rnn0` to `rnn3`.
See `python extract_activations.py --help` for more options.

### Computing ABX scores

There are two main ways to compute the ABX scores:

* using the [utility scripts from ZeroSpeech 2021](https://github.com/bootphon/zerospeech2021) to validate and evaluate a submission.

```bash
zerospeech2021-validate ~/corpora/zerospeech2021 data/submission/vg-rnn0 --no-lexical --no-syntactic --no-semantic --only-dev
zerospeech2021-evaluate ~/corpora/zerospeech2021 data/submission/vg-rnn0 --no-lexical --no-syntactic --no-semantic --force-cpu -o results/zerospeech2021/rnn0
```

* using [libri-light's evaluation script](https://github.com/facebookresearch/libri-light/tree/master/eval).

```bash
python <path_to_libri-light_eval>/eval_ABX.py data/activations/rnn0/  ~/corpora/zerospeech2021/phonetic/dev-clean/dev-clean.item --file_extension '.pt' --out results/abx/rnn0 --feature_size 0.02 --distance_mode 'cosine'
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
  --audio_features_fn cpc_small.memmap
```

Similary, you can choose to train from CPC representations by typing :

```bash
python -m platalea.utils.preprocessing flickr8k --flicrk8k_root ~/corpora/flickr8k \
  --cpc_model_path zr2021_models/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt \
  --audio_features_fn cpc_big_2nd_layer.memmap --cpc_gru_level 2
```

And train the visually ground model :

```bash
mkdir -p exps/cpc_vg
cd exps/cpc_vg
cp ../../train_cpc_vg.py .
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

[1] Chrupała, G. (2019). Symbolic Inductive Bias for Visually Grounded Learning of Spoken Language. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 6452–6462. https://doi.org/10.18653/v1/P19-1647

[2] Higy, B., Elliott, D., & Chrupała, G. (2020). Textual Supervision for Visually Grounded Spoken Language Understanding. Findings of the Association for Computational Linguistics: EMNLP 2020, 2698–2709. https://doi.org/10.18653/v1/2020.findings-emnlp.244

