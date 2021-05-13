## Instructions for running the baselines

The baselines are based on the baselines for the [Zerospeech 2021 challenge](https://github.com/bootphon/zerospeech2021_baseline) [[1]](README.md#reference), with the CPC-based acoustic model replaced by or complemented with a visually-grounded (VG) model similar to the *speech-image* model described in [[2-3]](README.md#references).

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

## References

[1] Nguyen, T. A., de Seyssel, M., Rozé, P., Rivière, M., Kharitonov, E., Baevski, A., Dunbar, E., & Dupoux, E. (2020). The Zero Resource Speech Benchmark 2021: Metrics and baselines for unsupervised spoken language modeling. http://arxiv.org/abs/2011.11588

[2] Chrupała, G. (2019). Symbolic Inductive Bias for Visually Grounded Learning of Spoken Language. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 6452–6462. https://doi.org/10.18653/v1/P19-1647

[3] Higy, B., Elliott, D., & Chrupała, G. (2020). Textual Supervision for Visually Grounded Spoken Language Understanding. Findings of the Association for Computational Linguistics: EMNLP 2020, 2698–2709. https://doi.org/10.18653/v1/2020.findings-emnlp.244

[4] Hsu, W.-N., Harwath, D., Song, C., & Glass, J. (2020). Text-Free Image-to-Speech Synthesis Using Learned Segmental Units. http://arxiv.org/abs/2012.15454
