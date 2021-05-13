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

## References

[1] Nguyen, T. A., de Seyssel, M., Rozé, P., Rivière, M., Kharitonov, E., Baevski, A., Dunbar, E., & Dupoux, E. (2020). The Zero Resource Speech Benchmark 2021: Metrics and baselines for unsupervised spoken language modeling. http://arxiv.org/abs/2011.11588

[2] Chrupała, G. (2019). Symbolic Inductive Bias for Visually Grounded Learning of Spoken Language. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 6452–6462. https://doi.org/10.18653/v1/P19-1647

[3] Higy, B., Elliott, D., & Chrupała, G. (2020). Textual Supervision for Visually Grounded Spoken Language Understanding. Findings of the Association for Computational Linguistics: EMNLP 2020, 2698–2709. https://doi.org/10.18653/v1/2020.findings-emnlp.244

[4] Hsu, W.-N., Harwath, D., Song, C., & Glass, J. (2020). Text-Free Image-to-Speech Synthesis Using Learned Segmental Units. http://arxiv.org/abs/2012.15454

