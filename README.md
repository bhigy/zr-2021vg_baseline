# ZeroSpeech2021-VG &mdash; Baselines

This repository contains the code to run the baselines for the Zero-Resource Speech Challenge using Visually-Grounded Models of Spoken Language, 2021 edition.

## Description of the baselines

## Overview of the baselines

The baselines are based on the baselines for the [Zerospeech 2021 challenge](https://github.com/bootphon/zerospeech2021_baseline) [[1]](README.md#reference), with the CPC-based acoustic model replaced by or complemented with a visually-grounded (VG) model similar to the *speech-image* model described in [[2-3]](README.md#references).

|| Low-budget baseline | High-budget baseline |
---|---|---
| Input | MFCCs | Waveform |
| Acooustic model (training set) | VG model (SpokenCOCO) | CPC-small (LibriSpeech-960) + <br> VG model (SpokenCOCO)|
| Layer used to extract features | 1st recurrent layer (rnn0) | 1st recurrent layer (rnn0) |
| Quantization | k-means (LibriSpeech-100) | k-means (LibriSpeech-100) |
| Language Model | LSTM (LibriSpeech-960) | BERT (LibriSpeech-100) |

## How to use ?

1) [Installation](./docs/INSTALLATION.md)
2) [Datasets](./docs/DATASETS.md)
3) [Low budget baseline](./docs/LOWBUDGET.md)
4) [High budget baseline](./docs/HIGHBUDGET.md)
5) [Evaluation](./docs/EVALUATION.md)
6) [Baselines' results](./docs/RESULTS.md)


## Some useful reads

If you want to gain knowledge about the approach adopted in the ZeroSpeech 2021 challenge, we highly recommend going through :

[1] **Description of the challenge in :** Nguyen, T. A., de Seyssel, M., Rozé, P., Rivière, M., Kharitonov, E., Baevski, A., Dunbar, E., & Dupoux, E. (2020). The Zero Resource Speech Benchmark 2021: Metrics and baselines for unsupervised spoken language modeling. http://arxiv.org/abs/2011.11588

[2] **Website of the challenge :** https://zerospeech.com/2021/news.html 

[3] **Description (1st) of the visually grounded models in :** Chrupała, G. (2019). Symbolic Inductive Bias for Visually Grounded Learning of Spoken Language. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 6452–6462. https://doi.org/10.18653/v1/P19-1647

[4] **Description (2nd) of the visually grounded models in :** Higy, B., Elliott, D., & Chrupała, G. (2020). Textual Supervision for Visually Grounded Spoken Language Understanding. Findings of the Association for Computational Linguistics: EMNLP 2020, 2698–2709. https://doi.org/10.18653/v1/2020.findings-emnlp.244
