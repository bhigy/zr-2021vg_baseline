# ZeroSpeech2021-VG &mdash; Baselines

This repository contains the code to run the baselines for the Zero-Resource Speech Challenge using Visually-Grounded Models of Spoken Language, 2021 edition.

## Overview of the baselines

Our baseline is directly inspired by the audio-only baseline used in the [Zerospeech 2021 challenge](https://github.com/bootphon/zerospeech2021_baseline). 
The main difference is that we use a visually grounded (VG) model to learn our speech representations. Then, the latter are fed to K-means and the language model.
The low-budget baseline replaces the contrastive predictive model (CPC) with a VG model. While the high-budget baseline adds the VG model on top of the CPC model.

| Step | Low-budget baseline | High-budget baseline |
---|---|---
| Input | MFCCs | Waveform |
| Acoustic model (training set) | VG model | CPC-small + VG model |
| Layer used to extract features | 1st recurrent layer | 1st recurrent layer |
| Quantization | k-means | k-means |
| Language Model | LSTM | BERT |

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
