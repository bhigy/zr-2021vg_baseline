# ZeroSpeech2021-VG &mdash; Baselines

This repository contains the code to run the baselines for the Zero-Resource Speech Challenge using Visually-Grounded Models of Spoken Language, 2021 edition.


## Installation

The package can be installed using following commands:

```bash
conda create --name zrvg python=3.8 & conda activate zrvg
python -m pip install -r requirements.txt
```

## Description of the baselines

## Overview of the baselines

The baselines are based on the baselines for the [Zerospeech 2021 challenge](https://github.com/bootphon/zerospeech2021_baseline) [[1]](README.md#reference), with the CPC-based acoustic model replaced by or complemented with a visually-grounded (VG) model similar to the *speech-image* model described in [[2-3]](README.md#references).

|| Low-budget baseline | High-budget baseline |
---|---|---
| Input | MFCCs | Waveform |
| Acooustic model (training set) | VG model (SpokenCOCO) | CPC-small (LibriSpeech-960) + <br> VG model (SpokenCOCO)|
| Quantization | k-means (LibriSpeech-100) | k-means (LibriSpeech-100) |


## References

[1] Nguyen, T. A., de Seyssel, M., Rozé, P., Rivière, M., Kharitonov, E., Baevski, A., Dunbar, E., & Dupoux, E. (2020). The Zero Resource Speech Benchmark 2021: Metrics and baselines for unsupervised spoken language modeling. http://arxiv.org/abs/2011.11588

[2] Chrupała, G. (2019). Symbolic Inductive Bias for Visually Grounded Learning of Spoken Language. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 6452–6462. https://doi.org/10.18653/v1/P19-1647

[3] Higy, B., Elliott, D., & Chrupała, G. (2020). Textual Supervision for Visually Grounded Spoken Language Understanding. Findings of the Association for Computational Linguistics: EMNLP 2020, 2698–2709. https://doi.org/10.18653/v1/2020.findings-emnlp.244

[4] Hsu, W.-N., Harwath, D., Song, C., & Glass, J. (2020). Text-Free Image-to-Speech Synthesis Using Learned Segmental Units. http://arxiv.org/abs/2012.15454

