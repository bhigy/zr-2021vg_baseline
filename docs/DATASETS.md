### Datasets

Several datasets are used to train the baseline :

| Data | Use | Download link |
---|---|---
| COCO images 2014 | VG | https://cocodataset.org/#download |
| SpokenCOCO | VG | https://groups.csail.mit.edu/sls/downloads/placesaudio/index.cgi|
| Librispeech | K-means and LM | http://www.openslr.org/12 |
| ZeroSpeech 2021 | Evaluation | https://download.zerospeech.com |

#### SpokenCOCO format

If you downloaded the train/val/test sets of COCO images 2014 under `~/corpora/spokencoco`, the SpokenCOCO dataset is expected to lie in the same folder.
You'll have a folder structure that looks like :

```
~/corpora/spokencoco 
│
└─── train2014
└─── val2014
└─── SpokenCOCO
     └─── wavs
          └─── train
          └─── val
```

#### LibriSpeech

If you downloaded LibriSpeech, you're supposed to have a folder structure that looks like this :

```
~/corpora/librispeech
│
└─── dev-clean
└─── dev-other
└─── test-clean
└─── test-other
└─── train-clean-100
└─── train-clean-360
└─── train-other-500
```

Under the same folder, you'll further need to create a folder `train-full-960` that contains `train-clean-100`, `train-clean-360` and `train-other-500`.


#### ZeroSpeech 2021

Nothing to do for the dev/test set of ZeroSpeech 2021. We'll assume that this dataset has been download under `~/corpora/zerospeech2021`.
