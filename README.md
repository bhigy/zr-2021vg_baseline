# zr-2021vg_baseline

Baselines for the Zero-Resources Speech Challenge using VisuallyGrounded Models of Spoken Language, 2021 edition

# Installation

```bash
conda create --name zrmm python=3.8 & conda activate zrmm
python -m pip install -r requirements.txt
```

# Instructions for running the baseline


# Instructions for training CPC (comparison with the audio-only baseline)

### Data Preparation

Please download [Flickr Audio](https://groups.csail.mit.edu/sls/downloads/flickraudio/) and [SpokenCOCO](https://groups.csail.mit.edu/sls/downloads/placesaudio/index.cgi).
Once the corpora have been downloaded, you can run the python scripts to convert the original format to the format needed by CPC.

```bash
python utils/spoken_coco_to_cpc_format.py --audio /path/to/SpokenCOCO/wavs \
  --output /path/to/SpokenCOCO_CPC
  
python flickr_audio_to_cpc_format.py --flickr_audio /path/to/FLICKR8K/flickr_audio/wavs \
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

