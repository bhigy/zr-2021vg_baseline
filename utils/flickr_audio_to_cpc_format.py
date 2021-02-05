import argparse
import glob
import os
import sys
from tqdm import tqdm


def load_wav2spkr(wav2spkr_file):
    wav2spkr = {}
    with open(wav2spkr_file) as fin:
        for line in fin:
            wav, spkr = line.replace('\n', '').split(' ')
            wav2spkr[wav] = spkr
    return wav2spkr


def main(argv):
    parser = argparse.ArgumentParser(description='This script convert Flickr8k audio files into the format'
                                                 'needed to train CPC. Example call : '
                                                 'python flickr_audio_to_cpc_format.py \
                                                    --flickr_audio /private/home/marvinlvn/DATA/MULTIMODAL/FLICKR8K/flickr_audio/wavs \
                                                    --flickr_wav2spkr /private/home/marvinlvn/DATA/MULTIMODAL/FLICKR8K/wav2spk.txt \
                                                    --output /private/home/marvinlvn/DATA/MULTIMODAL/FLICKR_CPC')
    parser.add_argument('--flickr_audio', type=str, required=True,
                        help='Path to the directory containing the flickr 8k audio files.')
    parser.add_argument('--flickr_wav2spkr', type=str, required=True,
                        help='Path to the file wav2spk.txt.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output folder.')
    args = parser.parse_args(argv)

    if not os.path.isdir(args.flickr_audio):
        raise ValueError("Can't found %s" % args.flickr_audio)

    if not os.path.isfile(args.flickr_wav2spkr):
        raise ValueError("Can't found %s" % args.flickr_wav2spkr)

    os.makedirs(args.output, exist_ok=True)

    audio_files = glob.glob(os.path.join(args.flickr_audio, "*.wav"))

    if len(audio_files) == 0:
        raise ValueError("Can't found any .wav files in %s" % audio_files)

    wav2spkr = load_wav2spkr(args.flickr_wav2spkr)

    assert(len(wav2spkr) == len(audio_files))

    for audio_file in tqdm(audio_files):
        bn = os.path.basename(audio_file)
        speaker = wav2spkr[bn]

        # Create directory whose name is the speaker id
        dirname = os.path.join(args.output, speaker)
        os.makedirs(dirname, exist_ok=True)

        dest_file = os.path.join(args.output, dirname, bn)
        os.symlink(audio_file, dest_file)


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)