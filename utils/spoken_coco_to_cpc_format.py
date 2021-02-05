import argparse
import glob
import os
import sys
from tqdm import tqdm


def main(argv):
    parser = argparse.ArgumentParser(description='This script convert SpokenCoco audio files into the format'
                                                 'needed to train CPC. Example call : '
                                                 'python spoken_coco_to_cpc_format.py \
                                                    --audio /private/home/marvinlvn/DATA/MULTIMODAL/SpeechCOCO/SpokenCOCO/wavs \
                                                    --output /private/home/marvinlvn/DATA/MULTIMODAL/SpokenCOCO_CPC')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to the directory containing the flickr 8k audio files.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output folder.')
    args = parser.parse_args(argv)

    if not os.path.isdir(args.audio):
        raise ValueError("Can't found %s" % args.audio)

    os.makedirs(args.output, exist_ok=True)

    audio_files = glob.iglob(os.path.join(args.audio, "**/*.wav"), recursive=True)

    for audio_file in tqdm(audio_files):
        bn = os.path.basename(audio_file)

        speaker = bn.split('-')[0]

        # Create directory whose name is the speaker id
        dirname = os.path.join(args.output, speaker)
        os.makedirs(dirname, exist_ok=True)

        dest_file = os.path.join(args.output, dirname, bn)
        os.symlink(audio_file, dest_file)


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)