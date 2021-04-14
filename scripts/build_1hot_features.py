# Based on https://github.com/bootphon/zerospeech2021_baseline/blob/master/scripts/build_1hot_features.py
import argparse
import numpy as np
import os
import progressbar
from pathlib import Path
import sys
from time import time
import torch

from utils.utils_functions import writeArgs


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(
        description='Export 1-hot features from quantized units of audio files.')
    parser.add_argument(
        'pathQuantizedUnits', type=str,
        help='Path to the quantized units. Each line of the input file must be'
        'of the form file_name[tab]pseudo_units (ex. hat  1,1,2,3,4,4)')
    parser.add_argument(
        'pathOutputDir', type=str,
        help='Path to the output directory.')
    parser.add_argument(
        '--n_units', type=int, default=50,
        help='Number of discrete units (default: 50). If a dictionary is given,'
        'this is automatically set as vocab size.')
    parser.add_argument(
        '--dict', type=str,
        help='Path to the dictionary file containing vocab of the pseudo units on the dataset'
        '(this is required if the quantized units are not digits, i.e. multi-group case).')
    parser.add_argument(
        '--debug', action='store_true',
        help="Load only a very small amount of files for debugging purposes.")
    parser.add_argument(
        '--output_file_extension', type=str, default=".txt",
        choices=['.txt', '.npy', '.pt'],
        help="Extension of the audio files in the dataset (default: .txt).")
    return parser.parse_args(argv)


def main(argv):
    # Args parser
    args = parseArgs(argv)

    print("=============================================================")
    print(f"Building 1-hot features from {args.pathQuantizedUnits}")
    print("=============================================================")

    # Load input file
    print("")
    print(f"Reading input file from {args.pathQuantizedUnits}")
    seqNames = []
    seqInputs = []
    with open(args.pathQuantizedUnits, 'r') as f:
        for line in f:
            file_name, file_seq = line.strip().split("\t")
            # Convert sequence to the desired input form
            file_seq = file_seq.replace(",", " ")
            # Add to lists
            seqNames.append(file_name)
            seqInputs.append(file_seq)
    print(f"Found {len(seqNames)} sequences!")

    # Verify the output directory
    pathOutputDir = Path(args.pathOutputDir)
    if pathOutputDir.exists():
        existing_files = set([x.stem for x in pathOutputDir.iterdir() if x.suffix == ".npy"])
        seqNames = [s for s in seqNames if Path(s[1]).stem not in existing_files]
        print(f"Found existing output directory at {pathOutputDir}, "
              f"continue to build features of {len(seqNames)} audio files left!")
    else:
        print("")
        print(f"Creating the output directory at {pathOutputDir}")
        pathOutputDir.mkdir(parents=True, exist_ok=True)
    writeArgs(pathOutputDir / "_info_args.json", args)

    # Debug mode
    if args.debug:
        nsamples = 20
        print("")
        print(f"Debug mode activated, only load {nsamples} samples!")
        # shuffle(seqNames)
        seqNames = seqNames[:nsamples]
        seqInputs = seqInputs[:nsamples]

    # Load 1hot dictionary in case we use it
    if seqInputs and not seqInputs[0].split()[0].isdigit():  # multi-group ie. 65-241
        assert args.dict is not None, \
            "A dictionary must be given when the quantized outputs is not digits (multi-group case)!"
    if args.dict:
        print("")
        print(f"Loading onehot dictionary from {args.dict}...")
        with open(args.dict, "r") as f:
            lines = f.read().split("\n")
        pair2idx = {word.split()[0]: i for i, word in enumerate(lines) if word and not word.startwith("madeupword")}
        args.n_units = len(pair2idx)

    # Define onehot_feature_function
    def onehot_feature_function(input_sequence):
        if args.dict:
            indexes_sequence = np.array([pair2idx[item] for item in input_sequence.split()])
        else:
            indexes_sequence = np.array([int(item) for item in input_sequence.split()])

        onehotFeatures = np.eye(args.n_units)[indexes_sequence]

        return onehotFeatures

    # Building features
    print("")
    print(f"Building 1-hot features and saving outputs to {pathOutputDir}...")
    bar = progressbar.ProgressBar(maxval=len(seqNames))
    bar.start()
    start_time = time()
    for index, (name_seq, input_seq) in enumerate(zip(seqNames, seqInputs)):
        bar.update(index)

        # Computing features
        onehot_features = onehot_feature_function(input_seq)

        # Save the outputs
        file_name = os.path.splitext(name_seq)[0] + args.output_file_extension
        file_out = pathOutputDir / file_name
        save(file_out, onehot_features)
    bar.finish()
    print(f"...done {len(seqNames)} files in {time()-start_time} seconds.")


def save(fpath_out, data):
    if fpath_out.suffix == '.txt':
        np.savetxt(fpath_out, data)
    elif fpath_out.suffix == '.npy':
        np.save(fpath_out, data)
    elif fpath_out.suffix == '.pt':
        torch.save(torch.tensor(data), fpath_out)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
