# Based on https://github.com/bootphon/zerospeech2021_baseline/blob/master/scripts/build_CPC_features.py
import argparse
import numpy as np
import os
from pathlib import Path
import sys
import torch
from tqdm import tqdm

from cpc.dataset import findAllSeqs, filterSeqs

import platalea.dataset as dataset
from platalea.utils.preprocessing import _audio_feat_config, audio_features

from utils_functions import writeArgs


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(
        description='Extract activations from a VG model.')
    parser.add_argument(
        'pathCheckpoint', type=str, help='Path to the VG model checkpoint.')
    parser.add_argument(
        'pathDB', type=str,
        help='Path to the dataset that we want to quantize.')
    parser.add_argument(
        'pathOutputDir', type=str, help='Path to the output directory.')
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size')
    parser.add_argument(
        '--debug', action='store_true',
        help='Load only a very small amount of files for debugging purposes.')
    parser.add_argument(
        '--file_extension', type=str, default=".wav",
        help="Extension of the audio files in the dataset (default: .wav).")
    parser.add_argument(
        '--layer', type=str, default='all',
        help='Name of the layer to extract (default: "all", all layers).')
    parser.add_argument(
        '--max_size_seq', type=int, default=64000,
        help='Maximal number of frames to consider in each chunk when '
             'extracting activations (defaut: 64000).')
    parser.add_argument(
        '--output_file_extension', type=str, default=".txt",
        choices=['.txt', '.npy', '.pt'],
        help="Extension of the audio files in the dataset (default: .txt).")
    parser.add_argument(
        '--recursionLevel', type=int, default=2,
        help="The speaker recursionLevel in the training dataset (default: 2).")
    parser.add_argument(
        '--seqList', type=str, default=None,
        help="Specific the training sequence list (default: None).")
    return parser.parse_args(argv)


def compute_audio_features(audio_fpaths, max_size_seq):
    audio_config = _audio_feat_config
    audio_config['max_size_seq'] = max_size_seq
    return audio_features(audio_fpaths, audio_config)


def main(argv):
    # Args parser
    args = parseArgs(argv)

    print("=============================================================")
    print(f"Extract activations from VG model for {args.pathDB}")
    print("=============================================================")

    # Find all sequences
    print("")
    print(f"Looking for all {args.file_extension} files in {args.pathDB}")
    seqNames, _ = findAllSeqs(args.pathDB,
                              speaker_level=args.recursionLevel,
                              extension=args.file_extension,
                              loadCache=True)
    if len(seqNames) == 0 or not os.path.splitext(seqNames[0][-1])[1].endswith(args.file_extension):
        print("Seems like the _seq_cache.txt does not contain the correct extension, reload the file list")
        seqNames, _ = findAllSeqs(args.pathDB,
                                  speaker_level=args.recursionLevel,
                                  extension=args.file_extension,
                                  loadCache=False)
    print(f"Done! Found {len(seqNames)} files!")
    if args.seqList is not None:
        seqNames = filterSeqs(args.seqList, seqNames)
        print(f"Done! {len(seqNames)} remaining files after filtering!")
    assert len(seqNames) > 0

    pathOutputDir = Path(args.pathOutputDir)
    print("")
    print(f"Creating the output directory at {args.pathOutputDir}")
    pathOutputDir.mkdir(parents=True, exist_ok=True)
    writeArgs(pathOutputDir / "_info_args.json", args)

    # Debug mode
    if args.debug:
        nsamples = 20
        print("")
        print(f"Debug mode activated, only load {nsamples} samples!")
        # shuffle(seqNames)
        seqNames = seqNames[:nsamples]

    # Loading audio features
    print("")
    print(f"Loading audio features for {args.pathDB}")
    pathDB = Path(args.pathDB)
    if args.seqList is None:
        cache_fpath = pathDB / '_mfcc_features.pt'
        if cache_fpath.exists():
            print(f"Found cached features ({cache_fpath}). Loading them.")
            features = torch.load(cache_fpath)
        else:
            print('No cached features. Computing them from scratch.')
            audio_fpaths = [pathDB / s[1] for s in seqNames]
            features = compute_audio_features(audio_fpaths, args.max_size_seq)
            print(f'Caching features ({cache_fpath}).')
            torch.save(features, cache_fpath)
    else:
        print('Computing features.')
        audio_fpaths = [pathDB / s[1] for s in seqNames]
        features = compute_audio_features(audio_fpaths, args.max_size_seq)

    # Load VG model
    print("")
    print(f"Loading VG model from {args.pathCheckpoint}")
    vg_model = torch.load(args.pathCheckpoint)
    print("VG model loaded!")

    # Extracting activations
    # TODO:
    #    * check if extraction needs to set eval mode to reduce memory consumption
    print("")
    print(f"Extracting activations and saving outputs to {args.pathOutputDir}...")
    data = torch.utils.data.DataLoader(dataset=features,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       collate_fn=lambda x: dataset.batch_audio(x, max_frames=None))
    i_next = 0
    for au, l in tqdm(data):
        activations = vg_model.SpeechEncoder.introspect(au.cuda(), l.cuda())
        fnames = [s[1] for s in seqNames[i_next: i_next + args.batch_size]]
        if args.layer == 'all':
            for k in activations:
                save_activations(activations[k], pathOutputDir / k, fnames,
                                 args.output_file_extension)
        elif args.layer in activations:
            save_activations(activations[args.layer],
                             pathOutputDir / args.layer, fnames,
                             args.output_file_extension)
        i_next += args.batch_size


def save_activations(activations, output_dir, fnames, output_format):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    for i, act in enumerate(activations):
        fpath = (output_dir / fnames[i]).with_suffix(f'{output_format}')
        fpath.parent.mkdir(parents=True, exist_ok=True)
        if output_format == '.txt':
            act = act.detach().cpu().numpy()
            np.savetxt(fpath, act)
        elif output_format == '.npy':
            act = act.detach().cpu().numpy()
            np.save(fpath, act)
        elif output_format == '.pt':
            act = act.detach().cpu()
            torch.save(act, fpath)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
