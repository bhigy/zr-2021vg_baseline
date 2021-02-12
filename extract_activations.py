import argparse
import numpy as np
import os
from pathlib import Path
import sys
import torch
from tqdm import tqdm

from cpc.dataset import findAllSeqs

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
        '--file_extension', type=str, default="wav",
        help="Extension of the audio files in the dataset (default: wav).")
    parser.add_argument(
        '--layer', type=str, default='all',
        help='Name of the layer to extract (default: "all", all layers).')
    parser.add_argument(
        '--max_size_seq', type=int, default=64000,
        help='Maximal number of frames to consider in each chunk when '
             'extracting activations (defaut: 64000).')
    parser.add_argument(
        '--output_file_extension', type=str, default="txt",
        choices=['txt', 'npy', 'pt'],
        help="Extension of the audio files in the dataset (default: txt).")
    return parser.parse_args(argv)


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
                              speaker_level=1,
                              extension=args.file_extension,
                              loadCache=True)
    if len(seqNames) == 0 or not os.path.splitext(seqNames[0][-1])[1].endswith(args.file_extension):
        print("Seems like the _seq_cache.txt does not contain the correct extension, reload the file list")
        seqNames, _ = findAllSeqs(args.pathDB,
                                  speaker_level=1,
                                  extension=args.file_extension,
                                  loadCache=False)
    print(f"Done! Found {len(seqNames)} files!")

    # Verify the output directory
    pathOutputDir = Path(args.pathOutputDir)
    if pathOutputDir.exists():
        pass
    #     existing_files = set([x.stem for x in pathOutputDir.glob('*.txt')])
    #     seqNames = [s for s in seqNames if Path(s[1]).stem not in existing_files]
    #     print(f'Found existing output directory at {args.pathOutputDir}, '
    #           f'continue to build features of {len(seqNames)} audio files left!')
    else:
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
    cache_fpath = pathDB / '_mfcc_features.pt'
    if cache_fpath.exists():
        print(f"Found cached features ({cache_fpath}). Loading them.")
        features = torch.load(cache_fpath)
    else:
        print('No cached features. Computing them from scratch.')
        audio_fpaths = [pathDB / s[1] for s in seqNames]
        audio_config = _audio_feat_config
        audio_config['max_size_seq'] = args.max_size_seq
        features = audio_features(audio_fpaths, audio_config)
        print(f'Caching features ({cache_fpath}).')
        torch.save(features, cache_fpath)

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
    batch_fn = lambda x: dataset.batch_audio(x, max_frames=None)
    data = torch.utils.data.DataLoader(dataset=features,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       collate_fn=batch_fn)
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
        fpath = (output_dir / fnames[i]).with_suffix(f'.{output_format}')
        if output_format == 'txt':
            act = act.detach().cpu().numpy()
            np.savetxt(fpath, act)
        elif output_format == 'npy':
            act = act.detach().cpu().numpy()
            np.save(fpath, act)
        elif output_format == 'pt':
            act = act.detach().cpu()
            torch.save(act, fpath)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
