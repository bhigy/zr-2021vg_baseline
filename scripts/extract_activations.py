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
from platalea.utils.preprocessing import audio_features

from scripts.utils.utils_functions import writeArgs


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(
        description='Extract activations from a VG model.')
    parser.add_argument(
        'pathCheckpoint', type=str, help='Path to the VG model checkpoint.')
    parser.add_argument(
        'pathDB', type=str,
        help='Path to the dataset that we want to process.')
    parser.add_argument(
        'pathOutputDir', type=str, help='Path to the output directory.')
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size used to compute activations (defaut: 8).')
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
        help="Extension of the audio files in the dataset (default: txt).")
    parser.add_argument(
        '--zr_format', action='store_true',
        help="Indicates if activations from zerospeech2021 must be extracted. In which case,"
             "the folder structure will be the same as the one used for the zerospeech 2021 submission.")
    parser.add_argument(
        '--audio_features_fn', type=str, default='mfcc_features.pt',
        help="Basename of the audio features file.")
    parser.add_argument(
        '--image_features_fn', type=str, default="resnet_features.pt",
        help="Basename of the audio features file.")
    parser.add_argument(
        '--cpc_model_path', type=str, default=None,
        help='path to a pretrained CPC model. If provided, will be used as a audio feature extractor.')
    parser.add_argument(
        '--cpc_gru_level', type=int, default=-1,
        help='The RNN layer that needs to be extracted. Default to -1, extracts the '
             'last RNN layer of the aggregator network. Ex : for CPC big, 1 will extract the first layer,'
             '2 will extract the second layer and so on.')
    parser.add_argument(
        '--recursionLevel', type=int, default=2,
        help="The speaker recursionLevel in the training dataset (default: 2).")
    parser.add_argument(
        '--seqList', type=str, default=None,
        help="Specify a txt file containing the list of sequences (file names)"
        'to be included (default: None). If not speficied, include all files found in pathActivations.')
    return parser.parse_args(argv)


def compute_audio_features(audio_fpaths, max_size_seq, audio_config):
    audio_config['max_size_seq'] = max_size_seq
    return audio_features(audio_fpaths, audio_config)


def main(pathCheckpoint, pathDB, pathOutputDir, batch_size=8, debug=False,
         file_extension='.wav', layer='all', max_size_seq=64000,
         output_file_extension='.txt', recursionLevel=2, seqList=None,
         audio_features_fn='mfcc_features.pt',
         image_features_fn='resnet_features.pt', cpc_model_path=None,
         cpc_gru_level=-1, zr_format=False):

    args = argparse.Namespace(**locals())
    print("=============================================================")
    print(f"Extract activations from VG model for {pathDB}")
    print("=============================================================")

    # Initializing feature extraction config
    # /!\ Code duplication with preprocessing.py
    # Should probably store the feature config on disk.
    _audio_feat_config = dict(type='mfcc', delta=True, alpha=0.97, n_filters=40,
                              window_size=0.025, frame_shift=0.010, audio_features_fn=audio_features_fn)
    _images_feat_config = dict(model='resnet', image_features_fn=image_features_fn)

    if cpc_model_path is not None:
        if audio_features_fn == 'mfcc_features.pt':
            audio_features_fn = 'cpc_features.pt'
        _audio_feat_config = dict(type='cpc', model_path=cpc_model_path, audio_features_fn=audio_features_fn,
                                  strict=False, seq_norm=False, max_size_seq=10240, gru_level=cpc_gru_level,
                                  on_gpu=True)

    # Find all sequences
    print("")
    print(f"Looking for all {file_extension} files in {pathDB}")
    seqNames, _ = findAllSeqs(pathDB,
                              speaker_level=recursionLevel,
                              extension=file_extension,
                              loadCache=True)
    if len(seqNames) == 0 or not os.path.splitext(seqNames[0][-1])[1].endswith(file_extension):
        print("Seems like the _seq_cache.txt does not contain the correct extension, reload the file list")
        seqNames, _ = findAllSeqs(pathDB,
                                  speaker_level=recursionLevel,
                                  extension=file_extension,
                                  loadCache=False)
    print(f"Done! Found {len(seqNames)} files!")

    # Filter specific sequences
    if seqList is not None:
        seqNames = filterSeqs(seqList, seqNames)
        print(f"Done! {len(seqNames)} remaining files after filtering!")
    assert len(seqNames) > 0, \
        "No file to be processed!"

    pathOutputDir = Path(pathOutputDir)
    print("")
    print(f"Creating the output directory at {pathOutputDir}")
    pathOutputDir.mkdir(parents=True, exist_ok=True)
    writeArgs(pathOutputDir / "_info_args.json", args)

    # Debug mode
    if debug:
        nsamples = 20
        print("")
        print(f"Debug mode activated, only load {nsamples} samples!")
        # shuffle(seqNames)
        seqNames = seqNames[:nsamples]

    # Loading audio features
    print("")
    print(f"Loading audio features for {pathDB}")
    pathDB = Path(pathDB)
    if seqList is None:
        cache_fpath = pathDB / args.audio_features_fn
        if cache_fpath.exists():
            print(f"Found cached features ({cache_fpath}). Loading them.")
            features = torch.load(cache_fpath)
        else:
            print('No cached features. Computing them from scratch.')
            audio_fpaths = [pathDB / s[1] for s in seqNames]
            features = compute_audio_features(audio_fpaths, max_size_seq, _audio_feat_config)
            print(f'Caching features ({cache_fpath}).')
            torch.save(features, cache_fpath)
    else:
        print('Computing features.')
        audio_fpaths = [pathDB / s[1] for s in seqNames]
        features = compute_audio_features(audio_fpaths, max_size_seq, _audio_feat_config)


    # Load VG model
    print("")
    print(f"Loading VG model from {pathCheckpoint}")
    vg_model = torch.load(pathCheckpoint)
    print("VG model loaded!")

    # Extracting activations
    print("")
    print(f"Extracting activations and saving outputs to {pathOutputDir}...")
    data = torch.utils.data.DataLoader(dataset=features,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       collate_fn=lambda x: dataset.batch_audio(x, max_frames=None))

    i_next = 0
    zr_keywords = ['phonetic', 'lexical', 'syntactic', 'semantic']
    if zr_format:
        splitted_path = str(pathDB).split('/')
        for keyword in zr_keywords:
            if keyword in splitted_path:
                keyword_idx = splitted_path.index(keyword)
                break
        suffix = '/'.join(splitted_path[keyword_idx:])
    else:
        suffix = ""

    for au, l in tqdm(data):
        activations = vg_model.SpeechEncoder.introspect(au.cuda(), l.cuda())
        fnames = [s[1] for s in seqNames[i_next: i_next + batch_size]]
        if layer == 'all':
            for k in activations:
                save_activations(activations[k], pathOutputDir / k / suffix, fnames,
                                 output_file_extension)
        elif layer in activations:
            save_activations(activations[layer],
                             pathOutputDir / layer / suffix, fnames,
                             output_file_extension)
        i_next += batch_size


def save_activations(activations, output_dir, fnames, output_format):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    for i, act in enumerate(activations):
        fpath = (output_dir / fnames[i]).with_suffix(f'{output_format}')
        fpath.parent.mkdir(parents=True, exist_ok=True)
        # hack to be able to use output of attention layer for sSIMI
        if len(act.shape) == 1:
            act = torch.cat((act[None, :], act[None, :]))
        if output_format == '.txt':
            act = act.detach().cpu().numpy()
            np.savetxt(fpath, act)
        elif output_format == '.npy':
            act = act.detach().cpu().numpy()
            np.save(fpath, act)
        elif output_format == '.pt':
            act = act.detach().cpu()
            torch.save(act, fpath)
        else:
            raise ValueError(f'Output format {output_format} not supported for activations')


if __name__ == "__main__":
    args = parseArgs(sys.argv[1:])
    main(**vars(args))
