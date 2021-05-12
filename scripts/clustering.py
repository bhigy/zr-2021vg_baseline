# Based on https://github.com/facebookresearch/CPC_audio/blob/zerospeech/cpc/criterion/clustering/clustering_script.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Originally released under the MIT license.
import argparse
from cpc.criterion.clustering import kMeanCluster, kMeanGPU
from cpc.dataset import findAllSeqs, filterSeqs
import json
import os
from pathlib import Path
from random import shuffle
import sys
import time
import torch

from scripts.data import SequentialData


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Clustering module using kmeans or dpmeans.')
    parser.add_argument(
        'pathActivations', type=str,
        help="Path to the activations to cluster.")
    parser.add_argument(
        'pathOutput', type=str,
        help="Path to the output clustering checkpoint.")
    parser.add_argument(
        '-g',  '--nGroups', type=int, default=1,
        help="Number of groups for kmeans algorithm (default: 1).")
    parser.add_argument(
        '-k', '--nClusters', type=int, default=50,
        help="Number of clusters for kmeans algorithm (default: 50).")
    parser.add_argument(
        '-n', '--MAX_ITER', type=int, default=100,
        help="Number of iterations (default: 150).")
    parser.add_argument(
        '--batchSizeGPU', type=int, default=50,
        help='Batch size of each GPU (default: 50).')
    parser.add_argument(
        '--debug', action='store_true',
        help='Debug mode, only use a small number of training data.')
    parser.add_argument(
        '--extension', type=str, default='.pt', choices=['.txt', '.npy', '.pt'],
        help="The activation file extension (default: .pt).")
    parser.add_argument(
        '--getDistanceEstimation', action='store_true',
        help='Get distance estimation')
    parser.add_argument(
        '--load', action='store_true',
        help='Load the last checkpoint from the same directory as the output.')
    parser.add_argument(
        '--perIterSize', type=int, default=-1,
        help='(Depreciated) Number of items per iteration (default: -1).')
    parser.add_argument(
        '--recursionLevel', type=int, default=2,
        help="The speaker recursionLevel in the training dataset (default: 2).")
    parser.add_argument(
        '--save', action='store_true',
        help='Save the intermediate checkpoints. The checkpoints will'
        'be saved in the same directory as the output.')
    parser.add_argument(
        '--save-last', type=int, default=5,
        help='Number of last checkpoints to be saved (default: 5).')
    parser.add_argument(
        '--seqList', type=str, default=None,
        help="Specific the training sequence list (default: None).")
    return parser.parse_args(argv)


def main(pathActivations, pathOutput, nGroups=1, nClusters=50, MAX_ITER=100,
         batchSizeGPU=50, debug=False, extension='.pt', getDistanceEstimation=False,
         load=False, perIterSize=-1, recursionLevel=2, save=False, save_last=5,
         seqList=None):
    # Test the extension is valid
    if extension not in ['.txt', '.npy', '.pt']:
        raise ValueError(f'Activation file extension invalid ({extension})')

    torch.cuda.empty_cache()

    args = argparse.Namespace(**locals())
    # Export absolute paths for later use
    pathActivations = os.path.abspath(pathActivations)
    pathOutput = os.path.abspath(pathOutput)

    if not load:
        assert os.path.exists(pathOutput) is False, \
            f"The output file {pathOutput} already exists, please check the option --load !"
        assert os.path.exists(os.path.join(os.path.dirname(pathOutput), "checkpoint_last.pt")) is False, \
            "Found last_checkpoint.pt in the output directory, please check the option --load !"

    print(args)
    seqNames, speakers = findAllSeqs(pathActivations,
                                     speaker_level=recursionLevel,
                                     extension=extension,
                                     loadCache=True)

    if seqList is not None:
        seqNames = filterSeqs(seqList, seqNames)
    if debug:
        nsamples = 1000
        print(f"Debug mode activated, get only {nsamples} samples!")
        shuffle(seqNames)
        seqNames = seqNames[:nsamples]
    if getDistanceEstimation:
        shuffle(seqNames)
        seqNames = seqNames[:5000]

    print("")
    print(f'Loading activations at {pathActivations}')
    start_time = time.time()
    dataset = SequentialData(pathActivations, seqNames, None)
    print(f"Dataset loaded in {time.time()-start_time} seconds !")
    print("")

    nGPUs = torch.cuda.device_count()
    if nGPUs == 0:
        raise RuntimeError('No GPU found')
    batchSize = batchSizeGPU * nGPUs
    dataloader = dataset.getDataLoader(batchSize, numWorkers=0)
    print(f"Length of dataLoader: {len(dataloader)}")
    print("")

    # Check if dir exists
    if not os.path.exists(os.path.dirname(pathOutput)) and os.path.dirname(pathOutput):
        Path(os.path.dirname(pathOutput)).mkdir(parents=True, exist_ok=True)

    pathConfig = f"{os.path.splitext(pathOutput)[0]}_args.json"
    with open(pathConfig, 'w') as file:
        json.dump(vars(args), file, indent=2)

    out_state_dict = {}
    print("Starting the clustering...")
    start_time = time.time()
    # Using a dumb lambda function to skip feature extraction as we start from
    # the activations
    clusters = kMeanGPU(dataloader, lambda x: x, nClusters, nGroups,
                        perIterSize=perIterSize,
                        MAX_ITER=MAX_ITER,
                        save=save, load=load,
                        save_dir=os.path.dirname(pathOutput),
                        save_last=save_last,
                        ).cpu()

    print(f'Ran clustering '
          f'in {time.time() - start_time:.2f} seconds')

    clusterModule = kMeanCluster(clusters)
    out_state_dict["state_dict"] = clusterModule.state_dict()
    out_state_dict["n_clusters"] = nClusters
    out_state_dict['dim'] = clusters.size(2)
    torch.save(out_state_dict, pathOutput)
    with open(pathConfig, 'w') as file:
        json.dump(vars(args), file, indent=2)


if __name__ == "__main__":
    args = parseArgs(sys.argv[1:])
    main(**vars(args))
