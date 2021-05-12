# Based on https://github.com/bootphon/zerospeech2021_baseline/blob/master/scripts/quantize_audio.py
import os
import sys
import json
import argparse
import progressbar
from pathlib import Path
from time import time

import torch
from cpc.dataset import filterSeqs, findAllSeqs

from scripts.utils.utils_functions import readArgs, writeArgs, loadClusterModule
from scripts.data import loadFile


def quantize_file(file_path, clusterModule, cpu=False):
    file_path = Path(file_path)
    activations = loadFile(file_path)
    if file_path.suffix != '.pt':
        activations = torch.tensor(activations)
    if not cpu:
        activations = activations.cuda()

    nGroups = activations.size(-1)//clusterModule.Ck.size(-1)  # groups information

    # Quantize the output of clustering on the CPC features
    activations = activations.view(1, -1, clusterModule.Ck.size(-1))
    if activations.size(1) > 50000:  # Librilight, to avoid GPU OOM, decrease when still OOM
        clusterModule = clusterModule.cpu()
        activations = activations.cpu()
        quantized = torch.argmin(clusterModule(activations), dim=-1)
        if not cpu:
            clusterModule = clusterModule.cuda()
    else:
        quantized = torch.argmin(clusterModule(activations), dim=-1)
    quantized = quantized[0].detach().cpu().numpy()

    # Transform to quantized line
    quantLine = ",".join(["-".join([str(i) for i in item]) for item in quantized.reshape(-1, nGroups)])

    return quantLine


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(
        description='Quantize audio files using CPC Clustering Module.')
    parser.add_argument(
        'pathClusteringCheckpoint', type=str,
        help='Path to the clustering checkpoint.')
    parser.add_argument(
        'pathActivations', type=str,
        help='Path to the activations that we want to quantize.')
    parser.add_argument(
        'pathOutputDir', type=str, help='Path to the output directory.')
    parser.add_argument(
        '--cpu', action='store_true', help="Run on a cpu machine.")
    parser.add_argument(
        '--debug', action='store_true',
        help="Load only a very small amount of files for debugging purposes.")
    parser.add_argument(
        '--file_extension', type=str, default=".pt",
        choices=['.npy', '.pt', '.txt'],
        help="Extension of the activation files in the dataset (default: .pt).")
    parser.add_argument(
        '--recursionLevel', type=int, default=2,
        help="The speaker recursionLevel in the training dataset (default: 2).")
    parser.add_argument(
        '--resume', action='store_true',
        help="Continue to quantize if an output file already exists.")
    parser.add_argument(
        '--seqList', type=str, default=None,
        help="Specify a txt file containing the list of sequences (file names)"
        'to be included (default: None). '
        'If not speficied, include all files found in pathActivations.')
    parser.add_argument(
        '--split', type=str, default=None,
        help='If you want to divide the dataset in small splits, specify it '
        'with idxSplit-numSplits (idxSplit > 0), eg. --split 1-20.')
    return parser.parse_args(argv)


def main(pathClusteringCheckpoint, pathActivations, pathOutputDir,
         cpu=False, debug=False, file_extension=".pt", recursionLevel=2,
         resume=False, seqList=None, split=None):
    # Test the extension is valid
    if file_extension not in ['.txt', '.npy', '.pt']:
        raise ValueError(f'Activation file extension invalid ({file_extension})')

    args = argparse.Namespace(**locals())
    print("=============================================================")
    print(f"Quantizing data from {pathActivations}")
    print("=============================================================")

    # Get splits
    if split:
        assert len(split.split("-")) == 2 \
           and int(split.split("-")[1]) >= int(split.split("-")[0]) >= 1, \
               "SPLIT must be under the form idxSplit-numSplits (numSplits >= idxSplit >= 1), eg. --split 1-20"
        idx_split, num_splits = split.split("-")
        idx_split = int(idx_split)
        num_splits = int(num_splits)

    # Find all sequences
    print("")
    print(f"Looking for all {file_extension} files in {pathActivations}")
    seqNames, _ = findAllSeqs(pathActivations,
                              speaker_level=recursionLevel,
                              extension=file_extension,
                              loadCache=True)
    if len(seqNames) == 0 or not os.path.splitext(seqNames[0][1])[1].endswith(file_extension):
        print("Seems like the _seq_cache.txt does not contain the correct extension, reload the file list")
        seqNames, _ = findAllSeqs(pathActivations,
                                  speaker_level=recursionLevel,
                                  extension=file_extension,
                                  loadCache=False)
    print(f"Done! Found {len(seqNames)} files!")

    # Filter specific sequences
    if seqList is not None:
        seqNames = filterSeqs(seqList, seqNames)
        print(f"Done! {len(seqNames)} remaining files after filtering!")
    assert len(seqNames) > 0, \
        "No file to be quantized!"

    # Check if directory exists
    pathOutputDir = Path(pathOutputDir)
    if not pathOutputDir.exists():
        print("")
        print(f"Creating the output directory at {pathOutputDir}")
        pathOutputDir.mkdir(parents=True, exist_ok=True)
    writeArgs(pathOutputDir / "_info_args.json", args)

    # Check if output file exists
    if not split:
        nameOutput = "quantized_outputs.txt"
    else:
        nameOutput = f"quantized_outputs_split_{idx_split}-{num_splits}.txt"
    outputFile = pathOutputDir / nameOutput

    # Get splits
    if split:
        startIdx = len(seqNames) // num_splits * (idx_split-1)
        if idx_split == num_splits:
            endIdx = len(seqNames)
        else:
            endIdx = min(len(seqNames) // num_splits * idx_split, len(seqNames))
        seqNames = seqNames[startIdx:endIdx]
        print("")
        print(f"Quantizing split {idx_split} out of {num_splits} splits, "
              f"with {len(seqNames)} files (idx in range({startIdx}, {endIdx})).")

    # Debug mode
    if debug:
        nsamples = 20
        print("")
        print(f"Debug mode activated, only load {nsamples} samples!")
        # shuffle(seqNames)
        seqNames = seqNames[:nsamples]

    # Continue
    addEndLine = False  # to add end line (\n) to first line or not
    if resume:
        if outputFile.exists():
            with open(outputFile, 'r') as f:
                lines = [line for line in f]
            existing_files = set([x.split()[0] for x in lines if x.split()])
            seqNames = [s for s in seqNames if os.path.splitext(s[1].split('/')[-1])[0] not in existing_files]
            print(f"Found existing output file, continue to quantize {len(seqNames)} audio files left!")
            if len(lines) > 0 and not lines[-1].endswith("\n"):
                addEndLine = True
    else:
        print(outputFile, outputFile.exists())
        assert not outputFile.exists(), \
            f"Output file {outputFile} already exists !!! " \
            f"If you want to continue quantizing audio files, please check the --resume option."

    assert len(seqNames) > 0, \
        "No file to be quantized!"

    # Load Clustering args
    pathCheckpoint = Path(pathClusteringCheckpoint)
    assert pathCheckpoint.suffix == ".pt"
    if Path(str(pathCheckpoint.with_suffix('')) + '_args.json').exists():
        pathConfig = Path(str(pathCheckpoint.with_suffix('')) + '_args.json')
    elif (pathCheckpoint.parent / "checkpoint_args.json").exists():
        pathConfig = pathCheckpoint.parent / "checkpoint_args.json"
    else:
        assert False, \
            f"Args file not found in the directory {pathCheckpoint.parent}"
    clustering_args = readArgs(pathConfig)
    print("")
    print(f"Clutering args:\n{json.dumps(vars(clustering_args), indent=4, sort_keys=True)}")
    print('-' * 50)

    # Load CluterModule
    print("")
    print(f"Loading ClusterModule at {pathCheckpoint}")
    clusterModule = loadClusterModule(pathCheckpoint)
    if not cpu:
        clusterModule.cuda()
    print("ClusterModule loaded!")

    # Quantization of files
    print("")
    print(f"Quantizing activation files and saving outputs to {outputFile}...")
    f = open(outputFile, "a")
    bar = progressbar.ProgressBar(maxval=len(seqNames))
    bar.start()
    start_time = time()
    for index, vals in enumerate(seqNames):
        bar.update(index)

        file_path = vals[1]
        file_path = os.path.join(pathActivations, file_path)

        # Quantizing
        quantLine = quantize_file(file_path, clusterModule, cpu=cpu)

        # Save the outputs
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        outLine = "\t".join([file_name, quantLine])
        if addEndLine:
            f.write("\n"+outLine)
        else:
            f.write(outLine)
            addEndLine = True
    bar.finish()
    print(f"...done {len(seqNames)} files in {time()-start_time} seconds.")
    f.close()


if __name__ == "__main__":
    args = parseArgs(sys.argv[1:])
    main(**vars(args))
