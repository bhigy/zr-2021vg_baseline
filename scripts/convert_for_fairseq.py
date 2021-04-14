import argparse
import sys


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(
        description='Convert quantized activations to fairseq format to train '
        'the language model.')
    parser.add_argument(
        'pathQuantizedActivations', type=str,
        help='Path to the file containing the quantized activations.')
    parser.add_argument(
        'pathOutput', type=str, help='Path to the output file.')
    parser.add_argument(
        '--seqList', type=str, default=None,
        help="Specify a txt file containing the list of sequences (file names)"
        'to be included (default: None). '
        'If not speficied, include all files found in pathQuantizedActivations.')
    return parser.parse_args(argv)


def main(argv):
    # Args parser
    args = parseArgs(argv)

    # Load quantized activations
    data = {}
    for line in open(args.pathQuantizedActivations).readlines():
        split = line.split()
        data[split[0]] = split[1].split(',')

    # Filter data
    if args.seqList is not None:
        seqList = [line.rstrip('\n') for line in open(args.seqList).readlines()]
        data_filtered = {k: data[k] for k in seqList}
        data = data_filtered

    with open(args.pathOutput, 'w') as f:
        f.writelines([(' ').join(d) + '\n' for d in data.values()])


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
