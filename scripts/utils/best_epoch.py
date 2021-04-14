import argparse
import json
import os
import sys


def main(argv):
    parser = argparse.ArgumentParser(description='This script takes as input the directory to the trained VG model '
                                                 'and returns the best epoch, selected on the r@10 metric.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the directory containing the .json log files')
    args = parser.parse_args(argv)

    if not os.path.isdir(args.model_path):
        raise ValueError("Can't find %s" % args.model_path)

    r10 = {}
    with open(os.path.join(args.model_path, 'result.json')) as fin:
        for line in fin:
            line = json.loads(line)
            r10[line['epoch']] = line['recall']['10']
    best_ep = max(r10, key=r10.get)
    print("With a recall@10 of %.2f, best epoch is : %d" % (r10[best_ep], best_ep))

if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)
