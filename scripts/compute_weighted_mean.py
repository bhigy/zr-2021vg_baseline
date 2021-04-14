#!/usr/bin/env python3

"""
Compute the weighted mean mean score.

Compute the weighted mean score from a CSV file containing 'score' and 'n'
columns. Can be used to compute mean sWUGGY and sBLIMP scores.
"""

import argparse
import numpy as np
import pandas as pd

def main(csv_files):
    wmeans = np.zeros(len(csv_files))
    for i, fpath in enumerate(csv_files):
        data = pd.read_csv(fpath)
        wmeans[i] = (data['score'] * data['n']).sum() / data['n'].sum()
    return wmeans


if __name__ == '__main__':
    # Parsing command line
    doc = __doc__.strip("\n").split("\n", 1)
    parser = argparse.ArgumentParser(
        description=doc[0], epilog=doc[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv_files', help='CSV file containing the individual scores', nargs='+')
    args = parser.parse_args()

    wmeans = main(args.csv_files)
    print(wmeans)
