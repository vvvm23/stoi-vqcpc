#!/usr/bin/env python

import numpy as np
from scipy.stats import spearmanr

import argparse
import csv
import datetime
from pathlib import Path

def get_uid(path):
    path = Path(path)
    ext = "".join(path.suffixes)
    return str(path.name).removesuffix(ext)

def main(args):
    assert not args.names or len(args.predictions) == len(args.names), "must specify zero or matching number of names"
    args.names = args.names if args.names else args.predictions

    print("> processing ground truth")
    with open(args.ground, mode='r') as f:
        csv_reader = csv.reader(f)
        ground = [(get_uid(n), float(s)) for n, *_, s in csv_reader]
    ground.sort()

    predictions = []
    for p, n in zip(args.predictions, args.names):
        print(f"> processing prediction '{n}'")
        with open(p, mode='r') as f:
            csv_reader = csv.reader(f)
            predictions.append([(get_uid(n), float(s)) for n, *_, s in csv_reader])
            predictions[-1].sort()

    for g, *p in zip(ground, *predictions):
        assert all([g[0] == x[0] for x in p]), "names did not match"

    ground_array = np.array([g[1] for g in ground])
    prediction_arrays = [np.array([x[1] for x in p]) for p in predictions]

    mses = [((p - ground_array)**2).mean() for p in prediction_arrays]
    lccs = [np.corrcoef(ground_array, p)[0,1] for p in prediction_arrays]
    srccs = [spearmanr(ground_array.T, p.T)[0] for p in prediction_arrays]

    best_mse = np.argmin(mses)
    best_lcc = np.argmax(lccs)
    best_srcc = np.argmax(srccs)

    names = args.names if args.names else args.predictions
    for i, p in enumerate(names):
        msg = (
            f"> '{p}' results:\n"
            f"\tMSE: \t{mses[i]} \t{'BEST' if i == best_mse else ''}\n"
            f"\tLCC: \t{lccs[i]} \t{'BEST' if i == best_lcc else ''}\n"
            f"\tSRCC: \t{srccs[i]} \t{'BEST' if i == best_srcc else ''}\n"
        )
        print(msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ground', type=str)
    parser.add_argument('predictions', nargs='+', type=str)
    parser.add_argument('--names', nargs='+', type=str)
    args = parser.parse_args()

    main(args)
