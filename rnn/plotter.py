# from matplotlib.pyplot import draw, figure, show
import pylab as p
import numpy as np
import time
import argparse as ap
import json
from collections import defaultdict

parser = ap.ArgumentParser()
parser.add_argument("file", type=str)


def fsource(fpath):
    with open(fpath) as fh:

        for line in fh:
            if not line.strip():
                continue
            try:
                data_dict = json.loads(line)
            except ValueError:
                pass
            yield data_dict


def run(args):
    data = defaultdict(list)
    for d in fsource(args.file):
        p.clf()
        for key, val in d.items():
            if key.startswith("_"):
                continue
            data[key].append(val)

    for key in d.keys():
        p.plot(data[key], label=key)

    p.legend()
    p.show(block=True)
    raw_input("press enter")

def main(args):
    while True:
        run(args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)