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
        while True:
            line = fh.readline()
            if not line.strip():
                continue
            try:
                data_dict = json.loads(line)
            except ValueError:
                pass
            yield data_dict

def main(args):
    data = defaultdict(list)
    for key, val in fsource(args.file):
        data[key].append(val)
        p.plot(data[key])
        p.show(block=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)