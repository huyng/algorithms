# from matplotlib.pyplot import draw, figure, show
import pylab as p
import numpy as np
import time
import argparse as ap
import json
from collections import defaultdict

parser = ap.ArgumentParser()
parser.add_argument("file", type=str)
parser.add_argument("--port", default=8080, type=int)
parser.add_argument("--host", default="0.0.0.0", type=str)
parser.add_argument("--debug", default=False, type=bool)



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
    from StringIO import StringIO
    data = defaultdict(list)
    for d in fsource(args.file):
        p.clf()
        for key, val in d.items():
            if key.startswith("_"):
                continue
            data[key].append(val)

    for key in d.keys():
        p.plot(data[key], label=key)

    p.xlabel("epoch")
    p.ylabel("cost")
    p.legend()
    buf = StringIO()
    p.savefig(buf, format='png')
    buf.seek(0)
    return buf.read()



    


from flask import Flask
from flask import make_response
app = Flask(__name__)
@app.route('/')
def index():
    image = run(args)
    response = make_response(image)
    response.headers['Content-Type'] = 'image/jpeg'
    # response.headers['Content-Disposition'] = 'attachment; filename=img.jpg'
    return response


def main(args):
    app.run(debug=True, host=args.host, port=args.port)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)