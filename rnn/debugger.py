import pylab as p
import theano
import numpy as np

def show_weights(*args):
    for i, W in enumerate(args):
        title = str("argument %i" % i)
        if isinstance(W, theano.tensor.sharedvar.TensorSharedVariable):
            title = W.name
            W = W.get_value(borrow=False)

        W = np.atleast_2d(W)
        p.figure()
        p.title(title)
        p.imshow(W, cmap=p.cm.gray_r, interpolation='nearest')

