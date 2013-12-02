import time
import numpy as np

def bench_convolve_with_patches():
    import conv
    import pylab as P

    images = gen_lena()
    kernel = gen_filter()

    for i in range(100):
        t0 = time.time()
        c = conv.convolve3d_with_patches(images, kernel, pad=0)

        # P.imshow(c.astype(np.uint32).squeeze(), cmap=P.cm.Greys_r)
        # P.pause(10)

        print "took :: %.5f secs to compute featmap %s"  % (time.time() - t0, c.shape) 


def gen_filter():
    """
    Generate a list of 10 filters with 2-channels each and with 
    a shape of 3 rows and 3 cols.

    Output shape (3, 3, 2, 10)
    """

    c1 = np.eye(3, dtype=np.float32)
    c2 = np.ones_like(c1)

    f = np.dstack([c1, c2])
    f = np.expand_dims(f, axis=3)
    # print f.shape
    filters = np.concatenate([f]*10, axis=3)
    # print filters.shape
    return filters



def gen_lena():
    """
    Generate a (512, 512, 2) matrix where first layer is
    lena image. And the second layer is an array of 0s
    """
    from scipy.misc import lena 
    lena = np.atleast_3d(lena()).astype(np.float32)
    zeros = np.zeros_like(lena)
    feats = np.dstack([lena, zeros])
    return feats
    


if __name__ == '__main__':
    bench_convolve_with_patches()
