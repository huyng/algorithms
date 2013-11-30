def bench_convolve_with_patches():
    pass

def gen_4d_image():
    # mat = np.arange(n_rows*n_cols*n_channels, dtype=np.float32).reshape(n_rows, n_cols, n_channels)
    # mat = np.zeros((n_rows, n_cols, n_channels), dtype=np.float32)
    # for i in range(n_channels):
    #     mat[:,:,i] = i
    pass

def gen_4d_lena():
    from scipy.misc import lena
    # from pylab import *
    im = np.atleast_3d(lena())
    im = im.astype(np.float32)
    # filters = np.eye(3).ravel()
    # c = convolve_with_patches(im, filters)
    # print c.shape
    # imshow(c, cmap=cm.Greys_r)


if __name__ == '__main__':
    bench_convolve_with_patches()
