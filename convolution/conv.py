import numpy as np
import ctypes as ct


_lib = np.ctypeslib.load_library("libconv.so", ".")
_lib.sliding_patches.restype = None
_lib.sliding_patches.argtypes = [
    ct.c_void_p,
    ct.c_void_p, 
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int
]

PAD_CACHE = None

def convolve3d_with_patches(feats, filters, stride=1, pad=None):
    """
    Computes convolution of features with filters. For every filter
    the convolution sums the product of filter weights and features 
    across channels.

    
    Parameters
    ----------

    feats: ndarray(dtype=np.float32)
        image as 3d array of shape (rows, cols, channels)

    filters: ndarray(dtype=np.float32)
        filters as 4d array of shape (p_size, p_size, channels, n_filters)

    stride: int
        This determines the step size to slide the convolution 
        window. You can use this to sub sample your convolution
        operation. Default 1

    pad: int or None
         Amount of padding to add to your images

    Returns
    -------

    featmap: ndarray
        The feature map corresponding to the convolution of filters
        with feats. The shape should be (rows, cols, n_filters)


    """

    assert feats.dtype == np.float32, "feats must be np.float32"
    assert filters.dtype == np.float32, "feats must be np.float32"
    
    pad = int(filters.shape[0]/2) if pad is None else pad
    padding = [(pad,pad), (pad,pad), (0,0)]
    if pad > 0:
        feats = np.pad(feats, padding, mode="constant") # pad with zeros
    
    n_rows = feats.shape[0]
    n_cols = feats.shape[1]
    n_channels = feats.shape[2]

    p_size = filters.shape[0]
    p_strd = 1
    p_rows = int((n_rows-p_size)/p_strd) + 1
    p_cols = int((n_cols-p_size)/p_strd) + 1
    p_dpth = p_size*p_size*n_channels

    # allocate container for convolution patches
    # global PAD_CACHE
    # if PAD_CACHE is not None:
    #     patches = PAD_CACHE
    # else:

    patches = np.zeros((p_rows, p_cols, p_dpth), 
                       dtype=np.float32)
    # PAD_CACHE = patches


    # extract patches
    _lib.sliding_patches(
        feats.ctypes.data_as(ct.c_void_p), 
        patches.ctypes.data_as(ct.c_void_p), 
        ct.c_int(n_rows), 
        ct.c_int(n_cols),
        ct.c_int(n_channels),
        ct.c_int(p_size),
        ct.c_int(p_strd))

    # convert filters into columns
    filters = filters.reshape(p_size**2*n_channels, -1)
    # print patches[0,0,:]
    # print "-- patch size %s" % (patches.shape,)
    # print "-- filter size %s" % (filters.shape,)

    featmap = np.dot(patches, filters)
    # print "-- featmap size %s" % (featmap.shape,)
    
    return featmap


if __name__ == '__main__':
    pass
