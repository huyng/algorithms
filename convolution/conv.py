import numpy as np
import ctypes as ct


_lib = np.ctypeslib.load_library("libconv.so", ".")
_lib.sliding_patches.restype = None
_lib.sliding_patches.argtypes = [ct.c_void_p, ct.c_void_p, 
                            ct.c_int, ct.c_int, ct.c_int, ct.c_int]

def convolve_with_patches(mat, filters, stride=1, pad=None):
    """
    
    Parameters
    ----------

    mat: ndarray(dtype=np.float32)
        4d array of shape (n, rows, cols, depth)

    filters: ndarray(dtype=np.float32)
        3d array of shape (rows, cols, depth)

    stride: int
        This determines the step size to slide the convolution 
        window. You can use this to sub sample your convolution
        operation. Default 1

    pad: int or None
         Amount of padding to add to your images


    """

    # 
    n_rows = mat.shape[0]
    n_cols = mat.shape[1]
    n_channels = mat.shape[2]
    p_size = filters.shape[0]
    p_strd = 1
    p_rows = int((n_rows-p_size)/p_strd) + 1
    p_cols = int((n_cols-p_size)/p_strd) + 1
    p_dpth = p_size**2*n_channels

    patches = np.zeros((p_rows, p_cols, p_dpth), dtype=np.float32)
    _lib.sliding_patches(
        mat.ctypes.data_as(ct.c_void_p), 
        patches.ctypes.data_as(ct.c_void_p), 
        ct.c_int(n_rows), 
        ct.c_int(n_cols),
        ct.c_int(n_channels),
        ct.c_int(p_size),
        ct.c_int(p_strd))

    return np.dot(patches, filters)


if __name__ == '__main__':
    pass
