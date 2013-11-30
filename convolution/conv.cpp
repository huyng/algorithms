#include <cstdlib>
#include <string.h>



extern "C" {

// Extract overlapping patches as flattened arrays 
// from `im` into `patches`. This function is modeled
// after matlab's im2col function.
// 
// 
// Parameters
// ----------
// 
// im:
//      pointer to array of type T[n_rows][n_cols][n_dpth]
// patches:
//      pointer to array of type T[p_rows][p_cols][p_size*p_size*n_dpth]
// n_rows:
//      number of rows in im
// n_cols:
//      number of cols in im
// n_dpth:
//      number of channels in im
// p_size:
//      the edge size of the square patch
// p_strd:
//      the "stride" or amount to move patches by when sliding
//      
void sliding_patches (
        const float* im,
        float* patches,
        const int n_rows,
        const int n_cols,
        const int n_dpth,
        const int p_size,
        const int p_strd)
{
    
    // dimension of patch if it was flattened
    int p_len = p_size * p_size;

    // compute p_rows & p_cols which is the
    // expected shape of patches output
    int p_rows = (n_rows - p_size) / p_strd + 1;
    int p_cols = (n_cols - p_size) / p_strd + 1;
    int p_dpth = n_dpth * p_size * p_size;


    // get number of array "steps" needed to reach
    // the next row in original image array
    int step_im = n_cols * n_dpth;

    // get number of array "steps" needed to reach
    // the next patch column
    int step_patch = p_size * n_dpth;


    // iterate over patch rows
    for (int idx_r = 0; idx_r < p_rows; ++idx_r) {

        // get pointer to patch[idx_r][0][0]
        float* ptr_patch = patches + idx_r * p_cols * p_len * n_dpth;

        // iterate over patch columns
        for (int idx_c = 0; idx_c < p_cols; ++idx_c) {

            // copy im[wnd_top:wnd_btm, wnd_lft:wnd_rht, :]
            int wnd_top = idx_r * p_strd;
            int wnd_btm = wnd_top + p_size;
            int wnd_lft = idx_c * p_strd;
            int wnd_rht = wnd_lft * p_size;

            const float* ptr_im = im + (wnd_top * n_cols + wnd_lft) * n_dpth;
            for (int i = wnd_top; i < wnd_btm; ++i) {
                
                // copy one patch row worth of image data into
                // patch channel 
                // e.g:  image[i, wnd_lft:wnd_rht, :] -> patches[idx_r][idx_c+i][p_size*n_dpth*i]
                memcpy(ptr_patch, ptr_im, sizeof(float) * step_patch);

                ptr_patch += step_patch; // step inward psize *nchans
                ptr_im += step_im; // step down one image row
            }
        }
    }
}

} // extern C

