#include "FFT.hpp"


/* Interlaced decomposition function for 1D cnum array of 8 elements */
void bit_rev_8(cnum* fft_arr)
{
    cnum temp[8];
    memcpy(temp, fft_arr, 8 * sizeof(cnum));

    for (int i =0; i< 8; i++)
    {
        fft_arr[i] = temp[bit_rev_lut[i]];
    }

}

/* 1D FFT for array of 8 elements */
void FFT_8(cnum* fft_arr)
{
    int s,k,m,j;
    cnum w,wm,t,u;

    /* FFT formula */

    /* Interlaced Decomposition */
    bit_rev_8(fft_arr);

    /* FFT algorithm */
    for (s = 1 ; s <= log2(8); s++)
    {
        m = std::pow(2, s);//can probably just bit shift this
        wm = std::exp( (((double) -2 * M_PI* 1i) / (double)m));
        for(k=0; k<8; k+=m)
        {
            w = 1;
            for (j = 0; j < m/2 ; j++)
            {
                t = w * fft_arr[ k + j + (m/2)];
                u = fft_arr[ k + j ];
                fft_arr[ k + j ] = u + t;
                fft_arr[ k + j + (m/2)] = u - t;
                w = w*wm;
                
            }
        }

    }
}


/* Computes the 2D FFT of a transform block */
/* The 2D FFT is just an FFT of rows then columns (or columns then rows) */
void FFT_8x8(xformBlock* blockptr)
{
    int row, col;
    cnum col_arr[8];
    
    /* 1D FFT on all rows of the block */
    for (row=0 ; row <8 ; row++)
        {
            FFT_8(blockptr->data[row]);
        }

    /* 1D FFT on all cols of the block */
    for (col=0 ; col <8 ; col++)
        {
            /* Copy col data */
            for(row = 0; row <8 ; row++)
                {
                    col_arr[row] = blockptr->data[row][col];
                }

            /* Col FFT */
            FFT_8(col_arr);

            /* Write col data back */
            for(row = 0; row <8 ; row++)
                {
                    blockptr->data[row][col] = col_arr[row];
                }
        }
}

/* 1D inverse FFT */
void IFFT_8(cnum* fft_arr)
{
    /* IFFT can be simplified into 1 / n * ( FFT(conjugate ) ) */

    /* Take the conjugate of every element */
    for ( int i = 0; i<8; i++)
        {
            fft_arr[i] = std::conj(fft_arr[i]);
        }

    /* FFT on the conjugate array */
    FFT_8(fft_arr);

    /* 1/n * the conjugate again */
    for ( int i = 0; i<8; i++)
        {
            fft_arr[i] = (0.125) * std::conj(fft_arr[i]);
        }
}

/* 2D IFFT of a transform block */
/* The 2D IFFT is just an FFT of rows then columns (or columns then rows) */
void IFFT_8x8(xformBlock* blockptr)
{
    int row, col;
    cnum col_arr[8];
    
    /* IFFT on rows */
    for (row=0 ; row <8 ; row++)
        {
            IFFT_8(blockptr->data[row]);
        }

    /* IFFT on columns */
    for (col=0 ; col <8 ; col++)
        {
            for(row = 0; row <8 ; row++)
                {
                    col_arr[row] = blockptr->data[row][col];
                }

            IFFT_8(col_arr);

            for(row = 0; row <8 ; row++)
                {
                    blockptr->data[row][col] = col_arr[row];
                }
            
        }
}
 