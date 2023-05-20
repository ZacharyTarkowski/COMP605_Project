#include "FFT_Serial.h"



void bit_rev_8(cnum* fft_arr)
{
    cnum temp[8];
    memcpy(temp, fft_arr, 8 * sizeof(cnum));

    for (int i =0; i< 8; i++)
    {
        fft_arr[i] = temp[bit_rev_lut[i]];
    }

}

void FFT_8(cnum* fft_arr)
{
    int s,k,m,j;
    cnum w,wm,t,u;
    bit_rev_8(fft_arr);

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



void FFT_8x8(xformBlock* blockptr)
{
    int row, col;
    cnum col_arr[8];
    
    for (row=0 ; row <8 ; row++)
        {
            FFT_8(blockptr->data[row]);
        }

    for (col=0 ; col <8 ; col++)
        {
            for(row = 0; row <8 ; row++)
                {
                    col_arr[row] = blockptr->data[row][col];
                }

            FFT_8(col_arr);

            for(row = 0; row <8 ; row++)
                {
                    blockptr->data[row][col] = col_arr[row];
                }
        }
}


void IFFT_8(cnum* fft_arr)
{
    for ( int i = 0; i<8; i++)
        {
            fft_arr[i] = std::conj(fft_arr[i]);
        }

    FFT_8(fft_arr);

    for ( int i = 0; i<8; i++)
        {
            fft_arr[i] = (0.125) * std::conj(fft_arr[i]);
        }
}

void IFFT_8x8(xformBlock* blockptr)
{
    int row, col;
    cnum col_arr[8];
    
    for (row=0 ; row <8 ; row++)
        {
            IFFT_8(blockptr->data[row]);
        }

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
 
int main()
{
    using namespace std::complex_literals;

    xformBlock block;

     cnum fft_arr2[8];

    for (int i =0; i< 8; i++)
    {
        for (int j =0; j< 8; j++)
        {
            block.data[i][j] = (double)i * 1;
            //std::cout << fft_arr[i][j] << '\n';
        }

        fft_arr2[i] =  (double)i * 1;
    }

    FFT_8x8(&block);
    IFFT_8x8(&block);

    for (int i =0; i< 8; i++)
    {
        for (int j =0; j< 8; j++)
        {
            
            std::cout << block.data[i][j];
        }

        std::cout << "\n";
    }

    std::cout <<" seperation "<<"\n";

    FFT_8(fft_arr2);
    IFFT_8(fft_arr2);

    for (int j =0; j< 8; j++)
        {
            
            std::cout << fft_arr2[j];
        }

}