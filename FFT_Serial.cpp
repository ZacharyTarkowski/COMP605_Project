#include <stdio.h>
#include <math.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

#define cnum std::complex<double>
using namespace std::complex_literals;

static const unsigned bit_rev_lut[] = {
    0, 4, 2, 6, 1, 5, 3, 7
};

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
        //std::cout<<"m="<<m<<" wm"<< wm<<"\n";
        //std::cout<<(double) -6.28 * 1i / (double)m<<"\n";
        for(k=0; k<8; k+=m)
        {
            w = 1;
            for (j = 0; j < m/2 ; j++)
            {
                //std::cout<<"k="<<k<<" j="<<j<<" k+j="<<k+j<<" kj (m/2)="<<k + j + (m/2)<<" w="<<w<<"\n";
                t = w * fft_arr[ k + j + (m/2)];
                u = fft_arr[ k + j ];
                fft_arr[ k + j ] = u + t;
                fft_arr[ k + j + (m/2)] = u - t;
                w = w*wm;
                
            }
        }

    }
}



void FFT_8x8(cnum **fft_arr)
{
    int row, col;
    cnum col_arr[8];
    
    for (row=0 ; row <8 ; row++)
        {
            FFT_8(fft_arr[row]);
        }

    for (col=0 ; col <8 ; col++)
        {
            for(row = 0; row <8 ; row++)
            {
                col_arr[row] = fft_arr[row][col];
                //std::cout<<" row"<<row<<" col"<<col;
            }
           

            for (int j =0; j< 8; j++)
        {
            //std::cout << col_arr[j];
        }
        // std::cout << "\n";

            FFT_8(col_arr);

            for(row = 0; row <8 ; row++)
            {
                fft_arr[row][col] = col_arr[row];
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

void IFFT_8x8(cnum **fft_arr)
{
    int row, col;
    cnum col_arr[8];
    
    for (row=0 ; row <8 ; row++)
        {
            IFFT_8(fft_arr[row]);
        }

    for (col=0 ; col <8 ; col++)
        {
            for(row = 0; row <8 ; row++)
            {
                col_arr[row] = fft_arr[row][col];
                //std::cout<<" row"<<row<<" col"<<col;
            }
           

            for (int j =0; j< 8; j++)
        {
            //std::cout << col_arr[j];
        }
        // std::cout << "\n";

            IFFT_8(col_arr);

            for(row = 0; row <8 ; row++)
            {
                fft_arr[row][col] = col_arr[row];
            }
            
        }
}
 
int main()
{
    using namespace std::complex_literals;

    cnum **fft_arr;
    fft_arr = new cnum *[8];
    for (int i =0; i< 8; i++)
    {
        fft_arr[i] = new cnum[8];
    }

    cnum fft_arr2[8];

    for (int i =0; i< 8; i++)
    {
        for (int j =0; j< 8; j++)
        {
            fft_arr[i][j] = (double)i * 1i;
            //std::cout << fft_arr[i][j] << '\n';
        }

        fft_arr2[i] =  (double)i * 1i;
    }

    FFT_8x8(fft_arr);
    IFFT_8x8(fft_arr);

    // for (int i =0; i< 8; i++)
    // {
    //      std::cout <<i<<" "<< fft_arr[i] << '\n';
    // }

    for (int i =0; i< 8; i++)
    {
        for (int j =0; j< 8; j++)
        {
            
            std::cout << fft_arr[i][j];
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




    // std::cout << std::fixed << std::setprecision(1);
 
    // std::complex<double> z1 = 1i * 1i; // imaginary unit squared
    // std::cout << "i * i = " << z1 << '\n';
 
    // std::complex<double> z2 = std::pow(1i, 2); // imaginary unit squared
    // std::cout << "pow(i, 2) = " << z2 << '\n';
 
    // const double PI = std::acos(-1); // or std::numbers::pi in C++20
    // std::complex<double> z3 = std::exp(1i * PI); // Euler's formula
    // std::cout << "exp(i * pi) = " << z3 << '\n';
 
    // std::complex<double> z4 = 1. + 2i, z5 = 1. - 2i; // conjugates
    // std::cout << "(1 + 2i) * (1 - 2i) = " << z4 * z5 << '\n';
}