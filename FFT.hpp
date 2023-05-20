#ifndef FFT
#define FFT
#include <stdio.h>
#include <math.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <stdint.h>


#define cnum std::complex<double>
#define uint8 uint8_t
#define int16 int16_t
#define uint32 uint32_t
using namespace std::complex_literals;

enum band_type {Y, CB, CR};

//takes input of current index value and returns decomposed index value
static const unsigned bit_rev_lut[] = {
    0, 4, 2, 6, 1, 5, 3, 7
};

//xformblock struct for block image processing
typedef struct xformBlock {
    cnum data[8][8];
    int row_index;
    int col_index;
    enum band_type band;

} xformBlock;

void FFT_8x8(xformBlock* blockptr);
void IFFT_8x8(xformBlock* blockptr);

#endif 