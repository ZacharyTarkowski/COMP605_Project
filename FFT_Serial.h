#ifndef FFT_SERIAL
#define FFT_SERIAL
#include <stdio.h>
#include <math.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

#define cnum std::complex<double>
using namespace std::complex_literals;

enum band_type {Y, CB, CR};

static const unsigned bit_rev_lut[] = {
    0, 4, 2, 6, 1, 5, 3, 7
};

typedef struct xformBlock {
    cnum data[8][8];
    int row_index;
    int col_index;
    enum band_type band;

} xformBlock;

void FFT_8x8(xformBlock* blockptr);
void IFFT_8x8(xformBlock* blockptr);

#endif 