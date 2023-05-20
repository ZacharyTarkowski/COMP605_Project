#ifndef ImageProcessing
#define ImageProcessing

#include<iostream>
#include "FFT.hpp"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <stdint.h>

using namespace cv;

//initialize 2D array of quantize matrix quality level 50 for quantize function
const int LuminanceQuantizationTable[8][8] = 
{
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

const int ChrominanceQuantizationTable[8][8] = 
{
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}
};

void QuantizeChrominance(xformBlock* blockptr);
void QuantizeLuminance(xformBlock* blockptr);
void InvQuantizeChrominance(xformBlock* blockptr);
void InvQuantizeLuminance(xformBlock* blockptr);

void SubSample(Mat bandmat, Mat bandmat420);
void UpSample(Mat bandmat, Mat bandmatUp);

#endif 