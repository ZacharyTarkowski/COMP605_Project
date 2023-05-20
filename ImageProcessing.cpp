#include "ImageProcessing.hpp"

using namespace cv;
using namespace std;

/*******************************************************************
    QuantizeChrominance

    inputs:
    xformBlock* blockptr

    outputs:
    xformBlock* blockptr

    description:
    Operate on pixels of image on a block by block level 
    Perform quantize operation of dividing by chrominance quantize value corresponding to location within block
    and then rounding resultant

*******************************************************************/
void QuantizeChrominance(xformBlock* blockptr)
{
    for (int row = 0; row<8; row++)
    {
        for (int col = 0; col<8; col++)
        {
            //store quantize value in a complex
            std::complex<double> quant_val(ChrominanceQuantizationTable[row][col],ChrominanceQuantizationTable[row][col]);
            //perform division of complex over scaler
            blockptr->data[row][col]=(blockptr->data[row][col]/quant_val);
            //create a temporary complex struct holding quantized values rounded
            std::complex<double> temp(round(blockptr->data[row][col].real()),round(blockptr->data[row][col].imag()));
            //copy this back into blockptr location copied from
            blockptr->data[row][col] = temp;
        }
    }
}

/*******************************************************************
    QuantizeLuminance

    inputs:
    xformBlock* blockptr

    outputs:
    xformBlock* blockptr

    description:
    Operate on pixels of image on a block by block level 
    Perform quantize operation of dividing by luminance quantize value corresponding to location within block
    and then rounding resultant

*******************************************************************/
void QuantizeLuminance(xformBlock* blockptr)
{
    for (int row = 0; row<8; row++)
    {
        for (int col = 0; col<8; col++)
        {
            std::complex<double> quant_val(LuminanceQuantizationTable[row][col],LuminanceQuantizationTable[row][col]);
            blockptr->data[row][col]=(blockptr->data[row][col]/quant_val);
            std::complex<double> temp(round(blockptr->data[row][col].real()),round(blockptr->data[row][col].imag()));
            blockptr->data[row][col] = temp;
        }
    }
}

/*******************************************************************
    InvQuantizeChrominance

    inputs:
    xformBlock* blockptr

    outputs:
    xformBlock* blockptr

    description:
    Operate on pixels of image on a block by block level 
    Undo quantize operation by multiplying chrominance value corresponding to location within block

*******************************************************************/
void InvQuantizeChrominance(xformBlock* blockptr)
{
    for (int row = 0; row<8; row++)
    {
        for (int col = 0; col<8; col++)
        {
            std::complex<double> quant_val(ChrominanceQuantizationTable[row][col],ChrominanceQuantizationTable[row][col]);
            blockptr->data[row][col]=(blockptr->data[row][col]*quant_val);
        }
    }
}

/*******************************************************************
    InvQuantizeLuminance

    inputs:
    xformBlock* blockptr

    outputs:
    xformBlock* blockptr

    description:
    Operate on pixels of image on a block by block level 
    Undo quantize operation by multiplying luminance value corresponding to location within block

*******************************************************************/
void InvQuantizeLuminance(xformBlock* blockptr)
{
    for (int row = 0; row<8; row++)
    {
        for (int col = 0; col<8; col++)
        {
            std::complex<double> quant_val(LuminanceQuantizationTable[row][col],LuminanceQuantizationTable[row][col]);
            blockptr->data[row][col]=(blockptr->data[row][col]*quant_val);
            
        }
    }
}

/*******************************************************************
    SubSample

    inputs:
    Mat bandmat
        Mat object carrying a chrominance band to be reduced to a 4:2:0 sample rate

    Mat bandmat420 
        has half as many cols and half as many rows as bandmat 

    outputs:
    Mat bandmat420

    description:
    Saves pixels at every other row and col into a smaller mat create a subsampled matrix of color band

*******************************************************************/
void SubSample(Mat bandmat, Mat bandmat420)
{
   int i,j; 
   for (int x = 0; x< bandmat.rows/2; x++) // from pixel 1 to end of new bound
    {
        for (int y = 0; y< bandmat.cols/2; y++) // from pixel 1 to end of new bound
        {
            
            i = 2*x + 1; // skips every other row
            j = 2*y + 1; // skips every other column
            bandmat420.at<uchar>(x,y) = bandmat.at<uchar>(i,j); // assign row and column to subsample variable
        }
    } 
}

/*******************************************************************
    UpSample

    inputs:
    Mat bandmat
        Mat object carrying a chrominance band to be Upsampled from 4:2:0 to 4:4:4

    Mat bandmat420 
        has twice as many cols and twice as many rows as bandmat 

    outputs:
    Mat bandmatUp

    description:
    Store orignal pixels in larger matrix and linearly interpolate the gaps through taking the average of adjacent pixels between columns
    and copying the previous row into gap rows

*******************************************************************/
void UpSample(Mat bandmat, Mat bandmatUp)
{

   for (int x = 0; x < bandmatUp.rows; x++) // from pixel 1 to end of new bound
    {
        if(x % 2 != 0)
        {
            for (int y = 0; y < bandmatUp.cols; y++) // from pixel 1 to end of new bound
            {
                bandmatUp.at<uchar>(x,y) = bandmatUp.at<uchar>(x-1,y);
            }
        }
        else
        {
            for (int y = 0; y < bandmatUp.cols; y++) // from pixel 1 to end of new bound
            {   
                if( (y%2 != 0))
                {
                    bandmatUp.at<uchar>(x,y) = bandmatUp.at<uchar>(x,y-1);
                }
                else
                {
                    bandmatUp.at<uchar>(x,y) = bandmat.at<uchar>(x/2,y/2);
                }
            }
        }
    } 
}