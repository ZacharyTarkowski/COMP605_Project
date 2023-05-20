#include<iostream>
#include "FFT.hpp"
#include "FFT.cpp"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <stdint.h>
//using namespace cv;
using namespace std;
using namespace cv;

void mat2block(Mat* mat, xformBlock* block);
void block2mat(Mat* mat, xformBlock* block);

int main(int argc, const char** argv) 
{
    Mat img = imread("/home/605/tarkow/Project/lena.jpg");
    Mat myImage_Converted;
    Mat different_Channels[3];
    //cv::imshow("Window",img);
    //cv::waitKey(0);

    cvtColor(img,myImage_Converted, COLOR_RGB2YCrCb);
    cout << " dims " << myImage_Converted.dims <<" cols " << myImage_Converted.cols <<" rows " 
    << myImage_Converted.rows <<endl;

    split(myImage_Converted, different_Channels);
    Mat Y = different_Channels[0];
    Mat Cb = different_Channels[1];
    Mat Cr = different_Channels[2];
    cout << " dims " << Y.dims <<" cols " << Y.cols <<" rows " 
    << Y.rows <<endl;
    cout << " dims " << Cb.dims <<" cols " << Cb.cols <<" rows " 
    << Cb.rows <<endl;

    xformBlock test;

    int numBlocks = ( myImage_Converted.rows / 8 ) * (  myImage_Converted.cols / 8  );
    int numRows = myImage_Converted.rows;
    int numCols = myImage_Converted.cols;

    cout << numBlocks <<  " "<<numRows <<" "<< numCols<<endl;

    Rect roi (0,0,8,8);
   // Mat im_roi = myImage_Converted(roi);
   // cout << im_roi << endl;

    Mat im_roi = Y(roi).clone();

    cout<<im_roi.cols<<endl;

    mat2block(&im_roi,&test);

    //FFT_8x8(&test);
   // IFFT_8x8(&test);

    xformBlock blockArr[numBlocks];
    int blockidx = 0;

    for (int currRow = 0; currRow < numRows; currRow+=8)
    {
        for (int currCol = 0; currCol < numCols; currCol+=8)
            {
                //cout<< currRow << " " << currCol << " "<<blockidx<<endl;
                Rect roi (currCol,currRow,8,8);
                im_roi = Y(roi).clone();
                mat2block(&im_roi,&blockArr[blockidx]);
                blockidx++;
                //remember to put metadata in the blocks
            }
    }

    for (int i = 0; i<numBlocks; i++)
    {
        FFT_8x8(&blockArr[i]);
        IFFT_8x8(&blockArr[i]);
    }

    Mat img2 = Mat(1080,1920, CV_8UC1, 0.0);
    Mat img3 = Mat(1080,1920, CV_8UC1, 0.0);

    blockidx = 0;
    for (int currRow = 0; currRow < numRows; currRow+=8)
    {
        for (int currCol = 0; currCol < numCols; currCol+=8)
            {
                //cout<< currRow << " " << currCol << " "<<blockidx<<endl;
                Rect roi (currCol,currRow,8,8);
                block2mat(&im_roi,&blockArr[blockidx]);
                //img2(roi) = im_roi.clone();
                im_roi.copyTo(img2(roi));
                //cout<<im_roi;
                blockidx++;
                //remember to put metadata in the blocks
            }
    }

    absdiff(Y,img2,img3);
    cout<<img2.dims<<" "<<img2.cols<<" "<<img2.rows<<" "<<endl;
    cout <<img3;

    imwrite("\MyImage.png", myImage_Converted);
    imwrite("\MyImage2.png", Y);
    imwrite("\MyImage3.png", Cb);
    imwrite("\MyImage4.png", Cr);
    imwrite("\MyImage5.png", img2);
    imwrite("\MyImage6.png", img3);
    

    return 0;
}

void mat2block(Mat* mat, xformBlock* block)
{
    for (int currRow = 0; currRow < 8; currRow++)
    {
        for (int currCol = 0; currCol < 8; currCol++)
            {
                block->data[currRow][currCol] = mat->at<uchar>(currRow,currCol);
            }
    }
}

void block2mat(Mat* mat, xformBlock* block)
{
    for (int currRow = 0; currRow < 8; currRow++)
    {
        for (int currCol = 0; currCol < 8; currCol++)
            {
                 mat->at<uchar>(currRow,currCol) = (uchar)block->data[currRow][currCol].real();
            }
    }
}