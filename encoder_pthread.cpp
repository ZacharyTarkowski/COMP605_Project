
#include <pthread.h>
#include<iostream>
#include "FFT.hpp"
#include "FFT.cpp"
#include "ImageProcessing.hpp"
#include "ImageProcessing.cpp"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <stdint.h>
#include <string> 
#include <sys/resource.h>

using namespace std;
using namespace cv;

#define pthread pthread_t
#define join pthread_join
#define create pthread_create
#define mutex pthread_mutex_t


typedef struct thread_args {
    int rank;
    int Y_blockstart;
    int Y_blockend;
    int C_blockstart;
    int C_blockend;  
    xformBlock* Y_blockarr; 
    int16* Y_bitstream16;
    xformBlock* Cb_blockarr; 
    int16* Cb_bitstream16; 
    xformBlock* Cr_blockarr; 
    int16* Cr_bitstream16;
} thread_args;



void mat2block(Mat* mat, xformBlock* block);
void block2mat(Mat* mat, xformBlock* block);
void getblockarray(Mat mat, xformBlock* blockarr);
void getmat(Mat mat, xformBlock* blockarr);

void block2bitstream(xformBlock *block, uint8 *arr );
void bitstream2block(xformBlock *block, uint8 *arr );

void block2bitstream16(xformBlock *block, int16_t *arr );
void bitstream2block16(xformBlock *block, int16_t *arr );

void* compress(void* args);
void* decompress(void* args);

double PSNR(Mat PreImg, Mat PostImg);
//void decompress(int blockstart, int blockend, block* blockarr, int16* bitstream16);

int main(int argc, const char** argv) 
{

    int my_rank;
    int num_process;
    int thread_count = stoi(argv[1],0,10);

    
     Mat img;
    Mat myImage_Converted;

    int Y_numBlocks ;
    int Y_numRows ;
    int Y_numCols ;

    
    int C_numBlocks ;
    int C_numRows ;
    int C_numCols ;

    Mat Cr_420 ;
    Mat Cb_420 ;
    Mat Y;

    //-------------------------------------------------------------------------------------------------//
    //Image Preprocessing in root
    //-------------------------------------------------------------------------------------------------//

    cout<<"Number of threads : "<<thread_count << endl;
    cout<<"Image to compress : "<<argv[2] << endl;
        img = imread(argv[2]);

        Mat different_Channels[3];
        //image conversion from RGB to YCrCb to perform encoding process
        cvtColor(img,myImage_Converted, COLOR_RGB2YCrCb);

    //split image matrix into multiple matrices for encoding on each band
        split(myImage_Converted, different_Channels);
        Y = different_Channels[0];
        Mat Cr = different_Channels[1];
        Mat Cb = different_Channels[2];
    //find number of blocks necessary for luminance 
        Y_numBlocks = ( myImage_Converted.rows / 8 ) * (  myImage_Converted.cols / 8  );
        Y_numRows = myImage_Converted.rows;
        Y_numCols = myImage_Converted.cols;

    //Chrominance bands will be subsampled to 4:2:0 and thus require half as many blocks in x and  y or 1/4 overall
        Cr_420 = Mat(myImage_Converted.rows/2,myImage_Converted.cols/2, CV_8UC1, 0.0);
        Cb_420 = Mat(myImage_Converted.rows/2,myImage_Converted.cols/2, CV_8UC1, 0.0);

    //subsample call and then pad rows/cols to make resultant matrix divisble by 8x8

        SubSample(Cr,Cr_420);
        SubSample(Cb,Cb_420);

        Mat row_cr = Mat(1, Cr_420.cols, CV_8UC1, 128.0 );
        Mat row_cb = Mat(1, Cr_420.cols, CV_8UC1, 128.0 );
        Mat row = Mat(1, Cr_420.cols, CV_8UC1, 128.0 );
        row_cr.row(0).copyTo(Cr_420.row(Cr_420.rows-1));
        row_cb.row(0).copyTo(Cr_420.row(Cr_420.rows-1));

        int padding = Cr_420.rows % 8;

        Cr_420.resize(Cr_420.rows + padding, 1.0);
        Cb_420.resize(Cb_420.rows + padding, 1.0);

        int change_row;

        for(int i =1; i<= padding; i++)
        {
            change_row=Cr_420.rows-i;
            Cr_420.row(change_row).copyTo(row_cr.row(0));
            Cb_420.row(change_row).copyTo(row_cb.row(0));
        }

        int padding_cols = 0;
        Mat col_cr = Mat(Cr_420.rows, 1, CV_8UC1, 128.0 );
        Mat col_cb = Mat(Cb_420.rows, 1, CV_8UC1, 128.0 );
        col_cr.col(0).copyTo(Cr_420.col(Cr_420.cols-1));
        col_cb.col(0).copyTo(Cr_420.col(Cr_420.cols-1));

        Mat matArray_Cr[] = { Cr_420, col_cr};
        Mat matArray_Cb[] = { Cb_420, col_cb};

        for(int i =1; i<= padding_cols; i++)
        {
            hconcat( matArray_Cr, 2, Cr_420 );
            hconcat( matArray_Cb, 2, Cb_420 );
            matArray_Cr[0] = Cr_420;
            matArray_Cb[0] = Cb_420;
        }

        C_numBlocks = ( Cr_420.rows / 8 ) * (  Cr_420.cols / 8  );
        C_numRows = Cr_420.rows;
        C_numCols = Cr_420.cols;
    //end chrominance subsampling, padding, and finding number of blocks




    //initialize block array of size number of blocks we calculated
    xformBlock my_Y_blockArr[Y_numBlocks];
    xformBlock my_Cb_blockArr[C_numBlocks];
    xformBlock my_Cr_blockArr[C_numBlocks];

    //initialize bitstream sizes
    int16* my_Y_bitstream16 = (int16*)malloc(Y_numBlocks*272);
    int16* Y_bitstream16 = (int16*)malloc(0);

    int16* my_Cr_bitstream16 = (int16*)malloc(C_numBlocks*272);
    int16* Cr_bitstream16 = (int16*)malloc(0);

    int16* my_Cb_bitstream16 = (int16*)malloc(C_numBlocks*272);
    int16* Cb_bitstream16 = (int16*)malloc(0);
    
    //populate block array with blocks from image matrix for corresponding band
    getblockarray(Y,my_Y_blockArr);
    getblockarray(Cb_420,my_Cb_blockArr);
    getblockarray(Cr_420,my_Cr_blockArr);
    pthread* thread_handles = (pthread*)malloc(thread_count*sizeof(pthread));
    thread_args* thread_args_arr = (thread_args*)malloc(thread_count*sizeof(thread_args));
    int thread; 
    
    /* take start time before thread creation*/
    struct rusage start, end;
    getrusage(RUSAGE_SELF, &start);
    for (thread = 0; thread < thread_count;thread++)
    {
        //calculate where in array the thread should start and end
        int Y_start = (Y_numBlocks / thread_count)* thread;
        int C_start = (C_numBlocks / thread_count)* thread;

        int Y_end = (thread != thread_count-1) ? (Y_numBlocks / thread_count) : (Y_numBlocks / thread_count) + (Y_numBlocks % thread_count);
        int C_end = (thread != thread_count-1) ? (C_numBlocks / thread_count) : (C_numBlocks / thread_count) + (C_numBlocks % thread_count);

        //populate thread args struct
        thread_args_arr[thread].rank = thread;
        thread_args_arr[thread].Y_blockstart=Y_start ;
        thread_args_arr[thread].Y_blockend=Y_start + Y_end;
        thread_args_arr[thread].C_blockstart=C_start;
        thread_args_arr[thread].C_blockend= C_start + C_end;
        thread_args_arr[thread].Y_blockarr= my_Y_blockArr;
        thread_args_arr[thread].Y_bitstream16=my_Y_bitstream16;
        thread_args_arr[thread].Cb_blockarr=my_Cb_blockArr;
        thread_args_arr[thread].Cb_bitstream16= my_Cb_bitstream16;
        thread_args_arr[thread].Cr_blockarr=my_Cr_blockArr;
        thread_args_arr[thread].Cr_bitstream16=my_Cr_bitstream16;

        //create threads for compress
        create(&thread_handles[thread],NULL, compress, (void*) &thread_args_arr[thread]);
    }

    //thread joins
    for (thread = 0; thread<thread_count;thread++)
    {
        join(thread_handles[thread],NULL);
    }

    //take time for compression finish
    getrusage(RUSAGE_SELF, &end);
    printf("Compress System Time: %.06f\n\r", ((end.ru_stime.tv_sec - start.ru_stime.tv_sec) 
        + 1e-6*(end.ru_stime.tv_usec - start.ru_stime.tv_usec)));
        printf("Compress User Time: %.06f\n\r", ((end.ru_utime.tv_sec - start.ru_utime.tv_sec) 
        + 1e-6*(end.ru_utime.tv_usec - start.ru_utime.tv_usec)));

    


        //This is where a compressed file could be written to and encoded.


    //-------------------------------------------------------------------------------------------------//
    // Image Processing to reconstruct image
    //-------------------------------------------------------------------------------------------------//

//take time for decompression
getrusage(RUSAGE_SELF, &start);
for (thread = 0; thread < thread_count;thread++)
    {
        //calculate where in array the thread should start and end
        int Y_start = (Y_numBlocks / thread_count)* thread;
        int C_start = (C_numBlocks / thread_count)* thread;

        int Y_end = (thread != thread_count-1) ? (Y_numBlocks / thread_count) : (Y_numBlocks / thread_count) + (Y_numBlocks % thread_count);
        int C_end = (thread != thread_count-1) ? (C_numBlocks / thread_count) : (C_numBlocks / thread_count) + (C_numBlocks % thread_count);
        
        //populate thread args struct
        thread_args_arr[thread].rank = thread;
        thread_args_arr[thread].Y_blockstart=Y_start ;
        thread_args_arr[thread].Y_blockend=Y_start + Y_end;
        thread_args_arr[thread].C_blockstart=C_start;
        thread_args_arr[thread].C_blockend= C_start + C_end;
        thread_args_arr[thread].Y_blockarr= my_Y_blockArr;
        thread_args_arr[thread].Y_bitstream16=my_Y_bitstream16;
        thread_args_arr[thread].Cb_blockarr=my_Cb_blockArr;
        thread_args_arr[thread].Cb_bitstream16= my_Cb_bitstream16;
        thread_args_arr[thread].Cr_blockarr=my_Cr_blockArr;
        thread_args_arr[thread].Cr_bitstream16=my_Cr_bitstream16;

        //create threads for decompress
        create(&thread_handles[thread],NULL, decompress, (void*) &thread_args_arr[thread]);
    }

    //thread joins
    for (thread = 0; thread<thread_count;thread++)
    {
        join(thread_handles[thread],NULL);
    }

    //take timing for end of decompress and report
    getrusage(RUSAGE_SELF, &end);
    printf("Decompress System Time: %.06f\n\r", ((end.ru_stime.tv_sec - start.ru_stime.tv_sec) 
    + 1e-6*(end.ru_stime.tv_usec - start.ru_stime.tv_usec)));
    printf("Decompress User Time: %.06f\n\r", ((end.ru_utime.tv_sec - start.ru_utime.tv_sec) 
    + 1e-6*(end.ru_utime.tv_usec - start.ru_utime.tv_usec)));


    //initialize matrices to hold decoded matrices

    Mat Y_compressed = Mat(img.rows,img.cols, CV_8UC1, 0.0);
    Mat Cb_compressed = Mat(img.rows,img.cols, CV_8UC1, 0.0);
    Mat Cr_compressed = Mat(img.rows,img.cols, CV_8UC1, 0.0);

    //calculate size needed for original chrominance matrices typically are read from header of the type of file image stored in
    //here we must calculate as we are not creating a new file type
    Mat Cr_420_compressed = Mat(myImage_Converted.rows/2,myImage_Converted.cols/2, CV_8UC1, 0.0);
    Mat Cb_420_compressed = Mat(myImage_Converted.rows/2,myImage_Converted.cols/2, CV_8UC1, 0.0);

    int padding_compressed = Cr_420_compressed.rows % 8;
    Cr_420_compressed.resize(Cr_420_compressed.rows + padding_compressed, 1.0);
    Cb_420_compressed.resize(Cb_420_compressed.rows + padding_compressed, 1.0);



    Mat differenceimage = Mat(img.rows,img.cols, CV_8UC1, 0.0);
    //get the blocks into matrix format stored in objects of size we calculated above
    getmat(Y_compressed,my_Y_blockArr);
    getmat(Cb_420_compressed,my_Cb_blockArr);
    getmat(Cr_420_compressed,my_Cr_blockArr);

    //upsample chrominance images
    UpSample(Cr_420_compressed,Cr_compressed);
    UpSample(Cb_420_compressed,Cb_compressed);

    //store in matrix array to be merged in OpenCV 3 dimensional mat object
    different_Channels[0] = Y_compressed;
    different_Channels[1] = Cr_compressed;
    different_Channels[2] = Cb_compressed;

    Mat img_compressed = Mat(img.rows,img.cols, CV_8UC1, 0.0);
    Mat img_compressed_rgb = Mat(img.rows,img.cols, CV_8UC1, 0.0);
    merge(different_Channels,3,img_compressed);
    
    //convert back to RGB from YCrCb
    cvtColor(img_compressed,img_compressed_rgb, COLOR_YCrCb2RGB);
    
    absdiff(myImage_Converted,img_compressed,differenceimage);

    imwrite("\Image_Compressed_pthread.png", img_compressed_rgb);
    //imwrite("\MyImage3.png", Cr_compressed);
    //imwrite("\MyImage4.png", Cb_compressed);
    //imwrite("\MyImage5.png", Y_compressed);
    imwrite("\Difference_Image_pthread.png", differenceimage);

    //split rgb channels of original and new image and use to find PSNR to determine lossyness of program
    Mat img_channels[3];
    split(img,img_channels);
    Mat R = img_channels[0];
    Mat G = img_channels[1];
    Mat B = img_channels[2];

    Mat img_channels_compress[3];
    split(img_compressed_rgb, img_channels_compress);
    Mat R_compress = img_channels_compress[0];
    Mat G_compress = img_channels_compress[1];
    Mat B_compress = img_channels_compress[2];

    //cout << "alive?" <<endl;
    double PSNR_R = PSNR(R, R_compress);
    double PSNR_G = PSNR(G, G_compress);
    double PSNR_B = PSNR(B, B_compress);

    cout <<"PSNR_R "<< PSNR_R <<endl;
    cout <<"PSNR_G "<< PSNR_G <<endl;
    cout <<"PSNR_B "<< PSNR_B <<endl;

    free(my_Y_bitstream16 );
    free(Y_bitstream16 );

    free(my_Cr_bitstream16 );
    free(Cr_bitstream16 );

    free(my_Cb_bitstream16 );
    free(Cb_bitstream16 );

    free(thread_handles);
    free(thread_args_arr);


    return 0;
}
/*******************************************************************
    getmat

    inputs:
        opencv mat object, pointer to block array
    outputs:
        modified mat to reflect block array data
    description:
        takes an array of xformBlock structs and copys them into the proper place in the opencv mat

*******************************************************************/
void getmat(Mat mat, xformBlock* blockarr)
{
    int blockidx = 0;
    Mat im_roi = Mat(8,8, CV_8UC1, 0.0);

    for (int currRow = 0; currRow < mat.rows; currRow+=8)
    {
        for (int currCol = 0; currCol < mat.cols; currCol+=8)
            {
                Rect roi (currCol,currRow,8,8);

                block2mat(&im_roi,&blockarr[blockidx]);

                im_roi.copyTo(mat(roi));

                blockidx++;
            }
    }
}

/*******************************************************************
    getblockarray

    inputs:
        mat object with image data, empty xformblock array. 
        mat dimensions must be divisible by 8 to work properly.
    outputs:
        xformblock array will be filled with image data and metadata
    description:
        Takes opencv mat object and converts it into a 1D array of
        xformblock objects, with associated metadata.

*******************************************************************/
void getblockarray(Mat mat, xformBlock* blockarr)
{
    int blockidx = 0;
    Mat im_roi = Mat(8,8, CV_8UC1, 0.0);

    for (int currRow = 0; currRow < mat.rows; currRow+=8)
    {
        for (int currCol = 0; currCol < mat.cols; currCol+=8)
            {
                //rect function is col,row
                Rect roi (currCol,currRow,8,8);
                
                im_roi = mat(roi).clone();

                mat2block(&im_roi,&blockarr[blockidx]);

                blockidx++;
            }
    }
}

/* Helper function, copies the data of a single xformBlock struct into an opencv mat  */
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

/* Helper function, copies the data of a single opencv mat struct into a xformBlock struct  */
void block2mat(Mat* mat, xformBlock* block)
{
    for (int currRow = 0; currRow < 8; currRow++)
    {
        for (int currCol = 0; currCol < 8; currCol++)
            {

                /* The math within openCV's color transformation functions will occasionally cause an overflow/underflow
                    with imprecise numbers caused by our quantization. This math instead saturates potential errors */
                uchar temp;
                if (block->data[currRow][currCol].real() > 255.0)
                {
                    temp = 255;
                }
                else if (block->data[currRow][currCol].real() < 0.0)
                {
                    temp = 0;
                }
                else 
                {
                    temp = round(block->data[currRow][currCol].real());
                }
                 mat->at<uchar>(currRow,currCol) = temp;
            }
    }
}

/*******************************************************************
    block2bitstream

    inputs:
        xformblock pointer with data, uint8 array pointer
    outputs:
        xformblock pointer with data, uint8 array pointer with data
    description:
        takes a xformblock struct and outputs a uint8 bitstream

*******************************************************************/
void block2bitstream(xformBlock *block, uint8 *arr )
{
    memcpy(arr,&block->row_index,sizeof(int));
    memcpy(&arr[5],&block->col_index,sizeof(int));
    
    int arridx = 9;

    for (int i =0; i< 8; i++)
    {
        for (int j =0; j< 8; j++)
        {
            /* The math within openCV's color transformation functions will occasionally cause an overflow/underflow
                    with imprecise numbers caused by our quantization. This math instead saturates potential errors */
            uint8 temp_real;
            uint8 temp_imag = (uint8_t)block->data[i][j].imag();
            if (block->data[i][j].real() > 255.0)
            {
                temp_real = 255;
            }
            else if (block->data[i][j].real() < 0.0)
            {
                temp_real = 0;
            }
            else
            {
                temp_real = (uint8_t)block->data[i][j].real();
            }
                         
            memcpy(&arr[arridx],&temp_real,sizeof(uint8_t));
            memcpy(&arr[arridx+1],&temp_imag,sizeof(uint8_t));
            arridx+=2;
        }
    }
}


/*******************************************************************
    bitstream2block

    inputs:
        xformblock pointer, uint8 array pointer with data
    outputs:
        xformblock pointer with data, uint8 array pointer with data
    description:
        takes a stream of uint8s and transforms it into an xformblock

*******************************************************************/
void bitstream2block(xformBlock *block, uint8 *arr )
{
    block->row_index = arr[0] | (arr[1] << 8) | (arr[2] << 16) | (arr[3] << 24);
    block->col_index = arr[5] | (arr[6] << 8) | (arr[7] << 16) | (arr[8] << 24);
    //cout<<block->row_index << " "<< block->col_index<<endl;
    
    int arridx = 9;

    for (int i =0; i< 8; i++)
    {
        for (int j = 0; j< 8; j++)
        {
            block->data[i][j].real(arr[arridx]);
            block->data[i][j].imag(arr[arridx+1]);
            arridx+=2;
        }
    }

}

/*******************************************************************
    block2bitstream16

    inputs:
        xformblock pointer with data, uint16 array pointer
    outputs:
        xformblock pointer with data, uint16 array pointer with data
    description:
        takes a xformblock struct and outputs a uint16 bitstream

*******************************************************************/
void block2bitstream16(xformBlock *block, int16_t *arr )
{
    //memcpy(arr,&block->row_index,sizeof(int));
    //memcpy(&arr[5],&block->col_index,sizeof(int));
    
    int arridx = 5;
    
    //copy data plus convert to uchar
    for (int i =0; i< 8; i++)
    {
        for (int j =0; j< 8; j++)
        {
            int16_t temp_real = (int16_t)block->data[i][j].real();
            int16_t temp_imag = (int16_t)block->data[i][j].imag();
            memcpy(&arr[arridx],&temp_real,sizeof(int16_t));
            memcpy(&arr[arridx+1],&temp_imag,sizeof(int16_t));
            arridx+=2;
        }
    }
}

/*******************************************************************
    bitstream2block16

    inputs:
        xformblock pointer, uint16 array pointer with data
    outputs:
        xformblock pointer with data, uint16 array pointer with data
    description:
        takes a stream of uint16s and transforms it into an xformblock

*******************************************************************/
void bitstream2block16(xformBlock *block, int16_t *arr )
{
    //block->row_index = arr[0] | (arr[1] << 8) | (arr[2] << 16) | (arr[3] << 24);
    //block->col_index = arr[5] | (arr[6] << 8) | (arr[7] << 16) | (arr[8] << 24);
    //cout<<block->row_index << " "<< block->col_index<<endl;
    
    int arridx = 5;

    for (int i =0; i< 8; i++)
    {
        for (int j = 0; j< 8; j++)
        {
            block->data[i][j].real(arr[arridx]);
            block->data[i][j].imag(arr[arridx+1]);
            arridx+=2;
        }
    }

}

/*******************************************************************
    PSNR

    inputs:
        2D mat object holding a single color band  
        another 2D mat object holding a single color band to be eveluated against
    outputs:
        double that is the result of the Peak-Signal to Noise Ratio of the two mats
    description:
        calculates Peak-Signal to Noise Ratio 

*******************************************************************/
double PSNR(Mat PreImg, Mat PostImg)
{
    double MSE =0.0;
    double PSNR = 0.0;
    double temp = 0.0;
    for (int currRow = 0; currRow < PreImg.rows; currRow++)
    {
        for (int currCol = 0; currCol < PreImg.cols; currCol++)
            {
                temp = (int )PreImg.at<uchar>(currRow,currCol) - (int )PostImg.at<uchar>(currRow,currCol);
                MSE += pow(temp,2);
            }
    }
    MSE = MSE/((int ) PreImg.rows * (int ) PostImg.cols);
    PSNR = 10*log10(pow(255,2)/MSE);
    return PSNR;
}

/* Compression computation for all bands */
void* compress(void* args)
{
    thread_args* t_args = (thread_args*)args;
    for (int i = t_args->Y_blockstart; i<t_args->Y_blockend; i++)
    {
        FFT_8x8(&t_args->Y_blockarr[i]);
        QuantizeLuminance(&t_args->Y_blockarr[i]);
        
        block2bitstream16(&t_args->Y_blockarr[i],&t_args->Y_bitstream16[i*136]);
       
    }


    for (int i = t_args->C_blockstart; i<t_args->C_blockend; i++)
    {
        FFT_8x8(&t_args->Cb_blockarr[i]);
        QuantizeChrominance(&t_args->Cb_blockarr[i]);

        FFT_8x8(&t_args->Cr_blockarr[i]);
        QuantizeChrominance(&t_args->Cr_blockarr[i]);
        
        block2bitstream16(&t_args->Cb_blockarr[i],&t_args->Cb_bitstream16[i*136]);
        block2bitstream16(&t_args->Cr_blockarr[i],&t_args->Cr_bitstream16[i*136]);
    }
}

/* Decompression computation for all bands */
void* decompress(void* args)
{
    thread_args* t_args = (thread_args*)args;
    for (int i = t_args->Y_blockstart; i<t_args->Y_blockend; i++)
    {
        bitstream2block16(&t_args->Y_blockarr[i],&t_args->Y_bitstream16[i*136]);
        
        
        InvQuantizeLuminance(&t_args->Y_blockarr[i]);
        IFFT_8x8(&t_args->Y_blockarr[i]);
    }

    for (int i = t_args->C_blockstart; i<t_args->C_blockend; i++)
    {
        bitstream2block16(&t_args->Cb_blockarr[i],&t_args->Cb_bitstream16[i*136]);
        bitstream2block16(&t_args->Cr_blockarr[i],&t_args->Cr_bitstream16[i*136]);
        
        InvQuantizeChrominance(&t_args->Cb_blockarr[i]);
        IFFT_8x8(&t_args->Cb_blockarr[i]);

        InvQuantizeChrominance(&t_args->Cr_blockarr[i]);
        IFFT_8x8(&t_args->Cr_blockarr[i]);
    }
}