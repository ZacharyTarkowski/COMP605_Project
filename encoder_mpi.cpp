#include <mpi.h>
#include<iostream>
#include "FFT.hpp"
#include "FFT.cpp"
#include "ImageProcessing.hpp"
#include "ImageProcessing.cpp"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <stdint.h>
using namespace std;
using namespace cv;



void mat2block(Mat* mat, xformBlock* block);
void block2mat(Mat* mat, xformBlock* block);
void getblockarray(Mat mat, xformBlock* blockarr);
void getmat(Mat mat, xformBlock* blockarr);

void block2bitstream(xformBlock *block, uint8 *arr );
void bitstream2block(xformBlock *block, uint8 *arr );

void block2bitstream16(xformBlock *block, int16_t *arr );
void bitstream2block16(xformBlock *block, int16_t *arr );

double PSNR(Mat PreImg, Mat PostImg);

int main(int argc, const char** argv) 
{

    /* Variable declarations */
    int my_rank;
    int num_process;

    /* MPI start */
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    /* OpenCV Mat objects for image I/O */
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
    double start, end;

    Mat different_Channels[3];


    //-------------------------------------------------------------------------------------------------//
    //Image Preprocessing in root
    //-------------------------------------------------------------------------------------------------//

    /* Root process does image I/O and pre-processing */
    if(my_rank==0)
    {
        /* OpenCV image read */
        img = imread(argv[1]);

        /* OpenCV convert to YCbCr */
        cvtColor(img,myImage_Converted, COLOR_RGB2YCrCb);

        /* OpenCV split into different color bands */
        split(myImage_Converted, different_Channels);
        Y = different_Channels[0];
        Mat Cr = different_Channels[1];
        Mat Cb = different_Channels[2];

        /* Number of luminance transform blocks needed */
        Y_numBlocks = ( myImage_Converted.rows / 8 ) * (  myImage_Converted.cols / 8  );
        Y_numRows = myImage_Converted.rows;
        Y_numCols = myImage_Converted.cols;

        Cr_420 = Mat(myImage_Converted.rows/2,myImage_Converted.cols/2, CV_8UC1, 0.0);
        Cb_420 = Mat(myImage_Converted.rows/2,myImage_Converted.cols/2, CV_8UC1, 0.0);

        /* Subsample chrominance */
        SubSample(Cr,Cr_420);
        SubSample(Cb,Cb_420);

        /* Below is padding to keep the subsampled array dimension lengths divisible by 8 */
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

        /* Number of blocks in subsampled xform block arrays */
        C_numBlocks = ( Cr_420.rows / 8 ) * (  Cr_420.cols / 8  );
        C_numRows = Cr_420.rows;
        C_numCols = Cr_420.cols;
    }
    
    /* Broadcast the total number of luminance and chrominance blocks */
    MPI_Bcast(&Y_numBlocks,1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&C_numBlocks,1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Individual processes number of blocks to operate on */
    int my_Y_numBlocks = (Y_numBlocks / num_process) + (my_rank !=num_process-1 ? 0 :  (Y_numBlocks % num_process));
    int my_C_numBlocks = (C_numBlocks / num_process) + (my_rank !=num_process-1 ? 0 :  (C_numBlocks % num_process));

    /* Allocate memory for process block operations */
    xformBlock* my_Y_blockArr = (xformBlock*)malloc(Y_numBlocks*sizeof(xformBlock));
    xformBlock* my_Cb_blockArr = (xformBlock*)malloc(C_numBlocks*sizeof(xformBlock));
    xformBlock* my_Cr_blockArr = (xformBlock*)malloc(C_numBlocks*sizeof(xformBlock));

    /* Memory allocation for bitstreams
       136 is the size of a xformblock object stored in 8 bit integers */
    uint8* my_Y_bitstream = (uint8*)malloc(Y_numBlocks*136);
    uint8* Y_bitstream = (uint8*)malloc(Y_numBlocks*136);

    uint8* my_Cr_bitstream = (uint8*)malloc(C_numBlocks*136);
    uint8* Cr_bitstream = (uint8*)malloc(C_numBlocks*136);

    uint8* my_Cb_bitstream = (uint8*)malloc(C_numBlocks*136);
    uint8* Cb_bitstream = (uint8*)malloc(C_numBlocks*136);

    xformBlock* Y_blockArr = (xformBlock*)malloc(Y_numBlocks*sizeof(xformBlock));
    xformBlock* Cb_blockArr = (xformBlock*)malloc(C_numBlocks*sizeof(xformBlock));
    xformBlock* Cr_blockArr = (xformBlock*)malloc(C_numBlocks*sizeof(xformBlock));

    /* Memory allocation for 16 bit bitstreams
       272 is the size of a xformblock object stored in 8 bit integers, using 16 bits for
       each real and imaginary value of a complex value */
    int16* my_Y_bitstream16 = (int16*)malloc(my_Y_numBlocks*272);
    int16* Y_bitstream16 = (int16*)malloc(Y_numBlocks*272);

    int16* my_Cr_bitstream16 = (int16*)malloc(my_C_numBlocks*272);
    int16* Cr_bitstream16 = (int16*)malloc(C_numBlocks*272);

    int16* my_Cb_bitstream16 = (int16*)malloc(my_C_numBlocks*272);
    int16* Cb_bitstream16 = (int16*)malloc(C_numBlocks*272);

    //-------------------------------------------------------------------------------------------------//
    // Send data bitstreams to child processes
    //-------------------------------------------------------------------------------------------------//

    /* Synchronize and take time */
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    /* Root process converts block array to bitstreams and sends to all processes */
    if(my_rank == 0)
    {
        
        int Y_nblocks_per_send;
        int C_nblocks_per_send;  

        /* Convert OpenCV mats to block array */
        getblockarray(Y,Y_blockArr);
        getblockarray(Cb_420,Cb_blockArr);
        getblockarray(Cr_420,Cr_blockArr);

        /* Convert block arrays into bitstream */
        for(int i = 0; i<Y_numBlocks; i++)
        {
            block2bitstream(&Y_blockArr[i],&Y_bitstream[i*136]);
        }

        for(int i = 0; i<C_numBlocks; i++)
        {
            block2bitstream(&Cr_blockArr[i],&Cr_bitstream[i*136]);
            block2bitstream(&Cb_blockArr[i],&Cb_bitstream[i*136]);
        }

        /* MPI send/recieves of bitstreams */
        for (int i = 1; i<num_process; i++)
        {
            /* Root sends bitstreams to all child processes */
            Y_nblocks_per_send = (i != num_process-1) ? (Y_numBlocks / num_process) : (Y_numBlocks / num_process) + (Y_numBlocks % num_process);
            C_nblocks_per_send = (i != num_process-1) ? (C_numBlocks / num_process) : (C_numBlocks / num_process) + (C_numBlocks % num_process);
            MPI_Send(&Y_bitstream[my_Y_numBlocks*i*136], Y_nblocks_per_send*136, MPI_INT8_T, i, 0, MPI_COMM_WORLD );
            MPI_Send(&Cr_bitstream[my_C_numBlocks*i*136], C_nblocks_per_send*136, MPI_INT8_T, i, 0, MPI_COMM_WORLD );
            MPI_Send(&Cb_bitstream[my_C_numBlocks*i*136], C_nblocks_per_send*136, MPI_INT8_T, i, 0, MPI_COMM_WORLD );
            
        }
    }
    else
    {
        /* Child processes recieve bitstreams from root */
        MPI_Recv(my_Y_bitstream, my_Y_numBlocks*136, MPI_INT8_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv(my_Cr_bitstream, my_C_numBlocks*136, MPI_INT8_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv(my_Cb_bitstream, my_C_numBlocks*136, MPI_INT8_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    }

    //-------------------------------------------------------------------------------------------------//
    // Image Processing in each process
    //-------------------------------------------------------------------------------------------------//
    
    /* Main Compression processing :
        Convert bitstream into a xform block
        FFT xform block
        Quantize xform block
        Convert xformblock into 16 bit bitstream */

    /* 16 bit bitstream is needed because FFT coefficient matrix can have signed values 
        exceeding 128 */    
    for (int i = 0; i<my_Y_numBlocks; i++)
    {
        bitstream2block(&my_Y_blockArr[i],&my_Y_bitstream[i*136]);
        FFT_8x8(&my_Y_blockArr[i]);
        QuantizeLuminance(&my_Y_blockArr[i]);
        
        block2bitstream16(&my_Y_blockArr[i],&my_Y_bitstream16[i*136]);
    }

    for (int i = 0; i<my_C_numBlocks; i++)
    {
        bitstream2block(&my_Cb_blockArr[i],&my_Cb_bitstream[i*136]);
        FFT_8x8(&my_Cb_blockArr[i]);
        QuantizeChrominance(&my_Cb_blockArr[i]);
        
        block2bitstream16(&my_Cb_blockArr[i],&my_Cb_bitstream16[i*136]);
    }

    for (int i = 0; i<my_C_numBlocks; i++)
    {
        bitstream2block(&my_Cr_blockArr[i],&my_Cr_bitstream[i*136]);
        FFT_8x8(&my_Cr_blockArr[i]);
        QuantizeChrominance(&my_Cr_blockArr[i]);

        block2bitstream16(&my_Cr_blockArr[i],&my_Cr_bitstream16[i*136]);
    }



    //-------------------------------------------------------------------------------------------------//
    // Send data to root to store
    //-------------------------------------------------------------------------------------------------//
    

    if(my_rank == 0)
    {
        int Y_nblocks_per_recv;
        int C_nblocks_per_recv;

        for (int i = 1; i<num_process; i++)
        {   
            
            Y_nblocks_per_recv = (i != num_process-1) ? (Y_numBlocks / num_process) : (Y_numBlocks / num_process) + (Y_numBlocks % num_process);
            C_nblocks_per_recv = (i != num_process-1) ? (C_numBlocks / num_process) : (C_numBlocks / num_process) + (C_numBlocks % num_process);
            MPI_Recv(&Y_bitstream16[my_Y_numBlocks*i*136], Y_nblocks_per_recv*136, MPI_INT16_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            MPI_Recv(&Cr_bitstream16[my_C_numBlocks*i*136], C_nblocks_per_recv*136, MPI_INT16_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            MPI_Recv(&Cb_bitstream16[my_C_numBlocks*i*136], C_nblocks_per_recv*136, MPI_INT16_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }
        /* Root takes time and reports. Implicit synchronization by above loop waiting till
        all processes send their compressed bitstream to root */
        end = MPI_Wtime();
        printf("Compress Time is : %lf\n\r",end-start);
    }
    else
    {
        
        MPI_Send(my_Y_bitstream16, my_Y_numBlocks*136, MPI_INT16_T, 0, 0, MPI_COMM_WORLD);
        MPI_Send(my_Cr_bitstream16, my_C_numBlocks*136, MPI_INT16_T, 0, 0, MPI_COMM_WORLD);
        MPI_Send(my_Cb_bitstream16, my_C_numBlocks*136, MPI_INT16_T, 0, 0, MPI_COMM_WORLD);
    }

    


    //-------------------------------------------------------------------------------------------------//
    //Store and read out in root
    //-------------------------------------------------------------------------------------------------//
        //This is where a compressed file could be written to and encoded.

    //-------------------------------------------------------------------------------------------------//
    // Send stored data
    //-------------------------------------------------------------------------------------------------//

    /* Synchronize and take time for decompression */
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    /* Root sends out bitstreams to all child processes for decompression*/
    if(my_rank == 0)
    {
        
        int Y_nblocks_per_send;
        int C_nblocks_per_send;

        for (int i = 1; i<num_process; i++)
        {
            Y_nblocks_per_send = (i != num_process-1) ? (Y_numBlocks / num_process) : (Y_numBlocks / num_process) + (Y_numBlocks % num_process);
            C_nblocks_per_send = (i != num_process-1) ? (C_numBlocks / num_process) : (C_numBlocks / num_process) + (C_numBlocks % num_process);
            MPI_Send(&Y_bitstream16[my_Y_numBlocks*i*136], Y_nblocks_per_send*136, MPI_INT16_T, i, 0, MPI_COMM_WORLD );
            MPI_Send(&Cr_bitstream16[my_C_numBlocks*i*136], C_nblocks_per_send*136, MPI_INT16_T, i, 0, MPI_COMM_WORLD );
            MPI_Send(&Cb_bitstream16[my_C_numBlocks*i*136], C_nblocks_per_send*136, MPI_INT16_T, i, 0, MPI_COMM_WORLD );
            
        }
    }
    else
    {
        MPI_Recv(my_Y_bitstream16, my_Y_numBlocks*136, MPI_INT16_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv(my_Cr_bitstream16, my_C_numBlocks*136, MPI_INT16_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv(my_Cb_bitstream16, my_C_numBlocks*136, MPI_INT16_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    }
    
    //-------------------------------------------------------------------------------------------------//
    // Image Processing to reconstruct image
    //-------------------------------------------------------------------------------------------------//

    /* Main Decompression processing :
        Convert 16 bit bitstream into a xform block
        Inverse Quantize xform block
        IFFT xform block
        Convert xformblock into bitstream */

    for (int i = 0; i<my_Y_numBlocks; i++)
    {
        bitstream2block16(&my_Y_blockArr[i],&my_Y_bitstream16[i*136]);
        
        InvQuantizeLuminance(&my_Y_blockArr[i]);
        IFFT_8x8(&my_Y_blockArr[i]);
        block2bitstream(&my_Y_blockArr[i],&my_Y_bitstream[i*136]);
    }

    for (int i = 0; i<my_C_numBlocks; i++)
    {
        bitstream2block16(&my_Cb_blockArr[i],&my_Cb_bitstream16[i*136]);
        
        InvQuantizeChrominance(&my_Cb_blockArr[i]);
        IFFT_8x8(&my_Cb_blockArr[i]);
        block2bitstream(&my_Cb_blockArr[i],&my_Cb_bitstream[i*136]);
    }

    for (int i = 0; i<my_C_numBlocks; i++)
    {
        bitstream2block16(&my_Cr_blockArr[i],&my_Cr_bitstream16[i*136]);

        InvQuantizeChrominance(&my_Cr_blockArr[i]);
        IFFT_8x8(&my_Cr_blockArr[i]);
        block2bitstream(&my_Cr_blockArr[i],&my_Cr_bitstream[i*136]);
    }

    //-------------------------------------------------------------------------------------------------//
    // Send all to root
    //-------------------------------------------------------------------------------------------------//

    /* Root recieves all decompressed bitstreams */
    if(my_rank == 0)
    {

        for (int i = 1; i<num_process; i++)
        {   
            
            int Y_nblocks_per_recv = (i != num_process-1) ? (Y_numBlocks / num_process) : (Y_numBlocks / num_process) + (Y_numBlocks % num_process);
            int C_nblocks_per_recv = (i != num_process-1) ? (C_numBlocks / num_process) : (C_numBlocks / num_process) + (C_numBlocks % num_process);
            MPI_Recv(&Y_bitstream[my_Y_numBlocks*i*136], Y_nblocks_per_recv*136, MPI_INT8_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            MPI_Recv(&Cr_bitstream[my_C_numBlocks*i*136], C_nblocks_per_recv*136, MPI_INT8_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            MPI_Recv(&Cb_bitstream[my_C_numBlocks*i*136], C_nblocks_per_recv*136, MPI_INT8_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }
        end = MPI_Wtime();
        printf("Decompress Time is : %lf\n\r",end-start);
    }
    else
    {
        
        MPI_Send(my_Y_bitstream, my_Y_numBlocks*136, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
        MPI_Send(my_Cr_bitstream, my_C_numBlocks*136, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
        MPI_Send(my_Cb_bitstream, my_C_numBlocks*136, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
        
    }
    
    //-------------------------------------------------------------------------------------------------//
    // Post Processing in root
    //-------------------------------------------------------------------------------------------------//

    if(my_rank==0)
    {

        for(int i = 0; i<Y_numBlocks; i++)
        {
            bitstream2block(&Y_blockArr[i],&Y_bitstream[i*136]);
        }

        for(int i = 0; i<C_numBlocks; i++)
        {
            bitstream2block(&Cr_blockArr[i],&Cr_bitstream[i*136]);
            bitstream2block(&Cb_blockArr[i],&Cb_bitstream[i*136]);
        }
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
        getmat(Y_compressed,Y_blockArr);
        getmat(Cb_420_compressed,Cb_blockArr);
        getmat(Cr_420_compressed,Cr_blockArr);


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
        //find difference of orignal image and new one
        absdiff(myImage_Converted,img_compressed,differenceimage);
        //save images for visual analysis
        imwrite("\Image_Compressed_MPI.png", img_compressed_rgb);
        //imwrite("\MyImage3.png", Cr_compressed);
        //imwrite("\MyImage4.png", Cb_compressed);
        //imwrite("\MyImage5.png", Y_compressed);
        imwrite("\Difference_Image_MPI.png", differenceimage);

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

        double PSNR_R = PSNR(R, R_compress);
        double PSNR_G = PSNR(G, G_compress);
        double PSNR_B = PSNR(B, B_compress);

        cout <<"PSNR_R "<< PSNR_R <<endl;
        cout <<"PSNR_G "<< PSNR_G <<endl;
        cout <<"PSNR_B "<< PSNR_B <<endl;
        
    }

    MPI_Finalize();


    free(my_Y_bitstream );
    free(Y_bitstream);

    free(my_Cr_bitstream );
    free(Cr_bitstream );

    free(my_Cb_bitstream );
    free(Cb_bitstream );

    free(my_Y_bitstream16 );
    free(Y_bitstream16 );

    free(my_Cr_bitstream16 );
    free(Cr_bitstream16 );

    free(my_Cb_bitstream16 );
    free(Cb_bitstream16 );

    free(my_Y_blockArr);
    free(my_Cb_blockArr);
    free(my_Cr_blockArr);

    free(Y_blockArr);
    free(Cb_blockArr);
    free(Cr_blockArr);



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
