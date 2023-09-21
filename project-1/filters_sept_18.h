#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <array>
#include <math.h>
#include <cmath>
#include <filesystem>
#include "filterMatrices.h"
using namespace std;
using namespace cv;

int formatRGBValue(double number)
{
    if (number < 0)
    {
        return 0;
    }

    if (number > 255)
    {
        return 255;
    }

    return int(number);
}

int matrixConvolutionNew(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel)
{

    for (int row = 0; row < src.rows; row++)
    {
        cv::Vec3b *rowPointer = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                int rowCenter = kernel.size() / 2;
                int colCenter = kernel[0].size() / 2;
                double sum = 0;

                // iterate through the kernel
                for (int kernalRow = -rowCenter; kernalRow <= rowCenter; kernalRow++)
                {
                    for (int kernalCol = -colCenter; kernalCol <= colCenter; kernalCol++)
                    {

                        int currentConvRow = row + kernalRow;
                        int currentConvCol = col + kernalCol;

                        // check if index is out of bounds
                        if (currentConvRow < 0 || currentConvRow >= src.rows || currentConvCol < 0 || currentConvCol >= src.cols)
                        {
                            continue;
                        }

                        int valFromSrc = src.at<cv::Vec3b>(currentConvRow, currentConvCol)[channel];
                        double valFromKernel = kernel[kernalRow + rowCenter][kernalCol + colCenter];

                        sum += valFromSrc * valFromKernel;
                    }
                }

                rowPointer[col][channel] = formatRGBValue(sum);
            }
        }
    }

    return 0;
}

int blur5x5(cv::Mat &src, cv::Mat &dst)
{
    dst.create(src.rows, src.cols, src.type());
    cv::Mat pass1 = cv::Mat::zeros(src.size(), src.type());
    matrixConvolutionNew(src, pass1, GAUSS_BLUR_KERNEL_1DX());
    matrixConvolutionNew(pass1, dst, GAUSS_BLUR_KERNEL_1DY());

    return 0;
}

int convertToGrayscale709(cv::Mat &src, cv::Mat &dst)
{

    // recreate the destination matrix
    dst.create(src.rows, src.cols, src.type());

    double bCoeff = 0.0722;
    double gCoeff = 0.7152;
    double rCoeff = 0.2126;

    for (int row = 0; row < src.rows; row++)
    {
        cv::Vec3b *rowPointer = dst.ptr<cv::Vec3b>(row);
        cv::Vec3b *rowPointerSrc = src.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++)
        {
            rowPointer[col] = formatRGBValue(
                rowPointerSrc[col][0] * bCoeff + rowPointerSrc[col][1] * gCoeff + rowPointerSrc[col][2] * rCoeff);
        }
    }

    return 0;
}

int concertToGreyScaleFast(cv::Mat &src, cv::Mat &dst)
{

    // recreate the destination matrix
    // dst.create(src.rows, src.cols, src.type());
    dst = cv::Mat::zeros(src.size(), src.type());
    // 8UC1 = 8-bit unsigned char with 1 channel (greyscale image)
    // 8UC3 = 8-bit unsigned chat with 3 channels (BGR/RGB image)
    // 16SC1
    // 32FC3

    double bCoeff = 0.0722;
    double gCoeff = 0.7152;
    double rCoeff = 0.2126;

    for (int row = 0; row < src.rows; row++)
    {
        // get pointer to the start row 1
        cv::Vec3b *rowPointer = src.ptr<cv::Vec3b>(row);
        cv::Vec3b *rowPointerDest = dst.ptr<cv::Vec3b>(row);

        /*
        // treat the image as an array of bytes
        uchar *uPointer = src.ptr<uchar>(row);

        // memory address just after the last pixel
        uchar *end = uPointer + src.cols * 3;

        while (uPointer < end) {
            *uPointer = 225 - *uPointer;
            uPointer++;
        }
        */

        for (int col = 0; col < src.cols; col++)
        {
            int greyScale = formatRGBValue(rowPointer[col][0] * bCoeff + rowPointer[col][1] * gCoeff + rowPointer[col][2] * rCoeff);

            rowPointerDest[col][0] = greyScale;
            rowPointerDest[col][1] = greyScale;
            rowPointerDest[col][2] = greyScale;

            // rowPointer[col][0] = 255 - rowPointer[col][0];
            // rowPointer[col][1] = 255 - rowPointer[col][1];
            // rowPointer[col][2] = 255 - rowPointer[col][2];

            // with uchar
            // uPointer[col] = 255 - uPointer[col];
        }
    }

    return 0;
}

int sobelX3x31(cv::Mat &src, cv::Mat &dst)
{

    dst = cv::Mat::zeros(src.size(), src.type());

    matrixConvolutionNew(src, dst, SOBEL_KERNEL_2DX());

    return 0;
}

int sobelPlain(cv::Mat &src, cv::Mat &dst) {

    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    for (int row = 0; row < src.rows; row++)
    {
        const cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(row - 1);
        const cv::Vec3b *rptr = src.ptr<cv::Vec3b>(row);
        const cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(row + 1);

        // destiation pointer
        cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(row);

        for (int j = 1; j < src.cols - 1; j++)
        {

            // over all color channels
            for (int c = 0; c < 3; c++)
            {
                dptr[j][c] = (-1 * rptrm1[j - 1][c] + 1 * rptrm1[j + 1][c] +
                              -2 * rptr[j - 1][c] + 2 * rptr[j + 1][c] +
                              -1 * rptrp1[j - 1][c] + 1 * rptrp1[j + 1][c]) /
                             4;
            }
        }
    }

    return 0;
}

int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    int count = 0;

    if (count == 0) {
        printf("Image type: %d \n",src.type());
        count++;
    }
    cv::Mat pass1 = cv::Mat::zeros(src.size(), src.type());
    dst = cv::Mat::zeros(src.size(), src.type());

    // cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    //  cv::Mat sobel = cv::Mat::zeros(src.size(), src.type());

    matrixConvolutionNew(src, pass1, SOBEL_KERNEL_1DX_H());
    matrixConvolutionNew(pass1, dst, SOBEL_KERNEL_1DX_V());
    // concertToGreyScaleFast(sobel, dst);

    return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    cv::Mat pass1 = cv::Mat::zeros(src.size(), src.type());
    dst = cv::Mat::zeros(src.size(), src.type());

    matrixConvolutionNew(src, pass1, SOBEL_KERNEL_1DY_V());
    matrixConvolutionNew(pass1, dst, SOBEL_KERNEL_1DX_H());

    return 0;
}

int combineSobelXY(cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &dst)
{
    for (int row = 0; row < sobelX.rows; row++)
    {
        cv::Vec3b *rowPointerSobelX = sobelX.ptr<cv::Vec3b>(row);
        cv::Vec3b *rowPointerSobelY = sobelY.ptr<cv::Vec3b>(row);
        cv::Vec3b *rowPointerDst = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < sobelX.cols; col++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                int x = rowPointerSobelX[col][channel];
                int y = rowPointerSobelY[col][channel];

                rowPointerDst[col][channel] = formatRGBValue(sqrt(x * x + y * y));
            }
        }
    }

    return 0;
}

int sobel3x3(cv::Mat &src, cv::Mat &dst)
{
    dst = cv::Mat::zeros(src.size(), src.type());
    cv::Mat sobelXpass1 = cv::Mat::zeros(src.size(), src.type());
    cv::Mat sobelXpass2 = cv::Mat::zeros(src.size(), src.type());
    cv::Mat sobelYpass1 = cv::Mat::zeros(src.size(), src.type());
    cv::Mat sobelYpass2 = cv::Mat::zeros(src.size(), src.type());

    matrixConvolutionNew(src, sobelXpass1, SOBEL_KERNEL_1DX_H());
    matrixConvolutionNew(sobelXpass1, sobelXpass2, SOBEL_KERNEL_1DX_V());
    matrixConvolutionNew(src, sobelYpass1, SOBEL_KERNEL_1DY_V());
    matrixConvolutionNew(sobelYpass1, sobelYpass2, SOBEL_KERNEL_1DX_H());

    combineSobelXY(sobelXpass2, sobelYpass2, dst);
    return 0;
}

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels)
{

    blur5x5(src, dst);

    const int bucketCount = 255 / levels;

    for (int row = 0; row < dst.rows; row++)
    {
        cv::Vec3b *rowPointerDst = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < dst.cols; col++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                int bucketIndex = rowPointerDst[col][channel] / bucketCount;

                rowPointerDst[col][channel] = bucketIndex * bucketCount;
            }
        }
    }

    return 0;
}

int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold)
{

    dst = cv::Mat::zeros(src.size(), src.type());
    // apply greyscale
    cv::Mat greyscaled;
    convertToGrayscale709(src, greyscaled);

    // sobel edge detect
    cv::Mat edgeDetected;
    sobel3x3(greyscaled, edgeDetected);

    // blur and quantize
    cv::Mat blurAndQuantize;
    blurQuantize(src, dst, levels);

    // printf("combine filters ");

    // combine them together
    for (int row = 0; row < dst.rows; row++)
    {
        cv::Vec3b *rowPointerEdge = edgeDetected.ptr<cv::Vec3b>(row);
        // cv::Vec3b *rowPointerBlur = blurAndQuantize.ptr<cv::Vec3b>(row);
        cv::Vec3b *rowPointerDst = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < dst.cols; col++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                if (rowPointerEdge[col][channel] > magThreshold)
                {

                    rowPointerDst[col][channel] = 0;
                }
            }
        }
    }

    return 0;
}

int sharpen(cv::Mat &src, cv::Mat &dst)
{

    dst = cv::Mat::zeros(src.size(), src.type());

    matrixConvolutionNew(src, dst, SHAPREN_KERNEL_5X5());

    return 0;
}