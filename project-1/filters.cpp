#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <array>
#include <math.h>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <future>
#include <chrono>
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

int matrixConvolution16bit(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel)
{
    if (src.type() != dst.type())
    {
        printf("src and dst not the same type. src is: %d, and dst is: %d.\n Will need 16 bits to work", src.type(), dst.type());
        dst = cv::Mat::zeros(src.size(), src.type());
        return -1;
    }

    for (int row = 0; row < src.rows; row++)
    {
        cv::Vec3s *rowPointer = dst.ptr<cv::Vec3s>(row);

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

                        int valFromSrc = src.at<cv::Vec3s>(currentConvRow, currentConvCol)[channel];
                        double valFromKernel = kernel[kernalRow + rowCenter][kernalCol + colCenter];

                        sum += valFromSrc * valFromKernel;
                    }
                }

                rowPointer[col][channel] = sum;
            }
        }
    }

    return 0;
}

int matrixConvolution8bit(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel)
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

int matrixConvolution8bitAsyncHelper(int startRow, int endRowNoInclu, int startCol, int endColNoInclu, cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel)
{

    for (int row = startRow; row < endRowNoInclu; row++)
    {
        cv::Vec3b *rowPointer = dst.ptr<cv::Vec3b>(row);

        for (int col = startCol; col < endColNoInclu; col++)
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

int matrixConvolution8bitAsync(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel)
{
    int halfCol = (src.cols / 2) + 1;
    int halfRow = (src.rows / 2) + 1;
    int rowCount = src.rows;
    int colCount = src.cols;

    auto convolutionTask1 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel]()
    {
        matrixConvolution8bitAsyncHelper(0, halfRow, 0, halfCol, src, dst, kernel);
    };

    auto convolutionTask2 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel]()
    {
        matrixConvolution8bitAsyncHelper(0, halfRow, halfCol, colCount, src, dst, kernel);
    };

    auto convolutionTask3 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel]()
    {
        matrixConvolution8bitAsyncHelper(halfRow, rowCount, 0, halfCol, src, dst, kernel);
    };

    auto convolutionTask4 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel]()
    {
        matrixConvolution8bitAsyncHelper(halfRow, rowCount, halfCol, colCount, src, dst, kernel);
    };

    std::future<void> future1 = std::async(std::launch::async, convolutionTask1);
    std::future<void> future2 = std::async(std::launch::async, convolutionTask2);
    std::future<void> future3 = std::async(std::launch::async, convolutionTask3);
    std::future<void> future4 = std::async(std::launch::async, convolutionTask4);

    // Wait for tasks to complete
    future1.get();
    future2.get();
    future3.get();
    future4.get();

    return 0;
}

int matrixConvolution16bitAsyncHelper(int startRow, int endRowNoInclu, int startCol, int endColNoInclu, cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel)
{

    for (int row = startRow; row < endRowNoInclu; row++)
    {
        cv::Vec3s *rowPointer = dst.ptr<cv::Vec3s>(row);

        for (int col = startCol; col < endColNoInclu; col++)
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

                        int valFromSrc = src.at<cv::Vec3s>(currentConvRow, currentConvCol)[channel];
                        double valFromKernel = kernel[kernalRow + rowCenter][kernalCol + colCenter];

                        sum += valFromSrc * valFromKernel;
                    }
                }

                rowPointer[col][channel] = sum;
            }
        }
    }

    return 0;
}

int matrixConvolution16bitAsync(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel)
{
    int halfCol = (src.cols / 2) + 1;
    int halfRow = (src.rows / 2) + 1;
    int rowCount = src.rows;
    int colCount = src.cols;

    auto convolutionTask1 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel]()
    {
        matrixConvolution16bitAsyncHelper(0, halfRow, 0, halfCol, src, dst, kernel);
    };

    auto convolutionTask2 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel]()
    {
        matrixConvolution16bitAsyncHelper(0, halfRow, halfCol, colCount, src, dst, kernel);
    };

    auto convolutionTask3 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel]()
    {
        matrixConvolution16bitAsyncHelper(halfRow, rowCount, 0, halfCol, src, dst, kernel);
    };

    auto convolutionTask4 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel]()
    {
        matrixConvolution16bitAsyncHelper(halfRow, rowCount, halfCol, colCount, src, dst, kernel);
    };

    std::future<void> future1 = std::async(std::launch::async, convolutionTask1);
    std::future<void> future2 = std::async(std::launch::async, convolutionTask2);
    std::future<void> future3 = std::async(std::launch::async, convolutionTask3);
    std::future<void> future4 = std::async(std::launch::async, convolutionTask4);

    // Wait for tasks to complete
    future1.get();
    future2.get();
    future3.get();
    future4.get();

    return 0;
}

int matrixMultiplication(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &transformer)
{
    for (int row = 0; row < src.rows; row++)
    {
        cv::Vec3b *rowPointer = src.ptr<cv::Vec3b>(row);
        cv::Vec3b *rowPointerDst = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++)
        {
            // note for openCV it is BGR
            int b = rowPointer[col][0];
            int g = rowPointer[col][1];
            int r = rowPointer[col][2];

            // i'm using the transformer for rgb
            rowPointerDst[col][2] = formatRGBValue(transformer[0][0] * r + transformer[0][2] * g + transformer[0][3] * b);
            rowPointerDst[col][1] = formatRGBValue(transformer[1][0] * r + transformer[1][2] * g + transformer[1][3] * b);
            rowPointerDst[col][0] = formatRGBValue(transformer[2][0] * r + transformer[2][2] * g + transformer[2][3] * b);
        }
    }
    return 0;
}

int blur5x5(cv::Mat &src, cv::Mat &dst)
{
    dst.create(src.rows, src.cols, src.type());

    cv::Mat pass1 = cv::Mat(src.size(), src.type(), Scalar(0, 0, 0));
    dst = cv::Mat(src.size(), src.type(), Scalar(0, 0, 0));

    matrixConvolution8bitAsync(src, pass1, GAUSS_BLUR_KERNEL_1DX);
    matrixConvolution8bitAsync(pass1, dst, GAUSS_BLUR_KERNEL_1DY);

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

int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    dst = cv::Mat(src.size(), src.type(), Scalar(0, 0, 0));
    cv::Mat src16 = cv::Mat(src.size(), CV_16SC3, Scalar(0, 0, 0));
    cv::Mat pass1 = cv::Mat(src.size(), CV_16SC3, Scalar(0, 0, 0));
    cv::Mat dst16 = cv::Mat(src.size(), CV_16SC3, Scalar(0, 0, 0));

    src.convertTo(src16, CV_16SC3);

    matrixConvolution16bitAsync(src16, pass1, SOBEL_KERNEL_1DX_H);
    matrixConvolution16bitAsync(pass1, dst16, SOBEL_KERNEL_1DX_V);

    cv::convertScaleAbs(dst16, dst);

    return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    dst = cv::Mat(src.size(), src.type(), Scalar(0, 0, 0));
    cv::Mat src16 = cv::Mat(src.size(), CV_16SC3, Scalar(0, 0, 0));
    cv::Mat pass1 = cv::Mat(src.size(), CV_16SC3, Scalar(0, 0, 0));
    cv::Mat dst16 = cv::Mat(src.size(), CV_16SC3, Scalar(0, 0, 0));

    src.convertTo(src16, CV_16SC3);

    matrixConvolution16bitAsync(src16, pass1, SOBEL_KERNEL_1DY_V);
    matrixConvolution16bitAsync(pass1, dst16, SOBEL_KERNEL_1DY_H);

    cv::convertScaleAbs(dst16, dst);

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
    cv::Mat sobelX = cv::Mat::zeros(src.size(), src.type());
    cv::Mat sobelY = cv::Mat::zeros(src.size(), src.type());

    sobelX3x3(src, sobelX);
    sobelY3x3(src, sobelY);

    combineSobelXY(sobelX, sobelY, dst);
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
    // sobel edge detect
    cv::Mat edgeDetected;
    sobel3x3(src, edgeDetected);

    // blur and quantize
    cv::Mat blurAndQuantize;
    blurQuantize(src, dst, levels);

    // combine them together
    for (int row = 0; row < dst.rows; row++)
    {
        cv::Vec3b *rowPointerEdge = edgeDetected.ptr<cv::Vec3b>(row);
        cv::Vec3b *rowPointerDst = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < dst.cols; col++)
        {
            bool outlined = false;

            for (int channel = 0; channel < 3; channel++)
            {
                if (outlined == false && rowPointerEdge[col][channel] > magThreshold)
                {
                    rowPointerDst[col][0] = 0;
                    rowPointerDst[col][1] = 0;
                    rowPointerDst[col][2] = 0;
                    outlined = true;
                }
                else
                {
                    channel++;
                }
            }
        }
    }

    return 0;
}

int sharpen(cv::Mat &src, cv::Mat &dst)
{
    printf("image type: %d", src.type());
    dst = cv::Mat::zeros(src.size(), src.type());

    matrixConvolution8bitAsync(src, dst, SHAPREN_KERNEL_5X5);
    return 0;
}

int sepia(cv::Mat &src, cv::Mat &dst)
{
    dst = cv::Mat::zeros(src.size(), src.type());
    matrixMultiplication(src, dst, SEPIA_TRANSFORM_MATRIX);
    return 0;
}