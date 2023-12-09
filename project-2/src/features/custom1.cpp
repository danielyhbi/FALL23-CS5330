/*
    Daniel Bi
    CS5530 Computer Vision -HW2

    Feature: Histogram matching - Find Grass lawn
*/
#include <future>
#include "feature.h"

using namespace cv;
using namespace std;

// This class implements the `Feature` object which includes getting the feature vector and ways to compare the feature vector(s)
class FindGrassMatching : public Feature
{
public:
    // the feature vector includes calcuating image histogram and sobel edge magnitude/orientation for texture
    int getFeatureVector(Mat &image, vector<vector<float>> &output) override
    {
        std::vector<std::vector<float>> histResult;
        std::vector<std::vector<float>> textureResult;
        std::vector<float> innerVector;

        getHistogram(image, histResult);

        for (int i = 0; i < histResult.size(); i++)
        {
            for (int j = 0; j < histResult[0].size(); j++)
            {
                innerVector.push_back(histResult[i][j]);
            }
        }

        output.push_back(innerVector);

        innerVector.clear();

        getSobelHistogram(image, textureResult);

        for (int i = 0; i < textureResult.size(); i++)
        {
            for (int j = 0; j < textureResult[0].size(); j++)
            {
                innerVector.push_back(textureResult[i][j]);
            }
        }

        output.push_back(innerVector);

        return 0;
    }

    int getHistogram(Mat &src, std::vector<std::vector<float>> &histogram)
    {

        const int histsize = 64;
        int scale = round(255 / histsize + 0.5);
        int max = 0;

        // initialize the matrix
        histogram = std::vector<std::vector<float>>(histsize, std::vector<float>(histsize, 0));

        for (int i = 0; i < src.rows; i++)
        {
            cv::Vec3b *rowPointer = src.ptr<cv::Vec3b>(i);

            for (int j = 0; j < src.cols; j++)
            {
                // grab the data
                float B = rowPointer[j][0];
                float G = rowPointer[j][1];
                float R = rowPointer[j][2];

                // compute the rg standard chromaticity
                float diviser = R + G + B;
                diviser = diviser > 0.0 ? diviser : 1.0;
                float r = R / (R + G + B);
                float g = G / (R + G + B);

                // compute indexes, r, g are in [0, 1]
                int rIndex = (int)(r * (histsize - 1) + 0.5); // rounds to the neatest value
                int gIndex = (int)(g * (histsize - 1) + 0.5);

                // increment the histogram
                histogram[rIndex][gIndex]++;

                float newValue = histogram[rIndex][gIndex];
                // conditional assignemnt
                max = newValue > max ? newValue : max;
            }
        }

        // normalized the histogram
        float sum = 0.0;
        for (const vector<float> &row : histogram)
        {
            for (float value : row)
            {
                sum += value;
            }
        }

        int totalPixel = src.rows * src.cols;

        for (int row = 0; row < histsize; row++)
        {
            for (int col = 0; col < histsize; col++)
            {
                histogram[row][col] /= totalPixel;
            }
        }
        return 0;
    }

    int getSobelHistogram(Mat &src, std::vector<std::vector<float>> &histogram)
    {

        Mat grayscaleImg, sobelX, sobelY, sobelXY, sobelPolar;

        const int histsize = 64;
        const int polarSize = 90;
        int max = 0;

        cv::Mat gradientMagnitude, gradientOrientation;
        convertToGrayscale(src, grayscaleImg);
        sobel3x3(grayscaleImg, gradientMagnitude, gradientOrientation);
        // extract the hist matrix

        histogram = std::vector<std::vector<float>>(histsize, std::vector<float>(polarSize, 0));

        for (int i = gradientMagnitude.rows / 2; i < gradientMagnitude.rows; i++)
        {
            cv::Vec3b *rowPointer = gradientMagnitude.ptr<cv::Vec3b>(i);
            cv::Vec3b *rowPointerO = gradientOrientation.ptr<cv::Vec3b>(i);

            for (int j = gradientMagnitude.cols / 2; j < gradientMagnitude.cols; j++)
            {
                // grab the data
                int test = rowPointer[j][0];
                // int scae = ceil(255 / histsize + 0.5);
                int scale = round(255 / histsize + 0.5);
                int B = rowPointer[j][0] / scale;
                if (B < 0)
                {
                    printf("Negative val detected. \n");
                }
                // Convert orientation from radians to degrees and ensure it's positive
                int orientation = rowPointerO[j][0] / (360 / polarSize) - 0.5;
                // increment the histogram
                histogram[B][orientation]++;

                float newValue = histogram[B][orientation];

                // conditional assignemnt
                max = newValue > max ? newValue : max;
            }
        }

        float sum = 0.0;
        for (const vector<float> &row : histogram)
        {
            for (float value : row)
            {
                sum += value;
            }
        }

        for (int row = 0; row < histsize; row++)
        {
            for (int col = 0; col < histsize; col++)
            {
                histogram[row][col] /= sum;
            }
        }

        return 0;
    }

    float compare(vector<vector<float>> originalImgFeature, vector<vector<float>> compareImgFeature) override
    {
        double totalScore = 0.0;

        vector<double> scores(originalImgFeature.size());

        // for now. each vector are weighted equally
        for (int vectorIndex = 0; vectorIndex < originalImgFeature.size(); vectorIndex++)
        {
            double intersection = 0.0;
            for (int i = 0; i < originalImgFeature[vectorIndex].size(); ++i)
            {
                intersection += std::min(originalImgFeature[vectorIndex][i], compareImgFeature[vectorIndex][i]); // Assuming hist1 and hist2 are histograms of two images
            }
            scores[vectorIndex] = intersection;
        }

        totalScore = 0.5 * scores[0] + 0.5 * scores[1];

        return totalScore;
    }

// private helper method obtained from hw1
private:
    // normalized from {1, 2, 1}
    const std::vector<std::vector<double>> SOBEL_KERNEL_1DX_V = {{0.25}, {0.5}, {0.25}};

    // normalized from {1, 0, -1}
    const std::vector<std::vector<double>> SOBEL_KERNEL_1DX_H = {{1, 0, -1}};

    const std::vector<std::vector<double>> SOBEL_KERNEL_1DY_V = {{1}, {0}, {-1}};

    const std::vector<std::vector<double>> SOBEL_KERNEL_1DY_H = {{0.25, 0.5, 0.25}};

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

    int getSobelOrientation(Mat &sobelX, Mat &sobelY, Mat &mag)
    {
        mag.create(sobelX.rows, sobelX.cols, sobelX.type());

        for (int row = 0; row < sobelX.rows; row++)
        {

            Vec3b *rowPointerX = sobelX.ptr<Vec3b>(row);
            Vec3b *rowPointerY = sobelY.ptr<Vec3b>(row);
            Vec3b *rowPointerM = mag.ptr<Vec3b>(row);

            for (int col = 0; col < sobelX.cols; col++)
            {

                int intensityX = rowPointerX[col][0];
                int intensityX1 = rowPointerX[col][1];
                int intensityY = rowPointerY[col][0];

                float orientation = 0.0;
                if (intensityX != 0)
                {
                    orientation = atan2(intensityY, intensityX);
                }
                else
                {
                    orientation = 1.57;
                }

                orientation = (orientation * 180) / M_PI;
                orientation = fmodf((orientation + 360), 360);

                rowPointerM[col] = orientation;
            }
        }

        return 0;

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

        auto convolutionTask1 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel, this]()
        {
            matrixConvolution16bitAsyncHelper(0, halfRow, 0, halfCol, src, dst, kernel);
        };

        auto convolutionTask2 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel, this]()
        {
            matrixConvolution16bitAsyncHelper(0, halfRow, halfCol, colCount, src, dst, kernel);
        };

        auto convolutionTask3 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel, this]()
        {
            matrixConvolution16bitAsyncHelper(halfRow, rowCount, 0, halfCol, src, dst, kernel);
        };

        auto convolutionTask4 = [halfRow, rowCount, &halfCol, &colCount, &src, &dst, &kernel, this]()
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
                for (int channel = 0; channel < 1; channel++)
                {
                    int x = rowPointerSobelX[col][channel];
                    int y = rowPointerSobelY[col][channel];

                    int magnitude = sqrt(x * x + y * y);

                    // if (magnitude < 80) {
                    //     //printf("ignoring value to black: %d\n", magnitude);
                    //     magnitude = 0.0;
                    // }

                    rowPointerDst[col][channel] = formatRGBValue(magnitude);
                    rowPointerDst[col][1] = formatRGBValue(magnitude);
                    rowPointerDst[col][2] = formatRGBValue(magnitude);
                }
            }
        }

        return 0;
    }

    int combineSobelYXOrientation(cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &dst)
    {
        for (int row = 0; row < sobelX.rows; row++)
        {
            cv::Vec3b *rowPointerSobelX = sobelX.ptr<cv::Vec3b>(row);
            cv::Vec3b *rowPointerSobelY = sobelY.ptr<cv::Vec3b>(row);
            cv::Vec3b *rowPointerDst = dst.ptr<cv::Vec3b>(row);

            for (int col = 0; col < sobelX.cols; col++)
            {
                for (int channel = 0; channel < 1; channel++)
                {
                    int x = rowPointerSobelX[col][channel];
                    int y = rowPointerSobelY[col][channel];

                    float orientation = 0.0;
                    if (x != 0)
                    {
                        orientation = atan2(x, y);
                    }
                    else
                    {
                        orientation = 1.57;
                    }

                    orientation = (orientation * 180) / M_PI;
                    orientation = fmodf((orientation + 360), 360);

                    rowPointerDst[col][channel] = orientation;
                    rowPointerDst[col][2] = orientation;
                    rowPointerDst[col][1] = orientation;
                }
            }
        }

        return 0;
    }

    int sobel3x3(cv::Mat &src, cv::Mat &dst, cv::Mat &orientation)
    {
        dst = cv::Mat::zeros(src.size(), src.type());
        orientation = cv::Mat::zeros(src.size(), src.type());
        cv::Mat sobelX = cv::Mat::zeros(src.size(), src.type());
        cv::Mat sobelY = cv::Mat::zeros(src.size(), src.type());

        sobelX3x3(src, sobelX);
        sobelY3x3(src, sobelY);

        combineSobelXY(sobelX, sobelY, dst);
        combineSobelYXOrientation(sobelX, sobelY, orientation);
        return 0;
    }

    int convertToGrayscale(cv::Mat &src, cv::Mat &dst)
    {
        dst = cv::Mat::zeros(src.size(), src.type());

        double bCoeff = 0.0722;
        double gCoeff = 0.7152;
        double rCoeff = 0.2126;

        for (int row = 0; row < src.rows; row++)
        {
            // get pointer to the start row 1
            cv::Vec3b *rowPointer = src.ptr<cv::Vec3b>(row);
            cv::Vec3b *rowPointerDest = dst.ptr<cv::Vec3b>(row);

            for (int col = 0; col < src.cols; col++)
            {
                int greyScale = formatRGBValue(rowPointer[col][0] * bCoeff + rowPointer[col][1] * gCoeff + rowPointer[col][2] * rCoeff);

                rowPointerDest[col][0] = greyScale;
                rowPointerDest[col][1] = greyScale;
                rowPointerDest[col][2] = greyScale;
            }
        }

        return 0;
    }
};