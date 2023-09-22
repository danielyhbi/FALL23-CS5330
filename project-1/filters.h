#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
// #include <array>
// #include <math.h>
// #include <cmath>
// #include <filesystem>
#include <iostream>
#include <future>
// #include <chrono>
using namespace std;
using namespace cv;

int formatRGBValue(double number);

int matrixConvolution16bit(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel);

int matrixConvolution8bit(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel);

int matrixConvolution8bitAsyncHelper(int startRow, int endRowNoInclu, int startCol, int endColNoInclu, cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel);

int matrixConvolution8bitAsync(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel);

int matrixConvolution16bitAsyncHelper(int startRow, int endRowNoInclu, int startCol, int endColNoInclu, cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel);

int matrixConvolution16bitAsync(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &kernel);

int matrixMultiplication(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<double>> &transformer);

int blur5x5(cv::Mat &src, cv::Mat &dst);

int convertToGrayscale709(cv::Mat &src, cv::Mat &dst);

int concertToGreyScaleFast(cv::Mat &src, cv::Mat &dst);

int sobelX3x3(cv::Mat &src, cv::Mat &dst);

int sobelY3x3(cv::Mat &src, cv::Mat &dst);

int combineSobelXY(cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &dst);

int sobel3x3(cv::Mat &src, cv::Mat &dst);

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold);

int sharpen(cv::Mat &src, cv::Mat &dst);

int sepia(cv::Mat &src, cv::Mat &dst);