/*
    Daniel Bi
    CS5530 Computer Vision -HW2

    Feature: Histogram matching
    Funciton: matching images with a single histogram
*/
#include "feature.h"

using namespace cv;
using namespace std;

// This class implements the `Feature` object which includes getting the feature vector and ways to compare the feature vector(s)
class HistogramMatchingFeature : public Feature
{
public:
    int getFeatureVector(Mat &image, vector<vector<float>> &output) override
    {
        std::vector<std::vector<float>> histResult;
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

        return 0;
    }

    int getHistogram(Mat &src, std::vector<std::vector<float>> &histogram)
    {

        const int histsize = 16;
        int max = 0;

        // std::vector<std::vector<float>> histogram(histsize, std::vector<float>(histsize, 0));

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
                histogram[rIndex][gIndex] += 1;

                float newValue = histogram[rIndex][gIndex];

                // conditional assignemnt
                // max = newValue > max ? newValue : max;
            }
        }

        // normalized the histogram
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

    float compare(vector<vector<float>>originalImgFeature, vector<vector<float>> compareImgFeature) override
    {
        double intersection = 0.0;
        for (int i = 0; i < originalImgFeature[0].size(); ++i)
        {
            intersection += std::min(originalImgFeature[0][i], compareImgFeature[0][i]); // Assuming hist1 and hist2 are histograms of two images
        }

        return intersection;
    }
};