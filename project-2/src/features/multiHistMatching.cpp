/*
    Daniel Bi
    CS5530 Computer Vision -HW2

    Feature: Histogram matching
    Funciton: matching images with multiple histograms. (4) for each corners of the image.
*/
#include "feature.h"

using namespace cv;
using namespace std;

// This class implements the `Feature` object which includes getting the feature vector and ways to compare the feature vector(s)
class MultiHistMatchingFeature : public Feature
{
public:
    // vector will be output in separate histograms
    int getFeatureVector(Mat &image, vector<vector<float>> &output) override
    {
        std::vector<std::vector<std::vector<float>>> histResult;
        std::vector<float> innerVector;

        getHistograms(image, histResult);

        for (int histCount = 0; histCount < histResult.size(); histCount++)
        {
            std::vector<float> innerVector;

            for (int row = 0; row < histResult[histCount].size(); row++)
            {
                for (int col = 0; col < histResult[histCount][row].size(); col++)
                {
                    innerVector.push_back(histResult[histCount][row][col]);
                }
            }

            output.push_back(innerVector);
        }
        return 0;
    }

    int getHistograms(Mat &src, std::vector<std::vector<std::vector<float>>> &histograms)
    {
        // initialize the matrix
        vector<vector<vector<float>>> histogramsResult;

        // initialize each individual histogram
        histograms.push_back(getHistogramHelper(src, 0, src.rows / 2,
                                                0, src.cols / 2));

        histograms.push_back(getHistogramHelper(src, src.rows / 2, src.rows,
                                                0, src.cols / 2));

        histograms.push_back(getHistogramHelper(src, 0, src.rows / 2,
                                                src.cols / 2, src.cols));

        histograms.push_back(getHistogramHelper(src, src.rows / 2, src.rows,
                                                src.cols / 2, src.cols));

        histogramsResult = histograms;

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
            scores[vectorIndex] = intersection / scores.size(); 
        }

        float threshold = (float)(1.0 / scores.size()) * 0.25;

        // get average score
        for (double score : scores) {
            float boost = 0.0;

            // if (score > threshold) {
            //     //printf("score reached above threshold\n");
            //     boost = 0.05;
            // }

            //totalScore = min(totalScore + score + boost, 0.9999);
            totalScore += score;
        }

        return totalScore;
    }

private:
    // produce 1 histogram per specification. Takes in arguments on start/end row/col
    vector<vector<float>> getHistogramHelper(Mat &src, int startRow, int endRow, int startCol, int endCol)
    {

        const int histsize = 16;
        int max = 0;

        // initialize the matrix
        std::vector<std::vector<float>> histogram(histsize, std::vector<float>(histsize, 0));

        for (int i = startRow; i < endRow; i++)
        {
            cv::Vec3b *rowPointer = src.ptr<cv::Vec3b>(i);

            for (int j = startCol; j < endCol; j++)
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
        int totalPixel = (endCol - startCol) * (endRow - startRow);

        for (int row = 0; row < histsize; row++)
        {
            for (int col = 0; col < histsize; col++)
            {
                histogram[row][col] /= totalPixel;
            }
        }

        return histogram;
    }
};