/*
Feature: Baseline matching

Funciton: use the 9x9 square in the middle of the image as a feature vector
*/
#include "feature.h"

using namespace cv;
using namespace std;

class BaselineMatchingFeature : public Feature
{
public:
    int getFeatureVector(Mat &image, vector<vector<float>> &output) override
    {
        int range = 9;

        // grab the 9x9 center pixel
        const int topLeft_row = (image.rows - range) / 2;
        const int topLeft_col = (image.cols - range) / 2;

        // obtain channel results
        cv::Vec3b *rowPointer = image.ptr<cv::Vec3b>(topLeft_row);

        vector<float> innerVector;

        for (int row = topLeft_row; row < topLeft_row + range; row++)
        {
            for (int col = topLeft_col; col < topLeft_col + range; col++)
            {
                for (int channel = 0; channel < 3; channel++)
                {
                    innerVector.push_back(rowPointer[col][channel]);
                }
            }
            rowPointer++;
        }

        output.push_back(innerVector);

        // sanity check, should be 81 of them (hardcodes)
        if (output[0].size() != range * range * 3)
        {
            printf("wrong result. There are %zu in the vector", output.size());
            return -1;
        }

        return 0;
    }

    float compare(vector<vector<float>> originalImgFeature, vector<vector<float>> compareImgFeature) override
    {
        if (originalImgFeature[0].size() != compareImgFeature[0].size())
        {
            printf("Error: Feature mismatch.");
            return -1;
        }

        int score = 0;

        for (int i = 0; i < originalImgFeature[0].size(); i++)
        {
            int diff = originalImgFeature[0][i] - compareImgFeature[0][i];
            score += diff * diff;
        }

        return score;
    }
};