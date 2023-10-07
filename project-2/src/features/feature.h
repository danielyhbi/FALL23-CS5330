#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cmath>
using namespace cv;
using namespace std;


class Feature {
    public:
        // each children class will define their own feature vector
        virtual int getFeatureVector(Mat &image, vector<vector<float>> &output) = 0;

        // compares two feature vector and find out the delta
        virtual float compare(vector<vector<float>> originalImgFeature, vector<vector<float>> compareImgFeature) = 0;
};