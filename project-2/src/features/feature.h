#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cmath>
using namespace cv;
using namespace std;

/*
    Abstract class that defines a feature for image regocnition.
    `getFeatureVector` should implement a way to obtain a feature from an iamge
    `compare` should implement a way to compare the feature vectors from two different images with custom distance metrics
*/
class Feature {
    public:
        // each children class will define their own feature vector
        virtual int getFeatureVector(Mat &image, vector<vector<float>> &output) = 0;

        // compares two feature vector and find out the delta
        virtual float compare(vector<vector<float>> originalImgFeature, vector<vector<float>> compareImgFeature) = 0;
};