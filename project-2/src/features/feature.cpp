// // #include <opencv2/core.hpp>
// // #include <opencv2/imgcodecs.hpp>
// // #include <opencv2/highgui.hpp>
// // #include <opencv2/opencv.hpp>
#include "feature.h"
// // #include <string>
// // #include <vector>
// // #include <filesystem>
// using namespace cv;
// using namespace std;

// class Feature {
//     public:
//         virtual int getFeatureVector(Mat &image, vector<int> &output) = 0;
//         virtual int compare(vector<int> originalImgFeature, vector<int> compareImgFeature) = 0;
// };
// int Feature::compare(vector<float> originalImgFeature, vector<float> compareImgFeature) {
//     if (originalImgFeature.size() != compareImgFeature.size()) {
//         printf("Error: Feature mismatch.");
//         return -1;
//     }

//     int score = 0;

//     for (int i = 0; i < originalImgFeature.size(); i++) {
//         int diff = originalImgFeature[i] - compareImgFeature[i];
//         score += diff * diff;
//     }

//     return score;
// }