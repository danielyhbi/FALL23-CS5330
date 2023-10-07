/*
    This is the main entry point of the code.
    Given in image. it will find similar images within the database
*/

#include <string>
#include <vector>
#include <filesystem>
#include <map>

#include "feature.h"
#include "baseline_matching.cpp"
#include "histogram_matching.cpp"
#include "multiHistMatching.cpp"
#include "multiHistTextureMatch.cpp"
#include "custom1.cpp"
#include "custom2.cpp"
#include "custom3_roads.cpp"
#include "csv_util.h"
#include "../src/util/readfiles.cpp"

using namespace cv;
using namespace std;

int main()
{

    string image_path = "/Users/danielbi/git-repo/FALL23-CS5330/project-2/examples/olympus/pic.0607.jpg";

    Mat img = imread(image_path, IMREAD_COLOR);
    int imgSize = img.rows * img.cols;

    std::vector<string> fileList;

    // BaselineMatchingFeature baselineMatching; //1016

    // HistogramMatchingFeature histogramMatching;
    // MultiHistMatchingFeature histogramMatching;
    // TextureHistogramMatchingFeature histogramMatching;
    // FindRoadMatching histogramMatching;  //0824
    // FindGrassMatching histogramMatching; //0413
    FindFaceMatching histogramMatching; // 0607

    std::map<float, string, std::greater<float>> simImages;
    vector<vector<float>> baselineVector;

    // std::map<float, string> simImages;
    // baselineMatching.getFeatureVector(img, baselineVector);
    // double intersectionItself = baselineMatching.compare(baselineVector, baselineVector);

    histogramMatching.getFeatureVector(img, baselineVector);
    double intersectionItself = histogramMatching.compare(baselineVector, baselineVector);

    printf("result %f\n", intersectionItself);

    fileList = readFiles("/Users/danielbi/git-repo/FALL23-CS5330/project-2/examples/olympus");

    double intersection = 0.0;

    for (string filePath : fileList)
    {
        vector<vector<float>> featureVector;
        Mat compareImg = imread(filePath, IMREAD_COLOR);

        // histogram based matching
        histogramMatching.getFeatureVector(compareImg, featureVector);
        double intersection = histogramMatching.compare(baselineVector, featureVector);
        simImages.emplace(intersection, filePath);

        // baseline matching
        // baselineMatching.getFeatureVector(compareImg, featureVector);
        // double intersection = baselineMatching.compare(baselineVector, featureVector);
        // double modifiedDelta = intersection / (compareImg.cols * compareImg.rows + imgSize);
        // simImages.emplace(baselineMatching.compare(baselineVector, featureVector), filePath);
    }

    int resultCount = 0;

    for (const auto &entry : simImages)
    {
        printf("Match #%d (error=%f):\n filepath= %s\n", resultCount + 1, entry.first, entry.second.c_str());

        // open windows
        Mat image = imread(entry.second);

        if (image.empty())
        {
            printf("Cannot read image: %s\n", entry.second.c_str());
            return -1;
        }

        // add text label
        string text = "Sim image rank #" + to_string(resultCount);
        string notification = "Press 'q' to quit";

        putText(image, text, cv::Point(10, 40), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        putText(image, notification, cv::Point(10, 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(255, 255, 0), 1);

        imshow(text, image);

        resultCount++;

        if (resultCount == 11 || resultCount == simImages.size())
        {
            break;
        }
    }

    char k = cv::waitKey(0);

    while (k != 'q')
    {
        // listen to the key stroke and apply image effect
        if (k == 'q')
        {
            printf("Program exited by user.");
            cv::destroyAllWindows();
            break;
        }
        char k = cv::waitKey(0);
    }
    return 0;
}