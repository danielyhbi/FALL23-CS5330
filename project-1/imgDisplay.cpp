#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include "filters.h"

using namespace cv;
using namespace std;

vector<int> getImageCompressionParam()
{
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(3);
    return compression_params;
}

int main()
{
    // can probably implement a file picker to select an image to display
    // can notify user if they want ot quit but kept on pressing the wrong keys
    // std::string saveFilePath = "/Users/danielbi/git-repo/FALL23-CS5330/project-1/ut/";
    // std::string image_path = "/Users/danielbi/git-repo/FALL23-CS5330/project-1/assets/jgrant.png";
    int saveCount = 1;

    string currentPath = std::__fs::filesystem::current_path();
    printf("current path %s\n", currentPath.c_str());

    std::string saveFilePath = currentPath + "/out/";
    std::string image_path = currentPath + "/assets/jgrant.png";

    Mat img = imread(image_path, IMREAD_COLOR);
    Mat imgModify = imread(image_path, IMREAD_COLOR);

    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    imshow("Display window", img);
    char key = waitKey(0); // Wait for a keystroke in the window

    while (key != 'q')
    {
        // listen to the key stroke and apply image effect
        if (key == 'q')
        {
            printf("Program exited by user.");
            break;
        }
        if (key == 's')
        {
            std::string saveFilePathAndName = saveFilePath + "imagecapture" + std::to_string(saveCount) + ".png";
            cv::imwrite(saveFilePathAndName, imgModify, getImageCompressionParam());
            printf("Screenshot saved to:\n    %s\n", saveFilePathAndName.c_str());
            saveCount++;
        }
        if (key == 'g')
        {
            cv::cvtColor(img, imgModify, cv::COLOR_BGR2GRAY);
            printf("toggled grey scale\n");
        }
        if (key == 'i')
        {
            blurQuantize(img, imgModify, 20);
            printf("toggled blur+quantize\n");
        }
        if (key == 'b')
        {
            blur5x5(img, imgModify);
            printf("toggled gauss blur\n");
        }
        if (key == 'e')
        {
            sobel3x3(img, imgModify);
            printf("toggled sobel edge detect\n");
        }
        if (key == 'c')
        {
            cartoon(img, imgModify, 10, 40);
            printf("toggled cartoon\n");
        }
        if (key == 'a')
        {
            sepia(img, imgModify);
            printf("toggled sepia\n");
        }
        if (key == 'x')
        {
            sharpen(img, imgModify);
            printf("toggled sharpen\n");
        }
        if (key == 'r')
        {
            imgModify = imread(image_path, IMREAD_COLOR);
            printf("toggled revert\n");
        }

        imshow("Display window", imgModify);
        key = waitKey(0);
    }

    cv::destroyAllWindows();
    return 0;
}