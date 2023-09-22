/*
    Daniel Bi
    Fall 2023
    CS 5330
    vidDisplay.cpp (main file)
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include "util.h"
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

int main(int argc, char *argv[])
{
    cv::VideoCapture *capdev;
    //string saveFilePath = "/Users/danielbi/git-repo/FALL23-CS5330/project-1/out/";

    string currentPath = std::__fs::filesystem::current_path();
    printf("current path %s\n", currentPath.c_str());
    std::string saveFilePath = currentPath + "/out/";

    int saveCount = 1;
    bool isGreyScale = false;
    bool isGreyScale709 = false;
    bool isGaussianBlur = false;
    bool isSobelX = false;
    bool isSobelY = false;
    bool isSobel = false;
    bool isBlurQuant = false;
    bool isCartoon = false;
    bool isTest = false;
    bool isSharpen = false;

    // open the video device
    capdev = new cv::VideoCapture(0);

    while (!capdev->isOpened())
    {
        printf("Unable to open video device.\n Accept the permission and press any key to continue.");
        waitKey(0);
        capdev = new cv::VideoCapture(0);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame, greyFrame, gaussFrame, sobelFrame, sobelFrame1;
    cv::Mat *frameOutput;

    for (;;)
    {
        int64 start = cv::getTickCount();

        *capdev >> frame; // get a new frame from the camera, treat as a stream

        if (frame.empty())
        {
            printf("frame is empty\n");
            break;
        }

        // check if any filters are applied
        if (isGreyScale)
        {
            cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
            frameOutput = &greyFrame;
        }
        else if (isGreyScale709)
        {
            concertToGreyScaleFast(frame, greyFrame);
            frameOutput = &greyFrame;
        }
        else if (isGaussianBlur)
        {
            blur5x5(frame, gaussFrame);
            frameOutput = &gaussFrame;
        }
        else if (isSobelX)
        {
            sobelX3x3(frame, sobelFrame);
            frameOutput = &sobelFrame;
        }
        else if (isSobelY)
        {
            sobelY3x3(frame, sobelFrame);
            frameOutput = &sobelFrame;
        }
        else if (isSobel)
        {
            sobel3x3(frame, sobelFrame);
            frameOutput = &sobelFrame;
        }
        else if (isBlurQuant)
        {
            blurQuantize(frame, gaussFrame, 15);
            frameOutput = &gaussFrame;
        }
        else if (isCartoon)
        {
            cartoon(frame, gaussFrame, 15, 20);
            frameOutput = &gaussFrame;
        }
        else if (isTest)
        {
            sepia(frame, sobelFrame);
            frameOutput = &sobelFrame;
        }
        else if (isSharpen)
        {
            sharpen(frame, gaussFrame);
            frameOutput = &gaussFrame;
        }
        else
        {
            frameOutput = &frame;
        }

        // display image
        cv::imshow("Video", *frameOutput);

        // very roughly measure fps
        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        // printf("FPS: %f\n", fps);

        // see if there is a waiting keystroke
        // ref: https://stackoverflow.com/questions/22148826/measure-opencv-fps
        char key = cv::waitKey(10);
        if (key == 'q')
        {
            printf("Program exited by user.");
            break;
        }
        if (key == 's')
        {
            string saveFilePathAndName = saveFilePath + "imagecapture" + std::to_string(saveCount) + ".png";
            cv::imwrite(saveFilePathAndName, *frameOutput, getImageCompressionParam());
            printf("Screenshot saved to:\n    %s\n", saveFilePathAndName.c_str());
            saveCount++;
        }
        if (key == 'g')
        {
            isGreyScale = toggleSwitch(isGreyScale);
            printf("toggled grey scale\n");
        }
        if (key == 'h')
        {
            isGreyScale709 = toggleSwitch(isGreyScale709);
            printf("toggled grey scale 709\n");
        }
        if (key == 'b')
        {
            isGaussianBlur = toggleSwitch(isGaussianBlur);
            printf("toggled gauss blur\n");
        }
        if (key == 'w')
        {
            isSobelX = toggleSwitch(isSobelX);
            printf("toggled sobel X\n");
        }
        if (key == 'r')
        {
            isSobelY = toggleSwitch(isSobelY);
            printf("toggled sobel Y\n");
        }
        if (key == 'e')
        {
            isSobel = toggleSwitch(isSobel);
            printf("toggled sobel -all \n");
        }
        if (key == 'i')
        {
            isBlurQuant = toggleSwitch(isBlurQuant);
            printf("toggled blurQuantize \n");
        }
        if (key == 'c')
        {
            isCartoon = toggleSwitch(isCartoon);
            printf("toggled cartoon \n");
        }
        if (key == 'a')
        {
            isTest = toggleSwitch(isTest);
            printf("toggled sepia \n");
        }
        if (key == 'x')
        {
            isSharpen = toggleSwitch(isSharpen);
            printf("toggled sharpen \n");
        }
    }

    delete capdev;
    return (0);
}
