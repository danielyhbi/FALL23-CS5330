/*
    Daniel Bi
    Fall 2023
    CS 5330
    (file for harris corner detection)
*/

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <filesystem>
using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    // setting up run environment
    cv::VideoCapture *capdev;
    string currentPath = std::__fs::filesystem::current_path();
    printf("current path %s\n", currentPath.c_str());
    std::string saveFilePath = currentPath + "/out/";

    // open the video device
    capdev = new cv::VideoCapture(0);

    while (!capdev->isOpened())
    {
        printf("Unable to open video device.\n Accept the permission and press any key to continue.");
        waitKey(0);
        capdev = new cv::VideoCapture(0);
    }

    // cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame, gray, cornerFrame;
    vector<Point2f> corners;
    int max_corners = 100; // Maximum number of corners to detect
    double quality_level = 0.01; // Quality level for corner detection
    double min_distance = 10; // Minimum distance between detected corners

    printf("starting window...\n");
    for (;;)
    {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        goodFeaturesToTrack(gray, corners, max_corners, quality_level, min_distance);

        // Draw detected corners on the original frame
        cornerFrame = frame.clone();
        for (const Point2f& corner : corners) {
            circle(cornerFrame, corner, 5, Scalar(0, 0, 255), -1); // Draw red circles at corner points
        }

        // display image
        cv::imshow("Video", cornerFrame);

        char key = cv::waitKey(10);
        if (key == 'q')
        {
            printf("Program exited by user.");
            break;
        }
    }

    delete capdev;
    return (0);
}
