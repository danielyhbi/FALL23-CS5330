/*
    Daniel Bi
    Fall 2023
    CS 5330
    vidDisplay.cpp (main file)
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

int printUWLogo(vector<Point3f> &points3D)
{

    points3D.clear();
    int altitude = 2;

    points3D.push_back(Point3f(0, 0, altitude));
    points3D.push_back(Point3f(3, 0, altitude));
    points3D.push_back(Point3f(3, -1, altitude));
    points3D.push_back(Point3f(2, -1, altitude));
    points3D.push_back(Point3f(3, -3.5, altitude));
    points3D.push_back(Point3f(4, 0, altitude));
    points3D.push_back(Point3f(5, 0, altitude));
    points3D.push_back(Point3f(6, -4, altitude));
    points3D.push_back(Point3f(6.5, -1, altitude));
    points3D.push_back(Point3f(6, -1, altitude));
    points3D.push_back(Point3f(6, 0, altitude));
    points3D.push_back(Point3f(8, 0, altitude));
    points3D.push_back(Point3f(8, -1, altitude));
    points3D.push_back(Point3f(7.5, -1, altitude));
    points3D.push_back(Point3f(7, -5, altitude));
    points3D.push_back(Point3f(5, -5, altitude));
    points3D.push_back(Point3f(4.5, -3, altitude));
    points3D.push_back(Point3f(3.5, -5, altitude));
    points3D.push_back(Point3f(2, -5, altitude));
    points3D.push_back(Point3f(1, -1, altitude));
    points3D.push_back(Point3f(0, -1, altitude));
    points3D.push_back(Point3f(0, 0, altitude));

    return 0;
}

int projectPointsToFrame(vector<Point3f> &points3D, Mat &photo, cv::Scalar &color, Mat &frame, Mat &rots, Mat &trans, Mat &cam, Mat &distort)
{

    vector<Point2f> imagePoints2D;

    cv::projectPoints(points3D, rots, trans, cam, distort, imagePoints2D);

    for (const auto &point : imagePoints2D)
    {
        circle(frame, point, 15, color, -1); // Draw projected points in red
    }

    // wrap and overlay images
    vector<Point2f> externalCorners = {
        Point2f(0, 0),                           // Top-left corner
        Point2f(photo.cols, 0),          // Top-right corner
        Point2f(photo.cols, photo.rows),  // Bottom-right corner
        Point2f(0, photo.rows)           // Bottom-left corner
    };
    // vector<Point2f> externalCorners = {
    //     Point2f(0, 0),                           // Top-left corner
    //     Point2f(photo.cols, 0),          // Top-right corner
    //     Point2f(0, photo.rows),           // Bottom-left corner
    //     Point2f(photo.cols, photo.rows)  // Bottom-right corner
    // };
    Mat perspectiveMatrix = getPerspectiveTransform(externalCorners, imagePoints2D);

    // Warp the external image onto the target image using the perspective transformation
    Mat warpedImage;
    warpPerspective(photo, warpedImage, perspectiveMatrix, frame.size());

    // Copy the warped image onto the target image
    warpedImage.copyTo(frame(Rect(0, 0, warpedImage.cols, warpedImage.rows)));


    return 0;
}

int projectLinesToFrame(vector<Point3f> &points3D, cv::Scalar &color, Mat &frame, Mat &rots, Mat &trans, Mat &cam, Mat &distort)
{

    vector<Point2f> imagePoints2D;

    cv::projectPoints(points3D, rots, trans, cam, distort, imagePoints2D);

    // Iterate through the vector of points and draw lines
    for (size_t i = 0; i < imagePoints2D.size() - 1; ++i)
    {
        cv::line(frame, cv::Point(imagePoints2D[i].x, imagePoints2D[i].y), cv::Point(imagePoints2D[i + 1].x, imagePoints2D[i + 1].y), color, 2);
    }

    // Draw the last line connecting the last and first points (for a closed shape)
    cv::line(frame, cv::Point(imagePoints2D.back().x, imagePoints2D.back().y), cv::Point(imagePoints2D.front().x, imagePoints2D.front().y), color, 2);

    return 0;
}

int overlayImage(vector<Point3f> &points3D, cv::Scalar &color, Mat &photo, Mat &frame, Mat &rots, Mat &trans, Mat &cam, Mat &distort) {
    return 0;
}

int main(int argc, char *argv[])
{
    // setting up run environment
    cv::VideoCapture *capdev;
    string currentPath = std::__fs::filesystem::current_path();
    printf("current path %s\n", currentPath.c_str());
    std::string saveFilePath = currentPath + "/out/";
    int boardWidth = 9;
    int boardHeight = 6;
    cv::Size boardSize(boardWidth, boardHeight);
    bool calibrated = false;

    // open the video device
    capdev = new cv::VideoCapture(0);

    while (!capdev->isOpened())
    {
        printf("Unable to open video device.\n Accept the permission and press any key to continue.");
        return -1;
    }

    // get some properties of the image
    cv::Size imageSize((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                       (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", imageSize.width, imageSize.height);

    cv::Mat huskyStadium;
    huskyStadium = cv::imread("/Users/danielbi/git-repo/FALL23-CS5330/project-4/src/greatestSetting.jpg");

    if (huskyStadium.empty()) {
        printf("Unable read image.");
        return -1;
    }

    cv::Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 2) = imageSize.width / 2;
    cameraMatrix.at<double>(1, 2) = imageSize.height / 2;
    cv::Mat distortionCoefficients = Mat::zeros(5, 1, CV_64F);

    // generate point set
    std::vector<cv::Vec3f> pointSet;          // the coordinate for the 9x6 board
    std::vector<cv::Point2f> cornerSet;       // corner set identified by the `findCorners`
    std::vector<cv::Point2f> cornerSetBackup; // back up corner set identified by the `findCorners`
    std::vector<std::vector<cv::Vec3f>> pointList;
    std::vector<std::vector<cv::Point2f>> cornerList;

    for (int col = 0; col < boardHeight; col++)
    {
        for (int row = 0; row < boardWidth; row++)
        {
            pointSet.push_back(cv::Vec3f(row, col * -1, 0));
        }
    }

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame, frame_bk, output;

    printf("starting window...\n");
    for (;;)
    {
        *capdev >> frame; // get a new frame from the camera, treat as a stream

        if (frame.empty())
        {
            printf("frame is empty\n");
            break;
        }

        // find the chessboard corners
        bool found = false;

        // convert to geyscale
        cv::Mat frameGray;
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);

        found = cv::findChessboardCorners(frameGray, boardSize, cornerSet,
                                          CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);

        if (found)
        {
            // Refine corner positions
            cv::cornerSubPix(frameGray, cornerSet, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

            // get the rotation and translation data
            if (calibrated)
            {
                Mat rotations, translations;
                cv::solvePnP(pointSet, cornerSet, cameraMatrix, distortionCoefficients,
                             rotations, translations);

                // cout << "\nRotation matrix:\n"
                //      << rotations
                //      << "\nTranslatin matrix:\n"
                //      << translations << endl;

                // project points on frame
                vector<Point3f> theCorners;
                theCorners.push_back(Point3f(0, 0, 0));
                theCorners.push_back(Point3f(8, 0, 0));
                theCorners.push_back(Point3f(0, -5, 0));
                theCorners.push_back(Point3f(8, -5, 0));

                cv::Scalar red(0, 0, 255);

                projectPointsToFrame(theCorners, huskyStadium, red, frame, rotations, translations, cameraMatrix, distortionCoefficients);

                vector<Point3f> uOfW;
                cv::Scalar purple(57, 39, 91);
                printUWLogo(uOfW);
                projectLinesToFrame(uOfW, purple, frame, rotations, translations, cameraMatrix, distortionCoefficients);

                // overlay husky stadium
                
                
            } else {
                // Draw chessboard corners on the frame
                cv::drawChessboardCorners(frame, boardSize, cornerSet, found);
            }

            // back up frames and cornerset
            frame_bk = frame.clone();
            // deep copy a vector list
            for (int index = 0; index < cornerSet.size(); index++)
            {
                cornerSetBackup.push_back(cornerSet[index]);
            }
        }

        // display image
        cv::imshow("Video", frame);

        char key = cv::waitKey(10);
        if (key == 'q')
        {
            printf("Program exited by user.");
            break;
        }
        else if (key == 's')
        {
            pointList.push_back(pointSet);

            if (found)
            {
                cornerList.push_back(cornerSet);
            }
            else
            {
                cornerList.push_back(cornerSetBackup);
            }

            printf("Image data #%zu saved! \n", pointList.size());
        }

        // check image calibration
        if (pointList.size() >= 5)
        {
            // set up calibration env
            double reprojectionError;
            std::vector<cv::Mat> rotations, translations;
            int flags = CALIB_FIX_ASPECT_RATIO;
            cameraMatrix = Mat::eye(3, 3, CV_64F);
            cameraMatrix.at<double>(0, 2) = imageSize.width / 2;
            cameraMatrix.at<double>(1, 2) = imageSize.height / 2;
            distortionCoefficients = Mat::zeros(5, 1, CV_64F);

            cv::calibrateCamera(pointList, cornerList, imageSize, cameraMatrix,
                                distortionCoefficients, rotations, translations, flags);

            vector<Point2f> projectedPoints;
            double totalError = 0;
            for (size_t i = 0; i < pointList.size(); ++i)
            {
                projectPoints(pointList[i], rotations[i], translations[i], cameraMatrix, distortionCoefficients, projectedPoints);
                double error = norm(cornerList[i], projectedPoints, NORM_L2) / projectedPoints.size();
                totalError += error;
            }
            reprojectionError = totalError / pointList.size();

            calibrated = true;

            // Print calibration results
            cout << "Camera Matrix after calibration:\n"
                 << cameraMatrix << endl;
            cout << "Distortion Coefficients after calibration:\n"
                 << distortionCoefficients << endl;
            cout << "Reprojection Error: " << reprojectionError << " pixels" << endl;

            // Save intrinsic parameters to a file
            FileStorage fs("intrinsic_parameters.yml", FileStorage::WRITE);
            fs << "Camera_Matrix" << cameraMatrix;
            fs << "Distortion_Coefficients" << distortionCoefficients;
            fs.release();

            // clear the queue
            pointList.clear();
            cornerList.clear();
            cout << "Press s to grab 5 more images for the next calibration.\n"
                 << endl;
        }
    }

    delete capdev;
    return (0);
}
