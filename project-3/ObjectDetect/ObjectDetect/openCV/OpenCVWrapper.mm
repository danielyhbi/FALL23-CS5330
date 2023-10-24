//
//  OpenCVWrapper.m
//  ObjectDetect
//
//  Created by Daniel Bi on 10/17/23.
//  CS5330 - hw3
//

#import <opencv2/opencv.hpp>
#import <opencv2/core.hpp>
#import <opencv2/imgcodecs/ios.h>  //UIImage <-> Mat
#import <opencv2/imgproc.hpp>
#import <opencv2/ml.hpp>
#import <Foundation/Foundation.h>
#import "OpenCVWrapper.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <map>

#pragma clang pop

using namespace std;
using namespace cv;

/*
 * add a method convertToMat to UIImage class
 */
@interface UIImage (OpenCVWrapper)
//- (void)convertToMat: (cv::Mat *)pMat: (bool)alphaExists;
- (void)convertToMat:(cv::Mat *)pMat withAlphaExists:(BOOL)alphaExists;
@end

@implementation UIImage (OpenCVWrapper)

- (void)convertToMat: (cv::Mat *)pMat withAlphaExists:(bool)alphaExists {
    if (self.imageOrientation == UIImageOrientationRight) {
        /*
         * When taking picture in portrait orientation,
         * convert UIImage to OpenCV Matrix in landscape right-side-up orientation,
         * and then rotate OpenCV Matrix to portrait orientation
         */
        UIImageToMat([UIImage imageWithCGImage:self.CGImage scale:1.0 orientation:UIImageOrientationUp], *pMat, alphaExists);
        cv::rotate(*pMat, *pMat, cv::ROTATE_90_CLOCKWISE);
    } else if (self.imageOrientation == UIImageOrientationLeft) {
        /*
         * When taking picture in portrait upside-down orientation,
         * convert UIImage to OpenCV Matrix in landscape right-side-up orientation,
         * and then rotate OpenCV Matrix to portrait upside-down orientation
         */
        UIImageToMat([UIImage imageWithCGImage:self.CGImage scale:1.0 orientation:UIImageOrientationUp], *pMat, alphaExists);
        cv::rotate(*pMat, *pMat, cv::ROTATE_90_COUNTERCLOCKWISE);
    } else {
        /*
         * When taking picture in landscape orientation,
         * convert UIImage to OpenCV Matrix directly,
         * and then ONLY rotate OpenCV Matrix for landscape left-side-up orientation
         */
        UIImageToMat(self, *pMat, alphaExists);
        if (self.imageOrientation == UIImageOrientationDown) {
            cv::rotate(*pMat, *pMat, cv::ROTATE_180);
        }
    }
}
@end

@interface CVWrapper ()

#ifdef __cplusplus

@property cv::Ptr<cv::ml::KNearest> cVkNNModel;
@property std::map<int, std::string> labelMap;
@property cv::Mat cvTrainDataBackup;
@property cv::Mat cvLabelBackup;

#endif

@end

@implementation CVWrapper

#pragma mark Singleton Methods
+ (CVWrapper *)getKNNModel {
    static CVWrapper *cvWrapper = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        cvWrapper = [[self alloc] init];
    });
    return cvWrapper;
}

-(id)init {
    if (self = [super init]) {
        self.cVkNNModel = cv::ml::KNearest::create();
    }
    return self;
}

-(void)trainModel:(NSArray<NSString *> *)trainData {
    
    _labelMap.clear();
    
    cv::Mat cvTrainData = cv::Mat_<float>(1, 3);
    cv::Mat cvLabel = cv::Mat_<float>(1, 1);
    std::vector<std::string> labels;
    std::string lastLabel = "";
    int index = 1;
    
    // convert NSArray to cv::Mat
    for (NSString *stringInput in trainData) {
        
        std::string input = std::string([stringInput UTF8String]);
        printf("Processing String: %s\n", input.c_str());
        std::istringstream ss(input);
        std::string token, label;
        std::vector<std::string> parsedInput;
        // parse input
        while (std::getline(ss, token, ',')) {
            parsedInput.push_back(token);
            printf("Added token:%s to queue\n", token.c_str());
        }
        
        // check if reached end of line
        if (parsedInput.size() == 0) {
            break;
        }
        
        cv::Mat newRow = cv::Mat_<float>(1, 3);
        cv::Mat newLabel = cv::Mat_<float>(1, 1);
        // add new label to array
        if (lastLabel.compare(parsedInput[0]) != 0) {
            // new label occured
            index += 1;
            _labelMap[index] = parsedInput[0];
            
            lastLabel = parsedInput[0];
        }
        
        newLabel.at<float>(0,0) = index;
        cvLabel.push_back(newLabel);
        printf("Added label:%f to queue\n", newLabel.at<float>(0,0));
        // add data to the training set
        for (int i = 1; i < parsedInput.size(); ++i)
        {
            double value = std::stof(parsedInput[i]);
            newRow.at<float>(0, i - 1) = value;
            printf("Added token:%f to new Row\n", newRow.at<float>(0, i - 1));
        }
        cvTrainData.push_back(newRow);
        printf("push back new Row to Mat. Current RowCount:%d\n", cvTrainData.rows);
    }
    
    cv::Mat cvTrainDataCorrected = cvTrainData.rowRange(1, cvTrainData.rows);
    cv::Mat cvLabelCorrected = cvLabel.rowRange(1, cvLabel.rows);
    
    // backup so the homeworks part 6 will work
    _cvLabelBackup = cvLabelCorrected.clone();
    _cvTrainDataBackup = cvTrainDataCorrected.clone();
    
    cout << cvTrainDataCorrected << endl;
    cout << cvLabelCorrected << endl;
    
    std::map<int, std::string>::iterator it = _labelMap.begin();
    while (it != _labelMap.end())
    {
        std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;
        ++it;
    }
    
    _cVkNNModel->train(cvTrainDataCorrected, cv::ml::ROW_SAMPLE, cvLabelCorrected);
}

#pragma mark Public
/// Main object detection function that uses pre-processed features to determine object's type
- (UIImage *) objectDetect: (UIImage *)source withKNN:(bool)knn {
    
    cv::Mat image, imageProcessed, binaryImg;
    [source convertToMat: &image withAlphaExists:false];
    [source convertToMat: &imageProcessed withAlphaExists:false];
    
    [CVWrapper getKNNModel];
    
    if (image.empty()) {
        printf("Image is empty!\n");
        return source;
    }
    
    cv::Mat binary = cv::Mat::zeros(image.size(), CV_8UC1);
    [CVWrapper processThreshold:imageProcessed withBinaryMat:binary];
    
    [CVWrapper processRegions:binary];
    
    vector<Vec4i> hierarchy;
    vector<vector<cv::Point>> contours;
    // find coutours only the outer layer from the binary
    cv::findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    std::vector<cv::Scalar> meanColor;
    
    //Mat drawing = Mat::zeros(binary.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(0, 255, 255);
        drawContours(image, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
        
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
        drawContours(mask, contours, (int)i, cv::Scalar(255), cv::FILLED);
        
        meanColor.push_back(cv::mean(image, mask));
    }
    
    if (contours.size() == 0) {
        // if all background, just return the image
        return MatToUIImage(image);
    }
    
    // compute moments for each contours
    std::vector<cv::Moments> mu(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i]);
    }
    
    // calcualte features
    for (int i = 0; i < mu.size(); ++i)
    {
        cv::Point centroidXY;
        cv::Vec2d axisMax;
        cv::Vec2d axisMin;
        
        [CVWrapper getGeometry:mu[i] withCentroid:centroidXY withMinAxis:axisMin withMaxAxis:axisMax];
        
        cv::RotatedRect boundingBox(cv::minAreaRect(contours[i]));
        double objectArea = cv::contourArea(contours[i]);
        int width = min(boundingBox.size.width, boundingBox.size.height);
        int height = max(boundingBox.size.width, boundingBox.size.height);
        float regionArea = boundingBox.size.area();
        double precentFilled = objectArea / regionArea * 100;
        float heightWeightRatio = (float)height / (float)width;
        float avgColor = meanColor[i][0];
        
        string infoHeader1 = "";
        
        if (knn) {
            infoHeader1 = [self classifyWithKNN:heightWeightRatio withPrecentFilled:precentFilled withAverageColor:avgColor];
        } else {
            infoHeader1 = [self classifyWithEuclid:heightWeightRatio withPrecentFilled:precentFilled withAverageColor:avgColor];
        }
        
        int scaleFactor = 50;
        cv::Point centroidVectoMax(centroidXY.x + axisMax[0] * scaleFactor * 0.5, centroidXY.y + axisMax[1] * scaleFactor * 0.5);
        cv::Point centroidVectoMin(centroidXY.x + axisMin[0] * scaleFactor, centroidXY.y + axisMin[1] * scaleFactor);
        cv::arrowedLine(image, centroidXY, centroidVectoMax, cv::Scalar(255, 0, 0, 255), 1, LINE_AA);
        cv::arrowedLine(image, centroidXY, centroidVectoMin, cv::Scalar(255, 255, 0, 255), 1, LINE_AA);
        // write information text
        string infoHeader2 = "- Height/Width Ratio: " + to_string(heightWeightRatio);
        string infoHeader3 = "- Precent Filled: " + to_string(precentFilled) + "%";
        string infoHeader4 = "- Avg Folor: " + to_string(avgColor);
        
        // obtain the 4 points from the bounding box
        Point2f vertices[4];
        boundingBox.points(vertices);
        float xTopLeft = 0;
        float yTopLeft = 0;
        for (int i = 0; i < 4; i++) {
            line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0, 255), 2, LINE_AA);
            if ((float)vertices[i].y > yTopLeft) {
                
                yTopLeft = vertices[i].y;
                
                if ((float)vertices[i].x > xTopLeft) {
                    xTopLeft = vertices[i].x;
                }
            }
        }
        
        cv::Scalar textColor(112, 41, 99, 255);
        
        cv::putText(image, infoHeader1, cv::Point2f(xTopLeft, yTopLeft + 5), cv::FONT_HERSHEY_DUPLEX, 0.25, textColor, 1, LINE_AA);
        cv::putText(image, infoHeader2, cv::Point2f(xTopLeft, yTopLeft + 20), cv::FONT_HERSHEY_DUPLEX, 0.25, textColor, 1, LINE_AA);
        cv::putText(image, infoHeader3, cv::Point2f(xTopLeft, yTopLeft + 35), cv::FONT_HERSHEY_DUPLEX, 0.25, textColor, 1, LINE_AA);
        cv::putText(image, infoHeader4, cv::Point2f(xTopLeft, yTopLeft + 50), cv::FONT_HERSHEY_DUPLEX, 0.25, textColor, 1, LINE_AA);
    }
    
    return MatToUIImage(image);
}


+ (NSString *)getOpenCVVersion {
    return [NSString stringWithFormat:@"OpenCV Version %s",  CV_VERSION];
}

+ (UIImage *)toGray:(UIImage *)source {
    // convert to mat
    cv::Mat temp, result;
    [source convertToMat: &temp withAlphaExists:false];
    
    if (temp.empty()) {
        printf("Image is empty!\n");
        return source;
    }
    
    cv::cvtColor(temp, result, COLOR_BGR2GRAY);
    return MatToUIImage(result);
}

+ (NSArray<NSString *> *)processBatchCaptures:(NSArray<UIImage *> *)imageArray withClassifier:(NSString *)name {
    
    printf("Count of the image array: %d\n", imageArray.count);
    
    NSMutableArray<NSString *> *resultStrings = [[NSMutableArray alloc] init];
    
    for (UIImage *image in imageArray) {
        [resultStrings addObject: [CVWrapper getFeature:image withClassifier:name]];
    }
    
    return resultStrings;
}

/// load a few words on suggestions as well as highlighting conoturs while in learning mode
+ (UIImage *) learningModePreview: (UIImage *)source {
    
    cv::Mat image, imageProcessed, binaryImg;
    [source convertToMat: &image withAlphaExists:false];
    [source convertToMat: &imageProcessed withAlphaExists:false];
    
    
    if (image.empty()) {
        printf("Image is empty!\n");
        return source;
    }
    
    
    cv::Mat binary = cv::Mat::zeros(image.size(), CV_8UC1);
    [CVWrapper processThreshold:imageProcessed withBinaryMat:binary];
    
    [CVWrapper processRegions:binary];
    
    vector<Vec4i> hierarchy;
    vector<vector<cv::Point>> contours;
    // find coutours only the outer layer from the binary
    cv::findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    cv::Scalar textColor(112, 41, 99, 255);
    cv::Scalar meanColor;
    
    if (contours.size() == 0) {
        // if all background, just return the image
        string infoWarning = "No Object in View!";
        cv::putText(image, infoWarning, cv::Point2f(50, image.rows / 2), cv::FONT_HERSHEY_DUPLEX, 0.4, textColor, 1, LINE_AA);
        return MatToUIImage(image);
    }
    
    if (contours.size() > 1) {
        string infoWarning = "Only ONE object in View Please";
        cv::putText(image, infoWarning, cv::Point2f(50, image.rows / 2), cv::FONT_HERSHEY_DUPLEX, 0.5, textColor, 1, LINE_AA);
        return MatToUIImage(image);
    }
    
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(0, 255, 255, 255);
        drawContours(image, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
        
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
        drawContours(mask, contours, (int)i, cv::Scalar(255), cv::FILLED);
        
        meanColor = cv::mean(image, mask);
    }
    
    cv::RotatedRect boundingBox(cv::minAreaRect(contours[0]));
    double objectArea = cv::contourArea(contours[0]);
    int width = min(boundingBox.size.width, boundingBox.size.height);
    int height = max(boundingBox.size.width, boundingBox.size.height);
    float regionArea = boundingBox.size.area();
    double precentFilled = objectArea / regionArea;
    float heightWeightRatio = (float)height / (float)width;
    int avgColor = (int)meanColor[0];
    
    // write information text
    string infoHeader1 = "Object Stats:";
    string infoHeader2 = "- Height/Width Ratio: " + to_string(heightWeightRatio);
    string infoHeader3 = "- Precent Filled: " + to_string(precentFilled * 100) + "%";
    string infoHeader4 = "- Avg Folor: " + to_string(avgColor);
    
    cv::putText(image, infoHeader1, cv::Point2f(50, 20 + 5), cv::FONT_HERSHEY_DUPLEX, 0.25, textColor, 1, LINE_AA);
    cv::putText(image, infoHeader2, cv::Point2f(50, 20 + 20), cv::FONT_HERSHEY_DUPLEX, 0.25, textColor, 1, LINE_AA);
    cv::putText(image, infoHeader3, cv::Point2f(50, 20 + 35), cv::FONT_HERSHEY_DUPLEX, 0.25, textColor, 1, LINE_AA);
    cv::putText(image, infoHeader4, cv::Point2f(50, 20 + 50), cv::FONT_HERSHEY_DUPLEX, 0.25, textColor, 1, LINE_AA);
    
    return MatToUIImage(image);
}

/// process image and output feature (height/width ratio and precentage fill area)
+ (NSString *) getFeature: (UIImage *)source withClassifier:(NSString *)name {
    
    cv::Mat image, imageProcessed, binaryImg;
    [source convertToMat: &image withAlphaExists:false];
    [source convertToMat: &imageProcessed withAlphaExists:false];
    
    if (image.empty()) {
        printf("Image is empty!\n");
        return @"";
    }
    
    cv::Mat binary = cv::Mat::zeros(image.size(), CV_8UC1);
    [CVWrapper processThreshold:imageProcessed withBinaryMat:binary];
    
    [CVWrapper processRegions:binary];
    
    vector<Vec4i> hierarchy;
    vector<vector<cv::Point>> contours;
    // find coutours only the outer layer from the binary
    cv::findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    cv::Scalar meanColor;
    
    //Mat drawing = Mat::zeros(binary.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(0, 255, 255);
        drawContours(image, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
        
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
        drawContours(mask, contours, (int)i, cv::Scalar(255), cv::FILLED);
        
        meanColor = cv::mean(image, mask);
    }
    
    if (contours.size() == 0) {
        // if all background, just return the image
        return @"";
    }
    
    // compute moments for each contours
    std::vector<cv::Moments> mu(contours.size());
    
    mu[0] = moments(contours[0]);
    cv::Point centroidXY;
    cv::Vec2d axisMax, axisMin;
    
    [CVWrapper getGeometry:mu[0] withCentroid:centroidXY withMinAxis:axisMin withMaxAxis:axisMax];
    
    cv::RotatedRect boundingBox(cv::minAreaRect(contours[0]));
    double objectArea = cv::contourArea(contours[0]);
    int width = min(boundingBox.size.width, boundingBox.size.height);
    int height = max(boundingBox.size.width, boundingBox.size.height);
    float regionArea = boundingBox.size.area();
    double precentFilled = objectArea / regionArea;
    float heightWeightRatio = (float)height / (float)width;
    float avgColor = meanColor[0];
    
//    string features = to_string(heightWeightRatio) + "," + to_string(precentFilled * 100);
    
    return [NSString stringWithFormat:@"%@,%.3f,%.3f,%.3f", name, heightWeightRatio, precentFilled * 100, avgColor];
}

/// For homework report demo only
+ (UIImage *)objectDetectCore:(UIImage *)source {
    
    cv::Mat image, imageProcessed, binaryImg;
    [source convertToMat: &image withAlphaExists:false];
    [source convertToMat: &imageProcessed withAlphaExists:false];
    
    //printf("Image type= %d\n", image.type());
    [CVWrapper getKNNModel];
    
    if (image.empty()) {
        printf("Image is empty!\n");
        return source;
    }
    
    cv::Mat binary = cv::Mat::zeros(image.size(), CV_8UC1);
    [CVWrapper processThreshold:imageProcessed withBinaryMat:binary];
    
    // for HW section 1 and 2 - threshold the input video
    //return MatToUIImage(binary);
    
    [CVWrapper processRegions:binary];
    
    // ====================================
    // for HW section 3 - threshold the input video
    // without colors
    // return MatToUIImage(binary);
    //
    // with colors
//    cv::Mat coloredBinary = cv::Mat::zeros(image.size(), CV_8UC4);
//    [CVWrapper colorRegions:coloredBinary withBinary:binary];
//    return MatToUIImage(coloredBinary);
    // ====================================
    
    vector<Vec4i> hierarchy;
    vector<vector<cv::Point>> contours;
    // find coutours only the outer layer from the binary
    cv::findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    //Mat drawing = Mat::zeros(binary.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(0, 255, 255, 255);
        drawContours(image, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
    }
    
    if (contours.size() == 0) {
        // if all background, just return the image
        return MatToUIImage(image);
    }
    
    // compute moments for each contours
    std::vector<cv::Moments> mu(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i]);
    }
    
    // calcualte features
    for (int i = 0; i < mu.size(); ++i)
    {
        cv::Point centroidXY;
        cv::Vec2d axisMax;
        cv::Vec2d axisMin;
        
        [CVWrapper getGeometry:mu[i] withCentroid:centroidXY withMinAxis:axisMin withMaxAxis:axisMax];
        
        cv::RotatedRect boundingBox(cv::minAreaRect(contours[i]));
        double objectArea = cv::contourArea(contours[i]);
        int width = min(boundingBox.size.width, boundingBox.size.height);
        int height = max(boundingBox.size.width, boundingBox.size.height);
        float regionArea = boundingBox.size.area();
        double precentFilled = objectArea / regionArea;
        float heightWeightRatio = (float)height / (float)width;

        
        int scaleFactor = 50;
        cv::Point centroidVectoMax(centroidXY.x + axisMax[0] * scaleFactor * 0.5, centroidXY.y + axisMax[1] * scaleFactor * 0.5);
        cv::Point centroidVectoMin(centroidXY.x + axisMin[0] * scaleFactor, centroidXY.y + axisMin[1] * scaleFactor);
        cv::arrowedLine(image, centroidXY, centroidVectoMax, cv::Scalar(255, 0, 0, 255), 1, LINE_AA);
        cv::arrowedLine(image, centroidXY, centroidVectoMin, cv::Scalar(255, 255, 0, 255), 1, LINE_AA);
        
        // obtain the 4 points from the bounding box
        Point2f vertices[4];
        boundingBox.points(vertices);
        float xTopLeft = 0;
        float yTopLeft = 0;
        for (int i = 0; i < 4; i++) {
            line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0, 255), 2, LINE_AA);
            if ((float)vertices[i].y > yTopLeft) {
                
                yTopLeft = vertices[i].y;
                
                if ((float)vertices[i].x > xTopLeft) {
                    xTopLeft = vertices[i].x;
                }
            }
        }
    }
    
    return MatToUIImage(image);
}



#pragma mark Private

+ (double) getAverage: (std::vector<int> &)arr withStartIndex:(int) startIndex withEndIndex:(int) endIndex {
    // convert to the actual end of index
    endIndex == -1 ? endIndex = arr.size() : endIndex;
    
    int sum = 0;
    
    for (int i = startIndex; i < endIndex; i++)
    {
        sum += arr[i];
    }
    
    return (double)(sum) / (endIndex - startIndex);
    
}

+ (double) KMeansCluster1D: (vector<int> &)arr {
    // sort the vector
    std::sort(arr.begin(), arr.end());
    // initialize 2 points
    double mean1 = arr[0];
    double mean2 = arr[arr.size() - 1];
    int divider = 0;
    
    // loop through each value and compare/"assign" to each mean
    for (int index = 0; index < arr.size(); index++)
    {
        int current = arr[index];
        
        if (abs(mean1 - current) > abs(mean2 - current))
        {
            divider = index;
            break;
        }
    }
    
    // calculate the new mean
    mean1 = [CVWrapper getAverage:arr withStartIndex:0 withEndIndex:divider];
    mean2 = [CVWrapper getAverage:arr withStartIndex:divider withEndIndex:-1];
    
    return (double)((mean1 + mean2) / 2);
}

/// this helper method processes the image binary threshhold for further object detection
+ (void) processThreshold: (cv::Mat &)image withBinaryMat:(cv::Mat &)binary {
    //initialize workspace
    std::vector<int> colors;
    double sampleRate = 0.25;
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    
    // takes in random 1/4 pixels
    for (int row = 0 * image.rows; row < 1 * image.rows; row++)
    {
        Vec3b *rowPointer = image.ptr<Vec3b>(row);
        
        for (int col = 0 * image.cols; col < 1 * image.cols; col++)
        {
            if ((double)rand() / RAND_MAX < sampleRate)
            {
                colors.push_back(rowPointer[col][0]);
            }
        }
    }
    
    // k means cluster - 1D
    double threshold = [CVWrapper KMeansCluster1D:colors];
    //threshold = threshold - 20 > 0 ? threshold - 20 : 0;
    //printf("result is %f\n", threshold);
    
    // takes in random 1/4 pixels
    for (int row = 0; row < image.rows; row++)
    {
        uchar *rowPointer = image.ptr<uchar>(row);
        uchar *rowPointerBinary = binary.ptr<uchar>(row);
        
        for (int col = 0; col < image.cols; col++)
        {
            if (rowPointer[col] > threshold)
            {
                // background
                rowPointerBinary[col] = 0;
            }
            else
            {
                // foreground
                rowPointerBinary[col] = 255;
            }
        }
    }
    
    // for HW section 1 - threshold the input video
    //return;
    
    // try the morphological operation
    cv::Mat structElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, structElement);
}

+ (void) processRegions: (cv::Mat &)binary {
    // Apply connected components analysis
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binary, labels, stats, centroids);
    std::set<int> inBound;
    std::set<int> outOfBound;
    
    // filter out smaller regions
    // Filter small regions (step 3)
    const int minRegionSize = 100; // Set your minimum region size threshold
    for (int i = 1; i < numLabels; ++i)
    {
        int* statsPointer = stats.ptr<int>(i);
        
        if (statsPointer[cv::CC_STAT_AREA] < minRegionSize)
        {
            outOfBound.insert(i);
        }
    }
    
    // scan the first 3 and last 3 row/col of the mat
    int rowCount = binary.rows;
    int colCount = binary.cols;
    std::vector<int> rowCounts{0, 1, 2, rowCount - 3, rowCount - 2, rowCount - 1};
    std::vector<int> colCounts{0, 1, 2, colCount - 3, colCount - 2, colCount - 1};
    
    for (int row : rowCounts)
    {
        int* rowPointer = labels.ptr<int>(row);
        for (int col = 0; col < binary.cols; col++)
        {
            if (rowPointer[col] != 0)
            {
                outOfBound.insert(rowPointer[col]);
            }
        }
    }
    
    for (int row = 0; row < binary.rows; row++)
    {
        int* rowPointer = labels.ptr<int>(row);
        
        for (int col : colCounts)
        {
            if (rowPointer[col] != 0)
            {
                outOfBound.insert(rowPointer[col]);
            }
        }
    }
    
    // populate inBoundIndex
    for (int i = 1; i < numLabels; ++i)
    {
        auto condition = outOfBound.find(i);
        
        if (condition == outOfBound.end()) {
            inBound.insert(i);
        }
    }
    
    // eliminate useless regions
    for (int row = 0; row < binary.rows; row++)
    {
        uchar* rowPointer = binary.ptr<uchar>(row);
        int* rowPointerLabel = labels.ptr<int>(row);
        
        for (int col = 0; col < binary.cols; col++)
        {
            auto condition = inBound.find(rowPointerLabel[col]);
            
            if (condition == inBound.end()) {
                // reaching the end means not in the set == useless pixels == eliminate
                rowPointer[col] = 0;
            }
        }
    }
    
}

/// using a processed binary Mat (with segmented object separated), will compute bounding box and
/// other stats with contour and moments
+ (void) getGeometry: (cv::Moments &) mu withCentroid:(cv::Point &) centroidXY
         withMinAxis:(cv::Vec2d &) axisMin withMaxAxis:(cv::Vec2d &) axisMax {
    
    double mu20 = mu.mu20;
    double mu11 = mu.mu11;
    double mu02 = mu.mu02;
    
    double centroidX = mu.m10 / (mu.m00 + 1e-5);
    double centroidY = mu.m01 / (mu.m00 + 1e-5);
    
    centroidXY = cv::Point(centroidX, centroidY);
    
    
    cv::Mat momentsMatrix = (cv::Mat_<double>(2, 2) << mu20, mu11, mu11, mu02);
    
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(momentsMatrix, eigenvalues, eigenvectors);
    
    // get the index of the smallest eigenvalue
    int minEigenvalueIndex = (eigenvalues.at<double>(0) < eigenvalues.at<double>(1)) ? 0 : 1;
    int maxEigenvalueIndex = (eigenvalues.at<double>(0) > eigenvalues.at<double>(1)) ? 0 : 1;
    
    double eigenXMax = eigenvectors.at<double>(minEigenvalueIndex, 0);
    double eigenYMax = eigenvectors.at<double>(minEigenvalueIndex, 1);
    double eigenXMin = eigenvectors.at<double>(maxEigenvalueIndex, 0);
    double eigenYMin = eigenvectors.at<double>(maxEigenvalueIndex, 1);
    
    axisMax = cv::Vec2d(eigenXMax, eigenYMax);
    axisMin = cv::Vec2d(eigenXMin, eigenYMin);
}

- (std::string) classifyWithKNN:(float)heightWeightRatio withPrecentFilled:(float) precentFilled withAverageColor:(float)avgColor {
    // detect object with the training model
    cv::Mat currentFeature = (cv::Mat_<float>(1, 3) << heightWeightRatio, precentFilled, avgColor);
    // Check and convert data type if necessary
    if (currentFeature.type() != CV_32F) {
        currentFeature.convertTo(currentFeature, CV_32F);
    }
    int k = 2;
    std::vector<float> results, neighborResponses, dists;
    self.cVkNNModel->findNearest(currentFeature, k, results, neighborResponses, dists);
    
    float condifenceRate = max(dists[0], dists[1]);
    
    // retrieve item from map
    string itemName = _labelMap[results[0]];
    
    if (condifenceRate > 200) {
        itemName += "??";
    } else if (condifenceRate > 400) {
        itemName = "???";
    }
    
    return "Object Stats: " + itemName + "  Error:" + to_string(condifenceRate);
}

- (std::string) classifyWithEuclid:(float)heightWeightRatio withPrecentFilled:(float) precentFilled withAverageColor:(float)avgColor {
    
    // loop through the set and find the closest match
    int savedIndex = 0;
    float minEuclid = 9999999;
    
    for (int row = 0; row < _cvTrainDataBackup.rows; row++) {
        
        float* trainPointer = _cvTrainDataBackup.ptr<float>(row);
        
        double feature1 = trainPointer[0];
        double feature2 = trainPointer[1];
        double feature3 = trainPointer[2];
        double euclid =
        sqrt((heightWeightRatio - feature1)*(heightWeightRatio - feature1)
             + (precentFilled - feature2)*(precentFilled - feature2)
             + (avgColor - feature3)*(avgColor - feature3));
        
        
        if (euclid < minEuclid) {
            minEuclid = euclid;
            savedIndex = row;
        }
    }
    
    float condifenceRate = minEuclid;
    int itemIndex = _cvLabelBackup.at<float>(savedIndex, 0);
    
    // retrieve item from map
    string itemName = _labelMap[itemIndex];
    
    if (condifenceRate > 20) {
        itemName += "??";
    } else if (condifenceRate > 50) {
        itemName = "???";
    }
    
    return "Object Stats: " + itemName + "  Error:" + to_string(condifenceRate);
}

/// performs connected component analysis
+ (void) colorRegions: (cv::Mat &) coloredBinary withBinary:(cv::Mat &)binary {
    
    cv::Mat labels, stats, centroids;
    
    int numLabels = cv::connectedComponentsWithStats(binary, labels, stats, centroids);
    
    std::vector<Vec4b> colorsAssign(numLabels);
    
    for (int label = 1; label < numLabels; ++label)
    {
        colorsAssign[label] = Vec4b((rand() & 255), (rand() & 255), (rand() & 255), 255);
    }
    
    // recolor the region with labels
    for (int r = 0; r < coloredBinary.rows; ++r)
    {
        int* rowPointerLabel = labels.ptr<int>(r);
        uchar* rowPointerBinary = binary.ptr<uchar>(r);
        cv::Vec4b* rowPointerColor = coloredBinary.ptr<cv::Vec4b>(r);
        
        for (int c = 0; c < coloredBinary.cols; ++c)
        {
            if (rowPointerBinary[c] != 0) {
                // detected an non-trivial region
                int currentLabel = rowPointerLabel[c];
                rowPointerColor[c] = colorsAssign[currentLabel];
            }
        }
    }
}


@end
