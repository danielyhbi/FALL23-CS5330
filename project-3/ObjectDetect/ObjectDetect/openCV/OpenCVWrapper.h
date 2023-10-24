//
//  OpenCVWrapper.h
//  ObjectDetect
//
//  Created by Daniel Bi on 10/17/23.
//  CS5330 - hw3
//
#import <UIKit/UIKit.h>
#ifndef OpenCVWrapper_h
#define OpenCVWrapper_h

/// Represents a wrapper written in objective-c++/c++ to utilize the openCV library in C++
@interface CVWrapper : NSObject

/// ====== Instance Methods ======

/// takes in a newer batch of feature array and re-train the data
- (void)trainModel:(NSArray<NSString *> *)trainData;

/// Main object detection function that uses pre-processed features to determine object's type
- (UIImage *) objectDetect: (UIImage *)source withKNN:(bool) knn;

/// ====== Static Methods ======

/// Output a string representing the current version of openCV
+ (NSString *)getOpenCVVersion;

/// Convert image to a simple grey filter
+ (UIImage *)toGray:(UIImage *)source;

/// Processes a batch of images and return processed feature to the Model in SwiftUI
+ (NSArray<NSString *> *)processBatchCaptures:(NSArray<UIImage *> *)imageArray withClassifier:(NSString *)name;

/// load a few words on suggestions as well as highlighting conoturs while in learning mode
+ (UIImage *) learningModePreview: (UIImage *)source;

/// process image and output feature (height/width ratio and precentage fill area)
+ (NSString *) getFeature: (UIImage *)source withClassifier:(NSString *)name;

/// Floater method manuplicated to represent various stage of the assignment -- not for official use
+ (UIImage *)objectDetectCore:(UIImage *)source;

/// Enable singleton
+ (CVWrapper *) getKNNModel;

@end

#endif /* OpenCVWrapper_h */
