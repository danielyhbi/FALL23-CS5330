//
//  LearningModeCaptureView.swift
//  ObjectDetect
//
//  Created by Daniel Bi on 10/17/23.
//  CS5330 - hw3
//

import SwiftUI
import AVFoundation

struct LearningModeCaptureView: View {
    
    @Environment(\.dismiss) private var dismiss
    
    @Binding var sessionName: String
    @Binding var capturedPhotos: [UIImage]
    @Binding var features: [String]
    @ObservedObject var model: DataModel
    
    var counter: Int = 0
    @State var isUnderFivePhotos: Bool = true
    
    var body: some View {
        
        NavigationStack {
            GeometryReader { geometry in
                ViewfinderView(image:  $model.viewfinderImage)
                    .overlay(alignment: .top) {
                        Text("Capture photos")
                    }
                    .overlay(alignment: .bottom) {
                        VStack {
                            CaptureProgressIconsView(count: capturedPhotos.count)
                            getShutterButton()
                        }
                    }
                    .overlay(alignment: .bottomTrailing) {
                        CapturedImageView(image: $model.capturedImage)
                            .frame(width: 100, height: 200)
                            .padding()
                    }
            }
        }.task {
            model.capturedImage = nil
            model.filterStatus = .learningMode
            await model.camera.start()
        }
        .onDisappear {
            model.camera.stop()
        }
    }
    
//    private func captureAndAppend() async {
//        var doneCapturing = false
//        
//        Task {
//            model.camera.takePhoto()
//            doneCapturing = true
//        }
//        
//        do {
//            guard let sampleImage = model.capturedImage else { return }
//            capturedPhotos.append(sampleImage)
//        }
//    }
    
    private func getShutterButton() -> some View {
        Button {
            if (capturedPhotos.count < 5) {
                model.camera.takePhoto()
                
                guard let sampleImage = model.capturedImage else { return }
                capturedPhotos.append(sampleImage)
            } else {
                isUnderFivePhotos = false
                // process data
                let feat = CVWrapper.processBatchCaptures(capturedPhotos, withClassifier: sessionName) as [NSString]
                
                let featuresInArray: [String] = feat.map {$0 as String}
                
                model.updateFeatureSets(newSet: featuresInArray)
                // then proceed to dismiss
                dismiss()
            }
        } label: {
            Label {
                Text("Take Photo")
            } icon: {
                ZStack {
                    Circle()
                        .strokeBorder(.white, lineWidth: 3)
                        .frame(width: 62, height: 62)
                    
                    if (capturedPhotos.count < 5) {
                        Circle()
                            .fill(.white)
                            .frame(width: 50, height: 50)
                    } else {
                        Image(systemName: "arrow.right")
                            .frame(width: 50, height: 50)
                    }
                }
            }
        }
        .buttonStyle(.plain)
        .labelStyle(.iconOnly)
        .padding()
    }
}

struct CaptureProgressIconsView: View {
    
    var count: Int
    var body: some View {
        HStack {
            if (count > 0) {
                ForEach (1...min(count, 5), id: \.self) {i in
                    getCompletedCircleWithCheck()
                }
            }
            
            if (count <= 4) {
                
                getCurrentCircleWithNumber()
                
                if (count <= 3) {
                    ForEach (count...3, id: \.self) {i in
                        getUpcomingCircleDot()
                    }
                }
            }
        }
    }
    
    private func getCurrentCircleWithNumber() -> some View {
        Image(systemName: "\(count+1).circle")
            .imageScale(.medium)
    }
    
    private func getUpcomingCircleDot() -> some View {
        Image(systemName: "circle.fill")
            .imageScale(.small)
    }
    
    private func getCompletedCircleWithCheck() -> some View {
        Image(systemName: "checkmark.circle")
            .imageScale(.medium)
            .foregroundColor(.green)
    }
}


