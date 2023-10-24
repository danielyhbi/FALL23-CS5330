//
//  DataModel.swift
//  ObjectDetect
//  Inspired by Apple Tutorial from their dev website
//
//  Created by Daniel Bi on 10/23/23.
//  CS5330 - hw3
//

import AVFoundation
import SwiftUI
import os.log

final class DataModel: ObservableObject {
    let camera = Camera()
    let csv = CSVProcess()
    
    @State var isCameraRunning: Bool
    @Published var viewfinderImage: Image?
    @Published var capturedImage: UIImage?
    @Published var filterStatus: FilterType = .recognition
    @Published var featureSet = [String]()
    @Published var detectWithKNearest = true
    
    var cvWrapper: CVWrapper
    
    enum FilterType {
        case recognition
        case learningMode
        case passThrough
        case demo
    }
    
    init() {
        isCameraRunning = camera.isRunning;
        self.cvWrapper = CVWrapper.getKNNModel()
        
        Task {
            await handleCameraPreviews()
        }
        
        Task {
            await handleCameraPhotos()
        }
        
        Task {
            await handleFeatureSets()
        }
    }
    
    func handleCameraPreviews() async {
        let imageStream = camera.previewStream
            .map { $0.getUIImage }
        
        for await uiimage in imageStream {
            Task { @MainActor in
                
                switch filterStatus {
                case .recognition:
                    viewfinderImage = Image(uiImage: cvWrapper.objectDetect(uiimage, withKNN: detectWithKNearest))
                case .learningMode:
                    viewfinderImage = Image(uiImage: CVWrapper.learningModePreview(uiimage))
                case .passThrough:
                    viewfinderImage = Image(uiImage: CVWrapper.toGray(uiimage))
                case .demo:
                    viewfinderImage = Image(uiImage: CVWrapper.objectDetectCore(uiimage))
                }
            }
        }
    }
    
    func handleCameraPhotos() async {
        let unpackedPhotoStream = camera.photoStream
            .compactMap { self.unpackPhotoToImage($0) }
        
        for await photoData in unpackedPhotoStream {
            Task { @MainActor in
                capturedImage = photoData
            }
        }
    }
    
    func handleFeatureSets() async {
//        Task {
        featureSet.removeAll()
        await csv.readCSV()
        self.featureSet.append(contentsOf: csv.csvData)
//        }
        
        Task {
            
            self.cvWrapper.trainModel(self.featureSet)
        }
    }
    
    func updateFeatureSets(newSet: [String]) {
        
        self.featureSet.append(contentsOf: newSet)
        
        Task {
            await csv.writeCSV(newCSVData:self.featureSet)
        }
        
        Task {
            self.cvWrapper.trainModel(self.featureSet)
        }
        print("cvWrapper")
    }
        
    private func unpackPhotoToImage(_ photo: AVCapturePhoto) -> UIImage? {
        guard let previewCGImage = photo.previewCGImageRepresentation(),
              let metadataOrientation = photo.metadata[String(kCGImagePropertyOrientation)] as? UInt32,
              let cgImageOrientation = CGImagePropertyOrientation(rawValue: metadataOrientation) else { return nil }
        
        let imageOrientation = UIImage.Orientation(cgImageOrientation)
        let height = previewCGImage.height
        let width = previewCGImage.width
        
        return UIImage(cgImage: previewCGImage, scale: 1, orientation: imageOrientation)
    }
}

fileprivate struct PhotoData {
    var thumbnailImage: Image
    var thumbnailSize: (width: Int, height: Int)
    var imageData: Data
    var imageSize: (width: Int, height: Int)
}

fileprivate extension CIImage {
    var image: Image? {
        let ciContext = CIContext()
        guard let cgImage = ciContext.createCGImage(self, from: self.extent) else { return nil }
        
        return Image(decorative: cgImage, scale: 1, orientation: .up)
    }
}

// wrote by Daniel
fileprivate extension CIImage {
    var getUIImage: UIImage? {
        let ciContext = CIContext()
        guard let cgImage = ciContext.createCGImage(self, from: self.extent) else { return nil }
        return UIImage(cgImage: cgImage, scale: 1, orientation: .up)
    }
}

fileprivate extension Image.Orientation {
    
    init(_ cgImageOrientation: CGImagePropertyOrientation) {
        switch cgImageOrientation {
        case .up: self = .up
        case .upMirrored: self = .upMirrored
        case .down: self = .down
        case .downMirrored: self = .downMirrored
        case .left: self = .left
        case .leftMirrored: self = .leftMirrored
        case .right: self = .right
        case .rightMirrored: self = .rightMirrored
        }
    }
}

fileprivate extension UIImage.Orientation {
    
    init(_ cgImageOrientation: CGImagePropertyOrientation) {
        switch cgImageOrientation {
        case .up: self = .up
        case .upMirrored: self = .upMirrored
        case .down: self = .down
        case .downMirrored: self = .downMirrored
        case .left: self = .left
        case .leftMirrored: self = .leftMirrored
        case .right: self = .right
        case .rightMirrored: self = .rightMirrored
        }
    }
}

fileprivate let logger = Logger(subsystem: "com.apple.swiftplaygroundscontent.capturingphotos", category: "DataModel")
