//
//  LearningModeView.swift
//  ObjectDetect
//
//  Created by Daniel Bi on 10/17/23.
//  CS5330 - hw3
//

import SwiftUI
import AVFoundation

struct LearningModeView: View {
    
    @Environment(\.dismiss) private var dismiss
    
    @State private var sessionName: String = "Sample Object"
    @State private var capturedPhotos: [UIImage] = []
    @State private var features: [String] = []
    @ObservedObject var model: DataModel
    
    var body: some View {
        
        VStack {
            
            TextField("Enter session name", text: $sessionName)
                .padding()
            
            NavigationLink(
                destination:
                    LearningModeCaptureView(
                        sessionName: $sessionName,
                        capturedPhotos: $capturedPhotos,
                        features: $features,
                        model: model)) {
                            ButtonObjectView(title:"Start Capture",
                                             textColor: .white,
                                             backgroundColor: .blue)
                        }

            Button {
                dismiss()
            } label: {
                Label {
                    Text("Return to Main Screen")
                } icon: {
                    Image(systemName: "return.left")
                }
            }.padding(100)
            
        }
        
    }
    
}
