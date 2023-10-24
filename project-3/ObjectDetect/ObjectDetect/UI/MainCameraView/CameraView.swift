//
//  CameraView.swift
//  ObjectDetect
//
//  Created by Daniel Bi on 10/17/23.
//  CS5330 - hw3
//

import SwiftUI

struct CameraView: View {
    
    @ObservedObject var model: DataModel
    private static let barHeightFactor = 0.15
    
    
    var body: some View {
        
        NavigationStack {
            GeometryReader { geometry in
                ViewfinderView(image:  $model.viewfinderImage)
                    .overlay(alignment: .top) {
                        Color.black
                            .opacity(0.75)
                            .frame(height: geometry.size.height * Self.barHeightFactor)
                    }
                    .overlay(alignment: .bottom) {
                        buttonsView()
                            .frame(height: geometry.size.height * Self.barHeightFactor)
                            .background(.black.opacity(0.75))
                    }
                    .overlay(alignment: .center)  {
                        Color.clear
                            .frame(height: geometry.size.height * (1 - (Self.barHeightFactor * 2)))
                            .accessibilityElement()
                            .accessibilityLabel("View Finder")
                            .accessibilityAddTraits([.isImage])
                    }
                    .background(.black)
            }
            .task {
                model.filterStatus = .recognition
                await model.camera.start()
            }
            .onDisappear {
                model.camera.stop()
            }
            .navigationTitle("Camera")
            .navigationBarTitleDisplayMode(.inline)
            .ignoresSafeArea()
            .statusBar(hidden: true)
        }
    }
    
    
    
}

extension CameraView {
    private func buttonsView() -> some View {
        HStack(spacing: 60) {
            
            Spacer()

            Image(systemName: model.detectWithKNearest ? "k.circle" : "e.circle")
                .imageScale(.large)
                .foregroundStyle(.tint)
            
            Button {
                //model.camera.takePhoto()
                model.detectWithKNearest.toggle()
            } label: {
                Label {
                    Text("Take Photo")
                } icon: {
                    ZStack {
                        Circle()
                            .strokeBorder(.white, lineWidth: 3)
                            .frame(width: 62, height: 62)
                        Circle()
                            .fill(.white)
                            .frame(width: 50, height: 50)
                    }
                }
            }
            
            Button {
                model.camera.switchCaptureDevice()
            } label: {
                Label("Switch Camera", systemImage: "arrow.triangle.2.circlepath")
                    .font(.system(size: 36, weight: .bold))
                    .foregroundColor(.white)
            }
            
            Spacer()
            
        }
        .buttonStyle(.plain)
        .labelStyle(.iconOnly)
        .padding()
    }
}
