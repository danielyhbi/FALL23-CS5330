//
//  HomeworkDemoView.swift
//  ObjectDetect
//
//  Created by Daniel Bi on 10/17/23.
//  CS5330 - hw3
//

import SwiftUI

struct HomeworkDemoView: View {
    
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
                model.filterStatus = .demo
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

extension HomeworkDemoView {
    private func buttonsView() -> some View {
        HStack(spacing: 60) {
            
            Spacer()

            Image(systemName: "book.pages")
                .imageScale(.large)
                .foregroundStyle(.tint)
            
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
