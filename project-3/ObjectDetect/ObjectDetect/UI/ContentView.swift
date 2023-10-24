//
//  ContentView.swift
//  ObjectDetect
//
//  Created by Daniel Bi on 10/17/23.
//  CS5330 - hw3
//

import SwiftUI

struct ContentView: View {
    @StateObject private var model = DataModel()
    
    var body: some View {
        NavigationStack {
            VStack (alignment: .center, spacing: 50) {
                Image(systemName: "globe")
                    .imageScale(.large)
                    .foregroundStyle(.tint)
                Text("Welcome")
                NavigationLink(destination: VersionView()) {
                    ButtonObjectView(title:"Check Version",
                                     textColor: .white,
                                     backgroundColor: .blue)
                }
                
                NavigationLink(destination: CameraView(model: model)) {
                    ButtonObjectView(title:"Go to Camera View",
                                     textColor: .white,
                                     backgroundColor: .blue)
                }
                
                NavigationLink(destination: LearningModeView(model: model)) {
                    ButtonObjectView(title:"Learning Mode",
                                     textColor: .white,
                                     backgroundColor: .blue)
                }
                                
                NavigationLink(destination: HomeworkDemoView(model: model)) {
                    ButtonObjectView(title:"HW Demo Mode",
                                     textColor: .white,
                                     backgroundColor: .blue)
                }
                
                Button {
                    // reset training data and clears CSV file
//                    let samplePenSet: [String] = ["pen,10.417,65.439,14",
//                                                  "pen,9.714,70.059,35",
//                                                  "pen,8.75,68.824,83",
//                                                  "pen,8.460,67.758,21",
//                                                  "pen,9.510,70.736,52",
//                                                  "pen,11.875,66.038,42",
//                                                  "pen,9.058,66.608,94"]
                    //model.featureSet.removeAll()
                    //model.updateFeatureSets(newSet: samplePenSet)
                    Task {
                        await model.handleFeatureSets()
                    }
                } label: {
                    Label("Reset Training File", systemImage: "xmark.circle")
                        .font(.system(size: 17, weight: .bold, design: .default))
                        .cornerRadius(10)
                }
            
            }
            .padding()
        }
    }
}

fileprivate extension UINavigationBar {
    
    static func applyCustomAppearance() {
        let appearance = UINavigationBarAppearance()
        appearance.backgroundEffect = UIBlurEffect(style: .systemUltraThinMaterial)
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().compactAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
    }
}
