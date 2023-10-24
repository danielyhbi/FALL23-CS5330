//
//  VersionView.swift
//  ObjectDetect
//
//  Created by Daniel Bi on 10/17/23.
//  CS5330 - hw3
//

import SwiftUI

struct VersionView: View {
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Welcome")
            Text("\(CVWrapper.getOpenCVVersion())")
        }
        .padding()
    }
}
