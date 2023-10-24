//
//  CapturedImageView.swift
//  ObjectDetect
//
//  Created by Daniel Bi on 10/17/23.
//  CS5330 - hw3
//

import SwiftUI
import AVFoundation

///Display a smaller preview of captured image by user
struct CapturedImageView: View {
    
    @Binding var image: UIImage?
    
    var body: some View {
        if let image = image {
            Image(uiImage: image)
                .resizable()
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10)
                    .stroke(Color.orange, lineWidth: 4))
                .shadow(radius: 10)
        }
    }
}
