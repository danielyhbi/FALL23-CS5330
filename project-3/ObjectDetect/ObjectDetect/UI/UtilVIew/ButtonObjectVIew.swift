//
//  ButtonObjectView.swift
//  ObjectDetect
//
//  Created by Daniel Bi on 10/17/23.
//  CS5330 - hw3
//

import SwiftUI

struct ButtonObjectView: View {
    
    var title: String
    var textColor: Color
    var backgroundColor: Color

    var body: some View {
        Text(title)
            .frame(width: 186, height: 37, alignment: .center)
            .background(backgroundColor)
            .foregroundColor(textColor)
            .font(.system(size: 17, weight: .bold, design: .default))
            .cornerRadius(10)
    }
}

struct ButtonObjectView_Previews: PreviewProvider {
    static var previews: some View {
        ButtonObjectView(title: "test tile",
                         textColor: .white,
                         backgroundColor: .blue)
    }
}
