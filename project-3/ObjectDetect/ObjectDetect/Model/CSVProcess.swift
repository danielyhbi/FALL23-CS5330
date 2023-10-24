//
//  CSVProcess.swift
//  ObjectDetect
//
//  Created by Daniel Bi on 10/23/23.
//  CS5330 - hw3
//

import Foundation

class CSVProcess: NSObject {
    var csvData: [String] = []
    
    private let currentDir: String = FileManager.default.currentDirectoryPath
    private let csvFileLocation: String = "modelFeatures.csv"
    
    func readCSV() async {
        csvData.removeAll()
        // Load CSV file
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = documentsURL.appendingPathComponent(csvFileLocation)
        
        if let path = Bundle.main.path(forResource: "modelFeatures", ofType: "csv") {
            //if let path = currentDir + csvFileLocation {
            do {
                print("obtain cvs filepath..\n\(path)")
                let csvString = try String(contentsOfFile: path, encoding: .utf8)
                print("read cvs file..\n\(csvString)")
                let csvLines = csvString.components(separatedBy: "\n")
                
                // Parse CSV lines
                for line in csvLines {
                    csvData.append(line)
                }
            } catch {
                print("Error reading CSV file: \(error)")
            }
        }
    }
    
    func writeCSV(newCSVData:[String]) async {
        // Inside the Button action to save CSV
        var csvText = ""
        for row in newCSVData {
            csvText += row + "\n"
        }
        
        let filePath = Bundle.main.path(forResource: "modelFeatures", ofType: "csv")
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = documentsURL.appendingPathComponent(csvFileLocation)
        
        do {
            try csvText.write(to: fileURL, atomically: true, encoding: .utf8)
            print("write cvs file..\n\(csvText)")
            print("CSV file saved at: \(fileURL)")
        } catch {
            print("Error saving CSV file: \(error)")
        }
    }
}
