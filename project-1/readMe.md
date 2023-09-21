# Real-time Filtering
Author: Daniel Bi

CS5330 - Fall 2023
9/22/2023

## Introduction
This package includes demostration of image/video filtering using: 
- Gaussian blur (`b`)
- Greyscale using openCV (`g`)
- Greyscale using 709 standard (`h`)
- Edge detection (`e`-all dir. `w` - X-dir, `r` - Y-dir)
- Quantization (`i`)
- Cartoon (`c`)
- Sharpen (`x`)
- Sepia (`a`)

Other features includes:
- Save image (`s`)
- Exit app (`q`)

### Image/Video Filtering Process
`imgDisplay.cpp` reads in an image file from a pre-defined path `image_path`. `vidDisplay.cpp` reads in the camera stream. Both process image with external library `filters.h` (`filters.cpp` for implemetations). Most filtering feature are done by using Matrix Convolution. However, Sepia is done by color transformation (matrix multiplications).

By pressing keys (mapping shown above), user can toggle on/off for filters in the video feed. Due to the scope of `imgDisplay` as an extension, user is limited to apply a "one-time" filter in the image window, with ability to restore image by pressing `r`.

## Demos
### 1 - Read an image from a file and display it

### 2 - Display live video

### 3 - Display greyscale live video

### 4 - Display alternative greyscale live video

### 5 - Implement a 5x5 Gaussian filter as separable 1x5 filters

### 6 - Implement a 3x3 Sobel X and 3x3 Sobel Y filter as separable 1x3 filters

### 7 - Implement a function that generates a gradient magnitude image from the X and Y Sobel images

### 8 - Implement a function that blurs and quantizes a color image

### 9 - Implement a live video cartoonization function using the gradient magnitude and blur/quantize filters

### 10 - Pick another effect to implement on your video

## Extensions
### Overview
### 1 - new filter
### 2 - refactor to imgDisplay.cpp
### 3 - performance improvements

## Reflection

## Acknowledgements and References