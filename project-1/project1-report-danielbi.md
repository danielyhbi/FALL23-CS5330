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
> No required image needed.

### 2 - Display live video
> No required image needed.

### 3 - Display greyscale live video
![vidDisplay Greyscale w OpenCV](/assets/report/3_video_greyscale.gif)
> By toggling 'g'. The feed switched to the greyscale filter by openCV (`cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);`)

According to opencv's website<sup>[1]</sup>. RGB/BGR <-> GRAY transform each channel through pipeline where (conform with  `rec601 luma` what is commonly used standard color TV system<sup>[2]</sup>):
$$ Y = 0.299 * R + 0.587 * G + 0.114 * B $$

In the end, each channel (R, G, B) are applied with the same intensity for a greyscale image.

### 4 - Display alternative greyscale live video
![vidDisplay Greyscale 709](/assets/report/4_video_greyscale709.gif)
> Although it looks similar to the image from section 3, a slightly different filter is used, where:
$$ Y = 0.2126 * R + 0.7152 * G + 0.0722 * B $$

This luma coding abides to the ITU-R BT.709 standard used for HDTV<sup>[2]</sup>, which emphasis more on the green color.

### 5 - Implement a 5x5 Gaussian filter as separable 1x5 filters
![vidDisplay Gaussian filter before](/assets/report/5_gaussblur_before.png)
![vidDisplay Gaussian filter](/assets/report/5_gaussblur_after.png)
> The first image is before the blur, the second image is after the blur. I included the photos instead because gif loses the fidelity of image, which would be hard to distinguish the blur filter.

The 5x5 gauss filter worked great on softening the edges around the fonts.

In addition, for faster convolution processing, (2) 1x5 kernels are utilized as separable filters. Where the original 5x5 gaussian filter is separated into (2) 1x5 matrix (one vertical, one horizontal) as shown below.

$$G_{pass1} = \begin{bmatrix} 0.1& 0.2& 0.4& 0.2& 0.1\end{bmatrix}$$
$$G_{pass2} = \begin{bmatrix} 0.1\\ 0.2\\ 0.4\\ 0.2\\ 0.1\end{bmatrix}$$

In order to properly perserve the intemediate steps between each convolutions, a temporary `Mat` is created. Psudocode is shown below.
```c++
cv::Mat temp;
cv::Mat dst;

MatrixConv(src, temp, G_pass1); // used for intermediate process
MatrixConv(temp, dst, G_pass2); // final output
```
Note each kernel is normalized to 1. For detail convolution steps, and gaussian filter processing. Please refer to function `matrixConvolution8bit` and `blur5x5` in the `filters.cpp` file.

### 6 - Implement a 3x3 Sobel X and 3x3 Sobel Y filter as separable 1x3 filters
> No required image needed. See `/assets/report` for all images.

### 7 - Implement a function that generates a gradient magnitude image from the X and Y Sobel images
![vidDisplay sobel X+Y](/assets/report/7_edge_detect_both.gif)
> The gif above shows sobel edge detection in a live video feed

$$Sobel_{x} = \begin{bmatrix} 1& 0& -1\\ 2& 0& -2 \\ 1& 0& -1\end{bmatrix} Sobel_{x} = \begin{bmatrix} 1& 2& 1\\ 0& 0& 0 \\ -1& -2& -1\end{bmatrix}$$

Similiar processing is utilized as the gaussian blur filter. However some difference remains:
- Due to the nature of sobel matrix, edge detection for each direction (X, Y) has to be processed separately.
- Sobel matrix highlights pixel that has neighbors with higher RGB differences. The operation zeros out pixels with low differences, and "255" pixels with higher differences. However, the math results both positive and negative values that needs to be properly recorded.

Once the kernel is properly separated (shown below), same process is done as the gauss blur. It doesn't matter which separated kernel is applied first due to math property.

$$Sobel_{xv} = \begin{bmatrix} 0.25\\ 0.5 \\ 0.25\end{bmatrix} Sobel_{xh} = \begin{bmatrix} 1& 0& -1\end{bmatrix}$$

$$Sobel_{yv} = \begin{bmatrix} 1\\ 0 \\ -1\end{bmatrix} Sobel_{yh} = \begin{bmatrix} 0.25& 0.5& 0.25\end{bmatrix}$$

The final step is to combine the edge detection result from x and y direction. Which is loop each pixel and perform an Euclidean distance calculation to get the gradient magnitidue.

To account for the negative value during the convolution, 16bit image coding is utilized (`CV_16SC3` and `Vec3s`). For detail convolution steps, and gaussian filter processing. Please refer to function `matrixConvolution16bit` and `sobel3X3` in the `filters.cpp` file.

### 8 - Implement a function that blurs and quantizes a color image
![vidDisplay blur + quantize before](/assets/report/8_blurQuantize_before.png)
![vidDisplay blur + quantize](/assets/report/8_blurQuantize_after.png)
> The first image is before, the second image is after. I included the photos instead because gif loses the fidelity of image, which would be hard to distinguish the quartize filter.

This is quite a straightforward task. You use the blur filter that you built, and follow the homework instruction to distinguist each RGB value into a set amount of bucket. For this filer, I used the value of `15`, which indicates there are only 15 sets of RGB values in that image.

For detail convolution steps, and gaussian filter processing. Please refer to function `matrixConvolution8bit` and `blurQuantize` in the `filters.cpp` file.

### 9 - Implement a live video cartoonization function using the gradient magnitude and blur/quantize filters
![vidDisplay cartoon](/assets/report/9_cartoon_video.gif)
> The gif above shows the cartoon filter in a live video feed

Granted this is quite a primitive take on the cartoon filter. For the scope of this homework, the process is listed below:
- Detect Edges (for the tracing effect)
- BlurQuantize (for mimicing the simple colors)
- Overlay the edges on top of the quantized frame (create the hand-drawn effect)

One of the slightly tricky part is to overlay the edges (black as an inked pen) since my `Sobel3X3` highlights edges as brighter color. With `magnitudeThreshold`, user can define at what threashold it wants to trace the edge--the answer is a surprisingly low value 20. Basically anything *not* black from my edge detection should be accounted.

For detail convolution steps, and gaussian filter processing. Please refer to function `cartoon` in the `filters.cpp` file.

### 10 - Pick another effect to implement on your video
![vidDisplay sharpen before](/assets/report/10_shapren_before.png)
![vidDisplay sharpen](/assets/report/10_sharpen_after.png)
> The first image is before, the second image is after. I included the photos instead because gif loses the fidelity of image, which would be hard to distinguish the sharpen filter.

I picked the sharpen filter. It utilizes the same process as the matrix convolution, where the kernel is:
$$Shapren = \begin{bmatrix} -.125& -.125& -.125& -.125& -.125\\
                          -.125& .25& .25& .25& -.125\\
                          -.125& .25& 1.0& .25& -.125\\
                          -.125& .25& .25& .25& -.125\\
                          -.125& -.125& -.125& -.125& -.125\end{bmatrix}$$

For detail convolution steps, and gaussian filter processing. Please refer to function `shapren` in the `filters.cpp` file.

## Extensions
### Overview
After looking into the extesion options, I wanted to focus on 3 things:
- Implementing a new filter from scratch (Color Transformations)
- Refactor filters so it applies to still images
- Performance improvements

### EXT1 - New Filter
Sepia filter requires matrix multiplication. In general, any color transformation filter requires matrix multiplication to transform colors based on its according RGB value. The transformation matrix for Sepia is shown below:

$$Sobel_{x} = \begin{bmatrix} 0.393& 0.769& 0.189\\ 0.349& 0.686& 0.168 \\ 0.272& 0.534& 0.131\end{bmatrix}$$

![imgDisplay Greyscale w OpenCV](/assets/report/ex1_sepia.gif)
> The gif above shows the sepia filter in a live video feed

For detail convolution steps, and gaussian filter processing. Please refer to function `matrixMultiplication`, `sepia` in the `filters.cpp` file.

### EXT2 - Refactor to imgDisplay.cpp

Naturally I thought about refactoring the code so all filters work in the `imgDisplay.cpp`. With the same capability as `vidDisplay.cpp`, user can apply a filter, and save to a location on disk.

I selected an image of my *second* favorite basketball player on the Portland Trailblazer team, Jeraimi Grant, as demonstration.

![imgDisplay Greyscale w OpenCV](/assets/report/extensions/1_og.png)

![imgDisplay Greyscale w OpenCV](/assets/report/3_img_greyscale.gif)
> gif of applying greyscale filter in action. His beautiful smile is still there despite the lack of RBGs

![imgDisplay ex blur w OpenCV](/assets/report/extensions/2_blur.png)
> A blur filter applied to the image. He is so sharp that the bluring filter barely did anything, just barely tho.. (my filter still worked)

![imgDisplay ex Edge detection ](/assets/report/extensions/3_edgeDetect.png)
> Probably the coolest filter effect I've seen so far, the edges are quite prominent since the original image has quite simple colors. The sobel detection filter was almost as perfect as his shooting (perimeter and in the paint).

![imgDisplay sepia w OpenCV](/assets/report/extensions/4_sepia.png)
> This is the sepia filter. Looks like Jerami is in the past but he is the future of the Blazers.

![imgDisplay cartoon w OpenCV](/assets/report/extensions/6_cartoon.png)
> This is the cartoon filter. It is almost giving me the Coby White vibe, Coby wishes tho.

![imgDisplay blur+quantize w OpenCV](/assets/report/extensions/7-blurquantize.png)
> This is the just the blur+Quantize filter. It didn't do much because the original image has simple distinct colors. However, look into the background and you can tell how many light sources it has. (Answer: not 2 but 3, since Jerami himself is a source of light..)

![imgDisplay shapren w OpenCV](/assets/report/extensions/8-sharpen.png)
> This is the the shapren filter. That's how the trailblazer fans would like--Jerami Grant, extra sharp ;)

### 3 - performance improvements

After me almost having too much fun on the image filters, and (probably) getting blocked by Grant himself on social media (if he sees my filters). I decided to focus on more a serious matter--performance.

My Macbook still runs on intel but it is not a fossil, however any matrix convolution filter runs quite slow on my machine.. See examples below:

![slow calcualtions](/assets/report/extensions/ex3_slow_edge.gif)
> The frame per second (FPS) goes from ~75 (no filter) all the way down to ~1.5FPS after edge detection is applied. It is even worse on the cartoon effect

![slow calcualtions](/assets/report/extensions/ex3_slow_cartoon.gif)
> The FPS dropped below 1 FPS.. This is definitely unacceptable even to my standard!

After learning that everything here is executed single threaded and with a single process, I decided to convert my convolution funtion to a more parallel-friendly code. Using C++'s asynchronous operations capability.

Mainly, I figured it would be efficient if the process can `divide and conquer`. For now, the frame (cv::Mat) is divided into (4) parts, where it is equally divided by the col and row (see figure below):

![slow calcualtions](/assets/report/extensions/ex_rect.jpeg)

Code-wise, I utilized the `future` library. Each `future` is an async task which is defined by a lambda. Pseduo-code below:
```c++
auto convoTask1 = [startRow, endRow, startCol, endCol, src, dst, kernel] {
    matrixConvolutionHelper(startRow, endRow, startCol, endCol, src, dst, kernel);
}
auto convoTask2 = ...

std::future<void> future1 = std::async(std::launch::async, convolutionTask1);
std::future<void> future2 = ...

future1.get();
future2.get();
...
```

With a modified function to perform matrix convolution, essentially it can start and end any square-area within the defined frame. I did (4) for now.

The result is almost un-surprising. the performance (FPS) increased by 4 times for each convolution tasks, shown below. 

![slow calcualtions](/assets/report/extensions/ex3_fast_edge.gif)
> The frame per second (FPS) increased to 4.5 FPS, which is about 4x more than the non-concurrent case

![slow calcualtions](/assets/report/extensions/ex3_fast_cartoon.gif)
> The frame per second (FPS) increased to 2.8 FPS, which is about 4x more than the non-concurrent case

I'm just happy that my laptop's fan no longer sounds exploding, and my program is way more responsive than before. If I had more time, I'd look into how to make it even faster.

For detail convolution steps, and gaussian filter processing. Please refer to function `matrixConvolution8bitAsync` and `matrixConvolution16bitAsync` in the `filters.cpp` file.

## Reflection
This is overall a very fun project for me. I get to programmatically edit image/video, learned how pixel works within an image, and how image filter works. Excited for the next project!

## Acknowledgements and References
[1]: OpenCV Doc (https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray)

[2]: Greyscale Wikipedia (https://en.wikipedia.org/wiki/Grayscale)

### People who I talked to
- Professor Bruce (on trouble shooting my edge detection code)
- Elsa Tamara (on how header works)