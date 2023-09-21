
// references:
// https://en.wikipedia.org/wiki/Grayscale
// https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray

// `COLOR_BGR2GREY` conforms with rec601 luma standard

// unlike the `COLOR_BGR2GREY`. This conforms with ITU-R BT.709 Standard
// where Y' = 0.2126R' + 0.7512G' + 0.0722B'
const std::vector<double> GREYSCALE_BGR = {0.0722, 0.7152, 0.2126};

const std::vector<std::vector<double>> GAUSS_BLUR_KERNEL_1DX = {{0.1, 0.2, 0.4, 0.2, 0.1}};

const std::vector<std::vector<double>> GAUSS_BLUR_KERNEL_1DY = {{0.1}, {0.2}, {0.4}, {0.2}, {0.1}};

// normalized from {1, 2, 1}
const std::vector<std::vector<double>> SOBEL_KERNEL_1DX_V = {{0.25}, {0.5}, {0.25}};

// normalized from {1, 0, -1}
const std::vector<std::vector<double>> SOBEL_KERNEL_1DX_H = {{1, 0, -1}};

const std::vector<std::vector<double>> SOBEL_KERNEL_1DY_V = {{1}, {0}, {-1}};

const std::vector<std::vector<double>> SOBEL_KERNEL_1DY_H = {{0.25, 0.5, 0.25}};

const std::vector<std::vector<double>> SOBEL_KERNEL_2DX = {{-0.25, 0, 0.25}, {-0.5, 0, 0.5}, {-0.25, 0, 0.25}};

const std::vector<std::vector<double>>
    SHAPREN_KERNEL_5X5 = {{-.125, -.125, -.125, -.125, -.125},
                          {-.125, .25, .25, .25, -.125},
                          {-.125, .25, 1.0, .25, -.125},
                          {-.125, .25, .25, .25, -.125},
                          {-.125, -.125, -.125, -.125, -.125}};

const std::vector<std::vector<double>> SEPIA_TRANSFORM_MATRIX = {{.393, .769, .189}, {.349, .686, .168}, {.272, .534, .131}};