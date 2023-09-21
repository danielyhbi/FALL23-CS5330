
int getMatrixConvolutionValue(int pixelRow, int pixelCol, int channel, cv::Mat &src, const std::vector<std::vector<double>> &kernel)
{

    // check if kernel is even number
    if (kernel.size() % 2 == 0 || kernel[0].size() % 2 == 0)
    {
        printf("Error. Kernel size should be odd number for matrix convolution!");
        return -1;
    }

    // determine the center of the convoluted matrix
    int rowCenter = kernel.size() / 2;
    int colCenter = kernel[0].size() / 2;
    int sum = 0;

    for (int row = 0; row < kernel.size(); row++)
    {
        int rowDiff = row - rowCenter;

        for (int col = 0; col < kernel[0].size(); col++)
        {
            int colDiff = col - colCenter;

            // check if index is out of bounds
            if (pixelRow + rowDiff < 0 || pixelRow + rowDiff > src.rows || pixelCol + colDiff < 0 || pixelCol + colDiff > src.cols)
            {
                continue;
            }

            // adding up sums
            sum += kernel[row][col] * src.at<cv::Vec3b>(pixelRow + rowDiff, pixelCol + colDiff)[channel];
        }
    }

    return formatRGBValue(sum);
}

int matrixConvolution(cv::Mat &src, cv::Mat &dst, const std::vector<std::vector<std::vector<double>>> &kernels)
{

    for (int row = 0; row < src.rows; row++)
    {
        cv::Vec3b *rowPointer = dst.ptr<cv::Vec3b>(row);

        for (int col = 0; col < src.cols; col++)
        {
            for (int channel = 0; channel < 3; channel++)
            {

                for (int i = 0; i < kernels.size(); i++)
                {
                    // dealing with one kernel at a time
                    rowPointer[col][channel] = getMatrixConvolutionValue(row, col, channel, src, kernels[i]);
                }
            }
        }
    }

    return 0;
}
