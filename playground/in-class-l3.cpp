#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>

int main(int argc, char const *argv[])
{
    cv::Mat src;
    cv::Mat dst;
    cv::Mat hist;

    char fileName[256];
    float max;
    const int histsize = 256;

    // error checking
    if (argc < 2)
    {
        printf("usageL %s <image fileame> \n", argv[0]);
        return -1;
    }

    // grab the filename
    strcpy(fileName, argv[1]);

    if (src.data == NULL)
    {
        return -2;
    }

    // build the hist gram
    // allocate the histogram, 2D single channel flaoting point array
    // initiated to zero
    hist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32FC1);

    // build the hisrogram
    max = 0; // keep track of the largest bucket

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            // grab the data
            float R = src.at<cv::Vec3b>(i, j)[0];
            float G = src.at<cv::Vec3b>(i, j)[1];
            float B = src.at<cv::Vec3b>(i, j)[2];

            // compute the rg standard chromaticity
            float diviser = R + G + B;
            diviser = diviser > 0 ? diviser : 1.0;
            float r = R / (R + G + B);
            float g = G / (R + G + B);

            // compute indexes, r, g are in [0, 1]
            int rIndex = (int)(r * (histsize - 1) + 0.5); // rounds to the neatest value
            int gIndex = (int)(g * (histsize - 1) + 0.5);

            // increment the histogram
            hist.at<float>(rIndex, gIndex)++;
            float newValue = hist.at<float>(rIndex, gIndex);

            // conditional assignemnt
            max = newValue > max ? newValue : max;
        }
    }

    // normalized the his gram
    // divide the whole mat by Max
    hist /= max;

    // after that, histogram is all in the range [0, 1]

    dst.create(hist.size(), CV_8UC3);

    // loop over hist
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            if (i + j > hist.rows)
            {
                dst.at<cv::Vec3b>(i, j) = cv::Vec3b(200, 120, 60);
            }
            else
            {
                float rColor = (float)i / histsize;
                float gColor = (float)j / histsize;
                float bColor = (float)((histsize - 1) - i - j) / histsize;

                dst.at<cv::Vec3b>(i, j)[0] = hist.at<float>(i, j) > 0 ? (unsigned char)(hist.at<float>(i, j) * 128 + 127) * bColor : 0;
                dst.at<cv::Vec3b>(i, j)[1] = hist.at<float>(i, j) > 0 ? (unsigned char)(hist.at<float>(i, j) * 128 + 127) * gColor : 0;
                dst.at<cv::Vec3b>(i, j)[2] = hist.at<float>(i, j) > 0 ? (unsigned char)(hist.at<float>(i, j) * 128 + 127) * rColor : 0;

                // dst.at<cv::Vec3b>(i, j)[1] = hist.at<float>(i, j) > 0 ? (unsigned char)(hist.at<float>(i, j) * 200 + 55.5 + 0.5) : 0;
                // dst.at<cv::Vec3b>(i, j)[2] = hist.at<float>(i, j) > 0 ? (unsigned char)(hist.at<float>(i, j) * 200 + 55.5 + 0.5) : 0;

                // dst.at<cv::Vec3b>(i, j)[1] = (unsigned char)(hist.at<float>(i, j) * 255 + 0.5);
                // dst.at<cv::Vec3b>(i, j)[2] = (unsigned char)(hist.at<float>(i, j) * 255 + 0.5);
            }
        }
    }

    // write image to a file
    cv::imwrite("histogram.png", dst);

    return 0;
}
