//创建于 21/3/2015
//研究 stereo matching, 李雪松 Xuesong Li

#include "imageprocessor.h"

ImageProcessor::ImageProcessor(float percentageOfDeletion)
{
    this->percentageOfDeletion = percentageOfDeletion;
}

Mat ImageProcessor::stretchHistogram(Mat image)
{
    Size imgSize = image.size();
    vector<Mat> channels;
    Mat output(imgSize, CV_8UC3);
    uint pixelThres = percentageOfDeletion * imgSize.height * imgSize.width;
    vector<uint> hist;
    hist.resize(std::numeric_limits<uchar>::max() + 1);


    cvtColor(image, output, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
    split(output, channels); //split the image into channels

    uchar min = std::numeric_limits<uchar>::max();
    uchar max = 0;

    for(size_t h = 0; h < imgSize.height; h++)
    {
        for(size_t w = 0; w < imgSize.width; w++)
        {
            uchar intensity = channels[0].at<uchar>(h, w);

            if(intensity < min)
                min = intensity;
            else if(intensity > max)
                max = intensity;

            hist[intensity]++;
        }
    }

    //update minimum
    bool foundMin = false;
    uint pixels = 0;
    for(size_t i = 0; i < hist.size() && !foundMin; i++)
    {
        pixels += hist[i];
        if(pixels <= pixelThres)
            min = i;
        else
            foundMin = true;
    }

    //update maximum
    bool foundMax = false;
    pixels = 0;
    for(size_t i = hist.size() - 1; i > 0 && !foundMax; i--)
    {
        pixels += hist[i];
        if(pixels <= pixelThres)
            max = i;
        else
            foundMax = true;
    }

    for(size_t h = 0; h < imgSize.height; h++)
    {
        for(size_t w = 0; w < imgSize.width; w++)
        {
            uchar intensity = channels[0].at<uchar>(h, w);
            uchar newIntensity;

            newIntensity = (intensity <= min)
                           ? 0
                           : (intensity >= max)
                             ? std::numeric_limits<uchar>::max()
                             : (std::numeric_limits<uchar>::max() / (float)(max - min)) * (intensity - min);

            channels[0].at<uchar>(h, w) = newIntensity;
        }
    }

    merge(channels,output); //merge 3 channels including the modified 1st channel into one image
    cvtColor(output, output, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)

    return output;
}

Mat ImageProcessor::unsharpMasking(Mat image, string blurMethod, int kernelSize, float alpha, float beta)
{
    Mat tempImage, output;
    float gamma = 0.0;
    float gaussianDevX = 0.0;
    float gaussianDevY = 0.0;

    if (blurMethod == "gauss")
    {
        GaussianBlur(image, tempImage, cv::Size(kernelSize, kernelSize), gaussianDevX, gaussianDevY);
        addWeighted(image, alpha, tempImage, beta, gamma, output);
    }
    else if (blurMethod == "median")
    {
        medianBlur(image, tempImage, kernelSize);
        addWeighted(image, alpha, tempImage, beta, gamma, output);
    }

    return output;
}

Mat ImageProcessor::laplacianSharpening(Mat image, int kernelSize, float alpha, float beta)
{
    Mat laplacianRes, abs_dst, output;
    int scale = 0;
    int delta = 0;
    float gamma = 0.0;

    Laplacian(image, laplacianRes, CV_8UC3, kernelSize, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(laplacianRes, abs_dst);
    addWeighted(image, alpha, abs_dst, beta, gamma, output);

    return output;
}
