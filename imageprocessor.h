#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H
#include <opencv2/opencv.hpp>
#include "common.h"

using namespace cv;

class ImageProcessor   //this class is used to filter the input images and cread the stereo methods
{
public:
    ImageProcessor(float percentageOfDeletion);
    Mat stretchHistogram(Mat image);
    Mat unsharpMasking(Mat image, string blurMethod, int kernelSize, float alpha, float beta);
    Mat laplacianSharpening(Mat image, int kernelSize, float alpha, float beta);
private:
    float percentageOfDeletion;
};

#endif // IMAGEPROCESSOR_H
