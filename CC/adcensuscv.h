#ifndef ADCENSUSCV_H
#define ADCENSUSCV_H

#include <opencv2/opencv.hpp>

using namespace cv;

class ADCensusCV
{
public:
    ADCensusCV(const Mat &leftImage, const Mat &rightImage, Size censusWin, float lambdaAD, float lambdaCensus);
    float ad(int wL, int hL, int wR, int hR) const;
    float census(int wL, int hL, int wR, int hR) const;
    float adCensus(int wL, int hL , int wR, int hR) const;
private:
    Mat leftImage;
    Mat rightImage;
    Size censusWin;
    float lambdaAD;
    float lambdaCensus;
};

#endif // ADCENSUSCV_H
 