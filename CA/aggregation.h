#ifndef AGGREGATION_H
#define AGGREGATION_H
#include <opencv2/opencv.hpp>
#include <omp.h>
#include "../common.h"

using namespace cv;
using namespace std;

class Aggregation
{
public:
    Aggregation(const Mat &leftImage, const Mat &rightImage, uint colorThreshold1, uint colorThreshold2,
                uint maxLength1, uint maxLength2 ,uchar dmax,uchar dmin);
    void aggregation2D(Mat &costMap, bool horizontalFirst, uchar imageNo,uchar dis);
    void getLimits(vector<Mat> &upLimits, vector<Mat> &downLimits, vector<Mat> &leftLimits, vector<Mat> &rightLimits) const;
	vector<vector<Mat> > temp_costMaps;

	Mat bl_adap_coagg(const int d,const int imageNo);

	Mat plane_map;
	float** param_plan;
private:
    Mat images[2];
	
    Size imgSize;
    uint colorThreshold1, colorThreshold2;
    uint maxLength1, maxLength2;
    vector<Mat> upLimits;
    vector<Mat> downLimits;
    vector<Mat> leftLimits;
    vector<Mat> rightLimits;
	uchar dis;
	uchar dmax;
	uchar dmin;
	uint num_plan;

	uint Window_Size;
	float Sig_clr;
	
	

	//float** param_plan;

    int colorDiff(const Vec3b &p1, const Vec3b &p2);
    int computeLimit(int height, int width, int directionH, int directionW, uchar imageNo);
    Mat computeLimits(int directionH, int directionW, int imageNo);

    Mat aggregation1D(const Mat &costMap, int directionH, int directionW, Mat &windowSizes, uchar imageNo,uchar dis);
};

#endif // AGGREGATION_H
