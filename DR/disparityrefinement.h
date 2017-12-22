#ifndef DISPARITYREFINEMENT_H
#define DISPARITYREFINEMENT_H

#include <opencv2/opencv.hpp>
#include "../CC/adcensuscv.h"
#include "../common.h"

using namespace cv;
using namespace std;

class DisparityRefinement
{
public:
    DisparityRefinement(uint dispTolerance, int dMin, int dMax,
                        uint votingThreshold, float votingRatioThreshold, uint maxSearchDepth,
                        uint blurKernelSize, uint cannyThreshold1, uint cannyThreshold2, uint cannyKernelSize , uint Tn_l, uint Tn_r, uint S_h );
    Mat outlierElimination(const Mat &leftDisp, const Mat &rightDisp);
    void regionVoting(Mat &disparity, const vector<Mat> &upLimits, const vector<Mat> &downLimits,
                      const vector<Mat> &leftLimits, const vector<Mat> &rightLimits, bool horizontalFirst);
    void properInterpolation(Mat &disparity, const Mat &leftImage);
    void discontinuityAdjustment(Mat &disparity, const vector<vector<Mat> > &costs);
    Mat subpixelEnhancement(Mat &disparity, const vector<vector<Mat> > &costs);
	

	void Deal_Occlusion(Mat &disparity,const Mat &leftImage, const Mat &plane_map , float** param_plan);


    static const int DISP_OCCLUSION;
    static const int DISP_MISMATCH;
private:
    int colorDiff(const Vec3b &p1, const Vec3b &p2);
    Mat convertDisp2Gray(const Mat &disparity);

    int occlusionValue;
    int mismatchValue;
    uint dispTolerance;
    int dMin;
    int dMax;
    uint votingThreshold;
    float votingRatioThreshold;
    uint maxSearchDepth;
    uint blurKernelSize;
    uint cannyThreshold1;
    uint cannyThreshold2;
    uint cannyKernelSize;
	//benzlee 2015 1 26
	uint Tn_l;
	uint Tn_r;
	uint S_h;
	int D_l;
 
 
};

#endif // DISPARITYREFINEMENT_H
