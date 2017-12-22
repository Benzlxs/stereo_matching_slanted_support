#ifndef STEREOPROCESSOR_H
#define STEREOPROCESSOR_H
#include "CC/adcensuscv.h"
#include "CA/aggregation.h"
#include "DC/scanlineoptimization.h"
#include "DR/disparityrefinement.h"
#include <omp.h>
#include "common.h"

 

using namespace std;

class StereoProcessor
{
public:
    StereoProcessor(uint dMin, uint dMax, Mat leftImage, Mat rightImage, Size censusWin, float defaultBorderCost,
                    float lambdaAD, float lambdaCensus, string savePath, uint aggregatingIterations,
                    uint colorThreshold1, uint colorThreshold2, uint maxLength1, uint maxLength2, uint colorDifference,
                    float pi1, float pi2, uint dispTolerance, uint votingThreshold, float votingRatioThreshold,
                    uint maxSearchDepth, uint blurKernelSize, uint cannyThreshold1, uint cannyThreshold2, uint cannyKernelSize);
    ~StereoProcessor();
    bool init(string &error);
    bool compute();
    Mat getDisparity();

private:
    int dMin;
    int dMax;
    Mat images[2];
    Size censusWin;
    float defaultBorderCost;
    float lambdaAD;
    float lambdaCensus;
    string savePath;
    uint aggregatingIterations;
    uint colorThreshold1;
    uint colorThreshold2;
    uint maxLength1;
    uint maxLength2;
    uint colorDifference;
    float pi1;
    float pi2;
    uint dispTolerance;
    uint votingThreshold;
    float votingRatioThreshold;
    uint maxSearchDepth;
    uint blurKernelSize;
    uint cannyThreshold1;
    uint cannyThreshold2;
    uint cannyKernelSize;
    bool validParams, dispComputed;

	

    vector<vector<Mat> > costMaps;
    Size imgSize;
    ADCensusCV *adCensus;
    Aggregation *aggregation;
    Mat disparityMap, floatDisparityMap;
    DisparityRefinement *dispRef;

    void costInitialization();
    void costAggregation();
    void scanlineOptimization();
    void outlierElimination();
	void deal_occlusion();   //benzleee  2015-1-26
    void regionVoting();
    void properInterpolation();
    void discontinuityAdjustment();
    void subpixelEnhancement();

	/*得到第二个视差图*/
	void outlierElimination_2();
    void regionVoting_2();
    void properInterpolation_2();
    void discontinuityAdjustment_2();
    void subpixelEnhancement_2();


    Mat cost2disparity(int imageNo);

    template <typename T>
    void saveDisparity(const Mat &disp, string filename, bool stretch = true);
};

#endif // STEREOPROCESSOR_H
