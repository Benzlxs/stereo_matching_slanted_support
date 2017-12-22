//创建于 21/3/2015
//研究 stereo matching, 李雪松 Xuesong Li

#include <iostream>
#include "stereoprocessor.h"
#include "imageprocessor.h"
#include <cstdlib>
#include <iomanip>

using namespace std;
using namespace cv;

#define KITTI

unsigned char kkk;

int main(int argc, char *argv[])
{
//  string xmlImages, ymlExtrinsic;
	uint dMin; uint dMax; Size censusWin; float defaultBorderCost;
    float lambdaAD; float lambdaCensus; string savePath; uint aggregatingIterations;
    uint colorThreshold1; uint colorThreshold2; uint maxLength1; uint maxLength2; uint colorDifference;
    float pi1; float pi2; uint dispTolerance; uint votingThreshold; float votingRatioThreshold;
    uint maxSearchDepth; uint blurKernelSize; uint cannyThreshold1; uint cannyThreshold2; uint cannyKernelSize;
    
	
	dMin = 0;
    dMax = 70;//107;//图像中最大视差值

    censusWin.height = 9;
    censusWin.width =  7;
    defaultBorderCost = 0.9990;
    lambdaAD =  10.0; // TODO Namen anpassen
    lambdaCensus = 30.0;
    savePath =  "../results/";
    aggregatingIterations = 1;//4
    colorThreshold1 = 20;
    colorThreshold2 = 6 ;
    maxLength1 = 34;
    maxLength2 = 17;
    colorDifference = 17;
    pi1 = 0.1;
    pi2 = 0.3;
    dispTolerance = 0;
    votingThreshold = 20;
    votingRatioThreshold = 0.4;
    maxSearchDepth = 20;
    blurKernelSize = 3;
    cannyThreshold1 = 20;
    cannyThreshold2 = 60;
    cannyKernelSize = 3;	
	#ifdef KITTI
		string img_l_file = "F:\\Datasets\\KITTI\\l_";
		string img_r_file = "F:\\Datasets\\KITTI\\r_";
	#else
		string img_l_file = "F:\\Datasets\\Middlebury\\l_";
		string img_r_file = "F:\\Datasets\\Middlebury\\r_";
	#endif
	string img_png = ".png";
	vector<Mat> images;//store the inmage
	uint numb_img =6;
	uint line = 60;
	kkk = 49;

	for(int i = 5;i< numb_img;i++)
	{
		stringstream lStr,rStr;
		//int j=5;
		lStr<<img_l_file<<i<< img_png;
		rStr<<img_r_file<<i<< img_png;
		string ldir = lStr.str();
		string rdir = rStr.str();

		Mat lmag = imread(ldir);
		Mat rmag = imread(rdir);

		images.push_back(lmag);
		images.push_back(rmag);
	 }

	 bool error = false;

	 for (int i = 0; i < (images.size() / 2) && !error; ++i){
		stringstream file;
		file << savePath << i;
		ImageProcessor iP(0.1); //创建ImageProcessor 类
		Mat eLeft, eRight;
		eLeft = iP.unsharpMasking(images[i * 2], "gauss", 3, 1.9, -1);
		eRight = iP.unsharpMasking(images[i * 2 + 1], "gauss", 3, 1.9, -1);

		StereoProcessor sP(dMin, dMax, images[i * 2], images[i * 2 + 1], censusWin, defaultBorderCost, lambdaAD, lambdaCensus, file.str(),
                                               aggregatingIterations, colorThreshold1, colorThreshold2, maxLength1, maxLength2,
                                               colorDifference, pi1, pi2, dispTolerance, votingThreshold, votingRatioThreshold,
                                               maxSearchDepth, blurKernelSize, cannyThreshold1, cannyThreshold2, cannyKernelSize);
		string errorMsg;
		error = !sP.init(errorMsg);   //create the memory

	  if(!error&&sP.compute())
	  {
		Mat disp = sP.getDisparity();
		stringstream fStr;
		string dest = "_result_l.png";
		fStr<<i<<dest;
		imwrite(fStr.str(),disp);
	  }

}

return 0;
}