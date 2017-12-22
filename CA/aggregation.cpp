#include "aggregation.h"
#include "../common.h"

#define CA_mod	

extern unsigned char kkk;

using namespace std;
#

Aggregation::Aggregation(const Mat &leftImage, const Mat &rightImage, uint colorThreshold1, uint colorThreshold2,
                         uint maxLength1, uint maxLength2,uchar dmax,uchar dmin)
{
    this->images[0] = leftImage;
    this->images[1] = rightImage;
    this->imgSize = leftImage.size();
    this->colorThreshold1 = colorThreshold1;
    this->colorThreshold2 = colorThreshold2;
    this->maxLength1 = maxLength1;
    this->maxLength2 = maxLength2;
    this->upLimits.resize(2);
    this->downLimits.resize(2);
    this->leftLimits.resize(2);
    this->rightLimits.resize(2);
	this->dmax = dmax;
	this->dmin = dmin;
	this->Window_Size= 25;
	this->Sig_clr = 7.65;//0.029;
	this->num_plan = 4;

    for(uchar imageNo = 0; imageNo < 2; imageNo++)
    {
        upLimits[imageNo] = computeLimits(-1, 0, imageNo);
        downLimits[imageNo] = computeLimits(1, 0, imageNo);
        leftLimits[imageNo] = computeLimits(0, -1, imageNo);
        rightLimits[imageNo] = computeLimits(0, 1, imageNo);
    }
	/*
	 this->imgSize = images[0].size();
     temp_costMaps.resize(2);
     for (size_t i = 0; i < 2; i++)
     {
       temp_costMaps[i].resize(abs(dMax - dMin) + 1);
       for(size_t j = 0; j < costMaps[i].size(); j++)
       {
           costMaps[i][j].create(imgSize, COST_MAP_TYPE);
       }
    }*/
 
	//para_volume = new Mat[num_plane];
	param_plan = new float*[num_plan];
	for(int i=0;i<num_plan;i++)
	param_plan[i] = new float[2];
	param_plan[0][0] =   0.0000     , param_plan[0][1] = 0.0000  ; //first group for ka and kb
/*	param_plan[1][0] =   0.0220     , param_plan[1][1] = 0.0170  ;//second group for ka and kb
	param_plan[2][0] =  -0.0288     , param_plan[2][1] = 0.0056  ;//second group for ka and kb 
 	param_plan[3][0] =   0.0066     , param_plan[3][1] = 0.0504 ;
	*/
	param_plan[1][0] =   0.0000     , param_plan[1][1] = 0.3200  ;//second group for ka and kb
	param_plan[2][0] =  -0.1200     , param_plan[2][1] = 0.0000  ;//second group for ka and kb 
 	param_plan[3][0] =   0.1200     , param_plan[3][1] = 0.0000 ;  
/*	param_plan[5][0] =  -0.1199     , param_plan[5][1] = 0.0240   ;
	*/
	
	temp_costMaps.resize(2);
	for (size_t i = 0; i < 2; i++){
	temp_costMaps[i].resize(abs(dmax - 0) + 1);
    for(size_t j = 0; j < temp_costMaps[i].size(); j++)
    {
     //   costMaps[i][j].create(imgSize, COST_MAP_TYPE);
		temp_costMaps[i][j].create(imgSize, CV_32F);
    }
  }
	Size img_size = leftImage.size();
	plane_map = Mat::zeros(img_size,CV_8UC1);
	for(int h =0; h<img_size.height;h++)
	{
		for(int w=0;w<img_size.width;w++)
		{
			plane_map.at<uchar>(h,w) = 0;
		}
	}
/*
	for(uint i=0;i<num_plane;i++)
	{
		para_volume[i] = Mat::zeros(2*maxLength1+1,2*maxLength1+1,CV_32F);
		for(uint x=0;x<=maxLength1;x++){
			for(uint y=0;y<=maxLength1;y++){
			
				float
			}
		}

	
	}
	*/
} 

Mat Aggregation::bl_adap_coagg(const int d,const int imageNo)
{
    //	CV_Assert( p.type() == CV_32FC1 );

	// range parameter is half window size
	int wndSZ  = kkk;//35
	float sig_sp;   //spatial distance
	float sig_clr = Sig_clr;    //color distance0.03*255
	sig_sp = wndSZ / 2.0f;   
	Mat I = images[imageNo];
	int H = I.rows;
	int W = I.cols;
	int H_WD = wndSZ / 2;
 
	Mat p =  temp_costMaps[imageNo][ d ];
	Mat dest = p.clone();
	                  //加快调试 k = 0；
		for( int y = 0; y < H ; y ++ ) {
		 	float* pData =  ( float* ) ( p.ptr<float>( y ) );
			float* destData =  ( float* ) ( dest.ptr<float>( y ) );//目标数组
		//	float* IP    =  ( float* ) ( I.ptr<float>( y ) ); //原图数组
			for( int x = 0; x < W; x ++ ) {
			//	float* pClr = IP + 3 * x;      //原图每次移动三个颜色单位
				Vec3b p1 = I.at<Vec3b>(y, x);
				float dis_pla_temp;
				float dis_pla_resu = 60000.0;
				char dis_pla = 0;
				char dis_displ = 0;
			// 	for(float ka = -0.2f;ka<=0.2f;ka=ka+0.2f)//用来调试，找出问什么内存访问发生冲突的原因 
				for(int  p_i= 0;p_i<num_plan;p_i++)//p_i represent parameters of ith planes
				{//25直线a参数
			 //   for(float kb = -0.2f;kb<=0.2f;kb=kb+0.2f)//
				//	for(float kb = 0;kb<=0.5;kb=kb+1)//
					{	
							float sum = 0.0f;
							float sumWgt = 0.0f;
							for( int  wy = - H_WD; wy <= H_WD; wy = wy+4 ) {  //windows大小
								int qy =  y + wy;
								if( qy < 0 ) {  //处理边界问题
									qy = 0;//qy += H;//benzlee
								}
								if( qy >= H ) { 
									qy = H-1;//qy -= H;
								}
							 	float* qData = ( float* ) ( p.ptr<float>( qy ) ); //cost volume 
							//	float* IQ    = ( float* ) ( I.ptr<float>( qy ) ); // original image
								for( int wx = - H_WD; wx <= H_WD; wx = wx+4 ) {

								//	double kz =  ka*wx+kb*wy+d;
									int qx = x + wx;
									if( qx < 0 ) {
										qx = 0;//qx += W;
										 
									}
									if( qx >= W ) {
										qx = W-1;//qx -= W;
										 
									}
								//	float* qClr = IQ + 3 * qx;
									Vec3b p2 = I.at<Vec3b>(qy, qx);
									float spDis = wx * wx + wy * wy; //distance//如果要优化算法的话，这样的就可以只计算一次
									float clrDis = 0.0f;
									for( int c = 0; c < 3; c++ ) {
									//	clrDis += fabs( pClr[ c ] - qClr[ c ] );  //difference between colors
										clrDis += abs( p1[ c ] - p2[ c ] );
									}
									///////get the result of the disparity plane  modified by benzlee
									#ifdef CA_mod
									float kz =  param_plan[p_i][0]*wx+param_plan[p_i][1]*wy+d;//ka to 0;and kb to 1   
									int integ_kz_1 = int(kz);//内存多就是任性
								 	int integ_kz_2 = integ_kz_1+1;
									float fron = float(kz-integ_kz_1);
								//	double backf = double(1-fron);
									if(kz<=dmin)      //stupid guy keep remember that disparity cannot be zero.
									{
										kz = dmin;
										integ_kz_1=dmin;
									 	integ_kz_2=dmin;
									}
									if(integ_kz_2>=(dmax-1))  //千万不能用kz
									{
										kz = dmax-1;
										integ_kz_1 =  dmax-1;
									 	integ_kz_2 =  dmax-1;
									}
									float* down_ad = ( float* ) (  temp_costMaps[imageNo][ integ_kz_1 ].ptr<float>( qy ) ); 
									float* up_ad = ( float* ) (  temp_costMaps[imageNo][ integ_kz_2 ].ptr<float>( qy ) );
									clrDis *= 0.333333333;
									float wgt = exp( - spDis / ( sig_sp * sig_sp ) - clrDis * clrDis / ( sig_clr * sig_clr ) );//内存有益处
								/*	if(fron<0.5){	//临近插值法
									down_data = spe_costVol[ integ_kz_1 ].at<double>(qy , qx);
									}else{
									down_data = spe_costVol[ integ_kz_2 ].at<double>(qy , qx);
									}*/
									//double down_data = spe_costVol[ integ_kz_1 ].at<double>(qy , qx);
									//double up_data = spe_costVol[ integ_kz_2 ].at<double>(qy , qx);
									//sum += wgt * (down_data+fron*(up_data-down_data));
						 			sum += wgt * (down_ad[ qx ]+fron*(up_ad[ qx ]-down_ad[ qx ])); 
									sumWgt += wgt;
									#else
									///////////
									clrDis *= 0.333333333;
									double wgt = exp( - spDis / ( sig_sp * sig_sp ) - clrDis * clrDis / ( sig_clr * sig_clr ) );
									sum += wgt * qData[ qx ];
									sumWgt += wgt;
									#endif
								}  //wx
							}    //wy
							dis_pla_temp = sum / sumWgt;
							dis_pla++;
							if(dis_pla_resu>dis_pla_temp)//gain the minimum
							{
								dis_pla_resu = dis_pla_temp; //dis_pla_resu always save the minimum;
								if(imageNo==0)
								 plane_map.at<uchar>(y,x) = p_i;
							//	dis_displ = dis_pla;	  //debug to display the results;					
							}
						}    //kb						
					}   //ka
					destData[ x ] = dis_pla_resu;
				 //	printf("%d-",dis_displ);
			}   //x
		}    //y
		return dest; //pass the processed data to the cosVol
 

}

int Aggregation::colorDiff(const Vec3b &p1, const Vec3b &p2)
{
    int colorDiff, diff = 0;

    for(uchar color = 0; color < 3; color++)
    {
        colorDiff = std::abs(p1[color] - p2[color]);
        diff = (diff > colorDiff)? diff: colorDiff;
    }

    return diff;
}

int Aggregation::computeLimit(int height, int width, int directionH, int directionW, uchar imageNo)
{
    // reference pixel
    Vec3b p = images[imageNo].at<Vec3b>(height, width);

    // coordinate of p1 the border patch pixel candidate
    int d = 1;
    int h1 = height + directionH;
    int w1 = width + directionW;

    // pixel value of p1 predecessor
    Vec3b p2 = p;

    // test if p1 is still inside the picture
    bool inside = (0 <= h1) && (h1 < imgSize.height) && (0 <= w1) && (w1 < imgSize.width);

    if(inside)
    {
        bool colorCond = true, wLimitCond = true, fColorCond = true;

        while(colorCond && wLimitCond && fColorCond && inside)
        {
            Vec3b p1 = images[imageNo].at<Vec3b>(h1, w1);

            // Do p1, p2 and p have similar color intensities?
            colorCond = colorDiff(p, p1) < colorThreshold1 && colorDiff(p1, p2) < colorThreshold1;

            // Is window limit not reached?
            wLimitCond = d < maxLength1;

            // Better color similarities for farther neighbors?
            fColorCond = (d <= maxLength2) || (d > maxLength2 && colorDiff(p, p1) < colorThreshold2);

            p2 = p1;
            h1 += directionH;
            w1 += directionW;

            // test if p1 is still inside the picture
            inside = (0 <= h1) && (h1 < imgSize.height) && (0 <= w1) && (w1 < imgSize.width);

            d++;
        }

        d--;
    }

    return d - 1;
}

Mat Aggregation::computeLimits(int directionH, int directionW, int imageNo)
{
    Mat limits(imgSize, CV_32S);
    int h, w;
    #pragma omp parallel default (shared) private(w, h) num_threads(omp_get_max_threads())
    #pragma omp for schedule(static)
    for(h = 0; h < imgSize.height; h++)
    {
        for(w = 0; w < imgSize.width; w++)
        {
            limits.at<int>(h, w) = computeLimit(h, w, directionH, directionW, imageNo);
        }
    }

    return limits;
}

Mat Aggregation::aggregation1D(const Mat &costMap, int directionH, int directionW, Mat &windowSizes, uchar imageNo,uchar dis)
{
    Mat tmpWindowSizes = Mat::zeros(imgSize, CV_32S);
    Mat aggregatedCosts(imgSize, CV_32F);
  //  int w_dmin, w_dmax,h_dmin, h_dmax, d;
   // int h, w;
	float temp_min;
	float* cost;
    int h_dmin, h_dmax,w_dmin, w_dmax ,h_d ,w_d ;
    int h, w ,temp;
	cost = new float[num_plan];

    #pragma omp parallel default (shared) private(w, h) num_threads(omp_get_max_threads())
    #pragma omp for schedule(static)
    for(h = 0; h < imgSize.height; h++)
    {
		int i =66;
        for(w = 0; w < imgSize.width; w++)
        {
            if(directionH == 0)
            {
                w_dmin = -leftLimits[imageNo].at<int>(h, w);
                w_dmax = rightLimits[imageNo].at<int>(h, w);
				for(uchar k=0;k<num_plan;k++)
				{
					cost[k] = 0;
				}
				for(w_d= w_dmin;w_d<=w_dmax;w_d++)
				{
 
					h_dmin =  -upLimits[imageNo].at<int>(h, w+w_d);
					h_dmax = downLimits[imageNo].at<int>(h, w+w_d);
					for(h_d =h_dmin;h_d<=h_dmax;h_d++)
					{
						for(uchar k=0;k<num_plan;k++)
						{
							float temp_d = param_plan[k][0]*w_d+param_plan[k][1]*h_d+dis;
							int integ_d1 = int(temp_d);
					    	if(integ_d1<=1)   //disparity cannot be zero
					    	{
						    	integ_d1 = 1;
						    	temp_d   = 1 ;
					    	}
					    	if((integ_d1+1)>=(dmax-dmin))
						   {
						    	integ_d1 = dmax -dmin-1;
						    	temp_d   = dmax-dmin-1;
					    	}
							float front = temp_d - integ_d1;
							//size_t dd = size_t(integ_d1);
							cost[k]+= temp_costMaps[imageNo][integ_d1].at<float>(h+h_d,w+w_d) + front*(temp_costMaps[imageNo][integ_d1+1].at<float>(h+h_d,w+w_d) -temp_costMaps[imageNo][integ_d1].at<float>(h+h_d,w+w_d));//temp_costMaps[imageNo][integ_d1].at<costType>(h, w)+;

						}//k
						tmpWindowSizes.at<int>(h, w) += 1;//windowSizes.at<int>(h + d * di
					}//h-d
				}//w-d
					temp_min = 3.40e+37;
					for(uchar k=0;k<num_plan;k++)  //的带cost中最小值，复制给
			       {
			        	if(cost[k]<temp_min)
				        {
							temp_min = cost[k];
							if(imageNo == 0)   //只计算leftimage
							plane_map.at<uchar>(h,w) = k;
						}
         			}
					aggregatedCosts.at<float>(h, w) = temp_min;
            }//if
            else
            {
				h_dmin =  -upLimits[imageNo].at<int>(h, w);
				h_dmax = downLimits[imageNo].at<int>(h, w);
 
				for(uchar k=0;k<num_plan;k++)
				{
					cost[k] = 0;
				}
				for(h_d= h_dmin;h_d<=h_dmax;h_d++)
				{
					//h_dmin =  -upLimits[imageNo].at<int>(h, w);
					//h_dmax = downLimits[imageNo].at<int>(h, w);
					w_dmin = -leftLimits[imageNo].at<int>(h+h_d, w);
					w_dmax = rightLimits[imageNo].at<int>(h+h_d, w);
					for(w_d =w_dmin;w_d<=w_dmax;w_d++)
					{
						for(uchar k=0;k<num_plan;k++)
						{
							float temp_d = param_plan[k][0]*w_d+param_plan[k][1]*h_d+dis;
							int integ_d1 = int(temp_d);
					    	if(integ_d1<=1)   //disparity cannot be zero
					    	{
						    	integ_d1 = 1;
						    	temp_d   = 1;
					    	}
					    	if((integ_d1+1)>=(dmax-dmin))
						   {
						    	integ_d1 = dmax -dmin - 1;
						    	temp_d   = dmax- dmin - 1;
					    	}
							float front = temp_d - integ_d1;
						//	float fir_dat = temp_costMaps[imageNo][integ_d1].at<costType>(h+h_d,w+w_d)/COST_FACTOR;
						//	float sec_dat = temp_costMaps[imageNo][integ_d1+1].at<costType>(h+h_d,w+w_d)/COST_FACTOR;
							cost[k]+= temp_costMaps[imageNo][integ_d1].at<float>(h+h_d,w+w_d) + front*(temp_costMaps[imageNo][integ_d1+1].at<float>(h+h_d,w+w_d) -temp_costMaps[imageNo][integ_d1].at<float>(h+h_d,w+w_d));//temp_costMaps[imageNo][integ_d1].at<costType>(h, w)+;

						}//k
						tmpWindowSizes.at<int>(h, w) += 1;//windowSizes.at<int>(h + d * di
					}//h-d
				}//w-d
					temp_min = 3.40e+37;
					for(uchar k=0;k<num_plan;k++)  //的带cost中最小值，复制给
			       {
			        	if(cost[k]<temp_min)
				        {
							temp_min = cost[k];
							if(imageNo == 0)   //只计算leftimage
							plane_map.at<uchar>(h,w) = k;
						}
         			}
					aggregatedCosts.at<float>(h, w) = temp_min;
            }
        /*
            float cost = 0;
            for(h_d = h_dmin; h_d <= h_dmax; h_d++)
            {
                cost += costMap.at<float>(h + d * directionH, w + d * directionW);
                tmpWindowSizes.at<int>(h, w) += windowSizes.at<int>(h + d * directionH, w + d * directionW);
            }
            aggregatedCosts.at<float>(h, w) = cost;
			*/
        }//x
    }//y

    tmpWindowSizes.copyTo(windowSizes);

    return aggregatedCosts;
}

void Aggregation::aggregation2D(Mat &costMap, bool horizontalFirst, uchar imageNo,uchar dis)
{
    int directionH = 1, directionW = 0;

    if (horizontalFirst)
        std::swap(directionH, directionW);

    Mat windowsSizes = Mat::ones(imgSize, CV_32S);

    for(uchar direction = 0; direction < 1; direction++)  //direction < 2
    {
        (aggregation1D(costMap, directionH, directionW, windowsSizes, imageNo,  dis)).copyTo(costMap);
        std::swap(directionH, directionW);
    }

    for(size_t h = 0; h < imgSize.height; h++)
    {
        for(size_t w = 0; w < imgSize.width; w++)
        {
            costMap.at<float>(h, w) /= windowsSizes.at<int>(h, w);
        }
    }

}

void Aggregation::getLimits(vector<Mat> &upLimits, vector<Mat> &downLimits,
                            vector<Mat> &leftLimits, vector<Mat> &rightLimits) const
{
    upLimits = this->upLimits;
    downLimits = this->downLimits;
    leftLimits = this->leftLimits;
    rightLimits = this->rightLimits;
}

