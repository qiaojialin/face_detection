/**
* @face_detection.cpp
* @author:wolf
* 这个demo使用opencv实现图片的人脸检测、以及绘画框出人脸、截取人脸等功能。
* 每个功能写成一个函数，方便移植使用。
* 参考：opencv基本绘画、物体检测模块文档。
*/
#include"tinyxml.h"
#include"tinystr.h"
#include "cv.h"
#include<fstream>
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "cxcore.h"
#include <stdlib.h>
#include"opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include<iostream>
#include<sstream>
#include<string>
#include<math.h>
using namespace cv;
using namespace std;

int FACES = 500;    //训练集人脸数
int NONFACES = 800;   //训练集非人脸数

int FACES_TEST = FACES;
int NONFACES_TEST = NONFACES;

#define WEAK_CLASSIFIERS 300   //每此训练最多存放300个弱分类器，强分类器最多包含300个弱分类器
#define THICKNESS  2       //画出的人脸框的粗细



//训练集图像结构体
struct Image
{
	Mat img;						//灰度图像矩阵
	float weight;					//每幅图的权重
	int flag;						//人脸为1，非人脸为0
	int img_integral[21][21];   	 //积分图像矩阵 1-FACES+NONFACES
	int feature_value;				//对应一个矩形特征的特征值
	float strong_value;				//强分类器算出来的特征值
}image[1501];	    //下标范围为1-1500


//测试集图像结构体
struct ImageTest
{
	Mat img;						//灰度图像矩阵
	float weight;					//每幅图的权重
	int flag;						//人脸为1，非人脸为0
	int img_integral[313][425];   	 //积分图像矩阵 1-FACES+NONFACES
	int feature_value;				//对应一个矩形特征的特征值
	float strong_value;				//强分类器算出来的特征值
};	   


Image P[301];  //当前级联分类器的人脸集
Image N[301];   //当前级联分类器的非人脸集
Image N_test[304];



//矩形特征结构体
struct Feature
{
	int templet;  //特征的类别
	int x_times;  //特征的X轴放大倍数
	int y_times;  //特征的Y轴放大倍数
	int x;        //特征的起始x坐标
	int y;        //特征的起始y坐标
};



//弱分类器结构体
struct WeakClassifier
{
	Feature feature;   //一个特征
	int threshold;     //阈值
	int p;             //调整方向 使判别不等式统一为小于号，即小于阈值为人脸
	float error_rate;  //错误率
	float alpha;      //弱分类器在强分类器中的权重
}wc[WEAK_CLASSIFIERS];             //下标为0----WEAK_CLASSIFIERS-1
int wc_index = 0;



//强分类器结构体
struct StrongClassifier
{
	WeakClassifier weakclassifier[WEAK_CLASSIFIERS];   //包含的弱分类器 最多WEAK_CLASSIFIERS个
	int number; //弱分类器个数
	float threshold;     //强分类器阈值  大于等于为人脸  小于为非人脸
};



//级联分类器结构体
struct CascadedClassifier
{
	StrongClassifier strongclassifier[51];  //每一级的强分类器   下标1-50  最多50层强分类器
	int number;     //级联层数
	float d;          //每一层可接受的最小灵敏度 
	float f;           //每一层可接受的最大误识率
	float f_target;    //级联分类器的目标误识率
}cc;



//人脸结构体,用于合并人脸
struct Face
{
	int x;  //人脸的起始x坐标
	int y;  //人脸的起始y坐标
	int length;  //人脸的宽度
	int number;  //这张人脸的个数  用于和其他人脸合并的权值
}face[1000];
int face_index = 0;



//读取MIT人脸库FACES+NONFACES张
void loadImg()
{
	 int i; 
	 //读取FACES张人脸灰度图 
	 for( i = 1; i <= FACES; i++ )
	 {
		 stringstream buffer;
         buffer<<i;
		 image[i].img = imread("faces\\"+buffer.str()+".bmp",0);
		 image[i].flag = 1;
	//	 P[i] = image[i];
	 }

	 //读取NONFACES张非人脸灰度图
	 for( i = 1; i <= NONFACES; i++ )
	 {
		 stringstream buffer;
         buffer<<i;
		 image[i+FACES].img = imread("nonfaces\\"+buffer.str()+".bmp",0);
		 image[i+FACES].flag = 0;
	//	 N[i] = image[i+FACES];
	//	 N_test[i] = image[i+FACES];
	 }

}



//计算一张图像的积分图像
void oneImageIntegral(Image &image)
{
	int x,y;
	//一列一列从左到右扫描
	for( x = 0; x < image.img.cols; x++ )
		for( y = 0; y < image.img.rows; y++ )
		{
			//（0,0）点直接返回其像素值
			if( x == 0 && y == 0)
				image.img_integral[y][x] = (int)image.img.at<uchar>(0,0);

			//Y轴上的点=上邻点+此点的像素
			else if( x == 0 )
				image.img_integral[y][x] = image.img_integral[y-1][0] + (int)image.img.at<uchar>(y,0);

			//X轴上的点=左邻点+此点像素
			else if( y == 0 )
				image.img_integral[y][x] = image.img_integral[0][x-1] + (int)image.img.at<uchar>(0,x);

			//其他点=左邻点+上邻点-左上点+此点像素
			else
				image.img_integral[y][x] = image.img_integral[y][x-1] + image.img_integral[y-1][x]   \
											 - image.img_integral[y-1][x-1] + (int)image.img.at<uchar>(y,x);
		}	 
}



//计算一张图像的积分图像
void drawoneImageIntegral(ImageTest &image)
{
	int x,y;
	//一行一行从左到右扫
	for( x = 0; x < image.img.cols; x++ )
		for( y = 0; y < image.img.rows; y++ )
		{
			//（0,0）点直接返回其像素值
			if( x == 0 && y == 0)
				image.img_integral[y][x] = (int)image.img.at<uchar>(0,0);

			//Y轴上的点=上邻点+此点的像素
			else if( x == 0 )
				image.img_integral[y][x] = image.img_integral[y-1][0] + (int)image.img.at<uchar>(y,0);

			//X轴上的点=左邻点+此点像素
			else if( y == 0 )
				image.img_integral[y][x] = image.img_integral[0][x-1] + (int)image.img.at<uchar>(0,x);

			//其他点=左邻点+上邻点-左上点+此点像素
			else
				image.img_integral[y][x] = image.img_integral[y][x-1] + image.img_integral[y-1][x]   \
											 - image.img_integral[y-1][x-1] + (int)image.img.at<uchar>(y,x);
		}	
}



//计算所有图像的积分图像
void imageIntegral()
{
	for( int index = 1; index <= FACES+NONFACES; index++ )
		oneImageIntegral(image[index]);
}



//计算一张图片相对于一个特征的特征值
int oneFeatureValue(Image image, float times, Feature feature)
{
	int templet = feature.templet;
	int x_times = feature.x_times;
	int y_times = feature.y_times;
	int x1 = feature.x;
	int y1 = feature.y;
	int feature_value = 0;

	//(s,t)特征为(1,2)的矩形特征，类别号为1
	if( templet == 1 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = (int)(1 * x_times * times);
		int y = (int)(2 * y_times * times);
		feature_value = 2 * ( image.img_integral[ y1 + y / 2 ][ x1 + x ] - image.img_integral[ y1 + y / 2 ][ x1 ] )  \
								- ( image.img_integral[ y1 ][ x1 + x ] - image.img_integral[ y1 ][ x1 ] )                    \
								- ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 + y ][ x1 ] );

	}

	//(s,t)特征为(2,1)的矩形特征,类别号为2
	if( templet == 2 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = (int)(2 * x_times * times);
		int y = (int)(1 * y_times * times);
		feature_value = 2 * ( image.img_integral[ y1 + y ][ x1 + x / 2 ] - image.img_integral[ y1 ][ x1 + x / 2 ] )    \
								- ( image.img_integral[ y1 + y ][ x1 ] - image.img_integral[ y1 ][ x1 ] )              \
								- ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 ][ x1 + x ] );
	}

	//(s,t)特征为(1,3)的矩形特征,类别号为3
	if( templet == 3 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = (int)(1 * x_times * times);
		int y = (int)(3 * y_times * times);
		feature_value = 3 * ( image.img_integral[ y1 + y / 3 ][ x1 + x ] - image.img_integral[ y1 + y / 3 ][ x1 ] )         \
								- 3 * ( image.img_integral[ y1 + 2 * y / 3 ][ x1 + x ] - image.img_integral[ y1 + 2 * y / 3 ][ x1 ] )    \
								+ ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 + y ][ x1 ] )                        \
								- ( image.img_integral[ y1 ][ x1 + x ] - image.img_integral[ y1 ][ x1 ] );		
	}

	//(s,t)特征为(3,1)的矩形特征,类别号为4
	if( templet == 4 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = (int)(3 * x_times * times);
		int y = (int)(1 * y_times * times);
		feature_value = 3 * ( image.img_integral[ y1 + y ][ x1 + x / 3 ] - image.img_integral[ y1 ][ x1 + x / 3 ] )      \
								- 3 * ( image.img_integral[ y1 + y ][ x1 + 2 * x / 3 ] - image.img_integral[ y1 ][ x1 + 2 * x / 3 ] )     \
								+ ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 ][ x1 + x ] )                         \
								- ( image.img_integral[ y1 + y ][ x1 ] - image.img_integral[ y1 ][ x1 ] );	
	}

	//(s,t)特征为(2,2)的矩形特征,类别号为5
	if( templet == 5 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = (int)(2 * x_times * times);
		int y = (int)(2 * y_times * times);
		feature_value = 4 * image.img_integral[ y1 + y / 2 ][ x1 + x / 2 ]                                                   \
									+ image.img_integral[ y1 ][ x1 ]  + image.img_integral[ y1 ][ x1 + x ]                            \
									+ image.img_integral[ y1 + y ][ x1 ] + image.img_integral[ y1 + y ][ x1 + x ]                     \
									- 2 * ( image.img_integral[ y1 + y / 2 ][ x1 + x ] + image.img_integral[ y1 + y / 2 ][ x1 ] )     \
									- 2 * ( image.img_integral[ y1 + y ][ x1 + x / 2 ] + image.img_integral[ y1 ][ x1 + x / 2 ] );		
	}
	return feature_value;
}




//计算一张图片相对于一个特征的特征值
void drawoneFeatureValue(ImageTest &image, float times, Feature feature, int x0, int y0)
{
	
	int templet = feature.templet;
	int x_times = feature.x_times;
	int y_times = feature.y_times;
	int x1 = feature.x + x0;
	int y1 = feature.y + y0;
	int feature_value = 0;

	//(s,t)特征为(1,2)的矩形特征，类别号为1
	if( templet == 1 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = (int)(1 * x_times * times);
		int y = (int)(2 * y_times * times);
		feature_value = 2 * ( image.img_integral[ y1 + y / 2 ][ x1 + x ] - image.img_integral[ y1 + y / 2 ][ x1 ] )  \
								- ( image.img_integral[ y1 ][ x1 + x ] - image.img_integral[ y1 ][ x1 ] )                    \
								- ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 + y ][ x1 ] );

	}

	//(s,t)特征为(2,1)的矩形特征,类别号为2
	if( templet == 2 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = (int)(2 * x_times * times);
		int y = (int)(1 * y_times * times);
		feature_value = 2 * ( image.img_integral[ y1 + y ][ x1 + x / 2 ] - image.img_integral[ y1 ][ x1 + x / 2 ] )    \
								- ( image.img_integral[ y1 + y ][ x1 ] - image.img_integral[ y1 ][ x1 ] )              \
								- ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 ][ x1 + x ] );
	}

	//(s,t)特征为(1,3)的矩形特征,类别号为3
	if( templet == 3 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = (int)(1 * x_times * times);
		int y = (int)(3 * y_times * times);
		feature_value = 3 * ( image.img_integral[ y1 + y / 3 ][ x1 + x ] - image.img_integral[ y1 + y / 3 ][ x1 ] )         \
								- 3 * ( image.img_integral[ y1 + 2 * y / 3 ][ x1 + x ] - image.img_integral[ y1 + 2 * y / 3 ][ x1 ] )    \
								+ ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 + y ][ x1 ] )                        \
								- ( image.img_integral[ y1 ][ x1 + x ] - image.img_integral[ y1 ][ x1 ] );		
	}

	//(s,t)特征为(3,1)的矩形特征,类别号为4
	if( templet == 4 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = (int)(3 * x_times * times);
		int y = (int)(1 * y_times * times);
		feature_value = 3 * ( image.img_integral[ y1 + y ][ x1 + x / 3 ] - image.img_integral[ y1 ][ x1 + x / 3 ] )      \
								- 3 * ( image.img_integral[ y1 + y ][ x1 + 2 * x / 3 ] - image.img_integral[ y1 ][ x1 + 2 * x / 3 ] )     \
								+ ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 ][ x1 + x ] )                         \
								- ( image.img_integral[ y1 + y ][ x1 ] - image.img_integral[ y1 ][ x1 ] );	
	}

	//(s,t)特征为(2,2)的矩形特征,类别号为5
	if( templet == 5 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = (int)(2 * x_times * times);
		int y = (int)(2 * y_times * times);
		feature_value = 4 * image.img_integral[ y1 + y / 2 ][ x1 + x / 2 ]                                                   \
									+ image.img_integral[ y1 ][ x1 ]  + image.img_integral[ y1 ][ x1 + x ]                            \
									+ image.img_integral[ y1 + y ][ x1 ] + image.img_integral[ y1 + y ][ x1 + x ]                     \
									- 2 * ( image.img_integral[ y1 + y / 2 ][ x1 + x ] + image.img_integral[ y1 + y / 2 ][ x1 ] )     \
									- 2 * ( image.img_integral[ y1 + y ][ x1 + x / 2 ] + image.img_integral[ y1 ][ x1 + x / 2 ] );		
	}
	
	image.feature_value = feature_value;
}



//把图像根据弱分类器特征值从小到大排序
void sortImageByFeatureValue()
{
	int i,j,k;
	for( i = 1; i < FACES + NONFACES; i++ )
	{
		k = i;
		for( j = i + 1; j <= FACES + NONFACES; j++ )
			if( image[j].feature_value < image[k].feature_value )
				k = j;
		if( k != i)
		{
			Image tmp = image[i];
			image[i] = image[k];
			image[k] = tmp;
		}
	}
}



//把图像根据强分类器特征值从小到大排序
void sortImageByStrongValue()
{
	int i,j,k;
	for( i = 1; i < FACES + NONFACES; i++ )
	{
		k = i;
		for( j = i + 1; j <= FACES + NONFACES; j++ )
			if( image[j].strong_value < image[k].strong_value )
				k = j;
		if( k != i)
		{
			Image tmp = image[i];
			image[i] = image[k];
			image[k] = tmp;
		}
	}
}



//通过特征feature训练弱分类器
void createWeakClassifier(WeakClassifier &weakclassifier, Feature feature)
{
	//根据矩形特征feature计算所有图片的特征值
	for( int i = 1; i <= FACES + NONFACES; i++ )
		image[i].feature_value = oneFeatureValue(image[i], 1, feature);


	//将图像特征值排序
	sortImageByFeatureValue();

	weakclassifier.threshold = image[0].feature_value;  //当前阈值
	weakclassifier.error_rate = (float)100;             //当前最小错误率
	weakclassifier.feature = feature;                   //分类器对应的矩形特征

	float faces_weight = 0;         //全部人脸样本权重
	float nonfaces_weight = 0;		//全部非人脸样本权重

	float faces_weight_before = 0;		     //在当前阈值之前的人脸权重和
	float nonfaces_weight_before = 0;		 //在当前阈值之前的非人脸权重和

	//统计此弱分类器构造环境中的所有人脸样本权重和faces_weight、非人脸样本权重和nonfaces_weight
	for(int i = 1; i <= FACES + NONFACES; i++)
	{
		if(image[i].flag == 1)
			faces_weight += image[i].weight;
		else if(image[i].flag == 0)
			nonfaces_weight += image[i].weight;
	}

	//计算此弱分类器的所有阈值可能，取错误率最小的阈值作为此弱分类器的阈值
	for(int i = 1; i <= FACES + NONFACES; i++)
	{
		//每次清空
		faces_weight_before = 0;
		nonfaces_weight_before = 0;

		//统计在当前元素之前的人脸样本权重和、非人脸样本权重和
		for(int j = 1; j <= i; j++ )  
		{
			if(image[j].flag == 1)  //人脸
				faces_weight_before += image[j].weight;
			else   //非人脸
				nonfaces_weight_before += image[j].weight;
		}

		//求最小错误率
		float e1,e2;
		e1 = faces_weight_before + ( nonfaces_weight - nonfaces_weight_before ); //非人脸 <= 阈值 < 人脸 的错误率  p_current = -1
		e2 = nonfaces_weight_before + ( faces_weight - faces_weight_before );   //人脸 <= 阈值 < 非人脸 的错误率  p_current = 1
		

		float error_current = 0;  //设置当前阈值的错误率
		int p_current;
		if( e1 <= e2 )
		{
			error_current = e1;
			p_current = -1;
		}
		else
		{
			error_current = e2;
			p_current = 1;
		}

		//当前错误率若小于分类器错误率，更新当前弱分类器的错误率，调节因子，阈值
		if( error_current < weakclassifier.error_rate )
		{
			weakclassifier.error_rate = error_current; 
			weakclassifier.p = p_current;
			weakclassifier.threshold = image[i].feature_value;
			
		}
	}

	float beta= weakclassifier.error_rate / (1 - weakclassifier.error_rate);  //调节因子
	weakclassifier.alpha = log(1/beta);

	float sum_weight = 0;  //调整后的权重和
	

	//调节权重，将分对的图片权重乘一个因子beta  原来的错误率和越高，beta越大
	for( int i = 1; i <= FACES + NONFACES; i++ )
	{
		if( image[i].feature_value * weakclassifier.p <= weakclassifier.threshold * weakclassifier.p && image[i].flag == 1 ) //分对的人脸
			image[i].weight *= beta;
		if( image[i].feature_value * weakclassifier.p >= weakclassifier.threshold * weakclassifier.p && image[i].flag == 0 ) //分对的非人脸
			image[i].weight *= beta;
		sum_weight += image[i].weight;
	}

	//归一化权重
	for( int i = 1; i <= FACES + NONFACES; i++)
		image[i].weight /= sum_weight;
	
}



//用一个弱分类器对一张图片进行探测,弱分类器判别公式  times为探测窗口放大倍数
int drawdetectFaceByWeakClassifier(ImageTest &image, float times, WeakClassifier weakclassifier, int x, int y)
{
	//计算图片相对于这个弱分类器的特征值
	drawoneFeatureValue(image, times, weakclassifier.feature, x, y);

	if(image.feature_value == -100)
		return 0;

	//根据弱分类器判别公式判断图片是否有人脸
	if( image.feature_value * weakclassifier.p <= weakclassifier.threshold * times * weakclassifier.p )
		return 1;

	return 0;
}



//用一个弱分类器对一张图片进行探测,弱分类器判别公式  times为探测窗口放大倍数
int detectFaceByWeakClassifier(Image &image, float times, WeakClassifier weakclassifier)
{
	//计算积分图
	oneImageIntegral(image);

	//计算图片相对于这个弱分类器的特征值
	image.feature_value = oneFeatureValue(image, times, weakclassifier.feature);

	//根据弱分类器判别公式判断图片是否有人脸
	if( image.feature_value * weakclassifier.p <= weakclassifier.threshold * times * weakclassifier.p )
		return 1;

	return 0;
}



//将弱分类器数组wc[200]按错误率从小到大排序
void sortWeakClassifier()
{
	int i,j;
	WeakClassifier tmp;

	for( i = 0; i < wc_index-1; i++ )
	{
		int k = i;
		for( j = i+1; j < wc_index; j++ )
			if( wc[j].error_rate < wc[k].error_rate )
				k = j;
		if( k != i )
		{
			tmp = wc[i];
			wc[i] = wc[k];
			wc[k] = tmp;
		}
	}
}



//生成窗口大小为m中的所有templet类 矩形特征，放入wc中，并按错误率由小到大排序
void trangleFeature(int length, int templet, int s, int t)
{
	Feature feature;//矩形特征
	int p,q;   //特征矩形距右边缘有p倍自身宽度  特征矩形距下边缘有q倍自身长度
	
	feature.templet = templet;  //矩形特征的种类	

	for( int x1 = 0; x1 <= length-1-s; x1++ )
	{
		p = (length-1-x1) / s;   //特征矩形距右边缘有p倍自身宽度
		for( int y1 = 0; y1 <= length-1-t; y1++ )
		{		
			q = (length-1-y1) / t;     //特征矩形距下边缘有q倍自身长度
			feature.x = x1;
			feature.y = y1;
			for( int x_times = 1; x_times <= p; x_times++ )
				for( int y_times = 1; y_times <= q; y_times++ )
				{
					//创建矩形特征的横纵轴放大倍数			
					feature.x_times = x_times;	
					feature.y_times = y_times;
					
					//构造此特征的弱分类器
					WeakClassifier weakclassifier;
					createWeakClassifier(weakclassifier,feature);		

					//弱分类器数组中可存放个数为300个 下标0-299，填满之后若此弱分类器错误率比最后一个小，则覆盖最后一个，并排序
					if( wc_index == WEAK_CLASSIFIERS && weakclassifier.error_rate < wc[WEAK_CLASSIFIERS-1].error_rate )
					{
						wc[wc_index-1] = weakclassifier;
						sortWeakClassifier();
					}
					//如果没有填满则一直填不排序，直到填满那次排一次序
					else if( wc_index < WEAK_CLASSIFIERS )
					{
						wc[wc_index++] = weakclassifier;
						if( wc_index == WEAK_CLASSIFIERS )
							sortWeakClassifier();
					}
				}
		}

	}
}



//探测窗口，为一个正方形，边长m，遍历窗口内所有矩形特征
void detectWindowCreateWeakClassifiers(int length)  
{
	wc_index = 0; //清空弱分类器数组

	trangleFeature(length, 1, 1, 2); //第1类矩形特征
	trangleFeature(length, 2, 2, 1); //第2类矩形特征
	trangleFeature(length, 3, 1, 3); //第3类矩形特征
	trangleFeature(length, 4, 3, 1); //第4类矩形特征
	trangleFeature(length, 5, 2, 2); //第5类矩形特征
}



//训练一个有 numberOfWeakClassifier 个弱分类器的强分类器   窗口大小length
void createStrongClassifier(StrongClassifier &strongclassifier, int numberOfWeakClassifier, int length)
{

	strongclassifier.number = numberOfWeakClassifier; //弱分类器个数
	
	//设置权重
	int i;
	float faces_initweight = (float)1.0 / ( (float)2.0 * (float)FACES );  //人脸的初始权重
	float nonfaces_initweight = (float)1.0 / ( (float)2.0 * (float)NONFACES );  //非人脸的初始权重 
	for( i = 1; i <= FACES + NONFACES; i++ )
		if(image[i].flag == 1)
			image[i].weight = faces_initweight;	
		else
			image[i].weight = nonfaces_initweight;
	

	//生成弱分类器组,窗口大小为length
	detectWindowCreateWeakClassifiers(length);


	//取最好的 numberOfWeakClassifier 个弱分类器进行集成
	for( i = 0; i < numberOfWeakClassifier; i++ )
		strongclassifier.weakclassifier[i] = wc[i];
	
	cout<<"强分类器中包含"<<i<<"个弱分类器."<<endl;
	//强分类器的阈值
	strongclassifier.threshold = 0;	
	for( int i = 0; i < strongclassifier.number; i++ )
		strongclassifier.threshold += strongclassifier.weakclassifier[i].alpha / 2;

}



//画一个矩形框
void drawFace(Mat &image, int x, int y, int length)
{
	if(x+length>=image.cols-1 || y+length>=image.rows-1)
		return;
	//中心点
	Point center((x+length)/2,(y+length)/2);
	//画正方形
	rectangle(image,Point(x,y),Point(x+length,y+length),Scalar(0,0,255),THICKNESS,8,0);
}



//用一个强分类器strongclassifier判断一张子图像是否是人脸
int detectFaceByStrongClassifier(Image &image, float times, StrongClassifier strongclassifier)
{
	float value = 0;  //强分类器对图片的判断值

	for( int i = 0; i < strongclassifier.number; i++ )
		value += (float)detectFaceByWeakClassifier(image, times, strongclassifier.weakclassifier[i]) * strongclassifier.weakclassifier[i].alpha;

	image.strong_value = value;

	if( value >= strongclassifier.threshold )
		return 1;
	return 0;
}



//训练级联分类器  每层的检测率最低为f 误识率最高为d  最终的误识率为f_target   返回一个级联分类器就有栈溢出错误************************
void createCascadedClassifier(float f, float d, float f_target, int length)      
{
	//ofstream fout("output.txt");
	cout<<"每层误判率："<<f<<" 检测率："<<d<<" 总误判率："<<f_target<<endl<<endl;
	cc.f = f;   //每一层可接受的最大误判率
	cc.d = d;   //每一层可接受的最小灵敏度
	cc.f_target = f_target;   //目标误判率
	int i = 0;    //当前级联的层数

	float F[30], D[30];  //各层误判率和灵敏度
	F[0] = 1.0;
	D[0] = 1.0;

	int n;   //每层强分类器中的弱分类器个数  即特征个数

	//不断增加层数直到总误差率达到要求
	while( F[i] > f_target )
	{
		i++;
		cout<<endl;
		cout<<"开始训练第"<<i<<"层强分类器"<<endl;
		cout<<"本次人脸集:"<<FACES<<" 本次非人脸集:"<<NONFACES<<endl;
		
		F[i] = F[i-1];

		n = 0;

		StrongClassifier strongclassifier;

		//用P和N作为新的训练集
		for( int j = 1; j <= FACES; j++ )
			image[j] = P[j];
		for( int j = 1; j <= NONFACES; j++ )
			image[j+FACES] = N[j];

		int face_face = 0;
		int nonface_face = 0;

		//本层误识率没有达到要求就一直生成强分类器
		while( F[i] > f * F[i-1] )   
		{
			n++;  //强分类器中的弱分类器个数
			if( n > 199 )
				break;
			//重新计算全部的积分图像
			for( int index = 1; index <= FACES+NONFACES; index++ )
				oneImageIntegral(image[index]);
	
			cout<<"正在训练有"<<n<<"个弱分类器的候选强分类器"<<endl;
			//训练一个有n个弱分类器的强分类器
		
			if( n == 1 )
			createStrongClassifier(strongclassifier, 200, length);

			strongclassifier.number = n;

			float a = 0;
			for( int i = 0; i < n; i++ )
				a += strongclassifier.weakclassifier[i].alpha / 2;
			strongclassifier.threshold = a;	


			face_face = 0;
			nonface_face = 0;
			for( int k = 1; k <= FACES + NONFACES; k++ )
			{
				if(detectFaceByStrongClassifier(image[k],1,strongclassifier) == 1)
				{
					if(image[k].flag == 1)
						face_face++;
					else
						nonface_face++;	
				}
			}			
			D[i] = (float)face_face/(float)FACES * D[i-1];
			F[i] = (float)nonface_face/(float)NONFACES * F[i-1];
			cout<<"生成n="<<n<<"的候选强分类器  阈值："<<strongclassifier.threshold<<" 灵敏度："<<D[i]<<" 误判率："<<F[i]<<endl;
		
			//降低第i层强分类器阈值   降低算法不详************************************************
			
			//找最接近阈值的样本特征值替代阈值
			sortImageByStrongValue();

			float dis = 99999;
			int index = 1;
			for( ;index <= FACES+NONFACES;index++ )
			{		
				if( image[index].strong_value >= strongclassifier.threshold )
				{	
					strongclassifier.threshold = image[index].strong_value;
					break;
				}
			}

			//循环直到当前层叠分类器的检测率达到 d * D[i-1]
		    while( D[i] < d * D[i-1] )
			{
			//	fout<<"本次调整的检测率条件  D[i]："<<D[i]<<" d*D[i-1]:"<<d*D[i-1]<<endl;
				//误判率坏了就不调了
				if( F[i] > f * F[i-1] )
				{
					cout<<"误判率调不好了！"<<endl;
					break;
				}

				//调不好D[i]了，就把F[i]也弄坏了重新生成新的强分类器
				if( index < 1 )
				{
					cout<<"灵敏度调不好了！"<<endl;
					F[i] = 1;
					break;
				}

				float face_weight = (float)1.0 / (float)FACES;  //一张人脸的初始权重
				int chazhi = ( d * D[i-1] - D[i] ) / face_weight;
			
				if( chazhi < 1 )
				{
					cout<<"灵敏度调不好了！"<<endl;
					F[i] = 1;
					break;
				}

				//阈值减小到上一个数据的强分类器特征值
				strongclassifier.threshold = image[index-chazhi].strong_value;

				index -= chazhi;

				//衡量当前层叠分类器的检测率和误识率
				face_face = 0;
				nonface_face = 0;
				for( int k = 1; k <= FACES + NONFACES; k++ )
				{
					if( detectFaceByStrongClassifier(image[k],1,strongclassifier) == 1 )
					{
						if(image[k].flag == 1)
							face_face++;
						else
							nonface_face++;	
					}
				}
				D[i] = (float)face_face/(float)FACES * D[i-1];
				F[i] = (float)nonface_face/(float)NONFACES * F[i-1];
				cout<<"调整后的检测率："<<D[i]<<" 误判率："<<F[i]<<endl;	 
			}
			
		}
		
		cout<<"生成第"<<i<<"层强分类器！"<<"包括"<<n<<"个弱分类器  阈值为："<<strongclassifier.threshold<<" 检测率："<<D[i]<<"  误判率："<<F[i]<<endl;
		
		cc.strongclassifier[i-1] = strongclassifier;

		int N_index = 1;
		//利用当前强分类器检测非人脸图像，将误判的图像放入非人脸集N
		for( int k = 1; k <= NONFACES; k++ )
		{
			if( N[k].flag == 0 && detectFaceByStrongClassifier(N[k],1,strongclassifier) == 1 )
				N[N_index++] = N[k];
		}
		NONFACES = N_index - 1;  //更新非人脸集个数

		if( NONFACES < 5 )
			break;
	}

	cc.number = i; //级联层数
}         



//用级联分类器对一张子图像进行分类，子图像初始长宽为20*20，可放大times倍
int detectSubWindowByCascadedClassifier(Image image, float times, CascadedClassifier cascadedclassifier)
{
	//级联分类器中有一层不通过则为非人脸，全通过为人脸
	for( int i = 0; i < cascadedclassifier.number; i++ )
		if( detectFaceByStrongClassifier(image, times, cascadedclassifier.strongclassifier[i]) == 0 )
			return 0;
	return 1;
}



//判断一个子窗口是否是人脸
int drawFaceByStrongClassifier(ImageTest &image, float times, StrongClassifier strongclassifier,int x, int y, int length)
{
	float value = 0;  //强分类器对图片的判断值
	int i;
	for( i = 0; i < strongclassifier.number; i++ )
		value += (float)drawdetectFaceByWeakClassifier(image, times, strongclassifier.weakclassifier[i], x, y) * strongclassifier.weakclassifier[i].alpha;

	//强分类器对样本的函数值
	image.strong_value = value;

	if( value >= strongclassifier.threshold )
	{

        //查找之前的人脸框，如果有相近的就合并
		for(i = 0; i < face_index; i++)
		{
			//合并相同尺寸不同位置的人脸框
			if(abs(face[i].x - x) <= 5 && abs(face[i].y - y) <= 5 && face[i].length >= length*0.8)
			{
				face[i].x = (face[i].x * face[i].number + x) / (face[i].number + 1);
				face[i].y = (face[i].y * face[i].number + y) / (face[i].number + 1);
				face[i].length = (face[i].length * face[i].number + length) / (face[i].number + 1);
				face[i].number++;
				break;
			}
		
		}
		//如果没有相近的就增加一条新的记录
		if(i == face_index)
		{
			face[face_index].x = x;
			face[face_index].y = y;
			face[face_index].length = length;
			face[face_index].number = 1;
			face_index++;
		}

		return 1;
	}
	return 0;
}



//用强分类器探测一张图片上的各个子窗口
void drawImageByStrongClassifier(Mat &image, StrongClassifier strongclassifier)
{
	int length_x = image.cols;
	int length_y = image.rows;

	float times_x = image.cols / 20;
	float times_y = image.rows / 20;
	float max_times;

	if(times_x < times_y)
		max_times = times_x;
	else
		max_times = times_y;
	
	cout<<"cols:"<<image.cols<<" rows:"<<image.rows<<" max_times:"<<max_times<<endl;

	float current = 1;

	ImageTest img;
	img.img = image;

	//计算积分图
	drawoneImageIntegral(img);
	int count;
	while(current < max_times)
	{
		count = 0;
		cout<<"当前放大倍数:"<<current;
		for(int i=0; i+20*current<length_x-1; i+=1)
			for(int j=0; j+20*current<length_y-1; j+=1)	
				count += drawFaceByStrongClassifier(img,current,strongclassifier,i,j,20*current);
		cout<<"  人脸个数："<<count<<endl;
		current *= 1.2;
	}

	for(int i = 0; i < face_index; i++)
		if(face[i].number>5)
			drawFace(image, face[i].x, face[i].y, face[i].length);

}



//改变图像尺寸
void resizeImage(Mat &image, double scale)
{
	Size dsize = Size(image.cols*scale,image.rows*scale);
	Mat image2 = Mat(dsize,CV_32S);
	resize(image, image2,dsize);
	image = image2;
}



int main()
{
	//读取MIT人脸库
//	loadImg();
	 

	//计算全部的积分图像
//	imageIntegral();


//xml的基本读写操作
#if 0
	const char * xmlFile = "school.xml"; 
    TiXmlDocument doc;  
    TiXmlDeclaration * decl = new TiXmlDeclaration("1.0", "", "");  
    TiXmlElement * schoolElement = new TiXmlElement( "School" );  
    TiXmlElement * classElement = new TiXmlElement( "Class" );  
    classElement->SetAttribute("name", "C++");

    TiXmlElement * stu1Element = new TiXmlElement("Student");
    stu1Element->SetAttribute("name", "tinyxml");
    stu1Element->SetAttribute("number", "123");
    TiXmlElement * stu1EmailElement = new TiXmlElement("email");
    stu1EmailElement->LinkEndChild(new TiXmlText("tinyxml@163.com") );
    TiXmlElement * stu1AddressElement = new TiXmlElement("address");
    stu1AddressElement->LinkEndChild(new TiXmlText("中国"));
    stu1Element->LinkEndChild(stu1EmailElement);
    stu1Element->LinkEndChild(stu1AddressElement);

    classElement->LinkEndChild(stu1Element);  
    schoolElement->LinkEndChild(classElement);  
    
    doc.LinkEndChild(decl);  
    doc.LinkEndChild(schoolElement); 
    doc.SaveFile(xmlFile);  

                            
    if (doc.LoadFile(xmlFile)) {
        doc.Print();
    } else {
        cout << "can not parse xml conf/school.xml" << endl;
        return 0;
    }

	

	TiXmlElement* rootElement = doc.RootElement();  //School元素  
    classElement = rootElement->FirstChildElement();  // Class元素
    TiXmlElement* studentElement = classElement->FirstChildElement();  //Students  
    for (; studentElement != NULL; studentElement = studentElement->NextSiblingElement() ) 
	{
        TiXmlAttribute* attributeOfStudent = studentElement->FirstAttribute();  //获得student的name属性  
        for (;attributeOfStudent != NULL; attributeOfStudent = attributeOfStudent->Next() ) {
            cout << attributeOfStudent->Name() << " : " << attributeOfStudent->Value() << std::endl;       
        }                                 

        TiXmlElement* studentContactElement = studentElement->FirstChildElement();//获得student的第一个联系方式 
        for (; studentContactElement != NULL; studentContactElement = studentContactElement->NextSiblingElement() ) {
            string contactType = studentContactElement->Value();
            string contactValue = studentContactElement->GetText();
            cout << contactType  << " : " << contactValue << std::endl;           
        }
	}


	system("Pause");
#endif



//C++与XML数据格式转换
#if 0
	int number = 100;
	float threshold = 9.89;
	char thres[20];
	sprintf(thres,"%f",threshold);

	const char * xmlFile = "StrongClassifier.xml"; 
	TiXmlDocument doc;  
	TiXmlDeclaration * decl = new TiXmlDeclaration("1.0", "", ""); 
	TiXmlElement * strongElement = new TiXmlElement( "StrongCLassifier" );
	strongElement->SetAttribute("number",number);
	strongElement->SetAttribute("threshold",thres);

	doc.LinkEndChild(decl);  
    doc.LinkEndChild(strongElement); 
    doc.SaveFile(xmlFile);  

	if (doc.LoadFile(xmlFile)) {
        doc.Print();
    } else {
        cout << "can not parse xml conf/school.xml" << endl;
        return 0;
    }


	TiXmlElement* rootElement = doc.RootElement();  //School元素  
	TiXmlAttribute* attributeOfStudent = rootElement->FirstAttribute();
	int haha=atoi(attributeOfStudent->Value());

	attributeOfStudent = attributeOfStudent->Next();
	float lala=atof(attributeOfStudent->Value());
	cout<<haha<<' '<<lala<<endl;

#endif



//生成满足一定误判率的强分类器
#if 0

	cout<<"训练分类器......"<<endl;

	//生成强分类器
	StrongClassifier strongclassifier; 
	createStrongClassifier(strongclassifier,WEAK_CLASSIFIERS,20);

	cout<<"分类器已生成......"<<endl;

	 //测试样本
	 for( int i = 1; i <= FACES; i++ )
	 {
		 stringstream buffer;
         buffer<<i+300;
		 image[i].img = imread("faces\\"+buffer.str()+".bmp",0);
		 image[i].flag = 1;
	 }
	 for( int i = 1; i <= NONFACES; i++ )
	 {
		 stringstream buffer;
         buffer<<i+300;
		 image[i+FACES].img = imread("nonfaces\\"+buffer.str()+".bmp",0);
		 image[i+FACES].flag = 0;
	 }


	imageIntegral();

	int nonface_face=0;
	int face_face =0;
	for(int i = 1; i <= FACES + NONFACES; i++)
		if(detectFaceByStrongClassifier(image[i],1,strongclassifier)==1)
		{	
			if(image[i].flag == 1)
				face_face++;
			else
				nonface_face++;
		}
		cout<<FACES<<':'<<face_face<<' '<<NONFACES<<':'<<nonface_face<<endl;
		cout<<"  检测率:"<<(float)face_face/(float)FACES;
		cout<<"  误识率:"<<(float)nonface_face/(float)NONFACES<<endl<<endl;


	//调整强分类器的阈值，使误判率达到要求
	sortImageByStrongValue();

	//找到最接近强分类器阈值的样本i
	int i;
	for(i=1; i<= FACES+NONFACES; i++)
		if(image[i].strong_value>=strongclassifier.threshold)
			break;

	cout<<"i:"<<i<<' '<<"threshold"<<strongclassifier.threshold<<endl;
	int count = ((float)nonface_face/(float)NONFACES - 0.001) * NONFACES;
	
	int c = 0;
	while(c<count)
	{
		
		if(image[i++].flag==0)
			c++;
	}
	
	strongclassifier.threshold = image[i].strong_value;
	cout<<"i:"<<i<<' '<<"new threshold:"<<strongclassifier.threshold<<endl;

	nonface_face = 0;
	face_face = 0;
	for(int i = 1; i <= FACES + NONFACES; i++)
		if(detectFaceByStrongClassifier(image[i],1,strongclassifier) == 1)
		{	
			if(image[i].flag == 1)
				face_face++;
			else
				nonface_face++;
		}
		cout<<FACES<<':'<<face_face<<' '<<NONFACES<<':'<<nonface_face<<endl;
		cout<<"  检测率:"<<(float)face_face/(float)FACES;
		cout<<"  误识率:"<<(float)nonface_face/(float)NONFACES<<endl<<endl;


#endif



//将强分类器输出到strongclassifier.xml文件中
#if 0
	const char * xmlFile = "strongclassifier.xml"; 
	TiXmlDocument doc;  
	TiXmlDeclaration * decl = new TiXmlDeclaration("1.0", "", ""); 
	TiXmlElement * strongElement = new TiXmlElement( "StrongCLassifier" );

	//strongcLassifier.number
	strongElement->SetAttribute("number",strongclassifier.number);

	//strongCLassifier.threshold
	char tmp[20];
	sprintf(tmp,"%f",strongclassifier.threshold);
	strongElement->SetAttribute("threshold",tmp);

	for(i = 0; i < strongclassifier.number; i++)
	{	

		cout<<i<<endl;
		TiXmlElement * weakElement = new TiXmlElement("WeakClassifier");
		

		//strongclassifier.weakclassifier[i].threshold
		TiXmlElement * weakthresholdElement = new TiXmlElement("threshold");
		sprintf(tmp,"%d",strongclassifier.weakclassifier[i].threshold);
		weakthresholdElement->LinkEndChild(new TiXmlText(tmp));
		weakElement->LinkEndChild(weakthresholdElement);

		//strongclassifier.weakclassifier[i].p
		TiXmlElement * weakpElement = new TiXmlElement("p");
		sprintf(tmp, "%d", strongclassifier.weakclassifier[i].p);
		weakpElement->LinkEndChild(new TiXmlText(tmp));
		weakElement->LinkEndChild(weakpElement);

		//strongclassifier.weakclassifier[i].error_rate
		TiXmlElement * weakerror_rateElement = new TiXmlElement("error_rate");
		sprintf(tmp,"%f",strongclassifier.weakclassifier[i].error_rate);
		weakerror_rateElement->LinkEndChild(new TiXmlText(tmp));
		weakElement->LinkEndChild(weakerror_rateElement);

		//strongclassifier.weakclassifier[i].alpha
		TiXmlElement * weakalphaElement = new TiXmlElement("alpha");
		sprintf(tmp,"%f",strongclassifier.weakclassifier[i].alpha);
		weakalphaElement->LinkEndChild(new TiXmlText(tmp));
		weakElement->LinkEndChild(weakalphaElement);

		//strongclassifier.weakclassifier[i].feature
		TiXmlElement * featureElement = new TiXmlElement("feature"); 
        weakElement->LinkEndChild(featureElement);

		//strongclassifier.weakclassifier[i].feature.templet
		TiXmlElement * featuretempletElement = new TiXmlElement("templet");
		sprintf(tmp, "%d", strongclassifier.weakclassifier[i].feature.templet);
		featuretempletElement->LinkEndChild(new TiXmlText(tmp));
		featureElement->LinkEndChild(featuretempletElement);

		//strongclassifier.weakclassifier[i].feature.x_times
		TiXmlElement * featurex_timesElement = new TiXmlElement("x_times");
		sprintf(tmp, "%d", strongclassifier.weakclassifier[i].feature.x_times);
		featurex_timesElement->LinkEndChild(new TiXmlText(tmp));
		featureElement->LinkEndChild(featurex_timesElement);

		//strongclassifier.weakclassifier[i].feature.y_times
		TiXmlElement * featurey_timesElement = new TiXmlElement("y_times");
		sprintf(tmp, "%d", strongclassifier.weakclassifier[i].feature.y_times);
		featurey_timesElement->LinkEndChild(new TiXmlText(tmp));
		featureElement->LinkEndChild(featurey_timesElement);

		//strongclassifier.weakclassifier[i].feature.x
		TiXmlElement * featurexElement = new TiXmlElement("x");
		sprintf(tmp, "%d", strongclassifier.weakclassifier[i].feature.x);
		featurexElement->LinkEndChild(new TiXmlText(tmp));
		featureElement->LinkEndChild(featurexElement);

		//strongclassifier.weakclassifier[i].feature.y
		TiXmlElement * featureyElement = new TiXmlElement("y");
		sprintf(tmp, "%d", strongclassifier.weakclassifier[i].feature.y);
		featureyElement->LinkEndChild(new TiXmlText(tmp));
		featureElement->LinkEndChild(featureyElement);

		strongElement->LinkEndChild(weakElement);
		
	}
	doc.LinkEndChild(decl);  
	doc.LinkEndChild(strongElement); 
	doc.SaveFile(xmlFile);  

#endif
	


//从xml里读取强分类器
#if 1
	const char * xmlFile = "strongclassifier300features.xml"; 
	TiXmlDocument doc;  
	doc.LoadFile(xmlFile);
	StrongClassifier strongclassifier;

	TiXmlElement* strongElement = doc.RootElement();  //StrongClassifier元素  

	//读取strongclassifier.number
	TiXmlAttribute* attributeOfStudent = strongElement->FirstAttribute();
	strongclassifier.number = atoi(attributeOfStudent->Value());

	//读取strongclassifier.threshold
	attributeOfStudent = attributeOfStudent->Next();
	strongclassifier.threshold = atof(attributeOfStudent->Value());

	int i = 0;

	TiXmlElement* weakElement = strongElement->FirstChildElement();  //WeakClassifier  
    for (; weakElement != NULL; weakElement = weakElement->NextSiblingElement() ) 
	{ 

		TiXmlElement* tmpelement;
		tmpelement = weakElement->FirstChildElement();  //WeakClassifier->threshold
		strongclassifier.weakclassifier[i].threshold = atoi(tmpelement->GetText());

		tmpelement = tmpelement->NextSiblingElement();  //p
		strongclassifier.weakclassifier[i].p = atoi(tmpelement->GetText());

		tmpelement = tmpelement->NextSiblingElement();  //WeakClassifier->error_rate
		
		tmpelement = tmpelement->NextSiblingElement();  //WeakClassifier->alpha
		strongclassifier.weakclassifier[i].alpha = atof(tmpelement->GetText());

		tmpelement = tmpelement->NextSiblingElement();  //WeakClassifier->feature

		tmpelement = tmpelement->FirstChildElement();   //WeakClassifier->feature->templet
		strongclassifier.weakclassifier[i].feature.templet = atoi(tmpelement->GetText());

		tmpelement = tmpelement->NextSiblingElement();   //WeakClassifier->feature->x_times
		strongclassifier.weakclassifier[i].feature.x_times = atoi(tmpelement->GetText());

		tmpelement = tmpelement->NextSiblingElement();   //WeakClassifier->feature->y_times
		strongclassifier.weakclassifier[i].feature.y_times = atoi(tmpelement->GetText());

		tmpelement = tmpelement->NextSiblingElement();   //WeakClassifier->feature->x
		strongclassifier.weakclassifier[i].feature.x = atoi(tmpelement->GetText());

		tmpelement = tmpelement->NextSiblingElement();   //WeakClassifier->feature->y
		strongclassifier.weakclassifier[i].feature.y = atoi(tmpelement->GetText());
		
		i++;

        
	}
	cout<<i<<endl;
#endif
	


//用强分类器对一张图片进行探测
#if 1

		Mat img_test = imread("test.bmp",0);

		resizeImage(img_test,0.5);

		cout<<"正在检测图片......"<<endl;

		drawImageByStrongClassifier(img_test,strongclassifier);
		
#endif


//显示测试图片
#if 1

		namedWindow("haha");
		imshow("haha",img_test);
		waitKey(0);

#endif



//生成强分类器，生成画ROC曲线的数据
#if 0
	
	//生成强分类器
	StrongClassifier strongclassifier = createStrongClassifier(200,20);

	 //测试样本
	 for( int i = 1; i <= FACES; i++ )
	 {
		 stringstream buffer;
         buffer<<i+300;
		 image[i].img = imread("faces\\"+buffer.str()+".bmp",0);
		 image[i].flag = 1;
	 }
	 for( int i = 1; i <= NONFACES; i++ )
	 {
		 stringstream buffer;
         buffer<<i+300;
		 image[i+FACES].img = imread("nonfaces\\"+buffer.str()+".bmp",0);
		 image[i+FACES].flag = 0;
	 }


	imageIntegral();

	for(int i = 1; i <= FACES + NONFACES; i++)
		if(detectFaceByStrongClassifier(image[i],1,strongclassifier)==1)
			;


	sortImageByStrongValue();

	ofstream fout("output.txt");

	for( int i = 1; i <= FACES+NONFACES; i++)
	{
		int faces_right=0;
		int nonfaces_faces=0;
		for( int j = i; j <= FACES+NONFACES; j++)
		{
			if(image[j].flag == 1)
				faces_right++;
			else
				nonfaces_faces++;
		}
		fout<<i<<"-阈值："<<image[i].strong_value;
		fout<<"  检测率:"<<(float)faces_right/(float)FACES;
		fout<<"  误识率:"<<(float)nonfaces_faces/(float)NONFACES<<endl;
	}
	fout.close();

#endif



//生成级联分类器
#if 0
	
	//生成级联分类器
	createCascadedClassifier((float)0.3, (float)0.99, (float)0.00001, 20);

	FACES = 300;
	NONFACES = 300;

	 //测试样本
	 for( int i = 1; i <= FACES; i++ )
	 {
		 stringstream buffer;
         buffer<<i+300;
		 image[i].img = imread("faces\\"+buffer.str()+".bmp",0);
		 image[i].flag = 1;
	 }
	 for( int i = 1; i <= NONFACES; i++ )
	 {
		 stringstream buffer;
         buffer<<i+2000;
		 image[i+FACES].img = imread("nonfaces\\"+buffer.str()+".bmp",0);
		 image[i+FACES].flag = 0;
	 }


	imageIntegral();

	int faces_right=0;
	int nonfaces_faces=0;

	for(int i = 1; i <= FACES + NONFACES; i++)
	{
		if(detectSubWindowByCascadedClassifier(image[i],1,cc)==1)
		{
			if(image[i].flag==1)			
				faces_right++;			
			else
				nonfaces_faces++;	
		}
	}
	fout<<endl<<endl<<"开始测试......"<<endl;
	fout<<"测试集人脸:"<<FACES<<"  识别成人脸:";
	fout<<faces_right<<endl;
	fout<<"检测率:";
	fout<<(float)faces_right/(float)FACES<<endl;

	fout<<"测试集非人脸:"<<NONFACES<<"  识别成人脸:";
	fout<<nonfaces_faces<<endl;
	fout<<"误识率：";
	fout<<(float)nonfaces_faces/(float)NONFACES<<endl;
	fout<<endl;
	fout.close();
#endif




//读取摄像头
#if 0
	cvNamedWindow("win");

    CvCapture* capture = cvCreateCameraCapture(0);
    IplImage* frame;

    while(1) {
        frame = cvQueryFrame(capture);
        if(!frame) break;
        cvShowImage("win", frame);

        char c = cvWaitKey(50);
        if(c==27) break;
    }

    cvReleaseCapture(&capture);
    cvDestroyWindow("win");

#endif	


 
    return 0;
}






