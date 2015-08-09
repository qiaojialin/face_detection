/**
* @face_detection.cpp
* @author:wolf
* 这个demo使用opencv实现图片的人脸检测、以及绘画框出人脸等功能。
* 每个功能写成一个函数，方便移植使用。
* 参考：opencv基本绘画、物体检测模块文档。
*/

#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include"opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include<iostream>
#include<sstream>
#include<string>
#include<math.h>
using namespace cv;
using namespace std;

int FACES = 10;  
int NONFACES = 50;

int FACES_TEST = FACES;
int NONFACES_TEST = NONFACES;

template <typename T>
void show(T a)
{
	cout<<a<<endl;
}

//图像结构体
struct Image
{
	Mat img;						//灰度图像矩阵
	float weight;					//每幅图的权重
	int flag;						//人脸为1，非人脸为0
	int img_integral[25][25];   	 //积分图像矩阵 1-FACES+NONFACES
	int feature_value;				//对应一个矩形特征的特征值
}image[1501];	    //下标范围为1-1500

Image P[501];  //当前级联分类器的人脸集
Image N[1001];   //当前级联分类器的非人脸集
Image N_test[1001];

//特征结构体
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
}wc[200];             //下标为0-199
int wc_index = 0;


//强分类器结构体
struct StrongClassifier
{
	WeakClassifier weakclassifier[200];   //包含的弱分类器 最多200个
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
		 P[i] = image[i];
	 }

	 //读取NONFACES张非人脸灰度图
	 for( i = 1; i <= NONFACES; i++ )
	 {
		 stringstream buffer;
         buffer<<i;
		 image[i+FACES].img = imread("nonfaces\\"+buffer.str()+".bmp",0);
		 image[i+FACES].flag = 0;
		 N[i] = image[i+FACES];
		 N_test[i] = image[i+FACES];
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
				image.img_integral[x][y] = (int)image.img.at<uchar>(0,0);

			//Y轴上的点=上邻点+此点的像素
			else if( x == 0 )
				image.img_integral[x][y] = image.img_integral[0][y-1] + (int)image.img.at<uchar>(0,y);

			//X轴上的点=左邻点+此点像素
			else if( y == 0 )
				image.img_integral[x][y] = image.img_integral[x-1][0] + (int)image.img.at<uchar>(x,0);

			//其他点=左邻点+上邻点-左上点+此点像素
			else
				image.img_integral[x][y] = image.img_integral[x-1][y] + image.img_integral[x][y-1]   \
											 - image.img_integral[x-1][y-1] + (int)image.img.at<uchar>(x,y);
		}	 
}



//计算所有图像的积分图像
void imageIntegral()
{
	for( int index = 1; index <= FACES+NONFACES; index++ )
		oneImageIntegral(image[index]);
}



//计算一张图片相对于一个特征的特征值
int oneFeatureValue(Image image, Feature feature)
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
		int x = 1 * x_times;
		int y = 2 * y_times;
		feature_value = 2 * ( image.img_integral[ x1 + x ][ y1 + y / 2 ] - image.img_integral[ x1 ][ y1 + y / 2 ] )  \
								- ( image.img_integral[ x1 + x ][ y1 ] - image.img_integral[ x1 ][ y1 ] )                    \
								- ( image.img_integral[ x1 + x ][ y1 + y ] - image.img_integral[ x1 ][ y1 + y ] );

	}

	//(s,t)特征为(2,1)的矩形特征,类别号为2
	if( templet == 2 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = 2 * x_times;
		int y = 1 * y_times;
		feature_value = 2 * ( image.img_integral[ x1 + x / 2 ][ y1 + y ] - image.img_integral[ x1 + x / 2 ][ y1 ] )    \
								- ( image.img_integral[ x1 ][ y1 + y ] - image.img_integral[ x1 ][ y1 ] )              \
								- ( image.img_integral[ x1 + x ][ y1 + y ] - image.img_integral[ x1 + x ][ y1 ] );
	}

	//(s,t)特征为(1,3)的矩形特征,类别号为3
	if( templet == 3 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = 1 * x_times;
		int y = 3 * y_times;
		feature_value = 3 * ( image.img_integral[ x1 + x ][ y1 + y / 3 ] - image.img_integral[ x1 ][ y1 + y / 3 ] )         \
								- 3 * ( image.img_integral[ x1 + x ][ y1 + 2 * y / 3 ] - image.img_integral[ x1 ][ y1 + 2 * y / 3 ] )    \
								+ ( image.img_integral[ x1 + x ][ y1 + y ] - image.img_integral[ x1 ][ y1 + y ] )                        \
								- ( image.img_integral[ x1 + x ][ y1 ] - image.img_integral[ x1 ][ y1 ] );		
	}

	//(s,t)特征为(3,1)的矩形特征,类别号为4
	if( templet == 4 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = 3 * x_times;
		int y = 1 * y_times;
		feature_value = 3 * ( image.img_integral[ x1 + x / 3 ][ y1 + y ] - image.img_integral[ x1 + x / 3 ][ y1 ] )      \
								- 3 * ( image.img_integral[ x1 + 2 * x / 3 ][ y1 + y ] - image.img_integral[ x1 + 2 * x / 3 ][ y1 ] )     \
								+ ( image.img_integral[ x1 + x ][ y1 + y ] - image.img_integral[ x1 + x ][ y1 ] )                         \
								- ( image.img_integral[ x1 ][ y1 + y ] - image.img_integral[ x1 ][ y1 ] );	
	}

	//(s,t)特征为(2,2)的矩形特征,类别号为5
	if( templet == 5 )
	{
		//每个特征的长和宽，单个弱分类器的依据
		int x = 2 * x_times;
		int y = 2 * y_times;
		feature_value = 4 * image.img_integral[ x1 + x / 2 ][ y1 + y / 2 ]                                                   \
									+ image.img_integral[ x1 ][ y1 ]  + image.img_integral[ x1 + x ][ y1 ]                            \
									+ image.img_integral[ x1 ][ y1 + y ] + image.img_integral[ x1 + x ][ y1 + y ]                     \
									- 2 * ( image.img_integral[ x1 + x ][ y1 + y / 2 ] + image.img_integral[ x1 ][ y1 + y / 2 ] )     \
									- 2 * ( image.img_integral[ x1 + x / 2 ][ y1 + y ] + image.img_integral[ x1 + x / 2 ][ y1 ] );		
	}

	return feature_value;
}



//把图像根据当前特征值从小到大排序
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



//通过特征feature训练弱分类器
WeakClassifier createWeakClassifier(Feature feature)
{
	//根据矩形特征feature计算所有图片的特征值
	for( int i = 1; i <= FACES + NONFACES; i++ )
		image[i].feature_value = oneFeatureValue(image[i], feature);


	//将图像特征值排序
	sortImageByFeatureValue();

	WeakClassifier weakclassifier; //弱分类器

	weakclassifier.threshold = image[0].feature_value;  //当前阈值
	weakclassifier.error_rate = (float)100;             //当前最小错误率
	weakclassifier.p = 1;                               //当前最小错误率下的不等式调节因子<
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
		int p_current = 0;
		if( e1 <= e2 )
		{
			error_current = e1;
			p_current = -1;
		}
		else if( e2 < e1 )
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
	
	return weakclassifier;
}



//用一个弱分类器对一张图片进行探测,弱分类器判别公式
int detectFaceByWeakClassifier(Image image, WeakClassifier weakclassifier)
{
	//计算积分图
	oneImageIntegral(image);

	//计算图片相对于这个弱分类器的特征值
	image.feature_value = oneFeatureValue(image, weakclassifier.feature);

	//根据弱分类器判别公式判断图片是否有人脸
	if( image.feature_value * weakclassifier.p <= weakclassifier.threshold * weakclassifier.p )
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
					WeakClassifier weakclassifier = createWeakClassifier(feature);		

					//弱分类器数组中可存放个数为200个 下标0-199，填满之后若此弱分类器错误率比最后一个小，则覆盖最后一个，并排序
					if( wc_index == 200 && weakclassifier.error_rate < wc[199].error_rate )
					{
						wc[wc_index-1] = weakclassifier;
						sortWeakClassifier();
					}
					//如果没有填满则一直填不排序，直到填满那次排一次序
					else if( wc_index < 200 )
					{
						wc[wc_index++] = weakclassifier;
						if( wc_index == 200 )
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



//训练一个有 numberOfWeakClassifier 个弱分类器的强分类器   窗口大小m
StrongClassifier createStrongClassifier(int numberOfWeakClassifier, int length)
{
	StrongClassifier strongclassifier;

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
	for( int i = 0; i < numberOfWeakClassifier; i++ )
		strongclassifier.weakclassifier[i] = wc[i];
	

	//强分类器的阈值
	strongclassifier.threshold = 0;	
	for( int i = 0; i < strongclassifier.number; i++ )
		strongclassifier.threshold += strongclassifier.weakclassifier[i].alpha / 2;

	return strongclassifier;
}



//用一个强分类器strongclassifier对一张图片image进行分类
int detectFaceByStrongClassifier(Image image, StrongClassifier strongclassifier)
{
	float value = 0;  //强分类器对图片的判断值

	for( int i = 0; i < strongclassifier.number; i++ )
		value += (float)detectFaceByWeakClassifier(image, strongclassifier.weakclassifier[i]) * strongclassifier.weakclassifier[i].alpha;

	if( value >= strongclassifier.threshold )
		return 1;
	return 0;
}



//训练级联分类器  每层的检测率最低为f 误识率最高为d  最终的误识率为f_target   返回一个级联分类器就有栈溢出错误************************
void createCascadedClassifier(float f, float d, float f_target, int length)      
{
	cout<<"每层误判率："<<f<<" 检测率："<<d<<" 总误判率："<<f_target<<endl<<endl;
	cc.f = f;   //每一层可接受的最大误判率
	cc.d = d;   //每一层可接受的最小灵敏度
	cc.f_target = f_target;   //目标误判率
	int i = 0;    //当前级联的层数

	float F[30], D[30];  //各层误判率和灵敏度
	F[0] = 1.0;
	D[0] = 1.0;
	
	//不断增加层数直到总误差率达到要求
	while( F[i] > f_target )
	{
		i++;
		cout<<endl<<endl<<endl;
		cout<<"开始训练第"<<i<<"层强分类器"<<endl;
		cout<<"本次人脸集:"<<FACES<<" 本次非人脸集:"<<NONFACES<<endl;
		int n = 0;   //每层强分类器中的弱分类器个数  即特征个数
		F[i] = F[i-1];

		StrongClassifier strongclassifier;

		int face_face = 0;
		int nonface_face = 0;

		//本层误识率没有达到要求就一直生成强分类器
		while( F[i] > f * F[i-1] )   
		{
			n++;  //强分类器中的弱分类器个数
			//用P和N作为新的训练集
			for( int j = 1; j <= FACES; j++ )
				image[j] = P[j];
			for( int j = 1; j <= NONFACES; j++ )
				image[j+FACES] = N[j];


			//重新计算全部的积分图像
			for( int index = 1; index <= FACES+NONFACES; index++ )
				oneImageIntegral(image[index]);
			
			cout<<"正在训练包含有"<<n<<"个弱分类器的强分类器......"<<endl;

			//训练一个有n个弱分类器的强分类器
			strongclassifier = createStrongClassifier(n, length);
			
			cout<<"候选强分类器中每个弱分类器:阈值  alpha  错误率："<<endl;
			for(int a=0;a<n;a++)
				cout<<strongclassifier.weakclassifier[a].threshold<<"  "<<strongclassifier.weakclassifier[a].alpha<<"  "<<strongclassifier.weakclassifier[a].error_rate<<endl;
			cout<<endl;
			
			cout<<"生成一个第"<<i<<"层的候选强分类器..."<<"包括"<<n<<"个弱分类器  阈值为："<<strongclassifier.threshold<<endl;
			//衡量当前层叠分类器的检测率D[i]和误识率F[i]
			face_face = 0;
			nonface_face = 0;
			for( int k = 1; k <= FACES + NONFACES; k++ )
			{
				if(detectFaceByStrongClassifier(image[k],strongclassifier) == 1)
				{
					if(image[k].flag == 1)
						face_face++;
					else
						nonface_face++;	
				}
			}			
			D[i] = (float)face_face/(float)FACES * D[i-1];
			F[i] = (float)nonface_face/(float)NONFACES * F[i-1];
			cout<<"候选强分类器的检测率："<<D[i]<<"  误判率："<<F[i]<<endl;

			//循环直到当前层叠分类器的检测率达到 d * D[i-1]
			while( D[i] < d * D[i-1] )
			{
				 //降低第i层强分类器阈值   降低算法不详************************************************
				if(strongclassifier.threshold > 0)
					strongclassifier.threshold *=0.8;

				//衡量当前层叠分类器的检测率和误识率
				face_face = 0;
				nonface_face = 0;
				for( int k = 1; k <= FACES + NONFACES; k++ )
				{
					if( detectFaceByStrongClassifier(image[k],strongclassifier) == 1 )
					{
						if(image[k].flag == 1)
							face_face++;
						else
							nonface_face++;	
					}
				}
				D[i] = (float)face_face/(float)FACES * D[i-1];
				F[i] = (float)nonface_face/(float)NONFACES * F[i-1];
				cout<<"候选强分类器调整后  阈值:"<<strongclassifier.threshold<<"  检测率："<<D[i]<<"  误判率："<<F[i]<<endl;
			}
		}
		
		cout<<endl<<"生成第"<<i<<"层强分类器！ 级联分类器的检测率："<<D[i]<<"  误判率："<<F[i]<<endl;
		
		cc.strongclassifier[i] = strongclassifier;

		int N_index = 1;
		//利用当前强分类器检测非人脸图像，将误判的图像放入非人脸集N
		for( int k = 1; k <= NONFACES; k++ )
		{
			if( N[k].flag == 0 && detectFaceByStrongClassifier(N[k],strongclassifier) == 1 )
				N[N_index++] = N[k];
		}
		NONFACES = N_index - 1;  //更新非人脸集个数
	}

	cc.number = i+1; //级联层数
}         



//画人脸  i为生成不同名称的窗口
void drawFace(Image image, int length,int i)
{
	stringstream buffer;
         buffer<<i;
		 namedWindow(buffer.str());
	Point center(length/2,length/2);
	rectangle(image.img,Point(0,0),Point(length,length),Scalar(0,0,255),2,8,0);
	imshow(buffer.str(),image.img);
}



//用级联分类器对一张子图像进行分类
int detectFaceByCascadedClassifier(Image image, CascadedClassifier cascadedclassifier)
{
	//级联分类器中有一层不通过则为非人脸，全通过为人脸
	for( int i = 0; i < cascadedclassifier.number; i++ )
		if( detectFaceByStrongClassifier(image, cascadedclassifier.strongclassifier[i]) == 0 )
			return 0;
	return 1;
}




int main()
{
	//读取MIT人脸库
	loadImg();
	 

	//计算全部的积分图像
	imageIntegral();
	

#if 0
	int faces_right=0;
	int nonfaces_faces=0;

	FACES = 200;
	NONFACES = 500;

	for( int i = 1; i <= FACES; i++)
		image[i] = P[i];
	for( int i = 1; i <= NONFACES; i++)
		image[i+FACES] = N_test[i];


	for(int i = 1; i <= FACES + NONFACES; i++)
	{
		if(detectFaceByCascadedClassifier(image[i],cascadedclassifier)==1)
		{
			if(image[i].flag==1)
				faces_right++;
			else
				nonfaces_faces++;	
		}		
	}
	cout<<"总人脸:"<<FACES<<"  识别成人脸:";
	show(faces_right);
	cout<<"检测率:";
	show((float)faces_right/(float)FACES);

	cout<<"总非人脸:"<<NONFACES<<"  识别成人脸:";
	show(nonfaces_faces);
	cout<<"误识率：";
	show((float)nonfaces_faces/(float)NONFACES);
#endif
	
#if 0
	
	//生成强分类器
	StrongClassifier strongclassifier = createStrongClassifier(1);

	int faces_right=0;
	int nonfaces_faces=0;

	for(int i = 1; i <= FACES + NONFACES; i++)
	{
		if(detectFaceByStrongClassifier(image[i],strongclassifier)==1)
		{
			if(image[i].flag==1)
				faces_right++;
			else
				nonfaces_faces++;	
		}
	}
	cout<<"阈值："<<strongclassifier.threshold<<endl;
	cout<<"总人脸:"<<FACES<<"  识别成人脸:";
	show(faces_right);
	cout<<"检测率:";
	show((float)faces_right/(float)FACES);

	cout<<"总非人脸:"<<NONFACES<<"  识别成人脸:";
	show(nonfaces_faces);
	cout<<"误识率：";
	show((float)nonfaces_faces/(float)NONFACES);
	cout<<endl;
	
#endif


#if 1
	
	//生成级联分类器
	createCascadedClassifier((float)0.3, (float)0.99, (float)0.00001, 20);


	//构造测试集
	for( int j = 1; j <= FACES_TEST; j++ )
		image[j] = P[j];
	for( int j = 1; j <= NONFACES_TEST; j++ )
		image[j+FACES] = N_test[j];

	int faces_right=0;
	int nonfaces_faces=0;

	for(int i = 1; i <= FACES + NONFACES; i++)
	{
		if(detectFaceByCascadedClassifier(image[i],cc)==1)
		{
			if(image[i].flag==1)
			{
				drawFace(image[i],20,i);
				faces_right++;
			}
			else
				nonfaces_faces++;	
		}
	}
	cout<<endl<<endl<<"开始测试......"<<endl;
	cout<<"测试集人脸:"<<FACES_TEST<<"  识别成人脸:";
	show(faces_right);
	cout<<"检测率:";
	show((float)faces_right/(float)FACES_TEST);

	cout<<"测试集非人脸:"<<NONFACES_TEST<<"  识别成人脸:";
	show(nonfaces_faces);
	cout<<"误识率：";
	show((float)nonfaces_faces/(float)NONFACES_TEST);
	cout<<endl;
	
#endif


	 
	

//	namedWindow("haha");
	//imshow("haha",image[1].img);
  
	waitKey(0);
    return 0;
}






