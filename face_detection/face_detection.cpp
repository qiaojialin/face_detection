/**
* @face_detection.cpp
* @author:wolf
* ���demoʹ��opencvʵ��ͼƬ��������⡢�Լ��滭�����������ȡ�����ȹ��ܡ�
* ÿ������д��һ��������������ֲʹ�á�
* �ο���opencv�����滭��������ģ���ĵ���
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

int FACES = 500;    //ѵ����������
int NONFACES = 800;   //ѵ������������

int FACES_TEST = FACES;
int NONFACES_TEST = NONFACES;

#define WEAK_CLASSIFIERS 300   //ÿ��ѵ�������300������������ǿ������������300����������
#define THICKNESS  2       //������������Ĵ�ϸ



//ѵ����ͼ��ṹ��
struct Image
{
	Mat img;						//�Ҷ�ͼ�����
	float weight;					//ÿ��ͼ��Ȩ��
	int flag;						//����Ϊ1��������Ϊ0
	int img_integral[21][21];   	 //����ͼ����� 1-FACES+NONFACES
	int feature_value;				//��Ӧһ����������������ֵ
	float strong_value;				//ǿ�����������������ֵ
}image[1501];	    //�±귶ΧΪ1-1500


//���Լ�ͼ��ṹ��
struct ImageTest
{
	Mat img;						//�Ҷ�ͼ�����
	float weight;					//ÿ��ͼ��Ȩ��
	int flag;						//����Ϊ1��������Ϊ0
	int img_integral[313][425];   	 //����ͼ����� 1-FACES+NONFACES
	int feature_value;				//��Ӧһ����������������ֵ
	float strong_value;				//ǿ�����������������ֵ
};	   


Image P[301];  //��ǰ������������������
Image N[301];   //��ǰ�����������ķ�������
Image N_test[304];



//���������ṹ��
struct Feature
{
	int templet;  //���������
	int x_times;  //������X��Ŵ���
	int y_times;  //������Y��Ŵ���
	int x;        //��������ʼx����
	int y;        //��������ʼy����
};



//���������ṹ��
struct WeakClassifier
{
	Feature feature;   //һ������
	int threshold;     //��ֵ
	int p;             //�������� ʹ�б𲻵�ʽͳһΪС�ںţ���С����ֵΪ����
	float error_rate;  //������
	float alpha;      //����������ǿ�������е�Ȩ��
}wc[WEAK_CLASSIFIERS];             //�±�Ϊ0----WEAK_CLASSIFIERS-1
int wc_index = 0;



//ǿ�������ṹ��
struct StrongClassifier
{
	WeakClassifier weakclassifier[WEAK_CLASSIFIERS];   //�������������� ���WEAK_CLASSIFIERS��
	int number; //������������
	float threshold;     //ǿ��������ֵ  ���ڵ���Ϊ����  С��Ϊ������
};



//�����������ṹ��
struct CascadedClassifier
{
	StrongClassifier strongclassifier[51];  //ÿһ����ǿ������   �±�1-50  ���50��ǿ������
	int number;     //��������
	float d;          //ÿһ��ɽ��ܵ���С������ 
	float f;           //ÿһ��ɽ��ܵ������ʶ��
	float f_target;    //������������Ŀ����ʶ��
}cc;



//�����ṹ��,���ںϲ�����
struct Face
{
	int x;  //��������ʼx����
	int y;  //��������ʼy����
	int length;  //�����Ŀ��
	int number;  //���������ĸ���  ���ں����������ϲ���Ȩֵ
}face[1000];
int face_index = 0;



//��ȡMIT������FACES+NONFACES��
void loadImg()
{
	 int i; 
	 //��ȡFACES�������Ҷ�ͼ 
	 for( i = 1; i <= FACES; i++ )
	 {
		 stringstream buffer;
         buffer<<i;
		 image[i].img = imread("faces\\"+buffer.str()+".bmp",0);
		 image[i].flag = 1;
	//	 P[i] = image[i];
	 }

	 //��ȡNONFACES�ŷ������Ҷ�ͼ
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



//����һ��ͼ��Ļ���ͼ��
void oneImageIntegral(Image &image)
{
	int x,y;
	//һ��һ�д�����ɨ��
	for( x = 0; x < image.img.cols; x++ )
		for( y = 0; y < image.img.rows; y++ )
		{
			//��0,0����ֱ�ӷ���������ֵ
			if( x == 0 && y == 0)
				image.img_integral[y][x] = (int)image.img.at<uchar>(0,0);

			//Y���ϵĵ�=���ڵ�+�˵������
			else if( x == 0 )
				image.img_integral[y][x] = image.img_integral[y-1][0] + (int)image.img.at<uchar>(y,0);

			//X���ϵĵ�=���ڵ�+�˵�����
			else if( y == 0 )
				image.img_integral[y][x] = image.img_integral[0][x-1] + (int)image.img.at<uchar>(0,x);

			//������=���ڵ�+���ڵ�-���ϵ�+�˵�����
			else
				image.img_integral[y][x] = image.img_integral[y][x-1] + image.img_integral[y-1][x]   \
											 - image.img_integral[y-1][x-1] + (int)image.img.at<uchar>(y,x);
		}	 
}



//����һ��ͼ��Ļ���ͼ��
void drawoneImageIntegral(ImageTest &image)
{
	int x,y;
	//һ��һ�д�����ɨ
	for( x = 0; x < image.img.cols; x++ )
		for( y = 0; y < image.img.rows; y++ )
		{
			//��0,0����ֱ�ӷ���������ֵ
			if( x == 0 && y == 0)
				image.img_integral[y][x] = (int)image.img.at<uchar>(0,0);

			//Y���ϵĵ�=���ڵ�+�˵������
			else if( x == 0 )
				image.img_integral[y][x] = image.img_integral[y-1][0] + (int)image.img.at<uchar>(y,0);

			//X���ϵĵ�=���ڵ�+�˵�����
			else if( y == 0 )
				image.img_integral[y][x] = image.img_integral[0][x-1] + (int)image.img.at<uchar>(0,x);

			//������=���ڵ�+���ڵ�-���ϵ�+�˵�����
			else
				image.img_integral[y][x] = image.img_integral[y][x-1] + image.img_integral[y-1][x]   \
											 - image.img_integral[y-1][x-1] + (int)image.img.at<uchar>(y,x);
		}	
}



//��������ͼ��Ļ���ͼ��
void imageIntegral()
{
	for( int index = 1; index <= FACES+NONFACES; index++ )
		oneImageIntegral(image[index]);
}



//����һ��ͼƬ�����һ������������ֵ
int oneFeatureValue(Image image, float times, Feature feature)
{
	int templet = feature.templet;
	int x_times = feature.x_times;
	int y_times = feature.y_times;
	int x1 = feature.x;
	int y1 = feature.y;
	int feature_value = 0;

	//(s,t)����Ϊ(1,2)�ľ�������������Ϊ1
	if( templet == 1 )
	{
		//ÿ�������ĳ��Ϳ�������������������
		int x = (int)(1 * x_times * times);
		int y = (int)(2 * y_times * times);
		feature_value = 2 * ( image.img_integral[ y1 + y / 2 ][ x1 + x ] - image.img_integral[ y1 + y / 2 ][ x1 ] )  \
								- ( image.img_integral[ y1 ][ x1 + x ] - image.img_integral[ y1 ][ x1 ] )                    \
								- ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 + y ][ x1 ] );

	}

	//(s,t)����Ϊ(2,1)�ľ�������,����Ϊ2
	if( templet == 2 )
	{
		//ÿ�������ĳ��Ϳ�������������������
		int x = (int)(2 * x_times * times);
		int y = (int)(1 * y_times * times);
		feature_value = 2 * ( image.img_integral[ y1 + y ][ x1 + x / 2 ] - image.img_integral[ y1 ][ x1 + x / 2 ] )    \
								- ( image.img_integral[ y1 + y ][ x1 ] - image.img_integral[ y1 ][ x1 ] )              \
								- ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 ][ x1 + x ] );
	}

	//(s,t)����Ϊ(1,3)�ľ�������,����Ϊ3
	if( templet == 3 )
	{
		//ÿ�������ĳ��Ϳ�������������������
		int x = (int)(1 * x_times * times);
		int y = (int)(3 * y_times * times);
		feature_value = 3 * ( image.img_integral[ y1 + y / 3 ][ x1 + x ] - image.img_integral[ y1 + y / 3 ][ x1 ] )         \
								- 3 * ( image.img_integral[ y1 + 2 * y / 3 ][ x1 + x ] - image.img_integral[ y1 + 2 * y / 3 ][ x1 ] )    \
								+ ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 + y ][ x1 ] )                        \
								- ( image.img_integral[ y1 ][ x1 + x ] - image.img_integral[ y1 ][ x1 ] );		
	}

	//(s,t)����Ϊ(3,1)�ľ�������,����Ϊ4
	if( templet == 4 )
	{
		//ÿ�������ĳ��Ϳ�������������������
		int x = (int)(3 * x_times * times);
		int y = (int)(1 * y_times * times);
		feature_value = 3 * ( image.img_integral[ y1 + y ][ x1 + x / 3 ] - image.img_integral[ y1 ][ x1 + x / 3 ] )      \
								- 3 * ( image.img_integral[ y1 + y ][ x1 + 2 * x / 3 ] - image.img_integral[ y1 ][ x1 + 2 * x / 3 ] )     \
								+ ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 ][ x1 + x ] )                         \
								- ( image.img_integral[ y1 + y ][ x1 ] - image.img_integral[ y1 ][ x1 ] );	
	}

	//(s,t)����Ϊ(2,2)�ľ�������,����Ϊ5
	if( templet == 5 )
	{
		//ÿ�������ĳ��Ϳ�������������������
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




//����һ��ͼƬ�����һ������������ֵ
void drawoneFeatureValue(ImageTest &image, float times, Feature feature, int x0, int y0)
{
	
	int templet = feature.templet;
	int x_times = feature.x_times;
	int y_times = feature.y_times;
	int x1 = feature.x + x0;
	int y1 = feature.y + y0;
	int feature_value = 0;

	//(s,t)����Ϊ(1,2)�ľ�������������Ϊ1
	if( templet == 1 )
	{
		//ÿ�������ĳ��Ϳ�������������������
		int x = (int)(1 * x_times * times);
		int y = (int)(2 * y_times * times);
		feature_value = 2 * ( image.img_integral[ y1 + y / 2 ][ x1 + x ] - image.img_integral[ y1 + y / 2 ][ x1 ] )  \
								- ( image.img_integral[ y1 ][ x1 + x ] - image.img_integral[ y1 ][ x1 ] )                    \
								- ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 + y ][ x1 ] );

	}

	//(s,t)����Ϊ(2,1)�ľ�������,����Ϊ2
	if( templet == 2 )
	{
		//ÿ�������ĳ��Ϳ�������������������
		int x = (int)(2 * x_times * times);
		int y = (int)(1 * y_times * times);
		feature_value = 2 * ( image.img_integral[ y1 + y ][ x1 + x / 2 ] - image.img_integral[ y1 ][ x1 + x / 2 ] )    \
								- ( image.img_integral[ y1 + y ][ x1 ] - image.img_integral[ y1 ][ x1 ] )              \
								- ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 ][ x1 + x ] );
	}

	//(s,t)����Ϊ(1,3)�ľ�������,����Ϊ3
	if( templet == 3 )
	{
		//ÿ�������ĳ��Ϳ�������������������
		int x = (int)(1 * x_times * times);
		int y = (int)(3 * y_times * times);
		feature_value = 3 * ( image.img_integral[ y1 + y / 3 ][ x1 + x ] - image.img_integral[ y1 + y / 3 ][ x1 ] )         \
								- 3 * ( image.img_integral[ y1 + 2 * y / 3 ][ x1 + x ] - image.img_integral[ y1 + 2 * y / 3 ][ x1 ] )    \
								+ ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 + y ][ x1 ] )                        \
								- ( image.img_integral[ y1 ][ x1 + x ] - image.img_integral[ y1 ][ x1 ] );		
	}

	//(s,t)����Ϊ(3,1)�ľ�������,����Ϊ4
	if( templet == 4 )
	{
		//ÿ�������ĳ��Ϳ�������������������
		int x = (int)(3 * x_times * times);
		int y = (int)(1 * y_times * times);
		feature_value = 3 * ( image.img_integral[ y1 + y ][ x1 + x / 3 ] - image.img_integral[ y1 ][ x1 + x / 3 ] )      \
								- 3 * ( image.img_integral[ y1 + y ][ x1 + 2 * x / 3 ] - image.img_integral[ y1 ][ x1 + 2 * x / 3 ] )     \
								+ ( image.img_integral[ y1 + y ][ x1 + x ] - image.img_integral[ y1 ][ x1 + x ] )                         \
								- ( image.img_integral[ y1 + y ][ x1 ] - image.img_integral[ y1 ][ x1 ] );	
	}

	//(s,t)����Ϊ(2,2)�ľ�������,����Ϊ5
	if( templet == 5 )
	{
		//ÿ�������ĳ��Ϳ�������������������
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



//��ͼ�����������������ֵ��С��������
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



//��ͼ�����ǿ����������ֵ��С��������
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



//ͨ������featureѵ����������
void createWeakClassifier(WeakClassifier &weakclassifier, Feature feature)
{
	//���ݾ�������feature��������ͼƬ������ֵ
	for( int i = 1; i <= FACES + NONFACES; i++ )
		image[i].feature_value = oneFeatureValue(image[i], 1, feature);


	//��ͼ������ֵ����
	sortImageByFeatureValue();

	weakclassifier.threshold = image[0].feature_value;  //��ǰ��ֵ
	weakclassifier.error_rate = (float)100;             //��ǰ��С������
	weakclassifier.feature = feature;                   //��������Ӧ�ľ�������

	float faces_weight = 0;         //ȫ����������Ȩ��
	float nonfaces_weight = 0;		//ȫ������������Ȩ��

	float faces_weight_before = 0;		     //�ڵ�ǰ��ֵ֮ǰ������Ȩ�غ�
	float nonfaces_weight_before = 0;		 //�ڵ�ǰ��ֵ֮ǰ�ķ�����Ȩ�غ�

	//ͳ�ƴ������������컷���е�������������Ȩ�غ�faces_weight������������Ȩ�غ�nonfaces_weight
	for(int i = 1; i <= FACES + NONFACES; i++)
	{
		if(image[i].flag == 1)
			faces_weight += image[i].weight;
		else if(image[i].flag == 0)
			nonfaces_weight += image[i].weight;
	}

	//���������������������ֵ���ܣ�ȡ��������С����ֵ��Ϊ��������������ֵ
	for(int i = 1; i <= FACES + NONFACES; i++)
	{
		//ÿ�����
		faces_weight_before = 0;
		nonfaces_weight_before = 0;

		//ͳ���ڵ�ǰԪ��֮ǰ����������Ȩ�غ͡�����������Ȩ�غ�
		for(int j = 1; j <= i; j++ )  
		{
			if(image[j].flag == 1)  //����
				faces_weight_before += image[j].weight;
			else   //������
				nonfaces_weight_before += image[j].weight;
		}

		//����С������
		float e1,e2;
		e1 = faces_weight_before + ( nonfaces_weight - nonfaces_weight_before ); //������ <= ��ֵ < ���� �Ĵ�����  p_current = -1
		e2 = nonfaces_weight_before + ( faces_weight - faces_weight_before );   //���� <= ��ֵ < ������ �Ĵ�����  p_current = 1
		

		float error_current = 0;  //���õ�ǰ��ֵ�Ĵ�����
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

		//��ǰ��������С�ڷ����������ʣ����µ�ǰ���������Ĵ����ʣ��������ӣ���ֵ
		if( error_current < weakclassifier.error_rate )
		{
			weakclassifier.error_rate = error_current; 
			weakclassifier.p = p_current;
			weakclassifier.threshold = image[i].feature_value;
			
		}
	}

	float beta= weakclassifier.error_rate / (1 - weakclassifier.error_rate);  //��������
	weakclassifier.alpha = log(1/beta);

	float sum_weight = 0;  //�������Ȩ�غ�
	

	//����Ȩ�أ����ֶԵ�ͼƬȨ�س�һ������beta  ԭ���Ĵ����ʺ�Խ�ߣ�betaԽ��
	for( int i = 1; i <= FACES + NONFACES; i++ )
	{
		if( image[i].feature_value * weakclassifier.p <= weakclassifier.threshold * weakclassifier.p && image[i].flag == 1 ) //�ֶԵ�����
			image[i].weight *= beta;
		if( image[i].feature_value * weakclassifier.p >= weakclassifier.threshold * weakclassifier.p && image[i].flag == 0 ) //�ֶԵķ�����
			image[i].weight *= beta;
		sum_weight += image[i].weight;
	}

	//��һ��Ȩ��
	for( int i = 1; i <= FACES + NONFACES; i++)
		image[i].weight /= sum_weight;
	
}



//��һ������������һ��ͼƬ����̽��,���������б�ʽ  timesΪ̽�ⴰ�ڷŴ���
int drawdetectFaceByWeakClassifier(ImageTest &image, float times, WeakClassifier weakclassifier, int x, int y)
{
	//����ͼƬ����������������������ֵ
	drawoneFeatureValue(image, times, weakclassifier.feature, x, y);

	if(image.feature_value == -100)
		return 0;

	//�������������б�ʽ�ж�ͼƬ�Ƿ�������
	if( image.feature_value * weakclassifier.p <= weakclassifier.threshold * times * weakclassifier.p )
		return 1;

	return 0;
}



//��һ������������һ��ͼƬ����̽��,���������б�ʽ  timesΪ̽�ⴰ�ڷŴ���
int detectFaceByWeakClassifier(Image &image, float times, WeakClassifier weakclassifier)
{
	//�������ͼ
	oneImageIntegral(image);

	//����ͼƬ����������������������ֵ
	image.feature_value = oneFeatureValue(image, times, weakclassifier.feature);

	//�������������б�ʽ�ж�ͼƬ�Ƿ�������
	if( image.feature_value * weakclassifier.p <= weakclassifier.threshold * times * weakclassifier.p )
		return 1;

	return 0;
}



//��������������wc[200]�������ʴ�С��������
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



//���ɴ��ڴ�СΪm�е�����templet�� ��������������wc�У�������������С��������
void trangleFeature(int length, int templet, int s, int t)
{
	Feature feature;//��������
	int p,q;   //�������ξ��ұ�Ե��p��������  �������ξ��±�Ե��q��������
	
	feature.templet = templet;  //��������������	

	for( int x1 = 0; x1 <= length-1-s; x1++ )
	{
		p = (length-1-x1) / s;   //�������ξ��ұ�Ե��p��������
		for( int y1 = 0; y1 <= length-1-t; y1++ )
		{		
			q = (length-1-y1) / t;     //�������ξ��±�Ե��q��������
			feature.x = x1;
			feature.y = y1;
			for( int x_times = 1; x_times <= p; x_times++ )
				for( int y_times = 1; y_times <= q; y_times++ )
				{
					//�������������ĺ�����Ŵ���			
					feature.x_times = x_times;	
					feature.y_times = y_times;
					
					//�������������������
					WeakClassifier weakclassifier;
					createWeakClassifier(weakclassifier,feature);		

					//�������������пɴ�Ÿ���Ϊ300�� �±�0-299������֮�������������������ʱ����һ��С���򸲸����һ����������
					if( wc_index == WEAK_CLASSIFIERS && weakclassifier.error_rate < wc[WEAK_CLASSIFIERS-1].error_rate )
					{
						wc[wc_index-1] = weakclassifier;
						sortWeakClassifier();
					}
					//���û��������һֱ�����ֱ�������Ǵ���һ����
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



//̽�ⴰ�ڣ�Ϊһ�������Σ��߳�m���������������о�������
void detectWindowCreateWeakClassifiers(int length)  
{
	wc_index = 0; //���������������

	trangleFeature(length, 1, 1, 2); //��1���������
	trangleFeature(length, 2, 2, 1); //��2���������
	trangleFeature(length, 3, 1, 3); //��3���������
	trangleFeature(length, 4, 3, 1); //��4���������
	trangleFeature(length, 5, 2, 2); //��5���������
}



//ѵ��һ���� numberOfWeakClassifier ������������ǿ������   ���ڴ�Сlength
void createStrongClassifier(StrongClassifier &strongclassifier, int numberOfWeakClassifier, int length)
{

	strongclassifier.number = numberOfWeakClassifier; //������������
	
	//����Ȩ��
	int i;
	float faces_initweight = (float)1.0 / ( (float)2.0 * (float)FACES );  //�����ĳ�ʼȨ��
	float nonfaces_initweight = (float)1.0 / ( (float)2.0 * (float)NONFACES );  //�������ĳ�ʼȨ�� 
	for( i = 1; i <= FACES + NONFACES; i++ )
		if(image[i].flag == 1)
			image[i].weight = faces_initweight;	
		else
			image[i].weight = nonfaces_initweight;
	

	//��������������,���ڴ�СΪlength
	detectWindowCreateWeakClassifiers(length);


	//ȡ��õ� numberOfWeakClassifier �������������м���
	for( i = 0; i < numberOfWeakClassifier; i++ )
		strongclassifier.weakclassifier[i] = wc[i];
	
	cout<<"ǿ�������а���"<<i<<"����������."<<endl;
	//ǿ����������ֵ
	strongclassifier.threshold = 0;	
	for( int i = 0; i < strongclassifier.number; i++ )
		strongclassifier.threshold += strongclassifier.weakclassifier[i].alpha / 2;

}



//��һ�����ο�
void drawFace(Mat &image, int x, int y, int length)
{
	if(x+length>=image.cols-1 || y+length>=image.rows-1)
		return;
	//���ĵ�
	Point center((x+length)/2,(y+length)/2);
	//��������
	rectangle(image,Point(x,y),Point(x+length,y+length),Scalar(0,0,255),THICKNESS,8,0);
}



//��һ��ǿ������strongclassifier�ж�һ����ͼ���Ƿ�������
int detectFaceByStrongClassifier(Image &image, float times, StrongClassifier strongclassifier)
{
	float value = 0;  //ǿ��������ͼƬ���ж�ֵ

	for( int i = 0; i < strongclassifier.number; i++ )
		value += (float)detectFaceByWeakClassifier(image, times, strongclassifier.weakclassifier[i]) * strongclassifier.weakclassifier[i].alpha;

	image.strong_value = value;

	if( value >= strongclassifier.threshold )
		return 1;
	return 0;
}



//ѵ������������  ÿ��ļ�������Ϊf ��ʶ�����Ϊd  ���յ���ʶ��Ϊf_target   ����һ����������������ջ�������************************
void createCascadedClassifier(float f, float d, float f_target, int length)      
{
	//ofstream fout("output.txt");
	cout<<"ÿ�������ʣ�"<<f<<" ����ʣ�"<<d<<" �������ʣ�"<<f_target<<endl<<endl;
	cc.f = f;   //ÿһ��ɽ��ܵ����������
	cc.d = d;   //ÿһ��ɽ��ܵ���С������
	cc.f_target = f_target;   //Ŀ��������
	int i = 0;    //��ǰ�����Ĳ���

	float F[30], D[30];  //���������ʺ�������
	F[0] = 1.0;
	D[0] = 1.0;

	int n;   //ÿ��ǿ�������е�������������  ����������

	//�������Ӳ���ֱ��������ʴﵽҪ��
	while( F[i] > f_target )
	{
		i++;
		cout<<endl;
		cout<<"��ʼѵ����"<<i<<"��ǿ������"<<endl;
		cout<<"����������:"<<FACES<<" ���η�������:"<<NONFACES<<endl;
		
		F[i] = F[i-1];

		n = 0;

		StrongClassifier strongclassifier;

		//��P��N��Ϊ�µ�ѵ����
		for( int j = 1; j <= FACES; j++ )
			image[j] = P[j];
		for( int j = 1; j <= NONFACES; j++ )
			image[j+FACES] = N[j];

		int face_face = 0;
		int nonface_face = 0;

		//������ʶ��û�дﵽҪ���һֱ����ǿ������
		while( F[i] > f * F[i-1] )   
		{
			n++;  //ǿ�������е�������������
			if( n > 199 )
				break;
			//���¼���ȫ���Ļ���ͼ��
			for( int index = 1; index <= FACES+NONFACES; index++ )
				oneImageIntegral(image[index]);
	
			cout<<"����ѵ����"<<n<<"�����������ĺ�ѡǿ������"<<endl;
			//ѵ��һ����n������������ǿ������
		
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
			cout<<"����n="<<n<<"�ĺ�ѡǿ������  ��ֵ��"<<strongclassifier.threshold<<" �����ȣ�"<<D[i]<<" �����ʣ�"<<F[i]<<endl;
		
			//���͵�i��ǿ��������ֵ   �����㷨����************************************************
			
			//����ӽ���ֵ����������ֵ�����ֵ
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

			//ѭ��ֱ����ǰ����������ļ���ʴﵽ d * D[i-1]
		    while( D[i] < d * D[i-1] )
			{
			//	fout<<"���ε����ļ��������  D[i]��"<<D[i]<<" d*D[i-1]:"<<d*D[i-1]<<endl;
				//�����ʻ��˾Ͳ�����
				if( F[i] > f * F[i-1] )
				{
					cout<<"�����ʵ������ˣ�"<<endl;
					break;
				}

				//������D[i]�ˣ��Ͱ�F[i]ҲŪ�������������µ�ǿ������
				if( index < 1 )
				{
					cout<<"�����ȵ������ˣ�"<<endl;
					F[i] = 1;
					break;
				}

				float face_weight = (float)1.0 / (float)FACES;  //һ�������ĳ�ʼȨ��
				int chazhi = ( d * D[i-1] - D[i] ) / face_weight;
			
				if( chazhi < 1 )
				{
					cout<<"�����ȵ������ˣ�"<<endl;
					F[i] = 1;
					break;
				}

				//��ֵ��С����һ�����ݵ�ǿ����������ֵ
				strongclassifier.threshold = image[index-chazhi].strong_value;

				index -= chazhi;

				//������ǰ����������ļ���ʺ���ʶ��
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
				cout<<"������ļ���ʣ�"<<D[i]<<" �����ʣ�"<<F[i]<<endl;	 
			}
			
		}
		
		cout<<"���ɵ�"<<i<<"��ǿ��������"<<"����"<<n<<"����������  ��ֵΪ��"<<strongclassifier.threshold<<" ����ʣ�"<<D[i]<<"  �����ʣ�"<<F[i]<<endl;
		
		cc.strongclassifier[i-1] = strongclassifier;

		int N_index = 1;
		//���õ�ǰǿ��������������ͼ�񣬽����е�ͼ������������N
		for( int k = 1; k <= NONFACES; k++ )
		{
			if( N[k].flag == 0 && detectFaceByStrongClassifier(N[k],1,strongclassifier) == 1 )
				N[N_index++] = N[k];
		}
		NONFACES = N_index - 1;  //���·�����������

		if( NONFACES < 5 )
			break;
	}

	cc.number = i; //��������
}         



//�ü�����������һ����ͼ����з��࣬��ͼ���ʼ����Ϊ20*20���ɷŴ�times��
int detectSubWindowByCascadedClassifier(Image image, float times, CascadedClassifier cascadedclassifier)
{
	//��������������һ�㲻ͨ����Ϊ��������ȫͨ��Ϊ����
	for( int i = 0; i < cascadedclassifier.number; i++ )
		if( detectFaceByStrongClassifier(image, times, cascadedclassifier.strongclassifier[i]) == 0 )
			return 0;
	return 1;
}



//�ж�һ���Ӵ����Ƿ�������
int drawFaceByStrongClassifier(ImageTest &image, float times, StrongClassifier strongclassifier,int x, int y, int length)
{
	float value = 0;  //ǿ��������ͼƬ���ж�ֵ
	int i;
	for( i = 0; i < strongclassifier.number; i++ )
		value += (float)drawdetectFaceByWeakClassifier(image, times, strongclassifier.weakclassifier[i], x, y) * strongclassifier.weakclassifier[i].alpha;

	//ǿ�������������ĺ���ֵ
	image.strong_value = value;

	if( value >= strongclassifier.threshold )
	{

        //����֮ǰ�����������������ľͺϲ�
		for(i = 0; i < face_index; i++)
		{
			//�ϲ���ͬ�ߴ粻ͬλ�õ�������
			if(abs(face[i].x - x) <= 5 && abs(face[i].y - y) <= 5 && face[i].length >= length*0.8)
			{
				face[i].x = (face[i].x * face[i].number + x) / (face[i].number + 1);
				face[i].y = (face[i].y * face[i].number + y) / (face[i].number + 1);
				face[i].length = (face[i].length * face[i].number + length) / (face[i].number + 1);
				face[i].number++;
				break;
			}
		
		}
		//���û������ľ�����һ���µļ�¼
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



//��ǿ������̽��һ��ͼƬ�ϵĸ����Ӵ���
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

	//�������ͼ
	drawoneImageIntegral(img);
	int count;
	while(current < max_times)
	{
		count = 0;
		cout<<"��ǰ�Ŵ���:"<<current;
		for(int i=0; i+20*current<length_x-1; i+=1)
			for(int j=0; j+20*current<length_y-1; j+=1)	
				count += drawFaceByStrongClassifier(img,current,strongclassifier,i,j,20*current);
		cout<<"  ����������"<<count<<endl;
		current *= 1.2;
	}

	for(int i = 0; i < face_index; i++)
		if(face[i].number>5)
			drawFace(image, face[i].x, face[i].y, face[i].length);

}



//�ı�ͼ��ߴ�
void resizeImage(Mat &image, double scale)
{
	Size dsize = Size(image.cols*scale,image.rows*scale);
	Mat image2 = Mat(dsize,CV_32S);
	resize(image, image2,dsize);
	image = image2;
}



int main()
{
	//��ȡMIT������
//	loadImg();
	 

	//����ȫ���Ļ���ͼ��
//	imageIntegral();


//xml�Ļ�����д����
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
    stu1AddressElement->LinkEndChild(new TiXmlText("�й�"));
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

	

	TiXmlElement* rootElement = doc.RootElement();  //SchoolԪ��  
    classElement = rootElement->FirstChildElement();  // ClassԪ��
    TiXmlElement* studentElement = classElement->FirstChildElement();  //Students  
    for (; studentElement != NULL; studentElement = studentElement->NextSiblingElement() ) 
	{
        TiXmlAttribute* attributeOfStudent = studentElement->FirstAttribute();  //���student��name����  
        for (;attributeOfStudent != NULL; attributeOfStudent = attributeOfStudent->Next() ) {
            cout << attributeOfStudent->Name() << " : " << attributeOfStudent->Value() << std::endl;       
        }                                 

        TiXmlElement* studentContactElement = studentElement->FirstChildElement();//���student�ĵ�һ����ϵ��ʽ 
        for (; studentContactElement != NULL; studentContactElement = studentContactElement->NextSiblingElement() ) {
            string contactType = studentContactElement->Value();
            string contactValue = studentContactElement->GetText();
            cout << contactType  << " : " << contactValue << std::endl;           
        }
	}


	system("Pause");
#endif



//C++��XML���ݸ�ʽת��
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


	TiXmlElement* rootElement = doc.RootElement();  //SchoolԪ��  
	TiXmlAttribute* attributeOfStudent = rootElement->FirstAttribute();
	int haha=atoi(attributeOfStudent->Value());

	attributeOfStudent = attributeOfStudent->Next();
	float lala=atof(attributeOfStudent->Value());
	cout<<haha<<' '<<lala<<endl;

#endif



//��������һ�������ʵ�ǿ������
#if 0

	cout<<"ѵ��������......"<<endl;

	//����ǿ������
	StrongClassifier strongclassifier; 
	createStrongClassifier(strongclassifier,WEAK_CLASSIFIERS,20);

	cout<<"������������......"<<endl;

	 //��������
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
		cout<<"  �����:"<<(float)face_face/(float)FACES;
		cout<<"  ��ʶ��:"<<(float)nonface_face/(float)NONFACES<<endl<<endl;


	//����ǿ����������ֵ��ʹ�����ʴﵽҪ��
	sortImageByStrongValue();

	//�ҵ���ӽ�ǿ��������ֵ������i
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
		cout<<"  �����:"<<(float)face_face/(float)FACES;
		cout<<"  ��ʶ��:"<<(float)nonface_face/(float)NONFACES<<endl<<endl;


#endif



//��ǿ�����������strongclassifier.xml�ļ���
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
	


//��xml���ȡǿ������
#if 1
	const char * xmlFile = "strongclassifier300features.xml"; 
	TiXmlDocument doc;  
	doc.LoadFile(xmlFile);
	StrongClassifier strongclassifier;

	TiXmlElement* strongElement = doc.RootElement();  //StrongClassifierԪ��  

	//��ȡstrongclassifier.number
	TiXmlAttribute* attributeOfStudent = strongElement->FirstAttribute();
	strongclassifier.number = atoi(attributeOfStudent->Value());

	//��ȡstrongclassifier.threshold
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
	


//��ǿ��������һ��ͼƬ����̽��
#if 1

		Mat img_test = imread("test.bmp",0);

		resizeImage(img_test,0.5);

		cout<<"���ڼ��ͼƬ......"<<endl;

		drawImageByStrongClassifier(img_test,strongclassifier);
		
#endif


//��ʾ����ͼƬ
#if 1

		namedWindow("haha");
		imshow("haha",img_test);
		waitKey(0);

#endif



//����ǿ�����������ɻ�ROC���ߵ�����
#if 0
	
	//����ǿ������
	StrongClassifier strongclassifier = createStrongClassifier(200,20);

	 //��������
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
		fout<<i<<"-��ֵ��"<<image[i].strong_value;
		fout<<"  �����:"<<(float)faces_right/(float)FACES;
		fout<<"  ��ʶ��:"<<(float)nonfaces_faces/(float)NONFACES<<endl;
	}
	fout.close();

#endif



//���ɼ���������
#if 0
	
	//���ɼ���������
	createCascadedClassifier((float)0.3, (float)0.99, (float)0.00001, 20);

	FACES = 300;
	NONFACES = 300;

	 //��������
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
	fout<<endl<<endl<<"��ʼ����......"<<endl;
	fout<<"���Լ�����:"<<FACES<<"  ʶ�������:";
	fout<<faces_right<<endl;
	fout<<"�����:";
	fout<<(float)faces_right/(float)FACES<<endl;

	fout<<"���Լ�������:"<<NONFACES<<"  ʶ�������:";
	fout<<nonfaces_faces<<endl;
	fout<<"��ʶ�ʣ�";
	fout<<(float)nonfaces_faces/(float)NONFACES<<endl;
	fout<<endl;
	fout.close();
#endif




//��ȡ����ͷ
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






