/**
* @face_detection.cpp
* @author:wolf
* ���demoʹ��opencvʵ��ͼƬ��������⡢�Լ��滭��������ȹ��ܡ�
* ÿ������д��һ��������������ֲʹ�á�
* �ο���opencv�����滭��������ģ���ĵ���
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

//ͼ��ṹ��
struct Image
{
	Mat img;						//�Ҷ�ͼ�����
	float weight;					//ÿ��ͼ��Ȩ��
	int flag;						//����Ϊ1��������Ϊ0
	int img_integral[25][25];   	 //����ͼ����� 1-FACES+NONFACES
	int feature_value;				//��Ӧһ����������������ֵ
}image[1501];	    //�±귶ΧΪ1-1500

Image P[501];  //��ǰ������������������
Image N[1001];   //��ǰ�����������ķ�������
Image N_test[1001];

//�����ṹ��
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
}wc[200];             //�±�Ϊ0-199
int wc_index = 0;


//ǿ�������ṹ��
struct StrongClassifier
{
	WeakClassifier weakclassifier[200];   //�������������� ���200��
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
		 P[i] = image[i];
	 }

	 //��ȡNONFACES�ŷ������Ҷ�ͼ
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
				image.img_integral[x][y] = (int)image.img.at<uchar>(0,0);

			//Y���ϵĵ�=���ڵ�+�˵������
			else if( x == 0 )
				image.img_integral[x][y] = image.img_integral[0][y-1] + (int)image.img.at<uchar>(0,y);

			//X���ϵĵ�=���ڵ�+�˵�����
			else if( y == 0 )
				image.img_integral[x][y] = image.img_integral[x-1][0] + (int)image.img.at<uchar>(x,0);

			//������=���ڵ�+���ڵ�-���ϵ�+�˵�����
			else
				image.img_integral[x][y] = image.img_integral[x-1][y] + image.img_integral[x][y-1]   \
											 - image.img_integral[x-1][y-1] + (int)image.img.at<uchar>(x,y);
		}	 
}



//��������ͼ��Ļ���ͼ��
void imageIntegral()
{
	for( int index = 1; index <= FACES+NONFACES; index++ )
		oneImageIntegral(image[index]);
}



//����һ��ͼƬ�����һ������������ֵ
int oneFeatureValue(Image image, Feature feature)
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
		int x = 1 * x_times;
		int y = 2 * y_times;
		feature_value = 2 * ( image.img_integral[ x1 + x ][ y1 + y / 2 ] - image.img_integral[ x1 ][ y1 + y / 2 ] )  \
								- ( image.img_integral[ x1 + x ][ y1 ] - image.img_integral[ x1 ][ y1 ] )                    \
								- ( image.img_integral[ x1 + x ][ y1 + y ] - image.img_integral[ x1 ][ y1 + y ] );

	}

	//(s,t)����Ϊ(2,1)�ľ�������,����Ϊ2
	if( templet == 2 )
	{
		//ÿ�������ĳ��Ϳ�������������������
		int x = 2 * x_times;
		int y = 1 * y_times;
		feature_value = 2 * ( image.img_integral[ x1 + x / 2 ][ y1 + y ] - image.img_integral[ x1 + x / 2 ][ y1 ] )    \
								- ( image.img_integral[ x1 ][ y1 + y ] - image.img_integral[ x1 ][ y1 ] )              \
								- ( image.img_integral[ x1 + x ][ y1 + y ] - image.img_integral[ x1 + x ][ y1 ] );
	}

	//(s,t)����Ϊ(1,3)�ľ�������,����Ϊ3
	if( templet == 3 )
	{
		//ÿ�������ĳ��Ϳ�������������������
		int x = 1 * x_times;
		int y = 3 * y_times;
		feature_value = 3 * ( image.img_integral[ x1 + x ][ y1 + y / 3 ] - image.img_integral[ x1 ][ y1 + y / 3 ] )         \
								- 3 * ( image.img_integral[ x1 + x ][ y1 + 2 * y / 3 ] - image.img_integral[ x1 ][ y1 + 2 * y / 3 ] )    \
								+ ( image.img_integral[ x1 + x ][ y1 + y ] - image.img_integral[ x1 ][ y1 + y ] )                        \
								- ( image.img_integral[ x1 + x ][ y1 ] - image.img_integral[ x1 ][ y1 ] );		
	}

	//(s,t)����Ϊ(3,1)�ľ�������,����Ϊ4
	if( templet == 4 )
	{
		//ÿ�������ĳ��Ϳ�������������������
		int x = 3 * x_times;
		int y = 1 * y_times;
		feature_value = 3 * ( image.img_integral[ x1 + x / 3 ][ y1 + y ] - image.img_integral[ x1 + x / 3 ][ y1 ] )      \
								- 3 * ( image.img_integral[ x1 + 2 * x / 3 ][ y1 + y ] - image.img_integral[ x1 + 2 * x / 3 ][ y1 ] )     \
								+ ( image.img_integral[ x1 + x ][ y1 + y ] - image.img_integral[ x1 + x ][ y1 ] )                         \
								- ( image.img_integral[ x1 ][ y1 + y ] - image.img_integral[ x1 ][ y1 ] );	
	}

	//(s,t)����Ϊ(2,2)�ľ�������,����Ϊ5
	if( templet == 5 )
	{
		//ÿ�������ĳ��Ϳ�������������������
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



//��ͼ����ݵ�ǰ����ֵ��С��������
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



//ͨ������featureѵ����������
WeakClassifier createWeakClassifier(Feature feature)
{
	//���ݾ�������feature��������ͼƬ������ֵ
	for( int i = 1; i <= FACES + NONFACES; i++ )
		image[i].feature_value = oneFeatureValue(image[i], feature);


	//��ͼ������ֵ����
	sortImageByFeatureValue();

	WeakClassifier weakclassifier; //��������

	weakclassifier.threshold = image[0].feature_value;  //��ǰ��ֵ
	weakclassifier.error_rate = (float)100;             //��ǰ��С������
	weakclassifier.p = 1;                               //��ǰ��С�������µĲ���ʽ��������<
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
	
	return weakclassifier;
}



//��һ������������һ��ͼƬ����̽��,���������б�ʽ
int detectFaceByWeakClassifier(Image image, WeakClassifier weakclassifier)
{
	//�������ͼ
	oneImageIntegral(image);

	//����ͼƬ����������������������ֵ
	image.feature_value = oneFeatureValue(image, weakclassifier.feature);

	//�������������б�ʽ�ж�ͼƬ�Ƿ�������
	if( image.feature_value * weakclassifier.p <= weakclassifier.threshold * weakclassifier.p )
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
					WeakClassifier weakclassifier = createWeakClassifier(feature);		

					//�������������пɴ�Ÿ���Ϊ200�� �±�0-199������֮�������������������ʱ����һ��С���򸲸����һ����������
					if( wc_index == 200 && weakclassifier.error_rate < wc[199].error_rate )
					{
						wc[wc_index-1] = weakclassifier;
						sortWeakClassifier();
					}
					//���û��������һֱ�����ֱ�������Ǵ���һ����
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



//ѵ��һ���� numberOfWeakClassifier ������������ǿ������   ���ڴ�Сm
StrongClassifier createStrongClassifier(int numberOfWeakClassifier, int length)
{
	StrongClassifier strongclassifier;

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
	for( int i = 0; i < numberOfWeakClassifier; i++ )
		strongclassifier.weakclassifier[i] = wc[i];
	

	//ǿ����������ֵ
	strongclassifier.threshold = 0;	
	for( int i = 0; i < strongclassifier.number; i++ )
		strongclassifier.threshold += strongclassifier.weakclassifier[i].alpha / 2;

	return strongclassifier;
}



//��һ��ǿ������strongclassifier��һ��ͼƬimage���з���
int detectFaceByStrongClassifier(Image image, StrongClassifier strongclassifier)
{
	float value = 0;  //ǿ��������ͼƬ���ж�ֵ

	for( int i = 0; i < strongclassifier.number; i++ )
		value += (float)detectFaceByWeakClassifier(image, strongclassifier.weakclassifier[i]) * strongclassifier.weakclassifier[i].alpha;

	if( value >= strongclassifier.threshold )
		return 1;
	return 0;
}



//ѵ������������  ÿ��ļ�������Ϊf ��ʶ�����Ϊd  ���յ���ʶ��Ϊf_target   ����һ����������������ջ�������************************
void createCascadedClassifier(float f, float d, float f_target, int length)      
{
	cout<<"ÿ�������ʣ�"<<f<<" ����ʣ�"<<d<<" �������ʣ�"<<f_target<<endl<<endl;
	cc.f = f;   //ÿһ��ɽ��ܵ����������
	cc.d = d;   //ÿһ��ɽ��ܵ���С������
	cc.f_target = f_target;   //Ŀ��������
	int i = 0;    //��ǰ�����Ĳ���

	float F[30], D[30];  //���������ʺ�������
	F[0] = 1.0;
	D[0] = 1.0;
	
	//�������Ӳ���ֱ��������ʴﵽҪ��
	while( F[i] > f_target )
	{
		i++;
		cout<<endl<<endl<<endl;
		cout<<"��ʼѵ����"<<i<<"��ǿ������"<<endl;
		cout<<"����������:"<<FACES<<" ���η�������:"<<NONFACES<<endl;
		int n = 0;   //ÿ��ǿ�������е�������������  ����������
		F[i] = F[i-1];

		StrongClassifier strongclassifier;

		int face_face = 0;
		int nonface_face = 0;

		//������ʶ��û�дﵽҪ���һֱ����ǿ������
		while( F[i] > f * F[i-1] )   
		{
			n++;  //ǿ�������е�������������
			//��P��N��Ϊ�µ�ѵ����
			for( int j = 1; j <= FACES; j++ )
				image[j] = P[j];
			for( int j = 1; j <= NONFACES; j++ )
				image[j+FACES] = N[j];


			//���¼���ȫ���Ļ���ͼ��
			for( int index = 1; index <= FACES+NONFACES; index++ )
				oneImageIntegral(image[index]);
			
			cout<<"����ѵ��������"<<n<<"������������ǿ������......"<<endl;

			//ѵ��һ����n������������ǿ������
			strongclassifier = createStrongClassifier(n, length);
			
			cout<<"��ѡǿ��������ÿ����������:��ֵ  alpha  �����ʣ�"<<endl;
			for(int a=0;a<n;a++)
				cout<<strongclassifier.weakclassifier[a].threshold<<"  "<<strongclassifier.weakclassifier[a].alpha<<"  "<<strongclassifier.weakclassifier[a].error_rate<<endl;
			cout<<endl;
			
			cout<<"����һ����"<<i<<"��ĺ�ѡǿ������..."<<"����"<<n<<"����������  ��ֵΪ��"<<strongclassifier.threshold<<endl;
			//������ǰ����������ļ����D[i]����ʶ��F[i]
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
			cout<<"��ѡǿ�������ļ���ʣ�"<<D[i]<<"  �����ʣ�"<<F[i]<<endl;

			//ѭ��ֱ����ǰ����������ļ���ʴﵽ d * D[i-1]
			while( D[i] < d * D[i-1] )
			{
				 //���͵�i��ǿ��������ֵ   �����㷨����************************************************
				if(strongclassifier.threshold > 0)
					strongclassifier.threshold *=0.8;

				//������ǰ����������ļ���ʺ���ʶ��
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
				cout<<"��ѡǿ������������  ��ֵ:"<<strongclassifier.threshold<<"  ����ʣ�"<<D[i]<<"  �����ʣ�"<<F[i]<<endl;
			}
		}
		
		cout<<endl<<"���ɵ�"<<i<<"��ǿ�������� �����������ļ���ʣ�"<<D[i]<<"  �����ʣ�"<<F[i]<<endl;
		
		cc.strongclassifier[i] = strongclassifier;

		int N_index = 1;
		//���õ�ǰǿ��������������ͼ�񣬽����е�ͼ������������N
		for( int k = 1; k <= NONFACES; k++ )
		{
			if( N[k].flag == 0 && detectFaceByStrongClassifier(N[k],strongclassifier) == 1 )
				N[N_index++] = N[k];
		}
		NONFACES = N_index - 1;  //���·�����������
	}

	cc.number = i+1; //��������
}         



//������  iΪ���ɲ�ͬ���ƵĴ���
void drawFace(Image image, int length,int i)
{
	stringstream buffer;
         buffer<<i;
		 namedWindow(buffer.str());
	Point center(length/2,length/2);
	rectangle(image.img,Point(0,0),Point(length,length),Scalar(0,0,255),2,8,0);
	imshow(buffer.str(),image.img);
}



//�ü�����������һ����ͼ����з���
int detectFaceByCascadedClassifier(Image image, CascadedClassifier cascadedclassifier)
{
	//��������������һ�㲻ͨ����Ϊ��������ȫͨ��Ϊ����
	for( int i = 0; i < cascadedclassifier.number; i++ )
		if( detectFaceByStrongClassifier(image, cascadedclassifier.strongclassifier[i]) == 0 )
			return 0;
	return 1;
}




int main()
{
	//��ȡMIT������
	loadImg();
	 

	//����ȫ���Ļ���ͼ��
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
	cout<<"������:"<<FACES<<"  ʶ�������:";
	show(faces_right);
	cout<<"�����:";
	show((float)faces_right/(float)FACES);

	cout<<"�ܷ�����:"<<NONFACES<<"  ʶ�������:";
	show(nonfaces_faces);
	cout<<"��ʶ�ʣ�";
	show((float)nonfaces_faces/(float)NONFACES);
#endif
	
#if 0
	
	//����ǿ������
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
	cout<<"��ֵ��"<<strongclassifier.threshold<<endl;
	cout<<"������:"<<FACES<<"  ʶ�������:";
	show(faces_right);
	cout<<"�����:";
	show((float)faces_right/(float)FACES);

	cout<<"�ܷ�����:"<<NONFACES<<"  ʶ�������:";
	show(nonfaces_faces);
	cout<<"��ʶ�ʣ�";
	show((float)nonfaces_faces/(float)NONFACES);
	cout<<endl;
	
#endif


#if 1
	
	//���ɼ���������
	createCascadedClassifier((float)0.3, (float)0.99, (float)0.00001, 20);


	//������Լ�
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
	cout<<endl<<endl<<"��ʼ����......"<<endl;
	cout<<"���Լ�����:"<<FACES_TEST<<"  ʶ�������:";
	show(faces_right);
	cout<<"�����:";
	show((float)faces_right/(float)FACES_TEST);

	cout<<"���Լ�������:"<<NONFACES_TEST<<"  ʶ�������:";
	show(nonfaces_faces);
	cout<<"��ʶ�ʣ�";
	show((float)nonfaces_faces/(float)NONFACES_TEST);
	cout<<endl;
	
#endif


	 
	

//	namedWindow("haha");
	//imshow("haha",image[1].img);
  
	waitKey(0);
    return 0;
}






