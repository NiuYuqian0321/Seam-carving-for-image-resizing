#include<iostream>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

//#define DEBUG 1

const int SIZE = 150;

void image_resize(Mat &image,Mat &image_new);
void image_toGray(Mat &image,Mat &image_gray);
void image_calEnergy(Mat &image,Mat &G);
void image_calEnergyAll(const Mat &G,Mat &M,Mat &R);
void image_findEnergyLine(Mat &image,Mat &M,const Mat &R);
void image_showEnergyLine(Mat &image,Mat &MinEnergyLine);
void image_deleteEnergyLine(const Mat &image,Mat &image_new,const Mat &MinEnergyLine);

int main()
{
	//原图
	//Mat image = imread("E:\\vs2012\\Projects\\ImageResizing\\ImageResizing\\p2.jpg");
	Mat image = imread("p2.jpg");
	if(!image.empty())
	{
		namedWindow("image");
		imshow("image",image);
	}
	
	Mat image_src;

	//一般缩小
	Mat image_dst;
	image.copyTo(image_src);
	resize(image_src,image_dst,Size(image_src.cols-SIZE,image_src.rows));
	if(!image_dst.empty())
	{
		namedWindow("image_dst");
		imshow("image_dst",image_dst);
		imwrite("image_dst.jpg",image_dst);
	}

	//智能缩小
	Mat image_new;
	for (int i = 0;i < SIZE;i++)
	{
		image_resize(image_src,image_new);//核心函数
		image_src = image_new;
		waitKey(50);
	}
	if(!image_src.empty())
	{
		namedWindow("image_src");
		imshow("image_src",image_src);
		imwrite("image_src.jpg",image_src);
	}
	waitKey(0);
	return 0;
}

void image_resize(Mat &image,Mat &image_new)
{
	//转换为单通道
	Mat image_gray(image.rows,image.cols,CV_8U,Scalar(0));
	image_toGray(image,image_gray);
	
	//计算能量矩阵G
	Mat G(image.rows,image.cols,CV_32F,Scalar(0));
	image_calEnergy(image_gray,G);
	
	//计算能量累计矩阵M，标记矩阵R
	Mat M (image.rows,image.cols,CV_32F,Scalar(0));
	Mat R (image.rows,image.cols,CV_32F,Scalar(0));
	image_calEnergyAll(G,M,R);
	
	//找到最小能量线
	Mat MinEnergyLine(image.rows,1,CV_32F,Scalar(0));//存列号
	image_findEnergyLine(MinEnergyLine,M,R);

	//显示最小能量线
	image_showEnergyLine(image,MinEnergyLine);

	//删除最小能量线
	image_deleteEnergyLine(image,image_new,MinEnergyLine);
}
void image_toGray(Mat &image,Mat &image_gray)
{
	if(image.channels() != 1)
	{
		cvtColor(image,image_gray,CV_BGR2GRAY);
	}
}
void image_calEnergy(Mat &image_gray,Mat &G)
{
	Mat image_x(image_gray.rows,image_gray.cols,CV_32F,Scalar(0));
	Mat image_y(image_gray.rows,image_gray.cols,CV_32F,Scalar(0));
	
	//Mat kernel_H = (Mat_<float>(3,3) << 0, 0, 0, 0, 1, -1, 0, 0, 0); //求水平梯度所使用的卷积核（赋初始值）
	//Mat kernel_V = (Mat_<float>(3,3) << 0, 0, 0, 0, 1, 0, 0, -1, 0); //求垂直梯度所使用的卷积核（赋初始值）
	
	Mat kernel_H = (Mat_<float>(3,3) << 1, 2, 1, 0, 0, 0, -1, -2, -1); //求水平梯度所使用的卷积核（赋初始值）
	Mat kernel_V = (Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1); //求垂直梯度所使用的卷积核（赋初始值）

	filter2D(image_gray,image_x,image_x.depth(),kernel_H);
	filter2D(image_gray,image_y,image_y.depth(),kernel_V);

	add(abs(image_x),abs(image_y),G);//水平与垂直滤波结果的绝对值相加，可以得到近似梯度大小

#ifdef DEBUG
	////如果要显示梯度大小这个图，因为G深度是CV_32F，所以需要先转换为CV_8U
	Mat showG;
	G.convertTo(showG,CV_8U,1,0);
	imshow("G",showG);
#endif
}
void image_calEnergyAll(const Mat &G,Mat &M,Mat &R)
{
	G.copyTo(M);

	for (int i = 1;i < G.rows;i++)  //从第2行开始计算
	{
		const float* currentG = G.ptr<const float>(i);
		float* previousR = R.ptr<float>(i-1);
		float* currentM = M.ptr<float>(i);
		float* currentR = R.ptr<float>(i);
		float* previousM = M.ptr<float>(i-1);

		//第一列
		if(previousM[0]<=previousM[1])
		{
			currentM[0] = currentG[0]+previousM[0];
			currentR[0] = 1;
		}
		else
		{
			currentM[0] = currentG[0]+previousM[1];
			currentR[0] = 2;
		}
		
		//中间列
		for (int j = 1;j < G.cols-1;j++)
		{
			float k[3];
			k[0] = previousM[j-1];
			k[1] = previousM[j];
			k[2] = previousM[j+1];

			int index = 0;
			if (k[1] < k[0])
				index = 1;
			if (k[2] < k[index])
				index = 2; 
			currentM[j] = currentG[j]+previousM[j-1+index];
			currentR[j] = index;
		}

		//最后一列
		if( previousM[G.cols-1] <= previousM[G.cols-2])
		{
			currentM[G.cols-1] = currentG[G.cols-1] + previousM[G.cols-1];
			currentR[G.cols-1] = 1;
		}
		else
		{
			currentM[G.cols-1] = currentG[G.cols-1] + previousM[G.cols-2];
			currentR[G.cols-1] = 0;
		}
	}
#ifdef DEBUG
	////如果要显示能量矩阵和标记矩阵，因为M和R深度是CV_32F，所以需要先转换为CV_8U
	Mat showM;
	M.convertTo(showM,CV_8U,1,0);
	imshow("M",showM);
	Mat showR;
	R.convertTo(showR,CV_8U,1,0);
	imshow("R",showR);
#endif
}
void image_findEnergyLine(Mat &MinEnergyLine,Mat &M,const Mat &R)
{
	int index = 0;

	// 获得index，即最后那行最小值的位置
	float *lastM = M.ptr<float>(M.rows - 1);
	for (int i = 1;i < M.cols;i++)
	{
		if (lastM[i] < lastM[index])
		{
			index = i;
		} 
	} 
	cout<<index<<endl;
	{
		MinEnergyLine.at<float>(MinEnergyLine.rows-1,0) = index;

		int tmpIndex = index;

		for (int i = MinEnergyLine.rows-1;i > 0;i--)
		{
			const float *currentR = R.ptr<const float>(i);
			int temp =currentR[tmpIndex];

			if (temp == 0) // 往左走
			{
				tmpIndex = tmpIndex - 1;
			}
			else if (temp == 2) // 往右走
			{
				tmpIndex = tmpIndex + 1;
			} // 如果temp = 1，则往正上走，tmpIndex不需要做修改

			float *previousE = MinEnergyLine.ptr<float>(i-1);
			previousE[0] = tmpIndex;
		}
	}
}
void image_showEnergyLine(Mat &image,Mat &MinEnergyLine)
{
	Mat tmpImage(image.rows,image.cols,image.type());
	image.copyTo(tmpImage);
	for (int i = 0;i < image.rows;i++)
	{
		int k = MinEnergyLine.at<float>(i,0);
		tmpImage.at<Vec3b>(i,k)[0] = 0;
		tmpImage.at<Vec3b>(i,k)[1] = 0;
		tmpImage.at<Vec3b>(i,k)[2] = 255;
	}
	imshow("MinEnergyLine",tmpImage);
}
void image_deleteEnergyLine(const Mat &image,Mat &image_new,const Mat &MinEnergyLine)
{
	Mat image2(image.rows,image.cols-1,image.type());
	for (int i = 0;i < image2.rows;i++)
	{
		int k = MinEnergyLine.at<float>(i,0);
		
		for (int j = 0;j < k;j++)
		{
			image2.at<cv::Vec3b>(i,j)[0] = image.at<cv::Vec3b>(i,j)[0];
			image2.at<cv::Vec3b>(i,j)[1] = image.at<cv::Vec3b>(i,j)[1];
			image2.at<cv::Vec3b>(i,j)[2] = image.at<cv::Vec3b>(i,j)[2];
		}
		for (int j = k;j < image2.cols-1;j++)
		{
			if (j == image2.cols-1)
			{
				int a = 1;
			}
			image2.at<cv::Vec3b>(i,j)[0] = image.at<cv::Vec3b>(i,j+1)[0];
			image2.at<cv::Vec3b>(i,j)[1] = image.at<cv::Vec3b>(i,j+1)[1];
			image2.at<cv::Vec3b>(i,j)[2] = image.at<cv::Vec3b>(i,j+1)[2];

		}
	}
	image2.copyTo(image_new);
#ifdef DEBUG
	namedWindow("image_new");
	imshow("image_new",image_new);
#endif 
}