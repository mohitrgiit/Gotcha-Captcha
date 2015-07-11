#include <opencv2\highgui\highgui.hpp>
#include <opencv\highgui.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <opencv\highgui.h>
#include <opencv\cv.h>
#include <Windows.h>
#include <opencv2\video\background_segm.hpp>
#include <opencv2\photo\photo.hpp>


using namespace cv;
using namespace std;


int main()
{
	char name[20],name2[20];
	
	for (int i = 2; i <= 30; i++)
	{
		sprintf(name, "boxedIn/ABRaa%d.jpg", i);
		Mat input = imread(name, 1);
		Mat img_gray;
		cvtColor(input, img_gray, COLOR_BGR2GRAY);
		Mat img_threshold2;
		adaptiveThreshold(img_gray, img_threshold2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 7);
		sprintf(name2, "gaussianboxedIn/ABRaa%d.jpg", i);
		imwrite(name2, img_threshold2);
	}
	
	/*
	Mat input = imread("sample/cg28.jpg");
	Mat img_gray;
	cvtColor(input, img_gray, COLOR_BGR2GRAY);
	imshow("Image", img_gray);

	Mat img_sobel;
	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 1, 3, 0);
	imshow("Sobel", img_sobel);

	Mat img_threshold1;
	threshold(img_gray, img_threshold1, 0, 255, THRESH_OTSU + THRESH_BINARY);
	imshow("Threshold-Otsu", img_threshold1);

	Mat img_threshold2;
	adaptiveThreshold(img_gray, img_threshold2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 17, 6);
	imshow("Threshold-adaptive(mean)",img_threshold2);

	Mat img_denoised;
	fastNlMeansDenoising(img_threshold2,img_denoised,25,7,27);
	imshow("Denoised", img_denoised);

	Mat img_threshold3;
	adaptiveThreshold(img_gray, img_threshold3, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 17, 6);
	imshow("Threshold-adaptive(gaussian)", img_threshold3);
	*/

	waitKey(0);
	return 0;

}