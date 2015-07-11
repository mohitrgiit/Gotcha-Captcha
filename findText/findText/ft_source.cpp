//Find Text 1

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

using namespace std;
using namespace cv;

vector<Rect> detectLetters(Mat img, int j)
{
	char name[100];
	vector<Rect> boundRect;
	Mat img_gray, img_sobel, img_threshold, element, img_laplacian;
	cvtColor(img, img_gray, CV_BGR2GRAY);
	Sobel(img_gray, img_sobel, CV_8U, 1, 1, 3, 1, 0, BORDER_DEFAULT);
	sprintf(name, "text_newFolder2(1)/sobel%d.jpg", j);
	imwrite(name, img_sobel);
	Laplacian(img_gray,img_laplacian,CV_8U,3,1,0,BORDER_DEFAULT);
	sprintf(name, "text_newFolder2(1)/laplacian%d.jpg", j);
	imwrite(name, img_laplacian);
	threshold(img_laplacian, img_threshold, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
	sprintf(name, "text_newFolder2(1)/thresh%d.jpg", j);
	imwrite(name, img_threshold);
	element = getStructuringElement(MORPH_RECT, Size(17, 3));
	morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);
	sprintf(name, "text_newFolder2(1)/close%d.jpg", j);
	imwrite(name, img_threshold);
	vector<vector<Point> > contours;
	findContours(img_threshold, contours, 0, 1);
	vector<vector<Point> > contours_poly(contours.size());
	for (int i = 0; i < contours.size(); i++)
	if (contours[i].size()>70)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		Rect appRect(boundingRect(Mat(contours_poly[i])));
		if (appRect.width>appRect.height)
			boundRect.push_back(appRect);
	}
	return boundRect;
}

int main(int argc, char** argv)
{
	char name2[100];
	for (int j = 10; j <= 50; j++)
	{
		//Read
		sprintf(name2, "sample/cg%d.jpg", j);
		Mat img1 = imread(name2,1);
		//Detect
		vector<Rect> letterBBoxes1 = detectLetters(img1,j);
		//Display
		for (int i = 0; i< letterBBoxes1.size(); i++)
			rectangle(img1, letterBBoxes1[i], Scalar(0, 255, 0), 3, 8, 0);
		sprintf(name2, "text_newFolder2(1)/ABR%d.jpg", j);
		imwrite(name2, img1);
	}
	return 0;
}



//Find Text 2
/*


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, char** argv[])
{
	char name[20];
	for (int j = 1; j <= 30; j++)
	{
		sprintf(name, "new/ABR%d.jpg", j);
		Mat large = imread(name, 1);
		Mat rgb = large;
		// downsample and use it for processing
		pyrDown(large, rgb);
		Mat small;
		cvtColor(large, small, CV_BGR2GRAY);
		// morphological gradient
		Mat grad;
		Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(2, 23));
		morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
		sprintf(name, "text_newFolder2(2)(2)/morphgrad%d.jpg", j);
		imwrite(name, grad);
		// binarize
		Mat bw;
		threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
		sprintf(name, "text_newFolder2(2)(2)/thresh%d.jpg", j);
		imwrite(name, bw);
		// connect horizontally oriented regions
		Mat connected;
		morphKernel = getStructuringElement(MORPH_RECT, Size(13, 3));
		morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
		sprintf(name, "text_newFolder2(2)(2)/morphclose%d.jpg", j);
		imwrite(name, connected);
		// find contours
		Mat mask = Mat::zeros(bw.size(), CV_8UC1);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		sprintf(name, "text_newFolder2(2)(2)/connected%d.jpg", j);
		imwrite(name, connected);

		// filter contours
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
		{
			Rect rect = boundingRect(contours[idx]);
			Mat maskROI(mask, rect);
			maskROI = Scalar(0, 0, 0);
			// fill the contour
			drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
			// ratio of non-zero pixels in the filled region
			double r = (double)countNonZero(maskROI) / (rect.width*rect.height);
			// assume at least 45% of the area is filled if it contains text ; constraints on region size ; these two conditions alone are not very robust. better to use something
			//	like the number of significant peaks in a horizontal projection as a third condition 
			if (r > .45 && (rect.height > 8 && rect.width > 8))
			{
				rectangle(rgb, rect, Scalar(0, 255, 0), 2);
			}
		}
		sprintf(name, "text_newFolder2(2)(2)/ABR%d.jpg", j);
		imwrite(name, rgb);
	}


	return 0;

}
*/
