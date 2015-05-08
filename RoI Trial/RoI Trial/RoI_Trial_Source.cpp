#include <opencv2\highgui\highgui.hpp>
#include <opencv\highgui.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>



using namespace cv;
using namespace std;

int main()
{

	Mat img_1 = imread("image15.jpg", -1);
	int lc_x = ((img_1.rows) / 2);
	int lc_y = ((img_1.cols) / 2);
	int br = 80;

	cout << lc_x << endl;
	cout << lc_y << endl;
	Rect rec = Rect(lc_y-(br)/2, lc_x-(br)/2, br, br);
	Mat RoI=img_1(rec);



	imshow("Original", img_1);
	imshow("RoI image", RoI);

	waitKey(0);

	return 0;

}