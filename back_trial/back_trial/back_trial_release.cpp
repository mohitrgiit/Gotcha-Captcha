#include<opencv\cv.h>
#include<opencv\highgui.h>

using namespace cv;

int main()
{
	Mat image;
	VideoCapture cap;
	cap.open(0);
	namedWindow("window", 1);
	int i=0;

	while (1)
	{
		if (i==0)
		{
			cap >> image;
			imshow("window", image);
			waitKey(9000);
			i++;
		}
		else
		{
			cap >> image;
			imshow("window", image);
			waitKey(30);
		}
	}
	return 0;
}
