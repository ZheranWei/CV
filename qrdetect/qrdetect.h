#include <opencv2/opencv.hpp>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include "zbar.h"  

using namespace cv;
using namespace std;
using namespace cv::ml;
using namespace zbar;


class Qrdetect {
public:
	
	Mat Qrcode(Mat & image);
	
};
