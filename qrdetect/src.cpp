#include "qrdetect.h"

bool isXCorner(Mat &image);
bool isYCorner(Mat &image);
Mat transformCorner(Mat &image, RotatedRect &rect);

Mat Qrdetect::Qrcode(Mat & image) {
	//图像二值化
	Mat src = image.clone();
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	//全局阈值分割
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

	// 码眼矩形轮廓分析
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	Moments monents;
	findContours(binary.clone(), contours, hireachy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point());
	Mat result = Mat::zeros(image.size(), CV_8UC1);
	for (size_t t = 0; t < contours.size(); t++) {
		double area = contourArea(contours[t]);
		//过滤面积小于100的轮廓
		if (area < 100) continue;
		//计算轮廓的最小外接矩形
		RotatedRect rect = minAreaRect(contours[t]);
		float w = rect.size.width;
		float h = rect.size.height;
		//计算轮廓的横纵比并进行过滤
		float rate = min(w, h) / max(w, h);
		if (rate > 0.85 && w < image.cols / 4 && h < image.rows / 4) {
			//对轮廓进行透视变换，得到轮廓最小外接矩形的俯视图
			Mat qr_roi = transformCorner(image, rect);
			// 根据矩形特征进行几何分析，如果符合码眼内部比例关系则画出轮廓
			if (isXCorner(qr_roi)) {
				//在原图上画出满足内部比例关系的后的轮廓（三个码眼）
				//drawContours(image, contours, static_cast<int>(t), Scalar(255, 0, 0), 2, 8);
				drawContours(result, contours, static_cast<int>(t), Scalar(255), 2, 8);
			}
		}
	}

	// 求取三个码眼的最小外接矩形以定位二维码
	vector<Point> pts;
	for (int row = 0; row < result.rows; row++) {
		for (int col = 0; col < result.cols; col++) {
			int pv = result.at<uchar>(row, col);
			if (pv == 255) {
				pts.push_back(Point(col, row));
			}
		}
	}
	if (pts.size() == 0) {
		cout << "Can not find any QRcode!" << endl;
		return src;
	}
	//对二维码的roi区域进行透视变换
	RotatedRect rrt = minAreaRect(pts);
	Mat res = transformCorner(src, rrt);
	namedWindow("qrcode-roi", WINDOW_FREERATIO);
	imshow("qrcode-roi", res);
	
	return res;
}
bool isXCorner(Mat &image) {
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	int xb = 0, yb = 0;
	int w1x = 0, w2x = 0;
	int b1x = 0, b2x = 0;

	int width = binary.cols;
	int height = binary.rows;
	int cy = height / 2;
	int cx = width / 2;
	int pv = binary.at<uchar>(cy, cx);
	if (pv == 255) return false;
	// 从中心向两边验证黑白宽度的比例关系
	bool findleft = false, findright = false;
	int start = 0, end = 0;
	int offset = 0;
	while (true) {
		offset++;
		if ((cx - offset) <= width / 8 || (cx + offset) >= width - 1) {
			start = -1;
			end = -1;
			break;
		}
		pv = binary.at<uchar>(cy, cx - offset);
		if (pv == 255) {
			start = cx - offset;
			findleft = true;
		}
		pv = binary.at<uchar>(cy, cx + offset);
		if (pv == 255) {
			end = cx + offset;
			findright = true;
		}
		if (findleft && findright) {
			break;
		}
	}

	if (start <= 0 || end <= 0) {
		return false;
	}
	xb = end - start;
	for (int col = start; col > 0; col--) {
		pv = binary.at<uchar>(cy, col);
		if (pv == 0) {
			w1x = start - col;
			break;
		}
	}
	for (int col = end; col < width - 1; col++) {
		pv = binary.at<uchar>(cy, col);
		if (pv == 0) {
			w2x = col - end;
			break;
		}
	}
	for (int col = (end + w2x); col < width; col++) {
		pv = binary.at<uchar>(cy, col);
		if (pv == 255) {
			b2x = col - end - w2x;
			break;
		}
		else {
			b2x++;
		}
	}
	for (int col = (start - w1x); col > 0; col--) {
		pv = binary.at<uchar>(cy, col);
		if (pv == 255) {
			b1x = start - col - w1x;
			break;
		}
		else {
			b1x++;
		}
	}

	float sum = xb + b1x + b2x + w1x + w2x;
	//printf("xb : %d, b1x = %d, b2x = %d, w1x = %d, w2x = %d\n", xb , b1x , b2x , w1x , w2x);
	xb = static_cast<int>((xb / sum)*7.0 + 0.5);
	b1x = static_cast<int>((b1x / sum)*7.0 + 0.5);
	b2x = static_cast<int>((b2x / sum)*7.0 + 0.5);
	w1x = static_cast<int>((w1x / sum)*7.0 + 0.5);
	w2x = static_cast<int>((w2x / sum)*7.0 + 0.5);
	printf("xb : %d, b1x = %d, b2x = %d, w1x = %d, w2x = %d\n", xb, b1x, b2x, w1x, w2x);
	if ((xb == 3 || xb == 4) && b1x == b2x && w1x == w2x && w1x == b1x && b1x == 1) { // 1:1:3:1:1
		return true;
	}
	else {
		return false;
	}
}
//为了节省计算量y方向可不做，如果要做判断依据是黑色高度大于白色高度
bool isYCorner(Mat &image) {
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	int width = binary.cols;
	int height = binary.rows;
	int cy = height / 2;
	int cx = width / 2;
	int pv = binary.at<uchar>(cy, cx);
	int bc = 0, wc = 0;
	bool found = true;
	for (int row = cy; row > 0; row--) {
		pv = binary.at<uchar>(row, cx);
		if (pv == 0 && found) {
			bc++;
		}
		else if (pv == 255) {
			found = false;
			wc++;
		}
	}
	bc = bc * 2;
	if (bc <= wc) {
		return false;
	}
	return true;
}

Mat transformCorner(Mat &image, RotatedRect &rect) {
	int width = static_cast<int>(rect.size.width);
	int height = static_cast<int>(rect.size.height);
	Mat result = Mat::zeros(height, width, image.type());
	Point2f vertices[4];

	rect.points(vertices);
	vector<Point> src_corners;
	vector<Point> dst_corners;
	//左下角为起点逆时针加入轮廓最小外接矩形的其他点
	dst_corners.push_back(Point(0, height));
	dst_corners.push_back(Point(0, 0));
	dst_corners.push_back(Point(width, 0));
	dst_corners.push_back(Point(width, height));
	
	for (int i = 0; i < 4; i++) {
		src_corners.push_back(vertices[i]);
	}
	Mat h = findHomography(src_corners, dst_corners);
	warpPerspective(image, result, h, result.size());
	return result;
}
