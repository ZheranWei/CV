#include "qrdetect.h"
#include <typeinfo>

int preNUm(unsigned char byte) {
	unsigned char mask = 0x80;
	int num = 0;
	for (int i = 0; i < 8; i++) {
		if ((byte & mask) == mask) {
			mask = mask >> 1;
			num++;
		}
		else {
			break;
		}
	}
	return num;
}

int main()
{
	Qrdetect qd;
	Mat src = imread("./qrcode/qrcode_07.jpg");
	
	
	if (src.empty()) {
		printf("No such file or directory...");
		return -1;
	}
	
	namedWindow("input", WINDOW_FREERATIO);
	imshow("input", src);
	Mat image = qd.Qrcode(src);
	
	ImageScanner scanner;
	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
	Mat Gray;
	cvtColor(image, Gray, COLOR_RGB2GRAY);
	//Mat imageGray = Gray(Rect(Point(338, 473), Point(1148, 652)));
	Mat imageGray = Gray.clone();
	int width = imageGray.cols;
	int height = imageGray.rows;
	Image imageZbar(width, height, "Y800", imageGray.data, width * height);
	int n = scanner.scan(imageZbar); //扫描二维码      

	if (imageZbar.symbol_begin() == imageZbar.symbol_end())
	{
		cout << "查询条码失败，请检查图片！" << endl;
	}

	for (Image::SymbolIterator symbol = imageZbar.symbol_begin(); symbol != imageZbar.symbol_end(); ++symbol)
	{
		String type = symbol->get_type_name();
		String msg = symbol->get_data();
		
		cout << "类型：" << type << endl;
		cout << "解码：" << msg << endl;

	}

	waitKey(0);
	imageZbar.set_data(NULL, 0);
	
	return 0;
}