#include <inference_engine.hpp>
#include<opencv2/opencv.hpp>
#include<fstream>
//using namespace InferenceEngine;
//using namespace std;
//using namespace cv;

InferenceEngine::InferRequest infer_request;

std::string input_name = "";
std::string output_name = "";

auto inferApi(cv::Mat &src, std::string xml, std::string bin, bool rgb = false, Precision ip = Precision::FP32, Precision op = Precision::FP32, Layout lo = Layout::NCHW) {
	Core ie;
	if (rgb)cvtColor(src, src, cv::COLOR_BGR2RGB);
	InferenceEngine::CNNNetwork network = ie.ReadNetwork(xml, bin);
	InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
	InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();

	for (auto item : inputs)
	{
		input_name = item.first;
		auto input_data = item.second;
		input_data->setPrecision(ip);
		input_data->setLayout(lo);
		std::cout << "input_name:" << input_name << std::endl;
	}

	for (auto item : outputs)
	{
		output_name = item.first;
		auto output_data = item.second;
		output_data->setPrecision(op);
		std::cout << "output_name:" << output_name << std::endl;
	}

	auto executable_network = ie.LoadNetwork(network, "CPU");
	infer_request = executable_network.CreateInferRequest();
	auto input = infer_request.GetBlob(input_name);
	return input;
}
void test() {
	std::string xml = "E:/PycharmProjects/pytorch/crack-segmentation/weights/road.xml";
	std::string bin = "E:/PycharmProjects/pytorch/crack-segmentation/weights/road.bin";

	cv::Mat src = cv::imread("E:/image/road.jpg");
	int im_h = src.rows;
	int im_w = src.cols;

	Core ie;
	InferenceEngine::CNNNetwork network = ie.ReadNetwork(xml, bin);
	InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
	InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
	std::string input_name = "";
	InferenceEngine::Precision ip = Precision::FP32;
	InferenceEngine::Precision op = Precision::FP32;
	InferenceEngine::Layout lo = Layout::NCHW;

	auto input = inferApi(src, xml, bin, true, ip, op, lo);
	size_t c = input->getTensorDesc().getDims()[1];
	size_t h = input->getTensorDesc().getDims()[2];
	size_t w = input->getTensorDesc().getDims()[3];
	size_t image_size = h * w;
	cv::Mat blob_image;
	cv::resize(src, blob_image, cv::Size(w, h));
	blob_image.convertTo(blob_image, CV_32F);
	blob_image = blob_image / 255.0;

	//HWC->NCHW 矩阵转换
	float*data = static_cast<float*>(input->buffer());
	for (size_t row = 0;row < h;row++) {
		for (size_t col = 0;col < w;col++) {
			for (size_t ch = 0;ch < c;ch++) {
				data[image_size*ch + row * w + col] = blob_image.at<cv::Vec3f>(row, col)[ch];
			}
		}
	}
	std::vector<cv::Vec3b> color_tab;
	color_tab.push_back(cv::Vec3b(0, 255, 0));
	color_tab.push_back(cv::Vec3b(255, 0, 0));
	infer_request.Infer();
	auto output = infer_request.GetBlob(output_name);
	const float* detection_out = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
	const SizeVector outputDims = output->getTensorDesc().getDims();

	const int out_c = outputDims[1];
	const int out_h = outputDims[2];
	const int out_w = outputDims[3];
	std::cout << output->getTensorDesc().getPrecision() << std::endl;
	std::cout << out_c << out_h << out_w << std::endl;
	cv::Mat result = cv::Mat::zeros(cv::Size(out_w, out_h), CV_8UC3);
	int step = out_h * out_w;
	for (int row = 0; row < out_h; row++) {
		for (int col = 0; col < out_w; col++) {
			int max_index = 0;
			float max_porb = detection_out[row*out_w + col];
			for (int cn = 1; cn < out_c; cn++) {
				float prob = detection_out[cn*step + row * out_w + col];
				if (prob > max_porb) {
					max_porb = prob;
					max_index = cn;
				}
			}
			result.at<cv::Vec3b>(row, col) = color_tab[max_index];
		}
	}
	cv::resize(result, result, cv::Size(im_w, im_h));
	cv::addWeighted(src, 0.7, result, 0.3, 0, src);

	cv::imshow("input", src);

}
int main(int argc, char** argv) {
	test();
	cv::waitKey(0);
	cv::destroyAllWindows();
}


