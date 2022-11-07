#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iomanip>
#include "FeatureExtraction.h"
#include "MomentsHelper.h"
#include "MomentsRecognizer.h"

using namespace cv;
using namespace std;
using namespace fe;

MomentsRecognizer MR;
MomentsHelper MH;
auto BP = CreateBlobProcessor();
auto PM = CreatePolynomialManager();
map<string, vector<ComplexMoments>> samples;
vector<int> layers;

void generateData()
{
	cout << "===Generate data!===" << endl;
	PM->InitBasis(16, 128);

	// Сортировка картинок для обучения по папкам
	string path_ld = "..\\Resources\\labeled_data";
	string path_gd = "..\\Resources\\ground_data";
	string path_td = "..\\Resources\\test_data";

	if (!MH.DistributeData(path_ld, path_gd, path_td, 50))
	{
		cout << "error" << endl;
	}

	// Подготовка и инициализация моментов перед обучением
	MH.GenerateMoments("..\\Resources\\ground_data", BP, PM, samples);
	MH.SaveMoments("ComplexMomentsGD", samples);
	MH.GenerateMoments("..\\Resources\\test_data", BP, PM, samples);
	MH.SaveMoments("ComplexMomentsTD", samples);

	cout << "===Generate complete!===" << endl;
}

void trainNetwork()
{
	cout << "===Train network!===" << endl;

	MH.ReadMoments("ComplexMomentsGD", samples);
	layers = { 256, 64, 64, 9 };

	if (MR.Train(samples, layers))
	{
		MR.Save("ANN");
		cout << "===Train network complete!===" << endl;
	};
}

void precisionTest()
{
	cout << "===Precision test!===" << endl;
	MR.Read("ANN");

	map<string, vector<ComplexMoments>> moments;
	MH.ReadMoments("ComplexMomentsTD", moments);

	double precision = MR.PrecisionTest(moments);
	cout << setprecision(5) << "Precision: " << precision << endl;
	cout << "===Precision test complete!===" << endl;
}

void recognizeImage()
{
	cout << "===Recognize single image!===" << endl;
	PM->InitBasis(16, 128);
	MR.Read("ANN");
	ComplexMoments result;

	Mat image = imread("..\\Resources\\numbers.png", cv::IMREAD_GRAYSCALE);

	if (image.empty()) {
		throw "Empty image";
	}

	threshold(image, image, 127, 255, CV_THRESH_BINARY);
	vector<Mat> blobs, nblobs;
	blobs = BP->DetectBlobs(image);

	nblobs = BP->NormalizeBlobs(blobs, (PM->GetBasis()[0][0].first.cols));
	for (int i = 0; i < nblobs.size(); i++)
	{
		string value = MR.Recognize(PM->Decompose(nblobs[i]));
		cv::imshow(to_string (i) + ") " + value, nblobs[i]);
		//cout << "Result: " << value << endl;
	}
	cv::waitKey();
	cout << "===Recognize single image complete!===" << endl;
}

int main(int argc, char** argv)
{
	string key;
	do 
	{
		cout << "===Enter next values to do something:===" << endl;
		cout << "  '1' - to generate data." << endl;
		cout << "  '2' - to train network." << endl;
		cout << "  '3' - to check recognizing precision." << endl;
		cout << "  '4' - to recognize single image." << endl;
		cout << "  'exit' - to close the application." << endl;
		cin >> key;
		cout << endl;
		if (key == "1") {
			generateData();
		}
		else if (key == "2") {
			trainNetwork();
		}
		else if (key == "3") {
			precisionTest();
		}
		else if (key == "4") {
			recognizeImage();
		}
		cout << endl;
	} while (key != "exit");
	return 0;
}

