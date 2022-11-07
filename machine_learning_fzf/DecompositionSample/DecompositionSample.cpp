#include <iostream>
#include <FeatureExtraction.h>
#include "Visualisation.h"

using namespace std;
using namespace cv;
using namespace fe;

int main() 
{
	Mat m = cv::imread("../Resources/numbers.png", CV_LOAD_IMAGE_GRAYSCALE);
	//imshow("Picture is here", m);
	//waitKey(0);

	int diameter, n_max;
	cout << "Diameter: ";
	cin >> diameter;
	cout << "Polynomial power: ";
	cin >> n_max;
	
	auto PM = CreatePolynomialManager();
	PM->InitBasis(n_max, diameter);
	ShowPolynomials("Polynomial", PM->GetBasis());
	waitKey(0);

	auto BP = CreateBlobProcessor();
	vector<Mat> dec =  BP->DetectBlobs(m);
	dec = BP->NormalizeBlobs(dec, diameter);

	for(int i = 0; i < dec.size(); i++)
	{
		ComplexMoments C = PM->Decompose(dec[i]);
		ShowBlobDecomposition(to_string(i),  dec[i], PM->Recovery(C));
	}
	waitKey(0);

	return 0;
}