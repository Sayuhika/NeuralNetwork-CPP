#define FEATURE_DLL_EXPORTS
#include "FeatureExtraction.h"
#include "RadialFunctions.h"

using namespace fe;
using namespace std;
using namespace cv;

namespace fe
{
	class BlobProcessor :public IBlobProcessor
	{
	public:
		FEATURE_DLL_API virtual vector<Mat> DetectBlobs(Mat image) override;

		FEATURE_DLL_API virtual vector<Mat> NormalizeBlobs(vector<Mat> & blobs, int side) override;

		FEATURE_DLL_API virtual string GetType() override;
	};

	class WalshManager :public PolynomialManager
	{
	public:
		FEATURE_DLL_API virtual ComplexMoments Decompose(Mat blob) override;

		FEATURE_DLL_API virtual Mat Recovery(ComplexMoments & decomposition) override;

		FEATURE_DLL_API virtual void InitBasis(int n_max, int diameter) override;

		FEATURE_DLL_API virtual string GetType() override;
	};
}

string GetTestString()
{
	return "You successfuly plug feature extraction library!";
}

/**
* Найти смежные области на изображении. Светлые области представляют собой фон, тёмные области объекты.
* @param image - изображение для поиска смежных областей, должно иметь тип CV_8UC1.
* @return неотмасштабированные смежные области, представляют собой белые символы на чёрном фоне, тип CV_8UC1.
*/
vector<Mat> fe::BlobProcessor::DetectBlobs(Mat image)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Mat> result;
	Mat img;

	threshold(image, img, 127, 255, THRESH_BINARY_INV);
	imshow("threshold", img);
	findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	for (int i = 0; i >= 0; i = hierarchy[i][0])
	{
		Point2f centre;
		float radius;
		minEnclosingCircle(contours[i], centre, radius);
		radius = (int)radius + 1;
		centre.x = radius - (int)centre.x;
		centre.y = radius - (int)centre.y;
		result.push_back(Mat::zeros(2 * radius, 2 * radius, CV_8UC1));
		drawContours(result.back(), contours, i, (255, 255, 255), FILLED, 8, hierarchy, 4, centre);
	}

	return result;
}

/**
* Привести размер смежных областей к единому масштабу.
* @param blobs - смежные области, должны представлять собой белые символы на чёрном фоне, тип CV_8UC1.
* @param normilized_blobs - буфер для записи смежных областей единого размера.
*               Смежные области после записи представляют собой белые буквы на чёрном фоне, тип CV_8UC1.
* @param side - сторона квадрата на котором будет отрисована нормализованная смежная область.
*/
vector<Mat> fe::BlobProcessor::NormalizeBlobs(vector<Mat> & blobs, int side)
{
	vector<Mat> result;
	result.resize(blobs.size());

	for (int i = 0; i < blobs.size(); i++)
	{
		resize(blobs[i], result[i], Size(side, side));
	}

	return result;
}

/**
* Получить описание используемого обработчика смежных областей.
* @return описание, содержит название метода выделения смежных областей и фамилию разработчика.
*/
string fe::BlobProcessor::GetType()
{
	return "Вы получили Type";
}

shared_ptr<IBlobProcessor> fe::CreateBlobProcessor()
{
	return make_shared<BlobProcessor>();
}

shared_ptr<PolynomialManager> fe::CreatePolynomialManager()
{
	return make_shared<WalshManager>();
}

ComplexMoments WalshManager::Decompose(Mat blob)
{
	ComplexMoments result;
	result.re = Mat::zeros(polynomials.size(), polynomials[0].size(), CV_64FC1);
	result.im = Mat::zeros(polynomials.size(), polynomials[0].size(), CV_64FC1);
	result.phase = Mat::zeros(polynomials.size(), polynomials[0].size(), CV_64FC1);
	result.abs = Mat::zeros(polynomials.size(), polynomials[0].size(), CV_64FC1);
	double norma = polynomials.size() * polynomials[0].size();
	Mat blum;
	blob.convertTo(blum, CV_64FC1);

	for (int i = 0; i < polynomials.size(); i++)
	{
		for (int j = 0; j < polynomials[0].size(); j++)
		{
			double temp = blum.dot(polynomials[i][j].first); // скалярное умножение
			double tabs = polynomials[i][j].first.dot(polynomials[i][j].first); // скалярное умножение
			result.re.at<double>(i, j) = (abs(temp) > 1e-20 ? (temp / tabs) / norma : 0.0); // избавляемся от проблем в случае деления на 0

			temp = blum.dot(polynomials[i][j].second); // скалярное умножение
			tabs = polynomials[i][j].second.dot(polynomials[i][j].second); // скалярное умножение
			result.im.at<double>(i, j) = (abs(temp) > 1e-20 ? (temp / tabs) / norma : 0.0); // избавляемся от проблем в случае деления на 0

			result.abs.at<double>(i, j) = sqrt(result.re.at<double>(i, j) * result.re.at<double>(i, j) + result.im.at<double>(i, j) * result.im.at<double>(i, j));
			result.phase.at<double>(i, j) = atan2(result.re.at<double>(i, j), result.im.at<double>(i, j));
		}
	}

	return result;
}

Mat WalshManager::Recovery(ComplexMoments & decomposition) // восстановление
{
	Mat result = Mat::zeros(polynomials[0][0].first.size[0], polynomials[0][0].first.size[1], CV_64FC1);

	for (int i = 0; i < polynomials.size(); i++)
	{
		for (int j = 0; j < polynomials[0].size(); j++)
		{
			result += polynomials[i][j].first * decomposition.re.at<double>(i, j); // действительная часть на массив коэффициентов 
			result += polynomials[i][j].second * decomposition.im.at<double>(i, j); // мнимая часть на массив коэффициентов 
		}
	}
	return result;
}

void WalshManager::InitBasis(int n_max, int diameter)
{
	polynomials.resize(n_max);

	// угловые и радиальные части итерируются по i и j
	for (int i = 0; i < n_max; i++)
	{
		polynomials[i].resize(n_max);

		for (int j = 0; j < n_max; j++)
		{
			polynomials[i][j].first = Mat::zeros(diameter, diameter, CV_64FC1);
			polynomials[i][j].second = Mat::zeros(diameter, diameter, CV_64FC1);

			// координаты пикселей итерируются по x и y
			for (int x = 0; x < diameter; x++)
			{
				for (int y = 0; y < diameter; y++)
				{
					double r = sqrt((x - diameter / 2)*(x - diameter / 2) + (y - diameter / 2)*(y - diameter / 2)) * 2 / diameter;

					if(r > 1)
					{
						polynomials[i][j].first.at<double>(x, y) = 0;
						polynomials[i][j].second.at<double>(x, y) = 0;
					}else
					{
						double walsh = rf::RadialFunctions::Walsh(r, i, n_max);
						double fi = j * atan2(y - diameter/2, x - diameter/2);
						polynomials[i][j].first.at<double>(x, y) = walsh * cos(fi);
						polynomials[i][j].second.at<double>(x, y) = walsh * sin(fi);
					}
				}
			}
		}
	}

	
}

std::string WalshManager::GetType()
{
	return "Вы получили Type";
}