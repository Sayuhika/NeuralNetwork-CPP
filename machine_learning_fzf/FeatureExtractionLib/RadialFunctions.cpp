 #include "RadialFunctions.h"
#include <math.h>
#include <opencv2/opencv.hpp>


#define PI 3.14159265359
#define SQRT_PI 1.77245385091
#define SQRT_2 1.41421356237

/* @brief взять факториал
* @param n		- число фактрориал которго необходимо вычислить
* @return		- значение факториала
*/
double factorial(double n)
{
	if (n < 2) return 1.;
	int res = 1;
	for (int _n = 2; _n < n + 1; _n++) {
		res *= _n;
	}
	return res;
}

double rf::RadialFunctions::Zernike(double rad, int n, int m)
{
	int absm = abs(m);
	//проверим корректность принятых данных
	if (n < 0 || absm > n || rad > 1 || rad < 0) return 0.;
	//не нулю равны только те полиномы, m-n у которых четное
	if ((n - absm) % 2 != 0) return 0.;

	if (n == 0) return SQRT_2;

	double rad_res = 0;
	//для ускорения вычислений не будем каждый раз расчитывать значение факториала,
	//а будем расчитывать его из значение на предидущем шаге
	double kfact = 1;
	double min1powk = 1;
	double nminkfact = factorial(n);
	double nminmdiv2minkfact = factorial((n - absm) / 2);
	double nplusmdiv2minkfact = factorial((n + absm) / 2);
	//суммирование полинома
	for (int k = 0; k < (n - absm) / 2 + 1; k++)
	{
		rad_res += (double)min1powk / kfact * nminkfact / nplusmdiv2minkfact /nminmdiv2minkfact * pow(rad, n - 2 * k);

		//вычисление новых значений факториала, на всех итерациях кроме последней
		if (k == (n - absm) / 2) break;

		nplusmdiv2minkfact /= int((n + absm)*0.5 - k);
		nminmdiv2minkfact /= int((n - absm)*0.5 - k);
		nminkfact /= (n - k);
		kfact *= k + 1;
		min1powk *= -1;
	}
	return rad_res * sqrt(2.*n+2.);
}

cv::Mat rf::RadialFunctions::WalshGenerator(cv::Mat walsh, int n)
{
	cv::Mat new_walsh = cv::Mat::zeros(walsh.rows * 2, walsh.cols * 2, CV_8SC1);
	cv::Mat roi;
	roi = new_walsh(cv::Rect(0, 0, walsh.cols, walsh.rows));
	walsh.copyTo(roi);
	roi = new_walsh(cv::Rect(walsh.cols, 0, walsh.cols, walsh.rows));
	walsh.copyTo(roi);
	roi = new_walsh(cv::Rect(0, walsh.rows, walsh.cols, walsh.rows));
	walsh.copyTo(roi);
	roi = new_walsh(cv::Rect(walsh.cols, walsh.rows, walsh.cols, walsh.rows));
	char * data = walsh.ptr<char>();
	for (int i = 0; i < walsh.cols * walsh.rows; i++) {
		data[i] *= -1;
	}
	walsh.copyTo(roi);
	return new_walsh.rows > n ? new_walsh : WalshGenerator(new_walsh, n);
}

double rf::RadialFunctions::ShiftedLegendre(double rad, int n)
{
	if (rad < 0. || rad > 1.0) return 0;
	return sqrt((2. * n + 1.) / 2.) * Legendre((rad - 0.5) * 2, n);
}

double rf::RadialFunctions::ShiftedChebyshev(double rad, int n)
{
	if (rad < 0. || rad > 1.0) return 0;
	double eps = 0.02;
	if (abs(rad - 1.0) < eps) rad = 1 - eps;
	if (rad < eps) rad = eps;
	double x = (rad - 0.5) * 2;
	double k = n == 1 ? SQRT_2 : 1.;
	return n == 0 ? pow(1 - x*x, 0.25) : cos(n * acos(x)) * pow(1 - x*x, 0.25) * SQRT_2 * k;
}

double rf::RadialFunctions::Walsh(double rad, int n, int n_max)
{
	if (rad < 0. || rad > 1.0) return 0;
	cv::Mat w;
	if (walsh_n_max != n_max) {
		w = WalshGenerator(cv::Mat(1, 1, CV_8SC1, cv::Scalar::all(1)), n_max);
		walsh_matrix = w;
		walsh_n_max = n_max;
	}
	else {
		w = walsh_matrix;
	}
	char a = (w.ptr<char>(n)[(int)(rad * w.cols)]);
	return a > 0 ? 1. : -1.;
	return 0.;
}

double rf::RadialFunctions::Legendre(double x, int n)
{
	if (n == 0) return 1;
	if (n == 1) return x; 
	double l2 = 1., l1 = x, l0;
	for (int _n = 2; _n < n + 1; _n++) {
		l0 = (2. * _n + 1) / (_n + 1.) * x * l1 - _n / (_n + 1.) * l2;
		l2 = l1;
		l1 = l0;
	}
	return l0;
}

rf::RadialFunctions::~RadialFunctions() = default;

cv::Mat rf::RadialFunctions::walsh_matrix;
int rf::RadialFunctions::walsh_n_max = 0;
