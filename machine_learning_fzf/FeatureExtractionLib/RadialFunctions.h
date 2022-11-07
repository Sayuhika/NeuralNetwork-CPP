#pragma once
#include <opencv2/opencv.hpp>

namespace rf {
	/** Класс для вычисления радиальных функций. */
	class RadialFunctions {
	protected:
		static int walsh_n_max;
		static cv::Mat walsh_matrix;
		static cv::Mat WalshGenerator(cv::Mat walsh, int n);
		static double Legendre(double x, int n);
	public:
		/**
		 * Радиальная часть полинома Цернике порядков n,m.
		 * @param rad - радиус на котором вычисляется значение, 0 <= rad <= 1.
		 * @param n - радиальный порядок полинома, n > 0.
		 * @param m - угловой порядок полинома, m > 0, n-m должно быть четное.
		 * @return значение полинома в точке rad.
		 */
		static double Zernike(double rad, int n, int m);

		/**
		 * Функция Уолша с номером n в базисе n_max.
		 * @param rad - радиус на котором вычисляется значение, 0 <= rad <= 1.
		 * @param n - номер функции.
		 * @param n_max - количество полиномов в базисе, должно быть степенью двойки.
		 * @return значение полинома в точке rad.
		 */
		static double Walsh(double rad, int n, int n_max);

		/**
		 * Вычислить значение сдвинутого полинома Лежанжра.
		 * @param rad - радиус на котором вычисляется значение, 0 <= rad <= 1.
		 * @param n - порядок полинома Лагера.
		 * @return значение полинома в точке rad.
		 */
		static double ShiftedLegendre(double rad, int n);

		/**
		 * Вычислить значение сдвинутого полинома Чебышева.
		 * @param rad - радиус на котором вычисляется значение, 0 <= rad <= 1.
		 * @param n - порядок полинома Чебышева.
		 * @return значение полинома в точке rad.
		 */
		static double ShiftedChebyshev(double rad, int n);

		/**
		 *  Деструктор.
		 */
		virtual ~RadialFunctions();
	};
}