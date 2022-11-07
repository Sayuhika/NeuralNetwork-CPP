#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include "ComplexMoments.h"

namespace fe {
	/**
	 * ѕереопределение типа набора полиномов. ƒл€ краткости.
	 *  омплексные полиномы хран€тс€ в двухмерном массиве. ѕервый индекс - радиальный пор€док, второй индекс - угловой пор€док.
	 * ѕолином представлен типом std::pair - в поле first хранитс€ реальна€ часть полинома, в поле second мнима€.
	 *  ажда€ часть представлена в cv::Mat типа CV_64FC1.
	 */
	typedef std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> OrthoBasis;

	/**
	 *  ласс, отвечающий за взаимодействие с полиномами ~ exp(jm*fi).
	 */
	class PolynomialManager
	{
	protected:

		/** ѕолиномы. */
		OrthoBasis polynomials;

	public:
		/**
		 * –азложить картинку в р€д по полиномам.
		 * @param blob - картинка (смежна€ область), должна быть типа CV_8UC1.
		 * @return decomposition разложение.
		 */
		virtual ComplexMoments Decompose(cv::Mat blob) = 0;

		/**
		 * ¬осстановить картинку из разложени€.
		 * @param decomposition - разложение картинки в р€д.
		 * @return восстановленное изображение, имеет тип CV_64FC1.
		 */
		virtual cv::Mat Recovery(ComplexMoments & decomposition) = 0;

		/**
		 * ѕроинициализировать базис ортогональных полиномов ~ exp(jm*fi).
		 * @param n_max - максимальный радиальный пор€док полиномов.
		 * @param diameter - диаметр окружности, на которой будут сгенерированы полиномы, пиксели.
		 */
		virtual void InitBasis(int n_max, int diameter) = 0;

		/**
		 * ѕолучить описание объекта дл€ работы с полиномами.
		 * @return - описание объекта.
		 */
		virtual std::string GetType() = 0;

		/**
		 * ѕолучить базис ортогональных полиномов. 
		 * @return базис ортогональных полиномов.  аждый полином представлен std::pair<cv::Mat, cv::Mat>.
		 *		   в поле first хранитс€ реальна€ часть полинома, в поле second мнима€.  ажда€ часть имеет тип CV_64FC1.
		 */
		virtual OrthoBasis GetBasis();

		/** ƒеструктор. */
		virtual ~PolynomialManager();
	};
};

