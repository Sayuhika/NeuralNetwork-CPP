#pragma once
#include <FeatureExtraction.h>
#include <opencv2/ml.hpp>

/**Объект для проведения распознавания моментов.*/
class MomentsRecognizer
{
protected:
	/**Массив распознанных значений. Ключ - номер активированного выходного нейрона. 
		Значение - значение цифры.*/
	std::vector<std::string> values;

	/**Нейронная сеть.*/
	cv::Ptr<cv::ml::ANN_MLP> pAnn;

	/*****************************************************************************/
	/**************Эту часть необходимо реализовать самостоятельно.***************/
	/*****************************************************************************/

	/**Преобразовать моменты изображения ко входу нейронной сети.
	 * @param moments - моменты изображения.
	 * @return - Значение входа нейронной сети.
	 */
	virtual cv::Mat MomentsToInput(fe::ComplexMoments& moments) 
	{
		cv::Mat result;
		result = cv::Mat::zeros(1, moments.abs.size[0] * moments.abs.size[1], CV_32F);
		
		for(int i = 0; i < moments.abs.size[0]; i++)
		{
			for (int j = 0; j < moments.abs.size[1]; j++)
			{
				result.at<float>(i * moments.abs.size[1] + j) = (float) moments.abs.at<double>(i, j);
			}
		}

		return result;
	};
	/**Преобразовать выход сети в значение символа.
	 * @param output - выход нейроной сети.
	 * @return - распознанное значение символа.
	 */
	virtual std::string OutputToValue(cv::Mat output) 
	{
		int m = 0;
		float max = output.at<float>(0, m);
		for (int i = 1; i < 9; i++)
		{
			float temp = output.at<float>(0, i);
			if(max < temp)
			{
				max = temp;
				m = i;
			}
		}
		switch (m)
		{
		case 0:
			return "0";
		case 1:
			return "1";
		case 2:
			return "2";
		case 3:
			return "3";
		case 4:
			return "4";
		case 5:
			return "5";
		case 6:
			return "6";
		case 7:
			return "7";
		case 8:
			return "8";
		default:
			return "произошло что-то плохое и непредвиденное";
		}
	};
	/********************************************************************************/
public:
	/**Конструктор по умолчанию*/
	MomentsRecognizer();
	
	/**Деструктор по умолчанию.*/
	virtual ~MomentsRecognizer();

	/**Сохранить объект для распознавания в файл.
	* @param filename - имя файла для сохранения
	* @return true - файл успшно сохранен, false - файл не сохранен.
	*/
	virtual bool Save(std::string filename);

	/**Прочитать объект для распознавания из файла.
	* @param filename - имя файла с объектом для распознавания.
	* @return true - сеть успешно считана, false - сеть не считана.
	*/
	virtual bool Read(std::string filename);

	/**Выполнить простейшую оценку точности распознавания сети.
	* @param moments - набор тестовых данных. Это ассоциативный массив.
	*					Ключ - значение символа (например "5")
	*                  Значение - набор разложений разчных вариаций этого символа.
	* @return - процент верного распознавания на тестовой выборке.
	*/
	virtual double PrecisionTest(std::map<std::string, std::vector<fe::ComplexMoments>> moments);

	/**Распознать символ по моментам
	 * @param moments - моменты по которым проводится распознавание.
	 * @return - распознанное значение символа.
	 */
	virtual std::string Recognize(fe::ComplexMoments & moments);

	/*********************************************************************************/
	/***********************Этот метод необходимо релизовать самостоятельно.**********/
	/*********************************************************************************/
	/**Обучить нейронную сеть.
	 * @param moments - набор обучающих данных. Это ассоциативный массив.
	 *					Ключ - значение символа (например "5")
	 *                  Значение - набор разложений разчных вариаций этого символа.
	 * @param layers - конфигурация скрытых слоев будущего персептрона.
	 *					Ключ - номер СКРЫТОГО слоя.
	 *					Значение - количество нейронов в слое.
	 * @param max_iters - максимальное количество итераций при обучении.
	 * @param eps - требуемая точность распознавания на обучающей выборке.
	 * @param speed - скорость обучения = коэффициент перед корректировкой к весу.
	 * @return true - сеть успешно обучена, false - сеть не обучена.
	 */
	virtual bool Train(
		std::map<std::string, std::vector<fe::ComplexMoments>> moments,
		std::vector<int> layers,
		int max_iters = 200000,
		float eps = 0.01,
		float speed = 0.1)
	{
		pAnn = cv::ml::ANN_MLP::create();

		// Конфигурация алгоритма обучения.
		pAnn->setLayerSizes(layers);
		pAnn->setBackpropMomentumScale(speed);
		pAnn->setBackpropWeightScale(0.1);
		pAnn->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1.0, 1.0);

		// Критерий остановки обучения. 
		cv::TermCriteria term_criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, max_iters, eps);
		pAnn->setTermCriteria(term_criteria);
		//pAnn->setTrainMethod(cv::ml::ANN_MLP::RPROP, 0.001);
		pAnn->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP);

		// Считаем число примеров
		int n = 0;
		for (int i = 0; i < 9; i++)
		{
			n += moments[std::to_string(i)].size();
		}

		cv::Mat input = cv::Mat::zeros(n, layers[0], CV_32F);
		cv::Mat output = cv::Mat::zeros(n, 9, CV_32F);

		n = 0;
		for (int i = 0; i < 9; i++)
		{
			for (int j = 0; j < moments[std::to_string(i)].size(); j++)
			{
				cv::Mat temp = MomentsToInput(moments[std::to_string(i)][j]);

				for (int k = 0; k < layers[0]; k++)
				{
					input.at<float>(n + j, k) = temp.at<float>(k);
				}

				for (int k = 0; k < 9; k++)
				{
					output.at<float>(n + j, k) = (k == i ? 1 : -1);
				}
			}
			n += moments[std::to_string(i)].size();
		}

		//auto trainData = cv::ml::TrainData::create(input, cv::ml::ROW_SAMPLE, output);
		//pAnn->train(trainData, cv::ml::ANN_MLP::NO_INPUT_SCALE + cv::ml::ANN_MLP::NO_OUTPUT_SCALE);
		//return pAnn->isTrained();

		return pAnn->train(input, cv::ml::ROW_SAMPLE, output);
	};
	/************************************************************************************/
};

