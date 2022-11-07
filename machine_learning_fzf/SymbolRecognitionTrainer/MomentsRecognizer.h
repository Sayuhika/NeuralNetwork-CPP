#pragma once
#include <FeatureExtraction.h>
#include <opencv2/ml.hpp>

/**������ ��� ���������� ������������� ��������.*/
class MomentsRecognizer
{
protected:
	/**������ ������������ ��������. ���� - ����� ��������������� ��������� �������. 
		�������� - �������� �����.*/
	std::vector<std::string> values;

	/**��������� ����.*/
	cv::Ptr<cv::ml::ANN_MLP> pAnn;

	/*****************************************************************************/
	/**************��� ����� ���������� ����������� ��������������.***************/
	/*****************************************************************************/

	/**������������� ������� ����������� �� ����� ��������� ����.
	 * @param moments - ������� �����������.
	 * @return - �������� ����� ��������� ����.
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
	/**������������� ����� ���� � �������� �������.
	 * @param output - ����� �������� ����.
	 * @return - ������������ �������� �������.
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
			return "��������� ���-�� ������ � ��������������";
		}
	};
	/********************************************************************************/
public:
	/**����������� �� ���������*/
	MomentsRecognizer();
	
	/**���������� �� ���������.*/
	virtual ~MomentsRecognizer();

	/**��������� ������ ��� ������������� � ����.
	* @param filename - ��� ����� ��� ����������
	* @return true - ���� ������ ��������, false - ���� �� ��������.
	*/
	virtual bool Save(std::string filename);

	/**��������� ������ ��� ������������� �� �����.
	* @param filename - ��� ����� � �������� ��� �������������.
	* @return true - ���� ������� �������, false - ���� �� �������.
	*/
	virtual bool Read(std::string filename);

	/**��������� ���������� ������ �������� ������������� ����.
	* @param moments - ����� �������� ������. ��� ������������� ������.
	*					���� - �������� ������� (�������� "5")
	*                  �������� - ����� ���������� ������� �������� ����� �������.
	* @return - ������� ������� ������������� �� �������� �������.
	*/
	virtual double PrecisionTest(std::map<std::string, std::vector<fe::ComplexMoments>> moments);

	/**���������� ������ �� ��������
	 * @param moments - ������� �� ������� ���������� �������������.
	 * @return - ������������ �������� �������.
	 */
	virtual std::string Recognize(fe::ComplexMoments & moments);

	/*********************************************************************************/
	/***********************���� ����� ���������� ���������� ��������������.**********/
	/*********************************************************************************/
	/**������� ��������� ����.
	 * @param moments - ����� ��������� ������. ��� ������������� ������.
	 *					���� - �������� ������� (�������� "5")
	 *                  �������� - ����� ���������� ������� �������� ����� �������.
	 * @param layers - ������������ ������� ����� �������� �����������.
	 *					���� - ����� �������� ����.
	 *					�������� - ���������� �������� � ����.
	 * @param max_iters - ������������ ���������� �������� ��� ��������.
	 * @param eps - ��������� �������� ������������� �� ��������� �������.
	 * @param speed - �������� �������� = ����������� ����� �������������� � ����.
	 * @return true - ���� ������� �������, false - ���� �� �������.
	 */
	virtual bool Train(
		std::map<std::string, std::vector<fe::ComplexMoments>> moments,
		std::vector<int> layers,
		int max_iters = 200000,
		float eps = 0.01,
		float speed = 0.1)
	{
		pAnn = cv::ml::ANN_MLP::create();

		// ������������ ��������� ��������.
		pAnn->setLayerSizes(layers);
		pAnn->setBackpropMomentumScale(speed);
		pAnn->setBackpropWeightScale(0.1);
		pAnn->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1.0, 1.0);

		// �������� ��������� ��������. 
		cv::TermCriteria term_criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, max_iters, eps);
		pAnn->setTermCriteria(term_criteria);
		//pAnn->setTrainMethod(cv::ml::ANN_MLP::RPROP, 0.001);
		pAnn->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP);

		// ������� ����� ��������
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

