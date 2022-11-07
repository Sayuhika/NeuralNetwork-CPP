#pragma once
#include <ANN.h>

namespace ANN 
{
	class NeuralNetwork:public ANeuralNetwork
	{
	public:
		ANNDLL_API NeuralNetwork(std::vector<size_t> & configuration = std::vector<size_t>(),
			ANeuralNetwork::ActivationType activation_type = ANeuralNetwork::POSITIVE_SYGMOID,
			float scale = 1.0);
		ANNDLL_API virtual std::string GetType() override;
		ANNDLL_API virtual std::vector<float> Predict(std::vector<float> &input) override;
	};
}