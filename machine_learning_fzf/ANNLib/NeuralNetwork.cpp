#define ANNDLL_EXPORTS
#include <NeuralNetwork.h>
#include <iostream>	
#include <time.h>

ANN::NeuralNetwork::NeuralNetwork(std::vector<size_t> & configuration,
	ANeuralNetwork::ActivationType activation_type,
	float scale) 
{
	this->configuration = configuration;
	this->activation_type = activation_type;
	this->scale = scale;
	if (configuration.size() > 0)
	{
		this->weights.resize(configuration.size() - 1);
		srand(time(0));

		for (int i = 0; i < weights.size(); i++)
		{
			weights[i].resize(configuration[i]);

			for (int j = 0; j < weights[i].size(); j++)
			{
				weights[i][j].resize(configuration[i + 1]);
				for (int k = 0; k < weights[i][j].size(); k++)
				{
					weights[i][j][k] = (((float)rand() / RAND_MAX) - 0.5);
				}
			}
		}
	}
	
}

std::shared_ptr <ANN::ANeuralNetwork> ANN::CreateNeuralNetwork(std::vector<size_t> & configuration,
	ANeuralNetwork::ActivationType activation_type,
	float scale)
{
	return std::make_shared<ANN::NeuralNetwork>(configuration, activation_type, scale);
}

std::string ANN::NeuralNetwork::GetType()
{
	return "You has got this type";
}

std::vector<float> ANN::NeuralNetwork::Predict(std::vector<float>&input)
{
	std::vector<float> result, buffer;
	buffer = input;

	for (int i = 0; i < weights.size(); i++) 
	{
		result.resize(configuration[i+1]);

		for (int j = 0; j < configuration[i+1]; j++) 
		{
			result[j] = 0;

			for (int k = 0; k < configuration[i]; k++) 
			{
				result[j] += buffer[k] * weights[i][k][j];
			}

			result[j] = Activation(result[j]);
		}

		buffer = result;
	}

	return result;
}

float ANN::BackPropTraining(std::shared_ptr<ANN::ANeuralNetwork> ann,
	std::vector<std::vector<float>> & inputs,
	std::vector<std::vector<float>> & outputs,
	int maxIters,
	float eps,
	float speed,
	bool std_dump)
{
	int count = 0;
	float error;

	while (count < maxIters)
	{
		error = 0;

		for (int i = 0; i < inputs.size(); i++)
		{
			error += BackPropTrainingIteration(ann, inputs[i], outputs[i], speed);
		}
		
		error /= inputs.size();
		count++;

		if (count%100==0) std::cout << error << std::endl;

		if (error < eps)
		{
			std::cout << "Done" << std::endl;
			break;
		}
	}

	std::cout << "Done" << std::endl;
	return error;
}

float ANN::BackPropTrainingIteration(std::shared_ptr<ANN::ANeuralNetwork> ann,
	const std::vector<float>& input,
	const std::vector<float>& output,
	float speed)
{
	std::vector<std::vector<float>> sums;
	sums.resize(ann->configuration.size());
    sums[0] = input;
	
	// Расчет взвешенных сум
	for (int i = 0; i < ann->weights.size(); i++)
	{
		sums[i + 1].resize(ann->configuration[i+1]);

		for (int j = 0; j < sums[i + 1].size(); j++)
		{
			sums[i+1][j] = 0.0f;
			
			for (int k = 0; k < ann->configuration[i]; k++)
			{
				sums[i + 1][j] += sums[i][k] * ann->weights[i][k][j];
			}

			sums[i + 1][j] = ann->Activation(sums[i + 1][j]);
		}

	}

	std::vector<float> sigmas, bufsigmas;
	float error = 0.0f;

	std::vector<std::vector<std::vector<float>>> deltas;
	deltas.resize(ann->weights.size());

	// Подготовка массива к использованию
	for (int i = 0; i < deltas.size(); i++)
	{
		deltas[i].resize(ann->weights[i].size());

		for (int j = 0; j < deltas[i].size(); j++)
		{
			deltas[i][j].resize(ann->weights[i][j].size());
		}
	}

	int layer_idx = ann->configuration.size() - 2;
	// Расчет изменения весов для выходного слоя
	for (int to = 0; to < ann->configuration[layer_idx + 1]; to++) 
	{
		sigmas.push_back(sums[layer_idx + 1][to] - output[to]);

		for (int from = 0; from < ann->configuration[layer_idx]; from++)
		{
			deltas[layer_idx][from][to] = -speed * sigmas[to] * sums[layer_idx][from] * ann->ActivationDerivative(sums[layer_idx + 1][to]);
		}
	}

	// Расчет изменения весов для скрытых слоев
	while (layer_idx > 0) {
		bufsigmas.clear();

		for (int from = 0; from < ann->configuration[layer_idx]; from++)
		{
			float sigma = 0.0f;

			for (int to = 0; to < ann->configuration[layer_idx + 1]; to++)
			{
				sigma += ann->weights[layer_idx][from][to] * sigmas[to];
			}

			bufsigmas.push_back(sigma);
		}

		layer_idx--;

		for (int to = 0; to < ann->configuration[layer_idx + 1]; to++)
		{
			for (int from = 0; from < ann->configuration[layer_idx]; from++)
			{
				deltas[layer_idx][from][to] = -speed * bufsigmas[to] * sums[layer_idx][from] * ann->ActivationDerivative(sums[layer_idx + 1][to]);
			}
		}

		sigmas = bufsigmas;
	}


	// Изменяем веса
	for (int i = 0; i < ann->weights.size(); i++)
	{
		for (int j = 0; j < ann->configuration[i + 1]; j++)
		{			
			for (int k = 0; k < ann->configuration[i]; k++)
			{
				ann->weights[i][k][j] += deltas[i][k][j];
			}
		}
	}

	sums[0] = input;

	// Расчет взвешенных сум
	for (int i = 0; i < ann->weights.size(); i++)
	{
		sums[i + 1].resize(ann->configuration[i + 1]);

		for (int j = 0; j < sums[i + 1].size(); j++)
		{
			sums[i + 1][j] = 0;

			for (int k = 0; k < ann->configuration[i]; k++)
			{
				sums[i + 1][j] += sums[i][k] * ann->weights[i][k][j];
			}

			sums[i + 1][j] = ann->Activation(sums[i + 1][j]);
		}

	}

	// Расчет квадратичной ошибки
	for (int i = 0; i < output.size();i++)
	{
		error += (output[i] - sums.back()[i]) * (output[i] - sums.back()[i]);
	}
	error /= output.size();

	return error;
}