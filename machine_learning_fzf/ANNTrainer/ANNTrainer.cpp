#include <iostream>
#include <ANN.h>
using namespace std;
using namespace ANN;

int main()
{
	vector<vector<float>> inputs, outputs;
	vector<size_t> conf = {2, 4, 1};
	float eps = 0.001;
	int max_iter = 10000;
	float speed = 0.1;

	if (!LoadData("../Resources/LFile.txt", inputs, outputs)) 
	{
		cout << "File LFile.txt not load or not found" << endl;
		return 1;
	} 

	auto NeuralNetwork = CreateNeuralNetwork(conf);
	BackPropTraining(NeuralNetwork, inputs, outputs, max_iter, eps, speed, true);
	NeuralNetwork->GetType();
	NeuralNetwork->Save("../Resources/NeuralNetworkData.txt");
	return 0;
}
