#include <iostream>
#include <ANN.h>
#include <vector>
using namespace std;
using namespace ANN;

int main()
{
	vector<vector<float>> inputs, outputs, results;

	shared_ptr<ANeuralNetwork> ANN = CreateNeuralNetwork();
	if (!ANN->Load("../Resources/NeuralNetworkData.txt"))
	{
		cout << "File NeuralNetworkData.txt not load or not found" << endl;
		return 1;
	}
	cout << ANN->GetType() << endl;
	if (!LoadData("../Resources/LFile.txt", inputs, outputs))
	{
		cout << "File LFile.txt not load or not found" << endl;
		return 1;
	}
	float error;
	results.resize(inputs.size());

	for(int i = 0; i < inputs.size(); i++)
	{
		results[i] = ANN->Predict(inputs[i]);
		error = 0;

		for (int j = 0; j < results[i].size(); j++)
		{
			error += (results[i][j] - outputs[i][j]) * (results[i][j] - outputs[i][j]);
			cout << results[i][j] << endl;
		}	

		error /= results[i].size();
		cout << "ERROR result: " << error << endl << endl;
	}
	system("pause");
	return 0;
}