#include "Individual.h"



ga::Individual::Individual()
{
}


ga::Individual::~Individual()
{
}

std::shared_ptr<ga::IIndividual> ga::Individual::Mutation()
{
	return std::shared_ptr<IIndividual>();
}

std::shared_ptr<ga::IIndividual> ga::Individual::Crossover(std::shared_ptr<IIndividual> individual)
{
	return std::shared_ptr<IIndividual>();
}

std::pair<int, int> ga::Individual::Spare(std::shared_ptr<IIndividual> individual)
{
	return std::pair<int, int>();
}

std::vector<float> ga::Individual::MakeDecision(std::vector<float>& input)
{
	return std::vector<float>();
}

std::shared_ptr<ga::IIndividual> ga::Individual::Clone()
{
	return std::shared_ptr<IIndividual>();
}
