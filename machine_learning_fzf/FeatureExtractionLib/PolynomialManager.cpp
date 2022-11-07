#include "PolynomialManager.h"

using namespace fe;

OrthoBasis PolynomialManager::GetBasis()
{
	return polynomials;
}

PolynomialManager::~PolynomialManager() = default;
