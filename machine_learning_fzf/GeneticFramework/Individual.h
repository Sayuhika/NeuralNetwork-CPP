#pragma once
#include "IIndividual.h"

namespace ga
{
	class Individual : public IIndividual
	{
	public:
		Individual();
		~Individual();

		/**
		 * ��������� ������� �����.
		 * @return ������������ �����.
		 */
		std::shared_ptr<IIndividual> Mutation();

		/**
		 * ��������� ����������� ������� ����� � ������ ������.
		 * @param individual - ����� � ������� ����� ��������� �����������.
		 * @return �������� ����� ����� �����������.
		 */
		std::shared_ptr<IIndividual> Crossover(std::shared_ptr<IIndividual> individual);

		/**
		 * �������� ������������ ����� ������� � ������ ������.
		 * @param individual - ������ �����.
		 * @return ���� ����. ������ �������� - ���������� ����� ��������� ������� ������.
		 *					  ������ �������� - ���������� �����, ��������� ������ ������.
		 */
		std::pair<int, int> Spare(std::shared_ptr<IIndividual> individual);

		/**
		 * ������� �������.
		 * � �������� ������������ ����� ���������� ��������� �������,
		 * �� ����� ������� ������� �������� ������������.
		 * @param input - ������� ������
		 * @return �������� ������.
		 */
		std::vector<float> MakeDecision(std::vector<float> & input);

		/**
		 * ����������� ������� �����.
		 * @return ����� ������� �����.
		 */
		std::shared_ptr<IIndividual> Clone();
	};
};


