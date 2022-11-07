#pragma once
#include "Epoch.h"

namespace ga
{
	// Генетический алгоритм.
	class GeneticAlgorithm
	{
	public:
		// Конструктор.
		GeneticAlgorithm();
		// Деструктор.
		virtual ~GeneticAlgorithm();

		// Текущая эпоха.
		pEpoch epoch;
		
		/**Провести отбор в этой эпохе. Проценты в параметрах указываются от 0 до 100.
		 * @param unchange_perc - процент популяции, который проходит в следующую эпоху без изменений.
		 * @param mutation_perc - процент новой популяции, который будут составлять мутировавшие особи.
		 * @param crossover_perc - процент новой популяции, который будут составлять особи после скрещивания.
		 * @return - последующая эпоха.
		 */
		pEpoch Selection(double unchange_perc, double mutation_perc, double crossover_perc);
	};
}



