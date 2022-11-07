#pragma once
#include "IIndividual.h"
namespace ga
{
	// Эпоха эволюции.
	class Epoch
	{
	public:
		// Конструктор.
		Epoch();
		// Деструктор.
		virtual ~Epoch();

		/* Популяция. Представляет собой вектор пар.
		 * Первое знаение в каждой паре - очки индивида, второе - указатель на индивид.
		 */
		std::vector<std::pair<int, pIIndividual>> population;

		/* Проведение битвы каждого с каждым.
		 * При вызове всем индивидам должны начислиться набранные за победы и проигрыши очки.
		 */
		void EpochBattle();
	};

	// Переопределение типа "умный указатель на эпоху".
	typedef std::shared_ptr<Epoch> pEpoch;
}