#pragma once

#include <vector>
#include <memory>

#include "ExportMacro.h"
#include "IBlobProcessor.h"
#include "PolynomialManager.h"
#include "ComplexMoments.h"

/** 
 * Функции, которые экспортируются из библиотеки. 
 */
namespace fe {
	/** 
	 * Получить тестовую строку.
	 * @return строка с поздравлениями.
	 */
	FEATURE_DLL_API std::string GetTestString();

	/** 
	 * Создать обработчик смежных областей.
	 * @return обработчик смежных областей.
	 */
	FEATURE_DLL_API std::shared_ptr<IBlobProcessor> CreateBlobProcessor();

	/** 
	 * Создать объект, ответственный за работу с полиномами.
	 * @return объект, ответсвенный за работу с полиномами.
	 */
	FEATURE_DLL_API std::shared_ptr<PolynomialManager> CreatePolynomialManager();
};

