#pragma once
#include <opencv2/opencv.hpp>
#include "ExportMacro.h"

namespace fe {
	/** 
	 * Структура, описывающая комплексные моменты. 
	 */
	class ComplexMoments
	{
	public:
		/** Реальные части. */
		cv::Mat re;

		/** Мнимые части. */
		cv::Mat im;

		/** Модули. */
		cv::Mat abs;

		/** Фазы. */
		cv::Mat phase;

		/** Конструтор по умолчанию. */
		FEATURE_DLL_API ComplexMoments();

		/** Деструктор. */
		FEATURE_DLL_API virtual ~ComplexMoments();
	};
}

