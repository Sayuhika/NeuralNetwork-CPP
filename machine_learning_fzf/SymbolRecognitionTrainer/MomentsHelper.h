#pragma once
#include "FeatureExtraction.h"

/************************************************************************/
/* Описание формата папки с примерами                                   */
/************************************************************************/
/*
	Папка с примерами имеет следующую структуру:
	В папке лежит набор подпапок с примерами. В каждой подпаке лежит множество реализаций
	одного и того же примера в формате *.png и файл "value.txt" с текстовым значением примера.
*/

/**Помощник для работы с моментами.*/
class MomentsHelper
{
public:
	/**Конструктор по умолчанию.*/
	MomentsHelper();
	/**Деструктор по умолчанию.*/
	virtual ~MomentsHelper();

	/**Сгенерировать моменты по примерам.
	 * @param path - путь до папки с примерами.
	 * @param blob_processor - обработчик смежных областей, который будет использоваться при генерации моментов.
	 * @param poly_manager - мэнэджер полиномов, который будет использоваться при генерации примеров.
	 * @param res - буфер для записи сгенерированных моментов. Это ассоциативный массив.
	 *	Ключ - значение символа (например "5")
	 *  Значение - набор разложений разчных вариаций этого символа.
	 * @return true - моменты успешно сгенерированы, false - моменты не сгенерированы.
	 */
	static bool GenerateMoments(
		std::string path,
		std::shared_ptr<fe::IBlobProcessor> blob_processor,
		std::shared_ptr<fe::PolynomialManager> poly_manager,
		std::map< std::string, std::vector<fe::ComplexMoments> > & res
		);

	/**Получить пути до всех подпапок с различными реализациями примеров, 
	 * расположенные в папке с примерами.
	 * @param base_path - путь до папки с примерами.
	 * @param paths - буфер для записи найденных путей.
	 * @return true - пути успешно найдены, false - пути не найдены.
	 */
	static bool GetSamplePaths(
		std::string base_path,
		std::vector<std::string> & paths
		);

	/**Обработать одно изображение с примером. На изображении должен присутствовать только один объект.
	 * @param image_path - путь и имя файла с прпимером.
	 * @param blob_processor - обработчик смежных областей применяемый прии обработке.
	 * @param poly_manager - мэнеджер полиномов, применяемый для разложения.
	 * @param res - буфер для записи разложения.
	 */
	static void ProcessOneImage(
		std::string image_path,
		std::shared_ptr<fe::IBlobProcessor> blob_processor,
		std::shared_ptr<fe::PolynomialManager> poly_manager,
		fe::ComplexMoments & res
		);

	/**Разложить размеченные данные по папкам с обучающими и тестовыми данными.
	 * ДЛЯ РАБОТЫ ФУНКЦИИ НЕОБХОДИМО, ЧТОБЫ ДИРЕКТОРИИ ДЛЯ ЗАПИСИ СУЩЕСТВОВАЛИ И БЫЛИ ПУСТЫ.
	 * @param labeled_data_path - путь до размеченных данных.
	 * @param ground_data_path - путь для записи обучающих данных.
	 * @param test_data_path - путь для записи тетстовых данных.
	 * @param percent - процент размеченных данных, который будет скопирован в обучающие данные.
	 *	Остальные данные будут скопированы в папку для тестовых данных.
	 * @return true - данные скопированы успешно, false - данные не скопированы.
	 */
	static bool DistributeData(
		std::string labeled_data_path,
		std::string ground_data_path,
		std::string test_data_path,
		double percent
		);

	/**Сохранить моменты.
	 * @param filename - имя файла для сохранения.
	 * @param moments - моменты для сохранения. Это ассоциативный массив.
	 *	Ключ - значение символа (например "5")
	 *  Значение - набор разложений разчных вариаций этого символа.
	 * @return - true - моменты сохранены, false - моменты не сохранены.
	 */
	static bool SaveMoments(
		std::string filename,
		std::map< std::string, std::vector<fe::ComplexMoments> > & moments
		);

	/**Считать моменты из файла.
	 * @param filename - имя файла для считывания.
	 * @param moment - буфер для записи считанных моментов. Это ассоциативный массив.
	 *	Ключ - значение символа (например "5")
	 *  Значение - набор разложений разчных вариаций этого символа.
	 * @return true - моменты считаны успешно, моменты не считаны.
	 */
	static bool ReadMoments(
		std::string filename,
		std::map< std::string, std::vector<fe::ComplexMoments> > & moments
		);
};

