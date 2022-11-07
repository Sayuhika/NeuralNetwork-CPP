#pragma once
#include <opencv2/opencv.hpp>

/*—реднее значение уровн€ €ркости дл€ полутоновых байтовых изображений*/
#define MIDDLE_LEVEL 127
/*ћаксимальное значение уровн€ €ркости дл€ полутоновых байтовых изображений*/
#define MAX_LEVEL 255

/**¬ывести на экран изображение типа CV_64FC1. ƒл€ отрисовки надо вызвать waitKey
 * @param wnd_name - название окна дл€ отображени€.
 * @param polynomial - изображение типа CV_64FC1 дл€ отображени€
 */
void Show64FC1Mat(std::string wnd_name, cv::Mat mat64fc1);

/**¬ывести на экран смежную область и ее воссстановление из разложени€ по ортогональному базису.
 * ƒл€ отрисовки надо вызвать waitKey
 * @param wnd_name - название окна дл€ отображени€.
 * @param blob - смежна€ область, которую раскладывали по ортогональному базису. ƒолжна иметь тип CV_8UC1.
 * @param decomposition - картинка восстановленна€ из разложени€ по ортогональному базису.
 *						  ƒолжна иметь CV_64FC1.
 */
void ShowBlobDecomposition(std::string wnd_name, cv::Mat blob, cv::Mat decomposition);


/**¬ывести на экран набор полиномов. ƒл€ отрисовки надо вызвать waitKey.
 * @param wnd_name - название дл€ окна отображени€.
 * @param polynomials - набор комплексных полиномов дл€ отображени€. 
 *						 аждый полином представлен парой картинок типа CV_64FC1.
 */
void ShowPolynomials(std::string wnd_name, std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> & polynomials);