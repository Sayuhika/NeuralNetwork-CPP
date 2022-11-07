#pragma once

#include "Visualisation.h"
#include <opencv2/opencv.hpp>

using namespace cv;

void Show64FC1Mat(std::string wnd_name, cv::Mat mat64fc1)
{
	if (mat64fc1.empty()) {
		throw std::invalid_argument("Empty polynomial");
	}
	Mat show_mat;
	double min, max;
	cv::minMaxLoc(mat64fc1, &min, &max);
	mat64fc1 -= min;
	mat64fc1 /= (max - min);
	mat64fc1.convertTo(show_mat, CV_8UC1, MAX_LEVEL);
	imshow(wnd_name, show_mat);
}

void ShowBlobDecomposition(std::string wnd_name, Mat blob, Mat decomposition)
{
 	if (blob.empty()) throw std::invalid_argument("Empty blob!");
	if (decomposition.empty()) throw std::invalid_argument("Empty blob!");
	if (blob.size != decomposition.size) throw std::invalid_argument("Incorrect size!");
	if (blob.type() != CV_8UC1) throw std::invalid_argument("Incorrect blob mat type!");
	if (decomposition.type() != CV_64FC1) throw std::invalid_argument("Incorrect decomposition mat type!");
	Mat show_decomposition;
	decomposition.convertTo(show_decomposition, CV_8UC1, MAX_LEVEL, MIDDLE_LEVEL);
	Mat show_mat = Mat::zeros(blob.rows, blob.cols * 2, CV_8UC1);
	Mat roi = show_mat(Rect(0, 0, blob.cols, blob.rows));
	blob.copyTo(roi);
	roi = show_mat(Rect(blob.cols, 0, blob.cols, blob.rows));
	show_decomposition.copyTo(roi);
	imshow(wnd_name, show_mat);
}

void ShowPolynomials(std::string wnd_name, std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> & polynomials)
{
	size_t j_max = 0;
	if (polynomials.size() == 0) {
		throw std::invalid_argument("Nothing to show");
	}
	for (size_t i = 0; i < polynomials.size(); i++) {
		if (polynomials[i].size() == 0) {
			throw std::invalid_argument("Empty polynomial line");
		}
		if (polynomials[i].size() > j_max) j_max = polynomials[i].size();
		for (size_t j = 0; j < polynomials[i].size(); j++) {
			if (polynomials[i][j].first.empty()) {
				throw std::invalid_argument("Empty polynomial");
			}
			if (polynomials[i][j].second.empty()) {
				throw std::invalid_argument("Empty polynomial");
			}
		}
	}
	int diameter = polynomials[0][0].first.cols;
	Mat show_mat = Mat::zeros(diameter * polynomials.size(), diameter * j_max * 2, CV_8UC1);
	show_mat.setTo(MIDDLE_LEVEL);
	Mat buf_mat;
	for (size_t i = 0; i < polynomials.size(); i++) {
		for (size_t j = 0; j < polynomials[i].size(); j++)	{
			polynomials[i][j].first.convertTo(buf_mat, CV_8UC1, MAX_LEVEL, MIDDLE_LEVEL);
			Mat roi = show_mat(Rect(2*j*diameter, i*diameter, diameter, diameter));
			buf_mat.copyTo(roi);
			polynomials[i][j].second.convertTo(buf_mat, CV_8UC1, MAX_LEVEL, MIDDLE_LEVEL);
			roi = show_mat(Rect((2 * j + 1)*diameter, i*diameter, diameter, diameter));
			buf_mat.copyTo(roi);
		}
	}
	imshow(wnd_name, show_mat);
}