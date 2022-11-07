#include "MomentsRecognizer.h"

using namespace cv;
using namespace std;
using namespace fe;

MomentsRecognizer::MomentsRecognizer()
{
}


MomentsRecognizer::~MomentsRecognizer()
{
}

std::string MomentsRecognizer::Recognize(fe::ComplexMoments & moments)
{
	Mat output;
	pAnn->predict(MomentsToInput(moments), output);
	return OutputToValue(output);
}

double MomentsRecognizer::PrecisionTest(std::map<std::string, std::vector<fe::ComplexMoments>> moments)
{
	int right_answers = 0;
	int tests = 0;
	for (auto primer = moments.begin(); primer != moments.end(); primer++) {
		for (int j = 0; j < (int)primer->second.size(); j++) {
			auto recognized = Recognize(primer->second[j]);
			if (recognized == primer->first) {
				right_answers++;
			}
			tests++;
		}
	}
	return (double)right_answers / (double)tests;
}

bool MomentsRecognizer::Save(std::string filename)
{
	FileStorage fs(filename, FileStorage::WRITE);
	if (!fs.isOpened()) {
		return false;
	}
	pAnn->write(fs);
	fs << "values" << values;
	fs.release();
	return true;
}

bool MomentsRecognizer::Read(std::string filename)
{
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened()) {
		return false;
	}
	pAnn = ml::ANN_MLP::create();
	pAnn->read(fs.root());
	values.clear();
	for (auto iter = fs["values"].begin(); iter != fs["values"].end(); iter++) {
		values.push_back(*iter);
	}
	fs.release();
	return true;
}

