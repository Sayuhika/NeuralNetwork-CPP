#include <windows.h>
#include <fstream>
#include <shellapi.h>
#include "MomentsHelper.h"

using namespace cv;
using namespace std;
using namespace fe;

#define MIDDLE_LEVEL 127
#define MAX_LEVEL 255
#define VAL_FILENAME "value.txt"

MomentsHelper::MomentsHelper()
{
}


MomentsHelper::~MomentsHelper()
{
}

bool MomentsHelper::GenerateMoments(
	std::string path,
	std::shared_ptr<fe::IBlobProcessor> blob_processor, 
	std::shared_ptr<fe::PolynomialManager> poly_manager, 
	std::map< std::string, std::vector<fe::ComplexMoments> > & res)
{
	vector<string> sample_dirs;
	GetSamplePaths(path, sample_dirs);
	WIN32_FIND_DATA findData;
	HANDLE handle;
	res.clear();

	//Перебираем все папки.
	for (size_t i = 0; i < sample_dirs.size(); i++) {
		// Ищем файл со значением буквы.
		ifstream f_val((path + "\\" + sample_dirs[i] + "\\" + VAL_FILENAME).c_str());

		if (!f_val.is_open()) {
			return false;
		}

		string key;
		f_val >> key;
		f_val.close();

		// Ищем первый файл png.
		handle = FindFirstFile((path + "\\" + sample_dirs[i] + "\\*.png").c_str(), &findData);
		res.insert(pair<string, vector<fe::ComplexMoments>>(key, vector< ComplexMoments >()));

		do {
			//Считываем картинку.
			ComplexMoments mom;

			//Обрабатываем.
			ProcessOneImage(path + "\\" + sample_dirs[i] + "\\" + findData.cFileName, blob_processor, poly_manager, mom);

			//Сохраняем
			res[key].push_back(mom);
		} while (FindNextFile(handle, &findData));

		FindClose(handle);
	}
	return true;
}

bool MomentsHelper::DistributeData(
	std::string labeled_data_path,
	std::string ground_data_path,
	std::string test_data_path,
	double percent)
{
	// Очищаем папки перед использованием
	SHFILEOPSTRUCT FileOp;

	FileOp.hwnd = NULL;
	FileOp.wFunc = FO_DELETE;
	FileOp.pTo = NULL;
	FileOp.fFlags = NULL;
	FileOp.lpszProgressTitle = NULL;

	//FileOp.pFrom = (ground_data_path).c_str();
	FileOp.pFrom = "..\\Resources\\ground_data";
	SHFileOperation(&FileOp);
	CreateDirectory((ground_data_path ).c_str(), NULL);

	//FileOp.pFrom = (test_data_path).c_str();
	FileOp.pFrom = "..\\Resources\\test_data";
	SHFileOperation(&FileOp);
	CreateDirectory((test_data_path).c_str(), NULL);


	// Конец очистки

	vector<string> sample_dirs;
	GetSamplePaths(labeled_data_path, sample_dirs);
	WIN32_FIND_DATA findData;
	HANDLE handle;
	// Для каждой папки создадим соответсвующую в каталоге для примеров и тестовых данных.
	for (size_t i = 0; i < sample_dirs.size(); i++) {
		if (!CreateDirectory((ground_data_path + "\\" + sample_dirs[i]).c_str(), NULL) ||
			!CreateDirectory((test_data_path + "\\" + sample_dirs[i]).c_str(), NULL))
		{
			return false;
		}
	}
	//Заглянем в каждую директорию и раскидаем файлы
	for (size_t i = 0; i < sample_dirs.size(); i++) {
		string find_path = labeled_data_path + "\\" + sample_dirs[i];
		string labeled_val = labeled_data_path + "\\" + sample_dirs[i] + "\\" + VAL_FILENAME;
		string ground_val = ground_data_path + "\\" + sample_dirs[i] + "\\" + VAL_FILENAME;
		string test_val = test_data_path + "\\" + sample_dirs[i] + "\\" + VAL_FILENAME;
		if (!CopyFile(labeled_val.c_str(), ground_val.c_str(), TRUE) ||
			!CopyFile(labeled_val.c_str(), test_val.c_str(), TRUE)
			) {
			return false;
		}
		// Ищем первый файл.
		handle = FindFirstFile((find_path + "\\*.png").c_str(), &findData);
		// И только теперь проходим по нужным нам файлам.
		do {
			string postfix = "\\" + sample_dirs[i] + "\\" + findData.cFileName;
			string copy_from = labeled_data_path + postfix;
			string copy_to = "";
			if (rand() / double(RAND_MAX) < percent / 100.) {
				copy_to = ground_data_path + postfix;
			}
			else {
				copy_to = test_data_path + postfix;
			}
			if (!CopyFile(copy_from.c_str(), copy_to.c_str(), TRUE)) {
				return false;
			}
		} while (FindNextFile(handle, &findData));
		FindClose(handle);
	}
	return true;
}

void MomentsHelper::ProcessOneImage(
		string image_path,
		std::shared_ptr<fe::IBlobProcessor> blob_processor,
		std::shared_ptr<fe::PolynomialManager> poly_manager,
		fe::ComplexMoments & res)
{
	Mat image = imread(image_path, cv::IMREAD_GRAYSCALE);

	if (image.empty()) {
		throw "Empty image";
	}

	threshold(image, image, MIDDLE_LEVEL, MAX_LEVEL, CV_THRESH_BINARY);
	vector<Mat> blobs, nblobs;
	blobs = blob_processor->DetectBlobs(image);

	if (blobs.size() != 1) {
		throw "Incorrect input data. More then one blob.";
	}

	nblobs = blob_processor->NormalizeBlobs(blobs, (poly_manager->GetBasis()[0][0].first.cols));
	res = poly_manager->Decompose(nblobs[0]);
}

bool MomentsHelper::GetSamplePaths(std::string base_path, std::vector<std::string> & paths)
{
	paths.clear();
	WIN32_FIND_DATA findData;
	HANDLE handle;   

	// Ищем первый файл.
	handle = FindFirstFile((base_path + "\\*").c_str(), &findData);

	// Находим ссылку на родительский каталог.
	FindNextFile(handle, &findData);

	// И только теперь проходим по нужным нам файлам.
	// Найдем каталоги с примерами.
	while (FindNextFile(handle, &findData))	{
		//производим необходимые операции, например запоминаем имена всех файлов в директории.
		if ((findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) > 0) 
		{
			paths.push_back(findData.cFileName);
		}
	}

	FindClose(handle);

	return true;
}

bool MomentsHelper::SaveMoments(std::string filename, std::map< std::string, std::vector<fe::ComplexMoments> > & moments)
{
	FileStorage fs(filename, FileStorage::WRITE);

	if (!fs.isOpened()) {
		return false;
	}

	fs << "train_data" << "[";

	for (auto it = moments.begin(); it != moments.end(); it++) 
	{
		fs << "{" << "value" << it->first << "moments" << "[";

		for each (auto mom in it->second) 
		{
			fs << "{" << "re" << mom.re << "im" << mom.im << "abs" << mom.abs << "phase" << mom.phase << "}";
		}

		fs << "]" << "}";
	}

	fs << "]";
	fs.release();

	return true;
}

bool MomentsHelper::ReadMoments(std::string filename, std::map< std::string, std::vector<fe::ComplexMoments> > & moments)
{
	FileStorage fs(filename, FileStorage::READ);

	if (!fs.isOpened()) {
		return false;
	}

	moments.clear();

	for (auto iter = fs["train_data"].begin(); iter != fs["train_data"].end(); iter++) 
	{
		pair<std::string, std::vector<fe::ComplexMoments>> example;
		(*iter)["value"] >> example.first;

		for (auto imoment = (*iter)["moments"].begin(); imoment != (*iter)["moments"].end(); imoment++) 
		{
			ComplexMoments moment;
			(*imoment)["abs"] >> moment.abs;
			(*imoment)["re"] >> moment.re;
			(*imoment)["im"] >> moment.im;
			(*imoment)["phase"] >> moment.phase;
			example.second.push_back(moment);
		}

		moments.insert(example);
	}
	return true;
}
