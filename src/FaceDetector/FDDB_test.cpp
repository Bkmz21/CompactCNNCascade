/*
*	Copyright (c) 2018, Ilya Kalinovskiy
*	All rights reserved.
*
*	This is an implementation of the algorithm described in the following paper:
*		I.A. Kalinovskiy, V.G. Spitsyn,
*		Compact Convolutional Neural Network Cascade for Face Detection,
*		http://arxiv.org/abs/1508.01292.
*
*	Redistribution and use of this program as source code or in binary form, with or without modifications, are permitted provided that the following conditions are met:
*		1. Redistributions may not be sold, nor may they be used in a commercial product or activity without prior permission from the copyright holder (contact him at kua_21@mail.ru).
*		2. Redistributions may not be used for military purposes.
*		3. Any published work which utilizes this program shall include the reference to the paper available at http://arxiv.org/abs/1508.01292
*		4. Redistributions must retain the above copyright notice and the reference to the algorithm on which the implementation is based on, this list of conditions and the following disclaimer.
*
*	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
*	IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "CompactCNNLibAPI_v2.h"
#include <opencv2/opencv.hpp>

#include <map>
#include <iostream>
#include <fstream>
#include <string>


//========================================================================================================


//--------------------------------------------------------------------------------------------------------
//GroundTruth and Statistic structures

struct GroundTruth
{
	struct TPerson {
		cv::Rect rect;
		int trackID;
		bool check;

		TPerson(int X, int Y, int Width, int Height, int _trackID) {
			rect = cv::Rect(X, Y, Width, Height);
			trackID = _trackID;
			check = false;
		}
	};

	int id;
	std::vector<TPerson> person;

	GroundTruth() {
		id = 0;
	}
};
struct Stat
{
	int image_count = 0;
	int face_count = 0;
	int curr_true_detections = 0;
	int curr_false_detections = 0;
	int true_detections = 0;
	int false_detections = 0;
};

//--------------------------------------------------------------------------------------------------------
//FD Benchmarks

//Face Detection Data Set and Benchmark (FDDB) 
//http://vis-www.cs.umass.edu/fddb/
void FDDBPraser(const std::string flist, const std::string gt_data,
				std::vector<std::string>& image_path, std::map<int, GroundTruth*>& ground_truth,
				const int border_size = 0, const std::string decimal_separator = ",")
{
	std::fstream f(flist.c_str(), std::ios_base::in);
	while (f) {
		std::string str;
		f >> str;
		image_path.push_back(str);
	}
	f.close();

	if (image_path.size() == 0) {
		printf("Images not found!");
		return;
	}

	std::ifstream fAnnot(gt_data.c_str());

	int index = 0;
	do {
		GroundTruth* frame_data = new GroundTruth;
		frame_data->id = index;

		std::string imS1;
		std::getline(fAnnot, imS1);

		int nAnnot;
		std::getline(fAnnot, imS1);

		std::stringstream ss1(imS1);
		ss1 >> nAnnot;

		for (int j = 0; j < nAnnot; j++)
		{
			double x, y, t, w, h, sc;

			std::string line;
			std::getline(fAnnot, line);
			std::stringstream ss(line);
			ss >> w >> h >> t >> x >> y >> sc;

			double width = 2. * sqrt(w * w + (h * h - w * w) * sin(t) * sin(t));
			double height = 2. * sqrt(w * w + (h * h - w * w) * cos(t) * cos(t));
			x = x - width * 0.5;
			y = y - height * 0.5;

			int TrackID = 1;
			float X = x + (double)border_size;
			float Y = y + (double)border_size;
			float Width = width;
			float Height = height;

			frame_data->person.push_back(GroundTruth::TPerson(int(X), int(Y), int(Width), int(Height), TrackID));
		}

		ground_truth[frame_data->id] = frame_data;

		index++;
	} while (!fAnnot.eof());
}

//--------------------------------------------------------------------------------------------------------
//Evaluation Detections

bool rectMatching(const cv::Rect& annot, const cv::Rect& det, const float threshold = 0.5f)
{
	std::vector<int> X(4);
	std::vector<int> Y(4);

	for (int j = -1; j <= 1; ++j)
	{
		for (int i = -1; i <= 1; ++i)
		{
			for (int s = -2; s <= 2; ++s)
			{
				double scale = 0.;
				if (s < 0)
				{
					scale = pow(1.1, double(-s));
				}
				else
				{
					scale = pow(0.95, double(s));
				}
				int scale_x = int(scale * double(annot.width));
				int scale_y = int(scale * double(annot.height));

				X[0] = annot.x - ((scale_x - annot.width) >> 1);
				Y[0] = annot.y - ((scale_y - annot.height) >> 1);
				X[1] = X[0] + scale_x;
				Y[1] = Y[0] + scale_y;

				int rx = int(double(X[1] - X[0]) * 0.2);
				if (i < 0)
				{
					X[0] += i * rx;
				}
				else
				{
					X[1] += i * rx;
				}

				int ry = int(double(Y[1] - Y[0]) * 0.2);
				Y[0] += j * ry;

				X[2] = det.x;
				Y[2] = det.y;
				X[3] = det.x + det.width;
				Y[3] = det.y + det.height;

				/*
				The Pascal Visual Object Classes (VOC) Challenge
				http://link.springer.com/article/10.1007%2Fs11263-009-0275-4
				*/
				if (!(X[0] >= X[3] || X[1] <= X[2] || Y[0] >= Y[3] || Y[1] <= Y[2]))
				{
					std::sort(X.begin(), X.end());
					std::sort(Y.begin(), Y.end());

					const float S_union = float((X[3] - X[0]) * (Y[3] - Y[0]));
					const float S_intersection = float((X[2] - X[1]) * (Y[2] - Y[1]));

					float overlapArea = S_intersection / S_union;

					if (overlapArea >= threshold)
					{
						return true;
					}
				}
			}
		}
	}
	return false;
}
void evaluationDetections(std::map<int, GroundTruth*>& ground_truth, const int index, std::vector<cv::Rect>& faces, Stat& stat)
{
	int annot_count = 0;
	int max_track_id = 0;
	int detection_rate = faces.size();
	int true_detections = 0;
	int false_detections = faces.size();
	int true_tracks = 0;
	int false_tracks = 0;

	GroundTruth* fgt = ground_truth[index];

	if (fgt != NULL && fgt->id == index) {
		annot_count = fgt->person.size();

		for (int det_id = 0; det_id < detection_rate; ++det_id) {
			bool bl = true;

			for (int annot_id = 0; annot_id < annot_count; ++annot_id) {
				max_track_id = MAX(max_track_id, fgt->person[annot_id].trackID);

				if (!fgt->person[annot_id].check) {
					if (rectMatching(fgt->person[annot_id].rect, faces[det_id])) {
						bl = false;
						true_detections++;
						fgt->person[annot_id].check = true;
						break;
					}
				}
			}
		}
		false_detections -= true_detections;

		for (int k = 0; k < annot_count; ++k) {
			fgt->person[k].check = false;
		}
	}

	stat.image_count++;
	stat.face_count += annot_count;
	stat.curr_true_detections = true_detections;
	stat.curr_false_detections = false_detections;
	stat.true_detections += true_detections;
	stat.false_detections += false_detections;
	if (1) {
		std::cout << " faces:" << annot_count \
			<< " TP:" << true_detections \
			<< " FP:" << false_detections << std::endl \
			<< "	total_faces:" << stat.face_count \
			<< " total_TP:" << stat.true_detections \
			<< " total_FP:" << stat.false_detections << std::endl;
	}
}

void toCVRect(std::vector<cv::Rect>& cv_faces, std::vector<float>& score, const CompactCNNLib::FaceDetector::Face* faces, const int N)
{
	for (int i = 0; i < N; ++i) {
		cv_faces.push_back(cv::Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height));
		score.push_back(faces[i].score);
	}
}
void rectExtension(std::vector<cv::Rect>& faces)
{
	for (int p = 0; p < (int)faces.size(); ++p) {
		faces[p].x -= int(0.1f * (float)faces[p].width);
		faces[p].y -= int(0.15f * (float)faces[p].height);
		faces[p].width += int(0.2f * (float)faces[p].width);
		faces[p].height += int(0.3f * (float)faces[p].height);
	}
}

//--------------------------------------------------------------------------------------------------------
//I/O

std::string FloatToStr(const float f)
{
	char sc[30];
	std::string str = "";

	sprintf_s(sc, "%f", f);
	str.append(sc);

	if (str.find(",") != -1) {
		str.replace(str.find(","), 1, ".");
	}

	return str;
}
void writeDetections(const std::string log_name, const std::string image_path, const int index,
	std::vector<cv::Rect>& rects, std::vector<float>& score, int border_size = 0)
{
	std::ofstream out_detect_list;
	if (index == 0) {
		out_detect_list.open(log_name, std::ios_base::out);
	}
	else {
		out_detect_list.open(log_name, std::ios_base::app);
	}

	out_detect_list << image_path << "\n";

	out_detect_list << rects.size() << "\n";
	for (int p = 0; p < rects.size(); ++p)
	{
		out_detect_list << FloatToStr(rects[p].x - border_size) << " "
			<< FloatToStr(rects[p].y - border_size) << " "
			<< FloatToStr(rects[p].width) << " "
			<< FloatToStr(rects[p].height) << " "
			<< score[p] << "\n";
	}

	out_detect_list.close();
}

//--------------------------------------------------------------------------------------------------------

int test(CompactCNNLib::FaceDetector& detector, 
	     const std::string data_root, const std::string flist, 
	     const std::string gt_data, const std::string FDDB_result,
	     const bool draw)
{
	const int border_size = 50;
	std::vector<std::string> image_path;
	std::map<int, GroundTruth*> ground_truth;
	FDDBPraser(flist, gt_data, image_path, ground_truth, border_size, ".");

	Stat stat;
	for (std::size_t i = 0; i < image_path.size(); ++i) {
		std::cout << image_path[i].c_str();
		cv::Mat img = cv::imread(data_root + image_path[i] + ".jpg");
		if (!img.data) {
			std::cout << "Could not open or find the image" << std::endl;
			continue;
		}

		cv::Mat img_border;
		cv::copyMakeBorder(img, img_border, border_size, border_size, border_size, border_size, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

		CompactCNNLib::FaceDetector::Face faces[100];
		CompactCNNLib::FaceDetector::ImageData frame_ref(img_border.cols, img_border.rows, img_border.channels(), img_border.data, img_border.step[0]);
		int num_faces = detector.Detect(faces, frame_ref);

		std::vector<cv::Rect> cv_faces;
		std::vector<float> score;
		toCVRect(cv_faces, score, faces, num_faces);
		rectExtension(cv_faces);

		evaluationDetections(ground_truth, i, cv_faces, stat);
		writeDetections(FDDB_result, image_path[i], i, cv_faces, score, border_size);

		if (draw) {
			for (auto rect : cv_faces) {
				cv::rectangle(img_border, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(0, 0, 255), 3, 8, 0);
			}

			GroundTruth* fgt = ground_truth[i];
			if (fgt != NULL && fgt->id == i) {
				for (int p = 0; p < fgt->person.size(); ++p) {
					cv::rectangle(img_border, cv::Point(fgt->person[p].rect.x, fgt->person[p].rect.y), 
						cv::Point(fgt->person[p].rect.x + fgt->person[p].rect.width, fgt->person[p].rect.y + fgt->person[p].rect.height), 
						cv::Scalar(0, 255, 255), 3, 8, 0);
				}
			}

			cv::imshow("Face Detection", img_border);
			cv::waitKey(0);
		}
	}

	std::cout << std::endl
		<< "	Recall: " << stat.true_detections / float(stat.face_count) << std::endl
		<< "	Precision: " << stat.true_detections / float(stat.true_detections + stat.false_detections) << std::endl;

	return 0;
}

//--------------------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	CompactCNNLib::FaceDetector face_detector;
	CompactCNNLib::FaceDetector::Param param;	

	param.max_image_size = CompactCNNLib::FaceDetector::Size(800, 800);
	param.min_face_height = 20;
	param.scale_factor = 1.1f;

	param.min_neighbors = 3;
	param.detect_precision = CompactCNNLib::FaceDetector::DetectPrecision::high;
	param.equalize = false;
	param.reflection = false;

	param.pipeline = CompactCNNLib::FaceDetector::Pipeline::GPU;
	param.num_threads = 4;
	param.drop_detect = false;

	if (face_detector.Init(param) < 0) return -1;

	std::string data_root = FDDBPath;
	std::string image_list = FDDBFold"FDDB-fold-all.txt";
	std::string ground_truth_data = FDDBFold"FDDB-fold-all-ellipseList.txt";
	std::string FDDB_result = "FDDB_result.txt";
	bool draw = false;
	test(face_detector, data_root, image_list, ground_truth_data, FDDB_result, draw);

	std::system("pause");
	return 0;
}