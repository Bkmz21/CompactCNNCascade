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

//OpenCV
#include <opencv2/opencv.hpp>
#pragma comment(lib, "opencv_core300.lib")
#pragma comment(lib, "opencv_highgui300.lib")
#pragma comment(lib, "opencv_imgcodecs300.lib")
#pragma comment(lib, "opencv_imgproc300.lib")
#pragma comment(lib, "opencv_videoio300.lib")
#pragma comment(lib, "opencv_video300.lib")
#pragma comment(lib, "opencv_objdetect300.lib")

//Boost
//#include <boost/filesystem.hpp>
#include <iostream>

//GRANSAC
#include "GRANSAC.hpp"
#include "CircleModel.hpp"

#include "cnn_detector_v3.h"
#include "hr_timer.h"

#ifdef USE_CUDA
#	pragma comment(lib, "cudart_static.lib")
#endif
#ifdef USE_CL
#	pragma comment(lib, "OpenCL.lib")
#endif

#include <iostream>
#include <vector>
#include <string>
#include <time.h>

//Dlib library
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#pragma comment(lib, "dlib.lib")
#pragma comment(lib, "dlib_cuda.lib")

dlib::shape_predictor dlib_shapePredictor;

using namespace NeuralNetworksLib;


//================================================================================================================================================


void genTrainData()
{
	return;

	CNNDetector::Param param;
	CNNDetector::AdvancedParam ad_param;

	param.pipeline = CNNDetector::Pipeline::CPU;
	param.min_obj_size = Size(80, 80);
	param.scale_factor = 1.025f;
	param.min_neighbors = 1;
	param.num_threads = 4;
	ad_param.treshold_1 = -1.5f;
	ad_param.treshold_2 = -1.0f;
	ad_param.treshold_3 = -0.5f;
	ad_param.detect_mode = CNNDetector::DetectMode::sync;
	ad_param.detect_precision = CNNDetector::DetectPrecision::default;
	ad_param.merger_detect = false;
	ad_param.drop_detect = false;

	NeuralNetworksLib::CNNDetector CNND(&param, &ad_param);

	cv::Size out_img_size(49, 51);

	std::string path_in = "I:/MTFL/";
	std::string fname = path_in + "training.txt";

	std::string path_out = "P:/Landmark_train/train/";

	std::ifstream fAnnot(fname);
	if (!fAnnot.is_open())
	{
		printf("error open file\n");
		std::system("pause");
		return;
	}

	std::ofstream fout_data(path_out + "landmark_data.txt");
	std::ofstream fout_labels(path_out + "landmark_labels.txt");

	int border = 25;
	int count = 0;
	do
	{
		std::string line, path_img;
		float px[5], py[5];
		int lb[4];

		std::getline(fAnnot, line);
		std::stringstream ss(line);
		ss >> path_img >> px[0] >> px[1] >> px[2] >> px[3] >> px[4] >> py[0] >> py[1] >> py[2] >> py[3] >> py[4] >> lb[0] >> lb[1] >> lb[2] >> lb[3];
		std::cout << path_img << std::endl;

		cv::Mat img = cv::imread(path_in + path_img);
		if (img.empty()) continue;

		cv::copyMakeBorder(img, img, border, border, border, border, cv::BORDER_CONSTANT, cv::Scalar(0));	
		for (int i = 0; i < 5; ++i)
		{
			px[i] += float(border);
			py[i] += float(border);
		}
		for (int i = 0; i < 4; ++i)
		{
			lb[i] -= 1;
		}

		for (int t = 0; t < 2; ++t)
		{
			if (t == 1)
			{
				cv::flip(img, img, 1);
				for (int i = 0; i < 5; ++i)
				{
					px[i] = img.cols - px[i];
				}
			}

			std::vector<CNNDetector::Detection> faces;
			SIMD::Image_8u in_img(img.cols, img.rows, img.channels(), img.data, img.step[0]);
			CNND.Detect(faces, in_img);

			for (size_t idx = 0; idx < faces.size(); ++idx)
			{
				CNNDetector::Detection face = faces[idx];

				bool bl = false;
				for (int i = 0; i < 5; ++i)
				{
					if (face.rect.intersects(Rect(px[i], py[i], 1, 1)) == 0)
					{
						bl = true;
						break;
					}
				}
				if (bl) continue;

				cv::Rect rect = cv::Rect(face.rect.x, face.rect.y, face.rect.width, face.rect.height);
				if (0)
				{
					path_img.erase(0, path_img.find('\\') + 1);
					std::string fname_img = path_out + "faces/" + path_img + "_" + std::to_string(idx) + "_" + std::to_string(t) + ".jpg";
					fout_data << fname_img + "	0" << std::endl;
					fout_labels << "|labels";
					for (int i = 0; i < 5; ++i)
					{
						fout_labels << " ";
						float rx = float(out_img_size.width) / float(rect.width);
						float ry = float(out_img_size.height) / float(rect.height);
						fout_labels << rx * (px[i] - float(rect.x)) / float(out_img_size.width) << " " << ry * (py[i] - float(rect.y)) / float(out_img_size.height);
					}
					for (int i = 0; i < 3; ++i)
					{
						fout_labels << " ";
						fout_labels << float(lb[i]);
					}
					fout_labels << std::endl;

					cv::Mat roi = img(rect);
					cv::Mat roi_gray;
					cv::cvtColor(roi, roi_gray, CV_BGR2GRAY);

					cv::Mat roi_gray_resize;
					cv::resize(roi_gray, roi_gray_resize, out_img_size);

					std::vector<cv::Mat> channels;
					channels.push_back(roi_gray_resize);
					channels.push_back(roi_gray_resize);
					channels.push_back(roi_gray_resize);

					cv::Mat megre;
					merge(channels, megre);
					cv::imwrite(fname_img, megre);
				}

				//if (t == 1) cv::rectangle(img, rect, cv::Scalar(125), 1);
			}

			//for (int i = 0; i < 5; ++i)
			//{
			//	if (t == 1) cv::ellipse(img, cv::Point(px[i], py[i]), cv::Size(5, 5), 0, 0, 360, cv::Scalar(255), -1);
			//}

			//cv::imshow("img", img);
			//cv::waitKey(0);
		}
		count++;
		if (count > 1000000000) break;
	}
	while (!fAnnot.eof());
	fAnnot.close();

	//std::string path_nofaces = "P:/face_train/train/bgr_gray/";
	//namespace fs = boost::filesystem;
	//for (fs::recursive_directory_iterator it(path_nofaces), end; it != end; ++it)
	//{
	//	if (it->path().extension() == ".jpg")
	//	{
	//		std::string fpath = fs::absolute(*it).string();
	//		std::cout << fpath << std::endl;

	//		fout_data << fpath + "	0" << std::endl;
	//		fout_labels << "|labels";
	//		for (int i = 0; i < 5; ++i)
	//		{
	//			fout_labels << " ";
	//			fout_labels << 0.f << " " << 0.f;
	//		}
	//		for (int i = 0; i < 3; ++i)
	//		{
	//			fout_labels << " ";
	//			fout_labels << 0.f;
	//		}
	//		fout_labels << std::endl;
	//	}
	//}

	fout_data.close();
	fout_labels.close();
}

void genCNKTModel()
{
	int L1_count = 24;
	int L2_count = 48;
	int L3_count = 96;
	int HL_connect = 96;
	int HL_scale = 1;
	int OL_count = 13;

	std::string path_out = "P:/Landmark_train/";
	std::ofstream fmodel(path_out + "genCNTK.txt");

	//gen L1
	for (int i = 0; i < L1_count; ++i)
	{
		fmodel << "            ";
		fmodel << "c1" << std::to_string(i + 1);
		fmodel << " = ConvolutionalLayer {1, (4:4), pad=false, activation=LeakyReLU, init=\"gaussian\", initValueScale=0.043} (featNorm)";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L1_count; ++i)
	{
		fmodel << "            ";
		fmodel << "bn1" << std::to_string(i + 1);
		fmodel << " = BatchNormalizationLayer {spatialRank=2, normalizationTimeConstant=5000, epsilon=0.00001, useCntkEngine=false} (c1";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L1_count; ++i)
	{
		fmodel << "            ";
		fmodel << "p1" << std::to_string(i + 1);
		fmodel << " = MaxPoolingLayer {(2:2), stride=(2:2)} (bn1";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L1_count; ++i)
	{
		fmodel << "            ";
		fmodel << "s1" << std::to_string(i + 1);
		fmodel << " = ";
		if (i == 0)
		{
			fmodel << "p1" + std::to_string(i + 1) << " + p1" + std::to_string(i + 2);
		}
		else
		{
			for (int j = 0; j < 3; ++j)
			{
				if (i + j > L1_count) continue;
				fmodel << "p1" + std::to_string(i + j);
				if (i + j + 1 > L1_count) continue;
				if (j < 2) fmodel << " + ";
			}
		}	
		fmodel << std::endl;
	}
	fmodel << std::endl;

	//gen L2
	for (int i = 0; i < L2_count; ++i)
	{
		fmodel << "            ";
		fmodel << "c2" << std::to_string(i + 1);
		fmodel << " = ConvolutionalLayer {1, (3:3), pad=false, activation=LeakyReLU, init=\"gaussian\", initValueScale=0.043} (s1";
		fmodel << std::to_string(i / 2 + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L2_count; ++i)
	{
		fmodel << "            ";
		fmodel << "bn2" << std::to_string(i + 1);
		fmodel << " = BatchNormalizationLayer {spatialRank=2, normalizationTimeConstant=5000, epsilon=0.00001, useCntkEngine=false} (c2";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L2_count; ++i)
	{
		fmodel << "            ";
		fmodel << "p2" << std::to_string(i + 1);
		fmodel << " = MaxPoolingLayer {(2:2), stride=(2:2)} (bn2";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L2_count; ++i)
	{
		fmodel << "            ";
		fmodel << "s2" << std::to_string(i + 1);
		fmodel << " = ";
		if (i == 0)
		{
			fmodel << "p2" + std::to_string(i + 1) << " + p2" + std::to_string(i + 2);
		}
		else
		{
			for (int j = 0; j < 3; ++j)
			{
				if (i + j > L2_count) continue;
				fmodel << "p2" + std::to_string(i + j);
				if (i + j + 1 > L2_count) continue;
				if (j < 2) fmodel << " + ";
			}
		}
		fmodel << std::endl;
	}
	fmodel << std::endl;

	//gen L3
	for (int i = 0; i < L3_count; ++i)
	{
		fmodel << "            ";
		fmodel << "c3" << std::to_string(i + 1);
		fmodel << " = ConvolutionalLayer {1, (10:11), pad=false, activation=LeakyReLU, init=\"gaussian\", initValueScale=0.0414} (s2";
		fmodel << std::to_string(i / 2 + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L3_count; ++i)
	{
		fmodel << "            ";
		fmodel << "bn3" << std::to_string(i + 1);
		fmodel << " = BatchNormalizationLayer {spatialRank=2, normalizationTimeConstant=5000, epsilon=0.00001, useCntkEngine=false} (c3";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	//gen HL
	int HL_count = L3_count - HL_connect + 1;
	for (int i = 0; i < HL_count; ++i)
	{
		fmodel << "            ";
		fmodel << "concat_" << std::to_string(i + 1);
		fmodel << " = RowStack(";
		for (int j = 0; j < HL_connect; ++j)
		{
			fmodel << "bn3" << std::to_string(i + j + 1);
			if (j < HL_connect - 1) fmodel << ":";
		}
		fmodel << ")";
		fmodel << std::endl;

		fmodel << "            ";
		fmodel << "h" << std::to_string(i + 1);
		fmodel << " = ConvolutionalLayer {" << std::to_string(HL_scale);
		fmodel << ", (" << std::to_string(HL_connect) << ":1), pad=false, activation=ScaleTanhApprox, init=\"gaussian\", initValueScale=0.55} (";
		fmodel << "concat_" << std::to_string(i + 1) << ")";
		fmodel << std::endl;

		fmodel << "            ";
		fmodel << "d" << std::to_string(i + 1) << " = Dropout (h" << std::to_string(i + 1) << ")";
		fmodel << std::endl << std::endl;
	}

	fmodel << "            ";
	fmodel << "concat_h = RowStack(";
	for (int i = 0; i < HL_count; ++i)
	{
		fmodel << "d" << std::to_string(i + 1);
		if (i < HL_count - 1) fmodel << ":";
	}
	fmodel << ")" << std::endl << std::endl;

	fmodel << "            ";
	fmodel << "bn_h = BatchNormalizationLayer {spatialRank=2, normalizationTimeConstant=15000, epsilon=0.00001, useCntkEngine=false} (concat_h)";
	fmodel << std::endl << std::endl;

	fmodel << "            ";
	fmodel << "z = DenseLayer {labelDim, bias=true, activation=ScaleTanhApprox, init=\"gaussian\", initValueScale=1.05} (bn_h)";
	fmodel << std::endl << std::endl;

	fmodel.close();
}

void genCNKTModel2()
{
	int L1_count = 4;
	int L2_count = 8;
	int L3_count = 16;
	int L4_count = 16;
	int HL_connect = L4_count;
	int HL_scale = 400;
	int OL_count = 13;

	std::string path_out = "P:/Landmark_train/";
	std::ofstream fmodel(path_out + "genCNTK.txt");

	//gen L1
	for (int i = 0; i < L1_count; ++i)
	{
		fmodel << "            ";
		fmodel << "c1" << std::to_string(i + 1);
		fmodel << " = ConvolutionalLayer {1, (4:4), pad=false, activation=LeakyReLU, init=\"gaussian\", initValueScale=0.043} (featNorm)";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L1_count; ++i)
	{
		fmodel << "            ";
		fmodel << "bn1" << std::to_string(i + 1);
		fmodel << " = BatchNormalizationLayer {spatialRank=2, normalizationTimeConstant=5000, epsilon=0.00001, useCntkEngine=false} (c1";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L1_count; ++i)
	{
		fmodel << "            ";
		fmodel << "p1" << std::to_string(i + 1);
		fmodel << " = MaxPoolingLayer {(2:2), stride=(2:2)} (bn1";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L1_count; ++i)
	{
		fmodel << "            ";
		fmodel << "s1" << std::to_string(i + 1);
		fmodel << " = ";
		if (i == 0)
		{
			fmodel << "p1" + std::to_string(i + 1) << " + p1" + std::to_string(i + 2);
		}
		else
		{
			for (int j = 0; j < 3; ++j)
			{
				if (i + j > L1_count) continue;
				fmodel << "p1" + std::to_string(i + j);
				if (i + j + 1 > L1_count) continue;
				if (j < 2) fmodel << " + ";
			}
		}
		fmodel << std::endl;
	}
	fmodel << std::endl;

	//gen L2
	for (int i = 0; i < L2_count; ++i)
	{
		fmodel << "            ";
		fmodel << "c2" << std::to_string(i + 1);
		fmodel << " = ConvolutionalLayer {1, (3:3), pad=false, activation=LeakyReLU, init=\"gaussian\", initValueScale=0.043} (s1";
		fmodel << std::to_string(i / 2 + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L2_count; ++i)
	{
		fmodel << "            ";
		fmodel << "bn2" << std::to_string(i + 1);
		fmodel << " = BatchNormalizationLayer {spatialRank=2, normalizationTimeConstant=5000, epsilon=0.00001, useCntkEngine=false} (c2";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L2_count; ++i)
	{
		fmodel << "            ";
		fmodel << "p2" << std::to_string(i + 1);
		fmodel << " = MaxPoolingLayer {(2:2), stride=(2:2)} (bn2";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L2_count; ++i)
	{
		fmodel << "            ";
		fmodel << "s2" << std::to_string(i + 1);
		fmodel << " = ";
		if (i == 0)
		{
			fmodel << "p2" + std::to_string(i + 1) << " + p2" + std::to_string(i + 2);
		}
		else
		{
			for (int j = 0; j < 3; ++j)
			{
				if (i + j > L2_count) continue;
				fmodel << "p2" + std::to_string(i + j);
				if (i + j + 1 > L2_count) continue;
				if (j < 2) fmodel << " + ";
			}
		}
		fmodel << std::endl;
	}
	fmodel << std::endl;

	//gen L3
	for (int i = 0; i < L3_count; ++i)
	{
		fmodel << "            ";
		fmodel << "c3" << std::to_string(i + 1);
		fmodel << " = ConvolutionalLayer {1, (3:3), pad=false, activation=LeakyReLU, init=\"gaussian\", initValueScale=0.0414} (s2";
		fmodel << std::to_string(i / 2 + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	for (int i = 0; i < L3_count; ++i)
	{
		fmodel << "            ";
		fmodel << "bn3" << std::to_string(i + 1);
		fmodel << " = BatchNormalizationLayer {spatialRank=2, normalizationTimeConstant=5000, epsilon=0.00001, useCntkEngine=false} (c3";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl;
	}
	fmodel << std::endl;

	fmodel << "            ";
	fmodel << "s3 = ";
	for (int i = 0; i < L3_count; ++i)
	{
		fmodel << "bn3" + std::to_string(i + 1);
		if (i < L3_count - 1) fmodel << " + ";
	}
	fmodel << std::endl << std::endl;

	//gen L4
	for (int i = 0; i < OL_count; ++i)
	{
		fmodel << "            ";
		fmodel << "c4" << std::to_string(i + 1);
		fmodel << " = ConvolutionalLayer {" << L4_count << ", (3:3), pad=false, activation=ScaleTanhApprox, init=\"gaussian\", initValueScale=0.0414} (s3)";
		fmodel << std::endl;

		fmodel << "            ";
		fmodel << "bn4" << std::to_string(i + 1);
		fmodel << " = BatchNormalizationLayer {spatialRank=2, normalizationTimeConstant=5000, epsilon=0.00001, useCntkEngine=false} (c4";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl;

		fmodel << "            ";
		fmodel << "p4" << std::to_string(i + 1);
		fmodel << " = MaxPoolingLayer {(6:7), stride=(1:1)} (bn4";
		fmodel << std::to_string(i + 1) << ")";
		fmodel << std::endl << std::endl;
	}

	//gen HL
	int HL_count = L4_count - HL_connect + 1;
	for (int i = 0; i < HL_count; ++i)
	{
		fmodel << "            ";
		fmodel << "concat_" << std::to_string(i + 1);
		fmodel << " = RowStack(";
		for (int j = 0; j < OL_count; ++j)
		{
			fmodel << "p4" << std::to_string(i + j + 1);
			if (j < OL_count - 1) fmodel << ":";
		}
		fmodel << ")";
		fmodel << std::endl;

		fmodel << "            ";
		fmodel << "h" << std::to_string(i + 1);
		fmodel << " = ConvolutionalLayer {" << std::to_string(HL_scale);
		fmodel << ", (" << std::to_string(OL_count) << ":1), pad=false, activation=ScaleTanhApprox, init=\"gaussian\", initValueScale=0.55} (";
		fmodel << "concat_" << std::to_string(i + 1) << ")";
		fmodel << std::endl;

		fmodel << "            ";
		fmodel << "d" << std::to_string(i + 1) << " = Dropout (h" << std::to_string(i + 1) << ")";
		fmodel << std::endl << std::endl;
	}

	fmodel << "            ";
	fmodel << "concat_h = RowStack(";
	for (int i = 0; i < HL_count; ++i)
	{
		fmodel << "d" << std::to_string(i + 1);
		if (i < HL_count - 1) fmodel << ":";
	}
	fmodel << ")" << std::endl << std::endl;

	fmodel << "            ";
	fmodel << "bn_h = BatchNormalizationLayer {spatialRank=2, normalizationTimeConstant=15000, epsilon=0.00001, useCntkEngine=false} (concat_h)";
	fmodel << std::endl << std::endl;

	fmodel << "            ";
	fmodel << "z = DenseLayer {labelDim, bias=true, activation=ScaleTanhApprox, init=\"gaussian\", initValueScale=1.05} (bn_h)";
	fmodel << std::endl << std::endl;

	fmodel.close();
}

void testModel()
{
	SIMD::ConvNeuralNetwork cnn4landmark;
//#ifdef USE_AVX
	cnn4landmark.LoadCNTKModel("P:/Landmark_train/landmark_model_dump.txt");
//	cnn4landmark.SaveToBinaryFile("P:/Landmark_train/landmark_model.bin");
//#else
//	cnn4landmark.Init("P:/Landmark_train/landmark_model.bin");
//#endif

	cnn4landmark.AllocateMemory(cnn4landmark.getMinInputImgSize());
	cnn4landmark.setNumThreads(1);

	std::string path_out = "P:/Landmark_train/train/";
	std::ifstream fdata(path_out + "landmark_data.txt");
	std::ifstream flabels(path_out + "landmark_labels.txt");

	Other::Timer tm;
	cv::namedWindow("Landmark Detection", cv::WINDOW_NORMAL);
	do
	{
		std::string line, path_img;
		int id;
		float px[5], py[5];

		//for (int i = 0; i < 100; ++i)
		{
			std::getline(fdata, line);
		}
		std::stringstream ss1(line);
		ss1 >> path_img >> id;

		//for (int i = 0; i < 100; ++i)
		{
			std::getline(flabels, line);
		}
		std::stringstream ss2(line);
		ss2 >> line >> px[0] >> py[0] >> px[1] >> py[1] >> px[2] >> py[2] >> px[3] >> py[3] >> px[4] >> py[4];
		std::cout << path_img << std::endl;

		cv::Mat img = cv::imread(path_img);
		if (img.empty()) continue;
		//cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
		//cv::flip(img, img, 0);

		SIMD::Image_8u in_img8u(img.cols, img.rows, img.channels(), img.data, img.step[0]);
		SIMD::Image_32f in_img32f(img.cols, img.rows, roundUpMul(in_img8u.width + 1, REG_SIZE), ALIGN_DEF);
		SIMD::Image_32f resp;
		
		SIMD::ImageConverter::Img8uToImg32fGRAY(in_img32f, in_img8u);
		tm.start();
		cnn4landmark.Forward(resp, in_img32f);
		printf("time detect = %f ms\n", tm.get(1000.));

		for (int i = 0; i < 5; ++i)
		{
			px[i] *= float(img.cols);
			py[i] *= float(img.rows);
			cv::Scalar color(255, 0, 0);
			if (i == 0) color = cv::Scalar(0, 0, 255);
			cv::ellipse(img, cv::Point(px[i], py[i]), cv::Size(2, 2), 0, 0, 360, color, -1);

			float rx = resp.data[2 * i * resp.widthStep] * float(img.cols);
			float ry = resp.data[(2 * i + 1) * resp.widthStep] * float(img.rows);
			color = cv::Scalar(0, 0, 0);
			if (i == 0) color = cv::Scalar(0, 255, 0);
			cv::ellipse(img, cv::Point(rx, ry), cv::Size(2, 2), 0, 0, 360, color, -1);
		}

		cv::imshow("Landmark Detection", img);
		cv::waitKey(0);
	} 
	while (!fdata.eof());
	fdata.close();
	flabels.close();
}

void convertLabels()
{
	std::string path_out = "P:/Landmark_train/train/";
	std::ifstream fdata(path_out + "landmark_data.txt");
	std::ifstream flabels_in(path_out + "landmark_labels.txt");
	std::ofstream flabels_out(path_out + "landmark_labels_2.txt");

	do
	{
		std::string line, path_img;
		int id;
		float px[5], py[5], ld[3];

		std::getline(fdata, line);
		std::stringstream ss1(line);
		ss1 >> path_img >> id;

		std::getline(flabels_in, line);
		std::stringstream ss2(line);
		ss2 >> line >> px[0] >> py[0] >> px[1] >> py[1] >> px[2] >> py[2] >> px[3] >> py[3] >> px[4] >> py[4] >> ld[0] >> ld[1] >> ld[2];

		if (path_img.find("_1.jpg") != std::string::npos)
		{
			flabels_out << line
				<< " " << px[1] << " " << py[1]
				<< " " << px[0] << " " << py[0]
				<< " " << px[2] << " " << py[2]
				<< " " << px[4] << " " << py[4]
				<< " " << px[3] << " " << py[3]
				<< " " << ld[0] << " " << ld[1] << " " << ld[2]
				<< std::endl;
		}
		else
		{
			flabels_out << line
				<< " " << px[0] << " " << py[0]
				<< " " << px[1] << " " << py[1]
				<< " " << px[2] << " " << py[2]
				<< " " << px[3] << " " << py[3]
				<< " " << px[4] << " " << py[4]
				<< " " << ld[0] << " " << ld[1] << " " << ld[2]
				<< std::endl;
		}
	} 
	while (!flabels_in.eof());

	fdata.close();
	flabels_out.close();
	flabels_in.close();
}

void genTestData()
{
	if (1)
	{
		std::string path = "W:/Projects/NeuralNetworks/CNNFaceRecognition/ext/OpenFaceCpp/models/dlib/shape_predictor_68_face_landmarks.dat";
		dlib::deserialize(path) >> dlib_shapePredictor;
	}

	CNNDetector::Param param;
	CNNDetector::AdvancedParam ad_param;

	param.pipeline = CNNDetector::Pipeline::CPU;
	param.min_obj_size = Size(80, 80);
	param.scale_factor = 1.1f;
	param.min_neighbors = 1;
	param.num_threads = 4;
	ad_param.treshold_1 = -1.5f;
	ad_param.treshold_2 = -1.0f;
	ad_param.treshold_3 = -0.5f;
	ad_param.detect_mode = CNNDetector::DetectMode::sync;
	ad_param.detect_precision = CNNDetector::DetectPrecision::default;
	ad_param.merger_detect = false;
	ad_param.drop_detect = false;

	NeuralNetworksLib::CNNDetector CNND(&param, &ad_param);

	SIMD::ConvNeuralNetwork cnn4landmark;
	cnn4landmark.LoadCNTKModel("P:/Landmark_train/landmark_model_dump.txt");
	//cnn4landmark.SaveToBinaryFile("P:/Landmark_train/landmark_model.bin");
	cnn4landmark.AllocateMemory(cnn4landmark.getMinInputImgSize());
	cnn4landmark.setNumThreads(1);

	cv::Size out_img_size(cnn4landmark.getMinInputImgSize().width, cnn4landmark.getMinInputImgSize().height);

	std::string path_in = "I:/MTFL/";
	std::string fname = path_in + "testing.txt";

	std::ifstream fAnnot(fname);
	if (!fAnnot.is_open())
	{
		printf("error open file\n");
		std::system("pause");
		return;
	}

	double dist_pix[5] = { 0., 0., 0., 0., 0. };
	int dist_count[5] = { 0, 0, 0, 0, 0 };
	int gender = 0;
	int smile = 0;
	int glasses = 0;
	int count = 0;

	int border = 25;
	Other::Timer tm;
	cv::namedWindow("Landmark Detection", cv::WINDOW_NORMAL);
	do
	{
		std::string line, path_img;
		float px[5], py[5];
		int lb[4];

		std::getline(fAnnot, line);
		std::stringstream ss(line);
		ss >> path_img >> px[0] >> px[1] >> px[2] >> px[3] >> px[4] >> py[0] >> py[1] >> py[2] >> py[3] >> py[4] >> lb[0] >> lb[1] >> lb[2] >> lb[3];
		std::cout << path_img << std::endl;

		cv::Mat img = cv::imread(path_in + path_img);
		if (img.empty()) continue;

		cv::copyMakeBorder(img, img, border, border, border, border, cv::BORDER_CONSTANT, cv::Scalar(0));
		for (int i = 0; i < 5; ++i)
		{
			px[i] += float(border);
			py[i] += float(border);
		}
		for (int i = 0; i < 4; ++i)
		{
			lb[i] -= 1;
		}

		std::vector<CNNDetector::Detection> faces;
		SIMD::Image_8u in_img(img.cols, img.rows, img.channels(), img.data, img.step[0]);
		CNND.Detect(faces, in_img);

		//printf("faces.size() = %d\n", faces.size());
		for (size_t idx = 0; idx < faces.size(); ++idx)
		{
			CNNDetector::Detection face = faces[idx];

			bool bl = false;
			for (int i = 0; i < 5; ++i)
			{
				if (face.rect.intersects(Rect(px[i], py[i], 1, 1)) == 0)
				{
					bl = true;
					break;
				}
			}
			if (bl) continue;

			cv::Rect rect = cv::Rect(face.rect.x, face.rect.y, face.rect.width, face.rect.height);
			cv::Mat roi = img(rect);
			cv::Mat roi_resize;
			cv::resize(roi, roi_resize, out_img_size);

			SIMD::Image_8u in_img8u(roi_resize.cols, roi_resize.rows, roi_resize.channels(), roi_resize.data, roi_resize.step[0]);
			SIMD::Image_32f in_img32f(roi_resize.cols, roi_resize.rows, roundUpMul(in_img8u.width + 1, REG_SIZE), ALIGN_DEF);
			SIMD::Image_32f resp;

			SIMD::ImageConverter::Img8uToImg32fGRAY(in_img32f, in_img8u);
			//tm.start();
			cnn4landmark.Forward(resp, in_img32f);
			//printf("time landmark = %f ms\n", tm.get(1000.));

			if (0)
			{
				dlib::array2d<dlib::bgr_pixel> dlib_img;
				dlib::rectangle bb(dlib::point(0, 0), dlib::point(roi_resize.cols, roi_resize.rows));

				dlib::assign_image(dlib_img, dlib::cv_image<dlib::bgr_pixel>(roi_resize));
				tm.start();
				dlib::full_object_detection shape = dlib_shapePredictor(dlib_img, bb);
				printf("time dlib landmark = %f ms\n", tm.get(1000.));

				std::vector<dlib::point> res;
				res.reserve(68);
				for (int i = 0; i < shape.num_parts(); i++)
				{
					res.push_back(shape.part(i));
				}
				if (res.size() < 68) continue;

				for (int i = 0; i < res.size(); ++i)
				{
					float px_ = res[i].x() /*/ float(rect.width) * float(roi_resize.cols)*/;
					float py_ = res[i].y() /*/ float(rect.height) * float(roi_resize.rows)*/;
					cv::ellipse(roi_resize, cv::Point(px_, py_), cv::Size(1, 1), 0, 0, 360, cv::Scalar(0, 255, 0), -1);
				}
			}

			for (int i = 0; i < 5; ++i)
			{
				float px_ = (px[i] - float(rect.x)) / float(rect.width) * float(roi_resize.cols);
				float py_ = (py[i] - float(rect.y)) / float(rect.height) * float(roi_resize.rows);
				cv::Scalar color(255, 0, 0);
				if (i == 0) color = cv::Scalar(0, 0, 255);
				cv::ellipse(roi_resize, cv::Point(px_, py_), cv::Size(1, 1), 0, 0, 360, color, -1);

				float rx = resp.data[2 * i * resp.widthStep] * float(roi_resize.cols);
				float ry = resp.data[(2 * i + 1) * resp.widthStep] * float(roi_resize.rows);
				color = cv::Scalar(0, 0, 0);
				//if (i == 0) color = cv::Scalar(0, 255, 0);
				cv::ellipse(roi_resize, cv::Point(rx, ry), cv::Size(1, 1), 0, 0, 360, color, -1);

				if (abs(px_ - rx) > 2 || abs(py_ - ry) > 2)
				{
					dist_pix[i] += sqrt(double((px_ - rx) * (px_ - rx) + (py_ - ry) * (py_ - ry)));
					dist_count[i]++;
				}
			}
			//printf("%f %f %f\n", resp.data[10 * resp.widthStep], resp.data[11 * resp.widthStep], resp.data[12 * resp.widthStep]);

			if (abs(resp.data[10 * resp.widthStep] - float(lb[0])) <= 0.5f) gender++;
			if (abs(resp.data[11 * resp.widthStep] - float(lb[1])) <= 0.5f) smile++;
			if (abs(resp.data[12 * resp.widthStep] - float(lb[2])) <= 0.5f) glasses++;

			//cv::imshow("Landmark Detection", roi_resize);
			//cv::waitKey(0);

			count++;
			//break;
		}
		//if (count > 200) break;
	} 
	while (!fAnnot.eof());
	fAnnot.close();

	gender = count - gender;
	smile = count - smile;
	glasses = count - glasses;
	printf("Result:\n");
	printf("	count: %d\n", count);
	for (int i = 0; i < 5; ++i)
	{
		printf("	dist_pix[%d]: %8.3f, dist_count[%d]: %d\n", i, dist_pix[i] /= double(dist_count[i]), i, dist_count[i]);
	}
	printf("	gender: %8.3f%\n", 100. - double(gender) / double(count) * 100.);
	printf("	smile: %8.3f%\n", 100. - double(smile) / double(count) * 100.);
	printf("	glasses: %8.3f%\n", 100. - double(glasses) / double(count) * 100.);

	system("pause");
}

void testVideo()
{
	if (1)
	{
		std::string path = "W:/Projects/NeuralNetworks/CNNFaceRecognition/ext/OpenFaceCpp/models/dlib/shape_predictor_68_face_landmarks.dat";
		dlib::deserialize(path) >> dlib_shapePredictor;
	}

	CNNDetector::Param param;
	CNNDetector::AdvancedParam ad_param;

	param.pipeline = CNNDetector::Pipeline::CPU;
	param.min_obj_size = Size(40, 40);
	param.scale_factor = 1.1f;
	param.min_neighbors = 1;
	param.num_threads = 4;
	ad_param.treshold_1 = -1.5f;
	ad_param.treshold_2 = -1.0f;
	ad_param.treshold_3 = -0.5f;
	ad_param.detect_mode = CNNDetector::DetectMode::sync;
	ad_param.detect_precision = CNNDetector::DetectPrecision::default;
	ad_param.merger_detect = true;
	ad_param.drop_detect = false;
	ad_param.facial_analysis = true;

	NeuralNetworksLib::CNNDetector CNND(&param, &ad_param);

	SIMD::ConvNeuralNetwork cnn4landmark;
	//cnn4landmark.LoadCNTKModel("P:/Landmark_train/landmark_model_dump.txt");
	cnn4landmark.Init("P:/Landmark_train/landmark_model.bin");
	//cnn4landmark.SaveToBinaryFile("P:/Landmark_train/landmark_model.cpt", CNND.hGrd);
	cnn4landmark.AllocateMemory(cnn4landmark.getMinInputImgSize());
	cnn4landmark.setNumThreads(1);

	cv::Size out_img_size(cnn4landmark.getMinInputImgSize().width, cnn4landmark.getMinInputImgSize().height);

	cv::VideoCapture capture("H:/clip/How.I.Met.Your.Mother.S05E12.rus.eng.720p.HDTV.[Kuraj-Bambey.Ru].mkv");
	if (!capture.isOpened())
	{
		printf("error load file\n");
		system("pause");
	}

	Other::Timer tm;
	cv::namedWindow("Landmark Detection", cv::WINDOW_NORMAL);
	do
	{
		cv::Mat img;
		if (!capture.grab()) break;
		capture.retrieve(img);
		if (img.empty()) continue;

		int frame_index = (int)capture.get(CV_CAP_PROP_POS_FRAMES);
		if (frame_index % 1 != 0 || frame_index < 100) continue;
		printf("\nframe = %d\n", frame_index);

		tm.start();
		std::vector<CNNDetector::Detection> faces;
		SIMD::Image_8u in_img(img.cols, img.rows, img.channels(), img.data, img.step[0]);
		//CNND.Detect(faces, in_img);
		printf("time detect = %f ms\n", tm.get(1000.));

		//printf("faces.size() = %d\n", faces.size());
		if (0)
		for (size_t idx = 0; idx < faces.size(); ++idx)
		{
			CNNDetector::Detection face = faces[idx];

			cv::Rect rect = cv::Rect(face.rect.x, face.rect.y, face.rect.width, face.rect.height);
			cv::Mat roi = img(rect);
			cv::Mat roi_resize;
			cv::resize(roi, roi_resize, out_img_size);

			SIMD::Image_8u in_img8u(roi_resize.cols, roi_resize.rows, roi_resize.channels(), roi_resize.data, roi_resize.step[0]);
			SIMD::Image_32f in_img32f(roi_resize.cols, roi_resize.rows, roundUpMul(in_img8u.width + 1, REG_SIZE), ALIGN_DEF);
			SIMD::Image_32f resp;

			SIMD::ImageConverter::Img8uToImg32fGRAY(in_img32f, in_img8u);
			//tm.start();
			cnn4landmark.Forward(resp, in_img32f);
			//printf("time landmark = %f ms\n", tm.get(1000.) / 1000.);

			if (1)
			{
				dlib::array2d<dlib::bgr_pixel> dlib_img;
				dlib::rectangle bb(dlib::point(0, 0), dlib::point(roi_resize.cols, roi_resize.rows));

				dlib::assign_image(dlib_img, dlib::cv_image<dlib::bgr_pixel>(roi_resize));
				tm.start();
				dlib::full_object_detection shape = dlib_shapePredictor(dlib_img, bb);
				printf("time dlib landmark = %f ms\n", tm.get(1000.));

				std::vector<dlib::point> res;
				res.reserve(68);
				for (int i = 0; i < shape.num_parts(); i++)
				{
					res.push_back(shape.part(i));
				}
				if (res.size() < 68) continue;

				for (int i = 0; i < res.size(); ++i)
				{
					float px_ = res[i].x() / float(roi_resize.cols) * float(rect.width) + rect.x;
					float py_ = res[i].y() / float(roi_resize.rows) * float(rect.height) + rect.y;
					cv::ellipse(img, cv::Point(px_, py_), cv::Size(3, 3), 0, 0, 360, cv::Scalar(255, 0, 0), -1);
				}
			}

			for (int i = 0; i < 5; ++i)
			{
				if (0)
				{
					float rx = resp.data[2 * i * resp.widthStep] * float(roi_resize.cols);
					float ry = resp.data[(2 * i + 1) * resp.widthStep] * float(roi_resize.rows);
					cv::Scalar color(0, 0, 0);
					//if (i == 0) color = cv::Scalar(0, 255, 0);
					cv::ellipse(roi_resize, cv::Point(rx, ry), cv::Size(1, 1), 0, 0, 360, color, -1);
				}
				else
				{
					float rx = resp.data[2 * i * resp.widthStep] * float(rect.width) + rect.x;
					float ry = resp.data[(2 * i + 1) * resp.widthStep] * float(rect.height) + rect.y;
					cv::Scalar color(0, 0, 255);
					//if (i == 0) color = cv::Scalar(0, 255, 0);
					cv::ellipse(img, cv::Point(rx, ry), cv::Size(5, 5), 0, 0, 360, color, -1);
				}
			}
			cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 3);
			//printf("%f %f %f\n", resp.data[10 * resp.widthStep], resp.data[11 * resp.widthStep], resp.data[12 * resp.widthStep]);
			//int l1 = (int)(resp.data[10 * resp.widthStep] + 0.5f) + 1;
			//int l2 = (int)(resp.data[11 * resp.widthStep] + 0.5f) + 1;
			//int l3 = (int)(resp.data[12 * resp.widthStep] + 0.5f) + 1;
			//printf("%d %d %d\n", l1, l2, l3);
		}

		if (1)
		for (size_t idx = 0; idx < faces.size(); ++idx)
		{
			CNNDetector::Detection face = faces[idx];
			cv::Rect rect = cv::Rect(face.rect.x, face.rect.y, face.rect.width, face.rect.height);
			
			size_t size = face.facial_data.size();
			if (size == 0)
			{
				cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 3);
				continue;
			}

			std::sort(face.facial_data.begin(), face.facial_data.end(),
					[](const CNNDetector::FacialData & a, const CNNDetector::FacialData & b) -> bool
			{
				return a.gender > b.gender;
			});
			byte gender = face.facial_data[size / 2].gender;

			if (gender == 0)
				cv::rectangle(img, rect, cv::Scalar(255, 0, 0), 3);
			else
				cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 3);

			//for (int t = 0; t < face.facial_data.size(); ++t)
			//{
			//	for (int i = 0; i < 5; ++i)
			//	{
			//		cv::ellipse(img, cv::Point(face.facial_data[t].landmarks[i].x, face.facial_data[t].landmarks[i].y), cv::Size(5, 5), 0, 0, 360, cv::Scalar(0, 0, 0), -1);
			//	}
			//}

			for (int i = 0; i < 5; ++i)
			{
				if (face.facial_data.size() < 3)
				{
					cv::ellipse(img, cv::Point(face.facial_data[0].landmarks[i].x, face.facial_data[0].landmarks[i].y), cv::Size(5, 5), 0, 0, 360, cv::Scalar(0, 0, 255), -1);
					continue;
				}

				std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> CandPoints;
				for (size_t t = 0; t < face.facial_data.size(); ++t)
				{
					std::shared_ptr<GRANSAC::AbstractParameter> CandPt = std::make_shared<GRANSAC::Point2D>(face.facial_data[t].landmarks[i].x, face.facial_data[t].landmarks[i].y);
					CandPoints.push_back(CandPt);
				}

				GRANSAC::RANSAC<GRANSAC::Circle2DModel, 2> Estimator;
				Estimator.Initialize(10, 25); // Threshold, iterations
				Estimator.Estimate(CandPoints);

				auto BestCircle = Estimator.GetBestModel();
				if (BestCircle)
				{
					cv::ellipse(img, cv::Point(BestCircle->m_x, BestCircle->m_y), cv::Size(5, 5), 0, 0, 360, cv::Scalar(0, 0, 255), -1);
				}
			}
		}
		
		cv::imshow("Landmark Detection", img);
		cv::waitKey(1);
	} 
	while (true);
	system("pause");
}

int main(int argc, char* argv[])
{
	//genTrainData();
	
	//genCNKTModel();

	//testModel();

	//convertLabels();

	//genTestData();

	testVideo();

	return 0;
}
