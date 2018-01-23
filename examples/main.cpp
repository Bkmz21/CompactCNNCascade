/*
*	Copyright (c) 2017, Ilya Kalinovskiy
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

#include "CompactCNNLib.h"
//#pragma comment(lib, "CompactCNNLib_AVX2_CPU_ONLY_x64.lib")
#pragma comment(lib, "CompactCNNLib_AVX_CUDA_x64.lib")
//#pragma comment(lib, "CompactCNNLib_AVX_OCL_x64.lib")

#include <opencv2/opencv.hpp>
#pragma comment(lib, "opencv_core300.lib")
#pragma comment(lib, "opencv_highgui300.lib")
#pragma comment(lib, "opencv_imgcodecs300.lib")
#pragma comment(lib, "opencv_imgproc300.lib")
#pragma comment(lib, "opencv_videoio300.lib")


//========================================================================================================


void video_test(std::string fname = "")
{
	CompactCNNLib::FaceDetector face_detector;
	CompactCNNLib::FaceDetector::Param param;

	////If you use your own models.
	//std::string cntk_bin[3];
	//for (int i = 0; i < 3; ++i)
	//{
	//	std::string cntk_dump = "W:/Projects/NeuralNetworks/CCNN-Cascade/cntk/cnn4face" + std::to_string(i + 1) + "_dump.txt";
	//	cntk_bin[i] = "W:/Projects/NeuralNetworks/CCNN-Cascade/cntk/cnn4face" + std::to_string(i + 1) + ".bin";
	//	CompactCNNLib::FaceDetector::CNTKDump2Binary(cntk_bin[i].c_str(), cntk_dump.c_str());
	//	param.models[i] = cntk_bin[i].c_str();
	//}

	param.facial_analysis = true;

	if (face_detector.Init(param) < 0) return;

	cv::VideoCapture capture;
	if (fname == "")
	{
		if (!capture.open(0)) return;
		capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
		capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
	}
	else
		if(!capture.open(fname)) return;

	cv::namedWindow("Face detection", cv::WINDOW_NORMAL);
	std::vector<CompactCNNLib::FaceDetector::Face> faces(1000);
	double time = 0.;
	for (;;)
	{
		cv::Mat frame, frame_gray;
		capture >> frame;
		if (frame.empty()) break;
		cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
		CompactCNNLib::FaceDetector::ImageData frame_shared(frame_gray.cols, frame_gray.rows, frame_gray.channels(), frame_gray.data, frame_gray.step[0]);
		
		int64 t0 = cv::getTickCount();
		int num_faces = face_detector.Detect(faces.data(), frame_shared);
		int64 t1 = cv::getTickCount();
		double secs = (t1 - t0) / cv::getTickFrequency();
		printf("time detect: %5.3f ms\n", secs * 1000.);

		for (int i = 0; i < num_faces; ++i)
		{
			cv::rectangle(frame, cv::Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), cv::Scalar(0, 0, 255), 3);
		
			std::stringstream ss_gender;
			ss_gender << "gender: " << (faces[i].gender ? "female" : "male");
			putText(frame, ss_gender.str().c_str(), cv::Point(faces[i].x, faces[i].y + faces[i].height + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
			
			std::stringstream ss_smile;
			ss_smile << "smile: " << (faces[i].smile ? "false" : "true");
			putText(frame, ss_smile.str().c_str(), cv::Point(faces[i].x, faces[i].y + faces[i].height + 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
			
			std::stringstream ss_glasses;
			ss_glasses << "glasses: " << (faces[i].glasses ? "false" : "true");
			putText(frame, ss_glasses.str().c_str(), cv::Point(faces[i].x, faces[i].y + faces[i].height + 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
		}

		int frame_index = abs(capture.get(cv::CAP_PROP_POS_FRAMES));
		time = frame_index > 1 ? time + secs : secs;

		std::stringstream ss; ss.precision(3);
		ss << "frame: " << frame_index << " | " << frame_index / time << " fps";
		putText(frame, ss.str().c_str(), cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

		cv::imshow("Face detection", frame);
		if (cv::waitKey(1) == 27) break;
	}
	
	face_detector.Clear();
}

void image_test(std::string path)
{
	if (path == "") return;

	CompactCNNLib::FaceDetector face_detector;
	CompactCNNLib::FaceDetector::Param param;
	param.max_image_size = CompactCNNLib::FaceDetector::Size(5000, 3000);
	param.min_face_height = 20;
	param.scale_factor = 1.1f;
	param.equalize = false;
	param.reflection = false;
	param.pipeline = CompactCNNLib::FaceDetector::Pipeline::GPU;
	param.num_threads = 4;
	if (face_detector.Init(param) < 0) return;

	std::vector<cv::String> fname;
	cv::glob(path, fname);
	cv::namedWindow("Face detection", cv::WINDOW_NORMAL);
	std::vector<CompactCNNLib::FaceDetector::Face> faces(1000);
	for (size_t i = 0; i < fname.size(); ++i)
	{
		cv::Mat im = cv::imread(fname[i]);
		if (im.empty()) continue; 
		CompactCNNLib::FaceDetector::ImageData im_shared(im.cols, im.rows, im.channels(), im.data, im.step[0]);

		int64 t0 = cv::getTickCount();
		int num_faces = face_detector.Detect(faces.data(), im_shared);
		int64 t1 = cv::getTickCount();
		double secs = (t1 - t0) / cv::getTickFrequency();
		printf("faces: %d, time detect: %5.3f ms\n", num_faces, secs * 1000.);

		for (int i = 0; i < num_faces; ++i)
			cv::rectangle(im, cv::Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), cv::Scalar(0, 0, 255), 3);

		std::stringstream ss; ss.precision(3);
		ss << secs * 1000. << " ms";
		putText(im, ss.str().c_str(), cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

		//cv::imwrite(fname[i] + "_result.jpg", im);
		cv::imshow("Face detection", im);
		if (cv::waitKey(0) == 27) break;
	}
}

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		printf("Usage: [-c] [-v file] [-i folder]\n");
		return 0;
	}

	std::string source = argv[1];
	std::string path = argv[std::min(2, argc - 1)];

	if (source == "-c") video_test();
	if (source == "-v") video_test(path);
	if (source == "-i") image_test(path);

	return 0;
}