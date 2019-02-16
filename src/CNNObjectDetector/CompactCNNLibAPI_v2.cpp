
#include "CompactCNNLibAPI_v2.h"
#include "cnn_detector_v3.h"
#include "cnn_models_converter.h"

#include <mutex>

#ifdef USE_CUDA
#	pragma comment(lib, "cudart_static.lib")
#endif
#ifdef USE_CL
#	pragma comment(lib, "OpenCL.lib")
#endif

//#define USE_OPENCV
#ifdef USE_OPENCV
#	include <opencv2/opencv.hpp>
#	pragma comment(lib, "P:/OpenCV_new/opencv-master/build_x64/lib/Release/opencv_core300_cst.lib")
#	pragma comment(lib, "P:/OpenCV_new/opencv-master/build_x64/lib/Release/opencv_imgproc300_cst.lib")
#	pragma comment(lib, "P:/OpenCV_new/opencv-master/build_x64/lib/Release/opencv_objdetect300_cst.lib")
#endif

using namespace NeuralNetworksLib;


//========================================================================================================


namespace CompactCNNLib
{

	static CNNDetector* CNND = nullptr;
	static std::mutex CNND_mutex;

#ifdef USE_OPENCV
	static void* CVVJFD = nullptr;
	bool use_VJD_check = true;
	float VJD_check_sf = 1.1f;
	int VJD_check_min_nb = 1;
#endif

	#define CHECK_HANDLE(expr)												\
		if (CNND != nullptr) expr;											\
		else printf("[CompactCNNLibAPI] FaceDetector no initialized!\n");	\

	#define MUTEX(expr)			\
		CNND_mutex.lock();		\
		expr;					\
		CNND_mutex.unlock();	\
	
	FaceDetector::Param paramConverter(CNNDetector::Param CNND_param, CNNDetector::AdvancedParam CNND_ad_param)
	{
		FaceDetector::Param param;

		param.max_image_size.width = CNND_param.max_image_size.width;
		param.max_image_size.height = CNND_param.max_image_size.height;
		param.min_face_height = CNND_param.min_obj_size.height;
		param.max_face_height = CNND_param.max_obj_size.height;
		param.scale_factor = CNND_param.scale_factor;
		param.min_neighbors = CNND_param.min_neighbors;

		param.detect_precision = static_cast<FaceDetector::DetectPrecision>(CNND_ad_param.detect_precision);
		param.treshold[0] = CNND_ad_param.treshold_1;
		param.treshold[1] = CNND_ad_param.treshold_2;
		param.treshold[2] = CNND_ad_param.treshold_3;
		param.drop_detect = CNND_ad_param.drop_detect;
		param.equalize = CNND_ad_param.equalize;
		param.reflection = CNND_ad_param.reflection;

		param.pipeline = static_cast<FaceDetector::Pipeline>(CNND_param.pipeline);
		param.detect_mode = static_cast<FaceDetector::DetectMode>(CNND_ad_param.detect_mode);
		param.num_threads = CNND_param.num_threads;

		for (int i = 0; i < 3; ++i)
		{
			param.models[i] = CNND_ad_param.path_model[i].c_str();
			param.index_output[i] = CNND_ad_param.index_output[i];
		}

		param.device_info = CNND_ad_param.device_info;
		param.cuda_device_id = CNND_ad_param.cuda_device_id;
		param.cl_platform_id = CNND_ad_param.cl_platform_id;
		param.cl_device_id = CNND_ad_param.cl_device_id;

		param.facial_analysis = CNND_ad_param.facial_analysis;

		return param;
	}
	std::pair<CNNDetector::Param, CNNDetector::AdvancedParam> paramConverter(FaceDetector::Param param)
	{
		CNNDetector::Param CNND_param;
		CNNDetector::AdvancedParam CNND_ad_param;

		CNND_param.max_image_size.width = param.max_image_size.width;
		CNND_param.max_image_size.height = param.max_image_size.height;
		CNND_param.min_obj_size.height = param.min_face_height;
		CNND_param.max_obj_size.height = param.max_face_height;
		CNND_param.scale_factor = param.scale_factor;
		CNND_param.min_neighbors = param.min_neighbors;

		CNND_ad_param.detect_precision = static_cast<CNNDetector::DetectPrecision>(param.detect_precision);
		CNND_ad_param.treshold_1 = param.treshold[0];
		CNND_ad_param.treshold_2 = param.treshold[1];
		CNND_ad_param.treshold_3 = param.treshold[2];
		CNND_ad_param.drop_detect = param.drop_detect;
		CNND_ad_param.equalize = param.equalize;
		CNND_ad_param.reflection = param.reflection;

		CNND_param.pipeline = static_cast<CNNDetector::Pipeline>(param.pipeline);
		CNND_ad_param.detect_mode = static_cast<CNNDetector::DetectMode>(param.detect_mode);
		CNND_param.num_threads = param.num_threads;

		for (int i = 0; i < 3; ++i)
		{
			if (param.models[i] != nullptr)
				CNND_ad_param.path_model[i] = std::string(param.models[i]);
			else
				CNND_ad_param.path_model[i] = "";
			CNND_ad_param.index_output[i] = param.index_output[i];
		}

		CNND_ad_param.device_info = param.device_info;
		CNND_ad_param.cuda_device_id = param.cuda_device_id;
		CNND_ad_param.cl_platform_id = param.cl_platform_id;
		CNND_ad_param.cl_device_id = param.cl_device_id;

		CNND_ad_param.facial_analysis = param.facial_analysis;

		CNND_ad_param.double_check = true;
		CNND_ad_param.adapt_min_neighbors = false;
		CNND_ad_param.gray_image_only = true;
		CNND_ad_param.packet_detection = false;
		CNND_ad_param.merger_detect = param.detect_mode == FaceDetector::DetectMode::disable ? false : true;
		CNND_ad_param.blur = false;
		CNND_ad_param.uniform_noise = false;

		return std::pair<CNNDetector::Param, CNNDetector::AdvancedParam>(CNND_param, CNND_ad_param);
	}

	FaceDetector::FaceDetector()
	{
		if (1)
		{
			printf("\n\n");
			printf("*******************************************************************\n");
			printf("* Compact Convolutional Neural Network Cascade for Face Detection *\n");
			printf("* http://arxiv.org/abs/1508.01292                                 *\n");
			printf("* Copyright (c) 2018, Ilya Kalinovskiy, e-mail: kua_21@mail.ru    *\n");
			printf("* All rights reserved                                             *\n");
			printf("*******************************************************************\n");
			printf("\n\n");

			printf("[CompactCNNLibAPI] CNN Cascade v2.0\n");
			printf("[CompactCNNLibAPI] Face Detector v3.1\n");
			printf("[CompactCNNLibAPI] API v2.0\n");
#ifdef USE_SSE
			printf("[CompactCNNLibAPI] SIMD support (SSE2)!\n");
#endif
#if defined(USE_AVX) && !defined(USE_AVX2)
			printf("[CompactCNNLibAPI] SIMD support (AVX)!\n");
#endif
#ifdef USE_AVX2
			printf("[CompactCNNLibAPI] SIMD support (AVX2)!\n");
#endif
#ifdef USE_CUDA
			printf("[CompactCNNLibAPI] CUDA support!\n");
#endif
#ifdef USE_OMP
			printf("[CompactCNNLibAPI] OpenMP support (maximum %d threads)!\n", MAX_NUM_THREADS);
#endif
#ifdef USE_CL
			printf("[CompactCNNLibAPI] OpenCL support!\n");
#endif
			printf("\n\n");
		}
	};
	FaceDetector::~FaceDetector() 
	{ 
		Clear(); 
	};

	int FaceDetector::Init(Param& param)
	{
		if (!isEmpty()) Clear();

		std::pair<CNNDetector::Param, CNNDetector::AdvancedParam> CNND_param = paramConverter(param);

		MUTEX(
		CNND = new CNNDetector(&CNND_param.first, &CNND_param.second);
		if (CNND->isEmpty())
		{
			printf("[CompactCNNLibAPI] Error loading FaceDetector!\n");
			CNND->~CNNDetector();
			CNND = nullptr;
		})

		if (isEmpty()) return -1;

#ifdef USE_OPENCV
		if (1)
		{
			CVVJFD = static_cast<void*>(new cv::CascadeClassifier());
			if (!((cv::CascadeClassifier*)CVVJFD)->load("haarcascade_frontalface_alt.xml") || ((cv::CascadeClassifier*)CVVJFD)->empty())
			{
				printf("[CompactCNNLibAPI] Error loading VJ cascade!\n");
				((cv::CascadeClassifier*)CVVJFD)->~CascadeClassifier();
				CVVJFD = 0;
				return -1;
			};

			cv::setUseOptimized(true);
		}
#endif

		return 0;
	};
	
	int FaceDetector::Detect(Face* faces, ImageData& img)
	{
		if (isEmpty())
		{
			printf("[CompactCNNLibAPI] FaceDetector no initialized!\n");
			return -1;
		}

		if (faces == nullptr) return -1;

		CNND_mutex.lock();

#ifdef PROFILE_DETECTOR
		CNND->stat.reset();
#endif

		std::vector<NeuralNetworksLib::CNNDetector::Detection> cnn_faces;
		SIMD::Image_8u image(img.cols, img.rows, img.channels, img.data, (int)img.step);

		//try {
			CNND->Detect(cnn_faces, image);
		//} catch (...) { }

#ifdef PROFILE_DETECTOR
		CNND->stat.print();
#endif

#ifdef USE_OPENCV
		if (use_VJD_check && CVVJFD != nullptr)
		{
			cv::Mat img_mat(img.rows, img.rows, CV_MAKETYPE(CV_8U, img.channels), img.data, img.step);
			cnn_faces.erase(std::remove_if(cnn_faces.begin(), cnn_faces.end(), [&](NeuralNetworksLib::CNNDetector::Detection& detect)
			{
				const float min_sf = 0.75f * 0.75f;
				const float max_sf = 1.f;

				const int dw = int(0.1f * float(detect.rect.width));
				const int dh = int(0.1f * float(detect.rect.height));

				cv::Rect rect;
				rect.x = MAX(0, detect.rect.x - dw);
				rect.y = MAX(0, detect.rect.y - dh);
				rect.width = MIN(img.cols, rect.x + detect.rect.width + (dw << 1)) - rect.x;
				rect.height = MIN(img.rows, rect.y + detect.rect.height + (dh << 1)) - rect.y;

				float w = float(rect.width);
				float h = float(rect.height);

				cv::Mat M_roi(img_mat, rect);
				cv::Mat M_scale;
				resize(M_roi, M_scale, cv::Size(int(80.f * w / h), 80));

				cv::Mat M_gray = M_scale;
				cv::cvtColor(M_gray, M_gray, cv::COLOR_BGR2GRAY);

				w = float(M_gray.cols);
				h = float(M_gray.rows);

				std::vector<cv::Rect> vj_faces(1);
				((cv::CascadeClassifier*)CVVJFD)->detectMultiScale(
					M_gray,
					vj_faces,
					VJD_check_sf,
					VJD_check_min_nb,
					0 | 2 /*CV_HAAR_SCALE_IMAGE*/,
					cv::Size(int(min_sf * w), int(min_sf * h)),
					cv::Size(int(max_sf * w), int(max_sf * h)));

				return vj_faces.size() == 0;
			}), cnn_faces.end());
		}
#endif

		int det_count = 0;
		for (auto it = cnn_faces.begin(); it != cnn_faces.end(); ++it)
		{
			faces[det_count] = Face(it->rect.x, it->rect.y, it->rect.width, it->rect.height, it->score);

			size_t size = it->facial_data.size();
			if (size != 0)
			{
				std::sort(it->facial_data.begin(), it->facial_data.end(),
					[](const CNNDetector::FacialData & a, const CNNDetector::FacialData & b) -> bool
				{
					return a.gender > b.gender;
				});
				faces[det_count].gender = it->facial_data[size / 2].gender;

				std::sort(it->facial_data.begin(), it->facial_data.end(),
					[](const CNNDetector::FacialData & a, const CNNDetector::FacialData & b) -> bool
				{
					return a.smile > b.smile;
				});
				faces[det_count].smile = it->facial_data[size / 2].smile;

				std::sort(it->facial_data.begin(), it->facial_data.end(),
					[](const CNNDetector::FacialData & a, const CNNDetector::FacialData & b) -> bool
				{
					return a.glasses > b.glasses;
				});
				faces[det_count].glasses = it->facial_data[size / 2].glasses;
			}

			det_count++;
		}
		cnn_faces.clear();

		/*
		if (n % write_log_frame_count == 0)
		{
			time /= double(n);

			std::fstream flog;
			flog.open("CCNN_time_log.txt", std::ios_base::app);
			if (flog.is_open())
			{
				CNNDetector::Param fd_param = CNND->getParam();
				CNNDetector::AdvancedParam fd_ad_param = CNND->getAdvancedParam();

				flog << fd_param.min_obj_size.height << " " << fd_param.max_obj_size.height << std::endl;
				flog << fd_param.min_neighbors << " " << fd_param.scale_factor << std::endl;
				flog << fd_ad_param.treshold_1 << " " << fd_ad_param.treshold_2 << " " << fd_ad_param.treshold_3 << std::endl;
				flog << fd_param.num_threads << " " << (int)fd_ad_param.detect_mode << std::endl;
				flog << time << " " << det_count << std::endl;
			}
			flog.close();
			
			time = 0.;
			n = 0;
		}
		*/

		CNND_mutex.unlock();

		return det_count;
	}

	int FaceDetector::Clear()
	{
		if (isEmpty()) return -1;

		MUTEX(
		delete CNND;
		CNND = nullptr;)

#ifdef USE_OPENCV
		if (CVVJFD != 0)
		{
			delete (cv::CascadeClassifier*)CVVJFD;
			CVVJFD = 0;
		}
#endif

		return 0;
	}

	bool FaceDetector::isEmpty() const
	{
		return CNND == nullptr;
	}

	FaceDetector::Param FaceDetector::getParam() const
	{
		MUTEX(
		FaceDetector::Param param;
		CHECK_HANDLE(param = paramConverter(CNND->getParam(), CNND->getAdvancedParam()));)
		return param;
	}
	void FaceDetector::setParam(FaceDetector::Param& param)
	{
		std::pair<CNNDetector::Param, CNNDetector::AdvancedParam> CNND_param = paramConverter(param);
		MUTEX(
		CHECK_HANDLE(CNND->setParam(CNND_param.first));
		CHECK_HANDLE(CNND->setAdvancedParam(CNND_param.second));)

#ifdef USE_OPENCV
		if (CVVJFD != nullptr)
		{
			CVVJFD = static_cast<void*>(new cv::CascadeClassifier());
			if (!((cv::CascadeClassifier*)CVVJFD)->load("haarcascade_frontalface_alt.xml") || ((cv::CascadeClassifier*)CVVJFD)->empty())
			{
				printf("[CompactCNNLibAPI] Error loading VJ cascade!\n");
				((cv::CascadeClassifier*)CVVJFD)->~CascadeClassifier();
				CVVJFD = 0;
				return;
			};

			cv::setUseOptimized(true);
		}
#endif
	}

	FaceDetector::Size FaceDetector::getMaxImageSize() const
	{
		NeuralNetworksLib::Size size;
		MUTEX(
		CHECK_HANDLE(size = CNND->getMaxImageSize());
		return FaceDetector::Size(size.width, size.height);)
	}
	void FaceDetector::setMaxImageSize(FaceDetector::Size max_image_size)
	{
		MUTEX(
		CHECK_HANDLE(CNND->setMaxImageSize(NeuralNetworksLib::Size(max_image_size.width, max_image_size.height)));)
	}

	int FaceDetector::getMinObjectHeight() const
	{
		NeuralNetworksLib::Size size;
		MUTEX(
		CHECK_HANDLE(size = CNND->getMinObjectSize());)
		return size.height;
	}
	void FaceDetector::setMinObjectHeight(int min_obj_height)
	{
		MUTEX(
		CHECK_HANDLE(CNND->setMinObjectSize(NeuralNetworksLib::Size(min_obj_height, min_obj_height)));)
	}

	int FaceDetector::getMaxObjectHeight() const
	{
		NeuralNetworksLib::Size size;
		MUTEX(
		CHECK_HANDLE(size = CNND->getMaxObjectSize());)
		return size.height;
	}
	void FaceDetector::setMaxObjectHeight(int max_obj_height)
	{
		MUTEX(
		CHECK_HANDLE(CNND->setMaxObjectSize(NeuralNetworksLib::Size(max_obj_height, max_obj_height)));)
	}

	int FaceDetector::getNumThreads() const
	{
		int num_threads = 0;
		MUTEX(
		CHECK_HANDLE(num_threads = CNND->getNumThreads());)
		return num_threads;
	}
	void FaceDetector::setNumThreads(int num_threads)
	{
		MUTEX(
		CHECK_HANDLE(CNND->setNumThreads(num_threads));)
	}

	int FaceDetector::getGrayImage32F(ImageData* img, Pipeline pipeline)
	{
		if (pipeline == Pipeline::CPU)
		{
			SIMD::Image_32f gray;
			if (CNND->getGrayImage(&gray) == 0)
			{
				img->cols = gray.width;
				img->rows = gray.height;
				img->channels = gray.nChannel;
				img->data = (unsigned char*)gray.data;
				img->step = gray.widthStep;
				return 0;
			}
			else return -1;
		}
		else
		{
#ifdef USE_CUDA
			CUDA::Image_32f_pinned gray;
			if (CNND->getGrayImage(&gray) == 0)
			{
				img->cols = gray.width;
				img->rows = gray.height;
				img->channels = gray.nChannel;
				img->data = (unsigned char*)gray.dataDevice;
				img->step = gray.widthStepDevice;
				return 0;
			}
			else return -1;
#endif
#ifdef USE_CL
			CL::Image_32f gray;
			if (CNND->getGrayImage(&gray) == 0)
			{
				img->cols = gray.width;
				img->rows = gray.height;
				img->channels = gray.nChannel;
				img->data = (unsigned char*)gray.dataDevice;
				img->step = gray.widthStepDevice;
				return 0;
			}
			else return -1;
#endif
		}
	}

	void FaceDetector::CNTKDump2Binary(const char* binary_file, const char* cntk_model_dump)
	{
#ifdef USE_CNTK_MODELS
		CNNModelsConverter cvtCNN;
		cvtCNN.LoadCNTKModel(cntk_model_dump);
		if (!cvtCNN.isEmpty())
		{
			cvtCNN.SaveToBinaryFile(binary_file);
		}
#endif
	}

}