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


#include "cnn_detector_v3.h"
#include "serialized_models.h"

//#include <opencv2/opencv.hpp>

#ifdef USE_OMP
#	include <omp.h>
#endif

#undef min
#undef max

#if defined(_MSC_VER)
#	define AtomicCompareExchangeSwap(destination, exchange, comparand) InterlockedCompareExchange(destination, exchange, comparand)
#else
#	define AtomicCompareExchangeSwap(destination, exchange, comparand) __sync_val_compare_and_swap(destination, comparand, exchange)
#endif


//========================================================================================================


namespace NeuralNetworksLib
{

	CNNDetector::CNNDetector(Param* _param, AdvancedParam* _advanced_param)
	{
		if (_param != 0)
		{
			param = *_param;
		}
		if (_advanced_param != 0)
		{
			advanced_param = *_advanced_param;
		}

		ext_pattern_offset = 11;
		x_pattern_offset = 2;

		Init();
	}
	CNNDetector::~CNNDetector()
	{
		Clear();
	}

	int CNNDetector::InitDevice()
	{
#if !defined(USE_CUDA) && !defined(USE_CL)
		param.pipeline = Pipeline::CPU;
#endif

		if (advanced_param.packet_detection)
		{
#ifdef USE_CUDA
			if ((int)param.pipeline > 0)
			{
				param.pipeline = Pipeline::GPU;
			}
			else
#endif
			{
				advanced_param.packet_detection = false;
			}
		}

		if ((int)param.pipeline > 0)
		{
#ifdef USE_CUDA
			if (CUDA::InitDevice(advanced_param.cuda_device_id, advanced_param.device_info) < 0)
			{
				param.pipeline = Pipeline::CPU;
				printf("[CNNDetector] Used CPU only!\n");
			}
			else
			{
				CUDA::ImageConverter::Init();
				CUDA::ImageResizer::Init();

				cuERR(cudaStreamCreate(&cu_stream0));
				cuERR(cudaEventCreate(&cu_event_img_gray));
				cuERR(cudaEventCreate(&cu_event_calc_cnn));

				advanced_param.concurrent_kernels = MAX(1, abs(advanced_param.concurrent_kernels));
				scl_id.resize(advanced_param.concurrent_kernels, -1);
				cu_stream.resize(advanced_param.concurrent_kernels);
				cu_event_response_map.resize(advanced_param.concurrent_kernels);
				for (int i = 0; i < advanced_param.concurrent_kernels; ++i)
				{
					cuERR(cudaStreamCreate(&cu_stream[i]));
					cuERR(cudaEventCreate(&cu_event_response_map[i]));
				}
			}
#endif
#ifdef USE_CL
			if (CL::InitDevice(advanced_param.cl_platform_id, advanced_param.cl_device_id, cl_device, cl_context, cl_queue, advanced_param.device_info) < 0) 
			{
				param.pipeline = Pipeline::CPU;
				printf("[CNNDetector] Used CPU only!\n");
			}
			else
			{
				CL::ImageConverter::create_cl_program(cl_device, cl_context);
				CL::ImageResizer::create_cl_program(cl_device, cl_context);
			}
#endif
		}

#if defined(_MSC_VER)
		check_detect_event = CreateEvent(NULL, TRUE, FALSE, NULL);
#endif

		//init threads
		OMP_RUNTIME(int max_num_threads = MAX_NUM_THREADS;)

		num_threads = 1;
		if (param.num_threads <= 0)
		{
			OMP_RUNTIME(num_threads = MIN(omp_get_num_procs(), max_num_threads))
		}
		else
		{
			OMP_RUNTIME(num_threads = MIN(param.num_threads, max_num_threads))
		}

#ifdef PROFILE_DETECTOR
		cpu_timer_detector = new Timer();
		cpu_timer_cnn = new Timer();
		cpu_timer_check1 = new Timer();
		cpu_timer_check2 = new Timer();
#	ifdef USE_CUDA
		cu_timer = new CUDA::Timer();
#	endif
#	ifdef USE_CL
		cl_timer = new CL::Timer(cl_queue);
#	endif
#endif

		return 0;
	}
	int CNNDetector::InitCNN()
	{
		//init CNN SIMD
#ifdef USE_AVX
		cpu_cnn = new SIMD::ConvNeuralNetwork_v2();
		//cpu_cnn = new Legacy::ConvNeuralNetwork();
		//advanced_param.path_model[0] = "test.bin";

		//Legacy::ConvNeuralNetwork* leg_cnn = new Legacy::ConvNeuralNetwork("test.bin");
		//SIMD::ConvNeuralNetwork* new_cnn = new SIMD::ConvNeuralNetwork();
		//new_cnn->Init(leg_cnn, true);
		//new_cnn->SaveToBinaryFile(advanced_param.path_model[0]);
#else
		cpu_cnn = new SIMD::ConvNeuralNetwork();
#endif

		/*
		cpu_cnn->LoadCNTKModel("P:/RTSD_train/cnn4rtsd1_cntk.txt");
		cv::Mat img = cv::imread("D:/RTSD/Classification/rtsd-r1/train/009478.png");
		cv::Mat img_resize = img;
		cv::resize(img, img_resize, cv::Size(cpu_cnn->getMinInputImgSize().width, cpu_cnn->getMinInputImgSize().height), 0, 0, 1);

		cv::Mat img_gray = cv::Mat::zeros(img_resize.size(), CV_8U);
		for (int y = 0; y < img_resize.rows; ++y)
		{
			for (int x = 0; x < img_resize.cols; ++x)
			{
				img_gray.at<uchar>(y, x) = img_resize.at<cv::Vec3b>(y, x)[0];
			}
		}
		cv::imshow("face", img_gray);
		cv::waitKey(0);

		SIMD::Image_8u i8u(img_gray.cols, img_gray.rows, img_gray.channels(), img_gray.data, (int)img_gray.step[0]);
		SIMD::Image_32f i32f(img_gray.cols, img_gray.rows);
		SIMD::ImageConverter::UCharToFloat(i32f, i8u);
		SIMD::Image_32f resp_map(1, 1);

		cpu_cnn->AllocateMemory(i32f.getSize());
		cpu_cnn->Forward(resp_map, i32f);
		printf("val = %f\n", resp_map.data[0]);

		system("pause");
		*/

		cpu_cnn->Init(advanced_param.path_model[0], advanced_param.index_output[0], hGrd);
		//cpu_cnn->Init(leg_cnn, true);
		//cpu_cnn->LoadCNTKModel("P:/face_train/dump.txt");
		if (cpu_cnn->isEmpty())
		{
			delete cpu_cnn;
			cpu_cnn = NULL;
			return -1;
		}

		/*
		cpu_cnn->Clear();
		cpu_cnn->Init(advanced_param.path_model[0]);
		cpu_cnn->SaveToBinaryFile("cnn4face1_cntk", hGrd);
		cpu_cnn->Clear();
		cpu_cnn->Init(advanced_param.path_model[1]);
		cpu_cnn->SaveToBinaryFile("cnn4face2_cntk", hGrd);
		cpu_cnn->Clear();
		cpu_cnn->Init(advanced_param.path_model[2]);
		cpu_cnn->SaveToBinaryFile("cnn4face3_cntk", hGrd);
		*/

		if ((int)param.pipeline > 0)
		{
#ifdef USE_CUDA
			//init CNN CUDA
			advanced_param.num_cnn_copy = 1; //MAX(1, abs(advanced_param.num_cnn_copy));
			cu_cnn.resize(advanced_param.num_cnn_copy);
			for (auto it = cu_cnn.begin(); it != cu_cnn.end(); ++it)
			{
				(*it) = new CUDA::ConvNeuralNetwork();
				(*it)->Init(advanced_param.path_model[0], advanced_param.index_output[0], hGrd);
				if ((*it)->isEmpty())
				{
					delete (*it);
					(*it) = NULL;
					cu_cnn.clear();
					return -1;
				}
			}
#endif
#ifdef USE_CL
			//init CNN CL
			cl_cnn = new CL::ConvNeuralNetwork();
			cl_cnn->Init(advanced_param.path_model[0], advanced_param.index_output[0], cl_device, cl_context, cl_queue);
			if (cl_cnn->isEmpty())
			{
				delete cl_cnn;
				cl_cnn = NULL;
				return -1;
			}
#endif
		}

#ifdef CHECK_TEST
		//init CNN
		Legacy::ConvNeuralNetwork* old_cnn = NULL;
		old_cnn = new Legacy::ConvNeuralNetwork(MODELSPATH"cnn4face1_old.bin");
		if (old_cnn->isEmpty())
		{
			delete old_cnn;
			old_cnn = NULL;
			return -1;
		}

		cpu_cnn->setCNNRef(old_cnn);

		if ((int)param.pipeline > 0)
		{
#	ifdef USE_CUDA
			for (auto it = cu_cnn.begin(); it != cu_cnn.end(); ++it)
			{
				(*it)->setCNNRef(old_cnn);
			}
#	endif
#	ifdef USE_CL
			cl_cnn->setCNNRef(old_cnn);
#	endif
		}
#endif

		return 0;
	}
	int CNNDetector::InitScales()
	{
		pattern_size = cpu_cnn->getMinInputImgSize();

		float max_scale = 1.f;
		if (param.min_obj_size.height > 0)
		{
			max_scale = (float)pattern_size.height / (float)param.min_obj_size.height;
			max_scale = MIN(3.f, max_scale);

			if (advanced_param.gray_image_only && max_scale < 0.8f)
			{
				Size input_size = param.max_image_size * max_scale;
				cpu_input_img = SIMD::Image_8u(input_size.width, input_size.height);
				cpu_input_img_resizer = new SIMD::ImageResizer(input_size, param.max_image_size);
				cpu_input_img_scale = max_scale;
				max_scale = 1.f;
			}
		}

		float min_scale = 1.f;
		if (param.max_obj_size.height > 0)
		{
			min_scale = (float)pattern_size.height / (float)param.max_obj_size.height;
		}
		else
		{
			min_scale = (float)pattern_size.height / (float)param.max_image_size.height;
		}
		min_scale = MIN(max_scale, min_scale);

		if (param.scale_factor < 1.f)
		{
			param.scale_factor = 1.1f;
		}

		float scale = max_scale;
		if (param.scale_factor != 1.f)
		{
			while (scale > min_scale)
			{
				scales.push_back(scale);
				scale /= param.scale_factor;
			}
		}
		else
		{
			scales.push_back(scale);
		}

		if (scales.size() == 0)
		{
			printf("[CNNDetector] Invalid configuration!\n");
			return -1;
		}

		return 0;
	}
	int CNNDetector::InitPaketCNN()
	{
		//init packet CNN
		if (advanced_param.packet_detection)
		{
			float area_ratio_max = packing2D.packing(&pack, &pack_size, scales, param.max_image_size, int(1.f + scales[0]) * param.max_image_size.width);
			for (int i = 1; i < MIN(4, (int)scales.size()); ++i)
			{
				std::vector<Rect> pack_temp;
				Size pack_size_temp;
				float area_ratio = packing2D.packing(&pack_temp, &pack_size_temp, scales, param.max_image_size, int(1.f + scales[i]) * param.max_image_size.width);
				if (area_ratio > area_ratio_max)
				{
					pack.clear();
					pack = pack_temp;
					pack_size = pack_size_temp;
					area_ratio_max = area_ratio;
				}
			}
			pack_size.height = int(1.2f * pack_size.height);

			if (param.pipeline != Pipeline::GPU)
			{
				pack_cpu_img_scale = SIMD::Image_32f(pack_size.width, pack_size.height, ALIGN_DEF, true);
			}

			if ((int)param.pipeline > 0)
			{
#ifdef USE_CUDA
				pack_cu_img_scale = CUDA::Image_32f(
					pack_size.width,
					pack_size.height,
					roundUpMul(pack_size.width, REG_SIZE),
					addRoundUpMul(pack_size.width, cu_cnn[0]->getBlockSize().width),
					addRoundUpMul(pack_size.height, cu_cnn[0]->getBlockSize().height),
					ALIGN_DEF);
#endif
#ifdef USE_CL
				pack_cl_img_scale = CL::Image_32f(
					cl_context,
					cl_queue,
					pack_size.width,
					pack_size.height,
					roundUpMul(pack_size.width, REG_SIZE),
					addRoundUpMul(pack_size.width, cl_cnn->getBlockSize().width),
					addRoundUpMul(pack_size.height, cl_cnn->getBlockSize().height),
					ALIGN_DEF);
#endif

				if (advanced_param.detect_mode != DetectMode::disable)
				{
#ifdef USE_CUDA
					cu_cnn_check = new CUDA::ConvNeuralNetwork();
					cu_cnn_check->Init(advanced_param.path_model[1], advanced_param.index_output[0], hGrd);
					if (cu_cnn_check->isEmpty())
					{
						delete cu_cnn_check;
						cu_cnn_check = NULL;
						return -1;
					}

					pattern_size_cd = cu_cnn_check->getMinInputImgSize();
					ext_pattern_size_cd = pattern_size_cd + ext_pattern_offset;

					pack_img_check_size = Size(
						roundUpMul(ext_pattern_size_cd.width + x_pattern_offset, 4) * pack_max_num_img_check.cols,
						roundUpMul(ext_pattern_size_cd.height, 4) * pack_max_num_img_check.rows);

					pack_cu_img_check_8u = CUDA::Image_8u_pinned(
						pack_img_check_size.width,
						pack_img_check_size.height,
						pack_img_check_size.width,
						addRoundUpMul(pack_img_check_size.width, cu_cnn_check->getBlockSize().width),
						addRoundUpMul(pack_img_check_size.height, cu_cnn_check->getBlockSize().height),
						ALIGN_DEF);

					pack_cu_img_check_32f = CUDA::Image_32f(
						pack_img_check_size.width,
						pack_img_check_size.height,
						pack_img_check_size.width,
						addRoundUpMul(pack_img_check_size.width, cu_cnn_check->getBlockSize().width),
						addRoundUpMul(pack_img_check_size.height, cu_cnn_check->getBlockSize().height),
						ALIGN_DEF);

					cu_cnn_check->AllocateMemory(pack_cu_img_check_32f.getSize());

					Size output_size = cu_cnn_check->getOutputImgSize();
					pack_cu_response_map_check = CUDA::Image_32f_pinned(
						output_size.width,
						output_size.height,
						roundUpMul(output_size.width, REG_SIZE),
						addRoundUpMul(output_size.width, cu_cnn_check->getBlockSize().width),
						addRoundUpMul(output_size.height, cu_cnn_check->getBlockSize().height),
						ALIGN_DEF);
#endif	
				}
			}
		}

		return 0;
	}
	int CNNDetector::InitBuffers()
	{
		//init img_buffer
		if (param.pipeline == Pipeline::CPU)
		{
			cpu_img_gray = SIMD::Image_32f(
				param.max_image_size.width, 
				param.max_image_size.height, 
				roundUpMul(param.max_image_size.width + 1, REG_SIZE), 
				ALIGN_DEF);
		}
		else
		{
#ifdef USE_CUDA
			cu_img_input = CUDA::Image_8u_pinned(
				param.max_image_size.width, 
				param.max_image_size.height, 
				4 * param.max_image_size.width, 
				roundUpMul(4 * param.max_image_size.width, 32),
				param.max_image_size.height, 
				ALIGN_DEF);
				cu_img_input.nChannel = 4;

			cu_img_gray = CUDA::Image_32f_pinned(
				param.max_image_size.width, 
				param.max_image_size.height, 
				roundUpMul(param.max_image_size.width, REG_SIZE), 
				roundUpMul(param.max_image_size.width, cu_cnn[0]->getBlockSize().width),
				roundUpMul(param.max_image_size.height, cu_cnn[0]->getBlockSize().height),
				ALIGN_DEF);
#endif
#ifdef USE_CL
			cl_img_input = CL::Image_8u(
				cl_context, 
				cl_queue, 
				param.max_image_size.width, 
				param.max_image_size.height, 
				4, 
				true,
				ALIGN_DEF);
			
			cl_img_gray = CL::Image_32f(
				cl_context, 
				cl_queue,
				param.max_image_size.width,
				param.max_image_size.height, 
				roundUpMul(param.max_image_size.width, REG_SIZE), 
				roundUpMul(param.max_image_size.width + 1, cl_cnn->getBlockSize().width), 
				roundUpMul(param.max_image_size.height + 1, cl_cnn->getBlockSize().height), 
				ALIGN_DEF);
			
			cl_img_temp = CL::Image_32f(
				cl_context, 
				cl_queue, 
				param.max_image_size.width, 
				param.max_image_size.height, 
				roundUpMul(param.max_image_size.width, REG_SIZE), 
				roundUpMul(param.max_image_size.width + 1, cl_cnn->getBlockSize().width), 
				roundUpMul(param.max_image_size.height + 1, cl_cnn->getBlockSize().height), 
				ALIGN_DEF);
#endif
		}

		cpu_img_scale.resize(scales.size());
#ifdef USE_CUDA
		cu_img_scale.resize(scales.size());
#endif
#ifdef USE_CL
		cl_img_scale.resize(scales.size());
#endif
		cpu_img_resizer.resize(scales.size());

		for (int scl = 0; scl < (int)scales.size(); ++scl)
		{
			Size img_resize = param.max_image_size * scales[scl];

			if (param.pipeline != Pipeline::GPU)
			{
				if (!advanced_param.packet_detection)
				{
					cpu_img_scale[scl] = SIMD::Image_32f(img_resize.width, img_resize.height, ALIGN_DEF, true);
				}
				else
				{
					int offset = pack[scl].y * pack_cpu_img_scale.widthStep + pack[scl].x;
					cpu_img_scale[scl] = SIMD::Image_32f(
						img_resize.width, 
						img_resize.height, 
						1, 
						pack_cpu_img_scale.data + offset, 
						pack_cpu_img_scale.widthStep);
				}

				cpu_img_resizer[scl] = new SIMD::ImageResizer(img_resize, param.max_image_size);
			}

			if ((int)param.pipeline > 0)
			{
				if (!advanced_param.packet_detection)
				{
#ifdef USE_CUDA
					cu_img_scale[scl] = CUDA::Image_32f(
						img_resize.width,
						img_resize.height,
						roundUpMul(img_resize.width, REG_SIZE),
						addRoundUpMul(img_resize.width, cu_cnn[0]->getBlockSize().width),
						addRoundUpMul(img_resize.height, cu_cnn[0]->getBlockSize().height),
						ALIGN_DEF);
#endif
#ifdef USE_CL
					cl_img_scale[scl] = CL::Image_32f(
						cl_context,
						cl_queue,
						img_resize.width,
						img_resize.height,
						roundUpMul(img_resize.width, REG_SIZE),
						addRoundUpMul(img_resize.width, cl_cnn->getBlockSize().width),
						addRoundUpMul(img_resize.height, cl_cnn->getBlockSize().height),
						ALIGN_DEF);
#endif
				}
				else
				{
#ifdef USE_CUDA
					int offsetHost = pack[scl].y * pack_cu_img_scale.widthStepHost + pack[scl].x;
					int offsetDevice = pack[scl].y * pack_cu_img_scale.widthStepDevice + pack[scl].x;
					cu_img_scale[scl] = CUDA::Image_32f(
						img_resize.width,
						img_resize.height,
						1,
						pack_cu_img_scale.dataHost + offsetHost,
						pack_cu_img_scale.dataDevice,
						pack_cu_img_scale.widthStepHost,
						pack_cu_img_scale.widthStepDevice,
						offsetDevice);
#endif
#ifdef USE_CL
					int offsetHost = pack[scl].y * pack_cl_img_scale.widthStepHost + pack[scl].x;
					int offsetDevice = pack[scl].y * pack_cl_img_scale.widthStepDevice + pack[scl].x;
					cl_img_scale[scl] = CL::Image_32f(
						cl_context,
						cl_queue,
						img_resize.width,
						img_resize.height,
						1,
						pack_cl_img_scale.dataHost + offsetHost,
						pack_cl_img_scale.dataDevice,
						pack_cl_img_scale.widthStepHost,
						pack_cl_img_scale.widthStepDevice,
						offsetDevice);
#endif
				}
			}
		}

		num_scales = (int)scales.size();
		data_transfer_flag = new volatile long[scales.size()];

#ifdef PROFILE_DETECTOR
		stat.max_image = param.max_image_size;
		stat.max_scale = param.max_image_size * scales[0];
		stat.min_scale = param.max_image_size * scales[(int)scales.size() - 1];
		stat.num_scales = (int)scales.size();
#endif

		return 0;
	}
	int CNNDetector::InitCNNBuffers()
	{
		//allocate buffers for CNN
		Size init_size = param.max_image_size * scales[0];
		if (advanced_param.packet_detection)
		{
			init_size = pack_size;
		}

		if (param.pipeline != Pipeline::GPU)
		{
			cpu_cnn->AllocateMemory(init_size);
			cpu_cnn->setNumThreads(num_threads);
			shift_pattern = (int)cpu_cnn->getInputOutputRatio();

			cpu_response_map.resize(cpu_img_scale.size());
			if (!advanced_param.packet_detection)
			{
				for (int scl = 0; scl < (int)cpu_img_scale.size(); ++scl)
				{
					Size output_size = cpu_cnn->getOutputImgSize(cpu_img_scale[scl].getSize());
					cpu_response_map[scl] = SIMD::Image_32f(output_size.width, output_size.height, ALIGN_DEF, true);
				}
			}
			else
			{
				Size pack_output_size = cpu_cnn->getOutputImgSize();
				pack_cpu_response_map = SIMD::Image_32f(pack_output_size.width, pack_output_size.height, ALIGN_DEF, true);

				for (int scl = 0; scl < (int)cpu_img_scale.size(); ++scl)
				{
					Size output_size = cpu_cnn->getOutputImgSize(cpu_img_scale[scl].getSize());
					int offset = (pack[scl].y * pack_cpu_response_map.widthStep + pack[scl].x) / shift_pattern;
					cpu_response_map[scl] = SIMD::Image_32f(
						output_size.width,
						output_size.height,
						1,
						pack_cpu_response_map.data + offset,
						pack_cpu_response_map.widthStep);
				}
			}
		}

		if ((int)param.pipeline > 0)
		{
#ifdef USE_CUDA
			for (auto it = cu_cnn.begin(); it != cu_cnn.end(); ++it)
			{
				(*it)->AllocateMemory(init_size);
			}
			shift_pattern = (int)cu_cnn[0]->getInputOutputRatio();

			cu_response_map.resize(cu_img_scale.size());
			if (!advanced_param.packet_detection)
			{
				for (int scl = 0; scl < (int)cu_img_scale.size(); ++scl)
				{
					Size output_size = cu_cnn[0]->getOutputImgSize(cu_img_scale[scl].getSize());
					cu_response_map[scl] = CUDA::Image_32f_pinned(output_size.width, output_size.height, 1, true, ALIGN_DEF);
				}
			}
			else
			{
				Size output_size = cu_cnn[0]->getOutputImgSize();
				pack_cu_response_map = CUDA::Image_32f_pinned(output_size.width, output_size.height, 1, true, ALIGN_DEF);

				for (int scl = 0; scl < (int)cu_img_scale.size(); ++scl)
				{
					Size output_size = cu_cnn[0]->getOutputImgSize(cu_img_scale[scl].getSize());
					int offsetHost = (pack[scl].y * pack_cu_response_map.widthStepHost + pack[scl].x) / shift_pattern;
					int offsetDevice = (pack[scl].y * pack_cu_response_map.widthStepDevice + pack[scl].x) / shift_pattern;
					cu_response_map[scl] = CUDA::Image_32f_pinned(
						output_size.width,
						output_size.height,
						1,
						pack_cu_response_map.dataHost + offsetHost,
						pack_cu_response_map.dataDevice,
						pack_cu_response_map.widthStepHost,
						pack_cu_response_map.widthStepDevice,
						offsetDevice);
				}
			}
#endif
#ifdef USE_CL
			cl_cnn->AllocateMemory(init_size);
			shift_pattern = (int)cl_cnn->getInputOutputRatio();

			cl_response_map.resize(cl_img_scale.size());
			if (!advanced_param.packet_detection)
			{
				for (int scl = 0; scl < (int)cl_img_scale.size(); ++scl)
				{
					Size output_size = cl_cnn->getOutputImgSize(cl_img_scale[scl].getSize());
					cl_response_map[scl] = CL::Image_32f(cl_context, cl_queue, output_size.width, output_size.height, 1, true, ALIGN_DEF);
				}
			}
			else
			{
				Size output_size = cl_cnn->getOutputImgSize();
				pack_cl_response_map = CL::Image_32f(cl_context, cl_queue, output_size.width, output_size.height, 1, true, ALIGN_DEF);

				for (int scl = 0; scl < (int)cl_img_scale.size(); ++scl)
				{
					Size output_size = cl_cnn->getOutputImgSize(cl_img_scale[scl].getSize());
					int offsetHost = (pack[scl].y * pack_cl_response_map.widthStepHost + pack[scl].x) / shift_pattern;
					int offsetDevice = (pack[scl].y * pack_cl_response_map.widthStepDevice + pack[scl].x) / shift_pattern;
					cl_response_map[scl] = CL::Image_32f(
						cl_context,
						cl_queue,
						output_size.width,
						output_size.height,
						1,
						pack_cl_response_map.dataHost + offsetHost,
						pack_cl_response_map.dataDevice,
						pack_cl_response_map.widthStepHost,
						pack_cl_response_map.widthStepDevice,
						offsetDevice);
				}
			}
#endif
		}

		return 0;
	}
	int CNNDetector::InitCNNCheck()
	{
		if (advanced_param.detect_mode != DetectMode::disable)
		{
			//init cpu_cnn_check1
			for (int i = 0; i < num_threads; ++i)
			{
				cpu_cnn_check1.push_back(new SIMD::ConvNeuralNetwork());
				//cpu_cnn_check1.push_back(new Legacy::ConvNeuralNetwork());
				//Legacy::ConvNeuralNetwork CNN_OLD(advanced_param.path_model[1]);

				auto it = cpu_cnn_check1.end() - 1;

				(*it)->Init(advanced_param.path_model[1], advanced_param.index_output[1], hGrd);
				//(*it)->Init(&CNN_OLD);
				if ((*it)->isEmpty())
				{
					delete (*it);
					cpu_cnn_check1.clear();
					return -1;
				}

				if (i == 0)
				{
					pattern_size_cd = (*it)->getMinInputImgSize();
					ext_pattern_size_cd = pattern_size_cd + ext_pattern_offset;
				}

				(*it)->AllocateMemory(ext_pattern_size_cd);
				(*it)->setNumThreads(1);
			}

			//init cpu_cnn_check2
			for (int i = 0; i < num_threads; ++i)
			{
				cpu_cnn_check2.push_back(new SIMD::ConvNeuralNetwork());
				//cpu_cnn_check2.push_back(new Legacy::ConvNeuralNetwork());
				//Legacy::ConvNeuralNetwork CNN_OLD(advanced_param.path_model[2]);
				auto it = cpu_cnn_check2.end() - 1;

				(*it)->Init(advanced_param.path_model[2], advanced_param.index_output[2], hGrd);
				//(*it)->Init(&CNN_OLD);
				if ((*it)->isEmpty())
				{
					delete (*it);
					cpu_cnn_check2.clear();
					return -1;
				}

				(*it)->AllocateMemory(ext_pattern_size_cd);
				(*it)->setNumThreads(1);
			}

			//init cpu_cnn_fa
			if (advanced_param.facial_analysis)
			{
				for (int i = 0; i < num_threads; ++i)
				{
					cpu_cnn_fa.push_back(new SIMD::ConvNeuralNetwork());
					auto it = cpu_cnn_fa.end() - 1;
					(*it)->Init(advanced_param.path_model[3], advanced_param.index_output[3], hGrd);
					if ((*it)->isEmpty())
					{
						delete (*it);
						cpu_cnn_fa.clear();
						break;
					}

					(*it)->AllocateMemory((*it)->getMinInputImgSize());
					(*it)->setNumThreads(1);
				}
			}

			if (cpu_cnn_check1[0]->getMinInputImgSize().width != cpu_cnn_check2[0]->getMinInputImgSize().width ||
				cpu_cnn_check1[0]->getMinInputImgSize().height != cpu_cnn_check2[0]->getMinInputImgSize().height)
			{
				printf("[CNNDetector] This configuration cnn models is not supported!\n");
				return -1;
			}

			//init cpu_img_check_buffer
			const int width = ext_pattern_size_cd.width;
			const int height = ext_pattern_size_cd.height;

			cpu_img_check_resizer.resize(num_threads);
			cpu_img_check_resize_32f.resize(num_threads);
			cpu_img_check_8u.resize(num_threads);
			cpu_img_check_32f.resize(num_threads);
			for (int i = 0; i < num_threads; ++i)
			{
				cpu_img_check_resizer[i] = new SIMD::ImageResizer(Size(width + x_pattern_offset, height), param.max_image_size);
				cpu_img_check_resize_32f[i] = SIMD::Image_32f(width + x_pattern_offset, height, ALIGN_DEF, true);
				cpu_img_check_8u[i] = SIMD::Image_8u(width + x_pattern_offset, height, ALIGN_DEF, true);
				cpu_img_check_32f[i] = SIMD::Image_32f(width, height, ALIGN_DEF, true);
			}

			if (advanced_param.uniform_noise)
			{
				cpu_img_urnd = SIMD::TmpImage<char>(width + x_pattern_offset, height, ALIGN_DEF);
				const float rnd = -20.f / (float)RAND_MAX;
				char* ptr = cpu_img_urnd.data;
				for (int k = 0; k < cpu_img_urnd.widthStep * cpu_img_urnd.height; ++k)
				{
					*ptr++ = char(10.f + rnd * (float)std::rand());
				}
			}

#ifdef CHECK_TEST
			Legacy::ConvNeuralNetwork* old_cnn_check1 = NULL;
			old_cnn_check1 = new Legacy::ConvNeuralNetwork(MODELSPATH"cnn4face2_old.bin");
			if (old_cnn_check1->isEmpty())
			{
				delete old_cnn_check1;
				old_cnn_check1 = NULL;
				return -1;
			}

			Legacy::ConvNeuralNetwork* old_cnn_check2 = NULL;
			old_cnn_check2 = new Legacy::ConvNeuralNetwork(MODELSPATH"cnn4face3_old.bin");
			if (old_cnn_check2->isEmpty())
			{
				delete old_cnn_check2;
				old_cnn_check2 = NULL;
				return -1;
			}

			for (int i = 0; i < num_threads; ++i)
			{
				cpu_cnn_check1[i]->setCNNRef(old_cnn_check1);
				cpu_cnn_check2[i]->setCNNRef(old_cnn_check2);
			}

#	ifdef USE_CUDA
			if (advanced_param.packet_detection && param.pipeline == Pipeline::GPU)
			{
				cu_cnn_check->setCNNRef(old_cnn_check1);
			}
#	endif
#endif
		}
		else
		{
			ext_pattern_size_cd = pattern_size;
		}

		return 0;
	}
	int CNNDetector::InitBlurFilters()
	{
		//init blur filter	
		const float kernel_matrix_col[4] = { 0.106450774f, 0.786570728f, 0.106450774f, 0.f };
		const float kernel_matrix_row[4] = { 0.106450774f, 0.786570728f, 0.106450774f, 0.f };

		row_filter3_kernel = SIMD::Array_32f(8, ALIGN_DEF);
		col_filter3_kernel = SIMD::Array_32f(8, ALIGN_DEF);

		for (int i = 0; i < 4; ++i)
		{
			row_filter3_kernel[i] = kernel_matrix_row[i];
			row_filter3_kernel[i + 4] = kernel_matrix_row[i];

			col_filter3_kernel[i] = kernel_matrix_col[i];
			col_filter3_kernel[i + 4] = kernel_matrix_col[i];
		}

		return 0;
	}

	int CNNDetector::Init()
	{
		printf("[CNNDetector] Initializing with max image size (%d, %d)!\n", param.max_image_size.width, param.max_image_size.height);

#	ifndef USE_CNTK_MODELS
		advanced_param.path_model[0] = DUMP::get_serialized_model("cnn4face1_new.bin");
		advanced_param.path_model[1] = DUMP::get_serialized_model("cnn4face2_new.bin");
		advanced_param.path_model[2] = DUMP::get_serialized_model("cnn4face3_new.bin");
		advanced_param.facial_analysis = false;
#	else
		advanced_param.path_model[0] = DUMP::get_serialized_model("cnn4face1_cntk.bin");
		advanced_param.path_model[1] = DUMP::get_serialized_model("cnn4face2_cntk.bin");
		advanced_param.path_model[2] = DUMP::get_serialized_model("cnn4face3_cntk.bin");
		advanced_param.path_model[3] = DUMP::get_serialized_model("cnn4landmarks_cntk.bin");
#	endif
		
		if (advanced_param.detect_precision == DetectPrecision::def)
		{
			advanced_param.gray_image_only = true;
			advanced_param.packet_detection = false;
			advanced_param.type_check = 1;
			advanced_param.equalize = true;
			advanced_param.reflection = true;
			advanced_param.adapt_min_neighbors = true;
			advanced_param.double_check = true;
			advanced_param.drop_detect = false;
			advanced_param.min_num_detect = 1;
			advanced_param.blur = false;
			advanced_param.uniform_noise = false;
			advanced_param.merger_detect = true;
		}

		switch (advanced_param.detect_precision)
		{
		case DetectPrecision::low:
			advanced_param.type_check = 0;
			advanced_param.drop_detect = false;
			//advanced_param.packet_detection = true;
			break;

		case DetectPrecision::normal:
			advanced_param.type_check = 1;
			break;

		case DetectPrecision::high:
			advanced_param.type_check = 2;
			break;

		case DetectPrecision::ultra:
			advanced_param.type_check = 2;
			advanced_param.min_num_detect = 2;
			break;
		}

		if (advanced_param.detect_mode == DetectMode::disable || advanced_param.facial_analysis)
		{
			advanced_param.drop_detect = false;
		}

		int err = 0;
		if ((err = InitDevice())      < 0) goto exit;
		if ((err = InitCNN())	 	  < 0) goto exit;
		if ((err = InitScales())      < 0) goto exit;
		if ((err = InitPaketCNN())    < 0) goto exit;
		if ((err = InitBuffers())     < 0) goto exit;
		if ((err = InitCNNBuffers())  < 0) goto exit;
		if ((err = InitCNNCheck())    < 0) goto exit;
		if ((err = InitBlurFilters()) < 0) goto exit;

		exit:

		if (err < 0)
		{
			printf("[CNNDetector] Load failed!\n");
			Clear();
		}

		return err;
	}
	void CNNDetector::Clear()
	{
		if (isEmpty()) return;

		//clear img_buffer
		if (param.pipeline == Pipeline::CPU)
		{
			cpu_img_gray.clear();
		}
		else
		{
#ifdef USE_CUDA	
			cu_img_input.clear();
			cu_img_gray.clear();
#endif
#ifdef USE_CL
			cl_img_input.clear();
			cl_img_gray.clear();
			cl_img_temp.clear();
#endif
		}

		//clear packed
		if (advanced_param.packet_detection)
		{
			pack.clear();
			pack_size = Size(0, 0);

			if (param.pipeline != Pipeline::GPU)
			{
				pack_cpu_img_scale.clear();
				pack_cpu_response_map.clear();
			}

			if ((int)param.pipeline > 0)
			{
#ifdef USE_CUDA
				pack_cu_img_scale.clear();
				pack_cu_response_map.clear();

				delete cu_cnn_check;
				pack_cu_img_check_8u.clear();
				pack_cu_img_check_32f.clear();
				pack_cu_response_map_check.clear();
				pack_pos_check.clear();
				pack_img_check_size = Size(0, 0);
#endif
#ifdef USE_CL
				pack_cl_img_scale.clear();
				pack_cl_response_map.clear();
#endif
			}
		}

		//clear CNN
		if (scales.size() > 0)
		{
			delete cpu_cnn;

			if (param.pipeline != Pipeline::GPU)
			{
				cpu_response_map.clear();
			}

			if ((int)param.pipeline > 0)
			{
#ifdef USE_CUDA
				for (auto it = cu_cnn.begin(); it != cu_cnn.end(); ++it)
				{
					delete (*it);
					(*it) = NULL;
				}
				cu_cnn.clear();

				cu_response_map.clear();
#endif
#ifdef USE_CL
				delete cl_cnn;

				cl_response_map.clear();
#endif
			}

			for (int i = 0; i < (int)cpu_cnn_check1.size(); ++i)
			{
				delete cpu_cnn_check1[i];
			}
			cpu_cnn_check1.clear();

			for (int i = 0; i < (int)cpu_cnn_check2.size(); ++i)
			{
				delete cpu_cnn_check2[i];
			}
			cpu_cnn_check2.clear();

			for (int i = 0; i < (int)cpu_cnn_fa.size(); ++i)
			{
				delete cpu_cnn_fa[i];
			}
			cpu_cnn_fa.clear();
		}

		scales.clear();

		//clear fast image resizing
		if (param.pipeline != Pipeline::GPU)
		{
			for (int i = 0; i < (int)cpu_img_scale.size(); ++i)
			{
				delete cpu_img_resizer[i];
			}
			cpu_img_resizer.clear();
		}

		cpu_img_scale.clear();
#ifdef USE_CUDA
		cu_img_scale.clear();
#endif
#ifdef USE_CL
		cl_img_scale.clear();
#endif

		if (cpu_input_img_resizer != nullptr)
		{
			delete cpu_input_img_resizer;
			cpu_input_img_resizer = nullptr;
		}
		cpu_input_img.clear();

		cpu_detect_rect.clear();
		gpu_detect_rect.clear();

		pattern_size = Size(0, 0);
		pattern_size_cd = Size(0, 0);
		ext_pattern_size_cd = Size(0, 0);
		shift_pattern = 0;

		num_scales = 0;

		delete[] data_transfer_flag;

		//clear cpu_img_check_buffer
		for (int i = 0; i < (int)cpu_img_check_resizer.size(); ++i)
		{
			delete cpu_img_check_resizer[i];
		}
		cpu_img_check_resizer.clear();

		cpu_img_check_resize_32f.clear();

		cpu_img_check_8u.clear();

		cpu_img_check_32f.clear();

		cpu_img_urnd.clear();

		col_filter3_kernel.clear();
		row_filter3_kernel.clear();

#if defined(_MSC_VER)
		CloseHandle(check_detect_event);
#endif

#ifdef PROFILE_DETECTOR
		delete cpu_timer_detector;
		delete cpu_timer_cnn;
		delete cpu_timer_check1;
		delete cpu_timer_check2;
#	ifdef USE_CUDA
		delete cu_timer;
#	endif
#	ifdef USE_CL
		delete cl_timer;
#	endif
#endif

		if ((int)param.pipeline > 0)
		{
#ifdef USE_CUDA
			cuERR(cudaStreamDestroy(cu_stream0));
			cuERR(cudaEventDestroy(cu_event_img_gray));
			cuERR(cudaEventDestroy(cu_event_calc_cnn));

			for (int i = 0; i < advanced_param.concurrent_kernels; ++i)
			{
				cuERR(cudaStreamDestroy(cu_stream[i]));
				cuERR(cudaEventDestroy(cu_event_response_map[i]));
			}
			scl_id.clear();
			cu_stream.clear();
			cu_event_response_map.clear();

			CUDA::ReleaseDevice();
#endif
#ifdef USE_CL
			if (cl_event_img_gray != NULL)
			{
				clERR(clReleaseEvent(cl_event_img_gray));
				cl_event_img_gray = NULL;
			}
			if (cl_event_response_map != NULL)
			{
				clERR(clReleaseEvent(cl_event_response_map));
				cl_event_response_map = NULL;
			}

			CL::ImageConverter::destroy_cl_program();
			CL::ImageResizer::destroy_cl_program();
			CL::ReleaseDevice(cl_device, cl_context, cl_queue);
#endif
		}
	}
	bool CNNDetector::isEmpty() const
	{
		if (advanced_param.detect_mode == DetectMode::disable)
		{
			return scales.size() == 0 || cpu_cnn == NULL
#ifdef USE_CUDA
				&& cu_cnn.size() == 0
#endif
#ifdef USE_CL
				&& cl_cnn == NULL
#endif
				;
		}
		else
		{
			return scales.size() == 0 || cpu_cnn == NULL
#ifdef USE_CUDA
				&& cu_cnn.size() == 0
#endif
#ifdef USE_CL
				&& cl_cnn == NULL
#endif	
				|| cpu_cnn_check1.size() == 0 || cpu_cnn_check2.size() == 0;
		}
	}

	int CNNDetector::PacketReallocate(Size size)
	{
		float area_ratio_max = packing2D.packing(&pack, &pack_size, scales, size, int(1.f + scales[0]) * size.width);
		for (int i = 1; i < MIN(4, (int)scales.size()); ++i)
		{
			std::vector<Rect> pack_temp;
			Size pack_size_temp;
			float area_ratio = packing2D.packing(&pack_temp, &pack_size_temp, scales, size, int(1.f + scales[i]) * size.width);
			if (area_ratio > area_ratio_max)
			{
				pack.clear();
				pack = pack_temp;
				pack_size = pack_size_temp;
				area_ratio_max = area_ratio;
			}
		}

		if (pack.size() == 0) return -1;

		if (param.pipeline != Pipeline::GPU)
		{
			pack_cpu_img_scale.setSize(pack_size);
		}

		if ((int)param.pipeline > 0)
		{
#ifdef USE_CUDA
			pack_cu_img_scale.setSize(pack_size);
#endif
#ifdef USE_CL
			pack_cl_img_scale.setSize(pack_size);
#endif
		}

		for (int scl = 0; scl < (int)pack.size(); ++scl)
		{
			if (param.pipeline != Pipeline::GPU)
			{
				int offset = pack[scl].y * pack_cpu_img_scale.widthStep + pack[scl].x;
				cpu_img_scale[scl].data = pack_cpu_img_scale.data + offset;
			}

			if ((int)param.pipeline > 0)
			{
#ifdef USE_CUDA
				int offsetHost = pack[scl].y * pack_cu_img_scale.widthStepHost + pack[scl].x;
				int offsetDevice = pack[scl].y * pack_cu_img_scale.widthStepDevice + pack[scl].x;
				cu_img_scale[scl].dataHost = pack_cu_img_scale.dataHost + offsetHost;
				cu_img_scale[scl].offsetDevice = offsetDevice;
#endif
#ifdef USE_CL
				int offsetHost = pack[scl].y * pack_cl_img_scale.widthStepHost + pack[scl].x;
				int offsetDevice = pack[scl].y * pack_cl_img_scale.widthStepDevice + pack[scl].x;
				cl_img_scale[scl].dataHost = pack_cl_img_scale.dataHost + offsetHost;
				cl_img_scale[scl].offsetDevice = offsetDevice;
#endif
			}
		}

		for (int scl = 0; scl < (int)pack.size(); ++scl)
		{
			Size img_resize = size * scales[scl];

			if (param.pipeline != Pipeline::GPU)
			{
				Size output_size = cpu_cnn->getOutputImgSize(img_resize);
				int offset = (pack[scl].y * pack_cpu_response_map.widthStep + pack[scl].x) / shift_pattern;
				cpu_response_map[scl].setSize(output_size);
				cpu_response_map[scl].data = pack_cpu_response_map.data + offset;
			}

			if ((int)param.pipeline > 0)
			{
#ifdef USE_CUDA
				Size output_size = cu_cnn[0]->getOutputImgSize(img_resize);
				int offsetHost = (pack[scl].y * pack_cu_response_map.widthStepHost + pack[scl].x) / shift_pattern;
				int offsetDevice = (pack[scl].y * pack_cu_response_map.widthStepDevice + pack[scl].x) / shift_pattern;
				cu_response_map[scl].setSize(output_size);
				cu_response_map[scl].dataHost = pack_cu_response_map.dataHost + offsetHost;
				cu_response_map[scl].offsetDevice = offsetDevice;
#endif
#ifdef USE_CL
				Size output_size = cl_cnn->getOutputImgSize(img_resize);
				int offsetHost = (pack[scl].y * pack_cl_response_map.widthStepHost + pack[scl].x) / shift_pattern;
				int offsetDevice = (pack[scl].y * pack_cl_response_map.widthStepDevice + pack[scl].x) / shift_pattern;
				cl_response_map[scl].setSize(output_size);
				cl_response_map[scl].dataHost = pack_cl_response_map.dataHost + offsetHost;
				cl_response_map[scl].offsetDevice = offsetDevice;
#endif
			}
		}

		return 0;
	}
	void CNNDetector::PacketCPUCheckDetect()
	{
#ifdef USE_CUDA
		cuERR(cudaEventSynchronize(cu_event_img_gray));

		Size pack_pattern = Size(
			roundUpMul(ext_pattern_size_cd.width + x_pattern_offset, shift_pattern),
			roundUpMul(ext_pattern_size_cd.height, shift_pattern));

		volatile long detect_id = 0;
		pack_pos_check.clear();

		PROFILE_TIMER(cpu_timer_check2, stat.time_pack_check_proc,
		for(int scl = 0; scl < num_scales; ++scl)
		{
			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < cu_response_map[scl].height; ++j)
			{
				int index = 0;
				OMP_RUNTIME(index = omp_get_thread_num())

				float* resp_map_ptr = cu_response_map[scl].dataHost + j * cu_response_map[scl].widthStepHost;
				for (int i = 0; i < cu_response_map[scl].width; ++i)
				{
					if (*resp_map_ptr++ > advanced_param.treshold_1)
					{
						long detect_id_temp = InterlockedAdd(&detect_id, 2) - 2;
						if (detect_id_temp >= pack_max_num_img_check.size) break;

						int offset = (roundUp(detect_id_temp + 1, pack_max_num_img_check.cols) - 1) * pack_pattern.height * pack_cu_img_check_8u.widthStepHost
							+ (detect_id_temp % pack_max_num_img_check.cols) * pack_pattern.width;

						const Point point(i * shift_pattern, j * shift_pattern);

						OMP_PRAGMA(omp critical(add_pack_pos))
						{
							PROFILE_COUNTER_ADD(stat.num_pack_check_detect, 1.)
							pack_pos_check.push_back(PackPos(scl, i, j, detect_id_temp));
						}

						const float inv_scale = 1.f / scales[scl];
						int x = static_cast<int>((point.x - (ext_pattern_offset + x_pattern_offset) / 2) * inv_scale);
						int y = static_cast<int>((point.y - ext_pattern_offset / 2) * inv_scale);
						int width = static_cast<int>((pattern_size.width + ext_pattern_offset + x_pattern_offset) * inv_scale);
						int height = static_cast<int>((pattern_size.height + ext_pattern_offset) * inv_scale);

						for (int mod = 0; mod < 2; ++mod)
						{
							if (mod == 1)
							{
								const float k = 0.2f;
								x = static_cast<int>(x + width * k / 2.f);
								width = static_cast<int>(width - width * k);

								y = static_cast<int>(y + height * k / 2.f);
								height = static_cast<int>(height - height * k);

								offset += pack_pattern.width;
							}

							const int rx = MIN(MAX(x, 0), cpu_img_gray.width);
							const int ry = MIN(MAX(y, 0), cpu_img_gray.height);
							const int rcols = MIN(MAX(x + width, 0), cpu_img_gray.width) - rx;
							const int rrows = MIN(MAX(y + height, 0), cpu_img_gray.height) - ry;

							SIMD::Image_32f img_temp(
								rcols, 
								rrows, 
								cpu_img_gray.nChannel, 
								cpu_img_gray.data + ry * cpu_img_gray.widthStep + rx, 
								cpu_img_gray.widthStep);

							cpu_img_check_resizer[index]->FastImageResize(cpu_img_check_resize_32f[index], img_temp, (int)ImgResize::NearestNeighbor);

							SIMD::Image_8u img_8u_temp(
								cpu_img_check_resize_32f[index].width, 
								cpu_img_check_resize_32f[index].height, 
								1, 
								pack_cu_img_check_8u.dataHost + offset, 
								pack_cu_img_check_8u.widthStepHost);
							
							SIMD::ImageConverter::FloatToUChar(
								img_8u_temp, 
								cpu_img_check_resize_32f[index], 
								Rect(0, 0, cpu_img_check_resize_32f[index].width, cpu_img_check_resize_32f[index].height));

							if (advanced_param.equalize)
							{
								equalizeImage(img_8u_temp);
							}
						}
					}
				}
			}
		})

		if (detect_id > 0)
		{
			detect_id = MIN((int)detect_id, pack_max_num_img_check.size);

			pack_cu_img_check_8u.height = (roundUp(detect_id, pack_max_num_img_check.cols) - 1);
			if (pack_cu_img_check_8u.height > 0)
				pack_cu_img_check_8u.width = pack_max_num_img_check.cols;
			else
				pack_cu_img_check_8u.width = detect_id - pack_cu_img_check_8u.height * pack_max_num_img_check.cols;
			pack_cu_img_check_8u.width *= pack_pattern.width;
			pack_cu_img_check_8u.height = (pack_cu_img_check_8u.height + 1) * pack_pattern.height;

			pack_cu_img_check_32f.width = pack_cu_img_check_8u.width;
			pack_cu_img_check_32f.height = pack_cu_img_check_8u.height;

			//run on GPU
			PROFILE_TIMER(cu_timer, stat.time_pack_check_gpu_cnn,
			pack_cu_img_check_8u.updateDataDevice(true, cu_stream0, true);
			CUDA::ImageConverter::Img8uToImg32fGRAY_tex((CUDA::Image_32f_pinned*)&pack_cu_img_check_32f, &pack_cu_img_check_8u, cu_stream0);
			cu_cnn_check->Forward(&pack_cu_response_map_check, &pack_cu_img_check_32f, cu_stream0);
			pack_cu_response_map_check.updateDataHost(true, cu_stream0, true);
			cuERR(cudaStreamSynchronize(cu_stream0));)
			
			Size resp_map_size = cu_cnn_check->getOutputImgSize(ext_pattern_size_cd);
			const Size pack_resp_map_step = pack_pattern * (1.f / (float)shift_pattern);
			
			PROFILE_TIMER(cpu_timer_check2, stat.time_pack_erase_detect,
			for (auto it = pack_pos_check.begin(); it != pack_pos_check.end();)
			{
				const int pack_offset = (roundUp(it->pack_id + 1, pack_max_num_img_check.cols) - 1) * pack_resp_map_step.height * pack_cu_response_map_check.widthStepHost
					+ (it->pack_id % pack_max_num_img_check.cols) * pack_resp_map_step.width;

				float* resp_ptr = pack_cu_response_map_check.dataHost + pack_offset + (x_pattern_offset / shift_pattern);

				bool bl = false;
				for (int j = 0; j < resp_map_size.height; ++j)
				{
					for (int i = 0; i < resp_map_size.width; ++i)
					{
						if (*(resp_ptr + i) > advanced_param.treshold_2)
						{
							bl = true;
							break;
						}
					}

					for (int i = pack_resp_map_step.width; i < pack_resp_map_step.width + resp_map_size.width; ++i)
					{
						if (*(resp_ptr + i) > advanced_param.treshold_2)
						{
							bl = true;
							break;
						}
					}

					if (bl) break;
					resp_ptr += pack_cu_response_map_check.widthStepHost;
				}

				if (!bl)
				{
					cu_response_map[it->scl].dataHost[it->y * cu_response_map[it->scl].widthStepHost + it->x] = -10.f;
					it = pack_pos_check.erase(it);
					continue;
				}
				it++;
			})
		}
#endif
	}

	void CNNDetector::CPUCheckDetect(std::vector<Detection>& rect, const int rect_size, const Point& point, const float score0,
		const SIMD::Image_32f& img, const float scale, const int mod, const int pack_id)
	{
		int index = 0;
		OMP_RUNTIME(index = omp_get_thread_num())

		const float inv_scale = 1.f / scale;
		int x = static_cast<int>((point.x - (ext_pattern_offset + x_pattern_offset) / 2) * inv_scale);
		int y = static_cast<int>((point.y - ext_pattern_offset / 2) * inv_scale);
		int width = static_cast<int>((pattern_size.width + ext_pattern_offset + x_pattern_offset) * inv_scale);
		int height = static_cast<int>((pattern_size.height + ext_pattern_offset) * inv_scale);
		
		Rect new_rect(
			static_cast<int>((float)point.x * inv_scale),
			static_cast<int>((float)point.y * inv_scale),
			static_cast<int>((float)pattern_size.width * inv_scale),
			static_cast<int>((float)pattern_size.height * inv_scale));

		Size pack_pattern;
		int pack_offset_x = 0;
		int pack_offset_y = 0;

		int rx = 0, ry = 0;
		int rcols = 0, rrows = 0;

		PROFILE_TIMER(cpu_timer_check2, stat.time_check_proc,
		if (!advanced_param.packet_detection || pack_id < 0)
		{
			if (advanced_param.drop_detect && mod == 0)
			{
				bool bl = false;

				OMP_PRAGMA(omp critical(add_rect))
				{				
					for (auto it = rect.begin() + rect_size; it != rect.end(); ++it)
					{
						const float overlap = new_rect.overlap(it->rect);
						if (overlap > 0.5f)
						{
							PROFILE_COUNTER_INC(stat.num_check_hor_drop)
							rect.push_back(Detection(new_rect, it->score, scale, it->knn));
							bl = true;
							break;
						}
					}
				}

				if (bl) return;
			}

			if (mod == 1)
			{
				const float k = 0.15f;
				x = static_cast<int>(x + width * k / 2.f);
				width = static_cast<int>(width - width * k);
				y = static_cast<int>(y + height * k / 2.f);
				height = static_cast<int>(height - height * k);
			}

			if (mod == 2)
			{
				const float k = 0.15f;
				x = static_cast<int>(x - width * k / 2.f);
				width = static_cast<int>(width + width * k);
				y = static_cast<int>(y - height * k / 2.f);
				height = static_cast<int>(height + height * k);
			}

			rx = MIN(MAX(x, 0), img.width);
			ry = MIN(MAX(y, 0), img.height);
			rcols = MIN(MAX(x + width, 0), img.width) - rx;
			rrows = MIN(MAX(y + height, 0), img.height) - ry;

			SIMD::Image_32f img_temp(
				rcols, 
				rrows, 
				img.nChannel, 
				img.data + ry * img.widthStep + rx, 
				img.widthStep);

			cpu_img_check_resizer[index]->FastImageResize(cpu_img_check_resize_32f[index], img_temp, (int)ImgResize::NearestNeighbor);

			SIMD::ImageConverter::FloatToUChar(
				cpu_img_check_8u[index], 
				cpu_img_check_resize_32f[index], 
				Rect(0, 0, cpu_img_check_resize_32f[index].width, cpu_img_check_resize_32f[index].height));

			if (advanced_param.equalize)
			{
				SIMD::equalizeImage(cpu_img_check_8u[index]);
			}
		}
		else
		{
			CUDA_CODE({
				pack_pattern = Size(
				roundUpMul(ext_pattern_size_cd.width + x_pattern_offset, shift_pattern),
				roundUpMul(ext_pattern_size_cd.height, shift_pattern));

				pack_offset_y = (roundUp(pack_id + 1, pack_max_num_img_check.cols) - 1) * pack_pattern.height;
				pack_offset_x = pack_id % pack_max_num_img_check.cols * pack_pattern.width;
				if (mod == 1) pack_offset_x += pack_pattern.width;

				const int pack_offset = pack_offset_y * pack_cu_img_check_8u.widthStepHost + pack_offset_x;
				uchar_* pack_img_ptr = pack_cu_img_check_8u.dataHost + pack_offset;
				cpu_img_check_8u[index].copyData(cpu_img_check_8u[index].width, cpu_img_check_8u[index].height, pack_img_ptr, pack_cu_img_check_8u.widthStepHost);
			})
		})

		if (mod == 0) { 
			PROFILE_COUNTER_INC(stat.num_check_mod0) }
		else { 
			PROFILE_COUNTER_INC(stat.num_check_mod1) }

		float max_score1 = -10.f;
		float max_score2 = -10.f;
		int knn_count1 = 0;
		int knn_count2 = 0;
		bool flag_object = false;

		const int min_neighbors = (advanced_param.adapt_min_neighbors && 
								   scale > 0.7f &&
								   param.min_neighbors > 1) ? 
								   param.min_neighbors - 1 : param.min_neighbors;

		for (int icx = -1; icx <= 1; icx += 2)
		{
			SIMD::Image_32f response_map;
			if (icx == -1)
			{
				PROFILE_COUNTER_INC(stat.num_check_call_cnn2)
				PROFILE_TIMER(cpu_timer_check2, stat.time_check_cpu_cnn2,
				if (!advanced_param.packet_detection || pack_id < 0)
				{
					SIMD::ImageConverter::UCharToFloat(cpu_img_check_32f[index], cpu_img_check_8u[index], x_pattern_offset);
					cpu_cnn_check1[index]->Forward(response_map, cpu_img_check_32f[index]);
				}
				else
				{
					CUDA_CODE({
						const Size resp_size = cpu_cnn_check1[index]->getOutputImgSize();
						const int pack_offset = pack_offset_y * pack_cu_response_map_check.widthStepHost + pack_offset_x;
						float* pack_resp_ptr = pack_cu_response_map_check.dataHost + (pack_offset + x_pattern_offset) / shift_pattern;
						response_map = SIMD::Image_32f(resp_size.width, resp_size.height, 1, pack_resp_ptr, pack_cu_response_map_check.widthStepHost);
					})
				})
			}
			else
			{
				PROFILE_COUNTER_INC(stat.num_check_call_cnn3)
				PROFILE_TIMER(cpu_timer_check2, stat.time_check_cpu_cnn3,
				if (!advanced_param.uniform_noise)
				{
					if (advanced_param.reflection)
					{
						SIMD::ImageConverter::UCharToFloat_inv(cpu_img_check_32f[index], cpu_img_check_8u[index]);
					}
					else
						SIMD::ImageConverter::UCharToFloat(cpu_img_check_32f[index], cpu_img_check_8u[index]);
				}
				else
				{
					SIMD::ImageConverter::UCharToFloat_add_rnd(cpu_img_check_32f[index], cpu_img_check_8u[index], cpu_img_urnd);
				}

				cpu_cnn_check2[index]->Forward(response_map, cpu_img_check_32f[index]);)
			}

			if (response_map.isEmpty()) break;

			PROFILE_TIMER(cpu_timer_check2, stat.time_check_find_detections,
			for (int j = 0; j < response_map.height; ++j)
			{
				float* resp_map_ptr = response_map.data + j * response_map.widthStep;
				for (int i = 0; i < response_map.width; ++i)
				{
					const float d = *resp_map_ptr++;

					if (icx == -1)
					{
						max_score1 = MAX(max_score1, d);
						if (d > advanced_param.treshold_2)
						{
							knn_count1++;
							PROFILE_COUNTER_INC(stat.num_detections_stage2)
						}
					}
					else
					{
						max_score2 = MAX(max_score2, d);
						if (d > advanced_param.treshold_3)
						{
							knn_count2++;
							PROFILE_COUNTER_INC(stat.num_detections_stage3)
						}
					}
				}
			})

#if 0
				if (1 /*&& max_score1 > 0*/)
				{
					if (max_score1 > advanced_param.treshold_2)
					{
						printf("max_score1 = %f, max_score2 = %f\n", max_score1, max_score2);
						cv::Mat temp_draw = cv::Mat::zeros(cpu_img_check_32f[index].height, cpu_img_check_32f[index].width, CV_8U);
						for (int ty = 0; ty < cpu_img_check_32f[index].height; ++ty)
							for (int tx = 0; tx < cpu_img_check_32f[index].width; ++tx)
							{
								temp_draw.data[ty*cpu_img_check_32f[index].width + tx] = (uchar)cpu_img_check_32f[index].data[ty*cpu_img_check_32f[index].widthStep + tx];
							}

						cv::imshow(std::to_string(response_map.height), temp_draw);
						//cv::imwrite("noEQ_img_check_max_score1=" + std::to_string(max_score1) + "_max_score2=" + std::to_string(max_score2) + ".bmp", temp_draw);
						cv::waitKey(0);
					}
				}
#endif

			if (advanced_param.type_check == 0 && knn_count1 >= min_neighbors) break;
			if (advanced_param.type_check > 0  && knn_count1 <= 0) break;
			if (advanced_param.type_check == 2 && knn_count1 < min_neighbors) break;
		}

		switch (advanced_param.type_check)
		{
		case 0: 
			if (knn_count1 >= min_neighbors || knn_count2 >= min_neighbors) 
				flag_object = true;
			break;

		case 1: 
			if ((knn_count1 >= min_neighbors && knn_count2 > 0) || (knn_count1 > 0 && knn_count2 >= min_neighbors)) 
				flag_object = true;
			break;

		case 2: 
			if (knn_count1 >= min_neighbors && knn_count2 >= min_neighbors) 
				flag_object = true;
			break;
		}

		if (flag_object)
		{
			PROFILE_COUNTER_INC(stat.num_check_add_rect)

			FacialData fd;
			if (cpu_cnn_fa.size() != 0)
			{
				SIMD::Image_32f response_map;
				cpu_cnn_fa[index]->Forward(response_map, cpu_img_check_resize_32f[index]);

				int zero_landmarks = 0;
				for (int t = 0; t < 5; ++t)
				{
					fd.landmarks[t].x = int(response_map.data[2 * t * response_map.widthStep] * float(rcols));
					fd.landmarks[t].y = int(response_map.data[(2 * t + 1) * response_map.widthStep] * float(rrows));

					if (fd.landmarks[t].x < (rcols >> 1) && fd.landmarks[t].y < (rrows >> 1))
						zero_landmarks++;

					fd.landmarks[t].x += rx;
					fd.landmarks[t].y += ry;
				}
				if (zero_landmarks == 5) return;

				fd.gender = int(response_map.data[10 * response_map.widthStep] + 0.5f);
				fd.smile = int(response_map.data[11 * response_map.widthStep] + 0.5f);
				fd.glasses = int(response_map.data[12 * response_map.widthStep] + 0.5f);
			}

			OMP_PRAGMA(omp critical(add_rect))
			{
				const float score = (score0) + float(knn_count1) * MAX(-1.7159f, max_score1) + float(knn_count2) * MAX(-1.7159f, max_score2);
				rect.push_back(Detection(new_rect, score/*MIN(score0, MIN(max_score1, max_score2))*/, scale, MIN(knn_count1, knn_count2)));
				//rect.push_back(Detection(Rect(rx, ry, rcols, rrows), MAX(max_score1, max_score2), scale, MIN(knn_count1, knn_count2)));

				if (cpu_cnn_fa.size() != 0)
				{
					rect.rbegin()->facial_data.push_back(fd);
				}

#if 0
				if (0)
				{
					//static bool bl = true;
					//static int cmp_id = -1;

					//if (bl)
					//{
					//	std::ifstream id_txt("id_txt.txt");
					//	std::string line;
					//	std::getline(id_txt, line);
					//	std::stringstream ss(line);
					//	ss >> cmp_id;
					//	id_txt.close();
					//	printf("cmp_id = %d\n", cmp_id);
					//	bl = false;
					//}

					static int count = 0;
					cv::Mat temp_draw = cv::Mat::zeros(cpu_img_check_32f[index]->height, cpu_img_check_32f[index]->width, CV_8U);
					for (int ty = 0; ty < cpu_img_check_32f[index]->height; ++ty)
						for (int tx = 0; tx < cpu_img_check_32f[index]->width; ++tx)
						{
							temp_draw.data[ty*cpu_img_check_32f[index]->width + tx] = (uchar)(*cpu_img_check_32f[index]).data[ty*cpu_img_check_32f[index]->widthStep + tx];
						}

					cv::imshow("1", temp_draw);
					//cv::imwrite("P:/RTSD_train/train/tt_sign/" + std::to_string(cmp_id) + "/" + std::to_string(cmp_id) + "_" + std::to_string(std::rand() + std::rand()) + ".png", temp_draw);
					cv::waitKey(1);
					//count++;
					//if (count > 25000)
					//{
					//	printf("STOP\n");
					//	(*cpu_img_check_32f[-1]).data[-1] /= 0.;
					//}
				}
#endif
			}
		}
		else
		{
			if (advanced_param.double_check && mod == 0)
			{
				CPUCheckDetect(rect, rect_size, point, score0, img, scale, 1, pack_id);
			}

			//if (advanced_param.double_check && mod == 1)
			//{
			//	CPUCheckDetect(rect, rect_size, point, img, scale, 2, pack_id);
			//}
		}
	}

	bool CNNDetector::DropDetection(Rect& new_rect, std::vector<Detection>& detect_rect_in, std::vector<Detection>& detect_rect_out, float scale)
	{	
		Rect exp_rect = new_rect * 0.7f;
		for (auto it = detect_rect_in.begin(); it != detect_rect_in.end(); ++it)
		{
			const float overlap = new_rect.overlap(it->rect);
			if (overlap == 0.f) continue;
			if (overlap > 0.5f)
			{
				detect_rect_out.push_back(Detection(new_rect, it->score, scale, it->knn));
				return true;
			}
			else
			{
				if (it->rect.intersects(exp_rect) == exp_rect.area())
				{
					return true;
				}
			}
		}
		return false;
	}
	void CNNDetector::RunCheckDetect(const int scl, const int device)
	{
		OMP_PRAGMA(omp critical(check_rect))
		{
			std::vector<Detection>* detect_rect;

#if defined(USE_CUDA) || defined(USE_CL)
			if (device > 0)
			{
				detect_rect = &gpu_detect_rect;
			}
			else
#endif
			{
				detect_rect = &cpu_detect_rect;
			}

			std::vector<std::pair<Point, float>> detect_point;
			detect_point.reserve(20);

			const float scale = scales[scl];
			const float inv_scale = 1.f / scale;

			PROFILE_TIMER(cpu_timer_check1, stat.time_check_ver_drop,
			GPU_ONLY(
			if (device > 0)
			{
				PROFILE_COUNTER_INC(stat.num_call_check_gpu)

				CUDA_CODE(
				PROFILE_COUNTER_ADD(stat.num_responses_stage1, cu_response_map[scl].size)
				for (int j = 0; j < cu_response_map[scl].height; ++j)
				{
					float* resp_map_ptr = cu_response_map[scl].dataHost + j * cu_response_map[scl].widthStepHost;
					for (int i = 0; i < cu_response_map[scl].width; ++i)
					{)
				
				CL_CODE(
				PROFILE_COUNTER_ADD(stat.num_responses_stage1, cl_response_map[scl].size)
				for (int j = 0; j < cl_response_map[scl].height; ++j)
				{
					float* resp_map_ptr = cl_response_map[scl].dataHost + j * cl_response_map[scl].widthStepHost;
					for (int i = 0; i < cl_response_map[scl].width; ++i)
					{)
						const float score = *(resp_map_ptr++);
						if (score > advanced_param.treshold_1)
						{
							const Point point(i * shift_pattern, j * shift_pattern);

							if (advanced_param.drop_detect)
							{
								Rect new_rect(
									static_cast<int>((float)point.x * inv_scale),
									static_cast<int>((float)point.y * inv_scale),
									static_cast<int>((float)pattern_size.width * inv_scale),
									static_cast<int>((float)pattern_size.height * inv_scale));

								if (DropDetection(new_rect, cpu_detect_rect,* detect_rect, scale) ||
									DropDetection(new_rect,* detect_rect,* detect_rect, scale))
								{
									PROFILE_COUNTER_ADD(stat.num_check_ver_drop, 1.)
										continue;
								}
							}

							detect_point.push_back(std::pair<Point, float>(point, score));
							PROFILE_COUNTER_ADD(stat.num_detections_stage1, 1.)
						}
					}
				}
			}
			else)	
			{
				PROFILE_COUNTER_INC(stat.num_call_check_cpu)
				PROFILE_COUNTER_ADD(stat.num_responses_stage1, cpu_response_map[scl].size)

				for (int j = 0; j < cpu_response_map[scl].height; ++j)
				{
					float* resp_map_ptr = cpu_response_map[scl].data + j * cpu_response_map[scl].widthStep;
					for (int i = 0; i < cpu_response_map[scl].width; ++i)
					{
						const float score = *(resp_map_ptr++);
						if (score > advanced_param.treshold_1)
						{
							const Point point(i * shift_pattern, j * shift_pattern);
							
							if (advanced_param.drop_detect)
							{
								Rect new_rect(
									static_cast<int>((float)point.x * inv_scale),
									static_cast<int>((float)point.y * inv_scale),
									static_cast<int>((float)pattern_size.width * inv_scale),
									static_cast<int>((float)pattern_size.height * inv_scale));

								if (DropDetection(new_rect,* detect_rect,* detect_rect, scale))
								{
									PROFILE_COUNTER_ADD(stat.num_check_ver_drop, 1.)
										continue;
								}
							}

							detect_point.push_back(std::pair<Point, float>(point, score));
							PROFILE_COUNTER_ADD(stat.num_detections_stage1, 1.)
						}
					}
				}
			})

			PROFILE_TIMER(cpu_timer_check1, stat.time_check,
			if (advanced_param.detect_mode != DetectMode::disable && detect_point.size() > 0)
			{
				if (advanced_param.uniform_noise)
				{
					const float rnd = -20.f / (float)RAND_MAX;
					char* ptr = cpu_img_urnd.data;
					for (int k = 0; k < cpu_img_urnd.widthStep * cpu_img_urnd.height; ++k)
					{
						*ptr++ = char(10.f + rnd * (float)std::rand());
					}
				}

				const int detect_rect_size = static_cast<const int>(detect_rect->size());

				int num_trd = num_threads;
				if (param.pipeline != Pipeline::GPU && advanced_param.detect_mode == DetectMode::async)
				{
					num_trd = 1;
				}

				OMP_PRAGMA(omp parallel for num_threads(num_trd) schedule(static))
				for (int p = 0; p < (int)detect_point.size(); ++p)
				{
					int pack_id = -1;
					if (advanced_param.packet_detection)
					{
						CUDA_CODE({
							for (auto it = pack_pos_check.begin(); it != pack_pos_check.end(); ++it)
							{
								if (it->scl == scl &&
									it->x == detect_point[p].first.x / shift_pattern &&
									it->y == detect_point[p].first.y / shift_pattern)
								{
									pack_id = it->pack_id;
									break;
								}
							}
						})
					}

					CPUCheckDetect(*detect_rect, detect_rect_size, detect_point[p].first, detect_point[p].second, cpu_img_gray, scale, 0, pack_id);
				}
			}
			else
			{
				for (int p = 0; p < (int)detect_point.size(); ++p)
				{
					detect_rect->push_back(Detection(
						int((float)detect_point[p].first.x * inv_scale),
						int((float)detect_point[p].first.y * inv_scale),
						int((float)pattern_size.width * inv_scale), 
						int((float)pattern_size.height * inv_scale), 
						detect_point[p].second,
						scale, 
						0));
				}
			})

			detect_point.clear();
		}
	}
	void CNNDetector::RunCheckDetectAsync()
	{
		if ((int)param.pipeline > 0)
		{
#ifdef USE_CUDA
			cuERR(cudaEventSynchronize(cu_event_img_gray));
#endif		
#ifdef USE_CL
			clERR(clWaitForEvents(1, &cl_event_img_gray));
#endif
		}

		const int scl_max = num_scales - 1;

		int inv_ord = 0;
		if (param.pipeline == Pipeline::GPU_CPU)
		{
			inv_ord = scl_max;
		}

		bool bl_exit;
		do
		{
#if defined(_MSC_VER)
			WaitForSingleObject(check_detect_event, INFINITE);
			ResetEvent(check_detect_event);
#endif

			bl_exit = true;
			for (int i_scl = scl_max; i_scl >= 0; --i_scl)
			{
				const int scl = abs(inv_ord - i_scl);
				const long sync_flag = AtomicCompareExchangeSwap(&data_transfer_flag[scl], 3, 2);

				if (sync_flag == 2)
				{
					RunCheckDetect(scl, (int)param.pipeline);
				}
				else
				{
					if (sync_flag != 3) bl_exit = false;
				}
			}
		} while (!bl_exit);

		if (param.pipeline == Pipeline::GPU)
		{
			std::reverse(gpu_detect_rect.begin(), gpu_detect_rect.end());
		}
	}

	void CNNDetector::RunCPUDetect()
	{
		if (param.pipeline == Pipeline::GPU_CPU)
		{
#ifdef USE_CUDA	
			cuERR(cudaEventSynchronize(cu_event_img_gray));
#endif
#ifdef USE_CL
			clERR(clWaitForEvents(1, &cl_event_img_gray));
#endif
		}

		const int scl_max = num_scales - 1;

		SIMD::Image_32f cpu_img_temp;
		for (int scl = scl_max; scl >= 0; --scl)
		{
			if (AtomicCompareExchangeSwap(&data_transfer_flag[scl], 1, 0) == 0)
			{
				const Size img_resize = cpu_img_gray.getSize() * scales[scl];
				if (img_resize.width < ext_pattern_size_cd.width || img_resize.height < ext_pattern_size_cd.height)
				{
					AtomicCompareExchangeSwap(&data_transfer_flag[scl], 3, data_transfer_flag[scl]);
					num_scales = MIN(scl, num_scales);
					continue;
				}
				cpu_img_scale[scl].setSize(img_resize);

				cpu_img_temp.clone(cpu_img_scale[scl]);

				if (scales[scl] != 1.f || advanced_param.packet_detection)
				{
					PROFILE_TIMER(cpu_timer_cnn, stat.time_cpu_img_resize,
					if (scales[scl] > 0.7f)
						cpu_img_resizer[scl]->FastImageResize(cpu_img_temp, cpu_img_gray, (int)ImgResize::NearestNeighbor, num_threads);
					else
						cpu_img_resizer[scl]->FastImageResize(cpu_img_temp, cpu_img_gray, (int)ImgResize::Bilinear, num_threads);)
				}
				else
				{
					cpu_img_temp.clone(cpu_img_gray);
				}

				if (!advanced_param.packet_detection)
				{
					PROFILE_TIMER(cpu_timer_cnn, stat.time_cpu_cnn,
					cpu_cnn->Forward(cpu_response_map[scl], cpu_img_temp);)

#if 0
					if (0)
					{
						{
							cv::Mat map = cv::Mat::zeros(cpu_cnn->cnn.input_buffer_size.rows, cpu_cnn->cnn.input_buffer_size.cols, CV_32F);
							float* data = cpu_cnn->cnn.input_data;
							int step = cpu_cnn->cnn.input_buffer_size.step;
							for (int ty = 0; ty < map.rows; ++ty)
							for (int tx = 0; tx < map.cols; ++tx)
							{
								map.at<float>(ty, tx) = data[ty * step + tx];
							}

							double min;
							double max;
							cv::minMaxIdx(map, &min, &max);

							cv::Mat adjMap;
							map = map - min;
							cv::convertScaleAbs(map, adjMap, 255 / (max - min));

							cv::imshow("input_" + std::to_string(scl), adjMap);
							cv::imwrite("input_" + std::to_string(scl) + ".bmp", adjMap);
						}

						for (int layer = 0; layer < cpu_cnn->cnn.layer_count; ++layer)
						{
							for (int surf = 0; surf < cpu_cnn->cnn.layer_buffer[layer].map_count; ++surf)
							{
								cv::Mat map = cv::Mat::zeros(cpu_cnn->cnn.layer_buffer[layer].conv_buffer_size.rows, cpu_cnn->cnn.layer_buffer[layer].conv_buffer_size.cols, CV_32F);
								float* data = cpu_cnn->cnn.layer_buffer[layer].conv_buffer[surf];
								int step = cpu_cnn->cnn.layer_buffer[layer].conv_buffer_size.cols;
								for (int ty = 0; ty < map.rows; ++ty)
								for (int tx = 0; tx < map.cols; ++tx)
								{
									map.at<float>(ty, tx) = data[ty * step + tx];
								}

								double min;
								double max;
								cv::minMaxIdx(map, &min, &max);

								cv::Mat adjMap;
								map = map - min;
								cv::convertScaleAbs(map, adjMap, 255 / (max - min));

								std::string sl = std::to_string(layer + 1);
								std::string ss = std::to_string(surf + 1);
								std::string s = "L_" + sl + " map_" + ss;
								cv::imshow(s.c_str(), adjMap);
								s = s + "_" + std::to_string(scl) + ".bmp";
								cv::imwrite(s.c_str(), adjMap);
							}
						}

						{
							cv::Mat map = cv::Mat::zeros(cpu_cnn->cnn.output_buffer_size.rows, cpu_cnn->cnn.output_buffer_size.cols, CV_32F);
							float* data = cpu_cnn->cnn.layer_buffer[2].pool_buffer[0];
							int step = cpu_cnn->cnn.output_buffer_size.step;
							for (int ty = 0; ty < map.rows; ++ty)
							for (int tx = 0; tx < map.cols; ++tx)
							{
								map.at<float>(ty, tx) = data[ty * step + tx];
							}

						{
							std::ofstream fout("cnn.output_" + std::to_string(scl) + "_" + std::to_string(scales[scl]) + ".txt");
							for (int i = 0; i < map.rows; i++)
							{
								for (int j = 0; j < map.cols; j++)
								{
								fout << map.at<float>(i, j) << "\t";
							}
							fout << std::endl;
						}
						fout.close();
					}

					double min = -1.7159;
					double max = 1.7159;
					//cv::minMaxIdx(map, &min, &max);

					cv::Mat adjMap;
					map = map - min;
					cv::convertScaleAbs(map, adjMap, 255 / (max - min));

					cv::imshow("output_" + std::to_string(scl), adjMap);
					cv::imwrite("output_" + std::to_string(scl) + ".bmp", adjMap);
				}
			}
#endif

#if defined(USE_CUDA) || defined(USE_CL)
					if (param.pipeline == Pipeline::GPU_CPU)
					{
						InterlockedExchange(&data_transfer_flag[scl], 3);
						RunCheckDetect(scl, (int)Pipeline::CPU);
					}
					else
#endif
					{
						AtomicCompareExchangeSwap(&data_transfer_flag[scl], 2, 1);
#if defined(_MSC_VER)
						SetEvent(check_detect_event);
#endif
					}
				}
				else
				{
					AtomicCompareExchangeSwap(&data_transfer_flag[scl], 2, 1);
				}
			}
		}

		if (advanced_param.packet_detection)
		{
			PROFILE_TIMER(cpu_timer_cnn, stat.time_pack_cpu_cnn,
			cpu_cnn->Forward(pack_cpu_response_map, pack_cpu_img_scale);)
		}
#if defined(_MSC_VER)
		SetEvent(check_detect_event);
#endif
	}

#ifdef USE_CUDA
	void CNNDetector::RunGPUDetect()
	{
		const int scl_max = num_scales - 1;

		int inv_ord = 0;
		if (param.pipeline == Pipeline::GPU)
		{
			inv_ord = scl_max;
		}

		int scl_calc = -1;
		CUDA::Image_32f cu_img_temp;
		cudaStream_t cu_stream_temp;
	
		for (int i_scl = 0; i_scl <= scl_max; ++i_scl)
		{
			const int scl = abs(inv_ord - i_scl);

			if (AtomicCompareExchangeSwap(&data_transfer_flag[scl], 1, 0) == 0)
			{
				const Size img_resize = cu_img_gray.getSize() * scales[scl];
				if (img_resize.width < ext_pattern_size_cd.width || img_resize.height < ext_pattern_size_cd.height)
				{
					InterlockedExchange(&data_transfer_flag[scl], 3);
					num_scales = MIN(num_scales, scl);
					continue;
				}
				cu_img_scale[scl].setSize(img_resize);

				int t = 0;
				while (true)
				{
					if (cudaEventQuery(cu_event_response_map[t]) == cudaSuccess)
					{
						if (scl_id[t] != -1)
						{
							AtomicCompareExchangeSwap(&data_transfer_flag[scl_id[t]], 2, 1);
							SetEvent(check_detect_event);
						}
						break;
					}
					t++;
					t = t % advanced_param.concurrent_kernels;
				}

				scl_calc = scl;
				cu_img_temp.clone(cu_img_scale[scl]);
				cu_stream_temp = cu_stream[t];

				if (scales[scl] != 1.f || advanced_param.packet_detection)
				{
					PROFILE_TIMER(cu_timer, stat.time_gpu_img_resize,
					if (scales[scl] > 0.7f)
						CUDA::ImageResizer::FastImageResize(&cu_img_temp, &cu_img_gray, (int)ImgResize::NearestNeighbor, cu_stream_temp);
					else
						CUDA::ImageResizer::FastImageResize(&cu_img_temp, &cu_img_gray, (int)ImgResize::Bilinear, cu_stream_temp);)
				}
				else
				{
					cu_img_temp.clone(*(CUDA::Image_32f*)&cu_img_gray);
				}

				if (!advanced_param.packet_detection)
				{
					scl_id[t] = scl;

					//Nvidia Hyper-Q not supported!
					cuERR(cudaStreamWaitEvent(cu_stream_temp, cu_event_calc_cnn, 0));

					PROFILE_TIMER(cu_timer, stat.time_gpu_cnn,
					cu_cnn[t % advanced_param.num_cnn_copy]->Forward(&cu_response_map[scl], &cu_img_scale[scl], cu_stream_temp);)
					
					cuERR(cudaEventRecord(cu_event_calc_cnn, cu_stream_temp));

					cu_response_map[scl].updateDataHost(true, cu_stream_temp);
				}

				cuERR(cudaEventRecord(cu_event_response_map[t], cu_stream_temp));
			}
		}

		if (advanced_param.packet_detection)
		{
			PROFILE_TIMER(cu_timer, stat.time_pack_gpu_cnn,
			cu_cnn[0]->Forward(&pack_cu_response_map, &pack_cu_img_scale, cu_stream0);
			pack_cu_response_map.updateDataHost(true, cu_stream0);)
		}

		cuERR(cudaDeviceSynchronize());

		if (advanced_param.packet_detection && advanced_param.detect_mode != DetectMode::disable)
		{
			PROFILE_TIMER(cpu_timer_cnn, stat.time_pack_check,
			PacketCPUCheckDetect();)
		}
		
		if (scl_calc >= 0)
		{
			for (auto it = scl_id.begin(); it != scl_id.end(); ++it)
			{
				*it = -1;
			}

			if (param.pipeline == Pipeline::GPU)
			{
				for (int i = scl_max; i >= scl_calc; --i)
				{
					AtomicCompareExchangeSwap(&data_transfer_flag[i], 2, 1);
				}
			}
			else
			{
				for (int i = 0; i <= scl_calc; ++i)
				{
					AtomicCompareExchangeSwap(&data_transfer_flag[i], 2, 1);
				}
			}
			SetEvent(check_detect_event);
		}
	}
#endif
#ifdef USE_CL
	void CNNDetector::RunGPUDetect()
	{
		const int scl_max = num_scales - 1;

		int inv_ord = 0;
		if (param.pipeline == Pipeline::GPU)
		{
			inv_ord = scl_max;
		}

		int scl_calc = -1;
		CL::Image_32f cl_img_temp;
		for (int i_scl = 0; i_scl <= scl_max; ++i_scl)
		{
			const int scl = abs(inv_ord - i_scl);

			if (AtomicCompareExchangeSwap(&data_transfer_flag[scl], 1, 0) == 0)
			{
				const Size img_resize = cl_img_gray.getSize() * scales[scl];
				if (img_resize.width < ext_pattern_size_cd.width || img_resize.height < ext_pattern_size_cd.height)
				{
					InterlockedExchange(&data_transfer_flag[scl], 3);
					num_scales = MIN(num_scales, scl);
					continue;
				}
				cl_img_scale[scl].setSize(img_resize);

				scl_calc = scl;
				cl_img_temp.clone(cl_img_scale[scl]);

				if (scales[scl] != 1.f || advanced_param.packet_detection)
				{
					PROFILE_TIMER(cl_timer, stat.time_gpu_img_resize,
					if (scales[scl] > 0.7f)
						CL::ImageResizer::FastImageResize(&cl_img_temp, &cl_img_gray, (int)ImgResize::NearestNeighbor, cl_queue);
					else
						CL::ImageResizer::FastImageResize(&cl_img_temp, &cl_img_gray, (int)ImgResize::Bilinear, cl_queue);)
				}
				else
				{
					cl_img_temp.clone(cl_img_gray);
				}

				if (!advanced_param.packet_detection)
				{
					PROFILE_TIMER(cl_timer, stat.time_gpu_cnn,
					cl_cnn->Forward(&cl_response_map[scl], &cl_img_temp);)

					if (i_scl % 3 == 0)
					{
						cl_response_map[scl].updateDataHost(true, &cl_event_response_map);
					}
					else
					{
						cl_response_map[scl].updateDataHost(true);
					}

					if (i_scl % 3 == 2)
					{
						if (cl_event_response_map != NULL)
						{
							clERR(clWaitForEvents(1, &cl_event_response_map));
							clERR(clReleaseEvent(cl_event_response_map));
							cl_event_response_map = NULL;
						}

						if (param.pipeline == Pipeline::GPU)
						{
							if (i_scl > 3)
							{
								AtomicCompareExchangeSwap(&data_transfer_flag[scl + 4], 2, 1);
								AtomicCompareExchangeSwap(&data_transfer_flag[scl + 3], 2, 1);
							}
							AtomicCompareExchangeSwap(&data_transfer_flag[scl + 2], 2, 1);
						}
						else
						{
							if (i_scl > 3)
							{
								AtomicCompareExchangeSwap(&data_transfer_flag[scl - 4], 2, 1);
								AtomicCompareExchangeSwap(&data_transfer_flag[scl - 3], 2, 1);
							}
							AtomicCompareExchangeSwap(&data_transfer_flag[scl - 2], 2, 1);
						}
						SetEvent(check_detect_event);
					}
				}
			}
		}

		if (advanced_param.packet_detection)
		{
			PROFILE_TIMER(cl_timer, stat.time_pack_gpu_cnn,
			cl_cnn->Forward(&pack_cl_response_map, &pack_cl_img_scale);
			pack_cl_response_map.updateDataHost(true);)
		}

		clERR(clFinish(cl_queue));
		if (cl_event_response_map != NULL)
		{
			clERR(clReleaseEvent(cl_event_response_map));
			cl_event_response_map = NULL;
		}

		if (scl_calc >= 0)
		{
			if (param.pipeline == Pipeline::GPU)
			{
				for (int i = scl_max; i >= scl_calc; --i)
				{
					AtomicCompareExchangeSwap(&data_transfer_flag[i], 2, 1);
				}
			}
			else
			{
				for (int i = 0; i <= scl_calc; ++i)
				{
					AtomicCompareExchangeSwap(&data_transfer_flag[i], 2, 1);
				}
			}
			SetEvent(check_detect_event);
		}
	}
#endif

	int  CNNDetector::Detect(std::vector<Detection>& detections, SIMD::Image_8u& image)
	{
		if (isEmpty())
		{
			if (Init() < 0)
			{
				printf("[CNNDetector] CNN detector no initialized!\n");
				return -1;
			}
		}

		if (image.width < ext_pattern_size_cd.width || image.height < ext_pattern_size_cd.height)
		{
			return 0;
		}

		if (image.width > param.max_image_size.width || image.height > param.max_image_size.height)
		{
			Clear();
			param.max_image_size.width = image.width;
			param.max_image_size.height = image.height;
			if (advanced_param.gray_image_only && image.nChannel != 1)
			{
				advanced_param.gray_image_only = false;
			}
			if (Init() < 0)
			{
				printf("[CNNDetector] CNN detector no initialized!\n");
				return -1;
			}
		}
		else
		{
			if (image.width != param.max_image_size.width || image.height != param.max_image_size.height)
			{
				num_scales = (int)scales.size();

				if (advanced_param.packet_detection)
				{
					if (PacketReallocate(image.getSize()) < 0)
					{
						printf("[CNNDetector] Packet buffers no initialized!\n");
						return -1;
					}
				}
			}
		}

		if (advanced_param.gray_image_only && cpu_input_img_resizer != nullptr)
		{
			if (image.nChannel == 1)
			{
				PROFILE_TIMER(cpu_timer_detector, stat.time_cpu_input_img_resize,
				cpu_input_img.setSize(image.getSize() * cpu_input_img_scale);
				cpu_input_img_resizer->FastImageResize(cpu_input_img, image, (int)ImgResize::Bilinear, num_threads);
				image.clone(cpu_input_img);)
			}
			else
			{
				printf("[CNNDetector] gray_image_only flag set to false!\n");
				advanced_param.gray_image_only = false;
				Clear();
				return Detect(detections, image);
			}
		}

		PROFILE_COUNTER_INC(stat.num_call_detect)

		if ((int)param.pipeline > 0)
		{
#ifdef USE_CUDA
			cu_img_input.width = image.width;
			cu_img_input.height = image.height;
			cu_img_input.nChannel = image.nChannel;

			PROFILE_TIMER(cu_timer, stat.time_img_HD_copy,
			cu_img_input.copyData(&image, true, cu_stream0);)

			cu_img_gray.width = image.width;
			cu_img_gray.height = image.height;

			int err = 0;
			PROFILE_TIMER(cu_timer, stat.time_gpu_RGBtoGray,
			if (advanced_param.blur)
			{
				err = CUDA::ImageConverter::Img8uToImg32fGRAY_blur_tex(&cu_img_gray, &cu_img_input, col_filter3_kernel(), row_filter3_kernel(), cu_stream0);
			}
			else
			{
				err = CUDA::ImageConverter::Img8uToImg32fGRAY_tex(&cu_img_gray, &cu_img_input, cu_stream0);
			})

			if (err < 0)
			{
				printf("[CNNDetector] This type image is not supported!\n");
				return -1;
			}

			PROFILE_TIMER(cu_timer, stat.time_img_DH_copy,
			cu_img_gray.updateDataHost(true, cu_stream0);)
			cuERR(cudaEventRecord(cu_event_img_gray, cu_stream0));

			cpu_img_gray = SIMD::Image_32f(
				cu_img_gray.width, 
				cu_img_gray.height, 
				cu_img_gray.nChannel, 
				cu_img_gray.dataHost, 
				cu_img_gray.widthStepHost);
#endif
#ifdef USE_CL
			cl_img_input.width = image.width;
			cl_img_input.height = image.height;
			cl_img_input.nChannel = image.nChannel;

			PROFILE_TIMER(cl_timer, stat.time_img_HD_copy,
			cl_img_input.copyData(&image, true);)

			cl_img_gray.width = image.width;
			cl_img_gray.height = image.height;

			int err = 0;
			PROFILE_TIMER(cl_timer, stat.time_gpu_RGBtoGray,
			if (advanced_param.blur)
			{
				err = CL::ImageConverter::Img8uToImg32fGRAY_blur(&cl_img_gray, &cl_img_input, col_filter3_kernel(), row_filter3_kernel(), &cl_img_temp, cl_queue);
			}
			else
			{
				err = CL::ImageConverter::Img8uToImg32fGRAY(&cl_img_gray, &cl_img_input, cl_queue);
			})

			if (err < 0)
			{
				printf("[CNNDetector] This type image is not supported!\n");
				return -1;
			}

			PROFILE_TIMER(cl_timer, stat.time_img_DH_copy,
			cl_img_gray.updateDataHost(true, &cl_event_img_gray);)

			cpu_img_gray = SIMD::Image_32f(
				cl_img_gray.width, 
				cl_img_gray.height, 
				cl_img_gray.nChannel,
				cl_img_gray.dataHost, 
				cl_img_gray.widthStepHost);
#endif
		}
		else
		{
			cpu_img_gray.width = image.width;
			cpu_img_gray.height = image.height;

			int err = 0;
			PROFILE_TIMER(cpu_timer_detector, stat.time_cpu_RGBtoGray,
			if (advanced_param.blur)
			{
				err = SIMD::ImageConverter::Img8uToImg32fGRAY_blur(cpu_img_gray, image, col_filter3_kernel(), row_filter3_kernel(), num_threads);
			}
			else
			{
				err = SIMD::ImageConverter::Img8uToImg32fGRAY(cpu_img_gray, image, num_threads);
			})

			if (err < 0)
			{
				printf("[CNNDetector] This type image is not supported!\n");
				return -1;
			}
		}

		SIMD::mm_erase((void*)data_transfer_flag, int(scales.size() * sizeof(data_transfer_flag[0])));

		OMP_RUNTIME(omp_set_nested(1))

		PROFILE_TIMER(cpu_timer_detector, stat.time_detect,
		GPU_ONLY(
		if ((int)param.pipeline > 0)
		{
			if (param.pipeline == Pipeline::GPU_CPU)
			{
				OMP_PRAGMA(omp parallel for num_threads(3))
				for (int run = 0; run < 3; ++run)
				{
					switch (run)
					{
					case 0: RunCPUDetect();			break;
					case 1: RunGPUDetect();			break;
					case 2: RunCheckDetectAsync();	break;
					default: continue;
					}
				}
			}
			else
			{			
				int num_trd = advanced_param.detect_mode == DetectMode::async ? 2 : 1;

				OMP_PRAGMA(omp parallel for num_threads(num_trd))
				for (int run = 0; run < 2; ++run)
				{
					switch (run)
					{
					case 0: RunGPUDetect();			break;
					case 1: RunCheckDetectAsync();	break;
					default: continue;
					}
				}
			}

			CL_CODE(
			clERR(clReleaseEvent(cl_event_img_gray));
			cl_event_img_gray = NULL;)
		}
		else)
		{
			int num_trd = advanced_param.detect_mode == DetectMode::async ? 2 : 1;

			OMP_PRAGMA(omp parallel for num_threads(num_trd))
			for (int run = 0; run < 2; ++run)
			{
				switch (run)
				{
				case 0: RunCPUDetect();			break;
				case 1: RunCheckDetectAsync();	break;
				default: continue;
				}
			}

			/*
			for (int shift = 0; shift <= 2; ++shift)
			{
				OMP_PRAGMA(omp parallel for num_threads(num_trd))
				for (int run = 0; run < 2; ++run)
				{
					switch (run)
					{
					case 0: RunCPUDetect();			break;
					case 1: RunCheckDetectAsync();	break;
					default: continue;
					}
				}
			
				if (0 || shift == 2) break;
				SIMD::mm_erase((void*)data_transfer_flag, scales.size() * sizeof(data_transfer_flag[0]));
				for (int sh_y = 2; sh_y < cpu_img_gray.height; ++sh_y)
				{
					for (int sh_x = 2; sh_x < cpu_img_gray.width; ++sh_x)
					{
						if (shift == 0)
						{
							cpu_img_gray.data[sh_y * cpu_img_gray.widthStep + sh_x - 2] = cpu_img_gray.data[sh_y * cpu_img_gray.widthStep + sh_x];
						}
						else
						{
							cpu_img_gray.data[(sh_y - 2) * cpu_img_gray.widthStep + sh_x] = cpu_img_gray.data[sh_y * cpu_img_gray.widthStep + sh_x];
						}
					}
				}
			}
			*/
		})

		PROFILE_TIMER(cpu_timer_detector, stat.time_post_proc,
		std::reverse_copy(cpu_detect_rect.begin(), cpu_detect_rect.end(), std::back_inserter(gpu_detect_rect));

		if (gpu_detect_rect.size() > 0)
		{
			std::sort(gpu_detect_rect.begin(), gpu_detect_rect.end(), [](Detection& a, Detection& b) { return b.scale < a.scale; });

			PROFILE_COUNTER_ADD(stat.num_detections_raw, gpu_detect_rect.size())
			if (advanced_param.merger_detect)
			{
				std::vector<Detection> detections_temp = detections;
				detections_temp.reserve((int)detections.size() + (int)gpu_detect_rect.size() / 2);
				Merger(detections_temp, gpu_detect_rect, 0.5f);

				if (advanced_param.min_num_detect > 1)
				{
					const int threshold = advanced_param.min_num_detect;
					detections_temp.erase(std::remove_if(detections_temp.begin(), detections_temp.end(), 
						[&threshold] (const Detection& detect) { return (detect.num_detect < threshold); }),
						detections_temp.end());
				}

				if (detections_temp.size() > 0)
				{
					detections.clear();
					detections.reserve((int)detections_temp.size());
					Merger(detections, detections_temp, 0.2f, true);
				}

				detections_temp.clear();
			}
			else
			{
				detections.resize((int)gpu_detect_rect.size());
				std::copy(gpu_detect_rect.begin(), gpu_detect_rect.end(), detections.begin());
			}
			PROFILE_COUNTER_ADD(stat.num_detections_result, detections.size())

			if (cpu_input_img_resizer != nullptr)
			{
				const float scale = (float)param.min_obj_size.height / (float)pattern_size.height;
				for (auto it = detections.begin(); it != detections.end(); ++it)
				{
					const int sx = int(float(it->rect.x) * scale);
					const int sy = int(float(it->rect.y) * scale);
					const int sw = int(float(it->rect.width) * scale);
					const int sh = int(float(it->rect.height) * scale);
					it->rect = Rect(sx, sy, sw, sh);

					if (cpu_cnn_fa.size() != 0)
					{
						for (size_t k = 0; k < it->facial_data.size(); ++k)
						{
							for (size_t t = 0; t < 5; ++t)
							{
								it->facial_data[k].landmarks[t].x = int(float(it->facial_data[k].landmarks[t].x) * scale);
								it->facial_data[k].landmarks[t].y = int(float(it->facial_data[k].landmarks[t].y) * scale);
							}
						}
					}
				}
			}

			cpu_detect_rect.clear();
			gpu_detect_rect.clear();
		})

		return 0;
	}
	void CNNDetector::Merger(std::vector<Detection>& detections, std::vector<Detection>& rect, float threshold, bool del)
	{
		std::vector<int> X(4);
		std::vector<int> Y(4);

		auto it1 = rect.begin();
		while (it1 != rect.end())
		{
			if ((*it1).isCheck())
			{
				it1++;
				continue;
			}

			Detection new_face_max = (*it1);
			Detection new_face_min = (*it1);

			auto it2 = it1 + 1;
			while (it2 != rect.end())
			{
				if ((*it2).isCheck())
				{
					it2++;
					continue;
				}

				X[0] = new_face_max.rect.x;
				Y[0] = new_face_max.rect.y;
				X[1] = new_face_max.rect.x2;
				Y[1] = new_face_max.rect.y2;

				X[2] = (*it2).rect.x;
				Y[2] = (*it2).rect.y;
				X[3] = (*it2).rect.x2;
				Y[3] = (*it2).rect.y2;

				if (!(X[0] >= X[3] || X[1] <= X[2] || Y[0] >= Y[3] || Y[1] <= Y[2]))
				{
					const float S_1 = float((X[1] - X[0]) * (Y[1] - Y[0]));
					const float S_2 = float((X[3] - X[2]) * (Y[3] - Y[2]));

					std::sort(X.begin(), X.end());
					std::sort(Y.begin(), Y.end());

					const float S_union = float((X[3] - X[0]) * (Y[3] - Y[0]));
					const float S_intersection = float((X[2] - X[1]) * (Y[2] - Y[1]));

					const float ratio_1 = S_intersection / S_1;
					const float ratio_2 = S_intersection / S_2;
					const float ratio_3 = S_intersection / S_union;

					float ratio = MIN(ratio_1, ratio_2);

					const int CX1 = X[0] + ((X[3] - X[0]) >> 1);
					const int CY1 = Y[0] + ((Y[3] - Y[0]) >> 1);
					const int CX2 = X[1] + ((X[2] - X[1]) >> 1);
					const int CY2 = Y[1] + ((Y[2] - Y[1]) >> 1);

					const int threshold_C = MIN((X[3] - X[0]), (X[2] - X[1])) >> 1;
					if (ratio_3 > 0.75f * threshold   && 
						MAX(ratio_1, ratio_2) > 0.7f  && 
						abs(CX1 - CX2) < threshold_C  && 
						abs(CY1 - CY2) < threshold_C)
					{
						ratio = 1.f;
					}

					if (ratio > threshold)
					{
						if (del)
						{
							if (S_1 > S_2)
							{
								(*it2).setCheckFlag();
								continue;
							}
							else
							{
								new_face_min.rect.x2 = 0;
								break;
							}
						}

						new_face_max.rect.x = MIN(new_face_max.rect.x, X[0]);
						new_face_max.rect.y = MIN(new_face_max.rect.y, Y[0]);
						new_face_max.rect.x2 = MAX(new_face_max.rect.x2, X[3]);
						new_face_max.rect.y2 = MAX(new_face_max.rect.y2, Y[3]);

						new_face_min.rect.x = MAX(new_face_min.rect.x, X[1]);
						new_face_min.rect.y = MAX(new_face_min.rect.y, Y[1]);
						new_face_min.rect.x2 = MIN(new_face_min.rect.x2, X[2]);
						new_face_min.rect.y2 = MIN(new_face_min.rect.y2, Y[2]);

						new_face_max.score = MAX(new_face_max.score, (*it2).score);
						new_face_max.knn = MAX(new_face_max.knn, (*it2).knn);
						new_face_max.scale = MIN(new_face_max.scale, (*it2).scale);
						
						for (int t = 0; t < (*it2).facial_data.size(); ++t)
						{
							new_face_max.facial_data.push_back((*it2).facial_data[t]);
						}

						new_face_max.num_detect++;

						(*it2).setCheckFlag();
					}
				}
				it2++;
			}
			it1++;

			if (del && new_face_min.rect.x2 == 0) continue;

			new_face_max.rect.x = (new_face_min.rect.x + new_face_max.rect.x) >> 1;
			new_face_max.rect.y = (new_face_min.rect.y + new_face_max.rect.y) >> 1;
			new_face_max.rect.x2 = (new_face_min.rect.x2 + new_face_max.rect.x2) >> 1;
			new_face_max.rect.y2 = (new_face_min.rect.y2 + new_face_max.rect.y2) >> 1;		
			
			new_face_max.rect.width = new_face_max.rect.x2 - new_face_max.rect.x;
			new_face_max.rect.height = new_face_max.rect.y2 - new_face_max.rect.y;
			detections.push_back(new_face_max);
		}
	}

	void CNNDetector::setNumThreads(int _num_threads)
	{
		OMP_RUNTIME(int max_num_threads = MAX_NUM_THREADS;)

		OMP_RUNTIME(
		if (_num_threads <= 0)
		{
			_num_threads = MIN(omp_get_num_procs(), max_num_threads);
		}
		param.num_threads = MIN(_num_threads, max_num_threads);

		if (num_threads < _num_threads)
		{
			Clear();
		}
		else
		{
			num_threads = param.num_threads;
			cpu_cnn->setNumThreads(num_threads);
		})
	}

	int CNNDetector::getGrayImage(SIMD::Image_32f* image) const
	{
		if (image != nullptr)
		{
			image->clone(cpu_img_gray);
			return 0;
		}
		else return -1;
	}
#ifdef USE_CUDA
	int CNNDetector::getGrayImage(CUDA::Image_32f_pinned* image) const
	{
		if (image != nullptr && param.pipeline != Pipeline::CPU)
		{
			image->clone((const CUDA::Image_32f_pinned)cu_img_gray);
			return 0;
		}
		else return -1;
	}
#endif
#ifdef USE_CL
	int CNNDetector::getGrayImage(CL::Image_32f* image) const
	{
		if (image != nullptr && param.pipeline != Pipeline::CPU)
		{
			image->clone(cl_img_gray);
			return 0;
		}
		else return -1;
	}
#endif

#ifdef PROFILE_DETECTOR
	void CNNDetector::Stat::reset()
	{
		//max_image = Size(0, 0);
		//max_scale = Size(0, 0);
		//min_scale = Size(0, 0);
		//num_scales = 0;

		num_call_detect = 0;
		num_detections_raw = 0;
		num_detections_result = 0;

		time_cpu_input_img_resize = 0.;
		time_img_HD_copy = 0.;
		time_cpu_RGBtoGray = 0.;
		time_gpu_RGBtoGray = 0.;
		time_img_DH_copy = 0.;
		time_detect = 0.;
		time_post_proc = 0.;

		num_call_check_cpu = 0;
		num_call_check_gpu = 0;
		num_responses_stage1 = 0.;
		num_detections_stage1 = 0.;

		num_check_ver_drop = 0.;
		time_check_ver_drop = 0.;

		time_cpu_img_resize = 0.;
		time_cpu_cnn = 0.;
		time_pack_cpu_cnn = 0.;

		time_gpu_img_resize = 0.;
		time_gpu_cnn = 0.;
		time_pack_gpu_cnn = 0.;

		num_check_mod0 = 0;
		num_check_mod1 = 0;
		num_check_hor_drop = 0;
		num_check_call_cnn2 = 0;
		num_detections_stage2 = 0;
		num_check_call_cnn3 = 0;
		num_detections_stage3 = 0;
		num_check_add_rect = 0;

		time_check = 0.;
		time_check_proc = 0.;
		time_check_cpu_cnn2 = 0.;
		time_check_cpu_cnn3 = 0.;
		time_check_find_detections = 0.;

		num_pack_check_detect = 0.;
		
		time_pack_check = 0.;
		time_pack_check_proc = 0.;
		time_pack_check_gpu_cnn = 0.;
		time_pack_erase_detect = 0.;
	}
	void CNNDetector::Stat::print()
	{
		printf("[CNNDetector] ---------------------- stat ----------------------\n");
		printf("[CNNDetector] stat: max_image_size = %dx%d\n", max_image.width, max_image.height);
		printf("[CNNDetector] stat: max_scale_size = %dx%d\n", max_scale.width, max_scale.height);
		printf("[CNNDetector] stat: min_scale_size = %dx%d\n", min_scale.width, min_scale.height);
		printf("[CNNDetector] stat: num_scales = %d\n", num_scales);

		printf("[CNNDetector] stat: num_call_detect = %d\n", num_call_detect);
		printf("[CNNDetector] stat: num_detections_raw = %d\n", num_detections_raw);
		printf("[CNNDetector] stat: num_detections_result = %d\n", num_detections_result);

		printf("[CNNDetector] stat:	 time_cpu_input_img_resize = %5.3f ms\n", time_cpu_input_img_resize / num_call_detect);
		printf("[CNNDetector] stat:	 time_img_HD_copy = %5.3f ms\n", time_img_HD_copy / num_call_detect);
		printf("[CNNDetector] stat:	 time_cpu_RGBtoGray = %5.3f ms\n", time_cpu_RGBtoGray / num_call_detect);
		printf("[CNNDetector] stat:	 time_gpu_RGBtoGray = %5.3f ms\n", time_gpu_RGBtoGray / num_call_detect);
		printf("[CNNDetector] stat:	 time_img_DH_copy = %5.3f ms\n", time_img_DH_copy / num_call_detect);
		printf("[CNNDetector] stat:	 time_detect = %5.3f ms\n", time_detect / num_call_detect);
		printf("[CNNDetector] stat:	 time_post_proc = %5.3f ms\n", time_post_proc / num_call_detect);

		printf("[CNNDetector] stat: num_call_check_cpu = %d\n", num_call_check_cpu);
		printf("[CNNDetector] stat: num_call_check_gpu = %d\n", num_call_check_gpu);
		printf("[CNNDetector] stat: num_responses_stage1 = %f\n", num_responses_stage1);
		printf("[CNNDetector] stat: num_detections_stage1 = %f\n", num_detections_stage1);

		printf("[CNNDetector] stat: num_check_ver_drop = %f\n", num_check_ver_drop);
		printf("[CNNDetector] stat:	 time_check_ver_drop_total = %5.3f ms\n", time_check_ver_drop / num_call_detect);
		printf("[CNNDetector] stat:	 time_check_ver_drop = %5.3f ms\n", time_check_ver_drop / (num_call_detect * (num_call_check_cpu + num_call_check_gpu)));

		printf("[CNNDetector] stat:	 time_cpu_img_resize = %5.3f ms\n", time_cpu_img_resize / num_call_detect);
		printf("[CNNDetector] stat:	 time_cpu_cnn1 = %5.3f ms\n", time_cpu_cnn / num_call_detect);
		printf("[CNNDetector] stat:	 time_pack_cpu_cnn1 = %5.3f ms\n", time_pack_cpu_cnn / num_call_detect);

		printf("[CNNDetector] stat:	 time_gpu_img_resize = %5.3f ms\n", time_gpu_img_resize / num_call_detect);
		printf("[CNNDetector] stat:	 time_gpu_cnn1 = %5.3f ms\n", time_gpu_cnn / num_call_detect);
		printf("[CNNDetector] stat:	 time_pack_gpu_cnn1 = %5.3f ms\n", time_pack_gpu_cnn / num_call_detect);

		printf("[CNNDetector] stat: num_check_mod0 = %d\n", num_check_mod0);
		printf("[CNNDetector] stat: num_check_mod1 = %d\n", num_check_mod1);
		printf("[CNNDetector] stat: num_check_hor_drop = %d\n", num_check_hor_drop);
		printf("[CNNDetector] stat: num_check_call_cnn2 = %d\n", num_check_call_cnn2);
		printf("[CNNDetector] stat: num_detections_stage2 = %d\n", num_detections_stage2);
		printf("[CNNDetector] stat: num_check_call_cnn3 = %d\n", num_check_call_cnn3);
		printf("[CNNDetector] stat: num_detections_stage3 = %d\n", num_detections_stage3);
		printf("[CNNDetector] stat: num_check_add_rect = %d\n", num_check_add_rect);

		printf("[CNNDetector] stat:	 time_check_total = %5.3f ms\n", (time_check + time_check_ver_drop) / num_call_detect);
		printf("[CNNDetector] stat:	 time_check = %5.3f ms\n", time_check / (num_check_mod0 + num_check_mod1));
		printf("[CNNDetector] stat:	 time_check_proc = %5.3f ms\n", time_check_proc / (num_check_mod0 + num_check_mod1));
		printf("[CNNDetector] stat:	 time_check_cpu_cnn2 = %5.3f ms\n", time_check_cpu_cnn2 / num_check_call_cnn2);
		printf("[CNNDetector] stat:	 time_check_cpu_cnn3 = %5.3f ms\n", time_check_cpu_cnn3 / num_check_call_cnn3);
		printf("[CNNDetector] stat:	 time_check_find_detections = %5.3f ms\n", time_check_find_detections / (num_check_call_cnn2 + num_check_call_cnn3));

		printf("[CNNDetector] stat: num_pack_check_detect = %f\n", num_pack_check_detect);

		printf("[CNNDetector] stat:	 time_pack_check_total = %5.3f ms\n", time_pack_check / num_call_detect);
		printf("[CNNDetector] stat:	 time_pack_check = %5.3f ms\n", time_pack_check / num_call_detect);
		printf("[CNNDetector] stat:	 time_pack_check_proc = %5.3f ms\n", time_pack_check_proc / num_call_detect);
		printf("[CNNDetector] stat:	 time_pack_check_gpu_cnn = %5.3f ms\n", time_pack_check_gpu_cnn / num_call_detect);
		printf("[CNNDetector] stat:	 time_pack_erase_detect = %5.3f ms\n", time_pack_erase_detect / num_call_detect);
	}
#endif
}