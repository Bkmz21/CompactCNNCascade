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


#pragma once

#include "config.h"

#include "image_proc.h"
#include "image_resize.h"

#ifndef USE_CNTK_MODELS
#	include "cnn_simd.h"
#	include "cnn_simd_v2.h"
#else
#	include "cnn_simd_cntk.h"
#	include "cnn_simd_v2_cntk.h"
#endif

#ifdef USE_CUDA
#	include "init_cuda.h"
#	include "image_proc_cuda.h"
#	include "image_resize_cuda.h"
#	ifdef USE_CNTK_MODELS
#		include "cnn_cuda_cntk.h"
#	endif
#endif

#ifdef USE_CL
#	include "init_cl.h"
#	include "image_proc_cl.h"
#	include "image_resize_cl.h"
#	ifndef USE_CNTK_MODELS
#		include "cnn_cl.h"
#	else
#		include "cnn_cl_cntk.h"
#	endif
#endif

#include "packing_2D.h"

#include <vector>
#include <iterator>
#include <list>
#include <sstream>
#include <fstream>

#if defined(_MSC_VER)
#	include <windows.h>
#endif

#ifdef PROFILE_DETECTOR
#	include "timer.h"
#endif

#ifdef CHECK_TEST
#	include "cnn.h"
#endif

//#define ConvNeuralNetwork_v2 ConvNeuralNetwork


//========================================================================================================


namespace NeuralNetworksLib
{

	class CNNDetector
	{
	public:
		enum struct Pipeline
		{
			CPU		= 0,
			GPU		= 1,
			GPU_CPU	= 2
		};
		enum struct ImgResize
		{
			NearestNeighbor = 0,
			Bilinear = 1
		};
		enum struct DetectMode
		{
			disable = 0,
			sync	= 1,
			async	= 2
		};
		enum struct DetectPrecision
		{
			def 	= 0,
			low  	= 1,
			normal	= 2,
			high	= 3,
			ultra	= 4
		};

		struct Param
		{
			Pipeline pipeline = Pipeline::CPU;
			Size max_image_size = Size(1280, 720);
			Size min_obj_size = Size(48, 48);
			Size max_obj_size = Size(0, 0);
			float scale_factor = 1.2f;
			int min_neighbors = 2;
			int num_threads = 0;
		};
		struct AdvancedParam
		{
		public:
			//main
			DetectMode detect_mode = DetectMode::sync;
			DetectPrecision detect_precision = DetectPrecision::normal;
			bool packet_detection = false;
			bool gray_image_only = false;
			bool facial_analysis = false;

			//precision
			bool equalize = true;
			bool reflection = true;
			int type_check = 1;
			float treshold_1 = 0.f;
			float treshold_2 = 0.f;
			float treshold_3 = 0.f;
			bool adapt_min_neighbors = false;
			bool double_check = true;
			bool drop_detect = true;
			int min_num_detect = 1;

			//device
			bool device_info = false;
			int cuda_device_id = 0;
			int cl_platform_id = -1;
			int cl_device_id = -1;
			int concurrent_kernels = 3;
			int num_cnn_copy = 1; //not supported!

			//secondary
			bool blur = false;
			ImgResize img_resize = ImgResize::NearestNeighbor;
			bool uniform_noise = false;
			bool merger_detect = true;

			std::string path_model[4];
			int index_output[4];

			AdvancedParam()
			{
				path_model[0] = "";
				path_model[1] = "";
				path_model[2] = "";
				path_model[3] = "";
				
				index_output[0] = 1;
				index_output[1] = 0;
				index_output[2] = 0;
				index_output[3] = -1;
			}
		};

		struct FacialData
		{
			Point landmarks[5];
			int gender = -1;
			int smile = -1;
			int glasses = -1;
		};

		struct Detection
		{
		private:
			bool checked = false;

		public:
			Rect rect;
			float score = 0.f;
			float scale = 0.f;
			int knn = 0;
			int num_detect = 0;
			std::vector<FacialData> facial_data;

			Detection() { }
			Detection(int _x, int _y, int _width, int _height, float _score, float _scale, int _knn)
			{
				rect = Rect(_x, _y, _width, _height);
				score = _score;
				scale = _scale;
				knn = _knn;
			}
			Detection(Rect _rect, float _score, float _scale, int _knn)
			{
				rect = _rect;
				score = _score;
				scale = _scale;
				knn = _knn;
			}

			void setCheckFlag() { checked = true; }
			bool isCheck() { return checked; }
		};

	private:
		struct PackPos
		{
			int scl = 0;
			int x = 0;
			int y = 0;
			int pack_id = 0;

			PackPos(int _scl, int _x, int _y, int _pack_id)
			{
				scl = _scl;
				x = _x;
				y = _y;
				pack_id = _pack_id;
			}
		};

		Param param;
		AdvancedParam advanced_param;

#ifdef USE_AVX
		SIMD::ConvNeuralNetwork_v2* cpu_cnn = NULL;
		//Legacy::ConvNeuralNetwork* cpu_cnn = NULL;
#else
		SIMD::ConvNeuralNetwork* cpu_cnn = NULL;
#endif

		SIMD::Image_32f					cpu_img_gray;
		std::vector<SIMD::Image_32f>	cpu_img_scale;
		std::vector<SIMD::Image_32f>	cpu_response_map;

		SIMD::Image_32f	pack_cpu_img_scale;
		SIMD::Image_32f	pack_cpu_response_map;

		std::vector<SIMD::ConvNeuralNetwork*> cpu_cnn_check1;
		std::vector<SIMD::ConvNeuralNetwork*> cpu_cnn_check2;
		std::vector<SIMD::ConvNeuralNetwork*> cpu_cnn_fa;
		//std::vector<Legacy::ConvNeuralNetwork*> cpu_cnn_check1;
		//std::vector<Legacy::ConvNeuralNetwork*> cpu_cnn_check2;

		std::vector<SIMD::Image_32f>	cpu_img_check_resize_32f;
		std::vector<SIMD::Image_8u>		cpu_img_check_8u;
		std::vector<SIMD::Image_32f>	cpu_img_check_32f;
		SIMD::TmpImage<char>			cpu_img_urnd;

		std::vector<SIMD::ImageResizer*> cpu_img_resizer;
		std::vector<SIMD::ImageResizer*> cpu_img_check_resizer;

		SIMD::Image_8u cpu_input_img;
		float cpu_input_img_scale = 0.f;
		SIMD::ImageResizer* cpu_input_img_resizer = nullptr;

		#if defined(_MSC_VER)
			HANDLE check_detect_event = NULL;
        #else
			int check_detect_event = 1;
        #endif

#ifdef USE_CUDA
		std::vector<CUDA::ConvNeuralNetwork*> cu_cnn;

		CUDA::Image_8u_pinned					cu_img_input;
		CUDA::Image_32f_pinned					cu_img_gray;
		std::vector<CUDA::Image_32f>			cu_img_scale;
		std::vector<CUDA::Image_32f_pinned>		cu_response_map;

		CUDA::Image_32f			pack_cu_img_scale;
		CUDA::Image_32f_pinned	pack_cu_response_map;

		CUDA::ConvNeuralNetwork* cu_cnn_check = NULL;
		CUDA::Image_8u_pinned	 pack_cu_img_check_8u;
		CUDA::Image_32f			 pack_cu_img_check_32f;
		CUDA::Image_32f_pinned	 pack_cu_response_map_check;
		
		Size pack_img_check_size;
		const Size2d pack_max_num_img_check = Size2d(50, 50);
		std::list<PackPos> pack_pos_check;

		cudaStream_t cu_stream0 = NULL;
		cudaEvent_t cu_event_img_gray = NULL;
		cudaEvent_t cu_event_calc_cnn = NULL;

		std::vector<int> scl_id;
		std::vector<cudaStream_t> cu_stream;
		std::vector<cudaEvent_t> cu_event_response_map;
#endif

#ifdef USE_CL
		CL::ConvNeuralNetwork* cl_cnn = NULL;

		CL::Image_8u				cl_img_input;
		CL::Image_32f				cl_img_gray;
		CL::Image_32f				cl_img_temp;
		std::vector<CL::Image_32f>	cl_img_scale;
		std::vector<CL::Image_32f>	cl_response_map;

		CL::Image_32f	pack_cl_img_scale;
		CL::Image_32f	pack_cl_response_map;

		cl_device_id cl_device = NULL;
		cl_context cl_context = NULL;
		cl_command_queue cl_queue = NULL;

		cl_event cl_event_img_gray = NULL;
		cl_event cl_event_response_map = NULL;
#endif

		std::vector<float> scales;
		int num_scales = 0;

		int num_threads = 0; //OpenMP only

		Packing2D packing2D;
		Size pack_size;
		std::vector<Rect> pack;

		std::vector<Detection> cpu_detect_rect;
		std::vector<Detection> gpu_detect_rect;
		volatile long* data_transfer_flag = NULL;

		Size pattern_size;
		Size pattern_size_cd;
		Size ext_pattern_size_cd;
		int shift_pattern = 0;
		int ext_pattern_offset = 0;		//15;
		int x_pattern_offset = 0;		//4;

		SIMD::Array_32f col_filter3_kernel;
		SIMD::Array_32f row_filter3_kernel;

#ifdef PROFILE_DETECTOR
		Timer* cpu_timer_detector;
		Timer* cpu_timer_cnn;
		Timer* cpu_timer_check1;
		Timer* cpu_timer_check2;
#	ifdef USE_CUDA
		CUDA::Timer* cu_timer;
#	endif
#	ifdef USE_CL
		CL::Timer* cl_timer;
#	endif
#endif

		int InitDevice();
		int InitCNN();
		int InitScales();
		int InitPaketCNN();
		int InitBuffers();
		int InitCNNBuffers();
		int InitCNNCheck();
		int InitBlurFilters();

		int  Init();
		void Clear();

		inline void CPUCheckDetect(std::vector<Detection>& rect, const int rect_size, const Point& point, const float score0,
									const SIMD::Image_32f& img, const float scale, const int mod = 0, const int pack_id = 0);

		int PacketReallocate(Size size);
		void PacketCPUCheckDetect();

		bool DropDetection(Rect& new_rect, std::vector<Detection>& detect_rect_in, std::vector<Detection>& detect_rect_out, float scale);
		
		void RunCheckDetect(const int scl, const int device);
		void RunCheckDetectAsync();

		void RunCPUDetect();
		void RunGPUDetect();

		inline void Merger(std::vector<Detection>& detections, std::vector<Detection>& rect, float threshold = 0.5f, bool del = false);

		public: void* hGrd = 0;

	public:
		CNNDetector(Param* _param = 0, AdvancedParam* _advanced_param = 0);
		~CNNDetector();

		bool isEmpty() const;
		
		int Detect(std::vector<Detection>& detections, SIMD::Image_8u& image);
		void NMS(std::vector<Detection>& detections, std::vector<Detection>& rect)
		{
			Merger(detections, rect);
		}

		Param getParam() const { return param; }
		void  setParam(Param& _param)
		{
			Clear();
			param = _param;
		}

		AdvancedParam getAdvancedParam() const { return advanced_param; }
		void setAdvancedParam(AdvancedParam& _advanced_param)
		{
			Clear();
			advanced_param = _advanced_param;
		}

		Size getMaxImageSize() const { return param.max_image_size; }
		void setMaxImageSize(Size _max_image_size)
		{
			Clear();
			param.max_image_size = _max_image_size;
		}

		Size getMinObjectSize() const { return param.min_obj_size; }
		void setMinObjectSize(Size _min_obj_size)
		{
			Clear();
			param.min_obj_size = _min_obj_size;
		}

		Size getMaxObjectSize() const { return param.max_obj_size; }
		void setMaxObjectSize(Size _max_obj_size)
		{
			Clear();
			param.max_obj_size = _max_obj_size;
		}

		int  getNumThreads() const { return num_threads; }
		void setNumThreads(int _num_threads);

		int getGrayImage(SIMD::Image_32f* image) const;
#ifdef USE_CUDA
		int getGrayImage(CUDA::Image_32f_pinned* image) const;
#endif
#ifdef USE_CL
		int getGrayImage(CL::Image_32f* image) const;
#endif

		CNNDetector& operator=(const CNNDetector& gpu_cnn_detector)
		{
			*this = gpu_cnn_detector;
			return *this;
		}

#ifdef PROFILE_DETECTOR
		struct Stat
		{
			Size max_image;
			Size max_scale;
			Size min_scale;
			int num_scales = 0;

			int num_call_detect = 0;
			int num_detections_raw = 0;
			int num_detections_result = 0;

			double time_cpu_input_img_resize = 0.;
			double time_img_HD_copy = 0.;
			double time_cpu_RGBtoGray = 0.;
			double time_gpu_RGBtoGray = 0.;
			double time_img_DH_copy = 0.;
			double time_detect = 0.;
			double time_post_proc = 0.;

			int num_call_check_cpu = 0;
			int num_call_check_gpu = 0;
			double num_responses_stage1 = 0;
			double num_detections_stage1 = 0;

			double num_check_ver_drop = 0;
			double time_check_ver_drop = 0.;

			double time_cpu_img_resize = 0.;
			double time_cpu_cnn = 0.;
			double time_pack_cpu_cnn = 0.;

			double time_gpu_img_resize = 0.;
			double time_gpu_cnn = 0.;
			double time_pack_gpu_cnn = 0.;

			volatile int num_check_mod0 = 0;
			volatile int num_check_mod1 = 0;
			volatile int num_check_hor_drop = 0;
			volatile int num_check_call_cnn2 = 0;
			volatile int num_detections_stage2 = 0;
			volatile int num_check_call_cnn3 = 0;
			volatile int num_detections_stage3 = 0;
			volatile int num_check_add_rect = 0;

			double time_check = 0.;
			double time_check_proc = 0.;
			double time_check_cpu_cnn2 = 0.;
			double time_check_cpu_cnn3 = 0.;
			double time_check_find_detections = 0.;

			double num_pack_check_detect = 0;

			double time_pack_check = 0.;
			double time_pack_check_proc = 0.;
			double time_pack_check_gpu_cnn = 0.;
			double time_pack_erase_detect = 0.;

			Stat() { reset(); }

			void reset();
			void print();
		} stat;
#endif
	};

}