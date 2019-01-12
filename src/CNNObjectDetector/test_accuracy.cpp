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


#include "config.h"
#include "serialized_models.h"

#include "image.h"
#include "image_proc.h"
#include "image_resize.h"

#ifndef USE_CNTK_MODELS
#	include "cnn_simd.h"
#	include "cnn_simd_v2.h"
#else
#	include "cnn_simd_cntk.h"
#	include "cnn_simd_v2_cntk.h"
#endif

#include "timer.h"

#include "resource.h"
#include <sstream>
#include <fstream>
#include <windows.h>

#ifdef USE_CUDA
#	include "init_cuda.h"
#	include "image_proc_cuda.h"
#	include "image_resize_cuda.h"
#	ifndef USE_CNTK_MODELS
#		include "cnn_cuda.h"
#	else
#		include "cnn_cuda_cntk.h"
#	endif
#endif

#ifdef USE_CL
#	include "init_cl.h"
#	include "image_proc_cl.h"
#	include "image_resize_cl.h"
#	include "cnn_cl_cntk.h"
#endif

//OpenCV
//#pragma comment(lib, "opencv_core300.lib")
//#pragma comment(lib, "opencv_highgui300.lib")
//#pragma comment(lib, "opencv_imgcodecs300.lib")
//#pragma comment(lib, "opencv_imgproc300.lib")
//#pragma comment(lib, "opencv_videoio300.lib")
//#pragma comment(lib, "opencv_video300.lib")
//#pragma comment(lib, "opencv_objdetect300.lib")

#ifdef USE_CUDA
#	pragma comment(lib, "cudart_static.lib")
#endif
#ifdef USE_CL
#	pragma comment(lib, "OpenCL.lib")
#endif

#include "cnn_detector_v3.h"

#include <iostream>
#include <vector>
#include <time.h>

//#define ConvNeuralNetwork_v2 ConvNeuralNetwork

using namespace NeuralNetworksLib;

CNNDetector* CNNGPUD;


//================================================================================================================================================


	template<typename type>
	int init_data(SIMD::TmpImage<type>& img)
	{
		srand(time(NULL));

		for (int j = 0; j < img.height; ++j)
		{
			for (int i = 0; i < img.width; ++i)
			{
				for (int c = 0; c < img.nChannel; ++c)
				{
					int offset = j * img.widthStep + img.nChannel * i + c;
					img.data[offset] = 255.f * (float)rand() / (float)RAND_MAX;
				}
			}
		}
		return 0;
	}

	template<typename type>
	int init_data(SIMD::TmpImage<type>& img_dst, SIMD::TmpImage<float>& img)
	{
		for (int j = 0; j < img_dst.height; ++j)
		{
			for (int i = 0; i < img_dst.width; ++i)
			{
				for (int c = 0; c < img_dst.nChannel; ++c)
				{
					int offset_dst = j * img_dst.widthStep + img.nChannel * i + c;
					int offset = j * img.widthStep + img.nChannel * i + c;
					img_dst.data[offset_dst] = (type)img.data[offset];
				}
			}
		}
		return 0;
	}

	template<typename type, const int pinned_mem = 0>
	int check_data(SIMD::TmpImage<type>& img, SIMD::TmpImage<type>& img_2, float eps = 1.E-3)
	{
		for (int j = 0; j < img.height; ++j)
		{
			for (int i = 0; i < img.width; ++i)
			{
				for (int c = 0; c < img.nChannel; ++c)
				{
					int offset = j * img.widthStep + img.nChannel * i + c;
					int offset_2 = j * img_2.widthStep + img.nChannel * i + c;

					if (abs(img.data[offset] - img_2.data[offset_2]) > eps)
					{
						printf("\n[TEST ACCURACY] y = %d, x = %d", j, i);
						printf("\n[TEST ACCURACY] img = %f, img_2 = %f\n", float(img.data[offset]), float(img_2.data[offset_2]));
						return -1;
					}
				}
			}
		}

		return 0;
	}

#ifdef USE_CUDA
	template<typename type, const int pinned_mem = 0>
	int init_data(CUDA::TmpImage<type, pinned_mem>& img_cu, SIMD::TmpImage<type>& img)
	{
		for (int j = 0; j < img_cu.height; ++j)
		{
			for (int i = 0; i < img_cu.width; ++i)
			{
				for (int c = 0; c < img_cu.nChannel; ++c)
				{
					int offset_cu = j * img_cu.widthStepHost + img.nChannel * i + c;
					int offset = j * img.widthStep + img.nChannel * i + c;
					img_cu.dataHost[offset_cu] = img.data[offset];
				}
			}
		}
		img_cu.updateDataDevice();
		return 0;
	}
	
	template<typename type, const int pinned_mem = 0>
	int check_data(SIMD::TmpImage<type>& img, CUDA::TmpImage<type, pinned_mem>& img_cu, float eps = 1.E-3)
	{
		img_cu.updateDataHost();

		for (int j = 0; j < img.height; ++j)
		{
			for (int i = 0; i < img.width; ++i)
			{
				for (int c = 0; c < img.nChannel; ++c)
				{
					int offset = j * img.widthStep + img.nChannel * i + c;
					int offset_cu = j * img_cu.widthStepHost + img.nChannel * i + c;

					if (abs(img.data[offset] - img_cu.dataHost[offset_cu]) > eps)
					{
						printf("\n[TEST ACCURACY] y = %d, x = %d", j, i);
						printf("\n[TEST ACCURACY] img = %f, img_cu = %f\n", float(img.data[offset]), float(img_cu.dataHost[offset_cu]));
						return -1;
					}
				}
			}
		}

		return 0;
	}
#endif

#ifdef USE_CL
	template<typename type>
	int init_data(CL::TmpImage<type, 0>& img_cl, SIMD::TmpImage<type>& img)
	{
		for (int j = 0; j < img_cl.height; ++j)
		{
			for (int i = 0; i < img_cl.width; ++i)
			{
				for (int c = 0; c < img_cl.nChannel; ++c)
				{
					int offset_cl = j * img_cl.widthStepHost + img.nChannel * i + c;
					int offset = j * img.widthStep + img.nChannel * i + c;
					img_cl.dataHost[offset_cl] = img.data[offset];
				}
			}
		}
		img_cl.updateDataDevice();
		return 0;
	}

	template<typename type>
	int check_data(SIMD::TmpImage<type>& img, CL::TmpImage<type, 0>& img_cl, float eps = 1.E-3)
	{
		img_cl.updateDataHost();

		for (int j = 0; j < img.height; ++j)
		{
			for (int i = 0; i < img.width; ++i)
			{
				for (int c = 0; c < img.nChannel; ++c)
				{
					int offset = j * img.widthStep + img.nChannel * i + c;
					int offset_cl = j * img_cl.widthStepHost + img.nChannel * i + c;

					if (abs(img.data[offset] - img_cl.dataHost[offset_cl]) > eps)
					{
						printf("\n[TEST ACCURACY] y = %d, x = %d", j, i);
						printf("\n[TEST ACCURACY] img = %f, img_cl = %f\n", float(img.data[offset]), float(img_cl.dataHost[offset_cl]));
						return -1;
					}
				}
			}
		}

		return 0;
	}
#endif

	int test(bool memory = 1, bool converter = 1, bool equalize = 1, bool resizer = 1, bool cnn = 1, bool format = 1)
	{
		printf("TEST ACCURACY\n\n");

		CNNDetector::Param param;
		CNNDetector::AdvancedParam ad_param;
		CNNGPUD = new CNNDetector(&param, &ad_param);

#ifdef USE_CUDA
		if (CUDA::InitDevice(0, true) < 0)
		{
			return -1;
		}
#endif

#ifdef USE_CL
		CL::InfoDevice();
		cl_device_id device;
		cl_context context;
		cl_command_queue queue;
		if (CL::InitDevice(-1, -1, device, context, queue, true) < 0)
		{
			return -1;
		}
#endif

		//allocate memory
		printf("\n[TEST ACCURACY]  test allocate memory\n");
		if (memory)
		{
			for (int i = 1; i <= 3; ++i)
			{
				int cannels = 1;
				switch (i)
				{
				case 2:
					cannels = 3;
					break;
				case 3:
					cannels = 4;
					break;
				}
				printf("[TEST ACCURACY] 	cannels = %d: ", cannels);

				Size size(999, 999);
				SIMD::Image_8u img_rgb_8u(size.width, size.height, cannels, ALIGN_DEF, true);
				init_data<uchar_>(img_rgb_8u);

#ifdef USE_CUDA
				CUDA::Image_8u img_rgb_8u_cu(size.width, size.height, cannels);
				init_data<uchar_>(img_rgb_8u_cu, img_rgb_8u);
				if (check_data<uchar_>(img_rgb_8u, img_rgb_8u_cu) < 0) return -1;
				
				int offset = 19 * img_rgb_8u.widthStep + cannels * 87;
				SIMD::Image_8u img_temp(179, 179, cannels, img_rgb_8u.data + offset, img_rgb_8u.widthStep);
				
				CUDA::Image_8u img_temp_cu(365, 348, cannels);
				img_temp_cu.offsetDevice = 57 * img_temp_cu.widthStepDevice + cannels * 74;
				
				int offset_cu = 19 * img_rgb_8u_cu.widthStepDevice + cannels * 87;
				img_rgb_8u_cu.offsetDevice = offset_cu;
				img_rgb_8u_cu.width = 179;
				img_rgb_8u_cu.height = 179;

				img_temp_cu.copyDataDeviceToDevice(&img_rgb_8u_cu, true);
				img_temp_cu.updateDataHost();

				int offset2_cu = 57 * img_temp_cu.widthStepHost + cannels * 74;
				SIMD::Image_8u img_temp2(179, 179, cannels, img_temp_cu.dataHost + offset2_cu, img_temp_cu.widthStepHost);
				if (check_data<uchar_>(img_temp, img_temp2) < 0) return -1;
#endif

#ifdef USE_CL
				CL::Image_8u img_rgb_8u_cl(context, queue, size.width, size.height, cannels);
				init_data<uchar_>(img_rgb_8u_cl, img_rgb_8u);
				if (check_data<uchar_>(img_rgb_8u, img_rgb_8u_cl) < 0) return -1;

				int offset = 19 * img_rgb_8u.widthStep + cannels * 87;
				SIMD::Image_8u img_temp(179, 179, cannels, img_rgb_8u.data + offset, img_rgb_8u.widthStep);

				CL::Image_8u img_temp_cl(context, queue, 365, 348, cannels);
				img_temp_cl.offsetDevice = 57 * img_temp_cl.widthStepDevice + cannels * 74;

				int offset_cl = 19 * img_rgb_8u_cl.widthStepDevice + cannels * 87;
				img_rgb_8u_cl.offsetDevice = offset_cl;
				img_rgb_8u_cl.width = 179;
				img_rgb_8u_cl.height = 179;

				img_temp_cl.copyDataDeviceToDevice(&img_rgb_8u_cl);
				img_temp_cl.updateDataHost();

				int offset2_cl = 57 * img_temp_cl.widthStepHost + cannels * 74;
				SIMD::Image_8u img_temp2(179, 179, cannels, img_temp_cl.dataHost + offset2_cl, img_temp_cl.widthStepHost);
				if (check_data<uchar_>(img_temp, img_temp2) < 0) return -1;
#endif
				
				printf("success\n");
			}
		}

		//image converter
		printf("\n[TEST ACCURACY]  test image converter\n");
		if (converter)
		{
			const float kernel_matrix_col[3] = { 0.106450774f, 0.786570728f, 0.106450774f };
			const float kernel_matrix_row[3] = { 0.106450774f, 0.786570728f, 0.106450774f };

			float* col_filter3_kernel = (float*)SIMD::mm_malloc(8 * sizeof(float), ALIGN_DEF);
			for (int i = 0; i < 3; ++i)
			{
				col_filter3_kernel[i] = kernel_matrix_col[i];
				col_filter3_kernel[i + 4] = kernel_matrix_col[i];
			}
			col_filter3_kernel[3] = 0.f;
			col_filter3_kernel[7] = 0.f;

			float* row_filter3_kernel = (float*)SIMD::mm_malloc(8 * sizeof(float), ALIGN_DEF);
			for (int i = 0; i < 3; ++i)
			{
				row_filter3_kernel[i] = kernel_matrix_row[i];
				row_filter3_kernel[i + 4] = kernel_matrix_row[i];
			}
			row_filter3_kernel[3] = 0.f;
			row_filter3_kernel[7] = 0.f;

#ifdef USE_CUDA
			CUDA::ImageConverter::Init();
#endif
#ifdef USE_CL
			CL::ImageConverter::create_cl_program(device, context);
#endif

			for (int i = 1; i <= 3; ++i)
			{
				int cannels = 1;
				switch (i)
				{
				case 2:
					cannels = 3;
					break;
				case 3:
					cannels = 4;
					break;
				}
				printf("[TEST ACCURACY] 	cannels = %d: ", cannels);

				Size size(1023, 1077);
				SIMD::Image_8u img_rgb_8u(size.width, size.height, cannels, ALIGN_DEF, true);
				SIMD::Image_32f img_32f(size.width, size.height);
				init_data<uchar_>(img_rgb_8u);

				if (cannels == 1)
				{
					SIMD::Image_8u img_8u(size.width, size.height, cannels, ALIGN_DEF, true);
					img_8u.copyData(size.width, size.height, img_rgb_8u.data, img_rgb_8u.widthStep);

					img_32f.erase();
					SIMD::ImageConverter::UCharToFloat(img_32f, img_8u);
					img_8u.erase();
					SIMD::ImageConverter::FloatToUChar(img_8u, img_32f, Rect(0, 0, img_32f.width, img_32f.height));
					img_32f.erase();
					SIMD::ImageConverter::UCharToFloat_inv(img_32f, img_8u);
					img_8u.erase();
					SIMD::ImageConverter::FloatToUChar(img_8u, img_32f, Rect(0, 0, img_32f.width, img_32f.height));
					img_32f.erase();
					SIMD::ImageConverter::UCharToFloat_inv(img_32f, img_8u);
					img_8u.erase();
					SIMD::ImageConverter::FloatToUChar(img_8u, img_32f, Rect(0, 0, img_32f.width, img_32f.height));

					SIMD::TmpImage<char> img_urnd1(size.width, size.height, ALIGN_SSE);
					SIMD::TmpImage<char> img_urnd2(size.width, size.height, ALIGN_SSE);
					const float rnd = -30.f / (float)RAND_MAX;
					char* ptr1 = img_urnd1.data;
					char* ptr2 = img_urnd2.data;
					for (int k = 0; k < img_urnd1.widthStep * img_urnd1.height; ++k)
					{
						const char d = char(15.f + rnd * (float)rand());
						*ptr1++ = d;
						*ptr2++ = -d;
					}

					img_32f.erase();
					SIMD::ImageConverter::UCharToFloat_add_rnd(img_32f, img_8u, img_urnd1);
					img_8u.erase();
					SIMD::ImageConverter::FloatToUChar(img_8u, img_32f, Rect(0, 0, img_32f.width, img_32f.height));
					img_32f.erase();
					SIMD::ImageConverter::UCharToFloat_add_rnd(img_32f, img_8u, img_urnd2);
					img_8u.erase();
					SIMD::ImageConverter::FloatToUChar(img_8u, img_32f, Rect(0, 0, img_32f.width, img_32f.height));

					if (check_data<uchar_>(img_rgb_8u, img_8u) < 0) return -1;
				}
				
				SIMD::ImageConverter::Img8uToImg32fGRAY(img_32f, img_rgb_8u, 4);

#ifdef USE_CUDA
				CUDA::Image_8u img_rgb_8u_cu(size.width, size.height, cannels * size.width, cannels * roundUpMul(size.width, 32), size.height, ALIGN_DEF);
				img_rgb_8u_cu.nChannel = cannels;

				CUDA::Image_32f img_32f_cu(size.width, size.height, size.width, roundUpMul(size.width, 32), size.height, ALIGN_DEF);
				init_data<uchar_>(img_rgb_8u_cu, img_rgb_8u);

				CUDA::ImageConverter::Img8uToImg32fGRAY((CUDA::Image_32f_pinned*)&img_32f_cu, (CUDA::Image_8u_pinned*)&img_rgb_8u_cu);
				if (check_data<float>(img_32f, img_32f_cu) < 0) return -1;
				
				CUDA::ImageConverter::Img8uToImg32fGRAY_tex((CUDA::Image_32f_pinned*)&img_32f_cu, (CUDA::Image_8u_pinned*)&img_rgb_8u_cu);
				if (check_data<float>(img_32f, img_32f_cu, 1.E-2) < 0) return -1;
#endif

#ifdef USE_CL
				CL::Image_8u img_rgb_8u_cl(context, queue, size.width, size.height, cannels);
				CL::Image_32f img_32f_cl(context, queue, size.width, size.height);
				init_data<uchar_>(img_rgb_8u_cl, img_rgb_8u);

				CL::ImageConverter::Img8uToImg32fGRAY(&img_32f_cl, &img_rgb_8u_cl, queue);
				if (check_data<float>(img_32f, img_32f_cl) < 0) return -1;
#endif

				SIMD::ImageConverter::Img8uToImg32fGRAY_blur(img_32f, img_rgb_8u, col_filter3_kernel, row_filter3_kernel, 1);

				img_32f.width -= 2;
				img_32f.height -= 2;

#ifdef USE_CUDA
				CUDA::ImageConverter::Img8uToImg32fGRAY_blur((CUDA::Image_32f_pinned*)&img_32f_cu, (CUDA::Image_8u_pinned*)&img_rgb_8u_cu, col_filter3_kernel, row_filter3_kernel);
				if (check_data<float>(img_32f, img_32f_cu) < 0) return -1;

				CUDA::ImageConverter::Img8uToImg32fGRAY_blur_tex((CUDA::Image_32f_pinned*)&img_32f_cu, (CUDA::Image_8u_pinned*)&img_rgb_8u_cu, col_filter3_kernel, row_filter3_kernel);
				if (check_data<float>(img_32f, img_32f_cu, 1.E-2) < 0) return -1;
#endif

#ifdef USE_CL
				CL::Image_32f img_buff_32f_cl(context, queue, size.width, size.height);
				CL::ImageConverter::Img8uToImg32fGRAY_blur(&img_32f_cl, &img_rgb_8u_cl, col_filter3_kernel, row_filter3_kernel, &img_buff_32f_cl, queue);
				if (check_data<float>(img_32f, img_32f_cl) < 0) return -1;
#endif

				printf("success\n");
			}

			SIMD::mm_free(col_filter3_kernel);
			SIMD::mm_free(row_filter3_kernel);

#ifdef USE_CL
			CL::ImageConverter::destroy_cl_program();
#endif
		}

		printf("\n[TEST ACCURACY]  test image equalize\n");
		if (equalize)
		{
			Size size(1024, 1024);
			SIMD::Image_8u img_8u(size.width, size.height, 1, ALIGN_DEF, true);
			for (int j = 0; j < img_8u.height; ++j)
			{
				uchar_* pData = img_8u.data + j * img_8u.widthStep;
				for (int i = 0; i < img_8u.width; ++i)
				{
					if (j > img_8u.height / 2 && i < img_8u.width / 2)* pData++ = 0;
					if (j < img_8u.height / 2 && i > img_8u.width / 2)* pData++ = 1;
					if (j > img_8u.height / 2 && i > img_8u.width / 2)* pData++ = 2;
					if (j < img_8u.height / 2 && i < img_8u.width / 2)* pData++ = 3;
				}
			}

			std::vector<int> hist(256);
			for (int j = 0; j < img_8u.height; ++j)
			{
				uchar_* pData = img_8u.data + j * img_8u.widthStep;
				for (int i = 0; i < img_8u.width; ++i)
				{
					hist[*pData++]++;
				}
			}

			printf("[TEST ACCURACY] 	hist: ");
			for (int k = 0; k < 256; ++k)
			{
				if (hist[k] != 0) printf("%d ", k);
			}
			printf("\n");

			SIMD::equalizeImage(img_8u);

			std::vector<int> hist_eq(256);
			for (int j = 0; j < img_8u.height; ++j)
			{
				uchar_* pData = img_8u.data + j * img_8u.widthStep;
				for (int i = 0; i < img_8u.width; ++i)
				{
					hist_eq[*pData++]++;
				}
			}

			printf("[TEST ACCURACY] 	equalize: ");
			int L_eq = 0;
			int n_bin_eq = 0;
			for (int k = 0; k < 256; ++k)
			{
				if (hist_eq[k] != 0)
				{
					L_eq += k - n_bin_eq;
					n_bin_eq = k;
					printf("%d ", k);
				}
			}
			printf("\n");

			if (256 - L_eq <= 2) printf("[TEST ACCURACY] 	success\n");
			else return -1;
		}

		printf("\n[TEST ACCURACY]  test image resizer\n");
		if (resizer)
		{
			Size size_in(1017, 1017);

			SIMD::ImageResizer image_resizer(size_in, size_in);

#ifdef USE_CUDA
			CUDA::ImageResizer::Init();
#endif
#ifdef USE_CL
			CL::ImageResizer::create_cl_program(device, context);
#endif

			printf("[TEST ACCURACY] 	size_in = (%d, %d): \n", size_in.width, size_in.height);
			for (int i = 1; i <= 30; i += 3)
			{
				Size size_out(50 * i, 50 * i);

				SIMD::Image_32f img_1_32f(size_in.width, size_in.height);
				SIMD::Image_32f	img_2_32f(size_out.width, size_out.height);
				init_data<float>(img_1_32f);		
				
				SIMD::Image_8u img_1_8u(size_in.width, size_in.height);
				SIMD::Image_8u	img_2_8u(size_out.width, size_out.height);
				init_data<uchar_>(img_1_8u, img_1_32f);

#ifdef USE_CUDA
				CUDA::Image_32f_pinned img_1_32f_cu(size_in.width, size_in.height, size_in.width, roundUpMul(size_in.width, 32), ALIGN_DEF);
				CUDA::Image_32f	img_2_32f_cu(size_out.width, size_out.height);
				init_data<float>(img_1_32f_cu, img_1_32f);

				img_1_32f_cu.updateDataHost();
				img_1_32f_cu.updateDataDevice();
#endif

#ifdef USE_CL
				CL::Image_32f img_1_32f_cl(context, queue, size_in.width, size_in.height);
				CL::Image_32f img_2_32f_cl(context, queue, size_out.width, size_out.height);
				init_data<float>(img_1_32f_cl, img_1_32f);
#endif

				printf("[TEST ACCURACY] 	size_out = (%d, %d)\n", size_out.width, size_out.height);
				printf("[TEST ACCURACY] 	NearestNeighbor\n");

				Timer timer(1, true);
				image_resizer.FastImageResize(img_2_8u, img_1_8u, 0, 4);
				printf("[TEST ACCURACY] 		simd 8u  %7.3f ms \n", timer.get(1000));
				img_2_8u.width--;
				img_2_8u.height--;

				image_resizer.FastImageResize(img_2_32f, img_1_32f, 0, 4);
				printf("[TEST ACCURACY] 		simd 32f %7.3f ms \n", timer.get(1000));
				img_2_32f.width--;
				img_2_32f.height--;

#ifdef USE_CUDA
				CUDA::Timer timer_cuda(true);
				CUDA::ImageResizer::FastImageResize(&img_2_32f_cu, &img_1_32f_cu, 0);
				printf("[TEST ACCURACY] 		cuda %7.3f ms \n", timer_cuda.get(1000));
				if (check_data<float>(img_2_32f, img_2_32f_cu, 1.f) < 0) return -1;

				SIMD::ImageConverter::UCharToFloat(img_2_32f, img_2_8u);
				if (check_data<float>(img_2_32f, img_2_32f_cu, 1.f) < 0) return -1;
#endif

#ifdef USE_CL
				CL::Timer timer_cl(queue, true);
				CL::ImageResizer::FastImageResize(&img_2_32f_cl, &img_1_32f_cl, 0, queue);
				printf("[TEST ACCURACY] 		cl   %7.3f ms\n", timer_cl.get(1000));
				if (check_data<float>(img_2_32f, img_2_32f_cl, 1.f) < 0) return -1;

				SIMD::ImageConverter::UCharToFloat(img_2_32f, img_2_8u);
				if (check_data<float>(img_2_32f, img_2_32f_cl, 1.f) < 0) return -1;
#endif

				img_2_8u.width++;
				img_2_8u.height++;
				img_2_32f.width++;
				img_2_32f.height++;

				printf("[TEST ACCURACY] 	Bilinear\n");

				timer.start();
				image_resizer.FastImageResize(img_2_8u, img_1_8u, 1, 4);
				printf("[TEST ACCURACY] 		simd 8u  %7.3f ms\n", timer.get(1000));
				img_2_8u.width--;
				img_2_8u.height--;

				timer.start();
				image_resizer.FastImageResize(img_2_32f, img_1_32f, 1, 4);
				printf("[TEST ACCURACY] 		simd 32f %7.3f ms\n", timer.get(1000));
				img_2_32f.width--;
				img_2_32f.height--;

#ifdef USE_CUDA
				timer_cuda.start();
				CUDA::ImageResizer::FastImageResize(&img_2_32f_cu, &img_1_32f_cu, 1);
				printf("[TEST ACCURACY] 		cuda %7.3f ms\n", timer_cuda.get(1000));
				if (check_data<float>(img_2_32f, img_2_32f_cu, 1.f) < 0) return -1;

				SIMD::ImageConverter::UCharToFloat(img_2_32f, img_2_8u);
				if (check_data<float>(img_2_32f, img_2_32f_cu, 2.f) < 0) return -1;
#endif

#ifdef USE_CL
				timer_cl.start();
				CL::ImageResizer::FastImageResize(&img_2_32f_cl, &img_1_32f_cl, 1, queue);
				printf("[TEST ACCURACY] 		cl   %7.3f ms\n", timer_cl.get(1000));
				if (check_data<float>(img_2_32f, img_2_32f_cl, 1.f) < 0) return -1;

				SIMD::ImageConverter::UCharToFloat(img_2_32f, img_2_8u);
				if (check_data<float>(img_2_32f, img_2_32f_cl, 2.f) < 0) return -1;
#endif

				printf("[TEST ACCURACY] 	success\n\n");
			}

#ifdef USE_CL
			CL::ImageResizer::destroy_cl_program();
#endif
		}
		
		printf("\n[TEST ACCURACY]  test cnn\n");
		if (cnn)
		{
#ifndef USE_CNTK_MODELS
			std::string model = DUMP::get_serialized_model("cnn4face1_new.bin");
#else
			std::string model = DUMP::get_serialized_model("cnn4face1_cntk.bin");
#endif

			Size init_size(3997, 3997);

#ifdef USE_AVX
			SIMD::ConvNeuralNetwork_v2* cnn_simd = new SIMD::ConvNeuralNetwork_v2();
#else
			SIMD::ConvNeuralNetwork* cnn_simd = new SIMD::ConvNeuralNetwork();
#endif

			const int index_output = 1;
			cnn_simd->Init(model, index_output, CNNGPUD->hGrd);
			if (cnn_simd->isEmpty())
			{
				delete cnn_simd;
				cnn_simd = NULL;
				return -1;
			}
			cnn_simd->AllocateMemory(init_size);
			cnn_simd->setNumThreads(4);

			SIMD::Image_32f img(init_size.width, init_size.height, ALIGN_DEF, true);
			SIMD::Image_32f resp(cnn_simd->getOutputImgSize(init_size).width, cnn_simd->getOutputImgSize(init_size).height);
			init_data<float>(img);

#ifdef USE_CUDA
			CUDA::ConvNeuralNetwork* cnn_cuda = new CUDA::ConvNeuralNetwork();
			cnn_cuda->Init(model, index_output, CNNGPUD->hGrd);
			if (cnn_cuda->isEmpty())
			{
				delete cnn_cuda;
				cnn_cuda = NULL;
				return -1;
			}
			cnn_cuda->AllocateMemory(init_size);

			CUDA::Image_32f img_cu(init_size.width, init_size.height, init_size.width,
				roundUpMul(init_size.width, cnn_cuda->getBlockSize().width), 
				roundUpMul(init_size.height, cnn_cuda->getBlockSize().height),
				ALIGN_DEF);
			CUDA::Image_32f_pinned resp_cu(cnn_cuda->getOutputImgSize(init_size).width, cnn_cuda->getOutputImgSize(init_size).height);
			init_data<float>(img_cu, img);
#endif

#ifdef USE_CL
			CL::ConvNeuralNetwork* cnn_cl = new CL::ConvNeuralNetwork();
			cnn_cl->Init(model, index_output, device, context, queue);
			if (cnn_cl->isEmpty())
			{
				delete cnn_cl;
				cnn_cl = NULL;
				return -1;
			}
			cnn_cl->AllocateMemory(init_size);

			CL::Image_32f img_cl(context, queue, init_size.width, init_size.height, init_size.width, 
				addRoundUpMul(init_size.width, cnn_cl->getBlockSize().width), 
				addRoundUpMul(init_size.height, cnn_cl->getBlockSize().height), 
				ALIGN_DEF);
			CL::Image_32f resp_cl(context, queue, 
				cnn_cl->getOutputImgSize(init_size).width, 
				cnn_cl->getOutputImgSize(init_size).height, 
				cnn_cl->getOutputImgSize(init_size).width, 
				roundUpMul(cnn_cl->getOutputImgSize(init_size).width, cnn_cl->getBlockSize().width),
				roundUpMul(cnn_cl->getOutputImgSize(init_size).height, cnn_cl->getBlockSize().height),
				ALIGN_DEF);
			init_data<float>(img_cl , img);
#endif

			//std::remove(path_model.c_str());

#ifdef CHECK_TEST
			Legacy::ConvNeuralNetwork* old_cnn = new Legacy::ConvNeuralNetwork(MODELSPATH"cnn4face1_old.bin");
			if (old_cnn->isEmpty())
			{
				delete old_cnn;
				old_cnn = NULL;
				return -1;
			}

			cnn_simd->setCNNRef(old_cnn);
			CUDA_CODE(cnn_cuda->setCNNRef(old_cnn);)
			CL_CODE(cnn_cl->setCNNRef(old_cnn);)
#endif

			printf("[TEST ACCURACY] 	init_size = (%d, %d): \n", init_size.width, init_size.height);
			for (int i = 0; i <= 61; ++i)
			{
				Size size(50 * i + i, 50 * i + i);
				printf("[TEST ACCURACY] 	size = (%d, %d):\n", size.width, size.height);

				img.width = size.width;
				img.height = size.height;
				init_data(img);

				Timer timer(1, true);
				cnn_simd->Forward(resp, img);
				printf("[TEST ACCURACY] 		cnn_simd %7.3f ms\n", timer.get(1000));

#ifdef USE_CUDA
				img_cu.width = size.width;
				img_cu.height = size.height;
				init_data<float>(img_cu, img);

				CUDA::Timer timer_cuda(true);
				cnn_cuda->Forward(&resp_cu, &img_cu);
				printf("[TEST ACCURACY] 		cnn_cuda %7.3f ms\n", timer_cuda.get(1000));
				if (check_data<float>(resp, resp_cu, 1.E-2) < 0) return -1;
#endif

#ifdef USE_CL
				img_cl.width = size.width;
				img_cl.height = size.height;
				init_data<float>(img_cl, img);

				CL::Timer timer_cl(queue, true);
				cnn_cl->Forward(&resp_cl, &img_cl);
				printf("[TEST ACCURACY] 		cnn_cl   %7.3f ms\n", timer_cl.get(1000));
				if (check_data<float>(resp, resp_cl, 1.E-2) < 0) return -1;
#endif

				printf("[TEST ACCURACY] 	success\n\n");
			}

			delete cnn_simd;

#ifdef USE_CUDA
			delete cnn_cuda;
			CUDA::ReleaseDevice();
#endif

#ifdef USE_CL	
			delete cnn_cl;
			CL::ReleaseDevice(device, context, queue);
#endif

#ifdef CHECK_TEST
			delete old_cnn;
#endif
		}

#if 0 && defined(CHECK_TEST) && !defined(USE_CNTK_MODELS)
		printf("\n[TEST ACCURACY]  test file format\n");
		if (format)
		{
			std::string path = MODELSPATH;
			std::string path_model = path + "cnn4face1_new.bin";
			std::string path_model_leg = path + "cnn4face1_old.bin";

			Size init_size(400, 400);
			SIMD::Image_32f img(init_size.width, init_size.height, ALIGN_DEF, true);
			init_data<float>(img);

			Legacy::ConvNeuralNetwork cnn_leg(path_model_leg);
			if (cnn_leg.isEmpty()) return -1;
			cnn_leg.AllocateMemory(init_size);
			cnn_leg.setNumThreads(4);
			SIMD::Image_32f resp_leg(cnn_leg.getOutputImgSize(init_size).width, cnn_leg.getOutputImgSize(init_size).height);

#ifdef USE_AVX
			SIMD::ConvNeuralNetwork_v2 cnn_simd;
#else
			SIMD::ConvNeuralNetwork cnn_simd;
#endif
			cnn_simd.Init(path_model, CNNGPUD->hGrd);
			if (cnn_simd.isEmpty())
			{
				return -1;
			}
			cnn_simd.AllocateMemory(init_size);
			cnn_simd.setNumThreads(4);
			cnn_simd.setCNNRef(&cnn_leg);
			SIMD::Image_32f resp_simd(cnn_simd.getOutputImgSize(init_size).width, cnn_simd.getOutputImgSize(init_size).height);

			SIMD::ConvNeuralNetwork cnn_simd_leg;
			cnn_simd_leg.Init(&cnn_leg, true, true);
			if (cnn_simd.isEmpty())
			{
				return -1;
			}
			cnn_simd_leg.AllocateMemory(init_size);
			cnn_simd_leg.setNumThreads(4);
			cnn_simd_leg.setCNNRef(&cnn_leg);
			SIMD::Image_32f resp_simd_leg(cnn_simd_leg.getOutputImgSize(init_size).width, cnn_simd_leg.getOutputImgSize(init_size).height);

			Timer timer(1, true);
			cnn_simd.Forward(resp_simd, img);
			printf("[TEST ACCURACY] 		cnn_simd %7.3f ms\n", timer.get(1000));

			timer.start();
			cnn_leg.Forward(resp_leg, img);
			printf("[TEST ACCURACY] 		cnn_leg %7.3f ms\n", timer.get(1000));

			timer.start();
			cnn_simd_leg.Forward(resp_simd_leg, img);
			printf("[TEST ACCURACY] 		cnn_simd_leg %7.3f ms\n", timer.get(1000));

			if (check_data<float>(resp_simd, resp_leg, 1.E-2) < 0) return -1;
			if (check_data<float>(resp_simd, resp_simd_leg, 1.E-2) < 0) return -1;

			//-----------------------------------------------------------------------------

			cnn_simd_leg.SaveToBinaryFile("temp");
			cnn_simd_leg.Clear();
			cnn_simd_leg.Init("temp.bin");
			cnn_simd_leg.AllocateMemory(init_size);
			cnn_simd_leg.setNumThreads(4);
			cnn_simd_leg.setCNNRef(&cnn_leg);
			resp_simd_leg.erase();

			timer.start();
			cnn_simd_leg.Forward(resp_simd_leg, img);
			printf("[TEST ACCURACY] 		cnn_simd_leg %7.3f ms\n", timer.get(1000));

			if (check_data<float>(resp_simd, resp_simd_leg, 1.E-2) < 0) return -1;

			printf("[TEST ACCURACY] 	success\n\n");
		}
#endif

		delete CNNGPUD;

		return 0;
	}

//--------------------------------------------------------------------------------------------------------

#ifdef _DEBUG
#include <crtdbg.h>
#define _CRTDBG_MAP_ALLOC

	void MemoryTest()
	{
		_CrtMemState _ms;
		_CrtMemCheckpoint(&_ms);

		{
			printf("[MemoryTest] Start\n");

			CNNDetector::Param param;
			CNNDetector::AdvancedParam ad_param;
			CNNGPUD = new CNNDetector(&param, &ad_param);

			SIMD::Image_8u img_8u(1920, 1080, 3, 1, false);
			std::vector<CNNDetector::Detection> detection;

			CNNGPUD->setMaxImageSize(NeuralNetworksLib::Size(320, 240));
			CNNGPUD->Detect(detection, img_8u);
			printf("Size(320, 240)\n");

			CNNGPUD->setMaxImageSize(NeuralNetworksLib::Size(640, 480));
			CNNGPUD->Detect(detection, img_8u);
			printf("Size(640, 480)\n");

			CNNGPUD->setMaxImageSize(NeuralNetworksLib::Size(800, 600));
			CNNGPUD->Detect(detection, img_8u);
			printf("Size(800, 600)\n");

			CNNGPUD->setMaxImageSize(NeuralNetworksLib::Size(1280, 720));
			CNNGPUD->Detect(detection, img_8u);
			printf("Size(1280, 720)\n");

			CNNGPUD->setMaxImageSize(NeuralNetworksLib::Size(1920, 1080));
			CNNGPUD->Detect(detection, img_8u);
			printf("Size(1920, 1080)\n");

			//delete CNNGPUD;

			std::vector<SIMD::ConvNeuralNetwork*> cnn_simd;
			cnn_simd.push_back(new SIMD::ConvNeuralNetwork());
			cnn_simd[0]->Init(DUMP::get_serialized_model("cnn4face1_cntk.bin"));
			delete cnn_simd[0];
			cnn_simd.clear();

			printf("[MemoryTest] End\n\n");
		}

		_CrtMemDumpAllObjectsSince(&_ms);
		_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
		_CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
	}
#endif

//--------------------------------------------------------------------------------------------------------

	int main(int argc, char* argv[])
	{
#ifdef _DEBUG
		MemoryTest();
#endif

		if (test(1, 1, 1, 1, 1, 1) == 0)
		{
			printf("\n[TEST ACCURACY] SUCCESS\n");
		}
		else
		{
			printf("\n[TEST ACCURACY] FAILED\n");
		}
		system("pause");
	}



