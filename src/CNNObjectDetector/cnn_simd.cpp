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


#include "cnn_simd.h"
#include <fstream>
#include <sstream>
#include <iterator>

#ifdef USE_OMP
#	include <omp.h>
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{
#ifndef USE_CNTK_MODELS

	namespace SIMD
	{
		void ConvNeuralNetwork::Init(std::string file_name, int index_output, void* hGrd)
		{
			std::stringstream data_bin;
			if (file_name.size() < 255)
			{
				std::fstream data_bin;
				data_bin.open(file_name.c_str(), std::fstream::binary | std::fstream::in);

				if (!data_bin.is_open())
				{
					printf("[SIMD::CNN] Configuration file not found!\n");
					return;
				}

				data_bin << data_bin.rdbuf();
				data_bin.close();
			}
			else
			{
				data_bin << file_name;
			}

			//version
			float format_version = 0.0f;
			FB_READ(data_bin, format_version);

			if (format_version > 1.0f)
			{
				printf("[SIMD::CNN] Configuration file format is not supported!\n");
				return;
			}

			//max pool
			FB_READ(data_bin, cnn.max_pool);

			//min input size
			FB_READ(data_bin, cnn.min_image_size.width);
			FB_READ(data_bin, cnn.min_image_size.height);

			//initial buffers
			FB_READ(data_bin, cnn.layer_count);
			cnn.layer_buffer.resize(cnn.layer_count);
			for (int i = 0; i < cnn.layer_count; ++i)
			{
				FB_READ(data_bin, cnn.layer_buffer[i].map_count);

				cnn.layer_buffer[i].conv_buffer.resize(cnn.layer_buffer[i].map_count);
				cnn.layer_buffer[i].pool_buffer.resize(cnn.layer_buffer[i].map_count);
				cnn.layer_buffer[i].sum_buffer.resize(cnn.layer_buffer[i].map_count);
			}

			//initial weight
			//conv kernels l1
			int kernel_width = 0;
			int kernel_height = 0;
			FB_READ(data_bin, kernel_width);
			FB_READ(data_bin, kernel_height);

			cnn.conv_l1.size = Size2d(kernel_width, kernel_height);
			cnn.conv_l1.kernels.resize(cnn.layer_buffer[0].map_count);

#if defined(USE_SSE) || defined(USE_AVX)
			int iBufferSize = MAX(1, REG_SIZE / 4) * kernel_width * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				cnn.conv_l1.kernels[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			for (int k = 0; k < cnn.layer_buffer[0].map_count; ++k)
			{
				int t = -(MAX(1, REG_SIZE / 4) - 1) * 4;
				for (int i = 0; i < cnn.conv_l1.size.size; ++i)
				{
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);

					if (i % 4 == 0) t += (MAX(1, REG_SIZE / 4) - 1) * 4;

					for (int p = 0; p < REG_SIZE; p += 4)
					{
						cnn.conv_l1.kernels[k][t + i + p] = kernel_val;
					}
				}
			}
#else
			int iBufferSize = kernel_width * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				cnn.conv_l1.kernels[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			for (int k = 0; k < cnn.layer_buffer[0].map_count; ++k)
			{
				for (int i = 0; i < cnn.conv_l1.size.size; ++i)
				{
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);

					cnn.conv_l1.kernels[k][i] = kernel_val;
				}
			}
#endif

			//conv kernels l2
			FB_READ(data_bin, kernel_width);
			FB_READ(data_bin, kernel_height);

			cnn.conv_l2.size = Size2d(kernel_width, kernel_height);
			cnn.conv_l2.kernels.resize(cnn.layer_buffer[1].map_count);

#if defined(USE_SSE) || defined(USE_AVX)
			iBufferSize = MAX(1, REG_SIZE / 4) * (kernel_width + 1) * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				cnn.conv_l2.kernels[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			for (int k = 0; k < cnn.layer_buffer[1].map_count; ++k)
			{
				int t = -(MAX(1, REG_SIZE / 4) - 1) * 4;
				for (int i = 0; i < cnn.conv_l2.size.size; ++i)
				{
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);

					if (i % 3 == 0)
					{
						t += (MAX(1, REG_SIZE / 4) - 1) * 4;
					}

					for (int p = 0; p < REG_SIZE; p += 4)
					{
						cnn.conv_l2.kernels[k][t + i + p] = kernel_val;
					}

					if ((i + 1) % 3 == 0)
					{
						t++;
						for (int p = 0; p < REG_SIZE; p += 4)
						{
							cnn.conv_l2.kernels[k][t + i + p] = 0.f;
						}
					}
				}
			}
#else
			iBufferSize = kernel_width * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				cnn.conv_l2.kernels[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			for (int k = 0; k < cnn.layer_buffer[1].map_count; ++k)
			{
				for (int i = 0; i < cnn.conv_l2.size.size; ++i)
				{
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);

					cnn.conv_l2.kernels[k][i] = kernel_val;
				}
			}
#endif

			//conv kernels l3
			FB_READ(data_bin, kernel_width);
			FB_READ(data_bin, kernel_height);

			cnn.conv_l3.size = Size2d(kernel_width, kernel_height);
			cnn.conv_l3.kernels.resize(cnn.layer_buffer[2].map_count);

#if defined(USE_SSE) || defined(USE_AVX)
			iBufferSize = MAX(1, REG_SIZE / 8) * 8 * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[2].map_count; ++i)
			{
				cnn.conv_l3.kernels[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			for (int k = 0; k < cnn.layer_buffer[2].map_count; ++k)
			{
				for (int i = 0; i < kernel_height; ++i)
				{
					for (int p = 0; p < REG_SIZE; p += 8)
					{
						int j = 0;
						for (j = 0; j < kernel_width; ++j)
						{
							float kernel_val = 0.f;
							FB_READ(data_bin, kernel_val);

							cnn.conv_l3.kernels[k][i * 8 + j + p] = kernel_val;
						}

						for (; j < 8; ++j)
						{
							cnn.conv_l3.kernels[k][i * 8 + j + p] = 0.f;
						}
					}
				}
			}
#else
			iBufferSize = kernel_width * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[2].map_count; ++i)
			{
				cnn.conv_l3.kernels[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			for (int k = 0; k < cnn.layer_buffer[2].map_count; ++k)
			{
				for (int i = 0; i < cnn.conv_l3.size.size; ++i)
				{
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);

					cnn.conv_l3.kernels[k][i] = kernel_val;
				}
			}
#endif

			//conv nn weight
			cnn.conv_bias.resize(cnn.layer_count);
			cnn.subs_weight.resize(cnn.layer_count);
			cnn.subs_bias.resize(cnn.layer_count);

			FB_READ(data_bin, cnn.af_scale);
			for (int i = 0; i < cnn.layer_count; ++i)
			{
				iBufferSize = cnn.layer_buffer[i].map_count;
				cnn.conv_bias[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.subs_weight[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.subs_bias[i] = Array_32f(iBufferSize, ALIGN_DEF);

				for (int j = 0; j < cnn.layer_buffer[i].map_count; ++j)
				{
					FB_READ(data_bin, cnn.conv_bias[i][j]);
					FB_READ(data_bin, cnn.subs_weight[i][j]);
					FB_READ(data_bin, cnn.subs_bias[i][j]);
				}
			}

			//simple nn weight
			FB_READ(data_bin, cnn.snn_full_connect);
			FB_READ(data_bin, cnn.snn_hl_size);

			if (cnn.layer_count != 3 ||
				cnn.conv_l1.size.size != 16 ||
				cnn.conv_l2.size.size != 9 ||
				(cnn.conv_l3.size.size != 30 && cnn.conv_l3.size.size != 56))
			{
				printf("[SIMD::CNN] This configuration cnn models is not supported!\n");
				Clear();
				return;
			}

			if (!cnn.snn_full_connect)
			{
				cnn.snn_hl_weight = Array_32f(cnn.snn_hl_size, ALIGN_DEF);
				for (int i = 0; i < cnn.snn_hl_size; ++i)
				{
					FB_READ(data_bin, cnn.snn_hl_weight[i]);
				}
			}
			else
			{
				cnn.snn_hl_weight_fc.resize(cnn.snn_hl_size);
				for (int i = 0; i < cnn.snn_hl_size; ++i)
				{
					int n = cnn.layer_buffer[cnn.layer_count - 1].map_count;
					cnn.snn_hl_weight_fc[i] = Array_32f(n, ALIGN_DEF);
					for (int j = 0; j < n; ++j)
					{
						FB_READ(data_bin, cnn.snn_hl_weight_fc[i][j]);
					}
				}

				cnn.hl_buffer.resize(cnn.snn_hl_size);
			}

			cnn.snn_hl_bias = Array_32f(cnn.snn_hl_size, ALIGN_DEF);
			for (int i = 0; i < cnn.snn_hl_size; ++i)
			{
				FB_READ(data_bin, cnn.snn_hl_bias[i]);
			}

			cnn.snn_ol_weight = Array_32f(cnn.snn_hl_size, ALIGN_DEF);
			for (int i = 0; i < cnn.snn_hl_size; ++i)
			{
				FB_READ(data_bin, cnn.snn_ol_weight[i]);
			}

			FB_READ(data_bin, cnn.snn_ol_bias);

			//set num threads
			num_threads = 1;
#ifdef USE_OMP
			num_threads = omp_get_num_procs();
#endif
		}

		void ConvNeuralNetwork::AllocateMemory(const Size size)
		{
			if (size.width < cnn.min_image_size.width || size.height < cnn.min_image_size.height)
			{
				return;
			}

			cnn.max_image_size = size;
			cnn.input_buffer_size = Size2d(size.width, size.height);

			//initial layer1
			cnn.conv_l1.ROI.cols = size.width - (cnn.conv_l1.size.cols - 1);
			cnn.conv_l1.ROI.rows = size.height - (cnn.conv_l1.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[0].conv_buffer_size.cols = (2 * REG_SIZE) * (int)ceilf((float)cnn.conv_l1.ROI.cols / float(2 * REG_SIZE));
			cnn.layer_buffer[0].conv_buffer_size.rows = 2 * (int)ceilf((float)cnn.conv_l1.ROI.rows / 2.0f);
			cnn.layer_buffer[0].conv_buffer_size.size = cnn.layer_buffer[0].conv_buffer_size.cols * cnn.layer_buffer[0].conv_buffer_size.rows;

			int iBufferSize = cnn.layer_buffer[0].conv_buffer_size.size;
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				cnn.layer_buffer[0].conv_buffer[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			//pool buffer
			cnn.layer_buffer[0].pool_buffer_size.cols = cnn.layer_buffer[0].conv_buffer_size.cols / 2;
			cnn.layer_buffer[0].pool_buffer_size.rows = cnn.layer_buffer[0].conv_buffer_size.rows / 2;

			cnn.layer_buffer[0].pool_buffer_size.cols = REG_SIZE * (int)ceilf((float)cnn.layer_buffer[0].pool_buffer_size.cols / float(REG_SIZE));
			cnn.layer_buffer[0].pool_buffer_size.size = cnn.layer_buffer[0].pool_buffer_size.cols * cnn.layer_buffer[0].pool_buffer_size.rows;

			iBufferSize = cnn.layer_buffer[0].pool_buffer_size.size;
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				cnn.layer_buffer[0].pool_buffer[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			//sum buffer
			cnn.layer_buffer[0].sum_buffer_size = cnn.layer_buffer[0].pool_buffer_size;

			iBufferSize = cnn.layer_buffer[0].sum_buffer_size.size;
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				cnn.layer_buffer[0].sum_buffer[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			//initial layer2
			cnn.conv_l2.ROI.cols = (cnn.conv_l1.ROI.cols >> 1) - (cnn.conv_l2.size.cols - 1);
			cnn.conv_l2.ROI.rows = (cnn.conv_l1.ROI.rows >> 1) - (cnn.conv_l2.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[1].conv_buffer_size.cols = (2 * REG_SIZE) * (int)ceilf((float)cnn.conv_l2.ROI.cols / float(2 * REG_SIZE));
			cnn.layer_buffer[1].conv_buffer_size.rows = 2 * (int)ceilf((float)cnn.conv_l2.ROI.rows / 2.0f);
			cnn.layer_buffer[1].conv_buffer_size.size = cnn.layer_buffer[1].conv_buffer_size.cols * cnn.layer_buffer[1].conv_buffer_size.rows;

			iBufferSize = cnn.layer_buffer[1].conv_buffer_size.size;
			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				cnn.layer_buffer[1].conv_buffer[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			//pool buffer
			cnn.layer_buffer[1].pool_buffer_size.cols = cnn.layer_buffer[1].conv_buffer_size.cols / 2;
			cnn.layer_buffer[1].pool_buffer_size.rows = cnn.layer_buffer[1].conv_buffer_size.rows / 2;

			cnn.layer_buffer[1].pool_buffer_size.cols = (REG_SIZE)* (int)ceilf((float)cnn.layer_buffer[1].pool_buffer_size.cols / float(REG_SIZE));
			cnn.layer_buffer[1].pool_buffer_size.size = cnn.layer_buffer[1].pool_buffer_size.cols * cnn.layer_buffer[1].pool_buffer_size.rows;

			iBufferSize = cnn.layer_buffer[1].pool_buffer_size.size;
			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				cnn.layer_buffer[1].pool_buffer[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			//sum buffer
			cnn.layer_buffer[1].sum_buffer_size = cnn.layer_buffer[1].pool_buffer_size;

			iBufferSize = cnn.layer_buffer[1].sum_buffer_size.size;
			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				cnn.layer_buffer[1].sum_buffer[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			//initial layer3
			cnn.conv_l3.ROI.cols = (cnn.conv_l2.ROI.cols >> 1) - (cnn.conv_l3.size.cols - 1);
			cnn.conv_l3.ROI.rows = (cnn.conv_l2.ROI.rows >> 1) - (cnn.conv_l3.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[2].conv_buffer_size.cols = (2 * REG_SIZE) * (int)ceilf((float)cnn.conv_l3.ROI.cols / float(2 * REG_SIZE));
			cnn.layer_buffer[2].conv_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.layer_buffer[2].conv_buffer_size.size = cnn.layer_buffer[2].conv_buffer_size.cols * cnn.layer_buffer[2].conv_buffer_size.rows;

			iBufferSize = cnn.layer_buffer[2].conv_buffer_size.size;
			for (int i = 0; i < cnn.layer_buffer[2].map_count; ++i)
			{
				cnn.layer_buffer[2].conv_buffer[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}
			cnn.pool3_buffer_ref = Array_32f_ref(cnn.layer_buffer[2].pool_buffer.data(), cnn.layer_buffer[2].pool_buffer.size());

			//pool buffer
			cnn.layer_buffer[2].pool_buffer_size = cnn.layer_buffer[2].conv_buffer_size;

			iBufferSize = cnn.layer_buffer[2].pool_buffer_size.size;
			for (int i = 0; i < cnn.layer_buffer[2].map_count; ++i)
			{
				cnn.layer_buffer[2].pool_buffer[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			//sum buffer
			cnn.layer_buffer[2].sum_buffer.clear();

			//hl buffer
			if (cnn.snn_full_connect)
			{
				cnn.hl_buffer_size = cnn.layer_buffer[2].pool_buffer_size;
				iBufferSize = cnn.hl_buffer_size.size;
				for (int i = 0; i < cnn.snn_hl_size; ++i)
				{
					cnn.hl_buffer[i] = Array_32f(iBufferSize, ALIGN_DEF);
				}
			}

			cnn.output_buffer_size.cols = cnn.conv_l3.ROI.cols;
			cnn.output_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.output_buffer_size.step = cnn.layer_buffer[2].pool_buffer_size.cols;
			cnn.output_buffer_size.size = cnn.output_buffer_size.rows * cnn.output_buffer_size.step;
		}
		void ConvNeuralNetwork::Clear()
		{
			if (isEmpty()) return;

			cnn.min_image_size = Size(0, 0);
			cnn.max_image_size = Size(0, 0);

			//clear buffers
			cnn.layer_buffer.clear();
			cnn.hl_buffer.clear();
			cnn.pool3_buffer_ref.clear();

			//clear weight
			//conv kernels L1
			cnn.conv_l1.kernels.clear();

			//conv kernels L2
			cnn.conv_l2.kernels.clear();

			//conv kernels L3
			cnn.conv_l3.kernels.clear();

			//conv nn weight
			cnn.conv_bias.clear();
			cnn.subs_weight.clear();
			cnn.subs_bias.clear();

			//simple nn weight
			cnn.snn_hl_weight.clear();
			cnn.snn_hl_weight_fc.clear();
			cnn.snn_hl_bias.clear();
			
			cnn.snn_ol_weight.clear();
		}

		void ConvNeuralNetwork::ResizeBuffers(const Size size)
		{
			cnn.input_buffer_size.cols = size.width;
			cnn.input_buffer_size.rows = size.height;

			//initial layer1
			cnn.conv_l1.ROI.cols = size.width - (cnn.conv_l1.size.cols - 1);
			cnn.conv_l1.ROI.rows = size.height - (cnn.conv_l1.size.rows - 1);

			cnn.layer_buffer[0].conv_buffer_size.cols = (2 * REG_SIZE) * (int)ceilf((float)cnn.conv_l1.ROI.cols / float(2 * REG_SIZE));
			cnn.layer_buffer[0].conv_buffer_size.rows = /*2* */ (int)ceilf((float)cnn.conv_l1.ROI.rows / 2.0f) << 1;

			cnn.layer_buffer[0].pool_buffer_size.cols = cnn.layer_buffer[0].conv_buffer_size.cols >> 1; // div on 2
			cnn.layer_buffer[0].pool_buffer_size.rows = cnn.layer_buffer[0].conv_buffer_size.rows >> 1; // div on 2

			cnn.layer_buffer[0].pool_buffer_size.cols = (REG_SIZE)* (int)ceilf((float)cnn.layer_buffer[0].pool_buffer_size.cols / float(REG_SIZE));
			cnn.layer_buffer[0].pool_buffer_size.size = cnn.layer_buffer[0].pool_buffer_size.cols * cnn.layer_buffer[0].pool_buffer_size.rows;

			//initial layer2
			cnn.conv_l2.ROI.cols = (cnn.conv_l1.ROI.cols >> 1) - (cnn.conv_l2.size.cols - 1);
			cnn.conv_l2.ROI.rows = (cnn.conv_l1.ROI.rows >> 1) - (cnn.conv_l2.size.rows - 1);

			cnn.layer_buffer[1].conv_buffer_size.cols = (2 * REG_SIZE) * (int)ceilf((float)cnn.conv_l2.ROI.cols / float(2 * REG_SIZE));
			cnn.layer_buffer[1].conv_buffer_size.rows = /*2* */ (int)ceilf((float)cnn.conv_l2.ROI.rows / 2.0f) << 1;

			cnn.layer_buffer[1].pool_buffer_size.cols = cnn.layer_buffer[1].conv_buffer_size.cols >> 1; // div on 2
			cnn.layer_buffer[1].pool_buffer_size.rows = cnn.layer_buffer[1].conv_buffer_size.rows >> 1; // div on 2

			cnn.layer_buffer[1].pool_buffer_size.cols = (REG_SIZE)* (int)ceilf((float)cnn.layer_buffer[1].pool_buffer_size.cols / float(REG_SIZE));
			cnn.layer_buffer[1].pool_buffer_size.size = cnn.layer_buffer[1].pool_buffer_size.cols * cnn.layer_buffer[1].pool_buffer_size.rows;

			//initial layer3
			cnn.conv_l3.ROI.cols = (cnn.conv_l2.ROI.cols >> 1) - (cnn.conv_l3.size.cols - 1);
			cnn.conv_l3.ROI.rows = (cnn.conv_l2.ROI.rows >> 1) - (cnn.conv_l3.size.rows - 1);

			cnn.layer_buffer[2].conv_buffer_size.cols = (2 * REG_SIZE) * (int)ceilf((float)cnn.conv_l3.ROI.cols / float(2 * REG_SIZE));
			cnn.layer_buffer[2].conv_buffer_size.rows = cnn.conv_l3.ROI.rows;

			cnn.layer_buffer[2].pool_buffer_size.cols = cnn.layer_buffer[2].conv_buffer_size.cols;
			cnn.layer_buffer[2].pool_buffer_size.rows = cnn.layer_buffer[2].conv_buffer_size.rows;
			cnn.layer_buffer[2].pool_buffer_size.size = cnn.layer_buffer[2].pool_buffer_size.cols * cnn.layer_buffer[2].pool_buffer_size.rows;

			cnn.hl_buffer_size = cnn.layer_buffer[2].pool_buffer_size;

			cnn.output_buffer_size.cols = cnn.conv_l3.ROI.cols;
			cnn.output_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.output_buffer_size.step = cnn.layer_buffer[2].pool_buffer_size.cols;
			cnn.output_buffer_size.size = cnn.output_buffer_size.rows * cnn.output_buffer_size.step;
		}
		void ConvNeuralNetwork::Run_single_thread(Image_32f& image)
		{
#ifdef CHECK_TEST
			if (cnn_ref == NULL) return;
			printf("\n	cnn_simd: run single thread");
			printf("\n	cnn_simd: image size = (%d, %d)\n", image.width, image.height);
			Legacy::ConvNeuralNetwork cnn_old(cnn_ref, image.width, image.height);
			SIMD::Image_32f r_map;
			cnn_old.Forward(r_map, image);
#endif

#ifdef PROFILE_CNN_SIMD
			printf("\n	cnn_simd: run single thread");
			printf("\n	cnn_simd: image size = (%d, %d)\n", image.width, image.height);
			Timer timer(1, true);
#endif

			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				if (cnn.max_pool)
				{
					cnnpp.conv_4x4(cnn.layer_buffer[0].conv_buffer[i](), cnn.layer_buffer[0].conv_buffer_size.cols, image(), image.widthStep, cnn.input_buffer_size.rows, cnn.conv_l1.kernels[i](), cnn.conv_l1.ROI.cols, cnn.conv_l1.ROI.rows);
					cnnpp.max_tanh_tanh(cnn.layer_buffer[0].pool_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].conv_buffer[i](), cnn.layer_buffer[0].conv_buffer_size.cols, cnn.layer_buffer[0].conv_buffer_size.rows, &(cnn.conv_bias[0][i]), &(cnn.subs_weight[0][i]), &(cnn.subs_bias[0][i]), &(cnn.af_scale));
				}
				else
				{
					cnnpp.conv_4x4(cnn.layer_buffer[0].conv_buffer[i](), cnn.layer_buffer[0].conv_buffer_size.cols, image(), image.widthStep, cnn.input_buffer_size.rows, cnn.conv_l1.kernels[i](), cnn.conv_l1.ROI.cols, cnn.conv_l1.ROI.rows);
					cnnpp.tanh_avr_tanh(cnn.layer_buffer[0].pool_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].conv_buffer[i](), cnn.layer_buffer[0].conv_buffer_size.cols, cnn.layer_buffer[0].conv_buffer_size.rows, &(cnn.conv_bias[0][i]), &(cnn.subs_weight[0][i]), &(cnn.subs_bias[0][i]), &(cnn.af_scale));
				}

#ifdef CHECK_TEST
				int count1 = 0;
				double d1 = 0;
				for (int y = 0; y < cnn_old.cnn.subs_neuron[0][i].row_count(); ++y)
				{
					for (int x = 0; x < cnn_old.cnn.subs_neuron[0][i].col_count(); ++x)
					{
						double d = cnn_old.cnn.subs_neuron[0][i][y][x] - cnn.layer_buffer[0].pool_buffer[i][y * cnn.layer_buffer[0].pool_buffer_size.cols + x];
						if (abs(d) > 1.E-4)
						{
							d1 += abs(d);
							count1++;
						}
					}
				}
				printf("	d1 = %f, count1 = %d\n", d1, count1);
				if (abs(d1) > 0.) system("pause");
#endif
			}

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd: run_L1 = %7.3f ms (conv_l1, tanh_avr_tanh)\n", timer.get(1000));
			timer.start();
#endif

			const int it1 = cnn.layer_buffer[1].map_count >> 1; // div on 2
			for (int i = 0; i < it1; ++i)
			{
				if (i > 0 && i < it1 - 1)
				{
					cnnpp.add2(cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer[i - 1](), cnn.layer_buffer[0].pool_buffer[i](), cnn.layer_buffer[0].pool_buffer[i + 1](), cnn.layer_buffer[0].pool_buffer_size.size);
				}
				else
				{
					if (i == 0)
					{
						cnnpp.add(cnn.layer_buffer[0].sum_buffer[0](), cnn.layer_buffer[0].pool_buffer[0](), cnn.layer_buffer[0].pool_buffer[1](), cnn.layer_buffer[0].pool_buffer_size.size);
					}
					else
					{
						const int t = cnn.layer_buffer[0].map_count;
						cnnpp.add(cnn.layer_buffer[0].sum_buffer[t - 1](), cnn.layer_buffer[0].pool_buffer[t - 2](), cnn.layer_buffer[0].pool_buffer[t - 1](), cnn.layer_buffer[0].pool_buffer_size.size);
					}
				}

				if (cnn.max_pool)
				{
					cnnpp.conv_3x3(cnn.layer_buffer[1].conv_buffer[2 * i](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].pool_buffer_size.rows, cnn.conv_l2.kernels[2 * i](), cnn.conv_l2.ROI.cols, cnn.conv_l2.ROI.rows);
					cnnpp.max_tanh_tanh(cnn.layer_buffer[1].pool_buffer[2 * i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].conv_buffer[2 * i](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[1].conv_buffer_size.rows, &(cnn.conv_bias[1][2 * i]), &(cnn.subs_weight[1][2 * i]), &(cnn.subs_bias[1][2 * i]), &(cnn.af_scale));

					cnnpp.conv_3x3(cnn.layer_buffer[1].conv_buffer[2 * i + 1](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].pool_buffer_size.rows, cnn.conv_l2.kernels[2 * i + 1](), cnn.conv_l2.ROI.cols, cnn.conv_l2.ROI.rows);
					cnnpp.max_tanh_tanh(cnn.layer_buffer[1].pool_buffer[2 * i + 1](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].conv_buffer[2 * i + 1](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[1].conv_buffer_size.rows, &(cnn.conv_bias[1][2 * i + 1]), &(cnn.subs_weight[1][2 * i + 1]), &(cnn.subs_bias[1][2 * i + 1]), &(cnn.af_scale));
				}
				else
				{
					cnnpp.conv_3x3(cnn.layer_buffer[1].conv_buffer[2 * i](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].pool_buffer_size.rows, cnn.conv_l2.kernels[2 * i](), cnn.conv_l2.ROI.cols, cnn.conv_l2.ROI.rows);
					cnnpp.tanh_avr_tanh(cnn.layer_buffer[1].pool_buffer[2 * i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].conv_buffer[2 * i](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[1].conv_buffer_size.rows, &(cnn.conv_bias[1][2 * i]), &(cnn.subs_weight[1][2 * i]), &(cnn.subs_bias[1][2 * i]), &(cnn.af_scale));

					cnnpp.conv_3x3(cnn.layer_buffer[1].conv_buffer[2 * i + 1](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].pool_buffer_size.rows, cnn.conv_l2.kernels[2 * i + 1](), cnn.conv_l2.ROI.cols, cnn.conv_l2.ROI.rows);
					cnnpp.tanh_avr_tanh(cnn.layer_buffer[1].pool_buffer[2 * i + 1](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].conv_buffer[2 * i + 1](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[1].conv_buffer_size.rows, &(cnn.conv_bias[1][2 * i + 1]), &(cnn.subs_weight[1][2 * i + 1]), &(cnn.subs_bias[1][2 * i + 1]), &(cnn.af_scale));
				}

#ifdef CHECK_TEST
				int count2 = 0;
				double d2 = 0;
				for (int y = 0; y < cnn_old.cnn.subs_neuron[1][2 * i].row_count(); ++y)
				{
					for (int x = 0; x < cnn_old.cnn.subs_neuron[1][2 * i].col_count(); ++x)
					{
						double d = cnn_old.cnn.subs_neuron[1][2 * i][y][x] - cnn.layer_buffer[1].pool_buffer[2 * i][y * cnn.layer_buffer[1].pool_buffer_size.cols + x];
						if (abs(d) > 1.E-4)
						{
							d2 += abs(d);
							count2++;
						}
					}
				}
				printf("	d2 = %f, count2 = %d\n", d2, count2);
				if (abs(d2) > 0.) system("pause");

				int count3 = 0;
				double d3 = 0;
				for (int y = 0; y < cnn_old.cnn.subs_neuron[1][2 * i + 1].row_count(); ++y)
				{
					for (int x = 0; x < cnn_old.cnn.subs_neuron[1][2 * i + 1].col_count(); ++x)
					{
						double d = cnn_old.cnn.subs_neuron[1][2 * i + 1][y][x] - cnn.layer_buffer[1].pool_buffer[2 * i + 1][y * cnn.layer_buffer[1].pool_buffer_size.cols + x];
						if (abs(d) > 1.E-4)
						{
							d3 += abs(d);
							count3++;
						}
					}
				}
				printf("	d3 = %f, count3 = %d\n", d3, count3);
				if (abs(d3) > 0.) system("pause");
#endif
			}

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd: run_L2 = %7.3f ms (sum, conv_l2, tanh_avr_tanh)\n", timer.get(1000));
#endif

			if (!cnn.snn_full_connect)
			{
#ifdef PROFILE_CNN_SIMD
				timer.start();
#endif

				const int it2 = cnn.layer_buffer[2].map_count >> 1; // div on 2
				for (int i = 0; i < it2; ++i)
				{
					if (i > 0 && i < it2 - 1)
					{
						cnnpp.add2(cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer[i - 1](), cnn.layer_buffer[1].pool_buffer[i](), cnn.layer_buffer[1].pool_buffer[i + 1](), cnn.layer_buffer[1].pool_buffer_size.size);
					}
					else
					{
						if (i == 0)
						{
							cnnpp.add(cnn.layer_buffer[1].sum_buffer[0](), cnn.layer_buffer[1].pool_buffer[0](), cnn.layer_buffer[1].pool_buffer[1](), cnn.layer_buffer[1].pool_buffer_size.size);
						}
						else
						{
							const int t = cnn.layer_buffer[1].map_count;
							cnnpp.add(cnn.layer_buffer[1].sum_buffer[t - 1](), cnn.layer_buffer[1].pool_buffer[t - 2](), cnn.layer_buffer[1].pool_buffer[t - 1](), cnn.layer_buffer[1].pool_buffer_size.size);
						}
					}

					if (cnn.conv_l3.size.rows == 8)
					{
						cnnpp.conv_8x7(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh_2tanh(cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i]), &(cnn.subs_weight[2][2 * i]), &(cnn.subs_bias[2][2 * i]), &(cnn.af_scale), &(cnn.snn_hl_weight[4 * i]), &(cnn.snn_hl_bias[4 * i]), &(cnn.snn_hl_weight[4 * i + 1]), &(cnn.snn_hl_bias[4 * i + 1]), &(cnn.snn_ol_weight[4 * i]), &(cnn.snn_ol_weight[4 * i + 1]));

						cnnpp.conv_8x7(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh_2tanh(cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i + 1]), &(cnn.subs_weight[2][2 * i + 1]), &(cnn.subs_bias[2][2 * i + 1]), &(cnn.af_scale), &(cnn.snn_hl_weight[4 * i + 2]), &(cnn.snn_hl_bias[4 * i + 2]), &(cnn.snn_hl_weight[4 * i + 3]), &(cnn.snn_hl_bias[4 * i + 3]), &(cnn.snn_ol_weight[4 * i + 2]), &(cnn.snn_ol_weight[4 * i + 3]));
					}
					else
					{
						cnnpp.conv_6x5(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh_2tanh(cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i]), &(cnn.subs_weight[2][2 * i]), &(cnn.subs_bias[2][2 * i]), &(cnn.af_scale), &(cnn.snn_hl_weight[4 * i]), &(cnn.snn_hl_bias[4 * i]), &(cnn.snn_hl_weight[4 * i + 1]), &(cnn.snn_hl_bias[4 * i + 1]), &(cnn.snn_ol_weight[4 * i]), &(cnn.snn_ol_weight[4 * i + 1]));

						cnnpp.conv_6x5(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh_2tanh(cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i + 1]), &(cnn.subs_weight[2][2 * i + 1]), &(cnn.subs_bias[2][2 * i + 1]), &(cnn.af_scale), &(cnn.snn_hl_weight[4 * i + 2]), &(cnn.snn_hl_bias[4 * i + 2]), &(cnn.snn_hl_weight[4 * i + 3]), &(cnn.snn_hl_bias[4 * i + 3]), &(cnn.snn_ol_weight[4 * i + 2]), &(cnn.snn_ol_weight[4 * i + 3]));
					}

					if (i > 0)
					{
						cnnpp.add(cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size);
					}
					else
					{
						cnnpp.add(cnn.layer_buffer[2].conv_buffer[0](), cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size);
					}
				}

#ifdef PROFILE_CNN_SIMD
				printf("	cnn_simd: run_L3 = %7.3f ms (sum, conv_l3, tanh_2tanh)\n", timer.get(1000));
				timer.start();
#endif

				for (int i = 1; i < it2; ++i)
				{
					cnnpp.add(cnn.layer_buffer[2].conv_buffer[0](), cnn.layer_buffer[2].conv_buffer[0](), cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer_size.size);
				}

				cnnpp.tanh(cnn.layer_buffer[2].pool_buffer[0](), cnn.layer_buffer[2].conv_buffer[0](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_ol_bias), &(cnn.af_scale));

#ifdef PROFILE_CNN_SIMD
				printf("	cnn_simd: run_HL = %7.3f ms (sum, tanh)\n", timer.get(1000));
#endif

			}
			else
			{
#ifdef PROFILE_CNN_SIMD
				timer.start();
#endif

				int it2 = cnn.layer_buffer[2].map_count >> 1; // div on 2
				for (int i = 0; i < it2; ++i)
				{
					if (i > 0 && i < it2 - 1)
					{
						cnnpp.add2(cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer[i - 1](), cnn.layer_buffer[1].pool_buffer[i](), cnn.layer_buffer[1].pool_buffer[i + 1](), cnn.layer_buffer[1].pool_buffer_size.size);
					}
					else
					{
						if (i == 0)
						{
							cnnpp.add(cnn.layer_buffer[1].sum_buffer[0](), cnn.layer_buffer[1].pool_buffer[0](), cnn.layer_buffer[1].pool_buffer[1](), cnn.layer_buffer[1].pool_buffer_size.size);
						}
						else
						{
							int t = cnn.layer_buffer[1].map_count;
							cnnpp.add(cnn.layer_buffer[1].sum_buffer[t - 1](), cnn.layer_buffer[1].pool_buffer[t - 2](), cnn.layer_buffer[1].pool_buffer[t - 1](), cnn.layer_buffer[1].pool_buffer_size.size);
						}
					}

					if (cnn.conv_l3.size.rows == 8)
					{
						cnnpp.conv_8x7(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh(cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i]), &(cnn.subs_weight[2][2 * i]), &(cnn.subs_bias[2][2 * i]), &(cnn.af_scale));

						cnnpp.conv_8x7(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh(cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i + 1]), &(cnn.subs_weight[2][2 * i + 1]), &(cnn.subs_bias[2][2 * i + 1]), &(cnn.af_scale));
					}
					else
					{
						cnnpp.conv_6x5(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh(cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i]), &(cnn.subs_weight[2][2 * i]), &(cnn.subs_bias[2][2 * i]), &(cnn.af_scale));

						cnnpp.conv_6x5(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh(cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i + 1]), &(cnn.subs_weight[2][2 * i + 1]), &(cnn.subs_bias[2][2 * i + 1]), &(cnn.af_scale));
					}
				}

#ifdef PROFILE_CNN_SIMD
				printf("	cnn_simd: run_L3 = %7.3f ms (sum, conv_l3, tanh_tanh)\n", timer.get(1000));
				timer.start();
#endif

				const int it3 = cnn.snn_hl_size;
				it2 = it2 << 1; // mul on 2;
				for (int i = 0; i < it3; ++i)
				{
					if (it2 == 24)
					{
						static Array_32f_ref temp_buff(cnn.layer_buffer[2].pool_buffer.data(), 24);
						cnnpp.mulC24_add_tanh(cnn.hl_buffer[i](), temp_buff(), cnn.layer_buffer[2].pool_buffer_size.size, cnn.snn_hl_weight_fc[i](), &(cnn.snn_hl_bias[i]), &(cnn.af_scale), &(cnn.snn_ol_weight[i]));
					}
					else
					{
						cnnpp.mulC2_add(cnn.hl_buffer[i](), cnn.layer_buffer[2].pool_buffer[0](), cnn.layer_buffer[2].pool_buffer[1](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_hl_weight_fc[i][0]), &(cnn.snn_hl_weight_fc[i][1]));

						for (int j = 2; j < it2; ++j)
						{
							cnnpp.mulC1_add(cnn.hl_buffer[i](), cnn.layer_buffer[2].pool_buffer[j](), cnn.hl_buffer[i](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_hl_weight_fc[i][j]));
						}

						cnnpp.tanh(cnn.hl_buffer[i](), cnn.hl_buffer[i](), cnn.hl_buffer_size.size, &(cnn.snn_hl_bias[i]), &(cnn.af_scale));
					}
				}

				for (int i = 1; i < it3; ++i)
				{
					cnnpp.add(cnn.hl_buffer[0](), cnn.hl_buffer[0](), cnn.hl_buffer[i](), cnn.layer_buffer[2].pool_buffer_size.size);
				}

				//Timer tm;
				//printf("	size = %d\n"(), cnn.layer_buffer[2].pool_buffer_size.size);
				//tm.start();
				cnnpp.tanh(cnn.layer_buffer[2].pool_buffer[0](), cnn.hl_buffer[0](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_ol_bias), &(cnn.af_scale));
				//printf("	tanh_approx = %7.3f ms\n", tm.get(1000));

				//tm.start();
				//cnnpp.tanh_approx_exp(cnn.layer_buffer[2].pool_buffer[0], cnn.hl_buffer[0](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_ol_bias), &(cnn.af_scale));
				//printf("	tanh_approx_exp = %7.3f ms\n", tm.get(1000));

				//tm.start();
				//for (int u = 0; u < cnn.layer_buffer[2].pool_buffer_size.size; ++u)
				//{
				//	cnn.layer_buffer[2].pool_buffer[0][u] = 1.7f * tanhf(1.f * cnn.hl_buffer[0][u] + 1.f);
				//}
				//printf("	tanh = %7.3f ms\n", tm.get(1000));

				//tm.start();
				//cnnpp.relu(cnn.layer_buffer[2].pool_buffer[0], cnn.hl_buffer[0](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_ol_bias), &(cnn.af_scale));
				//printf("	relu = %7.3f ms\n", tm.get(1000));

#ifdef PROFILE_CNN_SIMD
				printf("	cnn_simd: run_HL = %7.3f ms (sum, mul, tanh, sum, tanh)\n", timer.get(1000));
#endif
			}

#ifdef CHECK_TEST
			int count4 = 0;
			double d4 = 0;
			for (int y = 0; y < cnn_old.snn.output_neuron.row_count(); ++y)
			{
				for (int x = 0; x < cnn_old.snn.output_neuron.col_count(); ++x)
				{
					double d = cnn_old.snn.output_neuron[y][x][0] - cnn.layer_buffer[2].pool_buffer[0][y * cnn.output_buffer_size.step + x];
					if (abs(d) > 1.E-4)
					{
						d4 += abs(d);
						count4++;
					}
				}
			}
			printf("	d4 = %f, count4 = %d\n", d4, count4);
			if (abs(d4) > 0.) system("pause");
#endif
		}
		void ConvNeuralNetwork::Run_multi_threads(Image_32f& image)
		{
#ifdef CHECK_TEST
			if (cnn_ref == NULL) return;
			printf("\n	cnn_simd: run multi threads");
			printf("\n	cnn_simd: image size = (%d, %d)\n", image.width, image.height);
			Legacy::ConvNeuralNetwork cnn_old(cnn_ref, image.width, image.height);
			SIMD::Image_32f r_map;
			cnn_old.Forward(r_map, image);
#endif

#ifdef PROFILE_CNN_SIMD
			printf("\n	cnn_simd: run multi threads");
			printf("\n	cnn_simd: image size = (%d, %d)\n", image.width, image.height);
			Timer timer(1, true);
#endif

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				if (cnn.max_pool)
				{
					cnnpp.conv_4x4(cnn.layer_buffer[0].conv_buffer[i](), cnn.layer_buffer[0].conv_buffer_size.cols, image(), image.widthStep, cnn.input_buffer_size.rows, cnn.conv_l1.kernels[i](), cnn.conv_l1.ROI.cols, cnn.conv_l1.ROI.rows);
					cnnpp.max_tanh_tanh(cnn.layer_buffer[0].pool_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].conv_buffer[i](), cnn.layer_buffer[0].conv_buffer_size.cols, cnn.layer_buffer[0].conv_buffer_size.rows, &(cnn.conv_bias[0][i]), &(cnn.subs_weight[0][i]), &(cnn.subs_bias[0][i]), &(cnn.af_scale));
				}
				else
				{
					cnnpp.conv_4x4(cnn.layer_buffer[0].conv_buffer[i](), cnn.layer_buffer[0].conv_buffer_size.cols, image(), image.widthStep, cnn.input_buffer_size.rows, cnn.conv_l1.kernels[i](), cnn.conv_l1.ROI.cols, cnn.conv_l1.ROI.rows);
					cnnpp.tanh_avr_tanh(cnn.layer_buffer[0].pool_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].conv_buffer[i](), cnn.layer_buffer[0].conv_buffer_size.cols, cnn.layer_buffer[0].conv_buffer_size.rows, &(cnn.conv_bias[0][i]), &(cnn.subs_weight[0][i]), &(cnn.subs_bias[0][i]), &(cnn.af_scale));
				}

#ifdef CHECK_TEST
				int count1 = 0;
				double d1 = 0;
				for (int y = 0; y < cnn_old.cnn.subs_neuron[0][i].row_count(); ++y)
				{
					for (int x = 0; x < cnn_old.cnn.subs_neuron[0][i].col_count(); ++x)
					{
						double d = cnn_old.cnn.subs_neuron[0][i][y][x] - cnn.layer_buffer[0].pool_buffer[i][y * cnn.layer_buffer[0].pool_buffer_size.cols + x];
						if (abs(d) > 1.E-4)
						{
							d1 += abs(d);
							count1++;
						}
					}
				}
				printf("	d1 = %f, count1 = %d\n", d1, count1);
				if (abs(d1) > 0.) system("pause");
#endif
			}

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd: run_L1 = %7.3f ms (conv_l1, tanh_avr_tanh)\n", timer.get(1000));
			timer.start();
#endif

			const int it1 = cnn.layer_buffer[1].map_count >> 1; // div on 2

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int i = 0; i < it1; ++i)
			{
				if (i > 0 && i < it1 - 1)
				{
					cnnpp.add2(cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer[i - 1](), cnn.layer_buffer[0].pool_buffer[i](), cnn.layer_buffer[0].pool_buffer[i + 1](), cnn.layer_buffer[0].pool_buffer_size.size);
				}
				else
				{
					if (i == 0)
					{
						cnnpp.add(cnn.layer_buffer[0].sum_buffer[0](), cnn.layer_buffer[0].pool_buffer[0](), cnn.layer_buffer[0].pool_buffer[1](), cnn.layer_buffer[0].pool_buffer_size.size);
					}
					else
					{
						const int t = cnn.layer_buffer[0].map_count;
						cnnpp.add(cnn.layer_buffer[0].sum_buffer[t - 1](), cnn.layer_buffer[0].pool_buffer[t - 2](), cnn.layer_buffer[0].pool_buffer[t - 1](), cnn.layer_buffer[0].pool_buffer_size.size);
					}
				}

				if (cnn.max_pool)
				{
					cnnpp.conv_3x3(cnn.layer_buffer[1].conv_buffer[2 * i](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].pool_buffer_size.rows, cnn.conv_l2.kernels[2 * i](), cnn.conv_l2.ROI.cols, cnn.conv_l2.ROI.rows);
					cnnpp.max_tanh_tanh(cnn.layer_buffer[1].pool_buffer[2 * i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].conv_buffer[2 * i](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[1].conv_buffer_size.rows, &(cnn.conv_bias[1][2 * i]), &(cnn.subs_weight[1][2 * i]), &(cnn.subs_bias[1][2 * i]), &(cnn.af_scale));

					cnnpp.conv_3x3(cnn.layer_buffer[1].conv_buffer[2 * i + 1](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].pool_buffer_size.rows, cnn.conv_l2.kernels[2 * i + 1](), cnn.conv_l2.ROI.cols, cnn.conv_l2.ROI.rows);
					cnnpp.max_tanh_tanh(cnn.layer_buffer[1].pool_buffer[2 * i + 1](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].conv_buffer[2 * i + 1](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[1].conv_buffer_size.rows, &(cnn.conv_bias[1][2 * i + 1]), &(cnn.subs_weight[1][2 * i + 1]), &(cnn.subs_bias[1][2 * i + 1]), &(cnn.af_scale));
				}
				else
				{
					cnnpp.conv_3x3(cnn.layer_buffer[1].conv_buffer[2 * i](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].pool_buffer_size.rows, cnn.conv_l2.kernels[2 * i](), cnn.conv_l2.ROI.cols, cnn.conv_l2.ROI.rows);
					cnnpp.tanh_avr_tanh(cnn.layer_buffer[1].pool_buffer[2 * i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].conv_buffer[2 * i](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[1].conv_buffer_size.rows, &(cnn.conv_bias[1][2 * i]), &(cnn.subs_weight[1][2 * i]), &(cnn.subs_bias[1][2 * i]), &(cnn.af_scale));

					cnnpp.conv_3x3(cnn.layer_buffer[1].conv_buffer[2 * i + 1](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].pool_buffer_size.rows, cnn.conv_l2.kernels[2 * i + 1](), cnn.conv_l2.ROI.cols, cnn.conv_l2.ROI.rows);
					cnnpp.tanh_avr_tanh(cnn.layer_buffer[1].pool_buffer[2 * i + 1](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].conv_buffer[2 * i + 1](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[1].conv_buffer_size.rows, &(cnn.conv_bias[1][2 * i + 1]), &(cnn.subs_weight[1][2 * i + 1]), &(cnn.subs_bias[1][2 * i + 1]), &(cnn.af_scale));
				}

#ifdef CHECK_TEST
				int count2 = 0;
				double d2 = 0;
				for (int y = 0; y < cnn_old.cnn.subs_neuron[1][2 * i].row_count(); ++y)
				{
					for (int x = 0; x < cnn_old.cnn.subs_neuron[1][2 * i].col_count(); ++x)
					{
						double d = cnn_old.cnn.subs_neuron[1][2 * i][y][x] - cnn.layer_buffer[1].pool_buffer[2 * i][y * cnn.layer_buffer[1].pool_buffer_size.cols + x];
						if (abs(d) > 1.E-4)
						{
							d2 += abs(d);
							count2++;
						}
					}
				}
				printf("	d2 = %f, count2 = %d\n", d2, count2);
				if (abs(d2) > 0.) system("pause");

				int count3 = 0;
				double d3 = 0;
				for (int y = 0; y < cnn_old.cnn.subs_neuron[1][2 * i + 1].row_count(); ++y)
				{
					for (int x = 0; x < cnn_old.cnn.subs_neuron[1][2 * i + 1].col_count(); ++x)
					{
						double d = cnn_old.cnn.subs_neuron[1][2 * i + 1][y][x] - cnn.layer_buffer[1].pool_buffer[2 * i + 1][y * cnn.layer_buffer[1].pool_buffer_size.cols + x];
						if (abs(d) > 1.E-4)
						{
							d3 += abs(d);
							count3++;
						}
					}
				}
				printf("	d3 = %f, count3 = %d\n", d3, count3);
				if (abs(d3) > 0.) system("pause");
#endif
			}

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd: run_L2 = %7.3f ms (sum, conv_l2, tanh_avr_tanh)\n", timer.get(1000));
#endif

			if (!cnn.snn_full_connect)
			{
#ifdef PROFILE_CNN_SIMD
				timer.start();
#endif

				const int it2 = cnn.layer_buffer[2].map_count >> 1; // div on 2

				OMP_PRAGMA(omp parallel for num_threads(num_threads))
				for (int i = 0; i < it2; ++i)
				{
					if (i > 0 && i < it2 - 1)
					{
						cnnpp.add2(cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer[i - 1](), cnn.layer_buffer[1].pool_buffer[i](), cnn.layer_buffer[1].pool_buffer[i + 1](), cnn.layer_buffer[1].pool_buffer_size.size);
					}
					else
					{
						if (i == 0)
						{
							cnnpp.add(cnn.layer_buffer[1].sum_buffer[0](), cnn.layer_buffer[1].pool_buffer[0](), cnn.layer_buffer[1].pool_buffer[1](), cnn.layer_buffer[1].pool_buffer_size.size);
						}
						else
						{
							const int t = cnn.layer_buffer[1].map_count;
							cnnpp.add(cnn.layer_buffer[1].sum_buffer[t - 1](), cnn.layer_buffer[1].pool_buffer[t - 2](), cnn.layer_buffer[1].pool_buffer[t - 1](), cnn.layer_buffer[1].pool_buffer_size.size);
						}
					}

					if (cnn.conv_l3.size.rows == 8)
					{
						cnnpp.conv_8x7(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh_2tanh(cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i]), &(cnn.subs_weight[2][2 * i]), &(cnn.subs_bias[2][2 * i]), &(cnn.af_scale), &(cnn.snn_hl_weight[4 * i]), &(cnn.snn_hl_bias[4 * i]), &(cnn.snn_hl_weight[4 * i + 1]), &(cnn.snn_hl_bias[4 * i + 1]), &(cnn.snn_ol_weight[4 * i]), &(cnn.snn_ol_weight[4 * i + 1]));

						cnnpp.conv_8x7(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh_2tanh(cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i + 1]), &(cnn.subs_weight[2][2 * i + 1]), &(cnn.subs_bias[2][2 * i + 1]), &(cnn.af_scale), &(cnn.snn_hl_weight[4 * i + 2]), &(cnn.snn_hl_bias[4 * i + 2]), &(cnn.snn_hl_weight[4 * i + 3]), &(cnn.snn_hl_bias[4 * i + 3]), &(cnn.snn_ol_weight[4 * i + 2]), &(cnn.snn_ol_weight[4 * i + 3]));
					}
					else
					{
						cnnpp.conv_6x5(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh_2tanh(cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i]), &(cnn.subs_weight[2][2 * i]), &(cnn.subs_bias[2][2 * i]), &(cnn.af_scale), &(cnn.snn_hl_weight[4 * i]), &(cnn.snn_hl_bias[4 * i]), &(cnn.snn_hl_weight[4 * i + 1]), &(cnn.snn_hl_bias[4 * i + 1]), &(cnn.snn_ol_weight[4 * i]), &(cnn.snn_ol_weight[4 * i + 1]));

						cnnpp.conv_6x5(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh_2tanh(cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i + 1]), &(cnn.subs_weight[2][2 * i + 1]), &(cnn.subs_bias[2][2 * i + 1]), &(cnn.af_scale), &(cnn.snn_hl_weight[4 * i + 2]), &(cnn.snn_hl_bias[4 * i + 2]), &(cnn.snn_hl_weight[4 * i + 3]), &(cnn.snn_hl_bias[4 * i + 3]), &(cnn.snn_ol_weight[4 * i + 2]), &(cnn.snn_ol_weight[4 * i + 3]));
					}

					if (i > 0)
					{
						cnnpp.add(cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size);
					}
					else
					{
						cnnpp.add(cnn.layer_buffer[2].conv_buffer[0](), cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size);
					}
				}

#ifdef PROFILE_CNN_SIMD
				printf("	cnn_simd: run_L3 = %7.3f ms (sum, conv_l3, tanh_2tanh)\n", timer.get(1000));
				timer.start();
#endif

				const int block_size = 512;
				const int size = cnn.layer_buffer[2].pool_buffer_size.size;
				const int num_blocks = (int)ceilf((float)size / (float)block_size);

				OMP_PRAGMA(omp parallel for num_threads(num_threads))
				for (int t = 0; t < num_blocks; ++t)
				{
					int offset = t * block_size;
					int length = block_size;
					if (offset + block_size > size)
					{
						length = size - offset;
					}

					for (int i = 1; i < it2; ++i)
					{
						cnnpp.add(cnn.layer_buffer[2].conv_buffer[0](offset), cnn.layer_buffer[2].conv_buffer[0](offset), cnn.layer_buffer[2].pool_buffer[2 * i + 1](offset), length);
					}

					cnnpp.tanh(cnn.layer_buffer[2].pool_buffer[0](offset), cnn.layer_buffer[2].conv_buffer[0](offset), length, &(cnn.snn_ol_bias), &(cnn.af_scale));
				}

#ifdef PROFILE_CNN_SIMD
				printf("	cnn_simd: run_HL = %7.3f ms (sum, tanh)\n", timer.get(1000));
#endif

			}
			else
			{
#ifdef PROFILE_CNN_SIMD
				timer.start();
#endif

				int it2 = cnn.layer_buffer[2].map_count >> 1; // div on 2

				OMP_PRAGMA(omp parallel for num_threads(num_threads))
				for (int i = 0; i < it2; ++i)
				{
					if (i > 0 && i < it2 - 1)
					{
						cnnpp.add2(cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer[i - 1](), cnn.layer_buffer[1].pool_buffer[i](), cnn.layer_buffer[1].pool_buffer[i + 1](), cnn.layer_buffer[1].pool_buffer_size.size);
					}
					else
					{
						if (i == 0)
						{
							cnnpp.add(cnn.layer_buffer[1].sum_buffer[0](), cnn.layer_buffer[1].pool_buffer[0](), cnn.layer_buffer[1].pool_buffer[1](), cnn.layer_buffer[1].pool_buffer_size.size);
						}
						else
						{
							int t = cnn.layer_buffer[1].map_count;
							cnnpp.add(cnn.layer_buffer[1].sum_buffer[t - 1](), cnn.layer_buffer[1].pool_buffer[t - 2](), cnn.layer_buffer[1].pool_buffer[t - 1](), cnn.layer_buffer[1].pool_buffer_size.size);
						}
					}

					if (cnn.conv_l3.size.rows == 8)
					{
						cnnpp.conv_8x7(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh(cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i]), &(cnn.subs_weight[2][2 * i]), &(cnn.subs_bias[2][2 * i]), &(cnn.af_scale));

						cnnpp.conv_8x7(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh(cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i + 1]), &(cnn.subs_weight[2][2 * i + 1]), &(cnn.subs_bias[2][2 * i + 1]), &(cnn.af_scale));
					}
					else
					{
						cnnpp.conv_6x5(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh(cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i]), &(cnn.subs_weight[2][2 * i]), &(cnn.subs_bias[2][2 * i]), &(cnn.af_scale));

						cnnpp.conv_6x5(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
						cnnpp.tanh_tanh(cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i + 1]), &(cnn.subs_weight[2][2 * i + 1]), &(cnn.subs_bias[2][2 * i + 1]), &(cnn.af_scale));
					}
				}

#ifdef PROFILE_CNN_SIMD
				printf("	cnn_simd: run_L3 = %7.3f ms (sum, conv_l3, tanh_tanh)\n", timer.get(1000));
				timer.start();
#endif

				const int it3 = cnn.snn_hl_size;
				it2 = it2 << 1; // mul on 2;

				OMP_PRAGMA(omp parallel for num_threads(num_threads))
				for (int i = 0; i < it3; ++i)
				{
					if (it2 == 24)
					{
						cnnpp.mulC24_add_tanh(cnn.hl_buffer[i](), cnn.pool3_buffer_ref(), cnn.layer_buffer[2].pool_buffer_size.size, cnn.snn_hl_weight_fc[i](), &(cnn.snn_hl_bias[i]), &(cnn.af_scale), &(cnn.snn_ol_weight[i]));
					}
					else
					{
						cnnpp.mulC2_add(cnn.hl_buffer[i](), cnn.layer_buffer[2].pool_buffer[0](), cnn.layer_buffer[2].pool_buffer[1](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_hl_weight_fc[i][0]), &(cnn.snn_hl_weight_fc[i][1]));

						for (int j = 2; j < it2; ++j)
						{
							cnnpp.mulC1_add(cnn.hl_buffer[i](), cnn.layer_buffer[2].pool_buffer[j](), cnn.hl_buffer[i](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_hl_weight_fc[i][j]));
						}

						cnnpp.tanh(cnn.hl_buffer[i](), cnn.hl_buffer[i](), cnn.hl_buffer_size.size, &(cnn.snn_hl_bias[i]), &(cnn.af_scale));
					}
				}

				const int block_size = 512;
				const int size = cnn.layer_buffer[2].pool_buffer_size.size;
				const int num_blocks = (int)ceilf((float)size / (float)block_size);

				OMP_PRAGMA(omp parallel for num_threads(num_threads))
				for (int t = 0; t < num_blocks; ++t)
				{
					int offset = t * block_size;
					int length = block_size;
					if (offset + block_size > size)
					{
						length = size - offset;
					}

					for (int i = 1; i < it3; ++i)
					{
						cnnpp.add(cnn.hl_buffer[0](offset), cnn.hl_buffer[0](offset), cnn.hl_buffer[i](offset), length);
					}

					cnnpp.tanh(cnn.layer_buffer[2].pool_buffer[0](offset), cnn.hl_buffer[0](offset), length, &(cnn.snn_ol_bias), &(cnn.af_scale));
				}

#ifdef PROFILE_CNN_SIMD
				printf("	cnn_simd: run_HL = %7.3f ms (sum, mul, tanh, sum, tanh)\n", timer.get(1000));
#endif
			}

#ifdef CHECK_TEST
			int count4 = 0;
			double d4 = 0;
			for (int y = 0; y < cnn_old.snn.output_neuron.row_count(); ++y)
			{
				for (int x = 0; x < cnn_old.snn.output_neuron.col_count(); ++x)
				{
					double d = cnn_old.snn.output_neuron[y][x][0] - cnn.layer_buffer[2].pool_buffer[0][y * cnn.output_buffer_size.step + x];
					if (abs(d) > 1.E-4)
					{
						d4 += abs(d);
						count4++;
					}
				}
			}
			printf("	d4 = %f, count4 = %d\n", d4, count4);
			if (abs(d4) > 0.) system("pause");
#endif
		}

		void ConvNeuralNetwork::Forward(Image_32f& response_map, Image_32f& image)
		{
			if (image.width != cnn.input_buffer_size.cols || image.height != cnn.input_buffer_size.rows)
			{
				if (image.width < cnn.min_image_size.width || image.height < cnn.min_image_size.height ||
					image.width > cnn.max_image_size.width || image.height > cnn.max_image_size.height)
				{
					response_map.width = 0;
					response_map.height = 0;
					return;
				}

				ResizeBuffers(image.getSize());
			}

#ifdef USE_OMP
			if (num_threads > 1)
			{
				Run_multi_threads(image);
			}
			else
#endif
			{
				Run_single_thread(image);
			}

			if (response_map.isEmpty())
			{
				response_map.width = cnn.output_buffer_size.cols;
				response_map.height = cnn.output_buffer_size.rows;
				response_map.widthStep = cnn.output_buffer_size.step;
				response_map.data = cnn.layer_buffer[2].pool_buffer[0].data;
				response_map.sharingData = true;
			}
			else
			{
				response_map.width = cnn.output_buffer_size.cols;
				response_map.height = cnn.output_buffer_size.rows;
				response_map.copyData(cnn.output_buffer_size.cols, cnn.output_buffer_size.rows, cnn.layer_buffer[2].pool_buffer[0](), cnn.output_buffer_size.step);
			}
		}

		Size ConvNeuralNetwork::getOutputImgSize(const Size size)
		{
			//size layer1
			int cnn_conv_l1_ROI_cols = size.width - (cnn.conv_l1.size.cols - 1);
			int cnn_conv_l1_ROI_rows = size.height - (cnn.conv_l1.size.rows - 1);

			//size layer2
			int cnn_conv_l2_ROI_cols = (cnn_conv_l1_ROI_cols >> 1) - (cnn.conv_l2.size.cols - 1);
			int cnn_conv_l2_ROI_rows = (cnn_conv_l1_ROI_rows >> 1) - (cnn.conv_l2.size.rows - 1);

			//size layer3
			int cnn_conv_l3_ROI_cols = (cnn_conv_l2_ROI_cols >> 1) - (cnn.conv_l3.size.cols - 1);
			int cnn_conv_l3_ROI_rows = (cnn_conv_l2_ROI_rows >> 1) - (cnn.conv_l3.size.rows - 1);

			return Size(cnn_conv_l3_ROI_cols, cnn_conv_l3_ROI_rows);
		}
	}

#endif
}