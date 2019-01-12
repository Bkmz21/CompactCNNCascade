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


#include "cnn_simd_cntk.h"
#include <fstream>
#include <sstream>
#include <iterator>
#include <immintrin.h>
#include <cmath>

#ifdef USE_OMP
#	include <omp.h>
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_CNTK_MODELS

	namespace SIMD
	{
		void ConvNeuralNetwork::Init(std::string file_name, int index_output, void* hGrd)
		{
			//if (file_name.find(".txt") != std::string::npos)
			//{
			//	LoadCNTKModel(file_name);
			//	return;
			//}

			std::stringstream data_bin;
			if (file_name.size() < 255)
			{
				std::fstream file_bin;
				file_bin.open(file_name.c_str(), std::fstream::binary | std::fstream::in);

				if (!file_bin.is_open())
				{
					printf("[SIMD::CNN] Configuration file not found!\n");
					return;
				}

				data_bin << file_bin.rdbuf();
				file_bin.close();
			}
			else
			{
				data_bin << file_name;
			}

			//version
			float format_version = 0.0f;
			FB_READ(data_bin, format_version);

			if (format_version < 1.0f || format_version > 1.1f)
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
			if (kernel_width < 8)
			{
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
			}
			else
			{
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
			FB_READ(data_bin, cnn.af_scale);

			cnn.conv_bias.resize(cnn.layer_count);
			cnn.leakyReLU_w1.resize(cnn.layer_count);
			cnn.leakyReLU_w2.resize(cnn.layer_count);
			cnn.bn_weight.resize(cnn.layer_count);
			cnn.bn_bias.resize(cnn.layer_count);
			for (int i = 0; i < cnn.layer_count; ++i)
			{
				iBufferSize = cnn.layer_buffer[i].map_count;
				cnn.conv_bias[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.leakyReLU_w1[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.leakyReLU_w2[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.bn_weight[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.bn_bias[i] = Array_32f(iBufferSize, ALIGN_DEF);

				for (int j = 0; j < cnn.layer_buffer[i].map_count; ++j)
				{
					FB_READ(data_bin, cnn.conv_bias[i][j]);
					FB_READ(data_bin, cnn.leakyReLU_w1[i][j]);
					FB_READ(data_bin, cnn.leakyReLU_w2[i][j]);
					FB_READ(data_bin, cnn.bn_weight[i][j]);
					FB_READ(data_bin, cnn.bn_bias[i][j]);
				}
			}

			//simple nn weight
			FB_READ(data_bin, cnn.snn_full_connect);
			FB_READ(data_bin, cnn.snn_hl_size);
			FB_READ(data_bin, cnn.snn_connect_count);
			FB_READ(data_bin, cnn.hl_scale);

			if (!cnn.max_pool ||
				cnn.layer_count != 3 ||
				cnn.conv_l2.size.size != 9)
			{
				printf("[SIMD::CNN] This configuration cnn models is not supported!\n");
				Clear();
				return;
			}

			if (!cnn.snn_full_connect)
			{
				cnn.snn_hl_weight.resize(cnn.snn_hl_size);
				for (int i = 0; i < cnn.snn_hl_size; ++i)
				{
					cnn.snn_hl_weight[i] = Array_32f(cnn.snn_connect_count, ALIGN_DEF);
					for (int j = 0; j < cnn.snn_connect_count; ++j)
					{
						FB_READ(data_bin, cnn.snn_hl_weight[i][j]);
					}
				}
			}
			else
			{
				cnn.snn_hl_weight.resize(cnn.snn_hl_size);
				for (int i = 0; i < cnn.snn_hl_size; ++i)
				{
					int n = cnn.layer_buffer[cnn.layer_count - 1].map_count;
					cnn.snn_hl_weight[i] = Array_32f(n, ALIGN_DEF);
					for (int j = 0; j < n; ++j)
					{
						FB_READ(data_bin, cnn.snn_hl_weight[i][j]);
					}
				}
			}

			cnn.snn_hl_bias = Array_32f(cnn.snn_hl_size, ALIGN_DEF);
			for (int i = 0; i < cnn.snn_hl_size; ++i)
			{
				FB_READ(data_bin, cnn.snn_hl_bias[i]);
			}

			cnn.snn_hl_tanh_w = Array_32f(cnn.snn_hl_size / cnn.hl_scale, ALIGN_DEF);
			for (int i = 0; i < (cnn.snn_hl_size / cnn.hl_scale); ++i)
			{
				FB_READ(data_bin, cnn.snn_hl_tanh_w[i]);
			}

			cnn.snn_hl_bn_weight = Array_32f(cnn.hl_scale, ALIGN_DEF);
			cnn.snn_hl_bn_bias = Array_32f(cnn.hl_scale, ALIGN_DEF);
			for (int i = 0; i < cnn.hl_scale; ++i)
			{
				FB_READ(data_bin, cnn.snn_hl_bn_weight[i]);
				FB_READ(data_bin, cnn.snn_hl_bn_bias[i]);
			}

			FB_READ(data_bin, cnn.snn_ol_neuron_count);
			cnn.snn_ol_weight.resize(cnn.snn_ol_neuron_count);
			for (int i = 0; i < cnn.snn_ol_neuron_count; ++i)
			{
				cnn.snn_ol_weight[i] = Array_32f(cnn.snn_hl_size, ALIGN_DEF);
				for (int j = 0; j < cnn.snn_hl_size; ++j)
				{
					FB_READ(data_bin, cnn.snn_ol_weight[i][j]);
				}
			}

			cnn.snn_ol_bias = Array_32f(cnn.snn_ol_neuron_count, ALIGN_DEF);
			for (int i = 0; i < cnn.snn_ol_neuron_count; ++i)
			{
				FB_READ(data_bin, cnn.snn_ol_bias[i]);
			}

			FB_READ(data_bin, cnn.snn_ol_tanh_w);

			cnn.index_output = MIN(index_output, cnn.snn_ol_neuron_count - 1);
			cnn.af_scale = cnn.index_output == 0 ? -cnn.af_scale : cnn.af_scale;

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

			cnn.layer_buffer[1].pool_buffer_size.cols = (REG_SIZE) * (int)ceilf((float)cnn.layer_buffer[1].pool_buffer_size.cols / float(REG_SIZE));
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

			//pool buffer			
			const int t = cnn.conv_l3.ROI.rows == 1 ? 1 : 2;
			cnn.layer_buffer[2].pool_buffer_size.cols = (t * REG_SIZE) * (int)ceilf((float)cnn.conv_l3.ROI.cols / float(t * REG_SIZE));
			cnn.layer_buffer[2].pool_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.layer_buffer[2].pool_buffer_size.size = cnn.layer_buffer[2].pool_buffer_size.cols * cnn.layer_buffer[2].pool_buffer_size.rows;

			iBufferSize = cnn.layer_buffer[2].pool_buffer_size.size;
			for (int i = 0; i < cnn.layer_buffer[2].map_count; ++i)
			{
				cnn.layer_buffer[2].pool_buffer[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}
			cnn.pool3_buffer_ref = Array_32f_ref(cnn.layer_buffer[2].pool_buffer.data(), cnn.layer_buffer[2].pool_buffer.size());

			//sum buffer
			cnn.layer_buffer[2].sum_buffer.clear();

			//hl buffer
			cnn.hl_buffer.resize(cnn.snn_hl_size);
			cnn.hl_buffer_size = cnn.layer_buffer[2].pool_buffer_size;
			iBufferSize = cnn.hl_buffer_size.size;
			for (int i = 0; i < cnn.snn_hl_size; ++i)
			{
				cnn.hl_buffer[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			const int num_out_map = cnn.index_output < 0 ? cnn.snn_ol_neuron_count : 1;
			cnn.ol_buffer_size = Size2d(cnn.layer_buffer[2].pool_buffer_size.cols,
										num_out_map * cnn.layer_buffer[2].pool_buffer_size.rows,
										cnn.layer_buffer[2].pool_buffer_size.step);
			iBufferSize = cnn.ol_buffer_size.size;
			cnn.ol_buffer = Array_32f(iBufferSize, ALIGN_DEF);

			cnn.output_buffer_size = Size2d(cnn.conv_l3.ROI.cols,
											num_out_map * cnn.conv_l3.ROI.rows,
											cnn.ol_buffer_size.step);
		}
		void ConvNeuralNetwork::Clear()
		{
			if (isEmpty()) return;

			cnn.min_image_size = Size(0, 0);
			cnn.max_image_size = Size(0, 0);

			//clear buffers
			cnn.layer_buffer.clear();
			cnn.hl_buffer.clear();
			cnn.ol_buffer.clear();
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
			cnn.leakyReLU_w1.clear();
			cnn.leakyReLU_w2.clear();
			cnn.bn_weight.clear();
			cnn.bn_bias.clear();

			//simple nn weight
			cnn.snn_hl_weight.clear();
			cnn.snn_hl_size = 0;

			cnn.snn_hl_bias.clear();
			cnn.snn_hl_tanh_w.clear();

			cnn.snn_hl_bn_weight.clear();
			cnn.snn_hl_bn_bias.clear();

			cnn.snn_ol_weight.clear();
			cnn.snn_ol_neuron_count = 0;

			cnn.snn_ol_bias.clear();
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

			cnn.layer_buffer[1].pool_buffer_size.cols = (REG_SIZE) * (int)ceilf((float)cnn.layer_buffer[1].pool_buffer_size.cols / float(REG_SIZE));
			cnn.layer_buffer[1].pool_buffer_size.size = cnn.layer_buffer[1].pool_buffer_size.cols * cnn.layer_buffer[1].pool_buffer_size.rows;

			//initial layer3
			cnn.conv_l3.ROI.cols = (cnn.conv_l2.ROI.cols >> 1) - (cnn.conv_l3.size.cols - 1);
			cnn.conv_l3.ROI.rows = (cnn.conv_l2.ROI.rows >> 1) - (cnn.conv_l3.size.rows - 1);

			cnn.layer_buffer[2].conv_buffer_size.cols = (2 * REG_SIZE) * (int)ceilf((float)cnn.conv_l3.ROI.cols / float(2 * REG_SIZE));
			cnn.layer_buffer[2].conv_buffer_size.rows = cnn.conv_l3.ROI.rows;

			const int t = cnn.conv_l3.ROI.rows == 1 ? 1 : 2;
			cnn.layer_buffer[2].pool_buffer_size.cols = (t * REG_SIZE) * (int)ceilf((float)cnn.conv_l3.ROI.cols / float(t * REG_SIZE));
			cnn.layer_buffer[2].pool_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.layer_buffer[2].pool_buffer_size.size = cnn.layer_buffer[2].pool_buffer_size.cols * cnn.layer_buffer[2].pool_buffer_size.rows;

			cnn.hl_buffer_size = cnn.layer_buffer[2].pool_buffer_size;
			
			const int out_map = cnn.snn_ol_neuron_count > 3 ? cnn.snn_ol_neuron_count : 1;
			cnn.ol_buffer_size = Size2d(cnn.layer_buffer[2].pool_buffer_size.cols,
										out_map * cnn.layer_buffer[2].pool_buffer_size.rows,
										cnn.layer_buffer[2].pool_buffer_size.step);

			cnn.output_buffer_size = Size2d(cnn.conv_l3.ROI.cols,
											out_map * cnn.conv_l3.ROI.rows,
											cnn.ol_buffer_size.step);
		}
		void ConvNeuralNetwork::Run(Image_32f& image)
		{
#ifdef PROFILE_CNN_SIMD
			printf("\n	cnn_simd: run single thread");
			printf("\n	cnn_simd: image size = (%d, %d)\n", image.width, image.height);
			Timer timer(1, true);
#endif

			//OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				if (cnn.conv_l1.size.rows == 4 && cnn.conv_l1.size.cols == 4)
				{
					cnnpp.conv_4x4(cnn.layer_buffer[0].conv_buffer[i](), cnn.layer_buffer[0].conv_buffer_size.cols, image.data, image.widthStep, cnn.input_buffer_size.rows, cnn.conv_l1.kernels[i](), cnn.conv_l1.ROI.cols, cnn.conv_l1.ROI.rows);
					goto AF1;
				}
				if (cnn.conv_l1.size.rows == 3 && cnn.conv_l1.size.cols == 3)
				{
					cnnpp.conv_3x3(cnn.layer_buffer[0].conv_buffer[i](), cnn.layer_buffer[0].conv_buffer_size.cols, image.data, image.widthStep, cnn.input_buffer_size.rows, cnn.conv_l1.kernels[i](), cnn.conv_l1.ROI.cols, cnn.conv_l1.ROI.rows);
					goto AF1;
				}
				AF1:
				cnnpp.lrelu_bn_max(cnn.layer_buffer[0].pool_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].conv_buffer[i](), cnn.layer_buffer[0].conv_buffer_size.cols, cnn.layer_buffer[0].conv_buffer_size.rows, &(cnn.conv_bias[0][i]), &(cnn.leakyReLU_w1[0][i]), &(cnn.leakyReLU_w2[0][i]), &(cnn.bn_weight[0][i]), &(cnn.bn_bias[0][i]));
			}

			//for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			//{
			//	int count1 = 0;
			//	double d1 = 0;
			//	for (int y = 0; y < cnn.layer_buffer[0].pool_buffer_size.rows; ++y)
			//	{
			//		for (int x = 0; x < cnn.layer_buffer[0].pool_buffer_size.cols; ++x)
			//		{
			//			double d = cnn.layer_buffer[0].conv_buffer[i][y * cnn.layer_buffer[0].conv_buffer_size.cols + x];
			//			printf("%f ", d);
			//		}
			//		printf("\n");
			//	}
			//	printf("\n\n");
			//}
			//printf("\n\n");
			//for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			//{
			//	int count1 = 0;
			//	double d1 = 0;
			//	for (int y = 0; y < cnn.layer_buffer[0].pool_buffer_size.rows; ++y)
			//	{
			//		for (int x = 0; x < cnn.layer_buffer[0].pool_buffer_size.cols; ++x)
			//		{
			//			double d = cnn.layer_buffer[0].pool_buffer[i][y * cnn.layer_buffer[0].pool_buffer_size.cols + x];
			//			printf("%f ", d);
			//		}
			//		printf("\n");
			//	}
			//	printf("\n\n");
			//	break;
			//}
			//printf("\n\n");
			//printf("\n\n");
			
#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd: run_L1 = %7.3f ms (conv_l1, tanh_avr_tanh)\n", timer.get(1000));
			timer.start();
#endif

			const int it1 = cnn.layer_buffer[1].map_count >> 1; // div on 2
			//OMP_PRAGMA(omp parallel for num_threads(num_threads))
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

				cnnpp.conv_3x3(cnn.layer_buffer[1].conv_buffer[2 * i](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].pool_buffer_size.rows, cnn.conv_l2.kernels[2 * i](), cnn.conv_l2.ROI.cols, cnn.conv_l2.ROI.rows);
				cnnpp.lrelu_bn_max(cnn.layer_buffer[1].pool_buffer[2 * i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].conv_buffer[2 * i](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[1].conv_buffer_size.rows, &(cnn.conv_bias[1][2 * i]), &(cnn.leakyReLU_w1[1][2 * i]), &(cnn.leakyReLU_w2[1][2 * i]), &(cnn.bn_weight[1][2 * i]), &(cnn.bn_bias[1][2 * i]));

				cnnpp.conv_3x3(cnn.layer_buffer[1].conv_buffer[2 * i + 1](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[0].sum_buffer[i](), cnn.layer_buffer[0].pool_buffer_size.cols, cnn.layer_buffer[0].pool_buffer_size.rows, cnn.conv_l2.kernels[2 * i + 1](), cnn.conv_l2.ROI.cols, cnn.conv_l2.ROI.rows);
				cnnpp.lrelu_bn_max(cnn.layer_buffer[1].pool_buffer[2 * i + 1](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].conv_buffer[2 * i + 1](), cnn.layer_buffer[1].conv_buffer_size.cols, cnn.layer_buffer[1].conv_buffer_size.rows, &(cnn.conv_bias[1][2 * i + 1]), &(cnn.leakyReLU_w1[1][2 * i + 1]), &(cnn.leakyReLU_w2[1][2 * i + 1]), &(cnn.bn_weight[1][2 * i + 1]), &(cnn.bn_bias[1][2 * i + 1]));
			}

			//for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			//{
			//	int count1 = 0;
			//	double d1 = 0;
			//	for (int y = 0; y < cnn.layer_buffer[1].pool_buffer_size.rows; ++y)
			//	{
			//		for (int x = 0; x < cnn.layer_buffer[1].pool_buffer_size.cols; ++x)
			//		{
			//			double d = cnn.layer_buffer[1].conv_buffer[i][y * cnn.layer_buffer[1].pool_buffer_size.cols + x];
			//			double d2 = cnn.layer_buffer[1].pool_buffer[i][y * cnn.layer_buffer[1].pool_buffer_size.cols + x];
			//			printf("%f(%f) ", d, d2);
			//		}
			//		printf("\n");
			//	}
			//	printf("\n\n");
			//	break;
			//}
			//printf("\n\n");
			//printf("\n\n");

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd: run_L2 = %7.3f ms (sum, conv_l2, tanh_avr_tanh)\n", timer.get(1000));
			timer.start();
#endif

			int it2 = cnn.layer_buffer[2].map_count >> 1; // div on 2
			//OMP_PRAGMA(omp parallel for num_threads(num_threads))
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

				if (cnn.conv_l3.size.rows == 8 && cnn.conv_l3.size.cols == 7)
				{
					cnnpp.conv_8x7(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					cnnpp.conv_8x7(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					goto AF3;
				}
				if (cnn.conv_l3.size.rows == 3 && cnn.conv_l3.size.cols == 3)
				{
					cnnpp.conv_3x3(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					cnnpp.conv_3x3(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					goto AF3;
				}
				if (cnn.conv_l3.size.rows == 4 && cnn.conv_l3.size.cols == 4)
				{
					cnnpp.conv_4x4(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					cnnpp.conv_4x4(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					goto AF3;
				}
				if (cnn.conv_l3.size.rows == 5 && cnn.conv_l3.size.cols == 4)
				{
					cnnpp.conv_5x4(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					cnnpp.conv_5x4(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					goto AF3;
				}
				if (cnn.conv_l3.size.rows == 5 && cnn.conv_l3.size.cols == 5)
				{
					cnnpp.conv_5x5(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					cnnpp.conv_5x5(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					goto AF3;
				}
				if (cnn.conv_l3.size.rows == 6 && cnn.conv_l3.size.cols == 5)
				{
					cnnpp.conv_6x5(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					cnnpp.conv_6x5(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					goto AF3;
				}
				if (cnn.conv_l3.size.rows == 6 && cnn.conv_l3.size.cols == 6)
				{
					cnnpp.conv_6x6(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					cnnpp.conv_6x6(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					goto AF3;
				}
				if (cnn.conv_l3.size.rows == 7 && cnn.conv_l3.size.cols == 7)
				{
					cnnpp.conv_7x7(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					cnnpp.conv_7x7(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					goto AF3;
				}
				if (cnn.conv_l3.size.rows == 8 && cnn.conv_l3.size.cols == 8)
				{
					cnnpp.conv_8x8(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					cnnpp.conv_8x8(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					goto AF3;
				}
				if (cnn.conv_l3.size.rows == 11 && cnn.conv_l3.size.cols == 10)
				{
					cnnpp.conv_11x10(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					cnnpp.conv_11x10(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					goto AF3;
				}
				if (cnn.conv_l3.size.rows == 11 && cnn.conv_l3.size.cols == 11)
				{
					cnnpp.conv_11x11(cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					cnnpp.conv_11x11(cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer_size.cols, cnn.layer_buffer[1].sum_buffer[i](), cnn.layer_buffer[1].pool_buffer_size.cols, cnn.layer_buffer[1].pool_buffer_size.rows, cnn.conv_l3.kernels[2 * i + 1](), cnn.conv_l3.ROI.cols, cnn.conv_l3.ROI.rows);
					goto AF3;
				}
				AF3:
				cnnpp.lrelu_bn(cnn.layer_buffer[2].pool_buffer[2 * i](), cnn.layer_buffer[2].conv_buffer[2 * i](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i]), &(cnn.leakyReLU_w1[2][2 * i]), &(cnn.leakyReLU_w2[2][2 * i]), &(cnn.bn_weight[2][2 * i]), &(cnn.bn_bias[2][2 * i]));
				cnnpp.lrelu_bn(cnn.layer_buffer[2].pool_buffer[2 * i + 1](), cnn.layer_buffer[2].conv_buffer[2 * i + 1](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.conv_bias[2][2 * i + 1]), &(cnn.leakyReLU_w1[2][2 * i + 1]), &(cnn.leakyReLU_w2[2][2 * i + 1]), &(cnn.bn_weight[2][2 * i + 1]), &(cnn.bn_bias[2][2 * i + 1]));
			}

			//for (int i = 0; i < cnn.layer_buffer[2].map_count; ++i)
			//{
			//	int count1 = 0;
			//	double d1 = 0;
			//	for (int y = 0; y < cnn.layer_buffer[2].pool_buffer_size.rows; ++y)
			//	{
			//		for (int x = 0; x < cnn.layer_buffer[2].pool_buffer_size.cols; ++x)
			//		{
			//			double d = cnn.layer_buffer[2].pool_buffer[i][y * cnn.layer_buffer[2].pool_buffer_size.cols + x];
			//			printf("%f ", d);
			//		}
			//		printf("\n");
			//	}
			//	printf("\n\n");
			//	//break;
			//}
			//printf("\n\n");
			//printf("\n\n");

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd: run_L3 = %7.3f ms (sum, conv_l3, tanh_tanh)\n", timer.get(1000));
			timer.start();
#endif

			const int it3 = cnn.snn_hl_size;
			it2 = it2 << 1; // mul on 2;
			//OMP_PRAGMA(omp parallel for num_threads(num_threads))			
			for (int i = 0; i < (it3 / cnn.hl_scale); ++i)
			{
				for (int t = 0; t < cnn.hl_scale; ++t)
				{
					cnnpp.mulCN_add_tanhW(cnn.snn_connect_count, cnn.hl_buffer[cnn.hl_scale * i + t](), cnn.pool3_buffer_ref(i), cnn.layer_buffer[2].pool_buffer_size.size, cnn.snn_hl_weight[cnn.hl_scale * i + t](), &(cnn.snn_hl_bias[cnn.hl_scale * i + t]), &(cnn.snn_hl_tanh_w[i]), &(cnn.snn_hl_bn_weight[t]), &(cnn.snn_hl_bn_bias[t]));
				}
			}

			if (cnn.index_output >= 0)
			{
				for (int i = 0; i < (it3 / cnn.hl_scale); ++i)
				{
					for (int t = 0; t < cnn.hl_scale; ++t)
					{
						if (i == 0 && t == 0) {
							cnnpp.mulC(cnn.ol_buffer(), cnn.hl_buffer[cnn.hl_scale * i + t](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_ol_weight[cnn.index_output][t * (it3 / cnn.hl_scale) + i]));
						}
						else {
							cnnpp.mulC1_add(cnn.ol_buffer(), cnn.hl_buffer[cnn.hl_scale * i + t](), cnn.ol_buffer(), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_ol_weight[cnn.index_output][t * (it3 / cnn.hl_scale) + i]));
						}
					}
				}

				cnnpp.tanhW(cnn.ol_buffer(), cnn.ol_buffer(), cnn.ol_buffer_size.size, &(cnn.snn_ol_bias[cnn.index_output]), &(cnn.snn_ol_tanh_w), &(cnn.af_scale));			}
			else
			{
#if 0
				//RTSD
				float max_val = -1000.f;
				int max_id = 0;

				for (int out_id = 0; out_id < cnn.snn_ol_neuron_count; ++out_id)
				{
					if (out_id == 26) continue;
					if (out_id == 29) continue;
					if (out_id == 30) continue;
					if (out_id == 31) continue;

					for (int i = 0; i < (it3 / cnn.hl_scale); ++i)
					{
						for (int t = 0; t < cnn.hl_scale; ++t)
						{
							if (i == 0 && t == 0)
								cnnpp.mulC(cnn.ol_buffer(), cnn.hl_buffer[cnn.hl_scale * i + t](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_ol_weight[out_id][t * (it3 / cnn.hl_scale) + i]));
							else
								cnnpp.mulC1_add(cnn.ol_buffer(), cnn.hl_buffer[cnn.hl_scale * i + t](), cnn.ol_buffer(), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_ol_weight[out_id][t * (it3 / cnn.hl_scale) + i]));
						}
					}

					cnnpp.tanhW(cnn.ol_buffer(), cnn.ol_buffer(), cnn.ol_buffer_size.size, &(cnn.snn_ol_bias[out_id]), &(cnn.snn_ol_tanh_w), &(cnn.af_scale));
					//printf("%f\n", cnn.ol_buffer[0]);
					if (max_val < cnn.ol_buffer[0])
					{
						max_id = out_id;
						max_val = cnn.ol_buffer[0];
					}
				}

				if (max_id == 21) max_id = 27;
				cnn.ol_buffer[0] = float(max_id);
#endif

				for (int out_id = 0; out_id < cnn.snn_ol_neuron_count; ++out_id)
				{
					for (int i = 0; i < (it3 / cnn.hl_scale); ++i)
					{
						for (int t = 0; t < cnn.hl_scale; ++t)
						{
							if (i == 0 && t == 0)
								cnnpp.mulC(cnn.ol_buffer(out_id * cnn.layer_buffer[2].pool_buffer_size.size), cnn.hl_buffer[cnn.hl_scale * i + t](), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_ol_weight[out_id][t * (it3 / cnn.hl_scale) + i]));
							else
								cnnpp.mulC1_add(cnn.ol_buffer(out_id * cnn.layer_buffer[2].pool_buffer_size.size), cnn.hl_buffer[cnn.hl_scale * i + t](), cnn.ol_buffer(out_id * cnn.layer_buffer[2].pool_buffer_size.size), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_ol_weight[out_id][t * (it3 / cnn.hl_scale) + i]));
						}
					}

					cnnpp.tanhW(cnn.ol_buffer(out_id * cnn.layer_buffer[2].pool_buffer_size.size), cnn.ol_buffer(out_id * cnn.layer_buffer[2].pool_buffer_size.size), cnn.layer_buffer[2].pool_buffer_size.size, &(cnn.snn_ol_bias[out_id]), &(cnn.snn_ol_tanh_w), &(cnn.af_scale));
				}
			}

#ifdef USE_AVX
			_mm256_zeroupper();
#endif

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd: run_HL = %7.3f ms (sum, mul, tanh, sum, tanh)\n", timer.get(1000));
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

			Run(image);
			
			if (response_map.isEmpty())
			{
				response_map.width = cnn.output_buffer_size.cols;
				response_map.height = cnn.output_buffer_size.rows;
				response_map.widthStep = cnn.output_buffer_size.step;
				response_map.data = cnn.ol_buffer();
				response_map.sharingData = true;
			}
			else
			{
				response_map.width = cnn.output_buffer_size.cols;
				response_map.height = cnn.output_buffer_size.rows;
				response_map.copyData(cnn.output_buffer_size.cols, cnn.output_buffer_size.rows, cnn.ol_buffer(), cnn.output_buffer_size.step);
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

#if 0
		void ConvNeuralNetwork::SaveToBinaryFile(std::string file_name, void* hGrd)
		{
#if defined(USE_SSE) || defined(USE_AVX)
			printf("[SIMD::CNN] SIMD code is not supported!\n");
			return;
#endif

			if (file_name == "") return;

			if (file_name.find(".bin") == std::string::npos)
			{
				file_name.append(".bin");
			}

			std::fstream file_bin;
			file_bin.open(file_name.c_str(), std::fstream::binary | std::fstream::out);

			if (!file_bin.is_open()) return;

			//version
			float format_version = 1.1f;
			FB_WRITE(file_bin, format_version);

			//max pool
			FB_WRITE(file_bin, cnn.max_pool);

			//min input_image
			FB_WRITE(file_bin, cnn.min_image_size.width);
			FB_WRITE(file_bin, cnn.min_image_size.height);

			//cnn_layers
			FB_WRITE(file_bin, cnn.layer_count);
			for (int i = 0; i < cnn.layer_count; ++i)
			{
				FB_WRITE(file_bin, cnn.layer_buffer[i].map_count);
			}

			//conv l1
			FB_WRITE(file_bin, cnn.conv_l1.size.cols);
			FB_WRITE(file_bin, cnn.conv_l1.size.rows);
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				for (int j = 0; j < cnn.conv_l1.size.size; ++j)
				{
					FB_WRITE(file_bin, cnn.conv_l1.kernels[i][j]);
				}
			}

			//conv l2
			FB_WRITE(file_bin, cnn.conv_l2.size.cols);
			FB_WRITE(file_bin, cnn.conv_l2.size.rows);
			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				for (int j = 0; j < cnn.conv_l2.size.size; ++j)
				{
					FB_WRITE(file_bin, cnn.conv_l2.kernels[i][j]);
				}
			}

			//conv l3
			FB_WRITE(file_bin, cnn.conv_l3.size.cols);
			FB_WRITE(file_bin, cnn.conv_l3.size.rows);
			for (int i = 0; i < cnn.layer_buffer[2].map_count; ++i)
			{
				for (int j = 0; j < cnn.conv_l3.size.size; ++j)
				{
					FB_WRITE(file_bin, cnn.conv_l3.kernels[i][j]);
				}
			}

			//conv nn weight
			const float abs_af_scale = fabsf(cnn.af_scale);
			FB_WRITE(file_bin, abs_af_scale);

			for (int i = 0; i < cnn.layer_count; ++i)
			{
				for (int j = 0; j < cnn.layer_buffer[i].map_count; ++j)
				{
					FB_WRITE(file_bin, cnn.conv_bias[i][j]);
					FB_WRITE(file_bin, cnn.leakyReLU_w1[i][j]);
					FB_WRITE(file_bin, cnn.leakyReLU_w2[i][j]);
					FB_WRITE(file_bin, cnn.bn_weight[i][j]);
					FB_WRITE(file_bin, cnn.bn_bias[i][j]);
				}
			}

			//simple nn weight
			FB_WRITE(file_bin, cnn.snn_full_connect);
			FB_WRITE(file_bin, cnn.snn_hl_size);
			FB_WRITE(file_bin, cnn.snn_connect_count);
			FB_WRITE(file_bin, cnn.hl_scale);
			if (!cnn.snn_full_connect)
			{
				for (int i = 0; i < cnn.snn_hl_size; ++i)
				{
					for (int j = 0; j < cnn.snn_connect_count; ++j)
					{
						FB_WRITE(file_bin, cnn.snn_hl_weight[i][j]);
					}
				}
			}
			else
			{
				for (int i = 0; i < cnn.snn_hl_size; ++i)
				{
					for (int j = 0; j < cnn.layer_buffer[cnn.layer_count - 1].map_count; ++j)
					{
						FB_WRITE(file_bin, cnn.snn_hl_weight[i][j]);
					}
				}
			}

			for (int i = 0; i < cnn.snn_hl_size; ++i)
			{
				FB_WRITE(file_bin, cnn.snn_hl_bias[i]);
			}

			for (int i = 0; i < (cnn.snn_hl_size / cnn.hl_scale); ++i)
			{
				FB_WRITE(file_bin, cnn.snn_hl_tanh_w[i]);
			}

			for (int i = 0; i < cnn.hl_scale; ++i)
			{
				FB_WRITE(file_bin, cnn.snn_hl_bn_weight[i]);
				FB_WRITE(file_bin, cnn.snn_hl_bn_bias[i]);
			}

			FB_WRITE(file_bin, cnn.snn_ol_neuron_count);
			for (int i = 0; i < cnn.snn_ol_neuron_count; ++i)
			{
				for (int j = 0; j < cnn.snn_hl_size; ++j)
				{
					FB_WRITE(file_bin, cnn.snn_ol_weight[i][j]);
				}
			}

			for (int i = 0; i < cnn.snn_ol_neuron_count; ++i)
			{
				FB_WRITE(file_bin, cnn.snn_ol_bias[i]);
			}

			FB_WRITE(file_bin, cnn.snn_ol_tanh_w);

			file_bin.close();
		}

		
		void ConvNeuralNetwork::ParserDump(std::vector<float>& param, std::vector<std::string>& dump, std::string l_name, std::string p_name, int idx)
		{
			using namespace std;
			param.clear();

			if (l_name == "features")
			{
				for (string str : dump)
				{
					if (str.find("features") != std::string::npos)
					{
						string subs = strstr(str.c_str(), "[");
						istringstream iss(subs);
						vector<string> tokens{ istream_iterator<string>{iss}, istream_iterator<string>{} };
						param.push_back((float)stoi(tokens[1]));
						param.push_back((float)stoi(tokens[3]));
						break;
					}
				}
				return;
			}

			if (l_name == "cnn_struct")
			{
				int layer, surf, hl;
				for (layer = 1; layer < 10E6; ++layer)
				{
					bool bl = true;
					for (string str : dump)
					{
						string sl = "z.c" + to_string(layer);
						if (str.find(sl) != std::string::npos)
						{
							bl = false;
							break;
						}
					}
					if (bl) break;
				}
				param.push_back(--layer);

				for (int l = 1; l <= layer; ++l)
				{
					for (surf = 1; surf < 10E6; ++surf)
					{
						bool bl = true;
						for (string str : dump)
						{
							string ls = "z.c" + to_string(l) + to_string(surf);
							if (str.find(ls) != std::string::npos)
							{
								bl = false;
								break;
							}
						}
						if (bl) break;
					}
					param.push_back(--surf);
				}

				for (hl = 1; hl < 10E6; ++hl)
				{
					bool bl = true;
					for (string str : dump)
					{
						string ls = "z.h" + to_string(hl);
						if (str.find(ls) != std::string::npos)
						{
							bl = false;
							break;
						}
					}
					if (bl) break;
				}
				param.push_back(--hl);

				return;
			}

			if (l_name == "pooling")
			{
				for (int i = 0; i < dump.size(); ++i)
				{
					string str = dump[i];

					if (str.find("Pooling") != std::string::npos)
					{
						string str = dump[i+1];
						istringstream iss(str);
						vector<string> tokens{ istream_iterator<string>{iss}, istream_iterator<string>{} };
						param.push_back((float)stoi(tokens[1]));
						break;
					}
				}
				return;
			}

			if (l_name == "snn_full_connect")
			{
				for (int i = 0; i < dump.size(); ++i)
				{
					string str = dump[i];

					if (str.find("z.h1.c=Convolution") != std::string::npos)
					{
						string subs = strstr(str.c_str(), "Kernel:");
						istringstream iss(subs);
						vector<string> tokens{ istream_iterator<string>{iss}, istream_iterator<string>{} };
						param.push_back((float)stoi(tokens[1]));
						param.push_back((float)stoi(tokens[3]));
						break;
					}
				}
				return;
			}

			if (l_name == "hl_count")
			{
				int hl;
				for (hl = 1; hl < 10E6; ++hl)
				{
					bool bl = true;
					for (string str : dump)
					{
						string ls = "z.h" + to_string(hl);
						if (str.find(ls) != std::string::npos)
						{
							bl = false;
							break;
						}
					}
					if (bl) break;
				}
				param.push_back(--hl);

				for (int i = 0; i < dump.size(); ++i)
				{
					string str = dump[i];

					if (str.find("z.h1.c=Convolution") != std::string::npos)
					{
						string subs = strstr(str.c_str(), "Kernel:");
						istringstream iss(subs);
						vector<string> tokens{ istream_iterator<string>{iss}, istream_iterator<string>{} };
						param.push_back((float)stoi(tokens[1]));
						param.push_back((float)stoi(tokens[7]));
						break;
					}
				}
				return;
			}

			for (int i = 0; i < dump.size(); ++i)
			{
				string str = dump[i];
			
				string sl = "z." + l_name + "." + p_name + "=";
				if (dump[i].find(sl) != std::string::npos)
				{
					if (p_name == "c")
					{
						string subs = strstr(str.c_str(), "Kernel:");
						istringstream iss(subs);
						vector<string> tokens{ istream_iterator<string>{iss}, istream_iterator<string>{} };
						param.push_back((float)stoi(tokens[1]));
						param.push_back((float)stoi(tokens[3]));
						break;
					}

					int k = 0;
					for (int j = 1; j < 10E6; ++j)
					{
						string str = dump[i + j];
						if (str.find("#") != std::string::npos)
							break;

						istringstream iss(str);
						vector<string> tokens{ istream_iterator<string>{iss}, istream_iterator<string>{} };

						for (string sval : tokens)
						{
							if (idx == -1 || k == idx)
								param.push_back(stof(sval));
						}
						k++;
					}

					break;
				}
			}
		}
		void ConvNeuralNetwork::LoadCNTKModel(std::string file_name, bool preprocessing)
		{
#if defined(USE_SSE) || defined(USE_AVX)
			printf("[SIMD::CNN] SIMD code is not supported!\n");
			return;
#endif

			std::fstream file_dump;
			file_dump.open(file_name.c_str(), std::fstream::in);

			if (!file_dump.is_open())
			{
				printf("[SIMD::CNN] Configuration file not found!\n");
				return;
			}

			cnn.af_scale = 1.7159f;

			std::vector<std::string> dump_cntk;
			while (file_dump)
			{
				std::string str;
				std::getline(file_dump, str);

				std::string p("._");
				std::string::size_type n = p.length();
				for (std::string::size_type i = str.find(p);
					i != std::string::npos;
					i = str.find(p))
					str.erase(i, n);

				dump_cntk.push_back(str);
			}

			//min input size
			std::vector<float> param;
			ParserDump(param, dump_cntk, "features", "");
			cnn.min_image_size.width = (int)param[0];
			cnn.min_image_size.height = (int)param[1];

			//pooling type
			ParserDump(param, dump_cntk, "pooling", "");
			if ((int)param[0] == 1)
			{
				cnn.max_pool = true;
			}

			ParserDump(param, dump_cntk, "cnn_struct", "");

			//initial buffers
			cnn.layer_count = (int)param[0];
			cnn.layer_buffer.resize(cnn.layer_count);
			for (int i = 0; i < cnn.layer_count; ++i)
			{
				cnn.layer_buffer[i].map_count = (int)param[i+1];
				cnn.layer_buffer[i].conv_buffer.resize(cnn.layer_buffer[i].map_count);
				cnn.layer_buffer[i].pool_buffer.resize(cnn.layer_buffer[i].map_count);
				cnn.layer_buffer[i].sum_buffer.resize(cnn.layer_buffer[i].map_count);
			}

			//initial weight
			//conv kernels l1
			ParserDump(param, dump_cntk, "c11", "c");
			int kernel_width = (int)param[0];
			int kernel_height = (int)param[1];
			cnn.conv_l1.size = Size2d(kernel_width, kernel_height);
			cnn.conv_l1.kernels.resize(cnn.layer_buffer[0].map_count);

			int iBufferSize = kernel_width * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				cnn.conv_l1.kernels[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			std::vector<double> preproc(cnn.layer_buffer[0].map_count);
			for (int k = 0; k < cnn.layer_buffer[0].map_count; ++k)
			{
				ParserDump(param, dump_cntk, "c1" + std::to_string(k+1), "W");

				std::vector<float> param_mod;
				for (int y = 0; y < kernel_height; ++y)
				{
					for (int x = 0; x < kernel_width; ++x)
					{
						float val = param[3 * kernel_width * y + 0 * kernel_width + x] +
							        param[3 * kernel_width * y + 1 * kernel_width + x] +
							        param[3 * kernel_width * y + 2 * kernel_width + x];
						param_mod.push_back(val);
					}
				}

				//transpose
				std::vector<float> param_transp(iBufferSize);
				for (int n = 0; n < iBufferSize; ++n)
				{
					int x = n / kernel_width;
					int y = n % kernel_width;
					param_transp[n] = param_mod[kernel_height * y + x];
				}

				for (int i = 0; i < iBufferSize; ++i)
				{
					cnn.conv_l1.kernels[k][i] = param_transp[i];

					if (preprocessing)
					{
						cnn.conv_l1.kernels[k][i] *= 0.0078125f;

						preproc[k] += -1.0 * (double)param_transp[i];
					}
				}
			}

			//conv kernels l2
			ParserDump(param, dump_cntk, "c21", "c");
			kernel_width = (int)param[0];
			kernel_height = (int)param[1];
			cnn.conv_l2.size = Size2d(kernel_width, kernel_height);
			cnn.conv_l2.kernels.resize(cnn.layer_buffer[1].map_count);

			iBufferSize = kernel_width * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				cnn.conv_l2.kernels[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			for (int k = 0; k < cnn.layer_buffer[1].map_count; ++k)
			{
				ParserDump(param, dump_cntk, "c2" + std::to_string(k + 1), "W");

				//transpose
				std::vector<float> param_transp(iBufferSize);
				for (int n = 0; n < iBufferSize; ++n)
				{
					int x = n / kernel_width;
					int y = n % kernel_width;
					param_transp[n] = param[kernel_height * y + x];
				}

				for (int i = 0; i < iBufferSize; ++i)
				{
					cnn.conv_l2.kernels[k][i] = param_transp[i];
				}
			}

			//conv kernels l3
			ParserDump(param, dump_cntk, "c31", "c");
			kernel_width = (int)param[0];
			kernel_height = (int)param[1];
			cnn.conv_l3.size = Size2d(kernel_width, kernel_height);
			cnn.conv_l3.kernels.resize(cnn.layer_buffer[2].map_count);

			iBufferSize = kernel_width * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[2].map_count; ++i)
			{
				cnn.conv_l3.kernels[i] = Array_32f(iBufferSize, ALIGN_DEF);
			}

			for (int k = 0; k < cnn.layer_buffer[2].map_count; ++k)
			{
				ParserDump(param, dump_cntk, "c3" + std::to_string(k + 1), "W");

				//transpose
				std::vector<float> param_transp(iBufferSize);
				for (int n = 0; n < iBufferSize; ++n)
				{
					int x = n / kernel_width;
					int y = n % kernel_width;
					param_transp[n] = param[kernel_height * y + x];
				}

				for (int i = 0; i < iBufferSize; ++i)
				{
					cnn.conv_l3.kernels[k][i] = param_transp[i];
				}
			}

			//leakyReLU weight
			cnn.leakyReLU_w1.resize(cnn.layer_count);
			cnn.leakyReLU_w2.resize(cnn.layer_count);

			for (int i = 0; i < cnn.layer_count; ++i)
			{
				cnn.leakyReLU_w1[i] = Array_32f(cnn.layer_buffer[i].map_count, ALIGN_DEF);
				cnn.leakyReLU_w2[i] = Array_32f(cnn.layer_buffer[i].map_count, ALIGN_DEF);
				for (int j = 0; j < cnn.layer_buffer[i].map_count; ++j)
				{
					ParserDump(param, dump_cntk, "c" + std::to_string(i + 1) + std::to_string(j + 1), "res.PlusArgs[0].ElementTimesArgs[0]");
					cnn.leakyReLU_w1[i][j] = param[0];

					ParserDump(param, dump_cntk, "c" + std::to_string(i + 1) + std::to_string(j + 1), "res.PlusArgs[1].ElementTimesArgs[0]");
					cnn.leakyReLU_w2[i][j] = param[0];
				}
			}

			//conv nn weight
			cnn.conv_bias.resize(cnn.layer_count);
			cnn.bn_weight.resize(cnn.layer_count);
			cnn.bn_bias.resize(cnn.layer_count);

			for (int i = 0; i < cnn.layer_count; ++i)
			{
				iBufferSize = cnn.layer_buffer[i].map_count;
				cnn.conv_bias[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.bn_weight[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.bn_bias[i] = Array_32f(iBufferSize, ALIGN_DEF);

				for (int j = 0; j < cnn.layer_buffer[i].map_count; ++j)
				{
					ParserDump(param, dump_cntk, "c" + std::to_string(i + 1) + std::to_string(j + 1), "b");
					cnn.conv_bias[i][j] = param[0];

					ParserDump(param, dump_cntk, "bn" + std::to_string(i + 1) + std::to_string(j + 1), "scale");
					float bn_scale = param[0];

					ParserDump(param, dump_cntk, "bn" + std::to_string(i + 1) + std::to_string(j + 1), "bias");
					float bn_bias = param[0];

					ParserDump(param, dump_cntk, "bn" + std::to_string(i + 1) + std::to_string(j + 1), "runMean");
					float bn_runMean = param[0];

					ParserDump(param, dump_cntk, "bn" + std::to_string(i + 1) + std::to_string(j + 1), "runVariance");
					float bn_runVariance = param[0] + 0.00001f;

					cnn.bn_weight[i][j] = bn_scale / sqrtf(bn_runVariance);
					cnn.bn_bias[i][j] = bn_bias - bn_runMean * bn_scale / sqrtf(bn_runVariance);

					cnn.leakyReLU_w1[i][j] *= cnn.bn_weight[i][j];
					cnn.leakyReLU_w2[i][j] *= cnn.bn_weight[i][j];

					if (preprocessing && i == 0)
					{
						cnn.conv_bias[i][j] += (float)preproc[j];
					}
				}
			}

			//simple nn weight
			ParserDump(param, dump_cntk, "hl_count", "W");
			cnn.hl_scale = (int)param[2];
			cnn.snn_connect_count = (int)param[1];
			cnn.snn_hl_size = cnn.hl_scale * (int)param[0];

			ParserDump(param, dump_cntk, "snn_full_connect", "");
			if ((int)param[0] == cnn.layer_buffer[cnn.layer_count-1].map_count || (int)param[1] == cnn.layer_buffer[cnn.layer_count - 1].map_count)
			{
				cnn.snn_full_connect = true;
			}

			cnn.snn_hl_weight.resize(cnn.snn_hl_size);
			for (int i = 0; i < cnn.snn_hl_size / cnn.hl_scale; ++i)
			{
				ParserDump(param, dump_cntk, "h" + std::to_string(i + 1), "W");
				int w_count = cnn.snn_connect_count;

				//transpose
				std::vector<float> param_transp(param.size());
				for (int n = 0; n < param.size(); ++n)
				{
					int x = n / w_count;
					int y = n % w_count;
					param_transp[n] = param[cnn.hl_scale * y + x];
				}

				for (int j = 0; j < cnn.hl_scale; ++j)
				{
					cnn.snn_hl_weight[cnn.hl_scale * i + j] = Array_32f(w_count, ALIGN_DEF);
					for (int k = 0; k < w_count; ++k)
					{
						cnn.snn_hl_weight[cnn.hl_scale * i + j][k] = param_transp[j * w_count + k];
					}
				}
			}
		
			cnn.snn_hl_bias = Array_32f(cnn.snn_hl_size, ALIGN_DEF);
			cnn.snn_hl_tanh_w = Array_32f(cnn.snn_hl_size / cnn.hl_scale, ALIGN_DEF);
			for (int i = 0; i < cnn.snn_hl_size / cnn.hl_scale; ++i)
			{
				ParserDump(param, dump_cntk, "h" + std::to_string(i + 1), "b");
				for (int j = 0; j < cnn.hl_scale; ++j)
				{
					cnn.snn_hl_bias[cnn.hl_scale * i + j] = param[j];
				}

				ParserDump(param, dump_cntk, "h" + std::to_string(i + 1), "res.PlusArgs[0].ElementTimesArgs[1].x.ElementTimesArgs[0]");
				cnn.snn_hl_tanh_w[i] = param[0];
			}

			cnn.snn_hl_bn_weight = Array_32f(cnn.hl_scale, ALIGN_DEF);
			cnn.snn_hl_bn_bias = Array_32f(cnn.hl_scale, ALIGN_DEF);
			for (int i = 0; i < cnn.hl_scale; ++i)
			{
				ParserDump(param, dump_cntk, "bn_h", "scale", i);
				float bn_scale = param[0];

				ParserDump(param, dump_cntk, "bn_h", "bias", i);
				float bn_bias = param[0];

				ParserDump(param, dump_cntk, "bn_h", "runMean", i);
				float bn_runMean = param[0];

				ParserDump(param, dump_cntk, "bn_h", "runVariance", i);
				float bn_runVariance = param[0] + 0.00001f;

				cnn.snn_hl_bn_weight[i] = bn_scale / sqrtf(bn_runVariance);
				cnn.snn_hl_bn_bias[i] = bn_bias - bn_runMean * bn_scale / sqrtf(bn_runVariance);
			}

			ParserDump(param, dump_cntk, "z", "arrayOfFunctions[0].W");
			cnn.snn_ol_weight.resize(param.size() / cnn.snn_hl_size);

			for (int i = 0; i < param.size() / cnn.snn_hl_size; ++i)
			{
				cnn.snn_ol_weight[i] = Array_32f(cnn.snn_hl_size, ALIGN_DEF);
				for (int j = 0; j < cnn.snn_hl_size; ++j)
				{
					cnn.snn_ol_weight[i][j] = param[i * cnn.snn_hl_size + j];
				}
			}

			ParserDump(param, dump_cntk, "z", "arrayOfFunctions[0].b");
			cnn.snn_ol_neuron_count = (int)param.size();
			cnn.snn_ol_bias = Array_32f(cnn.snn_ol_neuron_count, ALIGN_DEF);
			for (int i = 0; i < cnn.snn_ol_neuron_count; ++i)
			{
				cnn.snn_ol_bias[i] = param[i];
			}

			if (cnn.snn_ol_neuron_count > 3)
			{
				cnn.af_scale = 0.f;
			}

			ParserDump(param, dump_cntk, "z", "PlusArgs[0].ElementTimesArgs[1].x.ElementTimesArgs[0]");
			cnn.snn_ol_tanh_w = param[0];

			file_dump.close();

			//set num threads
			num_threads = 1;
#ifdef USE_OMP
			num_threads = omp_get_num_procs();
#endif
		}
#endif

	}

#endif
}