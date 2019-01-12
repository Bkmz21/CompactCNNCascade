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


#include "cnn_simd_v2.h"
#include <fstream>
#include <sstream>
#include <iterator>

#ifdef USE_OMP
#	include <omp.h>
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{
#if defined(USE_AVX) && !defined(USE_CNTK_MODELS)

	namespace SIMD
	{
		void ConvNeuralNetwork_v2::Init(std::string file_name, int index_output, void* hGrd)
		{
			std::stringstream data_bin;
			if (file_name.size() < 255)
			{
				std::fstream data_bin;
				data_bin.open(file_name.c_str(), std::fstream::binary | std::fstream::in);

				if (!data_bin.is_open())
				{
					printf("[SIMD::CNN_v2] Configuration file not found!\n");
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
				printf("[SIMD::CNN_v2] Configuration file format is not supported!\n");
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
			}

			//initial weight
			//conv kernels l1
			int kernel_width = 0;
			int kernel_height = 0;
			FB_READ(data_bin, kernel_width);
			FB_READ(data_bin, kernel_height);

			cnn.conv_l1.size = Size2d(kernel_width, kernel_height);
			int iBufferSize = 2 * kernel_width * kernel_height * cnn.layer_buffer[0].map_count;
			cnn.conv_l1.kernels = Array_32f(iBufferSize, ALIGN_DEF);

			for (int k = 0; k < cnn.layer_buffer[0].map_count; ++k)
			{
				for (int i = 0; i < cnn.conv_l1.size.size; ++i)
				{				
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);
				
					int offset = i * 2 * cnn.layer_buffer[0].map_count;
					cnn.conv_l1.kernels[offset + k] = kernel_val;
				
					const int t = i % 4;
					if (t > 0 && t < 3)
					{
						cnn.conv_l1.kernels[offset + 4 + k] = cnn.conv_l1.kernels[offset - 2 * cnn.layer_buffer[0].map_count + k];
					}
					if (t == 3)
					{
						cnn.conv_l1.kernels[offset + 4 + k] = cnn.conv_l1.kernels[offset - 2 * cnn.layer_buffer[0].map_count + k];
						cnn.conv_l1.kernels[offset - 3 * 2 * cnn.layer_buffer[0].map_count + 4 + k] = cnn.conv_l1.kernels[offset + k];
					}
				}
			}

			//conv kernels l2
			FB_READ(data_bin, kernel_width);
			FB_READ(data_bin, kernel_height);

			cnn.conv_l2.size = Size2d(kernel_width, kernel_height);
			iBufferSize = kernel_width * kernel_height * cnn.layer_buffer[1].map_count;
			cnn.conv_l2.kernels = Array_32f(iBufferSize, ALIGN_DEF);

			for (int k = 0; k < cnn.layer_buffer[1].map_count; ++k)
			{
				for (int i = 0; i < cnn.conv_l2.size.size; ++i)
				{
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);
					
					if (k < 2) cnn.conv_l2.kernels[k + i * cnn.layer_buffer[1].map_count] = kernel_val;
					if (k >= 6) cnn.conv_l2.kernels[k - 4 + i * cnn.layer_buffer[1].map_count] = kernel_val;
					if (k >= 2 && k < 6) cnn.conv_l2.kernels[k + 2 + i * cnn.layer_buffer[1].map_count] = kernel_val;
				}
			}

			//conv kernels l3
			FB_READ(data_bin, kernel_width);
			FB_READ(data_bin, kernel_height);

			cnn.conv_l3.size = Size2d(kernel_width, kernel_height);
			iBufferSize = kernel_width * kernel_height * cnn.layer_buffer[2].map_count;
			cnn.conv_l3.kernels = Array_32f(iBufferSize, ALIGN_DEF);

			for (int k = 0; k < cnn.layer_buffer[2].map_count; ++k)
			{
				for (int i = 0; i < cnn.conv_l3.size.size; ++i)
				{
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);

					if (k % 2 == 0)
					{
						cnn.conv_l3.kernels[(k + i * cnn.layer_buffer[2].map_count) / 2] = kernel_val;
					}
					else
					{
						int offset = cnn.conv_l3.size.size * (cnn.layer_buffer[2].map_count / 2);
						cnn.conv_l3.kernels[offset + (k + i * cnn.layer_buffer[2].map_count) / 2] = kernel_val;
					}
				}
			}

			//conv nn weight
			cnn.conv_bias.resize(cnn.layer_count);
			cnn.subs_weight.resize(cnn.layer_count);
			cnn.subs_bias.resize(cnn.layer_count);

			FB_READ(data_bin, cnn.af_scale);
			for (int i = 0; i < cnn.layer_count; ++i)
			{
				int L = MAX(1, REG_SIZE / cnn.layer_buffer[i].map_count);
				iBufferSize = L * cnn.layer_buffer[i].map_count;
				cnn.conv_bias[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.subs_weight[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.subs_bias[i] = Array_32f(iBufferSize, ALIGN_DEF);

				if (i < 2)
				{
					for (int j = 0; j < cnn.layer_buffer[i].map_count; ++j)
					{
						FB_READ(data_bin, cnn.conv_bias[i][j]);
						FB_READ(data_bin, cnn.subs_weight[i][j]);
						FB_READ(data_bin, cnn.subs_bias[i][j]);

						for (int k = 1; k < L; ++k)
						{
							cnn.conv_bias[i][j + k * cnn.layer_buffer[i].map_count] = cnn.conv_bias[i][j];
							cnn.subs_weight[i][j + k * cnn.layer_buffer[i].map_count] = cnn.subs_weight[i][j];
							cnn.subs_bias[i][j + k * cnn.layer_buffer[i].map_count] = cnn.subs_bias[i][j];
						}
					}
				}
				else
				{
					for (int k = 0; k < cnn.layer_buffer[i].map_count; ++k)
					{
						float kernel_val = 0.f;
						FB_READ(data_bin, kernel_val);

						if (k % 2 == 0)
							cnn.conv_bias[i][k / 2] = kernel_val;
						else
							cnn.conv_bias[i][(k + cnn.layer_buffer[i].map_count) / 2] = kernel_val;

						kernel_val = 0.f;
						FB_READ(data_bin, kernel_val);

						if (k % 2 == 0)
							cnn.subs_weight[i][k / 2] = kernel_val;
						else
							cnn.subs_weight[i][(k + cnn.layer_buffer[i].map_count) / 2] = kernel_val;

						kernel_val = 0.f;
						FB_READ(data_bin, kernel_val);

						if (k % 2 == 0)
							cnn.subs_bias[i][k / 2] = kernel_val;
						else
							cnn.subs_bias[i][(k + cnn.layer_buffer[i].map_count) / 2] = kernel_val;
					}
				}
			}

			//simple nn weight
			FB_READ(data_bin, cnn.snn_full_connect);
			FB_READ(data_bin, cnn.snn_hl_size);

			if (cnn.layer_count != 3 ||
				!cnn.max_pool ||
				cnn.snn_full_connect ||
				cnn.conv_l1.size.size != 16 ||
				cnn.conv_l2.size.size != 9 ||
				cnn.conv_l3.size.size != 30)
			{
				printf("[SIMD::CNN_v2] This configuration cnn models is not supported!\n");
				Clear();
				return;
			}

			if (!cnn.snn_full_connect)
			{
				cnn.snn_hl_weight = Array_32f(cnn.snn_hl_size, ALIGN_DEF);
				for (int k = 0; k < cnn.snn_hl_size; ++k)
				{
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);

					if (k % 4 < 2)
					{
						if (k % 2 == 0)
							cnn.snn_hl_weight[k / 4] = kernel_val;
						else
							cnn.snn_hl_weight[k / 4 + 8] = kernel_val;
					}
					else
					{
						if (k % 2 == 0)
							cnn.snn_hl_weight[k / 4 + cnn.snn_hl_size / 2] = kernel_val;
						else
							cnn.snn_hl_weight[k / 4 + cnn.snn_hl_size / 2 + 8] = kernel_val;
					}
				}
			}
			else { }
			
			cnn.snn_hl_bias = Array_32f(cnn.snn_hl_size, ALIGN_DEF);
			for (int k = 0; k < cnn.snn_hl_size; ++k)
			{
				float kernel_val = 0.f;
				FB_READ(data_bin, kernel_val);

				if (k % 4 < 2)
				{
					if (k % 2 == 0)
						cnn.snn_hl_bias[k / 4] = kernel_val;
					else
						cnn.snn_hl_bias[k / 4 + 8] = kernel_val;
				}
				else
				{
					if (k % 2 == 0)
						cnn.snn_hl_bias[k / 4 + cnn.snn_hl_size / 2] = kernel_val;
					else
						cnn.snn_hl_bias[k / 4 + cnn.snn_hl_size / 2 + 8] = kernel_val;
				}
			}

			cnn.snn_ol_weight = Array_32f(cnn.snn_hl_size, ALIGN_DEF);
			for (int k = 0; k < cnn.snn_hl_size; ++k)
			{
				float kernel_val = 0.f;
				FB_READ(data_bin, kernel_val);

				if (k % 4 < 2)
				{
					if (k % 2 == 0)
						cnn.snn_ol_weight[k / 4] = kernel_val;
					else
						cnn.snn_ol_weight[k / 4 + 8] = kernel_val;
				}
				else
				{
					if (k % 2 == 0)
						cnn.snn_ol_weight[k / 4 + cnn.snn_hl_size / 2] = kernel_val;
					else
						cnn.snn_ol_weight[k / 4 + cnn.snn_hl_size / 2 + 8] = kernel_val;
				}
			}

			FB_READ(data_bin, cnn.snn_ol_bias);

			//set num threads
			num_threads = 1;
#ifdef USE_OMP
			num_threads = omp_get_num_procs();
#endif
		}

		void ConvNeuralNetwork_v2::AllocateMemory(const Size size)
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
			cnn.layer_buffer[0].conv_buffer_size.cols = roundUpMul(cnn.layer_buffer[0].map_count * cnn.conv_l1.ROI.cols, REG_SIZE);
			cnn.layer_buffer[0].conv_buffer_size.rows = roundUpMul(cnn.conv_l1.ROI.rows, 2);
			cnn.layer_buffer[0].conv_buffer_size.size = cnn.layer_buffer[0].conv_buffer_size.cols * cnn.layer_buffer[0].conv_buffer_size.rows;

			int iBufferSize = cnn.layer_buffer[0].conv_buffer_size.size;
			cnn.layer_buffer[0].conv_buffer = Array_32f(iBufferSize, ALIGN_DEF);

			//pool buffer
			cnn.layer_buffer[0].pool_buffer_size.cols = cnn.layer_buffer[0].conv_buffer_size.cols;
			cnn.layer_buffer[0].pool_buffer_size.rows = roundUp(cnn.layer_buffer[0].conv_buffer_size.rows, 2);

			cnn.layer_buffer[0].pool_buffer_size.cols = roundUpMul(cnn.layer_buffer[0].pool_buffer_size.cols, 2 * REG_SIZE);
			cnn.layer_buffer[0].pool_buffer_size.size = cnn.layer_buffer[0].pool_buffer_size.cols * cnn.layer_buffer[0].pool_buffer_size.rows;

			iBufferSize = cnn.layer_buffer[0].pool_buffer_size.size;
			cnn.layer_buffer[0].pool_buffer = Array_32f(iBufferSize, ALIGN_DEF);

			//initial layer2
			cnn.conv_l2.ROI.cols = (cnn.conv_l1.ROI.cols >> 1) - (cnn.conv_l2.size.cols - 1);
			cnn.conv_l2.ROI.rows = (cnn.conv_l1.ROI.rows >> 1) - (cnn.conv_l2.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[1].conv_buffer_size.cols = roundUpMul(cnn.layer_buffer[1].map_count * cnn.conv_l2.ROI.cols, REG_SIZE);
			cnn.layer_buffer[1].conv_buffer_size.rows = roundUpMul(cnn.conv_l2.ROI.rows, 2);
			cnn.layer_buffer[1].conv_buffer_size.size = cnn.layer_buffer[1].conv_buffer_size.cols * cnn.layer_buffer[1].conv_buffer_size.rows;

			iBufferSize = cnn.layer_buffer[1].conv_buffer_size.size;
			cnn.layer_buffer[1].conv_buffer = Array_32f(iBufferSize, ALIGN_DEF);

			//pool buffer
			cnn.layer_buffer[1].pool_buffer_size.cols = cnn.layer_buffer[1].conv_buffer_size.cols;
			cnn.layer_buffer[1].pool_buffer_size.rows = roundUp(cnn.layer_buffer[1].conv_buffer_size.rows, 2);

			cnn.layer_buffer[1].pool_buffer_size.cols = roundUpMul(cnn.layer_buffer[1].pool_buffer_size.cols, 2 * REG_SIZE);
			cnn.layer_buffer[1].pool_buffer_size.size = cnn.layer_buffer[1].pool_buffer_size.cols * cnn.layer_buffer[1].pool_buffer_size.rows;

			iBufferSize = cnn.layer_buffer[1].pool_buffer_size.size;
			cnn.layer_buffer[1].pool_buffer = Array_32f(iBufferSize, ALIGN_DEF);

			//initial layer3
			cnn.conv_l3.ROI.cols = (cnn.conv_l2.ROI.cols >> 1) - (cnn.conv_l3.size.cols - 1);
			cnn.conv_l3.ROI.rows = (cnn.conv_l2.ROI.rows >> 1) - (cnn.conv_l3.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[2].conv_buffer_size.cols = roundUpMul(cnn.layer_buffer[2].map_count * cnn.conv_l3.ROI.cols, REG_SIZE);
			cnn.layer_buffer[2].conv_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.layer_buffer[2].conv_buffer_size.size = cnn.layer_buffer[2].conv_buffer_size.cols * cnn.layer_buffer[2].conv_buffer_size.rows;

			iBufferSize = cnn.layer_buffer[2].conv_buffer_size.size;
			cnn.layer_buffer[2].conv_buffer = Array_32f(iBufferSize, ALIGN_DEF);

			//pool buffer
			cnn.layer_buffer[2].pool_buffer_size.cols = roundUpMul(2 * cnn.conv_l3.ROI.cols, 2 * REG_SIZE);
			cnn.layer_buffer[2].pool_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.layer_buffer[2].pool_buffer_size.size = cnn.layer_buffer[2].pool_buffer_size.cols * cnn.layer_buffer[2].pool_buffer_size.rows;

			iBufferSize = cnn.layer_buffer[2].pool_buffer_size.size;
			cnn.layer_buffer[2].pool_buffer = Array_32f(iBufferSize, ALIGN_DEF);

			cnn.output_buffer_size.cols = cnn.conv_l3.ROI.cols;
			cnn.output_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.output_buffer_size.step = cnn.layer_buffer[2].pool_buffer_size.cols;
			cnn.output_buffer_size.size = cnn.output_buffer_size.rows * cnn.output_buffer_size.step;
		}
		void ConvNeuralNetwork_v2::Clear()
		{
			if (isEmpty()) return;

			cnn.min_image_size = Size(0, 0);
			cnn.max_image_size = Size(0, 0);

			//clear buffers
			cnn.layer_buffer.clear();
			cnn.hl_buffer.clear();

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
			cnn.snn_hl_bias.clear();

			cnn.snn_ol_weight.clear();
		}

		void ConvNeuralNetwork_v2::ResizeBuffers(const Size size)
		{
			cnn.input_buffer_size.cols = size.width;
			cnn.input_buffer_size.rows = size.height;

			//initial layer1
			cnn.conv_l1.ROI.cols = size.width - (cnn.conv_l1.size.cols - 1);
			cnn.conv_l1.ROI.rows = size.height - (cnn.conv_l1.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[0].conv_buffer_size.cols = roundUpMul(cnn.layer_buffer[0].map_count * cnn.conv_l1.ROI.cols, REG_SIZE);
			cnn.layer_buffer[0].conv_buffer_size.rows = roundUpMul(cnn.conv_l1.ROI.rows, 2);

			//pool buffer
			cnn.layer_buffer[0].pool_buffer_size.cols = cnn.layer_buffer[0].conv_buffer_size.cols;
			cnn.layer_buffer[0].pool_buffer_size.rows = roundUp(cnn.layer_buffer[0].conv_buffer_size.rows, 2);

			cnn.layer_buffer[0].pool_buffer_size.cols = roundUpMul(cnn.layer_buffer[0].pool_buffer_size.cols, 2 * REG_SIZE);
			cnn.layer_buffer[0].pool_buffer_size.size = cnn.layer_buffer[0].pool_buffer_size.cols * cnn.layer_buffer[0].pool_buffer_size.rows;

			//initial layer2
			cnn.conv_l2.ROI.cols = (cnn.conv_l1.ROI.cols >> 1) - (cnn.conv_l2.size.cols - 1);
			cnn.conv_l2.ROI.rows = (cnn.conv_l1.ROI.rows >> 1) - (cnn.conv_l2.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[1].conv_buffer_size.cols = roundUpMul(cnn.layer_buffer[1].map_count * cnn.conv_l2.ROI.cols, REG_SIZE);
			cnn.layer_buffer[1].conv_buffer_size.rows = roundUpMul(cnn.conv_l2.ROI.rows, 2);

			//pool buffer
			cnn.layer_buffer[1].pool_buffer_size.cols = cnn.layer_buffer[1].conv_buffer_size.cols;
			cnn.layer_buffer[1].pool_buffer_size.rows = roundUp(cnn.layer_buffer[1].conv_buffer_size.rows, 2);

			cnn.layer_buffer[1].pool_buffer_size.cols = roundUpMul(cnn.layer_buffer[1].pool_buffer_size.cols, 2 * REG_SIZE);

			//initial layer3
			cnn.conv_l3.ROI.cols = (cnn.conv_l2.ROI.cols >> 1) - (cnn.conv_l3.size.cols - 1);
			cnn.conv_l3.ROI.rows = (cnn.conv_l2.ROI.rows >> 1) - (cnn.conv_l3.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[2].conv_buffer_size.cols = roundUpMul(cnn.layer_buffer[2].map_count * cnn.conv_l3.ROI.cols, REG_SIZE);
			cnn.layer_buffer[2].conv_buffer_size.rows = cnn.conv_l3.ROI.rows;

			//pool buffer
			cnn.layer_buffer[2].pool_buffer_size.cols = roundUpMul(2 * cnn.conv_l3.ROI.cols, 2 * REG_SIZE);
			cnn.layer_buffer[2].pool_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.layer_buffer[2].pool_buffer_size.size = cnn.layer_buffer[2].pool_buffer_size.cols * cnn.layer_buffer[2].pool_buffer_size.rows;

			cnn.output_buffer_size.cols = cnn.conv_l3.ROI.cols;
			cnn.output_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.output_buffer_size.step = cnn.layer_buffer[2].pool_buffer_size.cols;
			cnn.output_buffer_size.size = cnn.output_buffer_size.rows * cnn.output_buffer_size.step;
		}

		void ConvNeuralNetwork_v2::Forward(Image_32f& response_map, Image_32f& image)
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

#ifdef CHECK_TEST
			if (cnn_ref == NULL) return;
			printf("\n	cnn_simd_v2: run single thread");
			printf("\n	cnn_simd_v2: image size = (%d, %d)\n", image.width, image.height);
			Legacy::ConvNeuralNetwork cnn_old(cnn_ref, image.width, image.height);
			SIMD::Image_32f r_map;
			cnn_old.Forward(r_map, image);
#endif

#ifdef PROFILE_CNN_SIMD
			printf("\n	cnn_simd_v2: run single thread");
			printf("\n	cnn_simd_v2: image size = (%d, %d)\n", image.width, image.height);
			Timer timer(1, true);
#endif

			cnnpp.conv_4x4(
						cnn.layer_buffer[0].conv_buffer(), 
						cnn.layer_buffer[0].conv_buffer_size.cols, 
						image.data,
						image.widthStep,
						cnn.input_buffer_size.rows, 
						cnn.conv_l1.kernels(),
						cnn.conv_l1.ROI.cols,
						cnn.conv_l1.ROI.rows,
						num_threads);

			//max_pool only
			cnnpp.max_tanh_tanh(
							cnn.layer_buffer[0].pool_buffer(),
							cnn.layer_buffer[0].pool_buffer_size.cols, 
							cnn.layer_buffer[0].conv_buffer(),
							cnn.layer_buffer[0].conv_buffer_size.cols,
							cnn.layer_buffer[0].conv_buffer_size.rows, 
							cnn.conv_bias[0](), cnn.subs_weight[0](),
							cnn.subs_bias[0](),
							&(cnn.af_scale),
							num_threads);

#ifdef CHECK_TEST
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				int count1 = 0;
				double d1 = 0;
				for (int y = 0; y < cnn_old.cnn.subs_neuron[0][i].row_count(); ++y)
				{
					for (int x = 0; x < cnn_old.cnn.subs_neuron[0][i].col_count(); ++x)
					{
						double d = cnn_old.cnn.subs_neuron[0][i][y][x] - cnn.layer_buffer[0].pool_buffer[y * cnn.layer_buffer[0].pool_buffer_size.cols + cnn.layer_buffer[0].map_count * x + i];
						if (abs(d) > 1.E-4)
						{
							d1 += abs(d);
							count1++;
						}
					}
				}
				printf("	d1 = %f, count1 = %d\n", d1, count1);
				if (abs(d1) > 0.) system("pause");
			}
#endif

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd_v2: run_L1 = %7.3f ms (conv_l1, tanh_avr_tanh)\n", timer.get(1000));
			timer.start();
#endif

			cnnpp.conv_3x3(
						cnn.layer_buffer[1].conv_buffer(),
						cnn.layer_buffer[1].conv_buffer_size.cols, 
						cnn.layer_buffer[0].pool_buffer(),
						cnn.layer_buffer[0].pool_buffer_size.cols, 
						cnn.layer_buffer[0].pool_buffer_size.rows, 
						cnn.conv_l2.kernels(),
						cnn.conv_l2.ROI.cols, 
						cnn.conv_l2.ROI.rows,
						num_threads);
			
			//max_pool only
			cnnpp.max_tanh_tanh(
							cnn.layer_buffer[1].pool_buffer(),
							cnn.layer_buffer[1].pool_buffer_size.cols, 
							cnn.layer_buffer[1].conv_buffer(),
							cnn.layer_buffer[1].conv_buffer_size.cols,
							cnn.layer_buffer[1].conv_buffer_size.rows, 
							cnn.conv_bias[1](),
							cnn.subs_weight[1](),
							cnn.subs_bias[1](),
							&(cnn.af_scale),
							num_threads);

#ifdef CHECK_TEST
			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				int count2 = 0;
				double d2 = 0;
				for (int y = 0; y < cnn_old.cnn.subs_neuron[1][i].row_count(); ++y)
				{
					for (int x = 0; x < cnn_old.cnn.subs_neuron[1][i].col_count(); ++x)
					{
						double d = cnn_old.cnn.subs_neuron[1][i][y][x] - cnn.layer_buffer[1].pool_buffer[y * cnn.layer_buffer[1].pool_buffer_size.cols + cnn.layer_buffer[1].map_count * x + i];
						if (abs(d) > 1.E-4)
						{
							d2 += abs(d);
							count2++;
						}
					}
				}
				printf("	d2 = %f, count2 = %d\n", d2, count2);
				if (abs(d2) > 0.) system("pause");
			}
#endif

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd_v2: run_L2 = %7.3f ms (sum, conv_l2, tanh_avr_tanh)\n", timer.get(1000));
			timer.start();
#endif

			//6x5 only
			cnnpp.conv_6x5(
							cnn.layer_buffer[2].conv_buffer(),
							cnn.layer_buffer[2].conv_buffer_size.cols, 
							cnn.layer_buffer[1].pool_buffer(),
							cnn.layer_buffer[1].pool_buffer_size.cols, 
							cnn.layer_buffer[1].pool_buffer_size.rows, 
							cnn.conv_l3.kernels(),
							cnn.conv_l3.ROI.cols, 
							cnn.conv_l3.ROI.rows,
							num_threads);
			
			//full_connect no support
			cnnpp.tanh_tanh_2tanh(
						cnn.layer_buffer[2].pool_buffer(),
						cnn.layer_buffer[2].pool_buffer_size.cols,
						cnn.layer_buffer[2].conv_buffer(),
						cnn.layer_buffer[2].conv_buffer_size.cols,
						cnn.layer_buffer[2].conv_buffer_size.rows,
						cnn.conv_bias[2](), cnn.subs_weight[2](),
						cnn.subs_bias[2](),
						&(cnn.af_scale), 
						cnn.snn_hl_weight(),
						cnn.snn_hl_bias(),
						cnn.snn_hl_weight(8),
						cnn.snn_hl_bias(8),
						cnn.snn_ol_weight(),
						cnn.snn_ol_weight(8),
						num_threads);
			
			cnnpp.tanh_tanh_2tanh(
						cnn.layer_buffer[2].pool_buffer(1),
						cnn.layer_buffer[2].pool_buffer_size.cols,
						cnn.layer_buffer[2].conv_buffer(cnn.layer_buffer[2].conv_buffer_size.cols / 2),
						cnn.layer_buffer[2].conv_buffer_size.cols,
						cnn.layer_buffer[2].conv_buffer_size.rows,
						cnn.conv_bias[2](8),
						cnn.subs_weight[2](8),
						cnn.subs_bias[2](8),
						&(cnn.af_scale), 
						cnn.snn_hl_weight(16),
						cnn.snn_hl_bias(16),
						cnn.snn_hl_weight(16 + 8),
						cnn.snn_hl_bias(16 + 8),
						cnn.snn_ol_weight(16),
						cnn.snn_ol_weight(16 + 8),
						num_threads);

			cnnpp.tanh(
					cnn.layer_buffer[2].pool_buffer(),
					cnn.layer_buffer[2].pool_buffer_size.cols, 
					cnn.layer_buffer[2].pool_buffer(),
					cnn.layer_buffer[2].pool_buffer_size.cols,
					cnn.layer_buffer[2].pool_buffer_size.rows,
					&(cnn.snn_ol_bias), 
					&(cnn.af_scale),
					num_threads);

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd_v2: run_L3 = %7.3f ms (sum, conv_l3, tanh_tanh_2tanh_tanh)\n", timer.get(1000));
#endif

#ifdef CHECK_TEST
			int count4 = 0;
			double d4 = 0;
			for (int y = 0; y < cnn_old.snn.output_neuron.row_count(); ++y)
			{
				for (int x = 0; x < cnn_old.snn.output_neuron.col_count(); ++x)
				{
					double d = cnn_old.snn.output_neuron[y][x][0] - cnn.layer_buffer[2].pool_buffer[y * cnn.output_buffer_size.step + x];
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

			if (response_map.isEmpty())
			{
				response_map.width = cnn.output_buffer_size.cols;
				response_map.height = cnn.output_buffer_size.rows;
				response_map.widthStep = cnn.output_buffer_size.step;
				response_map.data = cnn.layer_buffer[2].pool_buffer();
				response_map.sharingData = true;
			}
			else
			{
				response_map.width = cnn.output_buffer_size.cols;
				response_map.height = cnn.output_buffer_size.rows;
				response_map.copyData(cnn.output_buffer_size.cols, cnn.output_buffer_size.rows, cnn.layer_buffer[2].pool_buffer(), cnn.output_buffer_size.step);
			}
		}

		Size ConvNeuralNetwork_v2::getOutputImgSize(const Size size)
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