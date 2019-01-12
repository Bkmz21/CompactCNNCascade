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


#include "cnn_simd_v2_cntk.h"
#include <fstream>
#include <sstream>
#include <iterator>
#include <immintrin.h>

#ifdef USE_OMP
#	include <omp.h>
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{
#if defined(USE_AVX) && defined(USE_CNTK_MODELS)

	namespace SIMD
	{
		void ConvNeuralNetwork_v2::Init(std::string file_name, int index_output, void* hGrd)
		{
			std::stringstream data_bin;
			if (file_name.size() < 255)
			{
				std::fstream file_bin;
				file_bin.open(file_name.c_str(), std::fstream::binary | std::fstream::in);

				if (!file_bin.is_open())
				{
					printf("[SIMD::CNN_v2] Configuration file not found!\n");
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
			FB_READ(data_bin, cnn.af_scale);

			cnn.conv_bias.resize(cnn.layer_count);
			cnn.leakyReLU_w1.resize(cnn.layer_count);
			cnn.leakyReLU_w2.resize(cnn.layer_count);
			cnn.bn_weight.resize(cnn.layer_count);
			cnn.bn_bias.resize(cnn.layer_count);
			for (int i = 0; i < cnn.layer_count; ++i)
			{
				int L = MAX(1, REG_SIZE / cnn.layer_buffer[i].map_count);
				iBufferSize = L * cnn.layer_buffer[i].map_count;
				cnn.conv_bias[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.leakyReLU_w1[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.leakyReLU_w2[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.bn_weight[i] = Array_32f(iBufferSize, ALIGN_DEF);
				cnn.bn_bias[i] = Array_32f(iBufferSize, ALIGN_DEF);

				if (i < 2)
				{
					for (int j = 0; j < cnn.layer_buffer[i].map_count; ++j)
					{
						int p = j;
						if (i == 1)
						{
							if (j == 2 || j == 3) p = j + 2;
							if (j == 4 || j == 5) p = j + 2;
							if (j == 6 || j == 7) p = j - 4;
						}

						FB_READ(data_bin, cnn.conv_bias[i][p]);
						FB_READ(data_bin, cnn.leakyReLU_w1[i][p]);
						FB_READ(data_bin, cnn.leakyReLU_w2[i][p]);
						FB_READ(data_bin, cnn.bn_weight[i][p]);
						FB_READ(data_bin, cnn.bn_bias[i][p]);

						for (int k = 1; k < L; ++k)
						{
							cnn.conv_bias[i][p + k * cnn.layer_buffer[i].map_count] = cnn.conv_bias[i][p];
							cnn.leakyReLU_w1[i][p + k * cnn.layer_buffer[i].map_count] = cnn.leakyReLU_w1[i][p];
							cnn.leakyReLU_w2[i][p + k * cnn.layer_buffer[i].map_count] = cnn.leakyReLU_w2[i][p];
							cnn.bn_weight[i][p + k * cnn.layer_buffer[i].map_count] = cnn.bn_weight[i][p];
							cnn.bn_bias[i][p + k * cnn.layer_buffer[i].map_count] = cnn.bn_bias[i][p];
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
							cnn.leakyReLU_w1[i][k / 2] = kernel_val;
						else
							cnn.leakyReLU_w1[i][(k + cnn.layer_buffer[i].map_count) / 2] = kernel_val;

						kernel_val = 0.f;
						FB_READ(data_bin, kernel_val);
						if (k % 2 == 0)
							cnn.leakyReLU_w2[i][k / 2] = kernel_val;
						else
							cnn.leakyReLU_w2[i][(k + cnn.layer_buffer[i].map_count) / 2] = kernel_val;

						kernel_val = 0.f;
						FB_READ(data_bin, kernel_val);
						if (k % 2 == 0)
							cnn.bn_weight[i][k / 2] = kernel_val;
						else
							cnn.bn_weight[i][(k + cnn.layer_buffer[i].map_count) / 2] = kernel_val;

						kernel_val = 0.f;
						FB_READ(data_bin, kernel_val);
						if (k % 2 == 0)
							cnn.bn_bias[i][k / 2] = kernel_val;
						else
							cnn.bn_bias[i][(k + cnn.layer_buffer[i].map_count) / 2] = kernel_val;
					}
				}
			}

			//simple nn weight
			FB_READ(data_bin, cnn.snn_full_connect);
			FB_READ(data_bin, cnn.snn_hl_size);
			FB_READ(data_bin, cnn.snn_connect_count);
			FB_READ(data_bin, cnn.hl_scale);

			if (!cnn.max_pool ||
				cnn.snn_full_connect ||
				cnn.layer_count != 3 ||
				cnn.conv_l1.size.size != 16 ||
				cnn.conv_l2.size.size != 9  ||
				cnn.conv_l3.size.size != 20 ||
				index_output < 0)
			{
				printf("[SIMD::CNN_v2] This configuration cnn models is not supported!\n");
				Clear();
				return;
			}

			//read snn_hl_weight
			cnn.snn_hl_weight.resize(cnn.hl_scale);
			for (int k = 0; k < cnn.hl_scale; ++k)
			{
				iBufferSize = cnn.snn_connect_count * roundUpMul(cnn.snn_hl_size / cnn.hl_scale, 2 * REG_SIZE);
				cnn.snn_hl_weight[k] = Array_32f(iBufferSize, ALIGN_DEF);
			}
			cnn.snn_hl_weight_ref = Array_32f_ref(cnn.snn_hl_weight.data(), cnn.snn_hl_weight.size());

			for (int p = 0; p < roundUpMul(cnn.snn_hl_size / cnn.hl_scale, REG_SIZE); ++p)
			{				
				for (int k = 0; k < cnn.hl_scale; ++k)
				{
					for (int i = 0; i < cnn.snn_connect_count; ++i)
					{
						float kernel_val = 0.f;
						if (p < cnn.snn_hl_size / cnn.hl_scale)
						{
							FB_READ(data_bin, kernel_val);
						}
						
						if (i % 2 == 0)
						{
							cnn.snn_hl_weight[k][(i >> 1) + (p >> 2) * (cnn.snn_connect_count >> 1) + (p % 4) * 2 * REG_SIZE] = kernel_val;
						}
						else
						{
							cnn.snn_hl_weight[k][(i >> 1) + (p >> 2) * (cnn.snn_connect_count >> 1) + REG_SIZE + (p % 4) * 2 * REG_SIZE] = kernel_val;
						}
					}
				}
			}

			//read snn_hl_bias
			cnn.snn_hl_bias.resize(cnn.hl_scale);
			for (int k = 0; k < cnn.hl_scale; ++k)
			{
				iBufferSize = roundUpMul(cnn.snn_hl_size / cnn.hl_scale, REG_SIZE);
				cnn.snn_hl_bias[k] = Array_32f(iBufferSize, ALIGN_DEF);
			}
			cnn.snn_hl_bias_ref = Array_32f_ref(cnn.snn_hl_bias.data(), cnn.snn_hl_bias.size());

			for (int p = 0; p < roundUpMul(cnn.snn_hl_size / cnn.hl_scale, REG_SIZE); ++p)
			{
				for (int k = 0; k < cnn.hl_scale; ++k)
				{
					float kernel_val = 0.f;
					if (p < cnn.snn_hl_size / cnn.hl_scale)
					{
						FB_READ(data_bin, kernel_val);
					}

					int i = -1;
					if (p == 0) i = 0;  if (p == 1) i = 2;  if (p == 2) i = 8;   if (p == 3) i = 10;
					if (p == 4) i = 1;  if (p == 5) i = 3;  if (p == 6) i = 9;   if (p == 7) i = 11;
					if (p == 8) i = 4;  if (p == 9) i = 6;  if (p == 10) i = 12; if (p == 11) i = 14;
					if (p == 12) i = 5; if (p == 13) i = 7; if (p == 14) i = 13; if (p == 15) i = 15;

					cnn.snn_hl_bias[k][i] = kernel_val;
				}
			}

			//read snn_hl_tanh_w
			iBufferSize = roundUpMul(cnn.snn_hl_size / cnn.hl_scale, REG_SIZE);
			cnn.snn_hl_tanh_w = Array_32f(iBufferSize, ALIGN_DEF);
			
			for (int p = 0; p < roundUpMul(cnn.snn_hl_size / cnn.hl_scale, REG_SIZE); ++p)
			{
				float kernel_val = 0.f;
				if (p < cnn.snn_hl_size / cnn.hl_scale)
				{
					FB_READ(data_bin, kernel_val);
				}

				int i = -1;
				if (p == 0) i = 0;  if (p == 1) i = 2;  if (p == 2) i = 8;   if (p == 3) i = 10;
				if (p == 4) i = 1;  if (p == 5) i = 3;  if (p == 6) i = 9;   if (p == 7) i = 11;
				if (p == 8) i = 4;  if (p == 9) i = 6;  if (p == 10) i = 12; if (p == 11) i = 14;
				if (p == 12) i = 5; if (p == 13) i = 7; if (p == 14) i = 13; if (p == 15) i = 15;

				cnn.snn_hl_tanh_w[i] = kernel_val;
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
			cnn.snn_ol_weight_ref.resize(cnn.snn_ol_neuron_count);
			for (int n = 0; n < cnn.snn_ol_neuron_count; ++n)
			{
				cnn.snn_ol_weight[n].resize(cnn.hl_scale);
				for (int k = 0; k < cnn.hl_scale; ++k)
				{
					iBufferSize = cnn.snn_connect_count * roundUpMul(cnn.snn_hl_size / cnn.hl_scale, REG_SIZE);
					cnn.snn_ol_weight[n][k] = Array_32f(iBufferSize, ALIGN_DEF);
				}
				cnn.snn_ol_weight_ref[n] = Array_32f_ref(cnn.snn_ol_weight[n].data(), cnn.snn_ol_weight[n].size());

				for (int k = 0; k < cnn.hl_scale; ++k)
				{
					for (int p = 0; p < roundUpMul(cnn.snn_hl_size / cnn.hl_scale, REG_SIZE); ++p)
					{
						float kernel_val = 0.f;
						if (p < cnn.snn_hl_size / cnn.hl_scale)
						{
							FB_READ(data_bin, kernel_val);
						}

						int i = -1;
						if (p == 0) i = 0;  if (p == 1) i = 2;  if (p == 2) i = 8;   if (p == 3) i = 10;
						if (p == 4) i = 1;  if (p == 5) i = 3;  if (p == 6) i = 9;   if (p == 7) i = 11;
						if (p == 8) i = 4;  if (p == 9) i = 6;  if (p == 10) i = 12; if (p == 11) i = 14;
						if (p == 12) i = 5; if (p == 13) i = 7; if (p == 14) i = 13; if (p == 15) i = 15;

						cnn.snn_ol_weight[n][k][i] = kernel_val;
					}
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
			cnn.layer_buffer[0].size.cols = roundUp(cnn.layer_buffer[0].map_count * cnn.conv_l1.ROI.cols, 2);
			cnn.layer_buffer[0].size.rows = roundUp(cnn.conv_l1.ROI.rows, 2);

			cnn.layer_buffer[0].size.cols = roundUpMul(cnn.layer_buffer[0].size.cols, REG_SIZE);
			cnn.layer_buffer[0].size.rows = 2 * cnn.layer_buffer[0].size.rows;
			cnn.layer_buffer[0].size.size = cnn.layer_buffer[0].size.cols * cnn.layer_buffer[0].size.rows;

			int iBufferSize = cnn.layer_buffer[0].size.size;
			cnn.layer_buffer[0].buffer = Array_32f(iBufferSize, ALIGN_DEF);

			//initial layer2
			cnn.conv_l2.ROI.cols = (cnn.conv_l1.ROI.cols >> 1) - (cnn.conv_l2.size.cols - 1);
			cnn.conv_l2.ROI.rows = (cnn.conv_l1.ROI.rows >> 1) - (cnn.conv_l2.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[1].size.cols = roundUp(cnn.layer_buffer[1].map_count * cnn.conv_l2.ROI.cols, 2);
			cnn.layer_buffer[1].size.rows = roundUp(cnn.conv_l2.ROI.rows, 2);

			cnn.layer_buffer[1].size.cols = roundUpMul(cnn.layer_buffer[1].size.cols, 2 * REG_SIZE);
			cnn.layer_buffer[1].size.rows = 2 * cnn.layer_buffer[1].size.rows;
			cnn.layer_buffer[1].size.size = cnn.layer_buffer[1].size.cols * cnn.layer_buffer[1].size.rows;

			iBufferSize = cnn.layer_buffer[1].size.size;
			cnn.layer_buffer[1].buffer = Array_32f(iBufferSize, ALIGN_DEF);

			//initial layer3
			cnn.conv_l3.ROI.cols = (cnn.conv_l2.ROI.cols >> 1) - (cnn.conv_l3.size.cols - 1);
			cnn.conv_l3.ROI.rows = (cnn.conv_l2.ROI.rows >> 1) - (cnn.conv_l3.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[2].size.cols = cnn.layer_buffer[2].map_count * cnn.conv_l3.ROI.cols;
			cnn.layer_buffer[2].size.rows = cnn.conv_l3.ROI.rows;

			cnn.layer_buffer[2].size.cols = roundUpMul(cnn.layer_buffer[2].size.cols, 4 * REG_SIZE);
			cnn.layer_buffer[2].size.rows = cnn.layer_buffer[2].size.rows;
			cnn.layer_buffer[2].size.size = cnn.layer_buffer[2].size.cols * cnn.layer_buffer[2].size.rows;

			iBufferSize = cnn.layer_buffer[2].size.size;
			cnn.layer_buffer[2].buffer = Array_32f(iBufferSize, ALIGN_DEF);

			cnn.output_buffer_size.cols = cnn.conv_l3.ROI.cols;
			cnn.output_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.output_buffer_size.step = cnn.layer_buffer[2].size.cols;
			cnn.output_buffer_size.size = cnn.output_buffer_size.rows * cnn.output_buffer_size.step;
		}
		void ConvNeuralNetwork_v2::Clear()
		{
			if (isEmpty()) return;

			cnn.min_image_size = Size(0, 0);
			cnn.max_image_size = Size(0, 0);

			//clear buffers
			cnn.layer_buffer.clear();

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
			cnn.snn_hl_weight_ref.clear();
			cnn.snn_hl_bias.clear();
			cnn.snn_hl_bias_ref.clear();

			cnn.snn_hl_tanh_w.clear();
			cnn.snn_hl_bn_weight.clear();
			cnn.snn_hl_bn_bias.clear();

			cnn.snn_ol_weight.clear();
			cnn.snn_ol_weight_ref.clear();
			cnn.snn_ol_bias.clear();
		}

		void ConvNeuralNetwork_v2::ResizeBuffers(const Size size)
		{
			cnn.input_buffer_size.cols = size.width;
			cnn.input_buffer_size.rows = size.height;

			//initial layer1
			cnn.conv_l1.ROI.cols = size.width - (cnn.conv_l1.size.cols - 1);
			cnn.conv_l1.ROI.rows = size.height - (cnn.conv_l1.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[0].size.cols = roundUp(cnn.layer_buffer[0].map_count * cnn.conv_l1.ROI.cols, 2);
			cnn.layer_buffer[0].size.rows = roundUp(cnn.conv_l1.ROI.rows, 2);

			cnn.layer_buffer[0].size.cols = roundUpMul(cnn.layer_buffer[0].size.cols, REG_SIZE);
			cnn.layer_buffer[0].size.rows = 2 * cnn.layer_buffer[0].size.rows;
			cnn.layer_buffer[0].size.size = cnn.layer_buffer[0].size.cols * cnn.layer_buffer[0].size.rows;

			//initial layer2
			cnn.conv_l2.ROI.cols = (cnn.conv_l1.ROI.cols >> 1) - (cnn.conv_l2.size.cols - 1);
			cnn.conv_l2.ROI.rows = (cnn.conv_l1.ROI.rows >> 1) - (cnn.conv_l2.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[1].size.cols = roundUp(cnn.layer_buffer[1].map_count * cnn.conv_l2.ROI.cols, 2);
			cnn.layer_buffer[1].size.rows = roundUp(cnn.conv_l2.ROI.rows, 2);

			cnn.layer_buffer[1].size.cols = roundUpMul(cnn.layer_buffer[1].size.cols, 2 * REG_SIZE);
			cnn.layer_buffer[1].size.rows = 2 * cnn.layer_buffer[1].size.rows;
			cnn.layer_buffer[1].size.size = cnn.layer_buffer[1].size.cols * cnn.layer_buffer[1].size.rows;

			//initial layer3
			cnn.conv_l3.ROI.cols = (cnn.conv_l2.ROI.cols >> 1) - (cnn.conv_l3.size.cols - 1);
			cnn.conv_l3.ROI.rows = (cnn.conv_l2.ROI.rows >> 1) - (cnn.conv_l3.size.rows - 1);

			//conv buffer
			cnn.layer_buffer[2].size.cols = cnn.layer_buffer[2].map_count * cnn.conv_l3.ROI.cols;
			cnn.layer_buffer[2].size.rows = cnn.conv_l3.ROI.rows;

			cnn.layer_buffer[2].size.cols = roundUpMul(cnn.layer_buffer[2].size.cols, 4 * REG_SIZE);
			cnn.layer_buffer[2].size.rows = cnn.layer_buffer[2].size.rows;
			cnn.layer_buffer[2].size.size = cnn.layer_buffer[2].size.cols * cnn.layer_buffer[2].size.rows;

			cnn.output_buffer_size.cols = cnn.conv_l3.ROI.cols;
			cnn.output_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.output_buffer_size.step = cnn.layer_buffer[2].size.cols;
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

#ifdef PROFILE_CNN_SIMD
			printf("\n	cnn_simd_v2: run single thread");
			printf("\n	cnn_simd_v2: image size = (%d, %d)\n", image.width, image.height);
			Timer timer(1, true);
#endif

			cnnpp.conv_4x4_lrelu_bn_max(
						cnn.layer_buffer[0].buffer(),
						cnn.layer_buffer[0].size.cols, 
						image.data,
						image.widthStep,
						cnn.input_buffer_size.rows, 
						cnn.conv_l1.kernels(),
						cnn.conv_bias[0](),
						cnn.leakyReLU_w1[0](), 
						cnn.leakyReLU_w2[0](), 
						cnn.bn_weight[0](),
						cnn.bn_bias[0](),
						cnn.conv_l1.ROI.cols,
						cnn.conv_l1.ROI.rows,
						num_threads);

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd_v2: run_L1 = %7.3f ms (conv_l1, tanh_avr_tanh)\n", timer.get(1000));
			timer.start();
#endif

			cnnpp.conv_3x3_lrelu_bn_max(
						cnn.layer_buffer[1].buffer(),
						cnn.layer_buffer[1].size.cols, 
						cnn.layer_buffer[0].buffer(),
						cnn.layer_buffer[0].size.cols,
						cnn.layer_buffer[0].size.rows,
						cnn.conv_l2.kernels(), 
						cnn.conv_bias[1](),
						cnn.leakyReLU_w1[1](), 
						cnn.leakyReLU_w2[1](), 
						cnn.bn_weight[1](),
						cnn.bn_bias[1](),
						cnn.conv_l2.ROI.cols, 
						cnn.conv_l2.ROI.rows,
						num_threads);

#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd_v2: run_L2 = %7.3f ms (sum, conv_l2, tanh_avr_tanh)\n", timer.get(1000));
			timer.start();
#endif

			//5x4 only
			cnnpp.conv_5x4_lrelu_bn(
						cnn.layer_buffer[2].buffer(), 
						cnn.layer_buffer[2].size.cols, 
						cnn.layer_buffer[1].buffer(),
						cnn.layer_buffer[1].size.cols,
						cnn.layer_buffer[1].size.rows,
						cnn.conv_l3.kernels(), 
						cnn.conv_bias[2](),
						cnn.leakyReLU_w1[2](),
						cnn.leakyReLU_w2[2](),
						cnn.bn_weight[2](),
						cnn.bn_bias[2](),
						cnn.conv_l3.ROI.cols, 
						cnn.conv_l3.ROI.rows,
						num_threads);
			
			//full_connect no support
			cnnpp.mulCN_add_tanhW_add(
						cnn.layer_buffer[2].buffer(),
						cnn.layer_buffer[2].size.cols,
						cnn.layer_buffer[2].buffer(),
						cnn.layer_buffer[2].size.cols,
						cnn.layer_buffer[2].size.rows,
						cnn.snn_hl_weight_ref(),
						cnn.snn_hl_bias_ref(),
						cnn.snn_hl_tanh_w(),
						cnn.snn_hl_bn_weight(),
						cnn.snn_hl_bn_bias(),
						cnn.snn_ol_weight_ref[cnn.index_output](),
						cnn.conv_l3.ROI.cols,
						cnn.conv_l3.ROI.rows,
						num_threads);

			cnnpp.tanhW(
						cnn.layer_buffer[2].buffer(),
						cnn.layer_buffer[2].size.cols,
						cnn.layer_buffer[2].buffer(),
						cnn.layer_buffer[2].size.cols,
						cnn.layer_buffer[2].size.rows,
						&(cnn.snn_ol_bias[cnn.index_output]),
						&(cnn.snn_ol_tanh_w),
						&(cnn.af_scale),
						cnn.conv_l3.ROI.cols,
						cnn.conv_l3.ROI.rows,
						num_threads);

#ifdef USE_AVX
			_mm256_zeroupper();
#endif
			
#ifdef PROFILE_CNN_SIMD
			printf("	cnn_simd_v2: run_L3 = %7.3f ms (sum, conv_l3, tanh_tanh_2tanh_tanh)\n", timer.get(1000));
#endif

			if (response_map.isEmpty())
			{
				response_map.width = cnn.output_buffer_size.cols;
				response_map.height = cnn.output_buffer_size.rows;
				response_map.widthStep = cnn.output_buffer_size.step;
				response_map.data = cnn.layer_buffer[2].buffer();
				response_map.sharingData = true;
			}
			else
			{
				response_map.width = cnn.output_buffer_size.cols;
				response_map.height = cnn.output_buffer_size.rows;
				response_map.copyData(cnn.output_buffer_size.cols, cnn.output_buffer_size.rows, cnn.layer_buffer[2].buffer(), cnn.output_buffer_size.step);
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