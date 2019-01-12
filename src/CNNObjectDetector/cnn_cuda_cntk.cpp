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


#include "cnn_cuda_cntk.h"
#include <fstream>
#include <sstream>


//================================================================================================================================================


namespace NeuralNetworksLib
{
#if defined(USE_CUDA) && defined(USE_CNTK_MODELS)

	namespace CUDA
	{
		#define FB_READ(FNAME, VAL) FNAME.read((char*)&(VAL), sizeof(VAL))
		#define FB_WRITE(FNAME, VAL) FNAME.write((const char*)&(VAL), sizeof(VAL))

		void ConvNeuralNetwork::Init(std::string file_name, int index_output, void* hGrd)
		{
			std::stringstream data_bin;
			if (file_name.size() < 255)
			{
				std::fstream data_bin;
				data_bin.open(file_name.c_str(), std::fstream::binary | std::fstream::in);

				if (!data_bin.is_open())
				{
					printf("[CUDA::CNN] Configuration file not found!\n");
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

			if (format_version < 1.0f || format_version > 1.1f)
			{
				printf("[CUDA::CNN] Configuration file format is not supported!\n");
				return;
			}

			block_size = Size(32, 32);

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
				cnn.layer_buffer[i].buffer.resize(cnn.layer_buffer[i].map_count);
			}

			//initial weight
			//conv kernels l1
			int kernel_width = 0;
			int kernel_height = 0;
			FB_READ(data_bin, kernel_width);
			FB_READ(data_bin, kernel_height);

			cnn.conv_l1.size = Size2d(kernel_width, kernel_height);
			cnn.conv_l1.kernels.resize(cnn.layer_buffer[0].map_count);
			int iBufferSize = kernel_width * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				cnn.conv_l1.kernels[i] = SIMD::Array_32f(iBufferSize, ALIGN_SSE);
			}

			for (int k = 0; k < cnn.layer_buffer[0].map_count; ++k)
			{
				for (int i = 0; i < cnn.conv_l1.size.size; ++i)
				{
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);

					cnn.conv_l1.kernels[k][i] = kernel_val;
				}

				CNNPP::set_kernel_on_device(model_type, cnn.conv_l1.kernels[k](), cnn.conv_l1.size.size, 1, k);
			}

			//conv kernels l2
			FB_READ(data_bin, kernel_width);
			FB_READ(data_bin, kernel_height);

			cnn.conv_l2.size = Size2d(kernel_width, kernel_height);
			cnn.conv_l2.kernels.resize(cnn.layer_buffer[1].map_count);
			iBufferSize = kernel_width * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				cnn.conv_l2.kernels[i] = SIMD::Array_32f(iBufferSize, ALIGN_SSE);
			}

			for (int k = 0; k < cnn.layer_buffer[1].map_count; ++k)
			{
				for (int i = 0; i < cnn.conv_l2.size.size; ++i)
				{
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);

					cnn.conv_l2.kernels[k][i] = kernel_val;
				}

				CNNPP::set_kernel_on_device(model_type, cnn.conv_l2.kernels[k](), cnn.conv_l2.size.size, 2, k);
			}

			//conv kernels l3
			FB_READ(data_bin, kernel_width);
			FB_READ(data_bin, kernel_height);

			cnn.conv_l3.size = Size2d(kernel_width, kernel_height);
			cnn.conv_l3.kernels.resize(cnn.layer_buffer[2].map_count);
			iBufferSize = kernel_width * kernel_height;
			for (int i = 0; i < cnn.layer_buffer[2].map_count; ++i)
			{
				cnn.conv_l3.kernels[i] = SIMD::Array_32f(iBufferSize, ALIGN_SSE);
			}

			for (int k = 0; k < cnn.layer_buffer[2].map_count; ++k)
			{
				for (int i = 0; i < cnn.conv_l3.size.size; ++i)
				{
					float kernel_val = 0.f;
					FB_READ(data_bin, kernel_val);

					cnn.conv_l3.kernels[k][i] = kernel_val;
				}

				CNNPP::set_kernel_on_device(model_type, cnn.conv_l3.kernels[k](), cnn.conv_l3.size.size, 3, k);
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
				iBufferSize = cnn.layer_buffer[i].map_count;
				cnn.conv_bias[i] = SIMD::Array_32f(iBufferSize, ALIGN_SSE);
				cnn.leakyReLU_w1[i] = SIMD::Array_32f(iBufferSize, ALIGN_SSE);
				cnn.leakyReLU_w2[i] = SIMD::Array_32f(iBufferSize, ALIGN_SSE);
				cnn.bn_weight[i] = SIMD::Array_32f(iBufferSize, ALIGN_SSE);
				cnn.bn_bias[i] = SIMD::Array_32f(iBufferSize, ALIGN_SSE);

				for (int j = 0; j < cnn.layer_buffer[i].map_count; ++j)
				{
					FB_READ(data_bin, cnn.conv_bias[i][j]);
					FB_READ(data_bin, cnn.leakyReLU_w1[i][j]);
					FB_READ(data_bin, cnn.leakyReLU_w2[i][j]);
					FB_READ(data_bin, cnn.bn_weight[i][j]);
					FB_READ(data_bin, cnn.bn_bias[i][j]);

					CNNPP::set_conv_b_on_device(model_type, &(cnn.conv_bias[i][j]), i + 1, j);
					CNNPP::set_lrelu_w1_on_device(model_type, &(cnn.leakyReLU_w1[i][j]), i + 1, j);
					CNNPP::set_lrelu_w2_on_device(model_type, &(cnn.leakyReLU_w2[i][j]), i + 1, j);
					CNNPP::set_bn_w_on_device(model_type, &(cnn.bn_weight[i][j]), i + 1, j);
					CNNPP::set_bn_b_on_device(model_type, &(cnn.bn_bias[i][j]), i + 1, j);
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
				cnn.conv_l2.size.size != 9 ||
				cnn.conv_l3.size.size != 20)
			{
				printf("[CUDA::CNN] This configuration cnn model_types is not supported!\n");
				Clear();
				return;
			}

			cnn.snn_hl_weight.resize(cnn.snn_hl_size);
			for (int i = 0; i < cnn.snn_hl_size; ++i)
			{
				cnn.snn_hl_weight[i] = SIMD::Array_32f(cnn.snn_connect_count, ALIGN_SSE);
				for (int j = 0; j < cnn.snn_connect_count; ++j)
				{
					FB_READ(data_bin, cnn.snn_hl_weight[i][j]);
				}
				CNNPP::set_hl_w_on_device(model_type, cnn.snn_hl_weight[i](), i);
			}

			cnn.snn_hl_bias = SIMD::Array_32f(cnn.snn_hl_size, ALIGN_SSE);
			for (int i = 0; i < cnn.snn_hl_size; ++i)
			{
				FB_READ(data_bin, cnn.snn_hl_bias[i]);
				CNNPP::set_hl_b_on_device(model_type, &(cnn.snn_hl_bias[i]), i);
			}

			cnn.snn_hl_tanh_w = SIMD::Array_32f(cnn.snn_hl_size / cnn.hl_scale, ALIGN_SSE);
			for (int i = 0; i < (cnn.snn_hl_size / cnn.hl_scale); ++i)
			{
				FB_READ(data_bin, cnn.snn_hl_tanh_w[i]);
				CNNPP::set_hl_tanh_w_on_device(model_type, &(cnn.snn_hl_tanh_w[i]), i);
			}

			cnn.snn_hl_bn_weight = SIMD::Array_32f(cnn.hl_scale, ALIGN_SSE);
			cnn.snn_hl_bn_bias = SIMD::Array_32f(cnn.hl_scale, ALIGN_SSE);
			for (int i = 0; i < cnn.hl_scale; ++i)
			{
				FB_READ(data_bin, cnn.snn_hl_bn_weight[i]);
				FB_READ(data_bin, cnn.snn_hl_bn_bias[i]);
	
				CNNPP::set_hl_bn_w_on_device(model_type, &(cnn.snn_hl_bn_weight[i]), i);
				CNNPP::set_hl_bn_b_on_device(model_type, &(cnn.snn_hl_bn_bias[i]), i);
			}

			FB_READ(data_bin, cnn.snn_ol_neuron_count);
			cnn.snn_ol_weight.resize(cnn.snn_ol_neuron_count);
			for (int i = 0; i < cnn.snn_ol_neuron_count; ++i)
			{
				cnn.snn_ol_weight[i] = SIMD::Array_32f(cnn.snn_hl_size, ALIGN_SSE);
				for (int j = 0; j < cnn.snn_hl_size; ++j)
				{
					FB_READ(data_bin, cnn.snn_ol_weight[i][j]);
				}

				CNNPP::set_ol_w_on_device(model_type, cnn.snn_ol_weight[i](), i);
			}

			cnn.snn_ol_bias = SIMD::Array_32f(cnn.snn_ol_neuron_count, ALIGN_SSE);
			for (int i = 0; i < cnn.snn_ol_neuron_count; ++i)
			{
				FB_READ(data_bin, cnn.snn_ol_bias[i]);
				CNNPP::set_ol_b_on_device(model_type, &(cnn.snn_ol_bias[i]), i);
			}

			FB_READ(data_bin, cnn.snn_ol_tanh_w);
			CNNPP::set_ol_tanh_w_on_device(model_type, &cnn.snn_ol_tanh_w);

			cnn.index_output = MIN(abs(index_output), cnn.snn_ol_neuron_count - 1);
			float af_scale = cnn.index_output == 0 ? -cnn.af_scale : cnn.af_scale;
			CNNPP::set_scale_on_device(model_type, &af_scale, 1);
			CNNPP::set_scale_on_device(model_type, &af_scale, 2);
			CNNPP::set_scale_on_device(model_type, &af_scale, 3);
			CNNPP::set_scale_on_device(model_type, &af_scale, 4);

			//init device
			CNNPP::set_texture_cache_params(model_type);
			CNNPP::set_cache_device_params(model_type);
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

			//pool buffer
			cnn.layer_buffer[0].size.cols = addRoundUpMul(cnn.conv_l1.ROI.cols, block_size.width) >> 1; // div on 2
			cnn.layer_buffer[0].size.rows = addRoundUpMul(cnn.conv_l1.ROI.rows, block_size.height) >> 1; // div on 2
			cnn.layer_buffer[0].size.size = cnn.layer_buffer[0].size.cols * cnn.layer_buffer[0].size.rows;

			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				cnn.layer_buffer[0].buffer[i] = Image_32f(cnn.layer_buffer[0].size.cols, cnn.layer_buffer[0].size.rows, 1, true, ALIGN_DEF);
			}

			//initial layer2
			cnn.conv_l2.ROI.cols = (cnn.conv_l1.ROI.cols >> 1) - (cnn.conv_l2.size.cols - 1);
			cnn.conv_l2.ROI.rows = (cnn.conv_l1.ROI.rows >> 1) - (cnn.conv_l2.size.rows - 1);

			//pool buffer
			cnn.layer_buffer[1].size.cols = addRoundUpMul(cnn.conv_l2.ROI.cols, block_size.width) >> 1; // div on 2
			cnn.layer_buffer[1].size.rows = addRoundUpMul(cnn.conv_l2.ROI.rows, block_size.height) >> 1; // div on 2
			cnn.layer_buffer[1].size.size = cnn.layer_buffer[1].size.cols * cnn.layer_buffer[1].size.rows;

			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				cnn.layer_buffer[1].buffer[i] = Image_32f(cnn.layer_buffer[1].size.cols, cnn.layer_buffer[1].size.rows, 1, true, ALIGN_DEF);
			}

			//initial layer3
			cnn.conv_l3.ROI.cols = (cnn.conv_l2.ROI.cols >> 1) - (cnn.conv_l3.size.cols - 1);
			cnn.conv_l3.ROI.rows = (cnn.conv_l2.ROI.rows >> 1) - (cnn.conv_l3.size.rows - 1);

			//pool buffer
			cnn.layer_buffer[2].size.cols = addRoundUpMul(cnn.conv_l3.ROI.cols, block_size.width);
			cnn.layer_buffer[2].size.rows = addRoundUpMul(cnn.conv_l3.ROI.rows, block_size.height);
			cnn.layer_buffer[2].size.size = cnn.layer_buffer[2].size.cols * cnn.layer_buffer[2].size.rows;

			for (int i = 0; i < cnn.layer_buffer[2].map_count; ++i)
			{
				cnn.layer_buffer[2].buffer[i] = Image_32f(cnn.layer_buffer[2].size.cols, cnn.layer_buffer[2].size.rows, 1, true, ALIGN_DEF);
			}

			cnn.output_buffer_size.cols = cnn.conv_l3.ROI.cols;
			cnn.output_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.output_buffer_size.step = cnn.layer_buffer[2].size.cols;
			cnn.output_buffer_size.size = cnn.output_buffer_size.rows * cnn.output_buffer_size.step;

			//bind texture on device
			CNNPP::bind_texture(model_type, cnn.layer_buffer[0].buffer.data(), 2, cnn.layer_buffer[0].map_count);
			CNNPP::bind_texture(model_type, cnn.layer_buffer[1].buffer.data(), 3, cnn.layer_buffer[1].map_count);
			CNNPP::bind_texture(model_type, cnn.layer_buffer[2].buffer.data(), 4, cnn.layer_buffer[2].map_count);

			//set dst surf
			CNNPP::set_dst_surf(model_type, cnn.layer_buffer[0].buffer.data(), 1, cnn.layer_buffer[0].map_count);
			CNNPP::set_dst_surf(model_type, cnn.layer_buffer[1].buffer.data(), 2, cnn.layer_buffer[1].map_count);
			CNNPP::set_dst_surf(model_type, cnn.layer_buffer[2].buffer.data(), 3, cnn.layer_buffer[2].map_count);
		}
		void ConvNeuralNetwork::Clear()
		{
			if (isEmpty()) return;

			block_size = Size(0, 0);

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

			cnn.snn_hl_bias.clear();
			cnn.snn_hl_tanh_w.clear();

			cnn.snn_hl_bn_weight.clear();
			cnn.snn_hl_bn_bias.clear();

			cnn.snn_ol_weight.clear();
			cnn.snn_ol_bias.clear();
		}

		void ConvNeuralNetwork::ResizeBuffers(const Size size)
		{
			cnn.input_buffer_size.cols = size.width;
			cnn.input_buffer_size.rows = size.height;

			//initial layer1
			cnn.conv_l1.ROI.cols = size.width - (cnn.conv_l1.size.cols - 1);
			cnn.conv_l1.ROI.rows = size.height - (cnn.conv_l1.size.rows - 1);

			//initial layer2
			cnn.conv_l2.ROI.cols = (cnn.conv_l1.ROI.cols >> 1) - (cnn.conv_l2.size.cols - 1);
			cnn.conv_l2.ROI.rows = (cnn.conv_l1.ROI.rows >> 1) - (cnn.conv_l2.size.rows - 1);

			//initial layer3
			cnn.conv_l3.ROI.cols = (cnn.conv_l2.ROI.cols >> 1) - (cnn.conv_l3.size.cols - 1);
			cnn.conv_l3.ROI.rows = (cnn.conv_l2.ROI.rows >> 1) - (cnn.conv_l3.size.rows - 1);

			cnn.output_buffer_size.cols = cnn.conv_l3.ROI.cols;
			cnn.output_buffer_size.rows = cnn.conv_l3.ROI.rows;
		}
		void ConvNeuralNetwork::Forward(CUDA::Image_32f_pinned* response_map, CUDA::Image_32f* image, cudaStream_t cuda_stream)
		{
			if (image->width != cnn.input_buffer_size.cols || image->height != cnn.input_buffer_size.rows)
			{
				if (image->width < cnn.min_image_size.width || image->height < cnn.min_image_size.height ||
					image->width > cnn.max_image_size.width || image->height > cnn.max_image_size.height)
				{
					response_map->width = 0;
					response_map->height = 0;
					return;
				}

				ResizeBuffers(image->getSize());
			}

			response_map->width = cnn.output_buffer_size.cols;
			response_map->height = cnn.output_buffer_size.rows;

#ifdef PROFILE_CNN_CUDA
			printf("\n	cnn_cuda: image size = (%d, %d)\n", image->width, image->height);
			Timer timer(true);
#endif

			CNNPP::bind_texture(model_type, image, 1, 1);
			CNNPP::run_L1(model_type, cnn.conv_l1.ROI, cuda_stream);
			CNNPP::unbind_texture(model_type, 1, 1);

#ifdef PROFILE_CNN_CUDA
			printf("	cnn_cuda: run_L1 = %7.3f ms\n", timer.get(1000));
			timer.start();
#endif

			CNNPP::run_L2(model_type, cnn.conv_l2.ROI, cuda_stream);

#ifdef PROFILE_CNN_CUDA
			printf("	cnn_cuda: run_L2 = %7.3f ms\n", timer.get(1000));
			timer.start();
#endif

			CNNPP::run_L3(model_type, cnn.conv_l3.ROI, cuda_stream);
	
#ifdef PROFILE_CNN_CUDA
			printf("	cnn_cuda: run_L3 = %7.3f ms\n", timer.get(1000));
			timer.start();
#endif

			CNNPP::set_dst_surf(model_type, (CUDA::Image_32f*)response_map, 4, 1);
			CNNPP::run_HL(model_type, cnn.conv_l3.ROI, cnn.index_output, cuda_stream);

#ifdef PROFILE_CNN_CUDA
			printf("	cnn_cuda: run_HL = %7.3f ms\n", timer.get(1000));
#endif
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