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


#include "cnn_cl.h"
#include <fstream>
#include <sstream>


//================================================================================================================================================


namespace NeuralNetworksLib
{
#if defined(USE_CL) && !defined(USE_CNTK_MODELS)

	namespace CL
	{
		void ConvNeuralNetwork::Init(std::string file_name, int, cl_device_id _device, cl_context _context, cl_command_queue _queue)
		{
			std::stringstream data_bin;
			if (file_name.size() < 255)
			{
				std::fstream data_bin;
				data_bin.open(file_name.c_str(), std::fstream::binary | std::fstream::in);

				if (!data_bin.is_open())
				{
					printf("[CL::CNN] Configuration file not found!\n");
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
				printf("[CL::CNN] Configuration file format is not supported!\n");
				return;
			}

			device = _device;
			context = _context;
			queue = _queue;
			block_size = Size(32, 32);

			CNNPP::create_cl_program(device, context);

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

				CNNPP::set_kernel_on_device(context, queue, cnn.conv_l1.kernels[k](), cnn.conv_l1.size.size, 1, k);
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

				CNNPP::set_kernel_on_device(context, queue, cnn.conv_l2.kernels[k](), cnn.conv_l2.size.size, 2, k);
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

				CNNPP::set_kernel_on_device(context, queue, cnn.conv_l3.kernels[k](), cnn.conv_l3.size.size, 3, k);
			}

			//conv nn weight
			cnn.conv_bias.resize(cnn.layer_count);
			cnn.subs_weight.resize(cnn.layer_count);
			cnn.subs_bias.resize(cnn.layer_count);

			FB_READ(data_bin, cnn.af_scale);
			CNNPP::set_scale_on_device(context, queue, &cnn.af_scale, 1);
			CNNPP::set_scale_on_device(context, queue, &cnn.af_scale, 2);
			CNNPP::set_scale_on_device(context, queue, &cnn.af_scale, 3);
			CNNPP::set_scale_on_device(context, queue, &cnn.af_scale, 4);

			for (int i = 0; i < cnn.layer_count; ++i)
			{
				iBufferSize = cnn.layer_buffer[i].map_count;
				cnn.conv_bias[i] = SIMD::Array_32f(iBufferSize, ALIGN_SSE);
				cnn.subs_weight[i] = SIMD::Array_32f(iBufferSize, ALIGN_SSE);
				cnn.subs_bias[i] = SIMD::Array_32f(iBufferSize, ALIGN_SSE);

				for (int j = 0; j < cnn.layer_buffer[i].map_count; ++j)
				{
					FB_READ(data_bin, cnn.conv_bias[i][j]);
					FB_READ(data_bin, cnn.subs_weight[i][j]);
					FB_READ(data_bin, cnn.subs_bias[i][j]);
				}

				CNNPP::set_conv_b_on_device(context, queue, cnn.conv_bias[i](), i + 1, cnn.layer_buffer[i].map_count);
				CNNPP::set_subs_w_on_device(context, queue, cnn.subs_weight[i](), i + 1, cnn.layer_buffer[i].map_count);
				CNNPP::set_subs_b_on_device(context, queue, cnn.subs_bias[i](), i + 1, cnn.layer_buffer[i].map_count);
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
				printf("[CL::CNN] This configuration cnn models is not supported!\n");
				Clear();
				return;
			}

			if (!cnn.snn_full_connect)
			{
				cnn.snn_hl_weight = SIMD::Array_32f(cnn.snn_hl_size, ALIGN_DEF);
				for (int i = 0; i < cnn.snn_hl_size; ++i)
				{
					FB_READ(data_bin, cnn.snn_hl_weight[i]);
				}
				CNNPP::set_hl_w_on_device(context, queue, cnn.snn_hl_weight(), cnn.snn_hl_size);
			}
			else { }

			cnn.snn_hl_bias = SIMD::Array_32f(cnn.snn_hl_size, ALIGN_DEF);
			for (int i = 0; i < cnn.snn_hl_size; ++i)
			{
				FB_READ(data_bin, cnn.snn_hl_bias[i]);
			}
			CNNPP::set_hl_b_on_device(context, queue, cnn.snn_hl_bias(), cnn.snn_hl_size);

			cnn.snn_ol_weight = SIMD::Array_32f(cnn.snn_hl_size, ALIGN_DEF);
			for (int i = 0; i < cnn.snn_hl_size; ++i)
			{
				FB_READ(data_bin, cnn.snn_ol_weight[i]);
			}
			CNNPP::set_ol_w_on_device(context, queue, cnn.snn_ol_weight(), cnn.snn_hl_size);

			FB_READ(data_bin, cnn.snn_ol_bias);
			CNNPP::set_ol_b_on_device(context, queue, &cnn.snn_ol_bias);

		}
		void ConvNeuralNetwork::AllocateMemory(Size size)
		{
			if (device == 0 || context == 0 || queue == 0)
			{
				return;
			}

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
				cnn.layer_buffer[0].buffer[i] = Image_32f(context, queue, cnn.layer_buffer[0].size.cols, cnn.layer_buffer[0].size.rows, 1, true, ALIGN_DEF);
				CNNPP::set_dst_surf(&cnn.layer_buffer[0].buffer[i], 1, i);
				CNNPP::set_src_surf(&cnn.layer_buffer[0].buffer[i], 1, i);
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
				cnn.layer_buffer[1].buffer[i] = Image_32f(context, queue, cnn.layer_buffer[1].size.cols, cnn.layer_buffer[1].size.rows, 1, true, ALIGN_DEF);
				CNNPP::set_dst_surf(&cnn.layer_buffer[1].buffer[i], 2, i);
				CNNPP::set_src_surf(&cnn.layer_buffer[1].buffer[i], 2, i);
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
				cnn.layer_buffer[2].buffer[i] = Image_32f(context, queue, cnn.layer_buffer[2].size.cols, cnn.layer_buffer[2].size.rows, 1, true, ALIGN_DEF);
				CNNPP::set_dst_surf(&cnn.layer_buffer[2].buffer[i], 3, i);
				CNNPP::set_src_surf(&cnn.layer_buffer[2].buffer[i], 3, i);
			}

			cnn.output_buffer_size.cols = cnn.conv_l3.ROI.cols;
			cnn.output_buffer_size.rows = cnn.conv_l3.ROI.rows;
			cnn.output_buffer_size.step = cnn.layer_buffer[2].size.cols;
			cnn.output_buffer_size.size = cnn.output_buffer_size.rows * cnn.output_buffer_size.step;

			//bind texture on device
			//bind_texture(cnn.input_data, 1, 1);
			//bind_texture(cnn.layer_buffer[0].buffer, 2, cnn.layer_buffer[0].map_count);
			//bind_texture(cnn.layer_buffer[1].buffer, 3, cnn.layer_buffer[1].map_count);
			//bind_texture(cnn.layer_buffer[2].buffer, 4, cnn.layer_buffer[2].map_count);

			//set dst surf
			//set_dst_surf(cnn.layer_buffer[0].buffer, 1, cnn.layer_buffer[0].map_count);
			//set_dst_surf(cnn.layer_buffer[1].buffer, 2, cnn.layer_buffer[1].map_count);
			//set_dst_surf(cnn.layer_buffer[2].buffer, 3, cnn.layer_buffer[2].map_count);
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
			cnn.subs_weight.clear();
			cnn.subs_bias.clear();

			//simple nn weight
			cnn.snn_hl_weight.clear();
			cnn.snn_hl_bias.clear();
			
			cnn.snn_ol_weight.clear();

			CNNPP::release_cl_buffers();
			CNNPP::destroy_cl_program();

			device = 0;
			context = 0;
			queue = 0;
		}

		void ConvNeuralNetwork::ResizeBuffers(Size size)
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
		void ConvNeuralNetwork::Forward(Image_32f* response_map, Image_32f* image)
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

#ifdef CHECK_TEST
			if (cnn_ref == NULL) return;
			printf("\n	cnn_cl: image size = (%d, %d)\n", image->width, image->height);
			Legacy::ConvNeuralNetwork cnn_old(cnn_ref, image->width, image->height);
			SIMD::Image_32f r_map;
			image->updateDataHost();
			SIMD::Image_32f temp(image->width, image->height, image->nChannel, image->dataHost, image->widthStepHost);
			cnn_old.Forward(r_map, temp);
			temp.clear();
#endif

#ifdef PROFILE_CNN_CL
			printf("\n	cnn_cl: image size = (%d, %d)\n", image->width, image->height);
			Timer timer(queue, true);
#endif

#ifdef PROFILE_CPUCNN
			printf("	cnn_cl: offset = %7.3f ms\n", timer.get(1000));
			timer.start();
#endif

			//cnn.input_data->copyData(image);

			CNNPP::set_src_surf(image, 0, 0);
			if (cnn.max_pool)
			{
				CNNPP::run_L1_max(cnn.conv_l1.ROI, block_size, queue);
			}
			//else
			//{
			//	CNNPP::run_L1_avr(cnn.conv_l1.ROI, queue);
			//}
			//unbind_texture(1, 1);

#ifdef PROFILE_CNN_CL
			printf("	cnn_cl: run_L1 = %7.3f ms\n", timer.get(1000));
#endif		

#ifdef CHECK_TEST
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{ 
				cnn.layer_buffer[0].buffer[i].updateDataHost();

				int count1 = 0;
				double d1 = 0;
				for (int y = 0; y < cnn_old.cnn.subs_neuron[0][i].row_count(); ++y)
				{
					for (int x = 0; x < cnn_old.cnn.subs_neuron[0][i].col_count(); ++x)
					{
						double d = cnn_old.cnn.subs_neuron[0][i][y][x] - cnn.layer_buffer[0].buffer[i].dataHost[y * cnn.layer_buffer[0].size.cols + x];
						if (abs(d) > 1.0E-4)
						{
							d1 += abs(d);
							count1++;
						}
					}
				}
				printf("	d1 = %f, count1 = %d\n", d1, count1);
				if (abs(d1) > 0.0) system("pause");
			}
#endif

#ifdef PROFILE_CNN_CL
			timer.start();
#endif

			if (cnn.max_pool)
			{
				CNNPP::run_L2_max(cnn.conv_l2.ROI, block_size, queue);
			}
			//else
			//{
			//	CNNPP::run_L2_avr(cnn.conv_l2.ROI, queue);
			//}

#ifdef PROFILE_CNN_CL
			printf("	cnn_cl: run_L2 = %7.3f ms\n", timer.get(1000));
#endif	

#ifdef CHECK_TEST
			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				cnn.layer_buffer[1].buffer[i].updateDataHost();

				int count2 = 0;
				double d2 = 0;
				for (int y = 0; y < cnn_old.cnn.subs_neuron[1][i].row_count(); ++y)
				{
					for (int x = 0; x < cnn_old.cnn.subs_neuron[1][i].col_count(); ++x)
					{
						double d = cnn_old.cnn.subs_neuron[1][i][y][x] - cnn.layer_buffer[1].buffer[i].dataHost[y * cnn.layer_buffer[1].size.cols + x];
						if (abs(d) > 1.0E-4)
						{
							d2 += abs(d);
							count2++;
						}
					}
				}
				printf("	d2 = %f, count2 = %d\n", d2, count2);
				if (abs(d2) > 0.0) system("pause");
			}
#endif

#ifdef PROFILE_CNN_CL
			timer.start();
#endif

			if (cnn.max_pool)
			{
				CNNPP::run_L3_max(cnn.conv_l3.ROI, block_size, queue);
			}
			//else
			//{
			//	CNNPP::run_L3_avr(cnn.conv_l3.ROI, block_size, queue);
			//}

#ifdef PROFILE_CNN_CL
			printf("	cnn_cl: run_L3 = %7.3f ms\n", timer.get(1000));
			timer.start();
#endif

			CNNPP::set_dst_surf(response_map, 4, 1);
			if (cnn.max_pool)
			{
				CNNPP::run_HL_max(cnn.conv_l3.ROI, block_size, queue);
			}
			//else
			//{
			//	CNNPP::run_HL_avr(cnn.conv_l3.ROI, block_size, queue);
			//}

#ifdef PROFILE_CNN_CL
			printf("	cnn_cl: run_HL = %7.3f ms\n", timer.get(1000));
#endif

#ifdef CHECK_TEST
			response_map->updateDataHost();

			int count4 = 0;
			double d4 = 0;
			for (int y = 0; y < cnn_old.snn.output_neuron.row_count(); ++y)
			{
				for (int x = 0; x < cnn_old.snn.output_neuron.col_count(); ++x)
				{
					double d = cnn_old.snn.output_neuron[y][x][0] - response_map->dataHost[y * response_map->widthStepHost + x];
					if (abs(d) > 1.0E-4)
					{
						d4 += abs(d);
						count4++;
					}
				}
			}
			printf("	d4 = %f, count4 = %d\n", d4, count4);
			if (abs(d4) > 0.0) system("pause");
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

#if 0
		void ConvNeuralNetwork::SaveToBinaryFile(std::string file_name)
		{
			if (file_name == "") return;

			file_name.append(".bin");

			std::fstream file_bin;
			file_bin.open(file_name.c_str(), std::fstream::binary | std::fstream::out);

			if (!file_bin.is_open()) return;

			//version
			float format_version = 1.0f;
			file_bin.write((const char*)&(format_version), sizeof(format_version));

			//max pool
			file_bin.write((const char*)&(cnn.max_pool), sizeof(cnn.max_pool));

			//input_image
			file_bin.write((const char*)&(cnn.min_image_size.width), sizeof(cnn.min_image_size.width));
			file_bin.write((const char*)&(cnn.min_image_size.height), sizeof(cnn.min_image_size.height));

			//cnn_layers
			file_bin.write((const char*)&(cnn.layer_count), sizeof(cnn.layer_count));
			for (int i = 0; i < cnn.layer_count; ++i)
			{
				file_bin.write((const char*)&(cnn.layer_buffer[i].map_count), sizeof(cnn.layer_buffer[i].map_count));
			}

			//conv l1
			file_bin.write((const char*)&(cnn.conv_l1.size.cols), sizeof(cnn.conv_l1.size.cols));
			file_bin.write((const char*)&(cnn.conv_l1.size.rows), sizeof(cnn.conv_l1.size.rows));
			for (int i = 0; i < cnn.layer_buffer[0].map_count; ++i)
			{
				for (int j = 0; j < cnn.conv_l1.size.size; ++j)
				{
					file_bin.write((const char*)&(cnn.conv_l1.kernels[i][j]), sizeof(cnn.conv_l1.kernels[i][j]));
				}
			}

			//conv l2
			file_bin.write((const char*)&(cnn.conv_l2.size.cols), sizeof(cnn.conv_l2.size.cols));
			file_bin.write((const char*)&(cnn.conv_l2.size.rows), sizeof(cnn.conv_l2.size.rows));
			for (int i = 0; i < cnn.layer_buffer[1].map_count; ++i)
			{
				for (int j = 0; j < cnn.conv_l2.size.size; ++j)
				{
					file_bin.write((const char*)&(cnn.conv_l2.kernels[i][j]), sizeof(cnn.conv_l2.kernels[i][j]));
				}
			}

			//conv l3
			file_bin.write((const char*)&(cnn.conv_l3.size.cols), sizeof(cnn.conv_l3.size.cols));
			file_bin.write((const char*)&(cnn.conv_l3.size.rows), sizeof(cnn.conv_l3.size.rows));
			for (int i = 0; i < cnn.layer_buffer[2].map_count; ++i)
			{
				for (int j = 0; j < cnn.conv_l3.size.size; ++j)
				{
					file_bin.write((const char*)&(cnn.conv_l3.kernels[i][j]), sizeof(cnn.conv_l3.kernels[i][j]));
				}
			}

			//conv nn weight
			file_bin.write((const char*)&(cnn.af_scale), sizeof(cnn.af_scale));

			for (int i = 0; i < cnn.layer_count; ++i)
			{
				for (int j = 0; j < cnn.layer_buffer[i].map_count; ++j)
				{
					file_bin.write((const char*)&(cnn.conv_bias[i][j]), sizeof(cnn.conv_bias[i][j]));
					file_bin.write((const char*)&(cnn.subs_weight[i][j]), sizeof(cnn.subs_weight[i][j]));
					file_bin.write((const char*)&(cnn.subs_bias[i][j]), sizeof(cnn.subs_bias[i][j]));
				}
			}

			//simple nn weight
			file_bin.write((const char*)&(cnn.snn_full_connect), sizeof(cnn.snn_full_connect));
			file_bin.write((const char*)&(cnn.snn_hl_size), sizeof(cnn.snn_hl_size));
			if (!cnn.snn_full_connect)
			{
				for (int i = 0; i < cnn.snn_hl_size; ++i)
				{
					file_bin.write((const char*)&(cnn.snn_hl_weight[i]), sizeof(cnn.snn_hl_weight[i]));
				}
			}
			else
			{
				for (int i = 0; i < cnn.snn_hl_size; ++i)
				{
					for (int j = 0; j < cnn.layer_buffer[cnn.layer_count - 1].map_count; ++j)
					{
						file_bin.write((const char*)&(cnn.snn_hl_weight_fc[i][j]), sizeof(cnn.snn_hl_weight_fc[i][j]));
					}
				}
			}

			for (int i = 0; i < cnn.snn_hl_size; ++i)
			{
				file_bin.write((const char*)&(cnn.snn_hl_bias[i]), sizeof(cnn.snn_hl_bias[i]));
			}

			for (int i = 0; i < cnn.snn_hl_size; ++i)
			{
				file_bin.write((const char*)&(cnn.snn_ol_weight[i]), sizeof(cnn.snn_ol_weight[i]));
			}

			file_bin.write((const char*)&(cnn.snn_ol_bias), sizeof(cnn.snn_ol_bias));

			file_bin.close();
		}
#endif
	}

#endif
}