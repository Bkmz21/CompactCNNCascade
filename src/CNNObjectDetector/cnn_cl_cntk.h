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
#include "image.h"
#include "image_cl.h"
#include "cnnpp_cl_cntk.h"

#include <vector>
#include <string>

#ifdef PROFILE_CNN_CL
#	include "timer.h"
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{
#if defined(USE_CL) && defined(USE_CNTK_MODELS)

	namespace CL
	{
		class ConvNeuralNetwork
		{
		private:
			struct Layer_buffer
			{
				int map_count = 0;

				Size2d size;
				std::vector<Image_32f> buffer;
			};
			struct Layer_filter
			{
				Size2d ROI;

				Size2d size;
				std::vector<SIMD::Array_32f> kernels;
			};
			struct CNN
			{
				Size min_image_size;
				Size max_image_size;

				Size2d input_buffer_size;
				Size2d output_buffer_size;

				int layer_count = 0;
				std::vector<Layer_buffer> layer_buffer;

				Layer_filter conv_l1;
				Layer_filter conv_l2;
				Layer_filter conv_l3;

				std::vector<SIMD::Array_32f> conv_bias;
				std::vector<SIMD::Array_32f> leakyReLU_w1;
				std::vector<SIMD::Array_32f> leakyReLU_w2;

				std::vector<SIMD::Array_32f> bn_weight;
				std::vector<SIMD::Array_32f> bn_bias;

				int snn_hl_size = 0;
				int snn_connect_count = 0;
				int hl_scale = 0;
				SIMD::Array_32f snn_hl_weight;
				SIMD::Array_32f snn_hl_bias;
				SIMD::Array_32f snn_hl_tanh_w;

				SIMD::Array_32f snn_hl_bn_weight;
				SIMD::Array_32f snn_hl_bn_bias;

				int snn_ol_neuron_count = 0;
				SIMD::Array_32f snn_ol_weight;
				SIMD::Array_32f snn_ol_bias;
				float snn_ol_tanh_w = 0.f;

				int index_output = 0;

				float af_scale = 0.f;
				bool max_pool = false;
				bool snn_full_connect = false;
			};

			cl_device_id device;
			cl_context context;
			cl_command_queue queue;

			Size block_size;
			
			CNN cnn;

			void ResizeBuffers(const Size size);

		public:
			ConvNeuralNetwork() { }
			~ConvNeuralNetwork() { Clear(); }

			void Init(std::string file_name, int index_output, cl_device_id _devices, cl_context _context, cl_command_queue _queue);
			void AllocateMemory(const Size size);
			void Clear();

			void Forward(Image_32f* response_map, Image_32f* image);

			inline bool isEmpty() const { return device == 0 || context == 0 || queue == 0 || cnn.min_image_size.width == 0 || cnn.min_image_size.height == 0; }

			inline Size getBlockSize()		   const { return block_size; }
			inline Size getMinInputImgSize()   const { return Size(cnn.min_image_size.width, cnn.min_image_size.height); }
			inline Size getMaxInputImgSize()   const { return Size(cnn.max_image_size.width, cnn.max_image_size.height); }
			inline Size getInputImgSize()	   const { return Size(cnn.input_buffer_size.cols, cnn.input_buffer_size.rows); }
			inline Size getOutputImgSize()	   const { return Size(cnn.output_buffer_size.cols, cnn.output_buffer_size.rows); }
			Size getOutputImgSize(const Size size);
			inline float getInputOutputRatio() const { return 4.f; /*(float)cnn.input_buffer_size.rows / (float)cnn.output_buffer_size.rows;*/ }

			inline int getIndexOutput() const { return cnn.index_output; }
		};
	}

#endif
}