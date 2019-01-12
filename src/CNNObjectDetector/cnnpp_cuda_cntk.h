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

#include "image.h"
#include "image_cuda.h"

#ifdef USE_CUDA
#	include <cuda.h>
#	include <cuda_runtime.h>
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{
#if defined(USE_CUDA) && defined(USE_CNTK_MODELS)

	namespace CUDA
	{
		class CNNPP
		{
		public:
			static void set_texture_cache_params(int model);
			static void bind_texture(int model, CUDA::Image_32f* src, int layer, int map_count);
			static void set_dst_surf(int model, CUDA::Image_32f* dst, int layer, int map_count);
			static void unbind_texture(int model, int layer, int map_count);
			static void set_cache_device_params(int model);

			static void set_kernel_on_device(int model, float* kernel, int size, int layer, int surface);
			static void set_conv_b_on_device(int model, float* conv_b, int layer, int surface);
			static void set_lrelu_w1_on_device(int model, float* lrelu_w1, int layer, int surface);
			static void set_lrelu_w2_on_device(int model, float* lrelu_w2, int layer, int surface);
			static void set_bn_w_on_device(int model, float* bn_w, int layer, int surface);
			static void set_bn_b_on_device(int model, float* bn_b, int layer, int surface);
			static void set_scale_on_device(int model, float* scale, int layer);
			static void set_hl_w_on_device(int model, float* hl_w, int surface);
			static void set_hl_b_on_device(int model, float* hl_b, int surface);
			static void set_hl_tanh_w_on_device(int model, float* hl_tanh_w, int surface);
			static void set_hl_bn_w_on_device(int model, float* hl_bn_w, int surface);
			static void set_hl_bn_b_on_device(int model, float* hl_bn_b, int surface);
			static void set_ol_w_on_device(int model, float* ol_w, int surface);
			static void set_ol_b_on_device(int model, float* ol_b, int surface);
			static void set_ol_tanh_w_on_device(int model, float* tanh_w);

			static void run_L1(int model, Size2d& ROI, cudaStream_t cuda_stream = 0);
			static void run_L2(int model, Size2d& ROI, cudaStream_t cuda_stream = 0);
			static void run_L3(int model, Size2d& ROI, cudaStream_t cuda_stream = 0);
			static void run_HL(int model, Size2d& ROI, int index_output, cudaStream_t cuda_stream = 0);
		};
	}

#endif
}