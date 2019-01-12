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


//========================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_FIXED_POINT

	namespace SIMD
	{
		class CNNPP_v3
		{
		private:
			const int Q = 14;
			const int Kq = (1 << (Q - 0));

			const int abs_mask = 0x7FFFFFFF;
			const float one = 1.f;
			const float tanh_a = 1.41645f;
			const float toFxP = float(Kq);
			const float toFP = 1.f / float(Kq);
			const float scale_data = 256.f;
			float fct = 1.f;

			//const float skt1 = 0.05f;
			//const float skt2 = 0.7f;
			//const float skt3 = 1.6f;
			//const float skt4 = 1.8f;
			//const float skt5 = 1.5f;
			//const float skt6 = 1.8f;

			const float skt1 = 0.05f;
			const float skt2 = 0.7f;
			const float skt3 = 1.4f;
			const float skt4 = 1.6f;
			const float skt5 = 1.5f;
			const float skt6 = 1.8f;

		public:
			CNNPP_v3() { }
			~CNNPP_v3() { }

			void conv_4x4_lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L = 0, size_t H = 0, int num_threads = 1);
			void conv_3x3_lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L = 0, size_t H = 0, int num_threads = 1);
			void conv_5x4_lrelu_bn(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L = 0, size_t H = 0, int num_threads = 1);
			void mulCN_add_tanhW_add(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float** __restrict snn_hl_w, float** __restrict snn_hl_b, float* __restrict snn_tanh_w, float* __restrict snn_bn_w, float* __restrict snn_bn_b, float** __restrict snn_ol_w, size_t L, size_t H, int num_threads = 1);
			void tanhW(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict snn_ol_b, float* __restrict tanh_w, float* __restrict scale, size_t L, size_t H, int num_threads = 1);

			//void conv_4x4_lrelu_bn_max_old(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L = 0, size_t H = 0, int num_threads = 1);
			//void conv_3x3_lrelu_bn_max_old(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L = 0, size_t H = 0, int num_threads = 1);
			//void conv_5x4_lrelu_bn_old(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L = 0, size_t H = 0, int num_threads = 1);
			//void mulCN_add_tanhW_add_old(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float** __restrict snn_hl_w, float** __restrict snn_hl_b, float* __restrict snn_tanh_w, float* __restrict snn_bn_w, float* __restrict snn_bn_b, float** __restrict snn_ol_w, size_t L, size_t H, int num_threads = 1);

			CNNPP_v3(const CNNPP_v3&) = delete;
			CNNPP_v3& operator=(const CNNPP_v3&) = delete;
		};
	}

#endif
}