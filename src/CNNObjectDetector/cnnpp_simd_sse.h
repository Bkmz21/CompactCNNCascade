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
#ifdef USE_SSE

	namespace SIMD
	{
		class CNNPP
		{
		private:
			const int abs_mask = 0x7FFFFFFF;
			const float one = 1.f;
			const float tanh_a = 1.41645f;

		public:
			CNNPP() { }
			~CNNPP() { }

			void conv_4x4(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L = 0, size_t H = 0);
			void conv_3x3(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L = 0, size_t H = 0);
			void conv_6x5(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L = 0, size_t H = 0);
			void conv_8x7(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L = 0, size_t H = 0);

			void tanh_avr_tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale);
			void max_tanh_tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale);

			void tanh_tanh_2tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale, float* __restrict snn_hl_w0, float* __restrict snn_hl_b0, float* __restrict snn_hl_w1, float* __restrict snn_hl_b1, float* __restrict snn_ol_w0, float* __restrict snn_ol_w1);
			void tanh_tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale);
			void tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict snn_ol_b, float* __restrict scale);

			void add(float* __restrict dst, float* __restrict src1, float* __restrict src2, int size_);
			void add2(float* __restrict dst, float* __restrict src1, float* __restrict src2, float* __restrict src3, int size_);

			void mulC(float* __restrict dst, float* __restrict src_mulC, int size_, float* __restrict snn_ol_w);
			void mulC1_add(float* __restrict dst, float* __restrict src1_mulC, float* __restrict src2, int size_, float* __restrict snn_hl_w);
			void mulC2_add(float* __restrict dst, float* __restrict src1_mulC0, float* __restrict src2_mulC1, int size_, float* __restrict snn_hl_w0, float* __restrict snn_hl_w1);

			void mulC24_add_tanh(float* __restrict dst, float* __restrict* src, int size_, float* __restrict snn_hl_w, float* __restrict snn_hl_b, float* __restrict scale, float* __restrict snn_ol_w);

			CNNPP(const CNNPP&) = delete;
			CNNPP& operator=(const CNNPP&) = delete;
		};
	}

#endif
}