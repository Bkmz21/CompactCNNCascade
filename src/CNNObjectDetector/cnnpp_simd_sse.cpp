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


#include "cnnpp_simd_sse.h"
#include <immintrin.h>


//========================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_SSE

	namespace SIMD
	{
#ifndef USE_FMA

		void CNNPP::conv_4x4(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;

			__m128 ymm12 = _mm_load_ps(kernel);
			__m128 ymm13 = _mm_load_ps(kernel + REG_SIZE);
			__m128 ymm14 = _mm_load_ps(kernel + 2 * REG_SIZE);
			__m128 ymm15 = _mm_load_ps(kernel + 3 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				__m128 ymm0 = _mm_load_ps(pSrc0);
				__m128 ymm1 = _mm_load_ps(pSrc1);
				__m128 ymm2 = _mm_load_ps(pSrc2);
				__m128 ymm3 = _mm_load_ps(pSrc3);
				pSrc0 += REG_SIZE;
				pSrc1 += REG_SIZE;
				pSrc2 += REG_SIZE;
				pSrc3 += REG_SIZE;

				size_t i = 0;
				for (; i < L - REG_SIZE; i += REG_SIZE)
				{
					__m128 ymm4 = _mm_load_ps(pSrc0);
					__m128 ymm5 = _mm_load_ps(pSrc1);
					__m128 ymm6 = _mm_load_ps(pSrc2);
					__m128 ymm7 = _mm_load_ps(pSrc3);
					pSrc0 += REG_SIZE;
					pSrc1 += REG_SIZE;
					pSrc2 += REG_SIZE;
					pSrc3 += REG_SIZE;

					__m128 ymm8 = _mm_mul_ps(ymm0, ymm12);
					__m128 ymm9 = _mm_mul_ps(ymm1, ymm13);
					__m128 ymm11 = _mm_add_ps(ymm8, ymm9);

					ymm8 = _mm_mul_ps(ymm2, ymm14);
					ymm11 = _mm_add_ps(ymm11, ymm8);

					ymm8 = _mm_mul_ps(ymm3, ymm15);
					ymm11 = _mm_add_ps(ymm11, ymm8);

					ymm8 = _mm_shuffle_ps(ymm11, ymm11, 14);
					ymm11 = _mm_add_ps(ymm11, ymm8);
					ymm8 = _mm_shuffle_ps(ymm11, ymm11, 1);
					ymm11 = _mm_add_ps(ymm11, ymm8);

					//-----------------------------------------

					ymm0 = _mm_blend_ps(ymm0, ymm4, 17);
					ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
					ymm8 = _mm_mul_ps(ymm0, ymm12);

					ymm1 = _mm_blend_ps(ymm1, ymm5, 17);
					ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
					ymm9 = _mm_mul_ps(ymm1, ymm13);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm2 = _mm_blend_ps(ymm2, ymm6, 17);
					ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
					ymm9 = _mm_mul_ps(ymm2, ymm14);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm3 = _mm_blend_ps(ymm3, ymm7, 17);
					ymm15 = _mm_shuffle_ps(ymm15, ymm15, 147);
					ymm9 = _mm_mul_ps(ymm3, ymm15);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 14);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 1);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 34);

					//-----------------------------------------

					ymm0 = _mm_blend_ps(ymm0, ymm4, 34);
					ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
					ymm8 = _mm_mul_ps(ymm0, ymm12);

					ymm1 = _mm_blend_ps(ymm1, ymm5, 34);
					ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
					ymm9 = _mm_mul_ps(ymm1, ymm13);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm2 = _mm_blend_ps(ymm2, ymm6, 34);
					ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
					ymm9 = _mm_mul_ps(ymm2, ymm14);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm3 = _mm_blend_ps(ymm3, ymm7, 34);
					ymm15 = _mm_shuffle_ps(ymm15, ymm15, 147);
					ymm9 = _mm_mul_ps(ymm3, ymm15);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 64);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 176);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 68);

					//-----------------------------------------

					ymm0 = _mm_blend_ps(ymm0, ymm4, 68);
					ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
					ymm8 = _mm_mul_ps(ymm0, ymm12);

					ymm1 = _mm_blend_ps(ymm1, ymm5, 68);
					ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
					ymm9 = _mm_mul_ps(ymm1, ymm13);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm2 = _mm_blend_ps(ymm2, ymm6, 68);
					ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
					ymm9 = _mm_mul_ps(ymm2, ymm14);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm3 = _mm_blend_ps(ymm3, ymm7, 68);
					ymm15 = _mm_shuffle_ps(ymm15, ymm15, 147);
					ymm9 = _mm_mul_ps(ymm3, ymm15);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 64);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 176);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 136);

					_mm_store_ps(pDst, ymm11);
					pDst += REG_SIZE;

					ymm0 = ymm4;
					ymm1 = ymm5;
					ymm2 = ymm6;
					ymm3 = ymm7;

					ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
					ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
					ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
					ymm15 = _mm_shuffle_ps(ymm15, ymm15, 147);
				}

				//============================================================================

				__m128 ymm4 = _mm_setzero_ps();
				__m128 ymm5 = _mm_setzero_ps();
				__m128 ymm6 = _mm_setzero_ps();
				__m128 ymm7 = _mm_setzero_ps();

				if (i + REG_SIZE < src_size_l)
				{
					ymm4 = _mm_load_ps(pSrc0);
					ymm5 = _mm_load_ps(pSrc1);
					ymm6 = _mm_load_ps(pSrc2);
					ymm7 = _mm_load_ps(pSrc3);
				}

				__m128 ymm8 = _mm_mul_ps(ymm0, ymm12);
				__m128 ymm9 = _mm_mul_ps(ymm1, ymm13);
				__m128 ymm11 = _mm_add_ps(ymm8, ymm9);

				ymm8 = _mm_mul_ps(ymm2, ymm14);
				ymm11 = _mm_add_ps(ymm11, ymm8);

				ymm8 = _mm_mul_ps(ymm3, ymm15);
				ymm11 = _mm_add_ps(ymm11, ymm8);

				ymm8 = _mm_shuffle_ps(ymm11, ymm11, 14);
				ymm11 = _mm_add_ps(ymm11, ymm8);
				ymm8 = _mm_shuffle_ps(ymm11, ymm11, 1);
				ymm11 = _mm_add_ps(ymm11, ymm8);

				//-----------------------------------------

				ymm0 = _mm_blend_ps(ymm0, ymm4, 17);
				ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
				ymm8 = _mm_mul_ps(ymm0, ymm12);

				ymm1 = _mm_blend_ps(ymm1, ymm5, 17);
				ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
				ymm9 = _mm_mul_ps(ymm1, ymm13);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm2 = _mm_blend_ps(ymm2, ymm6, 17);
				ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
				ymm9 = _mm_mul_ps(ymm2, ymm14);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm3 = _mm_blend_ps(ymm3, ymm7, 17);
				ymm15 = _mm_shuffle_ps(ymm15, ymm15, 147);
				ymm9 = _mm_mul_ps(ymm3, ymm15);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 14);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 1);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 34);

				//-----------------------------------------

				ymm0 = _mm_blend_ps(ymm0, ymm4, 34);
				ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
				ymm8 = _mm_mul_ps(ymm0, ymm12);

				ymm1 = _mm_blend_ps(ymm1, ymm5, 34);
				ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
				ymm9 = _mm_mul_ps(ymm1, ymm13);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm2 = _mm_blend_ps(ymm2, ymm6, 34);
				ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
				ymm9 = _mm_mul_ps(ymm2, ymm14);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm3 = _mm_blend_ps(ymm3, ymm7, 34);
				ymm15 = _mm_shuffle_ps(ymm15, ymm15, 147);
				ymm9 = _mm_mul_ps(ymm3, ymm15);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 64);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 176);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 68);

				//-----------------------------------------

				ymm0 = _mm_blend_ps(ymm0, ymm4, 68);
				ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
				ymm8 = _mm_mul_ps(ymm0, ymm12);

				ymm1 = _mm_blend_ps(ymm1, ymm5, 68);
				ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
				ymm9 = _mm_mul_ps(ymm1, ymm13);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm2 = _mm_blend_ps(ymm2, ymm6, 68);
				ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
				ymm9 = _mm_mul_ps(ymm2, ymm14);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm3 = _mm_blend_ps(ymm3, ymm7, 68);
				ymm15 = _mm_shuffle_ps(ymm15, ymm15, 147);
				ymm9 = _mm_mul_ps(ymm3, ymm15);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 64);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 176);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 136);

				_mm_store_ps(pDst, ymm11);

				ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
				ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
				ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
				ymm15 = _mm_shuffle_ps(ymm15, ymm15, 147);

				//============================================================================
			}
		}
		void CNNPP::conv_3x3(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 2;
			if (H == 0) H = src_size_h - 2;

			__m128 ymm12 = _mm_load_ps(kernel);
			__m128 ymm13 = _mm_load_ps(kernel + REG_SIZE);
			__m128 ymm14 = _mm_load_ps(kernel + 2 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				__m128 ymm0 = _mm_load_ps(pSrc0);
				__m128 ymm1 = _mm_load_ps(pSrc1);
				__m128 ymm2 = _mm_load_ps(pSrc2);
				pSrc0 += REG_SIZE;
				pSrc1 += REG_SIZE;
				pSrc2 += REG_SIZE;

				size_t i = 0;
				for (; i < L - REG_SIZE; i += REG_SIZE)
				{
					__m128 ymm4 = _mm_load_ps(pSrc0);
					__m128 ymm5 = _mm_load_ps(pSrc1);
					__m128 ymm6 = _mm_load_ps(pSrc2);
					pSrc0 += REG_SIZE;
					pSrc1 += REG_SIZE;
					pSrc2 += REG_SIZE;

					__m128 ymm8 = _mm_mul_ps(ymm0, ymm12);
					__m128 ymm9 = _mm_mul_ps(ymm1, ymm13);
					__m128 ymm11 = _mm_add_ps(ymm8, ymm9);

					ymm8 = _mm_mul_ps(ymm2, ymm14);
					ymm11 = _mm_add_ps(ymm11, ymm8);

					ymm8 = _mm_shuffle_ps(ymm11, ymm11, 14);
					ymm11 = _mm_add_ps(ymm11, ymm8);
					ymm8 = _mm_shuffle_ps(ymm11, ymm11, 1);
					ymm11 = _mm_add_ps(ymm11, ymm8);

					//-----------------------------------------

					ymm0 = _mm_blend_ps(ymm0, ymm4, 17);
					ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
					ymm8 = _mm_mul_ps(ymm0, ymm12);

					ymm1 = _mm_blend_ps(ymm1, ymm5, 17);
					ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
					ymm9 = _mm_mul_ps(ymm1, ymm13);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm2 = _mm_blend_ps(ymm2, ymm6, 17);
					ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
					ymm9 = _mm_mul_ps(ymm2, ymm14);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 14);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 1);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 34);

					//-----------------------------------------

					ymm0 = _mm_blend_ps(ymm0, ymm4, 34);
					ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
					ymm8 = _mm_mul_ps(ymm0, ymm12);

					ymm1 = _mm_blend_ps(ymm1, ymm5, 34);
					ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
					ymm9 = _mm_mul_ps(ymm1, ymm13);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm2 = _mm_blend_ps(ymm2, ymm6, 34);
					ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
					ymm9 = _mm_mul_ps(ymm2, ymm14);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 64);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 176);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 68);

					//-----------------------------------------

					ymm0 = _mm_blend_ps(ymm0, ymm4, 68);
					ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
					ymm8 = _mm_mul_ps(ymm0, ymm12);

					ymm1 = _mm_blend_ps(ymm1, ymm5, 68);
					ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
					ymm9 = _mm_mul_ps(ymm1, ymm13);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm2 = _mm_blend_ps(ymm2, ymm6, 68);
					ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
					ymm9 = _mm_mul_ps(ymm2, ymm14);
					ymm8 = _mm_add_ps(ymm8, ymm9);

					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 64);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm9 = _mm_shuffle_ps(ymm8, ymm8, 176);
					ymm8 = _mm_add_ps(ymm8, ymm9);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 136);

					_mm_store_ps(pDst, ymm11);
					pDst += REG_SIZE;

					ymm0 = ymm4;
					ymm1 = ymm5;
					ymm2 = ymm6;

					ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
					ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
					ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
				}

				//============================================================================

				__m128 ymm4 = _mm_setzero_ps();
				__m128 ymm5 = _mm_setzero_ps();
				__m128 ymm6 = _mm_setzero_ps();

				if (i + REG_SIZE < src_size_l)
				{
					ymm4 = _mm_load_ps(pSrc0);
					ymm5 = _mm_load_ps(pSrc1);
					ymm6 = _mm_load_ps(pSrc2);
				}

				__m128 ymm8 = _mm_mul_ps(ymm0, ymm12);
				__m128 ymm9 = _mm_mul_ps(ymm1, ymm13);
				__m128 ymm11 = _mm_add_ps(ymm8, ymm9);

				ymm8 = _mm_mul_ps(ymm2, ymm14);
				ymm11 = _mm_add_ps(ymm11, ymm8);

				ymm8 = _mm_shuffle_ps(ymm11, ymm11, 14);
				ymm11 = _mm_add_ps(ymm11, ymm8);
				ymm8 = _mm_shuffle_ps(ymm11, ymm11, 1);
				ymm11 = _mm_add_ps(ymm11, ymm8);

				//-----------------------------------------

				ymm0 = _mm_blend_ps(ymm0, ymm4, 17);
				ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
				ymm8 = _mm_mul_ps(ymm0, ymm12);

				ymm1 = _mm_blend_ps(ymm1, ymm5, 17);
				ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
				ymm9 = _mm_mul_ps(ymm1, ymm13);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm2 = _mm_blend_ps(ymm2, ymm6, 17);
				ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
				ymm9 = _mm_mul_ps(ymm2, ymm14);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 14);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 1);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 34);

				//-----------------------------------------

				ymm0 = _mm_blend_ps(ymm0, ymm4, 34);
				ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
				ymm8 = _mm_mul_ps(ymm0, ymm12);

				ymm1 = _mm_blend_ps(ymm1, ymm5, 34);
				ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
				ymm9 = _mm_mul_ps(ymm1, ymm13);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm2 = _mm_blend_ps(ymm2, ymm6, 34);
				ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
				ymm9 = _mm_mul_ps(ymm2, ymm14);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 64);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 176);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 68);

				//-----------------------------------------

				ymm0 = _mm_blend_ps(ymm0, ymm4, 68);
				ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
				ymm8 = _mm_mul_ps(ymm0, ymm12);

				ymm1 = _mm_blend_ps(ymm1, ymm5, 68);
				ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
				ymm9 = _mm_mul_ps(ymm1, ymm13);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm2 = _mm_blend_ps(ymm2, ymm6, 68);
				ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);
				ymm9 = _mm_mul_ps(ymm2, ymm14);
				ymm8 = _mm_add_ps(ymm8, ymm9);

				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 64);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm9 = _mm_shuffle_ps(ymm8, ymm8, 176);
				ymm8 = _mm_add_ps(ymm8, ymm9);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 136);

				_mm_store_ps(pDst, ymm11);

				ymm12 = _mm_shuffle_ps(ymm12, ymm12, 147);
				ymm13 = _mm_shuffle_ps(ymm13, ymm13, 147);
				ymm14 = _mm_shuffle_ps(ymm14, ymm14, 147);

				//============================================================================
			}
		}
		void CNNPP::conv_8x7(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 6;
			if (H == 0) H = src_size_h - 7;

			const __m128 ymm8_1 = _mm_load_ps(kernel);
			const __m128 ymm8_2 = _mm_load_ps(kernel + REG_SIZE);
			const __m128 ymm9_1 = _mm_load_ps(kernel + 2 * REG_SIZE);
			const __m128 ymm9_2 = _mm_load_ps(kernel + 3 * REG_SIZE);
			const __m128 ymm10_1 = _mm_load_ps(kernel + 4 * REG_SIZE);
			const __m128 ymm10_2 = _mm_load_ps(kernel + 5 * REG_SIZE);
			const __m128 ymm11_1 = _mm_load_ps(kernel + 6 * REG_SIZE);
			const __m128 ymm11_2 = _mm_load_ps(kernel + 7 * REG_SIZE);
			const __m128 ymm12_1 = _mm_load_ps(kernel + 8 * REG_SIZE);
			const __m128 ymm12_2 = _mm_load_ps(kernel + 9 * REG_SIZE);
			const __m128 ymm13_1 = _mm_load_ps(kernel + 10 * REG_SIZE);
			const __m128 ymm13_2 = _mm_load_ps(kernel + 11 * REG_SIZE);
			const __m128 ymm14_1 = _mm_load_ps(kernel + 12 * REG_SIZE);
			const __m128 ymm14_2 = _mm_load_ps(kernel + 13 * REG_SIZE);
			const __m128 ymm15_1 = _mm_load_ps(kernel + 14 * REG_SIZE);
			const __m128 ymm15_2 = _mm_load_ps(kernel + 15 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pSrc4 = src + (j + 4) * src_size_l;
				float* __restrict pSrc5 = src + (j + 5) * src_size_l;
				float* __restrict pSrc6 = src + (j + 6) * src_size_l;
				float* __restrict pSrc7 = src + (j + 7) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				for (size_t i = 0; i < L; ++i)
				{
					__m128 ymm0_1 = _mm_loadu_ps(pSrc0);
					__m128 ymm0_2 = _mm_loadu_ps(REG_SIZE + pSrc0++);
					__m128 ymm1_1 = _mm_loadu_ps(pSrc1);
					__m128 ymm1_2 = _mm_loadu_ps(REG_SIZE + pSrc1++);
					__m128 ymm2_1 = _mm_loadu_ps(pSrc2);
					__m128 ymm2_2 = _mm_loadu_ps(REG_SIZE + pSrc2++);
					__m128 ymm3_1 = _mm_loadu_ps(pSrc3);
					__m128 ymm3_2 = _mm_loadu_ps(REG_SIZE + pSrc3++);

					__m128 ymm7_1 = _mm_mul_ps(ymm0_1, ymm8_1);
					__m128 ymm6_1 = _mm_mul_ps(ymm1_1, ymm9_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_mul_ps(ymm2_1, ymm10_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_mul_ps(ymm3_1, ymm11_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);

					__m128 ymm7_2 = _mm_mul_ps(ymm0_2, ymm8_2);
					__m128 ymm6_2 = _mm_mul_ps(ymm1_2, ymm9_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);
					ymm6_2 = _mm_mul_ps(ymm2_2, ymm10_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);
					ymm6_2 = _mm_mul_ps(ymm3_2, ymm11_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);

					ymm0_1 = _mm_loadu_ps(pSrc4);
					ymm0_2 = _mm_loadu_ps(REG_SIZE + pSrc4++);
					ymm1_1 = _mm_loadu_ps(pSrc5);
					ymm1_2 = _mm_loadu_ps(REG_SIZE + pSrc5++);
					ymm2_1 = _mm_loadu_ps(pSrc6);
					ymm2_2 = _mm_loadu_ps(REG_SIZE + pSrc6++);
					ymm3_1 = _mm_loadu_ps(pSrc7);
					ymm3_2 = _mm_loadu_ps(REG_SIZE + pSrc7++);

					ymm6_1 = _mm_mul_ps(ymm0_1, ymm12_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_mul_ps(ymm1_1, ymm13_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_mul_ps(ymm2_1, ymm14_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_mul_ps(ymm3_1, ymm15_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);

					ymm6_2 = _mm_mul_ps(ymm0_2, ymm12_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);
					ymm6_2 = _mm_mul_ps(ymm1_2, ymm13_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);
					ymm6_2 = _mm_mul_ps(ymm2_2, ymm14_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);
					ymm6_2 = _mm_mul_ps(ymm3_2, ymm15_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);

					ymm7_1 = _mm_add_ps(ymm7_1, ymm7_2);

					ymm6_1 = _mm_shuffle_ps(ymm7_1, ymm7_1, 14);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_shuffle_ps(ymm7_1, ymm7_1, 1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);

					_mm_store_ss(pDst++, ymm7_1);
				}
			}
		}
		void CNNPP::conv_6x5(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 4;
			if (H == 0) H = src_size_h - 5;

			const __m128 ymm8_1 = _mm_load_ps(kernel);
			const __m128 ymm8_2 = _mm_load_ps(kernel + REG_SIZE);
			const __m128 ymm9_1 = _mm_load_ps(kernel + 2 * REG_SIZE);
			const __m128 ymm9_2 = _mm_load_ps(kernel + 3 * REG_SIZE);
			const __m128 ymm10_1 = _mm_load_ps(kernel + 4 * REG_SIZE);
			const __m128 ymm10_2 = _mm_load_ps(kernel + 5 * REG_SIZE);
			const __m128 ymm11_1 = _mm_load_ps(kernel + 6 * REG_SIZE);
			const __m128 ymm11_2 = _mm_load_ps(kernel + 7 * REG_SIZE);
			const __m128 ymm12_1 = _mm_load_ps(kernel + 8 * REG_SIZE);
			const __m128 ymm12_2 = _mm_load_ps(kernel + 9 * REG_SIZE);
			const __m128 ymm13_1 = _mm_load_ps(kernel + 10 * REG_SIZE);
			const __m128 ymm13_2 = _mm_load_ps(kernel + 11 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pSrc4 = src + (j + 4) * src_size_l;
				float* __restrict pSrc5 = src + (j + 5) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				for (size_t i = 0; i < L; ++i)
				{
					__m128 ymm0_1 = _mm_loadu_ps(pSrc0);
					__m128 ymm0_2 = _mm_loadu_ps(REG_SIZE + pSrc0++);
					__m128 ymm1_1 = _mm_loadu_ps(pSrc1);
					__m128 ymm1_2 = _mm_loadu_ps(REG_SIZE + pSrc1++);
					__m128 ymm2_1 = _mm_loadu_ps(pSrc2);
					__m128 ymm2_2 = _mm_loadu_ps(REG_SIZE + pSrc2++);

					__m128 ymm7_1 = _mm_mul_ps(ymm0_1, ymm8_1);
					__m128 ymm6_1 = _mm_mul_ps(ymm1_1, ymm9_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_mul_ps(ymm2_1, ymm10_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);

					__m128 ymm7_2 = _mm_mul_ps(ymm0_2, ymm8_2);
					__m128 ymm6_2 = _mm_mul_ps(ymm1_2, ymm9_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);
					ymm6_2 = _mm_mul_ps(ymm2_2, ymm10_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);

					ymm0_1 = _mm_loadu_ps(pSrc3);
					ymm0_2 = _mm_loadu_ps(REG_SIZE + pSrc3++);
					ymm1_1 = _mm_loadu_ps(pSrc4);
					ymm1_2 = _mm_loadu_ps(REG_SIZE + pSrc4++);
					ymm2_1 = _mm_loadu_ps(pSrc5);
					ymm2_2 = _mm_loadu_ps(REG_SIZE + pSrc5++);

					ymm6_1 = _mm_mul_ps(ymm0_1, ymm11_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_mul_ps(ymm1_1, ymm12_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_mul_ps(ymm2_1, ymm13_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);

					ymm6_2 = _mm_mul_ps(ymm0_2, ymm11_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);
					ymm6_2 = _mm_mul_ps(ymm1_2, ymm12_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);
					ymm6_2 = _mm_mul_ps(ymm2_2, ymm13_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);

					ymm7_1 = _mm_add_ps(ymm7_1, ymm7_2);

					ymm6_1 = _mm_shuffle_ps(ymm7_1, ymm7_1, 14);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_shuffle_ps(ymm7_1, ymm7_1, 1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);

					_mm_store_ss(pDst++, ymm7_1);
				}
			}
		}

#else

		void CNNPP::conv_4x4(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;

			const __m128 ymm12 = _mm_load_ps(kernel);
			const __m128 ymm13 = _mm_load_ps(kernel + REG_SIZE);
			const __m128 ymm14 = _mm_load_ps(kernel + 2 * REG_SIZE);
			const __m128 ymm15 = _mm_load_ps(kernel + 3 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				__m128 ymm0 = _mm_load_ps(pSrc0);
				__m128 ymm1 = _mm_load_ps(pSrc1);
				__m128 ymm2 = _mm_load_ps(pSrc2);
				__m128 ymm3 = _mm_load_ps(pSrc3);
				pSrc0 += REG_SIZE;
				pSrc1 += REG_SIZE;
				pSrc2 += REG_SIZE;
				pSrc3 += REG_SIZE;

				size_t i = 0;
				for (; i < L - REG_SIZE; i += REG_SIZE)
				{
					__m128 ymm4 = _mm_load_ps(pSrc0);
					__m128 ymm5 = _mm_load_ps(pSrc1);
					__m128 ymm6 = _mm_load_ps(pSrc2);
					__m128 ymm7 = _mm_load_ps(pSrc3);
					pSrc0 += REG_SIZE;
					pSrc1 += REG_SIZE;
					pSrc2 += REG_SIZE;
					pSrc3 += REG_SIZE;

					__m128 ymm8 = _mm_mul_ps(ymm0, ymm12);
					__m128 ymm11 = _mm_mul_ps(ymm2, ymm14);

					ymm8 = _mm_fmadd_ps(ymm1, ymm13, ymm8);
					ymm11 = _mm_fmadd_ps(ymm3, ymm15, ymm11);

					ymm11 = _mm_add_ps(ymm8, ymm11);

					ymm8 = _mm_shuffle_ps(ymm11, ymm11, 14);
					ymm11 = _mm_add_ps(ymm11, ymm8);
					ymm8 = _mm_shuffle_ps(ymm11, ymm11, 1);
					ymm11 = _mm_add_ps(ymm11, ymm8);

					//-----------------------------------------

					__m128 ymm10 = _mm_blend_ps(ymm0, ymm4, 1);
					ymm10 = _mm_permute_ps(ymm10, 57);
					ymm8 = _mm_mul_ps(ymm10, ymm12);

					ymm10 = _mm_blend_ps(ymm1, ymm5, 1);
					ymm10 = _mm_permute_ps(ymm10, 57);
					ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm_blend_ps(ymm2, ymm6, 1);
					ymm10 = _mm_permute_ps(ymm10, 57);
					ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm_blend_ps(ymm3, ymm7, 1);
					ymm10 = _mm_permute_ps(ymm10, 57);
					ymm8 = _mm_fmadd_ps(ymm10, ymm15, ymm8);

					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 14);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 1);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 34);

					//-----------------------------------------

					ymm10 = _mm_blend_ps(ymm0, ymm4, 3);
					ymm10 = _mm_permute_ps(ymm10, 78);
					ymm8 = _mm_mul_ps(ymm10, ymm12);

					ymm10 = _mm_blend_ps(ymm1, ymm5, 3);
					ymm10 = _mm_permute_ps(ymm10, 78);
					ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm_blend_ps(ymm2, ymm6, 3);
					ymm10 = _mm_permute_ps(ymm10, 78);
					ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm_blend_ps(ymm3, ymm7, 3);
					ymm10 = _mm_permute_ps(ymm10, 78);
					ymm8 = _mm_fmadd_ps(ymm10, ymm15, ymm8);

					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 64);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 176);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 68);

					//-----------------------------------------

					ymm10 = _mm_blend_ps(ymm0, ymm4, 7);
					ymm10 = _mm_permute_ps(ymm10, 147);
					ymm8 = _mm_mul_ps(ymm10, ymm12);

					ymm10 = _mm_blend_ps(ymm1, ymm5, 7);
					ymm10 = _mm_permute_ps(ymm10, 147);
					ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm_blend_ps(ymm2, ymm6, 7);
					ymm10 = _mm_permute_ps(ymm10, 147);
					ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm_blend_ps(ymm3, ymm7, 7);
					ymm10 = _mm_permute_ps(ymm10, 147);
					ymm8 = _mm_fmadd_ps(ymm10, ymm15, ymm8);

					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 64);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 176);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 136);

					_mm_store_ps(pDst, ymm11);
					pDst += REG_SIZE;

					ymm0 = ymm4;
					ymm1 = ymm5;
					ymm2 = ymm6;
					ymm3 = ymm7;
				}

				//============================================================================

				__m128 ymm4 = _mm_setzero_ps();
				__m128 ymm5 = _mm_setzero_ps();
				__m128 ymm6 = _mm_setzero_ps();
				__m128 ymm7 = _mm_setzero_ps();

				if (i + REG_SIZE < src_size_l)
				{
					ymm4 = _mm_load_ps(pSrc0);
					ymm5 = _mm_load_ps(pSrc1);
					ymm6 = _mm_load_ps(pSrc2);
					ymm7 = _mm_load_ps(pSrc3);
				}

				__m128 ymm8 = _mm_mul_ps(ymm0, ymm12);
				__m128 ymm11 = _mm_mul_ps(ymm2, ymm14);

				ymm8 = _mm_fmadd_ps(ymm1, ymm13, ymm8);
				ymm11 = _mm_fmadd_ps(ymm3, ymm15, ymm11);

				ymm11 = _mm_add_ps(ymm8, ymm11);

				ymm8 = _mm_shuffle_ps(ymm11, ymm11, 14);
				ymm11 = _mm_add_ps(ymm11, ymm8);
				ymm8 = _mm_shuffle_ps(ymm11, ymm11, 1);
				ymm11 = _mm_add_ps(ymm11, ymm8);

				//-----------------------------------------

				__m128 ymm10 = _mm_blend_ps(ymm0, ymm4, 1);
				ymm10 = _mm_permute_ps(ymm10, 57);
				ymm8 = _mm_mul_ps(ymm10, ymm12);

				ymm10 = _mm_blend_ps(ymm1, ymm5, 1);
				ymm10 = _mm_permute_ps(ymm10, 57);
				ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm_blend_ps(ymm2, ymm6, 1);
				ymm10 = _mm_permute_ps(ymm10, 57);
				ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm_blend_ps(ymm3, ymm7, 1);
				ymm10 = _mm_permute_ps(ymm10, 57);
				ymm8 = _mm_fmadd_ps(ymm10, ymm15, ymm8);

				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 14);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 1);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 34);

				//-----------------------------------------

				ymm10 = _mm_blend_ps(ymm0, ymm4, 3);
				ymm10 = _mm_permute_ps(ymm10, 78);
				ymm8 = _mm_mul_ps(ymm10, ymm12);

				ymm10 = _mm_blend_ps(ymm1, ymm5, 3);
				ymm10 = _mm_permute_ps(ymm10, 78);
				ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm_blend_ps(ymm2, ymm6, 3);
				ymm10 = _mm_permute_ps(ymm10, 78);
				ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm_blend_ps(ymm3, ymm7, 3);
				ymm10 = _mm_permute_ps(ymm10, 78);
				ymm8 = _mm_fmadd_ps(ymm10, ymm15, ymm8);

				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 64);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 176);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 68);

				//-----------------------------------------

				ymm10 = _mm_blend_ps(ymm0, ymm4, 7);
				ymm10 = _mm_permute_ps(ymm10, 147);
				ymm8 = _mm_mul_ps(ymm10, ymm12);

				ymm10 = _mm_blend_ps(ymm1, ymm5, 7);
				ymm10 = _mm_permute_ps(ymm10, 147);
				ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm_blend_ps(ymm2, ymm6, 7);
				ymm10 = _mm_permute_ps(ymm10, 147);
				ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm_blend_ps(ymm3, ymm7, 7);
				ymm10 = _mm_permute_ps(ymm10, 147);
				ymm8 = _mm_fmadd_ps(ymm10, ymm15, ymm8);

				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 64);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 176);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 136);

				_mm_store_ps(pDst, ymm11);

				//============================================================================
			}
		}
		void CNNPP::conv_3x3(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { 1, 2, 3, 4, 5, 6, 7, 0 };

			if (L == 0) L = src_size_l - 2;
			if (H == 0) H = src_size_h - 2;

			__m128 ymm12 = _mm_load_ps(kernel);
			__m128 ymm13 = _mm_load_ps(kernel + REG_SIZE);
			__m128 ymm14 = _mm_load_ps(kernel + 2 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				__m128 ymm0 = _mm_load_ps(pSrc0);
				__m128 ymm1 = _mm_load_ps(pSrc1);
				__m128 ymm2 = _mm_load_ps(pSrc2);
				pSrc0 += REG_SIZE;
				pSrc1 += REG_SIZE;
				pSrc2 += REG_SIZE;

				size_t i = 0;
				for (; i < L - REG_SIZE; i += REG_SIZE)
				{
					__m128 ymm4 = _mm_load_ps(pSrc0);
					__m128 ymm5 = _mm_load_ps(pSrc1);
					__m128 ymm6 = _mm_load_ps(pSrc2);
					pSrc0 += REG_SIZE;
					pSrc1 += REG_SIZE;
					pSrc2 += REG_SIZE;

					__m128 ymm8 = _mm_mul_ps(ymm0, ymm12);
					ymm8 = _mm_fmadd_ps(ymm1, ymm13, ymm8);
					ymm8 = _mm_fmadd_ps(ymm2, ymm14, ymm8);

					__m128 ymm11 = _mm_shuffle_ps(ymm8, ymm8, 14);
					ymm8 = _mm_add_ps(ymm8, ymm11);
					ymm11 = _mm_shuffle_ps(ymm8, ymm8, 1);
					ymm11 = _mm_add_ps(ymm8, ymm11);

					//-----------------------------------------

					__m128 ymm10 = _mm_blend_ps(ymm0, ymm4, 1);
					ymm10 = _mm_permute_ps(ymm10, 57);
					ymm8 = _mm_mul_ps(ymm10, ymm12);

					ymm10 = _mm_blend_ps(ymm1, ymm5, 1);
					ymm10 = _mm_permute_ps(ymm10, 57);
					ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm_blend_ps(ymm2, ymm6, 1);
					ymm10 = _mm_permute_ps(ymm10, 57);
					ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 14);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 1);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 34);

					//-----------------------------------------

					ymm10 = _mm_blend_ps(ymm0, ymm4, 3);
					ymm10 = _mm_permute_ps(ymm10, 78);
					ymm8 = _mm_mul_ps(ymm10, ymm12);

					ymm10 = _mm_blend_ps(ymm1, ymm5, 3);
					ymm10 = _mm_permute_ps(ymm10, 78);
					ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm_blend_ps(ymm2, ymm6, 3);
					ymm10 = _mm_permute_ps(ymm10, 78);
					ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 64);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 176);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 68);

					//-----------------------------------------

					ymm10 = _mm_blend_ps(ymm0, ymm4, 7);
					ymm10 = _mm_permute_ps(ymm10, 147);
					ymm8 = _mm_mul_ps(ymm10, ymm12);

					ymm10 = _mm_blend_ps(ymm1, ymm5, 7);
					ymm10 = _mm_permute_ps(ymm10, 147);
					ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm_blend_ps(ymm2, ymm6, 7);
					ymm10 = _mm_permute_ps(ymm10, 147);
					ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 64);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm10 = _mm_shuffle_ps(ymm8, ymm8, 176);
					ymm8 = _mm_add_ps(ymm8, ymm10);
					ymm11 = _mm_blend_ps(ymm11, ymm8, 136);

					_mm_store_ps(pDst, ymm11);
					pDst += REG_SIZE;

					ymm0 = ymm4;
					ymm1 = ymm5;
					ymm2 = ymm6;
				}

				//============================================================================

				__m128 ymm4 = _mm_setzero_ps();
				__m128 ymm5 = _mm_setzero_ps();
				__m128 ymm6 = _mm_setzero_ps();

				if (i + REG_SIZE < src_size_l)
				{
					ymm4 = _mm_load_ps(pSrc0);
					ymm5 = _mm_load_ps(pSrc1);
					ymm6 = _mm_load_ps(pSrc2);
				}

				__m128 ymm8 = _mm_mul_ps(ymm0, ymm12);
				ymm8 = _mm_fmadd_ps(ymm1, ymm13, ymm8);
				ymm8 = _mm_fmadd_ps(ymm2, ymm14, ymm8);

				__m128 ymm11 = _mm_shuffle_ps(ymm8, ymm8, 14);
				ymm8 = _mm_add_ps(ymm8, ymm11);
				ymm11 = _mm_shuffle_ps(ymm8, ymm8, 1);
				ymm11 = _mm_add_ps(ymm8, ymm11);

				//-----------------------------------------

				__m128 ymm10 = _mm_blend_ps(ymm0, ymm4, 1);
				ymm10 = _mm_permute_ps(ymm10, 57);
				ymm8 = _mm_mul_ps(ymm10, ymm12);

				ymm10 = _mm_blend_ps(ymm1, ymm5, 1);
				ymm10 = _mm_permute_ps(ymm10, 57);
				ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm_blend_ps(ymm2, ymm6, 1);
				ymm10 = _mm_permute_ps(ymm10, 57);
				ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 14);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 1);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 34);

				//-----------------------------------------

				ymm10 = _mm_blend_ps(ymm0, ymm4, 3);
				ymm10 = _mm_permute_ps(ymm10, 78);
				ymm8 = _mm_mul_ps(ymm10, ymm12);

				ymm10 = _mm_blend_ps(ymm1, ymm5, 3);
				ymm10 = _mm_permute_ps(ymm10, 78);
				ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm_blend_ps(ymm2, ymm6, 3);
				ymm10 = _mm_permute_ps(ymm10, 78);
				ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 64);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 176);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 68);

				//-----------------------------------------

				ymm10 = _mm_blend_ps(ymm0, ymm4, 7);
				ymm10 = _mm_permute_ps(ymm10, 147);
				ymm8 = _mm_mul_ps(ymm10, ymm12);

				ymm10 = _mm_blend_ps(ymm1, ymm5, 7);
				ymm10 = _mm_permute_ps(ymm10, 147);
				ymm8 = _mm_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm_blend_ps(ymm2, ymm6, 7);
				ymm10 = _mm_permute_ps(ymm10, 147);
				ymm8 = _mm_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 64);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm10 = _mm_shuffle_ps(ymm8, ymm8, 176);
				ymm8 = _mm_add_ps(ymm8, ymm10);
				ymm11 = _mm_blend_ps(ymm11, ymm8, 136);

				_mm_store_ps(pDst, ymm11);

				//============================================================================
			}
		}
		void CNNPP::conv_8x7(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 6;
			if (H == 0) H = src_size_h - 7;

			const __m128 ymm8_1 = _mm_load_ps(kernel);
			const __m128 ymm8_2 = _mm_load_ps(kernel + REG_SIZE);
			const __m128 ymm9_1 = _mm_load_ps(kernel + 2 * REG_SIZE);
			const __m128 ymm9_2 = _mm_load_ps(kernel + 3 * REG_SIZE);
			const __m128 ymm10_1 = _mm_load_ps(kernel + 4 * REG_SIZE);
			const __m128 ymm10_2 = _mm_load_ps(kernel + 5 * REG_SIZE);
			const __m128 ymm11_1 = _mm_load_ps(kernel + 6 * REG_SIZE);
			const __m128 ymm11_2 = _mm_load_ps(kernel + 7 * REG_SIZE);
			const __m128 ymm12_1 = _mm_load_ps(kernel + 8 * REG_SIZE);
			const __m128 ymm12_2 = _mm_load_ps(kernel + 9 * REG_SIZE);
			const __m128 ymm13_1 = _mm_load_ps(kernel + 10 * REG_SIZE);
			const __m128 ymm13_2 = _mm_load_ps(kernel + 11 * REG_SIZE);
			const __m128 ymm14_1 = _mm_load_ps(kernel + 12 * REG_SIZE);
			const __m128 ymm14_2 = _mm_load_ps(kernel + 13 * REG_SIZE);
			const __m128 ymm15_1 = _mm_load_ps(kernel + 14 * REG_SIZE);
			const __m128 ymm15_2 = _mm_load_ps(kernel + 15 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pSrc4 = src + (j + 4) * src_size_l;
				float* __restrict pSrc5 = src + (j + 5) * src_size_l;
				float* __restrict pSrc6 = src + (j + 6) * src_size_l;
				float* __restrict pSrc7 = src + (j + 7) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				for (size_t i = 0; i < L; ++i)
				{
					__m128 ymm0_1 = _mm_loadu_ps(pSrc0);
					__m128 ymm0_2 = _mm_loadu_ps(REG_SIZE + pSrc0++);
					__m128 ymm1_1 = _mm_loadu_ps(pSrc1);
					__m128 ymm1_2 = _mm_loadu_ps(REG_SIZE + pSrc1++);
					__m128 ymm2_1 = _mm_loadu_ps(pSrc2);
					__m128 ymm2_2 = _mm_loadu_ps(REG_SIZE + pSrc2++);
					__m128 ymm3_1 = _mm_loadu_ps(pSrc3);
					__m128 ymm3_2 = _mm_loadu_ps(REG_SIZE + pSrc3++);

					__m128 ymm7_1 = _mm_mul_ps(ymm0_1, ymm8_1);
					__m128 ymm6_1 = _mm_mul_ps(ymm1_1, ymm9_1);

					ymm7_1 = _mm_fmadd_ps(ymm2_1, ymm10_1, ymm7_1);
					ymm6_1 = _mm_fmadd_ps(ymm3_1, ymm11_1, ymm6_1);

					__m128 ymm7_2 = _mm_mul_ps(ymm0_2, ymm8_2);
					__m128 ymm6_2 = _mm_mul_ps(ymm1_2, ymm9_2);

					ymm7_2 = _mm_fmadd_ps(ymm2_2, ymm10_2, ymm7_2);
					ymm6_2 = _mm_fmadd_ps(ymm3_2, ymm11_2, ymm6_2);

					ymm0_1 = _mm_loadu_ps(pSrc4);
					ymm0_2 = _mm_loadu_ps(REG_SIZE + pSrc4++);
					ymm1_1 = _mm_loadu_ps(pSrc5);
					ymm1_2 = _mm_loadu_ps(REG_SIZE + pSrc5++);
					ymm2_1 = _mm_loadu_ps(pSrc6);
					ymm2_2 = _mm_loadu_ps(REG_SIZE + pSrc6++);
					ymm3_1 = _mm_loadu_ps(pSrc7);
					ymm3_2 = _mm_loadu_ps(REG_SIZE + pSrc7++);

					ymm7_1 = _mm_fmadd_ps(ymm0_1, ymm12_1, ymm7_1);
					ymm6_1 = _mm_fmadd_ps(ymm1_1, ymm13_1, ymm6_1);
					ymm7_1 = _mm_fmadd_ps(ymm2_1, ymm14_1, ymm7_1);
					ymm6_1 = _mm_fmadd_ps(ymm3_1, ymm15_1, ymm6_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);

					ymm7_2 = _mm_fmadd_ps(ymm0_2, ymm12_2, ymm7_2);
					ymm6_2 = _mm_fmadd_ps(ymm1_2, ymm13_2, ymm6_2);
					ymm7_2 = _mm_fmadd_ps(ymm2_2, ymm14_2, ymm7_2);
					ymm6_2 = _mm_fmadd_ps(ymm3_2, ymm15_2, ymm6_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);

					ymm7_1 = _mm_add_ps(ymm7_1, ymm7_2);

					ymm6_1 = _mm_shuffle_ps(ymm7_1, ymm7_1, 14);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_shuffle_ps(ymm7_1, ymm7_1, 1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);

					_mm_store_ss(pDst++, ymm7_1);
				}
			}
		}
		void CNNPP::conv_6x5(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 4;
			if (H == 0) H = src_size_h - 5;

			const __m128 ymm8_1 = _mm_load_ps(kernel);
			const __m128 ymm8_2 = _mm_load_ps(kernel + REG_SIZE);
			const __m128 ymm9_1 = _mm_load_ps(kernel + 2 * REG_SIZE);
			const __m128 ymm9_2 = _mm_load_ps(kernel + 3 * REG_SIZE);
			const __m128 ymm10_1 = _mm_load_ps(kernel + 4 * REG_SIZE);
			const __m128 ymm10_2 = _mm_load_ps(kernel + 5 * REG_SIZE);
			const __m128 ymm11_1 = _mm_load_ps(kernel + 6 * REG_SIZE);
			const __m128 ymm11_2 = _mm_load_ps(kernel + 7 * REG_SIZE);
			const __m128 ymm12_1 = _mm_load_ps(kernel + 8 * REG_SIZE);
			const __m128 ymm12_2 = _mm_load_ps(kernel + 9 * REG_SIZE);
			const __m128 ymm13_1 = _mm_load_ps(kernel + 10 * REG_SIZE);
			const __m128 ymm13_2 = _mm_load_ps(kernel + 11 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pSrc4 = src + (j + 4) * src_size_l;
				float* __restrict pSrc5 = src + (j + 5) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				for (size_t i = 0; i < L; ++i)
				{
					__m128 ymm0_1 = _mm_loadu_ps(pSrc0);
					__m128 ymm0_2 = _mm_loadu_ps(REG_SIZE + pSrc0++);
					__m128 ymm1_1 = _mm_loadu_ps(pSrc1);
					__m128 ymm1_2 = _mm_loadu_ps(REG_SIZE + pSrc1++);
					__m128 ymm2_1 = _mm_loadu_ps(pSrc2);
					__m128 ymm2_2 = _mm_loadu_ps(REG_SIZE + pSrc2++);
					__m128 ymm3_1 = _mm_loadu_ps(pSrc3);
					__m128 ymm3_2 = _mm_loadu_ps(REG_SIZE + pSrc3++);

					__m128 ymm7_1 = _mm_mul_ps(ymm0_1, ymm8_1);
					__m128 ymm6_1 = _mm_mul_ps(ymm1_1, ymm9_1);

					ymm7_1 = _mm_fmadd_ps(ymm2_1, ymm10_1, ymm7_1);
					ymm6_1 = _mm_fmadd_ps(ymm3_1, ymm11_1, ymm6_1);

					__m128 ymm7_2 = _mm_mul_ps(ymm0_2, ymm8_2);
					__m128 ymm6_2 = _mm_mul_ps(ymm1_2, ymm9_2);

					ymm7_2 = _mm_fmadd_ps(ymm2_2, ymm10_2, ymm7_2);
					ymm6_2 = _mm_fmadd_ps(ymm3_2, ymm11_2, ymm6_2);

					ymm0_1 = _mm_loadu_ps(pSrc4);
					ymm0_2 = _mm_loadu_ps(REG_SIZE + pSrc4++);
					ymm1_1 = _mm_loadu_ps(pSrc5);
					ymm1_2 = _mm_loadu_ps(REG_SIZE + pSrc5++);

					ymm7_1 = _mm_fmadd_ps(ymm0_1, ymm12_1, ymm7_1);
					ymm6_1 = _mm_fmadd_ps(ymm1_1, ymm13_1, ymm6_1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);

					ymm7_2 = _mm_fmadd_ps(ymm0_2, ymm12_2, ymm7_2);
					ymm6_2 = _mm_fmadd_ps(ymm1_2, ymm13_2, ymm6_2);
					ymm7_2 = _mm_add_ps(ymm7_2, ymm6_2);

					ymm7_1 = _mm_add_ps(ymm7_1, ymm7_2);

					ymm6_1 = _mm_shuffle_ps(ymm7_1, ymm7_1, 14);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);
					ymm6_1 = _mm_shuffle_ps(ymm7_1, ymm7_1, 1);
					ymm7_1 = _mm_add_ps(ymm7_1, ymm6_1);

					_mm_store_ss(pDst++, ymm7_1);
				}
			}
		}

#endif

		void CNNPP::tanh_avr_tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale)
		{
			ALIGN(ALIGN_DEF) float buff[2 * REG_SIZE];

			const __m128 ymm14 = _mm_set1_ps(*(float*)&abs_mask);
			const __m128 ymm15 = _mm_set1_ps(one);
			const __m128 ymm7 = _mm_set1_ps(tanh_a);
			const __m128 ymm1 = _mm_set1_ps(*conv_b);
			const __m128 ymm5 = _mm_set1_ps(*subs_w);
			__m128 ymm0 = _mm_set1_ps(*subs_b);
			__m128 ymm2 = _mm_set1_ps(*scale);
			_mm_store_ps(buff, ymm0);
			_mm_store_ps(buff + REG_SIZE, ymm2);

			size_t j2 = 0;
			for (size_t j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + j2 * dst_size_l;

				for (size_t i = 0; i < src_size_l; i += 2 * REG_SIZE)
				{
					ymm0 = _mm_load_ps(pSrc0 + i);

					__m128 ymm6 = _mm_load_ps(pSrc0 + i + REG_SIZE);
					ymm0 = _mm_add_ps(ymm0, ymm1);
					__m128 ymm4 = _mm_mul_ps(ymm0, ymm0);
					__m128 ymm3 = _mm_and_ps(ymm14, ymm0);

					ymm6 = _mm_add_ps(ymm6, ymm1);
					__m128 ymm12 = _mm_mul_ps(ymm4, ymm4);

					ymm3 = _mm_add_ps(ymm15, ymm3);
					ymm2 = _mm_andnot_ps(ymm14, ymm0);

					__m128 ymm10 = _mm_mul_ps(ymm6, ymm6);
					ymm3 = _mm_add_ps(ymm3, ymm4);
					__m128 ymm9 = _mm_and_ps(ymm14, ymm6);
					ymm0 = _mm_load_ps(pSrc1 + i);

					ymm0 = _mm_add_ps(ymm0, ymm1);
					__m128 ymm11 = _mm_mul_ps(ymm10, ymm10);

#ifdef USE_FMA
					ymm3 = _mm_fmadd_ps(ymm12, ymm7, ymm3);
#else
					ymm12 = _mm_mul_ps(ymm12, ymm7);
					ymm3 = _mm_add_ps(ymm3, ymm12);
#endif

					__m128 ymm8 = _mm_andnot_ps(ymm14, ymm6);

#ifdef USE_FAST_DIV
					ymm3 = _mm_rcp_ps(ymm3);
#else
					ymm3 = _mm_div_ps(ymm15, ymm3);
#endif	

					ymm9 = _mm_add_ps(ymm15, ymm9);
					ymm6 = _mm_load_ps(pSrc1 + i + REG_SIZE);

					ymm6 = _mm_add_ps(ymm6, ymm1);

					ymm4 = _mm_mul_ps(ymm0, ymm0);
					ymm3 = _mm_sub_ps(ymm15, ymm3);

					ymm9 = _mm_add_ps(ymm9, ymm10);
					ymm12 = _mm_or_ps(ymm3, ymm2);

					ymm10 = _mm_mul_ps(ymm6, ymm6);

#ifdef USE_FMA
					ymm9 = _mm_fmadd_ps(ymm11, ymm7, ymm9);
#else
					ymm11 = _mm_mul_ps(ymm11, ymm7);
					ymm9 = _mm_add_ps(ymm9, ymm11);
#endif				

					ymm3 = _mm_and_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm9 = _mm_rcp_ps(ymm9);
#else
					ymm9 = _mm_div_ps(ymm15, ymm9);
#endif	

					ymm9 = _mm_sub_ps(ymm15, ymm9);
					ymm2 = _mm_andnot_ps(ymm14, ymm0);

					ymm0 = _mm_mul_ps(ymm4, ymm4);


					ymm3 = _mm_add_ps(ymm15, ymm3);
					__m128 ymm13 = _mm_or_ps(ymm9, ymm8);

					ymm11 = _mm_mul_ps(ymm10, ymm10);

					ymm3 = _mm_add_ps(ymm3, ymm4);
					ymm8 = _mm_andnot_ps(ymm14, ymm6);

					ymm4 = _mm_load_ps(buff);
					ymm9 = _mm_and_ps(ymm14, ymm6);

#ifdef USE_FMA
					ymm3 = _mm_fmadd_ps(ymm0, ymm7, ymm3);
#else
					ymm0 = _mm_mul_ps(ymm0, ymm7);
					ymm3 = _mm_add_ps(ymm3, ymm0);
#endif

#ifdef USE_FAST_DIV
					ymm3 = _mm_rcp_ps(ymm3);
#else
					ymm3 = _mm_div_ps(ymm15, ymm3);
#endif	

					ymm9 = _mm_add_ps(ymm15, ymm9);

					ymm3 = _mm_sub_ps(ymm15, ymm3);

					ymm9 = _mm_add_ps(ymm9, ymm10);
					ymm3 = _mm_or_ps(ymm3, ymm2);

#ifdef USE_FMA
					ymm9 = _mm_fmadd_ps(ymm11, ymm7, ymm9);
#else
					ymm11 = _mm_mul_ps(ymm11, ymm7);
					ymm9 = _mm_add_ps(ymm9, ymm11);
#endif

#ifdef USE_FAST_DIV
					ymm9 = _mm_rcp_ps(ymm9);
#else
					ymm9 = _mm_div_ps(ymm15, ymm9);
#endif	

					ymm9 = _mm_sub_ps(ymm15, ymm9);
					ymm9 = _mm_or_ps(ymm9, ymm8);

					//-----------------------------

					ymm3 = _mm_add_ps(ymm3, ymm12);
					ymm9 = _mm_add_ps(ymm9, ymm13);

					ymm13 = _mm_load_ps(buff + REG_SIZE);
					//ymm6 = _mm_permute2f128_ps(ymm3, ymm9, 32);
					//ymm8 = _mm_permute2f128_ps(ymm9, ymm3, 19);

					//ymm2 = _mm_shuffle_ps(ymm6, ymm8, 221);
					//ymm3 = _mm_shuffle_ps(ymm6, ymm8, 136);		
					ymm2 = _mm_shuffle_ps(ymm3, ymm9, 221);
					ymm3 = _mm_shuffle_ps(ymm3, ymm9, 136);
					ymm0 = _mm_add_ps(ymm2, ymm3);
					//ymm0 = _mm_hadd_ps(ymm3, ymm9)

					//-----------------------------

					ymm0 = _mm_mul_ps(ymm0, ymm5);
					ymm0 = _mm_add_ps(ymm0, ymm4);

					//-----------------------------

					ymm3 = _mm_and_ps(ymm14, ymm0);

					ymm3 = _mm_add_ps(ymm15, ymm3);
					ymm4 = _mm_mul_ps(ymm0, ymm0);

					ymm3 = _mm_add_ps(ymm3, ymm4);
					ymm8 = _mm_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
					ymm3 = _mm_fmadd_ps(ymm8, ymm7, ymm3);
#else
					ymm8 = _mm_mul_ps(ymm8, ymm7);
					ymm3 = _mm_add_ps(ymm3, ymm8);
#endif	

					ymm2 = _mm_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm3 = _mm_rcp_ps(ymm3);
#else
					ymm3 = _mm_div_ps(ymm15, ymm3);
#endif	

					ymm3 = _mm_sub_ps(ymm15, ymm3);
					ymm3 = _mm_or_ps(ymm3, ymm2);

					//-----------------------------

					ymm3 = _mm_mul_ps(ymm3, ymm13);

					_mm_store_ps(pDst, ymm3);
					pDst += REG_SIZE;
				}
				j2++;
			}
		}
		void CNNPP::max_tanh_tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale)
		{
			const __m128 ymm14 = _mm_set1_ps(*(float*)&abs_mask);
			const __m128 ymm15 = _mm_set1_ps(one);
			const __m128 ymm7 = _mm_set1_ps(tanh_a);
			const __m128 ymm1 = _mm_set1_ps(*conv_b);
			const __m128 ymm5 = _mm_set1_ps(*subs_w);
			const __m128 ymm4 = _mm_set1_ps(*subs_b);
			const __m128 ymm11 = _mm_set1_ps(*scale);

			size_t j2 = 0;
			for (size_t j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + j2 * dst_size_l;

				for (size_t i = 0; i < src_size_l; i += 2 * REG_SIZE)
				{
					__m128 ymm3 = _mm_load_ps(pSrc0 + i);
					__m128 ymm9 = _mm_load_ps(pSrc0 + i + REG_SIZE);

					__m128 ymm12 = _mm_load_ps(pSrc1 + i);
					__m128 ymm13 = _mm_load_ps(pSrc1 + i + REG_SIZE);

					//-----------------------------

					ymm3 = _mm_max_ps(ymm3, ymm12);
					ymm9 = _mm_max_ps(ymm9, ymm13);

					//__m128 ymm6 = _mm_permute2f128_ps(ymm3, ymm9, 32);
					//__m128 ymm8 = _mm_permute2f128_ps(ymm9, ymm3, 19);

					//ymm3 = _mm_shuffle_ps(ymm6, ymm8, 221);
					//ymm9 = _mm_shuffle_ps(ymm6, ymm8, 136);
					__m128 ymm6 = _mm_shuffle_ps(ymm3, ymm9, 221);
					__m128 ymm8 = _mm_shuffle_ps(ymm3, ymm9, 136);
					__m128 ymm0 = _mm_max_ps(ymm6, ymm8);

					//-----------------------------

					ymm0 = _mm_add_ps(ymm0, ymm1);
					ymm3 = _mm_and_ps(ymm14, ymm0);

					ymm3 = _mm_add_ps(ymm15, ymm3);
					ymm6 = _mm_mul_ps(ymm0, ymm0);

					ymm3 = _mm_add_ps(ymm3, ymm6);
					ymm8 = _mm_mul_ps(ymm6, ymm6);

#ifdef USE_FMA
					ymm3 = _mm_fmadd_ps(ymm8, ymm7, ymm3);
#else
					ymm8 = _mm_mul_ps(ymm8, ymm7);
					ymm3 = _mm_add_ps(ymm3, ymm8);
#endif

					__m128 ymm2 = _mm_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm3 = _mm_rcp_ps(ymm3);
#else
					ymm3 = _mm_div_ps(ymm15, ymm3);
#endif	

					ymm3 = _mm_sub_ps(ymm15, ymm3);
					ymm0 = _mm_or_ps(ymm3, ymm2);

					//-----------------------------

#ifdef USE_FMA
					ymm0 = _mm_fmadd_ps(ymm0, ymm5, ymm4);
#else
					ymm0 = _mm_mul_ps(ymm0, ymm5);
					ymm0 = _mm_add_ps(ymm0, ymm4);
#endif

					//-----------------------------

					ymm3 = _mm_and_ps(ymm14, ymm0);

					ymm3 = _mm_add_ps(ymm15, ymm3);
					ymm6 = _mm_mul_ps(ymm0, ymm0);

					ymm3 = _mm_add_ps(ymm3, ymm6);
					ymm8 = _mm_mul_ps(ymm6, ymm6);

#ifdef USE_FMA
					ymm3 = _mm_fmadd_ps(ymm8, ymm7, ymm3);
#else
					ymm8 = _mm_mul_ps(ymm8, ymm7);
					ymm3 = _mm_add_ps(ymm3, ymm8);
#endif

					ymm2 = _mm_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm3 = _mm_rcp_ps(ymm3);
#else
					ymm3 = _mm_div_ps(ymm15, ymm3);
#endif	

					ymm3 = _mm_sub_ps(ymm15, ymm3);
					ymm3 = _mm_or_ps(ymm3, ymm2);

					//-----------------------------

					ymm3 = _mm_mul_ps(ymm3, ymm11);

					_mm_store_ps(pDst, ymm3);
					pDst += REG_SIZE;
				}
				j2++;
			}
		}

		void CNNPP::tanh_tanh_2tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale, float* __restrict snn_hl_w0, float* __restrict snn_hl_b0, float* __restrict snn_hl_w1, float* __restrict snn_hl_b1, float* __restrict snn_ol_w0, float* __restrict snn_ol_w1)
		{
			ALIGN(ALIGN_DEF) float buff[6 * REG_SIZE];

			const __m128 ymm14 = _mm_set1_ps(*(float*)&abs_mask);
			const __m128 ymm15 = _mm_set1_ps(one);
			const __m128 ymm13 = _mm_set1_ps(tanh_a);
			const __m128 ymm12 = _mm_set1_ps(*conv_b);
			const __m128 ymm11 = _mm_set1_ps(*subs_w);
			const __m128 ymm10 = _mm_set1_ps(*subs_b);
			const __m128 ymm9 = _mm_set1_ps(*scale);

			__m128 ymm8 = _mm_set1_ps(*snn_hl_w0);
			__m128 ymm7 = _mm_set1_ps(*snn_hl_b0);
			__m128 ymm6 = _mm_set1_ps(*snn_hl_w1);
			__m128 ymm5 = _mm_set1_ps(*snn_hl_b1);
			__m128 ymm4 = _mm_set1_ps(*snn_ol_w0);
			__m128 ymm3 = _mm_set1_ps(*snn_ol_w1);

			ymm8 = _mm_mul_ps(ymm8, ymm9);
			ymm6 = _mm_mul_ps(ymm6, ymm9);
			ymm4 = _mm_mul_ps(ymm4, ymm9);
			ymm3 = _mm_mul_ps(ymm3, ymm9);

			_mm_store_ps(buff, ymm8);
			_mm_store_ps(buff + REG_SIZE, ymm7);
			_mm_store_ps(buff + 2 * REG_SIZE, ymm6);
			_mm_store_ps(buff + 3 * REG_SIZE, ymm5);
			_mm_store_ps(buff + 4 * REG_SIZE, ymm4);
			_mm_store_ps(buff + 5 * REG_SIZE, ymm3);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m128 ymm0 = _mm_load_ps(src + i);
				ymm0 = _mm_add_ps(ymm0, ymm12);
				__m128 ymm1 = _mm_and_ps(ymm14, ymm0);

				ymm3 = _mm_add_ps(ymm15, ymm1);
				ymm4 = _mm_mul_ps(ymm0, ymm0);

				ymm3 = _mm_add_ps(ymm3, ymm4);
				ymm5 = _mm_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm_fmadd_ps(ymm5, ymm13, ymm3);
#else
				ymm5 = _mm_mul_ps(ymm5, ymm13);
				ymm3 = _mm_add_ps(ymm3, ymm5);
#endif	

				__m128 ymm2 = _mm_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm_rcp_ps(ymm3);
#else
				ymm3 = _mm_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm_sub_ps(ymm15, ymm3);
				ymm3 = _mm_or_ps(ymm3, ymm2);

#ifdef USE_FMA
				ymm0 = _mm_fmadd_ps(ymm3, ymm11, ymm10);
#else
				ymm3 = _mm_mul_ps(ymm3, ymm11);
				ymm0 = _mm_add_ps(ymm3, ymm10);
#endif	

				ymm1 = _mm_and_ps(ymm14, ymm0);

				ymm3 = _mm_add_ps(ymm15, ymm1);
				ymm4 = _mm_mul_ps(ymm0, ymm0);

				ymm3 = _mm_add_ps(ymm3, ymm4);
				ymm5 = _mm_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm_fmadd_ps(ymm5, ymm13, ymm3);
#else
				ymm5 = _mm_mul_ps(ymm5, ymm13);
				ymm3 = _mm_add_ps(ymm3, ymm5);
#endif

				ymm2 = _mm_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm_rcp_ps(ymm3);
#else
				ymm3 = _mm_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm_sub_ps(ymm15, ymm3);
				ymm6 = _mm_or_ps(ymm3, ymm2);

#ifdef USE_FMA
				ymm0 = _mm_fmadd_ps(ymm6, _mm_load_ps(buff), _mm_load_ps(buff + REG_SIZE));
#else
				ymm0 = _mm_mul_ps(ymm6, _mm_load_ps(buff));
				ymm0 = _mm_add_ps(ymm0, _mm_load_ps(buff + REG_SIZE));
#endif

				ymm1 = _mm_and_ps(ymm14, ymm0);

				ymm3 = _mm_add_ps(ymm15, ymm1);
				ymm4 = _mm_mul_ps(ymm0, ymm0);

				ymm3 = _mm_add_ps(ymm3, ymm4);
				ymm5 = _mm_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm_fmadd_ps(ymm5, ymm13, ymm3);
#else
				ymm5 = _mm_mul_ps(ymm5, ymm13);
				ymm3 = _mm_add_ps(ymm3, ymm5);
#endif

				ymm2 = _mm_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm_rcp_ps(ymm3);
#else
				ymm3 = _mm_div_ps(ymm15, ymm3);
#endif

				ymm3 = _mm_sub_ps(ymm15, ymm3);
				ymm7 = _mm_or_ps(ymm3, ymm2);

#ifdef USE_FMA
				ymm0 = _mm_fmadd_ps(ymm6, _mm_load_ps(buff + 2 * REG_SIZE), _mm_load_ps(buff + 3 * REG_SIZE));
#else
				ymm0 = _mm_mul_ps(ymm6, _mm_load_ps(buff + 2 * REG_SIZE));
				ymm0 = _mm_add_ps(ymm0, _mm_load_ps(buff + 3 * REG_SIZE));
#endif

				ymm1 = _mm_and_ps(ymm14, ymm0);

				ymm3 = _mm_add_ps(ymm15, ymm1);
				ymm4 = _mm_mul_ps(ymm0, ymm0);

				ymm3 = _mm_add_ps(ymm3, ymm4);
				ymm5 = _mm_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm_fmadd_ps(ymm5, ymm13, ymm3);
#else
				ymm5 = _mm_mul_ps(ymm5, ymm13);
				ymm3 = _mm_add_ps(ymm3, ymm5);
#endif

				ymm2 = _mm_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm_rcp_ps(ymm3);
#else
				ymm3 = _mm_div_ps(ymm15, ymm3);
#endif

				ymm3 = _mm_sub_ps(ymm15, ymm3);
				ymm8 = _mm_or_ps(ymm3, ymm2);

				ymm7 = _mm_mul_ps(ymm7, _mm_load_ps(buff + 4 * REG_SIZE));

#ifdef USE_FMA
				ymm8 = _mm_fmadd_ps(ymm8, _mm_load_ps(buff + 5 * REG_SIZE), ymm7);
#else
				ymm8 = _mm_mul_ps(ymm8, _mm_load_ps(buff + 5 * REG_SIZE));
				ymm8 = _mm_add_ps(ymm8, ymm7);
#endif

				_mm_store_ps(dst + i, ymm8);
			}
		}
		void CNNPP::tanh_tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale)
		{
			const __m128 ymm14 = _mm_set1_ps(*(float*)&abs_mask);
			const __m128 ymm15 = _mm_set1_ps(one);
			const __m128 ymm13 = _mm_set1_ps(tanh_a);
			const __m128 ymm12 = _mm_set1_ps(*conv_b);
			const __m128 ymm11 = _mm_set1_ps(*subs_w);
			const __m128 ymm10 = _mm_set1_ps(*subs_b);
			const __m128 ymm9 = _mm_set1_ps(*scale);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m128 ymm0 = _mm_load_ps(src + i);
				ymm0 = _mm_add_ps(ymm0, ymm12);
				__m128 ymm1 = _mm_and_ps(ymm14, ymm0);

				__m128 ymm3 = _mm_add_ps(ymm15, ymm1);
				__m128 ymm4 = _mm_mul_ps(ymm0, ymm0);

				ymm3 = _mm_add_ps(ymm3, ymm4);
				__m128 ymm5 = _mm_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm_fmadd_ps(ymm5, ymm13, ymm3);
#else	
				ymm5 = _mm_mul_ps(ymm5, ymm13);
				ymm3 = _mm_add_ps(ymm3, ymm5);
#endif

				__m128 ymm2 = _mm_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm_rcp_ps(ymm3);
#else
				ymm3 = _mm_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm_sub_ps(ymm15, ymm3);
				ymm3 = _mm_or_ps(ymm3, ymm2);

#ifdef USE_FMA
				ymm0 = _mm_fmadd_ps(ymm3, ymm11, ymm10);
#else	
				ymm3 = _mm_mul_ps(ymm3, ymm11);
				ymm0 = _mm_add_ps(ymm3, ymm10);
#endif

				ymm1 = _mm_and_ps(ymm14, ymm0);

				ymm3 = _mm_add_ps(ymm15, ymm1);
				ymm4 = _mm_mul_ps(ymm0, ymm0);

				ymm3 = _mm_add_ps(ymm3, ymm4);
				ymm5 = _mm_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm_fmadd_ps(ymm5, ymm13, ymm3);
#else	
				ymm5 = _mm_mul_ps(ymm5, ymm13);
				ymm3 = _mm_add_ps(ymm3, ymm5);
#endif

				ymm2 = _mm_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm_rcp_ps(ymm3);
#else
				ymm3 = _mm_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm_sub_ps(ymm15, ymm3);
				ymm3 = _mm_or_ps(ymm3, ymm2);

				__m128 ymm6 = _mm_mul_ps(ymm3, ymm9);

				_mm_store_ps(dst + i, ymm6);
			}
		}
		void CNNPP::tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict snn_ol_b, float* __restrict scale)
		{
			const __m128 ymm14 = _mm_set1_ps(*(float*)&abs_mask);
			const __m128 ymm15 = _mm_set1_ps(one);
			const __m128 ymm13 = _mm_set1_ps(tanh_a);
			const __m128 ymm12 = _mm_set1_ps(*snn_ol_b);
			const __m128 ymm11 = _mm_set1_ps(*scale);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m128 ymm0 = _mm_load_ps(src + i);
				ymm0 = _mm_add_ps(ymm0, ymm12);

				__m128 ymm1 = _mm_and_ps(ymm14, ymm0);
				__m128 ymm3 = _mm_add_ps(ymm15, ymm1);
				__m128 ymm4 = _mm_mul_ps(ymm0, ymm0);

				ymm3 = _mm_add_ps(ymm3, ymm4);
				__m128 ymm5 = _mm_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm_fmadd_ps(ymm5, ymm13, ymm3);
#else	
				ymm5 = _mm_mul_ps(ymm5, ymm13);
				ymm3 = _mm_add_ps(ymm3, ymm5);
#endif

				__m128 ymm2 = _mm_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm_rcp_ps(ymm3);
#else
				ymm3 = _mm_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm_sub_ps(ymm15, ymm3);
				ymm3 = _mm_or_ps(ymm3, ymm2);

				ymm3 = _mm_mul_ps(ymm3, ymm11);

				_mm_store_ps(dst + i, ymm3);
			}
		}

		void CNNPP::add(float* __restrict dst, float* __restrict src1, float* __restrict src2, int size_)
		{
			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m128 ymm0 = _mm_load_ps(src1 + i);
				__m128 ymm1 = _mm_load_ps(src2 + i);
				ymm0 = _mm_add_ps(ymm0, ymm1);
				_mm_store_ps(dst + i, ymm0);
			}
		}
		void CNNPP::add2(float* __restrict dst, float* __restrict src1, float* __restrict src2, float* __restrict src3, int size_)
		{
			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m128 ymm0 = _mm_load_ps(src1 + i);
				__m128 ymm1 = _mm_load_ps(src2 + i);
				__m128 ymm2 = _mm_load_ps(src3 + i);
				ymm0 = _mm_add_ps(ymm0, ymm1);
				ymm0 = _mm_add_ps(ymm0, ymm2);
				_mm_store_ps(dst + i, ymm0);
			}
		}

		void CNNPP::mulC(float* __restrict dst, float* __restrict src_mulC, int size_, float* __restrict snn_ol_w)
		{
			const __m128 ymm1 = _mm_set1_ps(*snn_ol_w);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m128 ymm0 = _mm_load_ps(src_mulC + i);
				__m128 ymm2 = _mm_mul_ps(ymm0, ymm1);
				_mm_store_ps(dst + i, ymm2);
			}
		}
		void CNNPP::mulC1_add(float* __restrict dst, float* __restrict src1_mulC, float* __restrict src2, int size_, float* __restrict snn_hl_w)
		{
			const __m128 ymm3 = _mm_set1_ps(*snn_hl_w);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m128 ymm0 = _mm_load_ps(src1_mulC + i);
				__m128 ymm1 = _mm_load_ps(src2 + i);

#ifdef USE_FMA
				ymm0 = _mm_fmadd_ps(ymm0, ymm3, ymm1);
#else
				ymm0 = _mm_mul_ps(ymm0, ymm3);
				ymm0 = _mm_add_ps(ymm0, ymm1);
#endif

				_mm_store_ps(dst + i, ymm0);
			}
		}
		void CNNPP::mulC2_add(float* __restrict dst, float* __restrict src1_mulC0, float* __restrict src2_mulC1, int size_, float* __restrict snn_hl_w0, float* __restrict snn_hl_w1)
		{
			const __m128 ymm3 = _mm_set1_ps(*snn_hl_w0);
			const __m128 ymm4 = _mm_set1_ps(*snn_hl_w1);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m128 ymm0 = _mm_load_ps(src1_mulC0 + i);
				__m128 ymm1 = _mm_load_ps(src2_mulC1 + i);

#ifdef USE_FMA
				ymm0 = _mm_mul_ps(ymm0, ymm3);
				ymm0 = _mm_fmadd_ps(ymm1, ymm4, ymm0);
#else
				ymm0 = _mm_mul_ps(ymm0, ymm3);
				ymm1 = _mm_mul_ps(ymm1, ymm4);
				ymm0 = _mm_add_ps(ymm0, ymm1);
#endif

				_mm_store_ps(dst + i, ymm0);
			}
		}

#ifndef USE_FMA

		void CNNPP::mulC24_add_tanh(float* __restrict dst, float* __restrict* src, int size_, float* __restrict snn_hl_w, float* __restrict snn_hl_b, float* __restrict scale, float* __restrict snn_ol_w)
		{
			__m128 ymm0 = _mm_set1_ps(*snn_hl_b);
			__m128 ymm9 = _mm_set1_ps(*snn_ol_w);

			__m128 ymm11 = _mm_set1_ps(*scale);
			ymm9 = _mm_mul_ps(ymm9, ymm11);

			for (size_t i = 0; i < size_; i += 2 * REG_SIZE)
			{
				__m128 ymm14 = ymm0;
				__m128 ymm15 = ymm0;

				//4
				//-----------------------------------

				__m128 ymm10 = _mm_set1_ps(*snn_hl_w);
				ymm11 = _mm_set1_ps(*(snn_hl_w + 1));
				__m128 ymm12 = _mm_set1_ps(*(snn_hl_w + 2));
				__m128 ymm13 = _mm_set1_ps(*(snn_hl_w + 3));

				__m128 ymm1 = _mm_load_ps(src[0] + i);
				__m128 ymm2 = _mm_load_ps(src[0] + REG_SIZE + i);
				__m128 ymm3 = _mm_load_ps(src[1] + i);
				__m128 ymm4 = _mm_load_ps(src[1] + REG_SIZE + i);
				__m128 ymm5 = _mm_load_ps(src[2] + i);
				__m128 ymm6 = _mm_load_ps(src[2] + REG_SIZE + i);
				__m128 ymm7 = _mm_load_ps(src[3] + i);
				__m128 ymm8 = _mm_load_ps(src[3] + REG_SIZE + i);

				ymm1 = _mm_mul_ps(ymm1, ymm10);
				ymm14 = _mm_add_ps(ymm14, ymm1);
				ymm2 = _mm_mul_ps(ymm2, ymm10);
				ymm15 = _mm_add_ps(ymm15, ymm2);

				ymm3 = _mm_mul_ps(ymm3, ymm11);
				ymm14 = _mm_add_ps(ymm14, ymm3);
				ymm4 = _mm_mul_ps(ymm4, ymm11);
				ymm15 = _mm_add_ps(ymm15, ymm4);

				ymm5 = _mm_mul_ps(ymm5, ymm12);
				ymm14 = _mm_add_ps(ymm14, ymm5);
				ymm6 = _mm_mul_ps(ymm6, ymm12);
				ymm15 = _mm_add_ps(ymm15, ymm6);

				ymm7 = _mm_mul_ps(ymm7, ymm13);
				ymm14 = _mm_add_ps(ymm14, ymm7);
				ymm8 = _mm_mul_ps(ymm8, ymm13);
				ymm15 = _mm_add_ps(ymm15, ymm8);

				//8
				//-----------------------------------

				ymm10 = _mm_set1_ps(*(snn_hl_w + 4));
				ymm11 = _mm_set1_ps(*(snn_hl_w + 5));
				ymm12 = _mm_set1_ps(*(snn_hl_w + 6));
				ymm13 = _mm_set1_ps(*(snn_hl_w + 7));

				ymm1 = _mm_load_ps(src[4] + i);
				ymm2 = _mm_load_ps(src[4] + REG_SIZE + i);
				ymm3 = _mm_load_ps(src[5] + i);
				ymm4 = _mm_load_ps(src[5] + REG_SIZE + i);
				ymm5 = _mm_load_ps(src[6] + i);
				ymm6 = _mm_load_ps(src[6] + REG_SIZE + i);
				ymm7 = _mm_load_ps(src[7] + i);
				ymm8 = _mm_load_ps(src[7] + REG_SIZE + i);

				ymm1 = _mm_mul_ps(ymm1, ymm10);
				ymm14 = _mm_add_ps(ymm14, ymm1);
				ymm2 = _mm_mul_ps(ymm2, ymm10);
				ymm15 = _mm_add_ps(ymm15, ymm2);

				ymm3 = _mm_mul_ps(ymm3, ymm11);
				ymm14 = _mm_add_ps(ymm14, ymm3);
				ymm4 = _mm_mul_ps(ymm4, ymm11);
				ymm15 = _mm_add_ps(ymm15, ymm4);

				ymm5 = _mm_mul_ps(ymm5, ymm12);
				ymm14 = _mm_add_ps(ymm14, ymm5);
				ymm6 = _mm_mul_ps(ymm6, ymm12);
				ymm15 = _mm_add_ps(ymm15, ymm6);

				ymm7 = _mm_mul_ps(ymm7, ymm13);
				ymm14 = _mm_add_ps(ymm14, ymm7);
				ymm8 = _mm_mul_ps(ymm8, ymm13);
				ymm15 = _mm_add_ps(ymm15, ymm8);

				//12
				//-----------------------------------

				ymm10 = _mm_set1_ps(*(snn_hl_w + 8));
				ymm11 = _mm_set1_ps(*(snn_hl_w + 9));
				ymm12 = _mm_set1_ps(*(snn_hl_w + 10));
				ymm13 = _mm_set1_ps(*(snn_hl_w + 11));

				ymm1 = _mm_load_ps(src[8] + i);
				ymm2 = _mm_load_ps(src[8] + REG_SIZE + i);
				ymm3 = _mm_load_ps(src[9] + i);
				ymm4 = _mm_load_ps(src[9] + REG_SIZE + i);
				ymm5 = _mm_load_ps(src[10] + i);
				ymm6 = _mm_load_ps(src[10] + REG_SIZE + i);
				ymm7 = _mm_load_ps(src[11] + i);
				ymm8 = _mm_load_ps(src[11] + REG_SIZE + i);

				ymm1 = _mm_mul_ps(ymm1, ymm10);
				ymm14 = _mm_add_ps(ymm14, ymm1);
				ymm2 = _mm_mul_ps(ymm2, ymm10);
				ymm15 = _mm_add_ps(ymm15, ymm2);

				ymm3 = _mm_mul_ps(ymm3, ymm11);
				ymm14 = _mm_add_ps(ymm14, ymm3);
				ymm4 = _mm_mul_ps(ymm4, ymm11);
				ymm15 = _mm_add_ps(ymm15, ymm4);

				ymm5 = _mm_mul_ps(ymm5, ymm12);
				ymm14 = _mm_add_ps(ymm14, ymm5);
				ymm6 = _mm_mul_ps(ymm6, ymm12);
				ymm15 = _mm_add_ps(ymm15, ymm6);

				ymm7 = _mm_mul_ps(ymm7, ymm13);
				ymm14 = _mm_add_ps(ymm14, ymm7);
				ymm8 = _mm_mul_ps(ymm8, ymm13);
				ymm15 = _mm_add_ps(ymm15, ymm8);

				//16
				//-----------------------------------

				ymm10 = _mm_set1_ps(*(snn_hl_w + 12));
				ymm11 = _mm_set1_ps(*(snn_hl_w + 13));
				ymm12 = _mm_set1_ps(*(snn_hl_w + 14));
				ymm13 = _mm_set1_ps(*(snn_hl_w + 15));

				ymm1 = _mm_load_ps(src[12] + i);
				ymm2 = _mm_load_ps(src[12] + REG_SIZE + i);
				ymm3 = _mm_load_ps(src[13] + i);
				ymm4 = _mm_load_ps(src[13] + REG_SIZE + i);
				ymm5 = _mm_load_ps(src[14] + i);
				ymm6 = _mm_load_ps(src[14] + REG_SIZE + i);
				ymm7 = _mm_load_ps(src[15] + i);
				ymm8 = _mm_load_ps(src[15] + REG_SIZE + i);

				ymm1 = _mm_mul_ps(ymm1, ymm10);
				ymm14 = _mm_add_ps(ymm14, ymm1);
				ymm2 = _mm_mul_ps(ymm2, ymm10);
				ymm15 = _mm_add_ps(ymm15, ymm2);

				ymm3 = _mm_mul_ps(ymm3, ymm11);
				ymm14 = _mm_add_ps(ymm14, ymm3);
				ymm4 = _mm_mul_ps(ymm4, ymm11);
				ymm15 = _mm_add_ps(ymm15, ymm4);

				ymm5 = _mm_mul_ps(ymm5, ymm12);
				ymm14 = _mm_add_ps(ymm14, ymm5);
				ymm6 = _mm_mul_ps(ymm6, ymm12);
				ymm15 = _mm_add_ps(ymm15, ymm6);

				ymm7 = _mm_mul_ps(ymm7, ymm13);
				ymm14 = _mm_add_ps(ymm14, ymm7);
				ymm8 = _mm_mul_ps(ymm8, ymm13);
				ymm15 = _mm_add_ps(ymm15, ymm8);

				//20
				//-----------------------------------

				ymm10 = _mm_set1_ps(*(snn_hl_w + 16));
				ymm11 = _mm_set1_ps(*(snn_hl_w + 17));
				ymm12 = _mm_set1_ps(*(snn_hl_w + 18));
				ymm13 = _mm_set1_ps(*(snn_hl_w + 19));

				ymm1 = _mm_load_ps(src[16] + i);
				ymm2 = _mm_load_ps(src[16] + REG_SIZE + i);
				ymm3 = _mm_load_ps(src[17] + i);
				ymm4 = _mm_load_ps(src[17] + REG_SIZE + i);
				ymm5 = _mm_load_ps(src[18] + i);
				ymm6 = _mm_load_ps(src[18] + REG_SIZE + i);
				ymm7 = _mm_load_ps(src[19] + i);
				ymm8 = _mm_load_ps(src[19] + REG_SIZE + i);

				ymm1 = _mm_mul_ps(ymm1, ymm10);
				ymm14 = _mm_add_ps(ymm14, ymm1);
				ymm2 = _mm_mul_ps(ymm2, ymm10);
				ymm15 = _mm_add_ps(ymm15, ymm2);

				ymm3 = _mm_mul_ps(ymm3, ymm11);
				ymm14 = _mm_add_ps(ymm14, ymm3);
				ymm4 = _mm_mul_ps(ymm4, ymm11);
				ymm15 = _mm_add_ps(ymm15, ymm4);

				ymm5 = _mm_mul_ps(ymm5, ymm12);
				ymm14 = _mm_add_ps(ymm14, ymm5);
				ymm6 = _mm_mul_ps(ymm6, ymm12);
				ymm15 = _mm_add_ps(ymm15, ymm6);

				ymm7 = _mm_mul_ps(ymm7, ymm13);
				ymm14 = _mm_add_ps(ymm14, ymm7);
				ymm8 = _mm_mul_ps(ymm8, ymm13);
				ymm15 = _mm_add_ps(ymm15, ymm8);

				//24
				//-----------------------------------

				ymm10 = _mm_set1_ps(*(snn_hl_w + 20));
				ymm11 = _mm_set1_ps(*(snn_hl_w + 21));
				ymm12 = _mm_set1_ps(*(snn_hl_w + 22));
				ymm13 = _mm_set1_ps(*(snn_hl_w + 23));

				ymm1 = _mm_load_ps(src[20] + i);
				ymm2 = _mm_load_ps(src[20] + REG_SIZE + i);
				ymm3 = _mm_load_ps(src[21] + i);
				ymm4 = _mm_load_ps(src[21] + REG_SIZE + i);
				ymm5 = _mm_load_ps(src[22] + i);
				ymm6 = _mm_load_ps(src[22] + REG_SIZE + i);
				ymm7 = _mm_load_ps(src[23] + i);
				ymm8 = _mm_load_ps(src[23] + REG_SIZE + i);

				ymm1 = _mm_mul_ps(ymm1, ymm10);
				ymm14 = _mm_add_ps(ymm14, ymm1);
				ymm2 = _mm_mul_ps(ymm2, ymm10);
				ymm15 = _mm_add_ps(ymm15, ymm2);

				ymm3 = _mm_mul_ps(ymm3, ymm11);
				ymm14 = _mm_add_ps(ymm14, ymm3);
				ymm4 = _mm_mul_ps(ymm4, ymm11);
				ymm15 = _mm_add_ps(ymm15, ymm4);

				ymm5 = _mm_mul_ps(ymm5, ymm12);
				ymm14 = _mm_add_ps(ymm14, ymm5);
				ymm6 = _mm_mul_ps(ymm6, ymm12);
				ymm15 = _mm_add_ps(ymm15, ymm6);

				ymm7 = _mm_mul_ps(ymm7, ymm13);
				ymm14 = _mm_add_ps(ymm14, ymm7);
				ymm8 = _mm_mul_ps(ymm8, ymm13);
				ymm15 = _mm_add_ps(ymm15, ymm8);

				//tanh
				ymm7 = _mm_set1_ps(*(float*)&abs_mask);
				ymm8 = _mm_set1_ps(one);
				ymm13 = _mm_set1_ps(tanh_a);

				ymm1 = _mm_and_ps(ymm7, ymm14);
				ymm3 = _mm_add_ps(ymm8, ymm1);
				ymm4 = _mm_mul_ps(ymm14, ymm14);

				ymm3 = _mm_add_ps(ymm3, ymm4);
				ymm5 = _mm_mul_ps(ymm4, ymm4);

				ymm5 = _mm_mul_ps(ymm5, ymm13);
				ymm3 = _mm_add_ps(ymm3, ymm5);
				ymm2 = _mm_andnot_ps(ymm7, ymm14);

#ifdef USE_FAST_DIV
				ymm3 = _mm_rcp_ps(ymm3);
#else
				ymm3 = _mm_div_ps(ymm8, ymm3);
#endif	

				ymm3 = _mm_sub_ps(ymm8, ymm3);
				ymm3 = _mm_or_ps(ymm3, ymm2);

				ymm14 = _mm_mul_ps(ymm3, ymm9);
				_mm_store_ps(dst + i, ymm14);

				ymm1 = _mm_and_ps(ymm7, ymm15);
				ymm3 = _mm_add_ps(ymm8, ymm1);
				ymm4 = _mm_mul_ps(ymm15, ymm15);

				ymm3 = _mm_add_ps(ymm3, ymm4);
				ymm5 = _mm_mul_ps(ymm4, ymm4);

				ymm5 = _mm_mul_ps(ymm5, ymm13);
				ymm3 = _mm_add_ps(ymm3, ymm5);
				ymm2 = _mm_andnot_ps(ymm7, ymm15);

#ifdef USE_FAST_DIV
				ymm3 = _mm_rcp_ps(ymm3);
#else
				ymm3 = _mm_div_ps(ymm8, ymm3);
#endif	

				ymm3 = _mm_sub_ps(ymm8, ymm3);
				ymm3 = _mm_or_ps(ymm3, ymm2);

				ymm15 = _mm_mul_ps(ymm3, ymm9);
				_mm_store_ps(dst + REG_SIZE + i, ymm15);
			}
		}

#else

		void CNNPP::mulC24_add_tanh(float* __restrict dst, float* __restrict* src, int size_, float* __restrict snn_hl_w, float* __restrict snn_hl_b, float* __restrict scale, float* __restrict snn_ol_w)
		{
			__m128 ymm0 = _mm_set1_ps(*snn_hl_b);
			__m128 ymm9 = _mm_set1_ps(*snn_ol_w);

			__m128 ymm11 = _mm_set1_ps(*scale);
			ymm9 = _mm_mul_ps(ymm9, ymm11);

			for (size_t i = 0; i < size_; i += 2 * REG_SIZE)
			{
				__m128 ymm14 = ymm0;
				__m128 ymm15 = ymm0;

				//4
				//-----------------------------------

				__m128 ymm10 = _mm_set1_ps(*snn_hl_w);
				ymm11 = _mm_set1_ps(*(snn_hl_w + 1));
				__m128 ymm12 = _mm_set1_ps(*(snn_hl_w + 2));
				__m128 ymm13 = _mm_set1_ps(*(snn_hl_w + 3));

				__m128 ymm1 = _mm_load_ps(src[0] + i);
				__m128 ymm2 = _mm_load_ps(src[0] + REG_SIZE + i);
				__m128 ymm3 = _mm_load_ps(src[1] + i);
				__m128 ymm4 = _mm_load_ps(src[1] + REG_SIZE + i);
				__m128 ymm5 = _mm_load_ps(src[2] + i);
				__m128 ymm6 = _mm_load_ps(src[2] + REG_SIZE + i);
				__m128 ymm7 = _mm_load_ps(src[3] + i);
				__m128 ymm8 = _mm_load_ps(src[3] + REG_SIZE + i);

				ymm14 = _mm_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm_fmadd_ps(ymm8, ymm13, ymm15);

				//8
				//-----------------------------------

				ymm10 = _mm_set1_ps(*(snn_hl_w + 4));
				ymm11 = _mm_set1_ps(*(snn_hl_w + 5));
				ymm12 = _mm_set1_ps(*(snn_hl_w + 6));
				ymm13 = _mm_set1_ps(*(snn_hl_w + 7));

				ymm1 = _mm_load_ps(src[4] + i);
				ymm2 = _mm_load_ps(src[4] + REG_SIZE + i);
				ymm3 = _mm_load_ps(src[5] + i);
				ymm4 = _mm_load_ps(src[5] + REG_SIZE + i);
				ymm5 = _mm_load_ps(src[6] + i);
				ymm6 = _mm_load_ps(src[6] + REG_SIZE + i);
				ymm7 = _mm_load_ps(src[7] + i);
				ymm8 = _mm_load_ps(src[7] + REG_SIZE + i);

				ymm14 = _mm_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm_fmadd_ps(ymm8, ymm13, ymm15);

				//12
				//-----------------------------------

				ymm10 = _mm_set1_ps(*(snn_hl_w + 8));
				ymm11 = _mm_set1_ps(*(snn_hl_w + 9));
				ymm12 = _mm_set1_ps(*(snn_hl_w + 10));
				ymm13 = _mm_set1_ps(*(snn_hl_w + 11));

				ymm1 = _mm_load_ps(src[8] + i);
				ymm2 = _mm_load_ps(src[8] + REG_SIZE + i);
				ymm3 = _mm_load_ps(src[9] + i);
				ymm4 = _mm_load_ps(src[9] + REG_SIZE + i);
				ymm5 = _mm_load_ps(src[10] + i);
				ymm6 = _mm_load_ps(src[10] + REG_SIZE + i);
				ymm7 = _mm_load_ps(src[11] + i);
				ymm8 = _mm_load_ps(src[11] + REG_SIZE + i);

				ymm14 = _mm_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm_fmadd_ps(ymm8, ymm13, ymm15);

				//16
				//-----------------------------------

				ymm10 = _mm_set1_ps(*(snn_hl_w + 12));
				ymm11 = _mm_set1_ps(*(snn_hl_w + 13));
				ymm12 = _mm_set1_ps(*(snn_hl_w + 14));
				ymm13 = _mm_set1_ps(*(snn_hl_w + 15));

				ymm1 = _mm_load_ps(src[12] + i);
				ymm2 = _mm_load_ps(src[12] + REG_SIZE + i);
				ymm3 = _mm_load_ps(src[13] + i);
				ymm4 = _mm_load_ps(src[13] + REG_SIZE + i);
				ymm5 = _mm_load_ps(src[14] + i);
				ymm6 = _mm_load_ps(src[14] + REG_SIZE + i);
				ymm7 = _mm_load_ps(src[15] + i);
				ymm8 = _mm_load_ps(src[15] + REG_SIZE + i);

				ymm14 = _mm_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm_fmadd_ps(ymm8, ymm13, ymm15);

				//20
				//-----------------------------------

				ymm10 = _mm_set1_ps(*(snn_hl_w + 16));
				ymm11 = _mm_set1_ps(*(snn_hl_w + 17));
				ymm12 = _mm_set1_ps(*(snn_hl_w + 18));
				ymm13 = _mm_set1_ps(*(snn_hl_w + 19));

				ymm1 = _mm_load_ps(src[16] + i);
				ymm2 = _mm_load_ps(src[16] + REG_SIZE + i);
				ymm3 = _mm_load_ps(src[17] + i);
				ymm4 = _mm_load_ps(src[17] + REG_SIZE + i);
				ymm5 = _mm_load_ps(src[18] + i);
				ymm6 = _mm_load_ps(src[18] + REG_SIZE + i);
				ymm7 = _mm_load_ps(src[19] + i);
				ymm8 = _mm_load_ps(src[19] + REG_SIZE + i);

				ymm14 = _mm_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm_fmadd_ps(ymm8, ymm13, ymm15);

				//24
				//-----------------------------------

				ymm10 = _mm_set1_ps(*(snn_hl_w + 20));
				ymm11 = _mm_set1_ps(*(snn_hl_w + 21));
				ymm12 = _mm_set1_ps(*(snn_hl_w + 22));
				ymm13 = _mm_set1_ps(*(snn_hl_w + 23));

				ymm1 = _mm_load_ps(src[20] + i);
				ymm2 = _mm_load_ps(src[20] + REG_SIZE + i);
				ymm3 = _mm_load_ps(src[21] + i);
				ymm4 = _mm_load_ps(src[21] + REG_SIZE + i);
				ymm5 = _mm_load_ps(src[22] + i);
				ymm6 = _mm_load_ps(src[22] + REG_SIZE + i);
				ymm7 = _mm_load_ps(src[23] + i);
				ymm8 = _mm_load_ps(src[23] + REG_SIZE + i);

				ymm14 = _mm_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm_fmadd_ps(ymm8, ymm13, ymm15);

				//tanh
				ymm7 = _mm_set1_ps(*(float*)&abs_mask);
				ymm8 = _mm_set1_ps(one);
				ymm13 = _mm_set1_ps(tanh_a);

				ymm1 = _mm_and_ps(ymm7, ymm14);
				ymm3 = _mm_add_ps(ymm8, ymm1);
				ymm4 = _mm_mul_ps(ymm14, ymm14);

				ymm3 = _mm_add_ps(ymm3, ymm4);
				ymm5 = _mm_mul_ps(ymm4, ymm4);

				//ymm5 = _mm_mul_ps(ymm5, ymm13);
				//ymm3 = _mm_add_ps(ymm3, ymm5);
				ymm3 = _mm_fmadd_ps(ymm5, ymm13, ymm3);
				ymm2 = _mm_andnot_ps(ymm7, ymm14);

#ifdef USE_FAST_DIV
				ymm3 = _mm_rcp_ps(ymm3);
#else
				ymm3 = _mm_div_ps(ymm8, ymm3);
#endif	

				ymm3 = _mm_sub_ps(ymm8, ymm3);
				ymm3 = _mm_or_ps(ymm3, ymm2);

				ymm14 = _mm_mul_ps(ymm3, ymm9);
				_mm_store_ps(dst + i, ymm14);

				ymm1 = _mm_and_ps(ymm7, ymm15);
				ymm3 = _mm_add_ps(ymm8, ymm1);
				ymm4 = _mm_mul_ps(ymm15, ymm15);

				ymm3 = _mm_add_ps(ymm3, ymm4);
				ymm5 = _mm_mul_ps(ymm4, ymm4);

				//ymm5 = _mm_mul_ps(ymm5, ymm13);
				//ymm3 = _mm_add_ps(ymm3, ymm5);
				ymm3 = _mm_fmadd_ps(ymm5, ymm13, ymm3);
				ymm2 = _mm_andnot_ps(ymm7, ymm15);

#ifdef USE_FAST_DIV
				ymm3 = _mm_rcp_ps(ymm3);
#else
				ymm3 = _mm_div_ps(ymm8, ymm3);
#endif	

				ymm3 = _mm_sub_ps(ymm8, ymm3);
				ymm3 = _mm_or_ps(ymm3, ymm2);

				ymm15 = _mm_mul_ps(ymm3, ymm9);
				_mm_store_ps(dst + REG_SIZE + i, ymm15);
			}
		}

#endif
	}

#endif
}