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


#include "cnnpp_simd_avx.h"
#include <immintrin.h>


//========================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_AVX

	namespace SIMD
	{
#ifndef USE_FMA

		void CNNPP::conv_3x3(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 2;
			if (H == 0) H = src_size_h - 2;

			__m256 ymm12 = _mm256_load_ps(kernel);
			__m256 ymm13 = _mm256_load_ps(kernel + REG_SIZE);
			__m256 ymm14 = _mm256_load_ps(kernel + 2 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				__m256 ymm0 = _mm256_load_ps(pSrc0);
				__m256 ymm1 = _mm256_load_ps(pSrc1);
				__m256 ymm2 = _mm256_load_ps(pSrc2);
				pSrc0 += REG_SIZE;
				pSrc1 += REG_SIZE;
				pSrc2 += REG_SIZE;

				size_t i = 0;
				for (; i < L - REG_SIZE; i += REG_SIZE)
				{
					__m256 ymm4 = _mm256_load_ps(pSrc0);
					__m256 ymm5 = _mm256_load_ps(pSrc1);
					__m256 ymm6 = _mm256_load_ps(pSrc2);
					pSrc0 += REG_SIZE;
					pSrc1 += REG_SIZE;
					pSrc2 += REG_SIZE;

					__m256 ymm8 = _mm256_mul_ps(ymm0, ymm12);
					__m256 ymm9 = _mm256_mul_ps(ymm1, ymm13);
					__m256 ymm11 = _mm256_add_ps(ymm8, ymm9);

					ymm8 = _mm256_mul_ps(ymm2, ymm14);
					ymm11 = _mm256_add_ps(ymm11, ymm8);

					ymm8 = _mm256_permute_ps(ymm11, 14);
					ymm11 = _mm256_add_ps(ymm11, ymm8);
					ymm8 = _mm256_permute_ps(ymm11, 1);
					ymm11 = _mm256_add_ps(ymm11, ymm8);

					__m256 ymm3 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
					__m256 ymm7 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
					__m256 ymm15 = _mm256_permute2f128_ps(ymm2, ymm6, 33);

					//-----------------------------------------

					ymm0 = _mm256_blend_ps(ymm0, ymm3, 17);
					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm8 = _mm256_mul_ps(ymm0, ymm12);

					ymm1 = _mm256_blend_ps(ymm1, ymm7, 17);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm9 = _mm256_mul_ps(ymm1, ymm13);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm2 = _mm256_blend_ps(ymm2, ymm15, 17);
					ymm14 = _mm256_permute_ps(ymm14, 147);
					ymm9 = _mm256_mul_ps(ymm2, ymm14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm9 = _mm256_permute_ps(ymm8, 14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm9 = _mm256_permute_ps(ymm8, 1);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 34);

					//-----------------------------------------

					ymm0 = _mm256_blend_ps(ymm0, ymm3, 34);
					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm8 = _mm256_mul_ps(ymm0, ymm12);

					ymm1 = _mm256_blend_ps(ymm1, ymm7, 34);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm9 = _mm256_mul_ps(ymm1, ymm13);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm2 = _mm256_blend_ps(ymm2, ymm15, 34);
					ymm14 = _mm256_permute_ps(ymm14, 147);
					ymm9 = _mm256_mul_ps(ymm2, ymm14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm9 = _mm256_permute_ps(ymm8, 64);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm9 = _mm256_permute_ps(ymm8, 176);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 68);

					//-----------------------------------------

					ymm0 = _mm256_blend_ps(ymm0, ymm3, 68);
					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm8 = _mm256_mul_ps(ymm0, ymm12);

					ymm1 = _mm256_blend_ps(ymm1, ymm7, 68);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm9 = _mm256_mul_ps(ymm1, ymm13);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm2 = _mm256_blend_ps(ymm2, ymm15, 68);
					ymm14 = _mm256_permute_ps(ymm14, 147);
					ymm9 = _mm256_mul_ps(ymm2, ymm14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm9 = _mm256_permute_ps(ymm8, 64);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm9 = _mm256_permute_ps(ymm8, 176);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 136);

					_mm256_store_ps(pDst, ymm11);
					pDst += REG_SIZE;

					ymm0 = ymm4;
					ymm1 = ymm5;
					ymm2 = ymm6;

					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm14 = _mm256_permute_ps(ymm14, 147);
				}

				//============================================================================

				__m256 ymm4 = _mm256_setzero_ps();
				__m256 ymm5 = _mm256_setzero_ps();
				__m256 ymm6 = _mm256_setzero_ps();

				if (i + REG_SIZE < src_size_l)
				{
					ymm4 = _mm256_load_ps(pSrc0);
					ymm5 = _mm256_load_ps(pSrc1);
					ymm6 = _mm256_load_ps(pSrc2);
				}

				__m256 ymm8 = _mm256_mul_ps(ymm0, ymm12);
				__m256 ymm9 = _mm256_mul_ps(ymm1, ymm13);
				__m256 ymm11 = _mm256_add_ps(ymm8, ymm9);

				ymm8 = _mm256_mul_ps(ymm2, ymm14);
				ymm11 = _mm256_add_ps(ymm11, ymm8);

				ymm8 = _mm256_permute_ps(ymm11, 14);
				ymm11 = _mm256_add_ps(ymm11, ymm8);
				ymm8 = _mm256_permute_ps(ymm11, 1);
				ymm11 = _mm256_add_ps(ymm11, ymm8);

				__m256 ymm3 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
				__m256 ymm7 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
				__m256 ymm15 = _mm256_permute2f128_ps(ymm2, ymm6, 33);

				//-----------------------------------------

				ymm0 = _mm256_blend_ps(ymm0, ymm3, 17);
				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm8 = _mm256_mul_ps(ymm0, ymm12);

				ymm1 = _mm256_blend_ps(ymm1, ymm7, 17);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm9 = _mm256_mul_ps(ymm1, ymm13);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm2 = _mm256_blend_ps(ymm2, ymm15, 17);
				ymm14 = _mm256_permute_ps(ymm14, 147);
				ymm9 = _mm256_mul_ps(ymm2, ymm14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm9 = _mm256_permute_ps(ymm8, 14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm9 = _mm256_permute_ps(ymm8, 1);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 34);

				//-----------------------------------------

				ymm0 = _mm256_blend_ps(ymm0, ymm3, 34);
				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm8 = _mm256_mul_ps(ymm0, ymm12);

				ymm1 = _mm256_blend_ps(ymm1, ymm7, 34);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm9 = _mm256_mul_ps(ymm1, ymm13);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm2 = _mm256_blend_ps(ymm2, ymm15, 34);
				ymm14 = _mm256_permute_ps(ymm14, 147);
				ymm9 = _mm256_mul_ps(ymm2, ymm14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm9 = _mm256_permute_ps(ymm8, 64);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm9 = _mm256_permute_ps(ymm8, 176);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 68);

				//-----------------------------------------

				ymm0 = _mm256_blend_ps(ymm0, ymm3, 68);
				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm8 = _mm256_mul_ps(ymm0, ymm12);

				ymm1 = _mm256_blend_ps(ymm1, ymm7, 68);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm9 = _mm256_mul_ps(ymm1, ymm13);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm2 = _mm256_blend_ps(ymm2, ymm15, 68);
				ymm14 = _mm256_permute_ps(ymm14, 147);
				ymm9 = _mm256_mul_ps(ymm2, ymm14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm9 = _mm256_permute_ps(ymm8, 64);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm9 = _mm256_permute_ps(ymm8, 176);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 136);

				_mm256_store_ps(pDst, ymm11);

				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm14 = _mm256_permute_ps(ymm14, 147);

				//============================================================================
			}
		}
		void CNNPP::conv_4x4(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;

			__m256 ymm12 = _mm256_load_ps(kernel);
			__m256 ymm13 = _mm256_load_ps(kernel + REG_SIZE);
			__m256 ymm14 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			__m256 ymm15 = _mm256_load_ps(kernel + 3 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				__m256 ymm0 = _mm256_load_ps(pSrc0);
				__m256 ymm1 = _mm256_load_ps(pSrc1);
				__m256 ymm2 = _mm256_load_ps(pSrc2);
				__m256 ymm3 = _mm256_load_ps(pSrc3);
				pSrc0 += REG_SIZE;
				pSrc1 += REG_SIZE;
				pSrc2 += REG_SIZE;
				pSrc3 += REG_SIZE;

				size_t i = 0;
				for (; i < L - REG_SIZE; i += REG_SIZE)
				{
					__m256 ymm4 = _mm256_load_ps(pSrc0);
					__m256 ymm5 = _mm256_load_ps(pSrc1);
					__m256 ymm6 = _mm256_load_ps(pSrc2);
					__m256 ymm7 = _mm256_load_ps(pSrc3);
					pSrc0 += REG_SIZE;
					pSrc1 += REG_SIZE;
					pSrc2 += REG_SIZE;
					pSrc3 += REG_SIZE;

					__m256 ymm8 = _mm256_mul_ps(ymm0, ymm12);
					__m256 ymm9 = _mm256_mul_ps(ymm1, ymm13);
					__m256 ymm11 = _mm256_add_ps(ymm8, ymm9);

					ymm8 = _mm256_mul_ps(ymm2, ymm14);
					ymm11 = _mm256_add_ps(ymm11, ymm8);

					ymm8 = _mm256_mul_ps(ymm3, ymm15);
					ymm11 = _mm256_add_ps(ymm11, ymm8);

					ymm8 = _mm256_permute_ps(ymm11, 14);
					ymm11 = _mm256_add_ps(ymm11, ymm8);
					ymm8 = _mm256_permute_ps(ymm11, 1);
					ymm11 = _mm256_add_ps(ymm11, ymm8);

					//-----------------------------------------

					__m256 ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
					ymm0 = _mm256_blend_ps(ymm0, ymm10, 17);
					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm8 = _mm256_mul_ps(ymm0, ymm12);

					ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
					ymm1 = _mm256_blend_ps(ymm1, ymm10, 17);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm9 = _mm256_mul_ps(ymm1, ymm13);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
					ymm2 = _mm256_blend_ps(ymm2, ymm10, 17);
					ymm14 = _mm256_permute_ps(ymm14, 147);
					ymm9 = _mm256_mul_ps(ymm2, ymm14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
					ymm3 = _mm256_blend_ps(ymm3, ymm10, 17);
					ymm15 = _mm256_permute_ps(ymm15, 147);
					ymm9 = _mm256_mul_ps(ymm3, ymm15);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm9 = _mm256_permute_ps(ymm8, 14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm9 = _mm256_permute_ps(ymm8, 1);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 34);

					//-----------------------------------------

					ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
					ymm0 = _mm256_blend_ps(ymm0, ymm10, 34);
					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm8 = _mm256_mul_ps(ymm0, ymm12);

					ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
					ymm1 = _mm256_blend_ps(ymm1, ymm10, 34);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm9 = _mm256_mul_ps(ymm1, ymm13);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
					ymm2 = _mm256_blend_ps(ymm2, ymm10, 34);
					ymm14 = _mm256_permute_ps(ymm14, 147);
					ymm9 = _mm256_mul_ps(ymm2, ymm14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
					ymm3 = _mm256_blend_ps(ymm3, ymm10, 34);
					ymm15 = _mm256_permute_ps(ymm15, 147);
					ymm9 = _mm256_mul_ps(ymm3, ymm15);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm9 = _mm256_permute_ps(ymm8, 64);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm9 = _mm256_permute_ps(ymm8, 176);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 68);

					//-----------------------------------------

					ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
					ymm0 = _mm256_blend_ps(ymm0, ymm10, 68);
					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm8 = _mm256_mul_ps(ymm0, ymm12);

					ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
					ymm1 = _mm256_blend_ps(ymm1, ymm10, 68);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm9 = _mm256_mul_ps(ymm1, ymm13);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
					ymm2 = _mm256_blend_ps(ymm2, ymm10, 68);
					ymm14 = _mm256_permute_ps(ymm14, 147);
					ymm9 = _mm256_mul_ps(ymm2, ymm14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
					ymm3 = _mm256_blend_ps(ymm3, ymm10, 68);
					ymm15 = _mm256_permute_ps(ymm15, 147);
					ymm9 = _mm256_mul_ps(ymm3, ymm15);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm9 = _mm256_permute_ps(ymm8, 64);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm9 = _mm256_permute_ps(ymm8, 176);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 136);

					_mm256_store_ps(pDst, ymm11);
					pDst += REG_SIZE;

					ymm0 = ymm4;
					ymm1 = ymm5;
					ymm2 = ymm6;
					ymm3 = ymm7;

					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm14 = _mm256_permute_ps(ymm14, 147);
					ymm15 = _mm256_permute_ps(ymm15, 147);
				}

				//============================================================================

				__m256 ymm4 = _mm256_setzero_ps();
				__m256 ymm5 = _mm256_setzero_ps();
				__m256 ymm6 = _mm256_setzero_ps();
				__m256 ymm7 = _mm256_setzero_ps();

				if (i + REG_SIZE < src_size_l)
				{
					ymm4 = _mm256_load_ps(pSrc0);
					ymm5 = _mm256_load_ps(pSrc1);
					ymm6 = _mm256_load_ps(pSrc2);
					ymm7 = _mm256_load_ps(pSrc3);
				}

				__m256 ymm8 = _mm256_mul_ps(ymm0, ymm12);
				__m256 ymm9 = _mm256_mul_ps(ymm1, ymm13);
				__m256 ymm11 = _mm256_add_ps(ymm8, ymm9);

				ymm8 = _mm256_mul_ps(ymm2, ymm14);
				ymm11 = _mm256_add_ps(ymm11, ymm8);

				ymm8 = _mm256_mul_ps(ymm3, ymm15);
				ymm11 = _mm256_add_ps(ymm11, ymm8);

				ymm8 = _mm256_permute_ps(ymm11, 14);
				ymm11 = _mm256_add_ps(ymm11, ymm8);
				ymm8 = _mm256_permute_ps(ymm11, 1);
				ymm11 = _mm256_add_ps(ymm11, ymm8);

				//-----------------------------------------

				__m256 ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
				ymm0 = _mm256_blend_ps(ymm0, ymm10, 17);
				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm8 = _mm256_mul_ps(ymm0, ymm12);

				ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
				ymm1 = _mm256_blend_ps(ymm1, ymm10, 17);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm9 = _mm256_mul_ps(ymm1, ymm13);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
				ymm2 = _mm256_blend_ps(ymm2, ymm10, 17);
				ymm14 = _mm256_permute_ps(ymm14, 147);
				ymm9 = _mm256_mul_ps(ymm2, ymm14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
				ymm3 = _mm256_blend_ps(ymm3, ymm10, 17);
				ymm15 = _mm256_permute_ps(ymm15, 147);
				ymm9 = _mm256_mul_ps(ymm3, ymm15);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm9 = _mm256_permute_ps(ymm8, 14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm9 = _mm256_permute_ps(ymm8, 1);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 34);

				//-----------------------------------------

				ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
				ymm0 = _mm256_blend_ps(ymm0, ymm10, 34);
				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm8 = _mm256_mul_ps(ymm0, ymm12);

				ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
				ymm1 = _mm256_blend_ps(ymm1, ymm10, 34);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm9 = _mm256_mul_ps(ymm1, ymm13);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
				ymm2 = _mm256_blend_ps(ymm2, ymm10, 34);
				ymm14 = _mm256_permute_ps(ymm14, 147);
				ymm9 = _mm256_mul_ps(ymm2, ymm14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
				ymm3 = _mm256_blend_ps(ymm3, ymm10, 34);
				ymm15 = _mm256_permute_ps(ymm15, 147);
				ymm9 = _mm256_mul_ps(ymm3, ymm15);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm9 = _mm256_permute_ps(ymm8, 64);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm9 = _mm256_permute_ps(ymm8, 176);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 68);

				//-----------------------------------------

				ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
				ymm0 = _mm256_blend_ps(ymm0, ymm10, 68);
				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm8 = _mm256_mul_ps(ymm0, ymm12);

				ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
				ymm1 = _mm256_blend_ps(ymm1, ymm10, 68);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm9 = _mm256_mul_ps(ymm1, ymm13);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
				ymm2 = _mm256_blend_ps(ymm2, ymm10, 68);
				ymm14 = _mm256_permute_ps(ymm14, 147);
				ymm9 = _mm256_mul_ps(ymm2, ymm14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
				ymm3 = _mm256_blend_ps(ymm3, ymm10, 68);
				ymm15 = _mm256_permute_ps(ymm15, 147);
				ymm9 = _mm256_mul_ps(ymm3, ymm15);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm9 = _mm256_permute_ps(ymm8, 64);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm9 = _mm256_permute_ps(ymm8, 176);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 136);

				_mm256_store_ps(pDst, ymm11);

				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm14 = _mm256_permute_ps(ymm14, 147);
				ymm15 = _mm256_permute_ps(ymm15, 147);

				//============================================================================
			}
		}
		void CNNPP::conv_5x4(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };

			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 4;

			const __m256i ymm_mask = _mm256_load_si256((__m256i*)set1_mask);

			const __m256 ymm8 = _mm256_load_ps(kernel);
			const __m256 ymm9 = _mm256_load_ps(kernel + REG_SIZE);
			const __m256 ymm10 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			const __m256 ymm11 = _mm256_load_ps(kernel + 3 * REG_SIZE);
			const __m256 ymm12 = _mm256_load_ps(kernel + 4 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pSrc4 = src + (j + 4) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				for (size_t i = 0; i < L; ++i)
				{
					__m256 ymm0 = _mm256_loadu_ps(pSrc0++);
					__m256 ymm1 = _mm256_loadu_ps(pSrc1++);
					__m256 ymm2 = _mm256_loadu_ps(pSrc2++);

					__m256 ymm7 = _mm256_mul_ps(ymm0, ymm8);
					__m256 ymm6 = _mm256_mul_ps(ymm1, ymm9);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_mul_ps(ymm2, ymm10);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm0 = _mm256_loadu_ps(pSrc3++);
					ymm1 = _mm256_loadu_ps(pSrc4++);

					ymm6 = _mm256_mul_ps(ymm0, ymm11);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_mul_ps(ymm1, ymm12);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm6 = _mm256_permute_ps(ymm7, 14);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_permute_ps(ymm7, 1);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm1 = _mm256_permute2f128_ps(ymm7, ymm7, 33);
					ymm7 = _mm256_add_ps(ymm7, ymm1);

					//_mm_store_ss(pDst + i, _mm256_castps256_ps128(ymm7));
					_mm256_maskstore_ps(pDst++, ymm_mask, ymm7);
				}
			}
		}
		void CNNPP::conv_5x5(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
		}
		void CNNPP::conv_6x5(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };

			if (L == 0) L = src_size_l - 4;
			if (H == 0) H = src_size_h - 5;

			const __m256i ymm_mask = _mm256_load_si256((__m256i*)set1_mask);

			const __m256 ymm8 = _mm256_load_ps(kernel);
			const __m256 ymm9 = _mm256_load_ps(kernel + REG_SIZE);
			const __m256 ymm10 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			const __m256 ymm11 = _mm256_load_ps(kernel + 3 * REG_SIZE);
			const __m256 ymm12 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			const __m256 ymm13 = _mm256_load_ps(kernel + 5 * REG_SIZE);

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
					__m256 ymm0 = _mm256_loadu_ps(pSrc0++);
					__m256 ymm1 = _mm256_loadu_ps(pSrc1++);
					__m256 ymm2 = _mm256_loadu_ps(pSrc2++);

					__m256 ymm7 = _mm256_mul_ps(ymm0, ymm8);
					__m256 ymm6 = _mm256_mul_ps(ymm1, ymm9);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_mul_ps(ymm2, ymm10);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm0 = _mm256_loadu_ps(pSrc3++);
					ymm1 = _mm256_loadu_ps(pSrc4++);
					ymm2 = _mm256_loadu_ps(pSrc5++);

					ymm6 = _mm256_mul_ps(ymm0, ymm11);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_mul_ps(ymm1, ymm12);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_mul_ps(ymm2, ymm13);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm6 = _mm256_permute_ps(ymm7, 14);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_permute_ps(ymm7, 1);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm1 = _mm256_permute2f128_ps(ymm7, ymm7, 33);
					ymm7 = _mm256_add_ps(ymm7, ymm1);

					//_mm_store_ss(pDst + i, _mm256_castps256_ps128(ymm7));
					_mm256_maskstore_ps(pDst++, ymm_mask, ymm7);
				}
			}
		}
		void CNNPP::conv_6x6(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
		}
		void CNNPP::conv_7x7(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
		}
		void CNNPP::conv_8x7(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };

			if (L == 0) L = src_size_l - 6;
			if (H == 0) H = src_size_h - 7;

			const __m256i ymm_mask = _mm256_load_si256((__m256i*)set1_mask);

			const __m256 ymm8 = _mm256_load_ps(kernel);
			const __m256 ymm9 = _mm256_load_ps(kernel + REG_SIZE);
			const __m256 ymm10 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			const __m256 ymm11 = _mm256_load_ps(kernel + 3 * REG_SIZE);
			const __m256 ymm12 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			const __m256 ymm13 = _mm256_load_ps(kernel + 5 * REG_SIZE);
			const __m256 ymm14 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			const __m256 ymm15 = _mm256_load_ps(kernel + 7 * REG_SIZE);

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
					__m256 ymm0 = _mm256_loadu_ps(pSrc0++);
					__m256 ymm1 = _mm256_loadu_ps(pSrc1++);
					__m256 ymm2 = _mm256_loadu_ps(pSrc2++);
					__m256 ymm3 = _mm256_loadu_ps(pSrc3++);

					__m256 ymm7 = _mm256_mul_ps(ymm0, ymm8);
					__m256 ymm6 = _mm256_mul_ps(ymm1, ymm9);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_mul_ps(ymm2, ymm10);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_mul_ps(ymm3, ymm11);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm0 = _mm256_loadu_ps(pSrc4++);
					ymm1 = _mm256_loadu_ps(pSrc5++);
					ymm2 = _mm256_loadu_ps(pSrc6++);
					ymm3 = _mm256_loadu_ps(pSrc7++);

					ymm6 = _mm256_mul_ps(ymm0, ymm12);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_mul_ps(ymm1, ymm13);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_mul_ps(ymm2, ymm14);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_mul_ps(ymm3, ymm15);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm6 = _mm256_permute_ps(ymm7, 14);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_permute_ps(ymm7, 1);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm1 = _mm256_permute2f128_ps(ymm7, ymm7, 33);
					ymm7 = _mm256_add_ps(ymm7, ymm1);

					//_mm_store_ss(pDst + i, _mm256_castps256_ps128(ymm7));
					_mm256_maskstore_ps(pDst++, ymm_mask, ymm7);
				}
			}
		}
		void CNNPP::conv_8x8(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
		}
		void CNNPP::conv_11x10(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 9;
			if (H == 0) H = src_size_h - 10;

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc_[11];
				for (size_t y = 0; y < 11; ++y)
				{
					pSrc_[y] = src + (j + y) * src_size_l;
				}
				float* __restrict pDst = dst + j * dst_size_l;

				for (size_t i = 0; i < L; ++i)
				{
					float d = 0.f;
					float *kernel_ref = kernel;
					for (size_t y = 0; y < 11; ++y)
					{
						d += *(pSrc_[y] + 0)* * (kernel_ref + 0)
							+ *(pSrc_[y] + 1)* * (kernel_ref + 1)
							+ *(pSrc_[y] + 2)* * (kernel_ref + 2)
							+ *(pSrc_[y] + 3)* * (kernel_ref + 3)
							+ *(pSrc_[y] + 4)* * (kernel_ref + 4)
							+ *(pSrc_[y] + 5)* * (kernel_ref + 5)
							+ *(pSrc_[y] + 6)* * (kernel_ref + 6)
							+ *(pSrc_[y] + 7)* * (kernel_ref + 7)
							+ *(pSrc_[y] + 8)* * (kernel_ref + 8)
							+ *(pSrc_[y] + 9)* * (kernel_ref + 9);
						kernel_ref += 10;
					}
					*(pDst++) = d;
				}
			}
		}
		void CNNPP::conv_11x11(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
		}

#else

		void CNNPP::conv_3x3(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { 1, 2, 3, 4, 5, 6, 7, 0 };

			if (L == 0) L = src_size_l - 2;
			if (H == 0) H = src_size_h - 2;

			const __m256i ymm_mask_temp = _mm256_load_si256((__m256i*)set1_mask);

			__m256 ymm12 = _mm256_load_ps(kernel);
			__m256 ymm13 = _mm256_load_ps(kernel + REG_SIZE);
			__m256 ymm14 = _mm256_load_ps(kernel + 2 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				__m256 ymm0 = _mm256_load_ps(pSrc0);
				__m256 ymm1 = _mm256_load_ps(pSrc1);
				__m256 ymm2 = _mm256_load_ps(pSrc2);
				pSrc0 += REG_SIZE;
				pSrc1 += REG_SIZE;
				pSrc2 += REG_SIZE;

				size_t i = 0;
				for (; i < L - REG_SIZE; i += REG_SIZE)
				{
					__m256 ymm4 = _mm256_load_ps(pSrc0);
					__m256 ymm5 = _mm256_load_ps(pSrc1);
					__m256 ymm6 = _mm256_load_ps(pSrc2);
					pSrc0 += REG_SIZE;
					pSrc1 += REG_SIZE;
					pSrc2 += REG_SIZE;

					__m256 ymm8 = _mm256_mul_ps(ymm0, ymm12);
					ymm8 = _mm256_fmadd_ps(ymm1, ymm13, ymm8);
					ymm8 = _mm256_fmadd_ps(ymm2, ymm14, ymm8);

					__m256 ymm11 = _mm256_permute_ps(ymm8, 14);
					ymm8 = _mm256_add_ps(ymm8, ymm11);
					ymm11 = _mm256_permute_ps(ymm8, 1);
					ymm11 = _mm256_add_ps(ymm8, ymm11);

					//-----------------------------------------

					__m256 ymm10 = _mm256_blend_ps(ymm0, ymm4, 1);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
					ymm8 = _mm256_mul_ps(ymm10, ymm12);

					ymm10 = _mm256_blend_ps(ymm1, ymm5, 1);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm256_blend_ps(ymm2, ymm6, 1);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm256_permute_ps(ymm8, 14);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm10 = _mm256_permute_ps(ymm8, 1);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 34);

					//-----------------------------------------

					__m256i ymm_mask = _mm256_permutevar8x32_epi32(ymm_mask_temp, ymm_mask_temp);

					ymm10 = _mm256_blend_ps(ymm0, ymm4, 3);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_mul_ps(ymm10, ymm12);

					ymm10 = _mm256_blend_ps(ymm1, ymm5, 3);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm256_blend_ps(ymm2, ymm6, 3);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm256_permute_ps(ymm8, 64);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm10 = _mm256_permute_ps(ymm8, 176);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 68);

					//-----------------------------------------

					ymm_mask = _mm256_permutevar8x32_epi32(ymm_mask, ymm_mask_temp);

					ymm10 = _mm256_blend_ps(ymm0, ymm4, 7);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_mul_ps(ymm10, ymm12);

					ymm10 = _mm256_blend_ps(ymm1, ymm5, 7);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm256_blend_ps(ymm2, ymm6, 7);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm256_permute_ps(ymm8, 64);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm10 = _mm256_permute_ps(ymm8, 176);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 136);

					_mm256_store_ps(pDst, ymm11);
					pDst += REG_SIZE;

					ymm0 = ymm4;
					ymm1 = ymm5;
					ymm2 = ymm6;
				}

				//============================================================================

				__m256 ymm4 = _mm256_setzero_ps();
				__m256 ymm5 = _mm256_setzero_ps();
				__m256 ymm6 = _mm256_setzero_ps();

				if (i + REG_SIZE < src_size_l)
				{
					ymm4 = _mm256_load_ps(pSrc0);
					ymm5 = _mm256_load_ps(pSrc1);
					ymm6 = _mm256_load_ps(pSrc2);
				}

				__m256 ymm8 = _mm256_mul_ps(ymm0, ymm12);
				ymm8 = _mm256_fmadd_ps(ymm1, ymm13, ymm8);
				ymm8 = _mm256_fmadd_ps(ymm2, ymm14, ymm8);

				__m256 ymm11 = _mm256_permute_ps(ymm8, 14);
				ymm8 = _mm256_add_ps(ymm8, ymm11);
				ymm11 = _mm256_permute_ps(ymm8, 1);
				ymm11 = _mm256_add_ps(ymm8, ymm11);

				//-----------------------------------------

				__m256 ymm10 = _mm256_blend_ps(ymm0, ymm4, 1);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
				ymm8 = _mm256_mul_ps(ymm10, ymm12);

				ymm10 = _mm256_blend_ps(ymm1, ymm5, 1);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm256_blend_ps(ymm2, ymm6, 1);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm256_permute_ps(ymm8, 14);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm10 = _mm256_permute_ps(ymm8, 1);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 34);

				//-----------------------------------------

				__m256i ymm_mask = _mm256_permutevar8x32_epi32(ymm_mask_temp, ymm_mask_temp);

				ymm10 = _mm256_blend_ps(ymm0, ymm4, 3);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_mul_ps(ymm10, ymm12);

				ymm10 = _mm256_blend_ps(ymm1, ymm5, 3);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm256_blend_ps(ymm2, ymm6, 3);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm256_permute_ps(ymm8, 64);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm10 = _mm256_permute_ps(ymm8, 176);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 68);

				//-----------------------------------------

				ymm_mask = _mm256_permutevar8x32_epi32(ymm_mask, ymm_mask_temp);

				ymm10 = _mm256_blend_ps(ymm0, ymm4, 7);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_mul_ps(ymm10, ymm12);

				ymm10 = _mm256_blend_ps(ymm1, ymm5, 7);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm256_blend_ps(ymm2, ymm6, 7);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm256_permute_ps(ymm8, 64);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm10 = _mm256_permute_ps(ymm8, 176);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 136);

				_mm256_store_ps(pDst, ymm11);

				//============================================================================
			}
		}
		void CNNPP::conv_4x4(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { 1, 2, 3, 4, 5, 6, 7, 0 };

			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;

			const __m256i ymm_mask_temp = _mm256_load_si256((__m256i*)set1_mask);

			const __m256 ymm12 = _mm256_load_ps(kernel);
			const __m256 ymm13 = _mm256_load_ps(kernel + REG_SIZE);
			const __m256 ymm14 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			const __m256 ymm15 = _mm256_load_ps(kernel + 3 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				__m256 ymm0 = _mm256_load_ps(pSrc0);
				__m256 ymm1 = _mm256_load_ps(pSrc1);
				__m256 ymm2 = _mm256_load_ps(pSrc2);
				__m256 ymm3 = _mm256_load_ps(pSrc3);
				pSrc0 += REG_SIZE;
				pSrc1 += REG_SIZE;
				pSrc2 += REG_SIZE;
				pSrc3 += REG_SIZE;

				size_t i = 0;
				for (; i < L - REG_SIZE; i += REG_SIZE)
				{
					__m256 ymm4 = _mm256_load_ps(pSrc0);
					__m256 ymm5 = _mm256_load_ps(pSrc1);
					__m256 ymm6 = _mm256_load_ps(pSrc2);
					__m256 ymm7 = _mm256_load_ps(pSrc3);
					pSrc0 += REG_SIZE;
					pSrc1 += REG_SIZE;
					pSrc2 += REG_SIZE;
					pSrc3 += REG_SIZE;

					__m256 ymm8 = _mm256_mul_ps(ymm0, ymm12);
					__m256 ymm11 = _mm256_mul_ps(ymm2, ymm14);

					ymm8 = _mm256_fmadd_ps(ymm1, ymm13, ymm8);
					ymm11 = _mm256_fmadd_ps(ymm3, ymm15, ymm11);

					ymm11 = _mm256_add_ps(ymm8, ymm11);

					ymm8 = _mm256_permute_ps(ymm11, 14);
					ymm11 = _mm256_add_ps(ymm11, ymm8);
					ymm8 = _mm256_permute_ps(ymm11, 1);
					ymm11 = _mm256_add_ps(ymm11, ymm8);

					//-----------------------------------------

					__m256 ymm10 = _mm256_blend_ps(ymm0, ymm4, 1);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
					ymm8 = _mm256_mul_ps(ymm10, ymm12);

					ymm10 = _mm256_blend_ps(ymm1, ymm5, 1);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm256_blend_ps(ymm2, ymm6, 1);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm256_blend_ps(ymm3, ymm7, 1);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm15, ymm8);

					ymm10 = _mm256_permute_ps(ymm8, 14);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm10 = _mm256_permute_ps(ymm8, 1);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 34);

					//-----------------------------------------

					__m256i ymm_mask = _mm256_permutevar8x32_epi32(ymm_mask_temp, ymm_mask_temp);

					ymm10 = _mm256_blend_ps(ymm0, ymm4, 3);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_mul_ps(ymm10, ymm12);

					ymm10 = _mm256_blend_ps(ymm1, ymm5, 3);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm256_blend_ps(ymm2, ymm6, 3);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm256_blend_ps(ymm3, ymm7, 3);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm15, ymm8);

					ymm10 = _mm256_permute_ps(ymm8, 64);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm10 = _mm256_permute_ps(ymm8, 176);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 68);

					//-----------------------------------------

					ymm_mask = _mm256_permutevar8x32_epi32(ymm_mask, ymm_mask_temp);

					ymm10 = _mm256_blend_ps(ymm0, ymm4, 7);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_mul_ps(ymm10, ymm12);

					ymm10 = _mm256_blend_ps(ymm1, ymm5, 7);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

					ymm10 = _mm256_blend_ps(ymm2, ymm6, 7);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

					ymm10 = _mm256_blend_ps(ymm3, ymm7, 7);
					ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
					ymm8 = _mm256_fmadd_ps(ymm10, ymm15, ymm8);

					ymm10 = _mm256_permute_ps(ymm8, 64);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm10 = _mm256_permute_ps(ymm8, 176);
					ymm8 = _mm256_add_ps(ymm8, ymm10);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 136);

					_mm256_store_ps(pDst, ymm11);
					pDst += REG_SIZE;

					ymm0 = ymm4;
					ymm1 = ymm5;
					ymm2 = ymm6;
					ymm3 = ymm7;
				}

				//============================================================================

				__m256 ymm4 = _mm256_setzero_ps();
				__m256 ymm5 = _mm256_setzero_ps();
				__m256 ymm6 = _mm256_setzero_ps();
				__m256 ymm7 = _mm256_setzero_ps();

				if (i + REG_SIZE < src_size_l)
				{
					ymm4 = _mm256_load_ps(pSrc0);
					ymm5 = _mm256_load_ps(pSrc1);
					ymm6 = _mm256_load_ps(pSrc2);
					ymm7 = _mm256_load_ps(pSrc3);
				}

				__m256 ymm8 = _mm256_mul_ps(ymm0, ymm12);
				__m256 ymm11 = _mm256_mul_ps(ymm2, ymm14);

				ymm8 = _mm256_fmadd_ps(ymm1, ymm13, ymm8);
				ymm11 = _mm256_fmadd_ps(ymm3, ymm15, ymm11);

				ymm11 = _mm256_add_ps(ymm8, ymm11);

				ymm8 = _mm256_permute_ps(ymm11, 14);
				ymm11 = _mm256_add_ps(ymm11, ymm8);
				ymm8 = _mm256_permute_ps(ymm11, 1);
				ymm11 = _mm256_add_ps(ymm11, ymm8);

				//-----------------------------------------

				__m256 ymm10 = _mm256_blend_ps(ymm0, ymm4, 1);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
				ymm8 = _mm256_mul_ps(ymm10, ymm12);

				ymm10 = _mm256_blend_ps(ymm1, ymm5, 1);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm256_blend_ps(ymm2, ymm6, 1);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm256_blend_ps(ymm3, ymm7, 1);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask_temp);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm15, ymm8);

				ymm10 = _mm256_permute_ps(ymm8, 14);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm10 = _mm256_permute_ps(ymm8, 1);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 34);

				//-----------------------------------------

				__m256i ymm_mask = _mm256_permutevar8x32_epi32(ymm_mask_temp, ymm_mask_temp);

				ymm10 = _mm256_blend_ps(ymm0, ymm4, 3);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_mul_ps(ymm10, ymm12);

				ymm10 = _mm256_blend_ps(ymm1, ymm5, 3);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm256_blend_ps(ymm2, ymm6, 3);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm256_blend_ps(ymm3, ymm7, 3);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm15, ymm8);

				ymm10 = _mm256_permute_ps(ymm8, 64);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm10 = _mm256_permute_ps(ymm8, 176);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 68);

				//-----------------------------------------

				ymm_mask = _mm256_permutevar8x32_epi32(ymm_mask, ymm_mask_temp);

				ymm10 = _mm256_blend_ps(ymm0, ymm4, 7);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_mul_ps(ymm10, ymm12);

				ymm10 = _mm256_blend_ps(ymm1, ymm5, 7);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm13, ymm8);

				ymm10 = _mm256_blend_ps(ymm2, ymm6, 7);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm14, ymm8);

				ymm10 = _mm256_blend_ps(ymm3, ymm7, 7);
				ymm10 = _mm256_permutevar8x32_ps(ymm10, ymm_mask);
				ymm8 = _mm256_fmadd_ps(ymm10, ymm15, ymm8);

				ymm10 = _mm256_permute_ps(ymm8, 64);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm10 = _mm256_permute_ps(ymm8, 176);
				ymm8 = _mm256_add_ps(ymm8, ymm10);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 136);

				_mm256_store_ps(pDst, ymm11);

				//============================================================================
			}
		}
		void CNNPP::conv_5x4(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };

			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 4;

			const __m256i ymm_mask = _mm256_load_si256((__m256i*)set1_mask);

			const __m256 ymm8 = _mm256_load_ps(kernel);
			const __m256 ymm9 = _mm256_load_ps(kernel + REG_SIZE);
			const __m256 ymm10 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			const __m256 ymm11 = _mm256_load_ps(kernel + 3 * REG_SIZE);
			const __m256 ymm12 = _mm256_load_ps(kernel + 4 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pSrc4 = src + (j + 4) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				for (size_t i = 0; i < L; ++i)
				{
					__m256 ymm0 = _mm256_loadu_ps(pSrc0++);
					__m256 ymm1 = _mm256_loadu_ps(pSrc1++);
					__m256 ymm2 = _mm256_loadu_ps(pSrc2++);
					__m256 ymm3 = _mm256_loadu_ps(pSrc3++);
					__m256 ymm4 = _mm256_loadu_ps(pSrc4++);

					__m256 ymm7 = _mm256_mul_ps(ymm0, ymm8);
					ymm7 = _mm256_fmadd_ps(ymm1, ymm9, ymm7);
					ymm7 = _mm256_fmadd_ps(ymm2, ymm10, ymm7);
					ymm7 = _mm256_fmadd_ps(ymm3, ymm11, ymm7);
					ymm7 = _mm256_fmadd_ps(ymm4, ymm12, ymm7);

					__m256 ymm6 = _mm256_permute_ps(ymm7, 14);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_permute_ps(ymm7, 1);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm1 = _mm256_permute2f128_ps(ymm7, ymm7, 33);
					ymm7 = _mm256_add_ps(ymm7, ymm1);

					//_mm_store_ss(pDst + i, _mm256_castps256_ps128(ymm7));
					_mm256_maskstore_ps(pDst++, ymm_mask, ymm7);
				}
			}
		}
		void CNNPP::conv_5x5(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
		}
		void CNNPP::conv_6x5(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };

			if (L == 0) L = src_size_l - 4;
			if (H == 0) H = src_size_h - 5;

			const __m256i ymm_mask = _mm256_load_si256((__m256i*)set1_mask);

			const __m256 ymm8 = _mm256_load_ps(kernel);
			const __m256 ymm9 = _mm256_load_ps(kernel + REG_SIZE);
			const __m256 ymm10 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			const __m256 ymm11 = _mm256_load_ps(kernel + 3 * REG_SIZE);
			const __m256 ymm12 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			const __m256 ymm13 = _mm256_load_ps(kernel + 5 * REG_SIZE);

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
					__m256 ymm0 = _mm256_loadu_ps(pSrc0++);
					__m256 ymm1 = _mm256_loadu_ps(pSrc1++);
					__m256 ymm2 = _mm256_loadu_ps(pSrc2++);
					__m256 ymm3 = _mm256_loadu_ps(pSrc3++);

					__m256 ymm7 = _mm256_mul_ps(ymm0, ymm8);
					__m256 ymm6 = _mm256_mul_ps(ymm1, ymm9);

					ymm7 = _mm256_fmadd_ps(ymm2, ymm10, ymm7);
					ymm6 = _mm256_fmadd_ps(ymm3, ymm11, ymm6);

					ymm0 = _mm256_loadu_ps(pSrc4++);
					ymm1 = _mm256_loadu_ps(pSrc5++);

					ymm7 = _mm256_fmadd_ps(ymm0, ymm12, ymm7);
					ymm6 = _mm256_fmadd_ps(ymm1, ymm13, ymm6);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm6 = _mm256_permute_ps(ymm7, 14);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_permute_ps(ymm7, 1);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm1 = _mm256_permute2f128_ps(ymm7, ymm7, 33);
					ymm7 = _mm256_add_ps(ymm7, ymm1);

					//_mm_store_ss(pDst + i, _mm256_castps256_ps128(ymm7));
					_mm256_maskstore_ps(pDst++, ymm_mask, ymm7);
				}
			}
		}
		void CNNPP::conv_6x6(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
		}
		void CNNPP::conv_7x7(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
		}
		void CNNPP::conv_8x7(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };

			if (L == 0) L = src_size_l - 6;
			if (H == 0) H = src_size_h - 7;

			const __m256i ymm_mask = _mm256_load_si256((__m256i*)set1_mask);

			const __m256 ymm8 = _mm256_load_ps(kernel);
			const __m256 ymm9 = _mm256_load_ps(kernel + REG_SIZE);
			const __m256 ymm10 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			const __m256 ymm11 = _mm256_load_ps(kernel + 3 * REG_SIZE);
			const __m256 ymm12 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			const __m256 ymm13 = _mm256_load_ps(kernel + 5 * REG_SIZE);
			const __m256 ymm14 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			const __m256 ymm15 = _mm256_load_ps(kernel + 7 * REG_SIZE);

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
					__m256 ymm0 = _mm256_loadu_ps(pSrc0++);
					__m256 ymm1 = _mm256_loadu_ps(pSrc1++);
					__m256 ymm2 = _mm256_loadu_ps(pSrc2++);
					__m256 ymm3 = _mm256_loadu_ps(pSrc3++);

					__m256 ymm7 = _mm256_mul_ps(ymm0, ymm8);
					__m256 ymm6 = _mm256_mul_ps(ymm1, ymm9);

					ymm7 = _mm256_fmadd_ps(ymm2, ymm10, ymm7);
					ymm6 = _mm256_fmadd_ps(ymm3, ymm11, ymm6);

					ymm0 = _mm256_loadu_ps(pSrc4++);
					ymm1 = _mm256_loadu_ps(pSrc5++);
					ymm2 = _mm256_loadu_ps(pSrc6++);
					ymm3 = _mm256_loadu_ps(pSrc7++);

					ymm7 = _mm256_fmadd_ps(ymm0, ymm12, ymm7);
					ymm6 = _mm256_fmadd_ps(ymm1, ymm13, ymm6);
					ymm7 = _mm256_fmadd_ps(ymm2, ymm14, ymm7);
					ymm6 = _mm256_fmadd_ps(ymm3, ymm15, ymm6);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm6 = _mm256_permute_ps(ymm7, 14);
					ymm7 = _mm256_add_ps(ymm7, ymm6);
					ymm6 = _mm256_permute_ps(ymm7, 1);
					ymm7 = _mm256_add_ps(ymm7, ymm6);

					ymm1 = _mm256_permute2f128_ps(ymm7, ymm7, 33);
					ymm7 = _mm256_add_ps(ymm7, ymm1);

					//_mm_store_ss(pDst + i, _mm256_castps256_ps128(ymm7));
					_mm256_maskstore_ps(pDst++, ymm_mask, ymm7);
				}
			}
		}
		void CNNPP::conv_8x8(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
		}
		void CNNPP::conv_11x10(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 9;
			if (H == 0) H = src_size_h - 10;

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc_[11];
				for (size_t y = 0; y < 11; ++y)
				{
					pSrc_[y] = src + (j + y) * src_size_l;
				}
				float* __restrict pDst = dst + j * dst_size_l;

				for (size_t i = 0; i < L; ++i)
				{
					float d = 0.f;
					float *kernel_ref = kernel;
					for (size_t y = 0; y < 11; ++y)
					{
						d += *(pSrc_[y] + 0)* * (kernel_ref + 0)
							+ *(pSrc_[y] + 1)* * (kernel_ref + 1)
							+ *(pSrc_[y] + 2)* * (kernel_ref + 2)
							+ *(pSrc_[y] + 3)* * (kernel_ref + 3)
							+ *(pSrc_[y] + 4)* * (kernel_ref + 4)
							+ *(pSrc_[y] + 5)* * (kernel_ref + 5)
							+ *(pSrc_[y] + 6)* * (kernel_ref + 6)
							+ *(pSrc_[y] + 7)* * (kernel_ref + 7)
							+ *(pSrc_[y] + 8)* * (kernel_ref + 8)
							+ *(pSrc_[y] + 9)* * (kernel_ref + 9);
						kernel_ref += 10;
					}
					*(pDst++) = d;
				}
			}
		}
		void CNNPP::conv_11x11(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
		}

#endif

		void CNNPP::tanh_avr_tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale)
		{
			ALIGN(ALIGN_DEF) float buff[2 * REG_SIZE];

			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm7 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm1 = _mm256_broadcast_ss(conv_b);
			const __m256 ymm5 = _mm256_broadcast_ss(subs_w);
			__m256 ymm0 = _mm256_broadcast_ss(subs_b);
			__m256 ymm2 = _mm256_broadcast_ss(scale);
			_mm256_store_ps(buff, ymm0);
			_mm256_store_ps(buff + REG_SIZE, ymm2);

			size_t j2 = 0;
			for (size_t j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + j2 * dst_size_l;

				for (size_t i = 0; i < src_size_l; i += 2 * REG_SIZE)
				{
					ymm0 = _mm256_load_ps(pSrc0 + i);

					__m256 ymm6 = _mm256_load_ps(pSrc0 + i + REG_SIZE);
					ymm0 = _mm256_add_ps(ymm0, ymm1);
					__m256 ymm4 = _mm256_mul_ps(ymm0, ymm0);
					__m256 ymm3 = _mm256_and_ps(ymm14, ymm0);

					ymm6 = _mm256_add_ps(ymm6, ymm1);
					__m256 ymm12 = _mm256_mul_ps(ymm4, ymm4);

					ymm3 = _mm256_add_ps(ymm15, ymm3);
					ymm2 = _mm256_andnot_ps(ymm14, ymm0);

					__m256 ymm10 = _mm256_mul_ps(ymm6, ymm6);
					ymm3 = _mm256_add_ps(ymm3, ymm4);
					__m256 ymm9 = _mm256_and_ps(ymm14, ymm6);
					ymm0 = _mm256_load_ps(pSrc1 + i);

					ymm0 = _mm256_add_ps(ymm0, ymm1);
					__m256 ymm11 = _mm256_mul_ps(ymm10, ymm10);

#ifdef USE_FMA
					ymm3 = _mm256_fmadd_ps(ymm12, ymm7, ymm3);
#else
					ymm12 = _mm256_mul_ps(ymm12, ymm7);
					ymm3 = _mm256_add_ps(ymm3, ymm12);
#endif

					__m256 ymm8 = _mm256_andnot_ps(ymm14, ymm6);

#ifdef USE_FAST_DIV
					ymm3 = _mm256_rcp_ps(ymm3);
#else
					ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

					ymm9 = _mm256_add_ps(ymm15, ymm9);
					ymm6 = _mm256_load_ps(pSrc1 + i + REG_SIZE);

					ymm6 = _mm256_add_ps(ymm6, ymm1);

					ymm4 = _mm256_mul_ps(ymm0, ymm0);
					ymm3 = _mm256_sub_ps(ymm15, ymm3);

					ymm9 = _mm256_add_ps(ymm9, ymm10);
					ymm12 = _mm256_or_ps(ymm3, ymm2);

					ymm10 = _mm256_mul_ps(ymm6, ymm6);

#ifdef USE_FMA
					ymm9 = _mm256_fmadd_ps(ymm11, ymm7, ymm9);
#else
					ymm11 = _mm256_mul_ps(ymm11, ymm7);
					ymm9 = _mm256_add_ps(ymm9, ymm11);
#endif				

					ymm3 = _mm256_and_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm9 = _mm256_rcp_ps(ymm9);
#else
					ymm9 = _mm256_div_ps(ymm15, ymm9);
#endif	

					ymm9 = _mm256_sub_ps(ymm15, ymm9);
					ymm2 = _mm256_andnot_ps(ymm14, ymm0);

					ymm0 = _mm256_mul_ps(ymm4, ymm4);


					ymm3 = _mm256_add_ps(ymm15, ymm3);
					__m256 ymm13 = _mm256_or_ps(ymm9, ymm8);

					ymm11 = _mm256_mul_ps(ymm10, ymm10);

					ymm3 = _mm256_add_ps(ymm3, ymm4);
					ymm8 = _mm256_andnot_ps(ymm14, ymm6);

					ymm4 = _mm256_load_ps(buff);
					ymm9 = _mm256_and_ps(ymm14, ymm6);

#ifdef USE_FMA
					ymm3 = _mm256_fmadd_ps(ymm0, ymm7, ymm3);
#else
					ymm0 = _mm256_mul_ps(ymm0, ymm7);
					ymm3 = _mm256_add_ps(ymm3, ymm0);
#endif

#ifdef USE_FAST_DIV
					ymm3 = _mm256_rcp_ps(ymm3);
#else
					ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

					ymm9 = _mm256_add_ps(ymm15, ymm9);

					ymm3 = _mm256_sub_ps(ymm15, ymm3);

					ymm9 = _mm256_add_ps(ymm9, ymm10);
					ymm3 = _mm256_or_ps(ymm3, ymm2);

#ifdef USE_FMA
					ymm9 = _mm256_fmadd_ps(ymm11, ymm7, ymm9);
#else
					ymm11 = _mm256_mul_ps(ymm11, ymm7);
					ymm9 = _mm256_add_ps(ymm9, ymm11);
#endif

#ifdef USE_FAST_DIV
					ymm9 = _mm256_rcp_ps(ymm9);
#else
					ymm9 = _mm256_div_ps(ymm15, ymm9);
#endif	

					ymm9 = _mm256_sub_ps(ymm15, ymm9);
					ymm9 = _mm256_or_ps(ymm9, ymm8);

					//-----------------------------

					ymm3 = _mm256_add_ps(ymm3, ymm12);
					ymm9 = _mm256_add_ps(ymm9, ymm13);

					ymm13 = _mm256_load_ps(buff + REG_SIZE);
					ymm6 = _mm256_permute2f128_ps(ymm3, ymm9, 32);
					ymm8 = _mm256_permute2f128_ps(ymm9, ymm3, 19);

					ymm2 = _mm256_shuffle_ps(ymm6, ymm8, 221);
					ymm3 = _mm256_shuffle_ps(ymm6, ymm8, 136);
					ymm0 = _mm256_add_ps(ymm2, ymm3);

					//-----------------------------

					ymm0 = _mm256_mul_ps(ymm0, ymm5);
					ymm0 = _mm256_add_ps(ymm0, ymm4);

					//-----------------------------

					ymm3 = _mm256_and_ps(ymm14, ymm0);

					ymm3 = _mm256_add_ps(ymm15, ymm3);
					ymm4 = _mm256_mul_ps(ymm0, ymm0);

					ymm3 = _mm256_add_ps(ymm3, ymm4);
					ymm8 = _mm256_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
					ymm3 = _mm256_fmadd_ps(ymm8, ymm7, ymm3);
#else
					ymm8 = _mm256_mul_ps(ymm8, ymm7);
					ymm3 = _mm256_add_ps(ymm3, ymm8);
#endif	

					ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm3 = _mm256_rcp_ps(ymm3);
#else
					ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

					ymm3 = _mm256_sub_ps(ymm15, ymm3);
					ymm3 = _mm256_or_ps(ymm3, ymm2);

					//-----------------------------

					ymm3 = _mm256_mul_ps(ymm3, ymm13);

					_mm256_store_ps(pDst, ymm3);
					pDst += REG_SIZE;
				}
				j2++;
			}
		}
		void CNNPP::max_tanh_tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale)
		{
			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm7 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm1 = _mm256_broadcast_ss(conv_b);
			const __m256 ymm5 = _mm256_broadcast_ss(subs_w);
			const __m256 ymm4 = _mm256_broadcast_ss(subs_b);
			const __m256 ymm11 = _mm256_broadcast_ss(scale);

			size_t j2 = 0;
			for (size_t j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + j2 * dst_size_l;

				for (size_t i = 0; i < src_size_l; i += 2 * REG_SIZE)
				{
					__m256 ymm3 = _mm256_load_ps(pSrc0 + i);
					__m256 ymm9 = _mm256_load_ps(pSrc0 + i + REG_SIZE);

					__m256 ymm12 = _mm256_load_ps(pSrc1 + i);
					__m256 ymm13 = _mm256_load_ps(pSrc1 + i + REG_SIZE);

					//-----------------------------

					ymm3 = _mm256_max_ps(ymm3, ymm12);
					ymm9 = _mm256_max_ps(ymm9, ymm13);

					__m256 ymm6 = _mm256_permute2f128_ps(ymm3, ymm9, 32);
					__m256 ymm8 = _mm256_permute2f128_ps(ymm9, ymm3, 19);

					ymm3 = _mm256_shuffle_ps(ymm6, ymm8, 221);
					ymm9 = _mm256_shuffle_ps(ymm6, ymm8, 136);
					__m256 ymm0 = _mm256_max_ps(ymm3, ymm9);

					//-----------------------------

					ymm0 = _mm256_add_ps(ymm0, ymm1);
					ymm3 = _mm256_and_ps(ymm14, ymm0);

					ymm3 = _mm256_add_ps(ymm15, ymm3);
					ymm6 = _mm256_mul_ps(ymm0, ymm0);

					ymm3 = _mm256_add_ps(ymm3, ymm6);
					ymm8 = _mm256_mul_ps(ymm6, ymm6);

#ifdef USE_FMA
					ymm3 = _mm256_fmadd_ps(ymm8, ymm7, ymm3);
#else
					ymm8 = _mm256_mul_ps(ymm8, ymm7);
					ymm3 = _mm256_add_ps(ymm3, ymm8);
#endif

					__m256 ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm3 = _mm256_rcp_ps(ymm3);
#else
					ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

					ymm3 = _mm256_sub_ps(ymm15, ymm3);
					ymm0 = _mm256_or_ps(ymm3, ymm2);

					//-----------------------------

#ifdef USE_FMA
					ymm0 = _mm256_fmadd_ps(ymm0, ymm5, ymm4);
#else
					ymm0 = _mm256_mul_ps(ymm0, ymm5);
					ymm0 = _mm256_add_ps(ymm0, ymm4);
#endif

					//-----------------------------

					ymm3 = _mm256_and_ps(ymm14, ymm0);

					ymm3 = _mm256_add_ps(ymm15, ymm3);
					ymm6 = _mm256_mul_ps(ymm0, ymm0);

					ymm3 = _mm256_add_ps(ymm3, ymm6);
					ymm8 = _mm256_mul_ps(ymm6, ymm6);

#ifdef USE_FMA
					ymm3 = _mm256_fmadd_ps(ymm8, ymm7, ymm3);
#else
					ymm8 = _mm256_mul_ps(ymm8, ymm7);
					ymm3 = _mm256_add_ps(ymm3, ymm8);
#endif

					ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm3 = _mm256_rcp_ps(ymm3);
#else
					ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

					ymm3 = _mm256_sub_ps(ymm15, ymm3);
					ymm3 = _mm256_or_ps(ymm3, ymm2);

					//-----------------------------

					ymm3 = _mm256_mul_ps(ymm3, ymm11);

					_mm256_store_ps(pDst, ymm3);
					pDst += REG_SIZE;
				}
				j2++;
			}
		}
		void CNNPP::max_tanh_bn(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict bn_w, float* __restrict bn_b, float* __restrict scale) { }

		void CNNPP::lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b)
		{
			//int  j20 = 0;
			//for (size_t j = 0; j < src_size_h; j += 2)
			//{
			//	float* __restrict pSrc0 = src + j * src_size_l;
			//	float* __restrict pSrc1 = src + (j + 1) * src_size_l;
			//	float* __restrict pDst = dst + j20 * dst_size_l;

			//	for (size_t i = 0; i < src_size_l; i += 2)
			//	{
			//		float c1 = *(pSrc0++) +* conv_b;
			//		c1 = *lrelu_w1 * c1 +* lrelu_w2 * fmaxf(0.f, c1) +* bn_b;

			//		float c2 = *(pSrc0++) +* conv_b;
			//		c2 = *lrelu_w1 * c2 +* lrelu_w2 * fmaxf(0.f, c2) +* bn_b;

			//		float c3 = *(pSrc1++) +* conv_b;
			//		c3 = *lrelu_w1 * c3 +* lrelu_w2 * fmaxf(0.f, c3) +* bn_b;

			//		float c4 = *(pSrc1++) +* conv_b;
			//		c4 = *lrelu_w1 * c4 +* lrelu_w2 * fmaxf(0.f, c4) +* bn_b;

			//		float pool = fmaxf(fmaxf(c1, c2), fmaxf(c3, c4));

			//		*(pDst++) = pool;
			//		break;
			//	}
			//	j20++;
			//	break;
			//}

			const __m256 ymm1 = _mm256_broadcast_ss(conv_b);
			const __m256 ymm4 = _mm256_broadcast_ss(lrelu_w1);
			const __m256 ymm5 = _mm256_broadcast_ss(lrelu_w2);
			const __m256 ymm2 = _mm256_broadcast_ss(bn_b);
			const __m256 ymm0 = _mm256_setzero_ps();

			size_t j2 = 0;
			for (size_t j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + j2 * dst_size_l;

				for (size_t i = 0; i < src_size_l; i += 2 * REG_SIZE)
				{
					__m256 ymm3 = _mm256_load_ps(pSrc0 + i);
					__m256 ymm9 = _mm256_load_ps(pSrc0 + i + REG_SIZE);

					__m256 ymm12 = _mm256_load_ps(pSrc1 + i);
					__m256 ymm13 = _mm256_load_ps(pSrc1 + i + REG_SIZE);

					//-----------------------------

					ymm3 = _mm256_add_ps(ymm3, ymm1);
					__m256 ymm7 = _mm256_max_ps(ymm3, ymm0);

#ifdef USE_FMA
					ymm3 = _mm256_fmadd_ps(ymm3, ymm4, ymm2);
					ymm3 = _mm256_fmadd_ps(ymm7, ymm5, ymm3);
#else
					ymm3 = _mm256_mul_ps(ymm3, ymm4);
					ymm7 = _mm256_mul_ps(ymm7, ymm5);

					ymm3 = _mm256_add_ps(ymm3, ymm7);
					ymm3 = _mm256_add_ps(ymm3, ymm2);
#endif

					//-----------------------------

					ymm9 = _mm256_add_ps(ymm9, ymm1);
					ymm7 = _mm256_max_ps(ymm9, ymm0);

#ifdef USE_FMA
					ymm9 = _mm256_fmadd_ps(ymm9, ymm4, ymm2);
					ymm9 = _mm256_fmadd_ps(ymm7, ymm5, ymm9);
#else
					ymm9 = _mm256_mul_ps(ymm9, ymm4);
					ymm7 = _mm256_mul_ps(ymm7, ymm5);

					ymm9 = _mm256_add_ps(ymm9, ymm7);
					ymm9 = _mm256_add_ps(ymm9, ymm2);
#endif

					//-----------------------------

					ymm12 = _mm256_add_ps(ymm12, ymm1);
					ymm7 = _mm256_max_ps(ymm12, ymm0);

#ifdef USE_FMA
					ymm12 = _mm256_fmadd_ps(ymm12, ymm4, ymm2);
					ymm12 = _mm256_fmadd_ps(ymm7, ymm5, ymm12);
#else
					ymm12 = _mm256_mul_ps(ymm12, ymm4);
					ymm7 = _mm256_mul_ps(ymm7, ymm5);

					ymm12 = _mm256_add_ps(ymm12, ymm7);
					ymm12 = _mm256_add_ps(ymm12, ymm2);
#endif

					//-----------------------------

					ymm13 = _mm256_add_ps(ymm13, ymm1);
					ymm7 = _mm256_max_ps(ymm13, ymm0);

#ifdef USE_FMA
					ymm13 = _mm256_fmadd_ps(ymm13, ymm4, ymm2);
					ymm13 = _mm256_fmadd_ps(ymm7, ymm5, ymm13);
#else
					ymm13 = _mm256_mul_ps(ymm13, ymm4);
					ymm7 = _mm256_mul_ps(ymm7, ymm5);

					ymm13 = _mm256_add_ps(ymm13, ymm7);
					ymm13 = _mm256_add_ps(ymm13, ymm2);
#endif

					//-----------------------------

					ymm3 = _mm256_max_ps(ymm3, ymm12);
					ymm9 = _mm256_max_ps(ymm9, ymm13);

					__m256 ymm6 = _mm256_permute2f128_ps(ymm3, ymm9, 32);
					__m256 ymm8 = _mm256_permute2f128_ps(ymm9, ymm3, 19);

					ymm3 = _mm256_shuffle_ps(ymm6, ymm8, 221);
					ymm9 = _mm256_shuffle_ps(ymm6, ymm8, 136);
					ymm8 = _mm256_max_ps(ymm3, ymm9);

					//-----------------------------

					_mm256_store_ps(pDst, ymm8);
					pDst += REG_SIZE;
				}
				j2++;
			}
		}
		void CNNPP::lrelu_bn(float* __restrict dst, float* __restrict src, int size_, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b)
		{
			//float* __restrict pSrc = src;
			//float* __restrict pDst = dst;
			//for (size_t i = 0; i < size_; i += 4)
			//{
			//	float c1 = *(pSrc++) +* conv_b;
			//	c1 = *lrelu_w1 * c1 +* lrelu_w2 * fmaxf(0.f, c1) +* bn_b;
			//	*(pDst++) = c1;

			//	float c2 = *(pSrc++) +* conv_b;
			//	c2 = *lrelu_w1 * c2 +* lrelu_w2 * fmaxf(0.f, c2) +* bn_b;
			//	*(pDst++) = c2;

			//	float c3 = *(pSrc++) +* conv_b;
			//	c3 = *lrelu_w1 * c3 +* lrelu_w2 * fmaxf(0.f, c3) +* bn_b;
			//	*(pDst++) = c3;

			//	float c4 = *(pSrc++) +* conv_b;
			//	c4 = *lrelu_w1 * c4 +* lrelu_w2 * fmaxf(0.f, c4) +* bn_b;
			//	*(pDst++) = c4;

			//	break;
			//}

			const __m256 ymm1 = _mm256_broadcast_ss(conv_b);
			const __m256 ymm4 = _mm256_broadcast_ss(lrelu_w1);
			const __m256 ymm5 = _mm256_broadcast_ss(lrelu_w2);
			const __m256 ymm2 = _mm256_broadcast_ss(bn_b);
			const __m256 ymm0 = _mm256_setzero_ps();

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 ymm3 = _mm256_load_ps(src + i);

				//-----------------------------

				ymm3 = _mm256_add_ps(ymm3, ymm1);
				__m256 ymm7 = _mm256_max_ps(ymm3, ymm0);

#ifdef USE_FMA
				ymm3 = _mm256_fmadd_ps(ymm3, ymm4, ymm2);
				ymm3 = _mm256_fmadd_ps(ymm7, ymm5, ymm3);
#else
				ymm3 = _mm256_mul_ps(ymm3, ymm4);
				ymm7 = _mm256_mul_ps(ymm7, ymm5);

				ymm3 = _mm256_add_ps(ymm3, ymm7);
				ymm3 = _mm256_add_ps(ymm3, ymm2);
#endif

				//-----------------------------

				_mm256_store_ps(dst + i, ymm3);
			}

		}
		void CNNPP::mulCN_add_tanhW(int N, float* __restrict dst, float** __restrict src_N, int size_, float* __restrict hl_w_N, float* __restrict hl_b, float* __restrict tanh_w, float* __restrict bn_w, float* __restrict bn_b)
		{
			//float** __restrict pSrc = new float*[N];
			//for (size_t j = 0; j < N; ++j)
			//{
			//	pSrc[j] = src_N[j];
			//}
			//float* __restrict pDst = dst;
			//for (size_t i = 0; i < size_; ++i)
			//{
			//	float c1 = *hl_b;
			//	for (size_t j = 0; j < N; ++j)
			//	{
			//		c1 += *(pSrc[j]++) * hl_w_N[j];
			//	}
			//	c1 *= *tanh_w;
			//	float sgn = 1.f;
			//	if (c1 < 0.f) sgn = -1.f;
			//	c1 = sgn * (1.f - 1.f / (1.f + fabsf(c1) + c1 * c1 + tanh_a * c1 * c1 * c1 * c1));
			//	c1 = 0.5f * c1 + 0.5f;

			//	*(pDst++) = *bn_w * c1 +* bn_b;

			//	break;
			//}
			//delete[] pSrc;

			const float scale = 0.5f;
			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm13 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm12 = _mm256_broadcast_ss(hl_b);
			const __m256 ymm11 = _mm256_broadcast_ss(&scale);
			const __m256 ymm10 = _mm256_broadcast_ss(tanh_w);
			const __m256 ymm9 = _mm256_broadcast_ss(bn_w);
			const __m256 ymm8 = _mm256_broadcast_ss(bn_b);

			__m256 ymm_hl_w_N[96];
			for (size_t i = 0; i < N; ++i)
			{
				ymm_hl_w_N[i] = _mm256_broadcast_ss(hl_w_N + i);
			}

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 ymm0 = ymm12;

				if (N % 4 == 0)
				{
					for (size_t j = 0; j < N; j += 4)
					{
#ifdef USE_FMA
						__m256 ymm7 = _mm256_load_ps(src_N[j] + i);
						ymm0 = _mm256_fmadd_ps(ymm7, ymm_hl_w_N[j], ymm0);

						ymm7 = _mm256_load_ps(src_N[j + 1] + i);
						ymm0 = _mm256_fmadd_ps(ymm7, ymm_hl_w_N[j + 1], ymm0);

						ymm7 = _mm256_load_ps(src_N[j + 2] + i);
						ymm0 = _mm256_fmadd_ps(ymm7, ymm_hl_w_N[j + 2], ymm0);

						ymm7 = _mm256_load_ps(src_N[j + 3] + i);
						ymm0 = _mm256_fmadd_ps(ymm7, ymm_hl_w_N[j + 3], ymm0);
#else
						__m256 ymm7 = _mm256_load_ps(src_N[j] + i);
						ymm7 = _mm256_mul_ps(ymm7, ymm_hl_w_N[j]);
						ymm0 = _mm256_add_ps(ymm0, ymm7);

						ymm7 = _mm256_load_ps(src_N[j + 1] + i);
						ymm7 = _mm256_mul_ps(ymm7, ymm_hl_w_N[j + 1]);
						ymm0 = _mm256_add_ps(ymm0, ymm7);

						ymm7 = _mm256_load_ps(src_N[j + 2] + i);
						ymm7 = _mm256_mul_ps(ymm7, ymm_hl_w_N[j + 2]);
						ymm0 = _mm256_add_ps(ymm0, ymm7);

						ymm7 = _mm256_load_ps(src_N[j + 3] + i);
						ymm7 = _mm256_mul_ps(ymm7, ymm_hl_w_N[j + 3]);
						ymm0 = _mm256_add_ps(ymm0, ymm7);
#endif
					}
				}
				else
				{
					for (size_t j = 0; j < N; ++j)
					{
						__m256 ymm7 = _mm256_load_ps(src_N[j] + i);
						__m256 ymm6 = _mm256_broadcast_ss(hl_w_N + j);
#ifdef USE_FMA
						ymm0 = _mm256_fmadd_ps(ymm6, ymm7, ymm0);
#else
						ymm6 = _mm256_mul_ps(ymm6, ymm7);
						ymm0 = _mm256_add_ps(ymm0, ymm6);
#endif
					}
				}

				ymm0 = _mm256_mul_ps(ymm0, ymm10);
				
				__m256 ymm1 = _mm256_and_ps(ymm14, ymm0);
				__m256 ymm3 = _mm256_add_ps(ymm15, ymm1);
				__m256 ymm4 = _mm256_mul_ps(ymm0, ymm0);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				__m256 ymm5 = _mm256_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);
#else	
				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm3 = _mm256_add_ps(ymm3, ymm5);
#endif

				__m256 ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm256_sub_ps(ymm15, ymm3);
				ymm3 = _mm256_or_ps(ymm3, ymm2);

				ymm3 = _mm256_mul_ps(ymm3, ymm11);
				ymm3 = _mm256_add_ps(ymm3, ymm11);

				ymm3 = _mm256_mul_ps(ymm3, ymm9);
				ymm3 = _mm256_add_ps(ymm3, ymm8);

				_mm256_store_ps(dst + i, ymm3);
			}
		}
		void CNNPP::tanhW(float* dst, float* src, int size_, float* __restrict snn_ol_b, float* __restrict tanh_w, float* __restrict scale)
		{
			//float* pSrc = src;
			//float* pDst = dst;
			//for (size_t i = 0; i < size_; ++i)
			//{
			//	float c1 = *(pSrc++) +* snn_ol_b;
			//	c1 *= *tanh_w;
			//	float sgn = 1.f;
			//	if (c1 < 0.f) sgn = -1.f;
			//	c1 = sgn * (1.f - 1.f / (1.f + fabsf(c1) + c1 * c1 + tanh_a * c1 * c1 * c1 * c1));
			//	//c1 = 0.5f * c1 + 0.5f;

			//	//if (index_output == 0)
			//	//	c1 *= -1.7159f;
			//	//else
			//	c1 *= 1.7159f;

			//	*(pDst++) = c1;

			//	break;
			//}

			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm13 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm12 = _mm256_broadcast_ss(snn_ol_b);
			const __m256 ymm11 = _mm256_broadcast_ss(scale);
			const __m256 ymm10 = _mm256_broadcast_ss(tanh_w);
			const __m256 ymm9 = _mm256_broadcast_ss(&half);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 ymm0 = _mm256_load_ps(src + i);
				ymm0 = _mm256_add_ps(ymm0, ymm12);
				ymm0 = _mm256_mul_ps(ymm0, ymm10);

				__m256 ymm1 = _mm256_and_ps(ymm14, ymm0);
				__m256 ymm3 = _mm256_add_ps(ymm15, ymm1);
				__m256 ymm4 = _mm256_mul_ps(ymm0, ymm0);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				__m256 ymm5 = _mm256_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);
#else	
				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm3 = _mm256_add_ps(ymm3, ymm5);
#endif

				__m256 ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm256_sub_ps(ymm15, ymm3);
				ymm3 = _mm256_or_ps(ymm3, ymm2);

				if (*scale == 0.f)
				{
					ymm3 = _mm256_mul_ps(ymm3, ymm9);
					ymm3 = _mm256_add_ps(ymm3, ymm9);
				}
				else
				{
					ymm3 = _mm256_mul_ps(ymm3, ymm11);
				}

				_mm256_store_ps(dst + i, ymm3);
			}
		}

		void CNNPP::tanh_tanh_2tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale, float* __restrict snn_hl_w0, float* __restrict snn_hl_b0, float* __restrict snn_hl_w1, float* __restrict snn_hl_b1, float* __restrict snn_ol_w0, float* __restrict snn_ol_w1)
		{
			ALIGN(ALIGN_DEF) float buff[6 * REG_SIZE];

			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm13 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm12 = _mm256_broadcast_ss(conv_b);
			const __m256 ymm11 = _mm256_broadcast_ss(subs_w);
			const __m256 ymm10 = _mm256_broadcast_ss(subs_b);
			const __m256 ymm9 = _mm256_broadcast_ss(scale);

			__m256 ymm8 = _mm256_broadcast_ss(snn_hl_w0);
			__m256 ymm7 = _mm256_broadcast_ss(snn_hl_b0);
			__m256 ymm6 = _mm256_broadcast_ss(snn_hl_w1);
			__m256 ymm5 = _mm256_broadcast_ss(snn_hl_b1);
			__m256 ymm4 = _mm256_broadcast_ss(snn_ol_w0);
			__m256 ymm3 = _mm256_broadcast_ss(snn_ol_w1);

			ymm8 = _mm256_mul_ps(ymm8, ymm9);
			ymm6 = _mm256_mul_ps(ymm6, ymm9);
			ymm4 = _mm256_mul_ps(ymm4, ymm9);
			ymm3 = _mm256_mul_ps(ymm3, ymm9);

			_mm256_store_ps(buff, ymm8);
			_mm256_store_ps(buff + REG_SIZE, ymm7);
			_mm256_store_ps(buff + 2 * REG_SIZE, ymm6);
			_mm256_store_ps(buff + 3 * REG_SIZE, ymm5);
			_mm256_store_ps(buff + 4 * REG_SIZE, ymm4);
			_mm256_store_ps(buff + 5 * REG_SIZE, ymm3);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 ymm0 = _mm256_load_ps(src + i);
				ymm0 = _mm256_add_ps(ymm0, ymm12);
				__m256 ymm1 = _mm256_and_ps(ymm14, ymm0);

				ymm3 = _mm256_add_ps(ymm15, ymm1);
				ymm4 = _mm256_mul_ps(ymm0, ymm0);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				ymm5 = _mm256_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);
#else
				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm3 = _mm256_add_ps(ymm3, ymm5);
#endif	

				__m256 ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm256_sub_ps(ymm15, ymm3);
				ymm3 = _mm256_or_ps(ymm3, ymm2);

#ifdef USE_FMA
				ymm0 = _mm256_fmadd_ps(ymm3, ymm11, ymm10);
#else
				ymm3 = _mm256_mul_ps(ymm3, ymm11);
				ymm0 = _mm256_add_ps(ymm3, ymm10);
#endif	

				ymm1 = _mm256_and_ps(ymm14, ymm0);

				ymm3 = _mm256_add_ps(ymm15, ymm1);
				ymm4 = _mm256_mul_ps(ymm0, ymm0);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				ymm5 = _mm256_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);
#else
				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm3 = _mm256_add_ps(ymm3, ymm5);
#endif

				ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm256_sub_ps(ymm15, ymm3);
				ymm6 = _mm256_or_ps(ymm3, ymm2);

#ifdef USE_FMA
				ymm0 = _mm256_fmadd_ps(ymm6, _mm256_load_ps(buff), _mm256_load_ps(buff + REG_SIZE));
#else
				ymm0 = _mm256_mul_ps(ymm6, _mm256_load_ps(buff));
				ymm0 = _mm256_add_ps(ymm0, _mm256_load_ps(buff + REG_SIZE));
#endif

				ymm1 = _mm256_and_ps(ymm14, ymm0);

				ymm3 = _mm256_add_ps(ymm15, ymm1);
				ymm4 = _mm256_mul_ps(ymm0, ymm0);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				ymm5 = _mm256_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);
#else
				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm3 = _mm256_add_ps(ymm3, ymm5);
#endif

				ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif

				ymm3 = _mm256_sub_ps(ymm15, ymm3);
				ymm7 = _mm256_or_ps(ymm3, ymm2);

#ifdef USE_FMA
				ymm0 = _mm256_fmadd_ps(ymm6, _mm256_load_ps(buff + 2 * REG_SIZE), _mm256_load_ps(buff + 3 * REG_SIZE));
#else
				ymm0 = _mm256_mul_ps(ymm6, _mm256_load_ps(buff + 2 * REG_SIZE));
				ymm0 = _mm256_add_ps(ymm0, _mm256_load_ps(buff + 3 * REG_SIZE));
#endif

				ymm1 = _mm256_and_ps(ymm14, ymm0);

				ymm3 = _mm256_add_ps(ymm15, ymm1);
				ymm4 = _mm256_mul_ps(ymm0, ymm0);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				ymm5 = _mm256_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);
#else
				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm3 = _mm256_add_ps(ymm3, ymm5);
#endif

				ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif

				ymm3 = _mm256_sub_ps(ymm15, ymm3);
				ymm8 = _mm256_or_ps(ymm3, ymm2);

				ymm7 = _mm256_mul_ps(ymm7, _mm256_load_ps(buff + 4 * REG_SIZE));

#ifdef USE_FMA
				ymm8 = _mm256_fmadd_ps(ymm8, _mm256_load_ps(buff + 5 * REG_SIZE), ymm7);
#else
				ymm8 = _mm256_mul_ps(ymm8, _mm256_load_ps(buff + 5 * REG_SIZE));
				ymm8 = _mm256_add_ps(ymm8, ymm7);
#endif

				_mm256_store_ps(dst + i, ymm8);
			}
		}
		void CNNPP::tanh_tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale)
		{
			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm13 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm12 = _mm256_broadcast_ss(conv_b);
			const __m256 ymm11 = _mm256_broadcast_ss(subs_w);
			const __m256 ymm10 = _mm256_broadcast_ss(subs_b);
			const __m256 ymm9 = _mm256_broadcast_ss(scale);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 ymm0 = _mm256_load_ps(src + i);
				ymm0 = _mm256_add_ps(ymm0, ymm12);
				__m256 ymm1 = _mm256_and_ps(ymm14, ymm0);

				__m256 ymm3 = _mm256_add_ps(ymm15, ymm1);
				__m256 ymm4 = _mm256_mul_ps(ymm0, ymm0);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				__m256 ymm5 = _mm256_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);
#else	
				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm3 = _mm256_add_ps(ymm3, ymm5);
#endif

				__m256 ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm256_sub_ps(ymm15, ymm3);
				ymm3 = _mm256_or_ps(ymm3, ymm2);

#ifdef USE_FMA
				ymm0 = _mm256_fmadd_ps(ymm3, ymm11, ymm10);
#else	
				ymm3 = _mm256_mul_ps(ymm3, ymm11);
				ymm0 = _mm256_add_ps(ymm3, ymm10);
#endif

				ymm1 = _mm256_and_ps(ymm14, ymm0);

				ymm3 = _mm256_add_ps(ymm15, ymm1);
				ymm4 = _mm256_mul_ps(ymm0, ymm0);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				ymm5 = _mm256_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);
#else	
				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm3 = _mm256_add_ps(ymm3, ymm5);
#endif

				ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm256_sub_ps(ymm15, ymm3);
				ymm3 = _mm256_or_ps(ymm3, ymm2);

				__m256 ymm6 = _mm256_mul_ps(ymm3, ymm9);

				_mm256_store_ps(dst + i, ymm6);
			}
		}
		void CNNPP::tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict snn_ol_b, float* __restrict scale)
		{
			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm13 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm12 = _mm256_broadcast_ss(snn_ol_b);
			const __m256 ymm11 = _mm256_broadcast_ss(scale);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 ymm0 = _mm256_load_ps(src + i);
				ymm0 = _mm256_add_ps(ymm0, ymm12);

				__m256 ymm1 = _mm256_and_ps(ymm14, ymm0);
				__m256 ymm3 = _mm256_add_ps(ymm15, ymm1);
				__m256 ymm4 = _mm256_mul_ps(ymm0, ymm0);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				__m256 ymm5 = _mm256_mul_ps(ymm4, ymm4);

#ifdef USE_FMA
				ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);
#else	
				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm3 = _mm256_add_ps(ymm3, ymm5);
#endif

				__m256 ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

				ymm3 = _mm256_sub_ps(ymm15, ymm3);
				ymm3 = _mm256_or_ps(ymm3, ymm2);

				ymm3 = _mm256_mul_ps(ymm3, ymm11);

				_mm256_store_ps(dst + i, ymm3);
			}
		}

		void CNNPP::tanh_approx_exp(float* __restrict dst, float* __restrict src, int size_, float* __restrict snn_ol_b, float* __restrict scale)
		{
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			/// Fast logistic using Angus Grandison's:																											  ///
			/// "A Fast, Streaming SIMD Extensions 2, Logistic Squashing Function", Milner J. J., Grandison A. J., Neural Computation, 20(12), pp2967-2972, 2008. ///
			/// C++/SSE-intrinsics translation by Guille D. Canas.																								  ///
			/// source: http://web.mit.edu/guilledc/www/ssefastlogistic.h																						  ///
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			const __m256 maxx = _mm256_set1_ps(87.f / 2.f);
			const __m256 minx = _mm256_set1_ps(-87.f / 2.f);
			const __m256 c = _mm256_set1_ps(2.f * 8388608.f / 0.6931471806f);
			const __m256 b = _mm256_set1_ps(1065353216.f);

			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm12 = _mm256_broadcast_ss(snn_ol_b);
			const __m256 ymm11 = _mm256_broadcast_ss(scale);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 x = _mm256_load_ps(src + i);
				x = _mm256_add_ps(x, ymm12);

				const __m256 y = _mm256_max_ps(minx, _mm256_min_ps(maxx, x));	// clamp to [-87, 87]

#ifdef USE_FMA
				__m256 z = _mm256_fmadd_ps(y, c, b);
#else	
				__m256 z = _mm256_add_ps(_mm256_mul_ps(y, c), b);
#endif

				const __m256i zi = _mm256_cvtps_epi32(z);
				const __m256 exp = _mm256_load_ps((const float* )&zi);
				const __m256 ax = _mm256_sub_ps(exp, ymm15);
				const __m256 bx = _mm256_add_ps(exp, ymm15);

#ifdef USE_FAST_DIV
				z = _mm256_rcp_ps(bx);
				z = _mm256_mul_ps(ax, z);
#else
				z = _mm256_div_ps(ax, bx);
#endif	

				z = _mm256_mul_ps(z, ymm11);

				_mm256_store_ps(dst + i, z);
			}
		}
		void CNNPP::relu(float* __restrict dst, float* __restrict src, int size_, float* __restrict snn_ol_b, float* __restrict scale)
		{
			const __m256 zero = _mm256_set1_ps(0.f);
			const __m256 ymm12 = _mm256_broadcast_ss(snn_ol_b);
			const __m256 ymm11 = _mm256_broadcast_ss(scale);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 x = _mm256_load_ps(src + i);
				x = _mm256_add_ps(x, ymm12);

				__m256 y = _mm256_max_ps(zero, x);

				y = _mm256_mul_ps(y, ymm11);

				_mm256_store_ps(dst + i, y);
			}
		}

		void CNNPP::add(float* __restrict dst, float* __restrict src1, float* __restrict src2, int size_)
		{
			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 ymm0 = _mm256_load_ps(src1 + i);
				__m256 ymm1 = _mm256_load_ps(src2 + i);
				ymm0 = _mm256_add_ps(ymm0, ymm1);
				_mm256_store_ps(dst + i, ymm0);
			}
		}
		void CNNPP::add2(float* __restrict dst, float* __restrict src1, float* __restrict src2, float* __restrict src3, int size_)
		{
			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 ymm0 = _mm256_load_ps(src1 + i);
				__m256 ymm1 = _mm256_load_ps(src2 + i);
				__m256 ymm2 = _mm256_load_ps(src3 + i);
				ymm0 = _mm256_add_ps(ymm0, ymm1);
				ymm0 = _mm256_add_ps(ymm0, ymm2);
				_mm256_store_ps(dst + i, ymm0);
			}
		}

		void CNNPP::mulC(float* __restrict dst, float* __restrict src_mulC, int size_, float* __restrict snn_ol_w)
		{
			const __m256 ymm1 = _mm256_broadcast_ss(snn_ol_w);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 ymm0 = _mm256_load_ps(src_mulC + i);
				__m256 ymm2 = _mm256_mul_ps(ymm0, ymm1);
				_mm256_store_ps(dst + i, ymm2);
			}
		}
		void CNNPP::mulC1_add(float* __restrict dst, float* __restrict src1_mulC, float* __restrict src2, int size_, float* __restrict snn_hl_w)
		{
			const __m256 ymm3 = _mm256_broadcast_ss(snn_hl_w);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 ymm0 = _mm256_load_ps(src1_mulC + i);
				__m256 ymm1 = _mm256_load_ps(src2 + i);

#ifdef USE_FMA
				ymm0 = _mm256_fmadd_ps(ymm0, ymm3, ymm1);
#else
				ymm0 = _mm256_mul_ps(ymm0, ymm3);
				ymm0 = _mm256_add_ps(ymm0, ymm1);
#endif

				_mm256_store_ps(dst + i, ymm0);
			}
		}
		void CNNPP::mulC2_add(float* __restrict dst, float* __restrict src1_mulC0, float* __restrict src2_mulC1, int size_, float* __restrict snn_hl_w0, float* __restrict snn_hl_w1)
		{
			const __m256 ymm3 = _mm256_broadcast_ss(snn_hl_w0);
			const __m256 ymm4 = _mm256_broadcast_ss(snn_hl_w1);

			for (size_t i = 0; i < size_; i += REG_SIZE)
			{
				__m256 ymm0 = _mm256_load_ps(src1_mulC0 + i);
				__m256 ymm1 = _mm256_load_ps(src2_mulC1 + i);

#ifdef USE_FMA
				ymm0 = _mm256_mul_ps(ymm0, ymm3);
				ymm0 = _mm256_fmadd_ps(ymm1, ymm4, ymm0);
#else
				ymm0 = _mm256_mul_ps(ymm0, ymm3);
				ymm1 = _mm256_mul_ps(ymm1, ymm4);
				ymm0 = _mm256_add_ps(ymm0, ymm1);
#endif

				_mm256_store_ps(dst + i, ymm0);
			}
		}

#ifndef USE_FMA

		void CNNPP::mulC24_add_tanh(float* __restrict dst, float* __restrict* src, int size_, float* __restrict snn_hl_w, float* __restrict snn_hl_b, float* __restrict scale, float* __restrict snn_ol_w)
		{
			__m256 ymm0 = _mm256_broadcast_ss(snn_hl_b);
			__m256 ymm9 = _mm256_broadcast_ss(snn_ol_w);

			__m256 ymm11 = _mm256_broadcast_ss(scale);
			ymm9 = _mm256_mul_ps(ymm9, ymm11);

			for (size_t i = 0; i < size_; i += 2 * REG_SIZE)
			{
				__m256 ymm14 = ymm0;
				__m256 ymm15 = ymm0;

				//4
				//-----------------------------------

				__m256 ymm10 = _mm256_broadcast_ss(snn_hl_w);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 1);
				__m256 ymm12 = _mm256_broadcast_ss(snn_hl_w + 2);
				__m256 ymm13 = _mm256_broadcast_ss(snn_hl_w + 3);

				__m256 ymm1 = _mm256_load_ps(src[0] + i);
				__m256 ymm2 = _mm256_load_ps(src[0] + REG_SIZE + i);
				__m256 ymm3 = _mm256_load_ps(src[1] + i);
				__m256 ymm4 = _mm256_load_ps(src[1] + REG_SIZE + i);
				__m256 ymm5 = _mm256_load_ps(src[2] + i);
				__m256 ymm6 = _mm256_load_ps(src[2] + REG_SIZE + i);
				__m256 ymm7 = _mm256_load_ps(src[3] + i);
				__m256 ymm8 = _mm256_load_ps(src[3] + REG_SIZE + i);

				ymm1 = _mm256_mul_ps(ymm1, ymm10);
				ymm14 = _mm256_add_ps(ymm14, ymm1);
				ymm2 = _mm256_mul_ps(ymm2, ymm10);
				ymm15 = _mm256_add_ps(ymm15, ymm2);

				ymm3 = _mm256_mul_ps(ymm3, ymm11);
				ymm14 = _mm256_add_ps(ymm14, ymm3);
				ymm4 = _mm256_mul_ps(ymm4, ymm11);
				ymm15 = _mm256_add_ps(ymm15, ymm4);

				ymm5 = _mm256_mul_ps(ymm5, ymm12);
				ymm14 = _mm256_add_ps(ymm14, ymm5);
				ymm6 = _mm256_mul_ps(ymm6, ymm12);
				ymm15 = _mm256_add_ps(ymm15, ymm6);

				ymm7 = _mm256_mul_ps(ymm7, ymm13);
				ymm14 = _mm256_add_ps(ymm14, ymm7);
				ymm8 = _mm256_mul_ps(ymm8, ymm13);
				ymm15 = _mm256_add_ps(ymm15, ymm8);

				//8
				//-----------------------------------

				ymm10 = _mm256_broadcast_ss(snn_hl_w + 4);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 5);
				ymm12 = _mm256_broadcast_ss(snn_hl_w + 6);
				ymm13 = _mm256_broadcast_ss(snn_hl_w + 7);

				ymm1 = _mm256_load_ps(src[4] + i);
				ymm2 = _mm256_load_ps(src[4] + REG_SIZE + i);
				ymm3 = _mm256_load_ps(src[5] + i);
				ymm4 = _mm256_load_ps(src[5] + REG_SIZE + i);
				ymm5 = _mm256_load_ps(src[6] + i);
				ymm6 = _mm256_load_ps(src[6] + REG_SIZE + i);
				ymm7 = _mm256_load_ps(src[7] + i);
				ymm8 = _mm256_load_ps(src[7] + REG_SIZE + i);

				ymm1 = _mm256_mul_ps(ymm1, ymm10);
				ymm14 = _mm256_add_ps(ymm14, ymm1);
				ymm2 = _mm256_mul_ps(ymm2, ymm10);
				ymm15 = _mm256_add_ps(ymm15, ymm2);

				ymm3 = _mm256_mul_ps(ymm3, ymm11);
				ymm14 = _mm256_add_ps(ymm14, ymm3);
				ymm4 = _mm256_mul_ps(ymm4, ymm11);
				ymm15 = _mm256_add_ps(ymm15, ymm4);

				ymm5 = _mm256_mul_ps(ymm5, ymm12);
				ymm14 = _mm256_add_ps(ymm14, ymm5);
				ymm6 = _mm256_mul_ps(ymm6, ymm12);
				ymm15 = _mm256_add_ps(ymm15, ymm6);

				ymm7 = _mm256_mul_ps(ymm7, ymm13);
				ymm14 = _mm256_add_ps(ymm14, ymm7);
				ymm8 = _mm256_mul_ps(ymm8, ymm13);
				ymm15 = _mm256_add_ps(ymm15, ymm8);

				//12
				//-----------------------------------

				ymm10 = _mm256_broadcast_ss(snn_hl_w + 8);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 9);
				ymm12 = _mm256_broadcast_ss(snn_hl_w + 10);
				ymm13 = _mm256_broadcast_ss(snn_hl_w + 11);

				ymm1 = _mm256_load_ps(src[8] + i);
				ymm2 = _mm256_load_ps(src[8] + REG_SIZE + i);
				ymm3 = _mm256_load_ps(src[9] + i);
				ymm4 = _mm256_load_ps(src[9] + REG_SIZE + i);
				ymm5 = _mm256_load_ps(src[10] + i);
				ymm6 = _mm256_load_ps(src[10] + REG_SIZE + i);
				ymm7 = _mm256_load_ps(src[11] + i);
				ymm8 = _mm256_load_ps(src[11] + REG_SIZE + i);

				ymm1 = _mm256_mul_ps(ymm1, ymm10);
				ymm14 = _mm256_add_ps(ymm14, ymm1);
				ymm2 = _mm256_mul_ps(ymm2, ymm10);
				ymm15 = _mm256_add_ps(ymm15, ymm2);

				ymm3 = _mm256_mul_ps(ymm3, ymm11);
				ymm14 = _mm256_add_ps(ymm14, ymm3);
				ymm4 = _mm256_mul_ps(ymm4, ymm11);
				ymm15 = _mm256_add_ps(ymm15, ymm4);

				ymm5 = _mm256_mul_ps(ymm5, ymm12);
				ymm14 = _mm256_add_ps(ymm14, ymm5);
				ymm6 = _mm256_mul_ps(ymm6, ymm12);
				ymm15 = _mm256_add_ps(ymm15, ymm6);

				ymm7 = _mm256_mul_ps(ymm7, ymm13);
				ymm14 = _mm256_add_ps(ymm14, ymm7);
				ymm8 = _mm256_mul_ps(ymm8, ymm13);
				ymm15 = _mm256_add_ps(ymm15, ymm8);

				//16
				//-----------------------------------

				ymm10 = _mm256_broadcast_ss(snn_hl_w + 12);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 13);
				ymm12 = _mm256_broadcast_ss(snn_hl_w + 14);
				ymm13 = _mm256_broadcast_ss(snn_hl_w + 15);

				ymm1 = _mm256_load_ps(src[12] + i);
				ymm2 = _mm256_load_ps(src[12] + REG_SIZE + i);
				ymm3 = _mm256_load_ps(src[13] + i);
				ymm4 = _mm256_load_ps(src[13] + REG_SIZE + i);
				ymm5 = _mm256_load_ps(src[14] + i);
				ymm6 = _mm256_load_ps(src[14] + REG_SIZE + i);
				ymm7 = _mm256_load_ps(src[15] + i);
				ymm8 = _mm256_load_ps(src[15] + REG_SIZE + i);

				ymm1 = _mm256_mul_ps(ymm1, ymm10);
				ymm14 = _mm256_add_ps(ymm14, ymm1);
				ymm2 = _mm256_mul_ps(ymm2, ymm10);
				ymm15 = _mm256_add_ps(ymm15, ymm2);

				ymm3 = _mm256_mul_ps(ymm3, ymm11);
				ymm14 = _mm256_add_ps(ymm14, ymm3);
				ymm4 = _mm256_mul_ps(ymm4, ymm11);
				ymm15 = _mm256_add_ps(ymm15, ymm4);

				ymm5 = _mm256_mul_ps(ymm5, ymm12);
				ymm14 = _mm256_add_ps(ymm14, ymm5);
				ymm6 = _mm256_mul_ps(ymm6, ymm12);
				ymm15 = _mm256_add_ps(ymm15, ymm6);

				ymm7 = _mm256_mul_ps(ymm7, ymm13);
				ymm14 = _mm256_add_ps(ymm14, ymm7);
				ymm8 = _mm256_mul_ps(ymm8, ymm13);
				ymm15 = _mm256_add_ps(ymm15, ymm8);

				//20
				//-----------------------------------

				ymm10 = _mm256_broadcast_ss(snn_hl_w + 16);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 17);
				ymm12 = _mm256_broadcast_ss(snn_hl_w + 18);
				ymm13 = _mm256_broadcast_ss(snn_hl_w + 19);

				ymm1 = _mm256_load_ps(src[16] + i);
				ymm2 = _mm256_load_ps(src[16] + REG_SIZE + i);
				ymm3 = _mm256_load_ps(src[17] + i);
				ymm4 = _mm256_load_ps(src[17] + REG_SIZE + i);
				ymm5 = _mm256_load_ps(src[18] + i);
				ymm6 = _mm256_load_ps(src[18] + REG_SIZE + i);
				ymm7 = _mm256_load_ps(src[19] + i);
				ymm8 = _mm256_load_ps(src[19] + REG_SIZE + i);

				ymm1 = _mm256_mul_ps(ymm1, ymm10);
				ymm14 = _mm256_add_ps(ymm14, ymm1);
				ymm2 = _mm256_mul_ps(ymm2, ymm10);
				ymm15 = _mm256_add_ps(ymm15, ymm2);

				ymm3 = _mm256_mul_ps(ymm3, ymm11);
				ymm14 = _mm256_add_ps(ymm14, ymm3);
				ymm4 = _mm256_mul_ps(ymm4, ymm11);
				ymm15 = _mm256_add_ps(ymm15, ymm4);

				ymm5 = _mm256_mul_ps(ymm5, ymm12);
				ymm14 = _mm256_add_ps(ymm14, ymm5);
				ymm6 = _mm256_mul_ps(ymm6, ymm12);
				ymm15 = _mm256_add_ps(ymm15, ymm6);

				ymm7 = _mm256_mul_ps(ymm7, ymm13);
				ymm14 = _mm256_add_ps(ymm14, ymm7);
				ymm8 = _mm256_mul_ps(ymm8, ymm13);
				ymm15 = _mm256_add_ps(ymm15, ymm8);

				//24
				//-----------------------------------

				ymm10 = _mm256_broadcast_ss(snn_hl_w + 20);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 21);
				ymm12 = _mm256_broadcast_ss(snn_hl_w + 22);
				ymm13 = _mm256_broadcast_ss(snn_hl_w + 23);

				ymm1 = _mm256_load_ps(src[20] + i);
				ymm2 = _mm256_load_ps(src[20] + REG_SIZE + i);
				ymm3 = _mm256_load_ps(src[21] + i);
				ymm4 = _mm256_load_ps(src[21] + REG_SIZE + i);
				ymm5 = _mm256_load_ps(src[22] + i);
				ymm6 = _mm256_load_ps(src[22] + REG_SIZE + i);
				ymm7 = _mm256_load_ps(src[23] + i);
				ymm8 = _mm256_load_ps(src[23] + REG_SIZE + i);

				ymm1 = _mm256_mul_ps(ymm1, ymm10);
				ymm14 = _mm256_add_ps(ymm14, ymm1);
				ymm2 = _mm256_mul_ps(ymm2, ymm10);
				ymm15 = _mm256_add_ps(ymm15, ymm2);

				ymm3 = _mm256_mul_ps(ymm3, ymm11);
				ymm14 = _mm256_add_ps(ymm14, ymm3);
				ymm4 = _mm256_mul_ps(ymm4, ymm11);
				ymm15 = _mm256_add_ps(ymm15, ymm4);

				ymm5 = _mm256_mul_ps(ymm5, ymm12);
				ymm14 = _mm256_add_ps(ymm14, ymm5);
				ymm6 = _mm256_mul_ps(ymm6, ymm12);
				ymm15 = _mm256_add_ps(ymm15, ymm6);

				ymm7 = _mm256_mul_ps(ymm7, ymm13);
				ymm14 = _mm256_add_ps(ymm14, ymm7);
				ymm8 = _mm256_mul_ps(ymm8, ymm13);
				ymm15 = _mm256_add_ps(ymm15, ymm8);

				//tanh
				ymm7 = _mm256_broadcast_ss((float*)&abs_mask);
				ymm8 = _mm256_broadcast_ss(&one);
				ymm13 = _mm256_broadcast_ss(&tanh_a);

				ymm1 = _mm256_and_ps(ymm7, ymm14);
				ymm3 = _mm256_add_ps(ymm8, ymm1);
				ymm4 = _mm256_mul_ps(ymm14, ymm14);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				ymm5 = _mm256_mul_ps(ymm4, ymm4);

				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm3 = _mm256_add_ps(ymm3, ymm5);
				ymm2 = _mm256_andnot_ps(ymm7, ymm14);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm8, ymm3);
#endif	

				ymm3 = _mm256_sub_ps(ymm8, ymm3);
				ymm3 = _mm256_or_ps(ymm3, ymm2);

				ymm14 = _mm256_mul_ps(ymm3, ymm9);
				_mm256_store_ps(dst + i, ymm14);

				ymm1 = _mm256_and_ps(ymm7, ymm15);
				ymm3 = _mm256_add_ps(ymm8, ymm1);
				ymm4 = _mm256_mul_ps(ymm15, ymm15);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				ymm5 = _mm256_mul_ps(ymm4, ymm4);

				ymm5 = _mm256_mul_ps(ymm5, ymm13);
				ymm3 = _mm256_add_ps(ymm3, ymm5);
				ymm2 = _mm256_andnot_ps(ymm7, ymm15);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm8, ymm3);
#endif	

				ymm3 = _mm256_sub_ps(ymm8, ymm3);
				ymm3 = _mm256_or_ps(ymm3, ymm2);

				ymm15 = _mm256_mul_ps(ymm3, ymm9);
				_mm256_store_ps(dst + REG_SIZE + i, ymm15);
			}
		}

#else

		void CNNPP::mulC24_add_tanh(float* __restrict dst, float* __restrict* src, int size_, float* __restrict snn_hl_w, float* __restrict snn_hl_b, float* __restrict scale, float* __restrict snn_ol_w)
		{
			__m256 ymm0 = _mm256_broadcast_ss(snn_hl_b);
			__m256 ymm9 = _mm256_broadcast_ss(snn_ol_w);

			__m256 ymm11 = _mm256_broadcast_ss(scale);
			ymm9 = _mm256_mul_ps(ymm9, ymm11);

			for (size_t i = 0; i < size_; i += 2 * REG_SIZE)
			{
				__m256 ymm14 = ymm0;
				__m256 ymm15 = ymm0;

				//4
				//-----------------------------------

				__m256 ymm10 = _mm256_broadcast_ss(snn_hl_w);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 1);
				__m256 ymm12 = _mm256_broadcast_ss(snn_hl_w + 2);
				__m256 ymm13 = _mm256_broadcast_ss(snn_hl_w + 3);

				__m256 ymm1 = _mm256_load_ps(src[0] + i);
				__m256 ymm2 = _mm256_load_ps(src[0] + REG_SIZE + i);
				__m256 ymm3 = _mm256_load_ps(src[1] + i);
				__m256 ymm4 = _mm256_load_ps(src[1] + REG_SIZE + i);
				__m256 ymm5 = _mm256_load_ps(src[2] + i);
				__m256 ymm6 = _mm256_load_ps(src[2] + REG_SIZE + i);
				__m256 ymm7 = _mm256_load_ps(src[3] + i);
				__m256 ymm8 = _mm256_load_ps(src[3] + REG_SIZE + i);

				ymm14 = _mm256_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm8, ymm13, ymm15);

				//8
				//-----------------------------------

				ymm10 = _mm256_broadcast_ss(snn_hl_w + 4);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 5);
				ymm12 = _mm256_broadcast_ss(snn_hl_w + 6);
				ymm13 = _mm256_broadcast_ss(snn_hl_w + 7);

				ymm1 = _mm256_load_ps(src[4] + i);
				ymm2 = _mm256_load_ps(src[4] + REG_SIZE + i);
				ymm3 = _mm256_load_ps(src[5] + i);
				ymm4 = _mm256_load_ps(src[5] + REG_SIZE + i);
				ymm5 = _mm256_load_ps(src[6] + i);
				ymm6 = _mm256_load_ps(src[6] + REG_SIZE + i);
				ymm7 = _mm256_load_ps(src[7] + i);
				ymm8 = _mm256_load_ps(src[7] + REG_SIZE + i);

				ymm14 = _mm256_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm8, ymm13, ymm15);

				//12
				//-----------------------------------

				ymm10 = _mm256_broadcast_ss(snn_hl_w + 8);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 9);
				ymm12 = _mm256_broadcast_ss(snn_hl_w + 10);
				ymm13 = _mm256_broadcast_ss(snn_hl_w + 11);

				ymm1 = _mm256_load_ps(src[8] + i);
				ymm2 = _mm256_load_ps(src[8] + REG_SIZE + i);
				ymm3 = _mm256_load_ps(src[9] + i);
				ymm4 = _mm256_load_ps(src[9] + REG_SIZE + i);
				ymm5 = _mm256_load_ps(src[10] + i);
				ymm6 = _mm256_load_ps(src[10] + REG_SIZE + i);
				ymm7 = _mm256_load_ps(src[11] + i);
				ymm8 = _mm256_load_ps(src[11] + REG_SIZE + i);

				ymm14 = _mm256_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm8, ymm13, ymm15);

				//16
				//-----------------------------------

				ymm10 = _mm256_broadcast_ss(snn_hl_w + 12);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 13);
				ymm12 = _mm256_broadcast_ss(snn_hl_w + 14);
				ymm13 = _mm256_broadcast_ss(snn_hl_w + 15);

				ymm1 = _mm256_load_ps(src[12] + i);
				ymm2 = _mm256_load_ps(src[12] + REG_SIZE + i);
				ymm3 = _mm256_load_ps(src[13] + i);
				ymm4 = _mm256_load_ps(src[13] + REG_SIZE + i);
				ymm5 = _mm256_load_ps(src[14] + i);
				ymm6 = _mm256_load_ps(src[14] + REG_SIZE + i);
				ymm7 = _mm256_load_ps(src[15] + i);
				ymm8 = _mm256_load_ps(src[15] + REG_SIZE + i);

				ymm14 = _mm256_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm8, ymm13, ymm15);

				//20
				//-----------------------------------

				ymm10 = _mm256_broadcast_ss(snn_hl_w + 16);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 17);
				ymm12 = _mm256_broadcast_ss(snn_hl_w + 18);
				ymm13 = _mm256_broadcast_ss(snn_hl_w + 19);

				ymm1 = _mm256_load_ps(src[16] + i);
				ymm2 = _mm256_load_ps(src[16] + REG_SIZE + i);
				ymm3 = _mm256_load_ps(src[17] + i);
				ymm4 = _mm256_load_ps(src[17] + REG_SIZE + i);
				ymm5 = _mm256_load_ps(src[18] + i);
				ymm6 = _mm256_load_ps(src[18] + REG_SIZE + i);
				ymm7 = _mm256_load_ps(src[19] + i);
				ymm8 = _mm256_load_ps(src[19] + REG_SIZE + i);

				ymm14 = _mm256_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm8, ymm13, ymm15);

				//24
				//-----------------------------------

				ymm10 = _mm256_broadcast_ss(snn_hl_w + 20);
				ymm11 = _mm256_broadcast_ss(snn_hl_w + 21);
				ymm12 = _mm256_broadcast_ss(snn_hl_w + 22);
				ymm13 = _mm256_broadcast_ss(snn_hl_w + 23);

				ymm1 = _mm256_load_ps(src[20] + i);
				ymm2 = _mm256_load_ps(src[20] + REG_SIZE + i);
				ymm3 = _mm256_load_ps(src[21] + i);
				ymm4 = _mm256_load_ps(src[21] + REG_SIZE + i);
				ymm5 = _mm256_load_ps(src[22] + i);
				ymm6 = _mm256_load_ps(src[22] + REG_SIZE + i);
				ymm7 = _mm256_load_ps(src[23] + i);
				ymm8 = _mm256_load_ps(src[23] + REG_SIZE + i);

				ymm14 = _mm256_fmadd_ps(ymm1, ymm10, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm2, ymm10, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm3, ymm11, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm4, ymm11, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm5, ymm12, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm6, ymm12, ymm15);

				ymm14 = _mm256_fmadd_ps(ymm7, ymm13, ymm14);
				ymm15 = _mm256_fmadd_ps(ymm8, ymm13, ymm15);

				//tanh
				ymm7 = _mm256_broadcast_ss((float*)&abs_mask);
				ymm8 = _mm256_broadcast_ss(&one);
				ymm13 = _mm256_broadcast_ss(&tanh_a);

				ymm1 = _mm256_and_ps(ymm7, ymm14);
				ymm3 = _mm256_add_ps(ymm8, ymm1);
				ymm4 = _mm256_mul_ps(ymm14, ymm14);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				ymm5 = _mm256_mul_ps(ymm4, ymm4);

				//ymm5 = _mm256_mul_ps(ymm5, ymm13);
				//ymm3 = _mm256_add_ps(ymm3, ymm5);
				ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);
				ymm2 = _mm256_andnot_ps(ymm7, ymm14);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm8, ymm3);
#endif	

				ymm3 = _mm256_sub_ps(ymm8, ymm3);
				ymm3 = _mm256_or_ps(ymm3, ymm2);

				ymm14 = _mm256_mul_ps(ymm3, ymm9);
				_mm256_store_ps(dst + i, ymm14);

				ymm1 = _mm256_and_ps(ymm7, ymm15);
				ymm3 = _mm256_add_ps(ymm8, ymm1);
				ymm4 = _mm256_mul_ps(ymm15, ymm15);

				ymm3 = _mm256_add_ps(ymm3, ymm4);
				ymm5 = _mm256_mul_ps(ymm4, ymm4);

				//ymm5 = _mm256_mul_ps(ymm5, ymm13);
				//ymm3 = _mm256_add_ps(ymm3, ymm5);
				ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);
				ymm2 = _mm256_andnot_ps(ymm7, ymm15);

#ifdef USE_FAST_DIV
				ymm3 = _mm256_rcp_ps(ymm3);
#else
				ymm3 = _mm256_div_ps(ymm8, ymm3);
#endif	

				ymm3 = _mm256_sub_ps(ymm8, ymm3);
				ymm3 = _mm256_or_ps(ymm3, ymm2);

				ymm15 = _mm256_mul_ps(ymm3, ymm9);
				_mm256_store_ps(dst + REG_SIZE + i, ymm15);
			}
		}

#endif

		//Legacy
		void CNNPP::conv_4x4_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;

			__m256 ymm12 = _mm256_load_ps(kernel);
			__m256 ymm13 = _mm256_load_ps(kernel + REG_SIZE);
			__m256 ymm14 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			__m256 ymm15 = _mm256_load_ps(kernel + 3 * REG_SIZE);

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				__m256 ymm0 = _mm256_load_ps(pSrc0);
				__m256 ymm1 = _mm256_load_ps(pSrc1);
				__m256 ymm2 = _mm256_load_ps(pSrc2);
				__m256 ymm3 = _mm256_load_ps(pSrc3);

				size_t i = 0;
				for (; i < L - REG_SIZE; i += REG_SIZE)
				{
					__m256 ymm4 = _mm256_load_ps(pSrc0 + i + REG_SIZE);
					__m256 ymm5 = _mm256_load_ps(pSrc1 + i + REG_SIZE);
					__m256 ymm6 = _mm256_load_ps(pSrc2 + i + REG_SIZE);
					__m256 ymm7 = _mm256_load_ps(pSrc3 + i + REG_SIZE);

					__m256 ymm8 = _mm256_mul_ps(ymm0, ymm12);
					__m256 ymm9 = _mm256_mul_ps(ymm1, ymm13);
					__m256 ymm11 = _mm256_add_ps(ymm8, ymm9);

					ymm8 = _mm256_mul_ps(ymm2, ymm14);
					ymm11 = _mm256_add_ps(ymm11, ymm8);

					ymm8 = _mm256_mul_ps(ymm3, ymm15);
					ymm11 = _mm256_add_ps(ymm11, ymm8);

					ymm8 = _mm256_permute_ps(ymm11, 14);
					ymm11 = _mm256_add_ps(ymm11, ymm8);
					ymm8 = _mm256_permute_ps(ymm11, 1);
					ymm11 = _mm256_add_ps(ymm11, ymm8);

					//-----------------------------------------

					__m256 ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
					ymm0 = _mm256_blend_ps(ymm0, ymm10, 17);
					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm8 = _mm256_mul_ps(ymm0, ymm12);

					ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
					ymm1 = _mm256_blend_ps(ymm1, ymm10, 17);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm9 = _mm256_mul_ps(ymm1, ymm13);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
					ymm2 = _mm256_blend_ps(ymm2, ymm10, 17);
					ymm14 = _mm256_permute_ps(ymm14, 147);
					ymm9 = _mm256_mul_ps(ymm2, ymm14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
					ymm3 = _mm256_blend_ps(ymm3, ymm10, 17);
					ymm15 = _mm256_permute_ps(ymm15, 147);
					ymm9 = _mm256_mul_ps(ymm3, ymm15);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm9 = _mm256_permute_ps(ymm8, 14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm9 = _mm256_permute_ps(ymm8, 1);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 34);

					//-----------------------------------------

					ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
					ymm0 = _mm256_blend_ps(ymm0, ymm10, 34);
					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm8 = _mm256_mul_ps(ymm0, ymm12);

					ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
					ymm1 = _mm256_blend_ps(ymm1, ymm10, 34);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm9 = _mm256_mul_ps(ymm1, ymm13);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
					ymm2 = _mm256_blend_ps(ymm2, ymm10, 34);
					ymm14 = _mm256_permute_ps(ymm14, 147);
					ymm9 = _mm256_mul_ps(ymm2, ymm14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
					ymm3 = _mm256_blend_ps(ymm3, ymm10, 34);
					ymm15 = _mm256_permute_ps(ymm15, 147);
					ymm9 = _mm256_mul_ps(ymm3, ymm15);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm9 = _mm256_permute_ps(ymm8, 64);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm9 = _mm256_permute_ps(ymm8, 176);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 68);

					//-----------------------------------------

					ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
					ymm0 = _mm256_blend_ps(ymm0, ymm10, 68);
					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm8 = _mm256_mul_ps(ymm0, ymm12);

					ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
					ymm1 = _mm256_blend_ps(ymm1, ymm10, 68);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm9 = _mm256_mul_ps(ymm1, ymm13);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
					ymm2 = _mm256_blend_ps(ymm2, ymm10, 68);
					ymm14 = _mm256_permute_ps(ymm14, 147);
					ymm9 = _mm256_mul_ps(ymm2, ymm14);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
					ymm3 = _mm256_blend_ps(ymm3, ymm10, 68);
					ymm15 = _mm256_permute_ps(ymm15, 147);
					ymm9 = _mm256_mul_ps(ymm3, ymm15);
					ymm8 = _mm256_add_ps(ymm8, ymm9);

					ymm9 = _mm256_permute_ps(ymm8, 64);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm9 = _mm256_permute_ps(ymm8, 176);
					ymm8 = _mm256_add_ps(ymm8, ymm9);
					ymm11 = _mm256_blend_ps(ymm11, ymm8, 136);

					//_mm256_store_ps(pDst + i, ymm11);
					//-----------------------------------------
					ymm8 = _mm256_permute2f128_ps(ymm11, ymm11, 32);
					ymm9 = _mm256_permute2f128_ps(ymm11, ymm11, 19);

					ymm10 = _mm256_shuffle_ps(ymm8, ymm9, 221);
					ymm11 = _mm256_shuffle_ps(ymm8, ymm9, 136);
					ymm11 = _mm256_max_ps(ymm10, ymm11);

					_mm_store_ps(pDst + (i >> 1), _mm256_castps256_ps128(ymm11));
					//-----------------------------------------

					ymm0 = ymm4;
					ymm1 = ymm5;
					ymm2 = ymm6;
					ymm3 = ymm7;

					ymm12 = _mm256_permute_ps(ymm12, 147);
					ymm13 = _mm256_permute_ps(ymm13, 147);
					ymm14 = _mm256_permute_ps(ymm14, 147);
					ymm15 = _mm256_permute_ps(ymm15, 147);
				}

				//============================================================================

				__m256 ymm4 = _mm256_setzero_ps();
				__m256 ymm5 = _mm256_setzero_ps();
				__m256 ymm6 = _mm256_setzero_ps();
				__m256 ymm7 = _mm256_setzero_ps();

				if (i + REG_SIZE < src_size_l)
				{
					ymm4 = _mm256_load_ps(pSrc0 + i + REG_SIZE);
					ymm5 = _mm256_load_ps(pSrc1 + i + REG_SIZE);
					ymm6 = _mm256_load_ps(pSrc2 + i + REG_SIZE);
					ymm7 = _mm256_load_ps(pSrc3 + i + REG_SIZE);
				}

				__m256 ymm8 = _mm256_mul_ps(ymm0, ymm12);
				__m256 ymm9 = _mm256_mul_ps(ymm1, ymm13);
				__m256 ymm11 = _mm256_add_ps(ymm8, ymm9);

				ymm8 = _mm256_mul_ps(ymm2, ymm14);
				ymm11 = _mm256_add_ps(ymm11, ymm8);

				ymm8 = _mm256_mul_ps(ymm3, ymm15);
				ymm11 = _mm256_add_ps(ymm11, ymm8);

				ymm8 = _mm256_permute_ps(ymm11, 14);
				ymm11 = _mm256_add_ps(ymm11, ymm8);
				ymm8 = _mm256_permute_ps(ymm11, 1);
				ymm11 = _mm256_add_ps(ymm11, ymm8);

				//-----------------------------------------

				__m256 ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
				ymm0 = _mm256_blend_ps(ymm0, ymm10, 17);
				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm8 = _mm256_mul_ps(ymm0, ymm12);

				ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
				ymm1 = _mm256_blend_ps(ymm1, ymm10, 17);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm9 = _mm256_mul_ps(ymm1, ymm13);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
				ymm2 = _mm256_blend_ps(ymm2, ymm10, 17);
				ymm14 = _mm256_permute_ps(ymm14, 147);
				ymm9 = _mm256_mul_ps(ymm2, ymm14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
				ymm3 = _mm256_blend_ps(ymm3, ymm10, 17);
				ymm15 = _mm256_permute_ps(ymm15, 147);
				ymm9 = _mm256_mul_ps(ymm3, ymm15);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm9 = _mm256_permute_ps(ymm8, 14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm9 = _mm256_permute_ps(ymm8, 1);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 34);

				//-----------------------------------------

				ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
				ymm0 = _mm256_blend_ps(ymm0, ymm10, 34);
				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm8 = _mm256_mul_ps(ymm0, ymm12);

				ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
				ymm1 = _mm256_blend_ps(ymm1, ymm10, 34);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm9 = _mm256_mul_ps(ymm1, ymm13);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
				ymm2 = _mm256_blend_ps(ymm2, ymm10, 34);
				ymm14 = _mm256_permute_ps(ymm14, 147);
				ymm9 = _mm256_mul_ps(ymm2, ymm14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
				ymm3 = _mm256_blend_ps(ymm3, ymm10, 34);
				ymm15 = _mm256_permute_ps(ymm15, 147);
				ymm9 = _mm256_mul_ps(ymm3, ymm15);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm9 = _mm256_permute_ps(ymm8, 64);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm9 = _mm256_permute_ps(ymm8, 176);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 68);

				//-----------------------------------------

				ymm10 = _mm256_permute2f128_ps(ymm0, ymm4, 33);
				ymm0 = _mm256_blend_ps(ymm0, ymm10, 68);
				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm8 = _mm256_mul_ps(ymm0, ymm12);

				ymm10 = _mm256_permute2f128_ps(ymm1, ymm5, 33);
				ymm1 = _mm256_blend_ps(ymm1, ymm10, 68);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm9 = _mm256_mul_ps(ymm1, ymm13);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 33);
				ymm2 = _mm256_blend_ps(ymm2, ymm10, 68);
				ymm14 = _mm256_permute_ps(ymm14, 147);
				ymm9 = _mm256_mul_ps(ymm2, ymm14);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm10 = _mm256_permute2f128_ps(ymm3, ymm7, 33);
				ymm3 = _mm256_blend_ps(ymm3, ymm10, 68);
				ymm15 = _mm256_permute_ps(ymm15, 147);
				ymm9 = _mm256_mul_ps(ymm3, ymm15);
				ymm8 = _mm256_add_ps(ymm8, ymm9);

				ymm9 = _mm256_permute_ps(ymm8, 64);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm9 = _mm256_permute_ps(ymm8, 176);
				ymm8 = _mm256_add_ps(ymm8, ymm9);
				ymm11 = _mm256_blend_ps(ymm11, ymm8, 136);

				//_mm256_store_ps(pDst + i, ymm11);
				//-----------------------------------------
				ymm8 = _mm256_permute2f128_ps(ymm11, ymm11, 32);
				ymm9 = _mm256_permute2f128_ps(ymm11, ymm11, 19);

				ymm10 = _mm256_shuffle_ps(ymm8, ymm9, 221);
				ymm11 = _mm256_shuffle_ps(ymm8, ymm9, 136);
				ymm11 = _mm256_max_ps(ymm10, ymm11);

				_mm_store_ps(pDst + (i >> 1), _mm256_castps256_ps128(ymm11));
				//-----------------------------------------

				ymm12 = _mm256_permute_ps(ymm12, 147);
				ymm13 = _mm256_permute_ps(ymm13, 147);
				ymm14 = _mm256_permute_ps(ymm14, 147);
				ymm15 = _mm256_permute_ps(ymm15, 147);

				//============================================================================
			}
		}
		void CNNPP::conv_4x4_block(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			const size_t BLOCK_SIZE_X = 128 / 2;
			const size_t BLOCK_SIZE_Y = 80 / 2;

			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;

			int n = src_size_l / BLOCK_SIZE_X;
			int m = src_size_h / BLOCK_SIZE_Y;

			__m256 ymm12 = _mm256_load_ps(kernel);
			__m256 ymm13 = _mm256_load_ps(kernel + REG_SIZE);
			__m256 ymm14 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			__m256 ymm15 = _mm256_load_ps(kernel + 3 * REG_SIZE);

			for (size_t y = 0; y < m; ++y)
			{
				for (size_t x = 0; x < n; ++x)
				{
					const size_t offset_src = y * BLOCK_SIZE_Y * src_size_l + x * BLOCK_SIZE_X;
					const size_t offset_dst = y * BLOCK_SIZE_Y * dst_size_l + x * BLOCK_SIZE_X;
					//#pragma loop(hint_parallel(8))
					for (size_t j = 0; j < BLOCK_SIZE_Y; ++j)
					{
						if (y + 1 == m && j + 3 == BLOCK_SIZE_Y) break;

						float* __restrict pSrc0 = src + offset_src + j * src_size_l;
						float* __restrict pSrc1 = src + offset_src + (j + 1) * src_size_l;
						float* __restrict pSrc2 = src + offset_src + (j + 2) * src_size_l;
						float* __restrict pSrc3 = src + offset_src + (j + 3) * src_size_l;
						float* __restrict pDst = dst + offset_dst + j * dst_size_l;

						__m256 ymm0 = _mm256_load_ps(pSrc0);
						__m256 ymm1 = _mm256_load_ps(pSrc1);
						__m256 ymm2 = _mm256_load_ps(pSrc2);
						__m256 ymm3 = _mm256_load_ps(pSrc3);

						//#pragma loop(hint_parallel(8))
						for (size_t i = 0; i < BLOCK_SIZE_X; i++)
						{
							if (x + 1 == n && i + 3 == BLOCK_SIZE_X) break;

							//__m256 ymm4 = _mm256_load_ps(pSrc0 + i);
							//__m256 ymm5 = _mm256_load_ps(pSrc1 + i);
							//__m256 ymm6 = _mm256_load_ps(pSrc2 + i);
							//__m256 ymm7 = _mm256_load_ps(pSrc3 + i);

							__m256 ymm8 = _mm256_mul_ps(ymm0, ymm12);
							__m256 ymm9 = _mm256_mul_ps(ymm1, ymm13);
							__m256 ymm11 = _mm256_add_ps(ymm8, ymm9);

							ymm8 = _mm256_mul_ps(ymm2, ymm14);
							ymm11 = _mm256_add_ps(ymm11, ymm8);

							ymm8 = _mm256_mul_ps(ymm3, ymm15);
							ymm11 = _mm256_add_ps(ymm11, ymm8);

							ymm8 = _mm256_permute_ps(ymm11, 14);
							ymm11 = _mm256_add_ps(ymm11, ymm8);
							ymm8 = _mm256_permute_ps(ymm11, 1);
							ymm11 = _mm256_add_ps(ymm11, ymm8);

							//_mm256_store_ps(pDst + i, ymm11);
							_mm_store_ss(pDst + i, _mm256_castps256_ps128(ymm11));
						}
					}
				}
			}
		}
		void CNNPP::tanh_max_tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale)
		{
			ALIGN(ALIGN_DEF) float buff[16];

			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm7 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm1 = _mm256_broadcast_ss(conv_b);
			const __m256 ymm5 = _mm256_broadcast_ss(subs_w);
			__m256 ymm0 = _mm256_broadcast_ss(subs_b);
			__m256 ymm2 = _mm256_broadcast_ss(scale);
			_mm256_store_ps(buff, ymm0);
			_mm256_store_ps(buff + REG_SIZE, ymm2);

			size_t j2 = 0;
			for (size_t j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + j2 * dst_size_l;

				for (size_t i = 0; i < src_size_l; i += 2 * REG_SIZE)
				{
					ymm0 = _mm256_load_ps(pSrc0 + i);

					__m256 ymm6 = _mm256_load_ps(pSrc0 + i + REG_SIZE);
					ymm0 = _mm256_add_ps(ymm0, ymm1);
					__m256 ymm4 = _mm256_mul_ps(ymm0, ymm0);
					__m256 ymm3 = _mm256_and_ps(ymm14, ymm0);

					ymm6 = _mm256_add_ps(ymm6, ymm1);
					__m256 ymm12 = _mm256_mul_ps(ymm4, ymm4);

					ymm12 = _mm256_mul_ps(ymm12, ymm7);
					ymm3 = _mm256_add_ps(ymm15, ymm3);
					ymm2 = _mm256_andnot_ps(ymm14, ymm0);

					__m256 ymm10 = _mm256_mul_ps(ymm6, ymm6);
					ymm3 = _mm256_add_ps(ymm3, ymm4);
					__m256 ymm9 = _mm256_and_ps(ymm14, ymm6);
					ymm0 = _mm256_load_ps(pSrc1 + i);

					ymm0 = _mm256_add_ps(ymm0, ymm1);
					__m256 ymm11 = _mm256_mul_ps(ymm10, ymm10);

					ymm11 = _mm256_mul_ps(ymm11, ymm7);
					ymm3 = _mm256_add_ps(ymm3, ymm12);
					__m256 ymm8 = _mm256_andnot_ps(ymm14, ymm6);

#ifdef USE_FAST_DIV
					ymm3 = _mm256_rcp_ps(ymm3);
#else
					ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

					ymm9 = _mm256_add_ps(ymm15, ymm9);
					ymm6 = _mm256_load_ps(pSrc1 + i + REG_SIZE);

					ymm6 = _mm256_add_ps(ymm6, ymm1);

					ymm4 = _mm256_mul_ps(ymm0, ymm0);
					ymm3 = _mm256_sub_ps(ymm15, ymm3);

					ymm9 = _mm256_add_ps(ymm9, ymm10);
					ymm12 = _mm256_or_ps(ymm3, ymm2);

					ymm10 = _mm256_mul_ps(ymm6, ymm6);
					ymm9 = _mm256_add_ps(ymm9, ymm11);
					ymm3 = _mm256_and_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm9 = _mm256_rcp_ps(ymm9);
#else
					ymm9 = _mm256_div_ps(ymm15, ymm9);
#endif	

					ymm9 = _mm256_sub_ps(ymm15, ymm9);
					ymm2 = _mm256_andnot_ps(ymm14, ymm0);

					ymm0 = _mm256_mul_ps(ymm4, ymm4);

					ymm0 = _mm256_mul_ps(ymm0, ymm7);
					ymm3 = _mm256_add_ps(ymm15, ymm3);
					__m256 ymm13 = _mm256_or_ps(ymm9, ymm8);

					ymm11 = _mm256_mul_ps(ymm10, ymm10);

					ymm11 = _mm256_mul_ps(ymm11, ymm7);
					ymm3 = _mm256_add_ps(ymm3, ymm4);
					ymm8 = _mm256_andnot_ps(ymm14, ymm6);

					ymm4 = _mm256_load_ps(buff);
					ymm9 = _mm256_and_ps(ymm14, ymm6);
					ymm3 = _mm256_add_ps(ymm3, ymm0);

#ifdef USE_FAST_DIV
					ymm3 = _mm256_rcp_ps(ymm3);
#else
					ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

					ymm9 = _mm256_add_ps(ymm15, ymm9);

					ymm3 = _mm256_sub_ps(ymm15, ymm3);

					ymm9 = _mm256_add_ps(ymm9, ymm10);
					ymm3 = _mm256_or_ps(ymm3, ymm2);

					ymm9 = _mm256_add_ps(ymm9, ymm11);

#ifdef USE_FAST_DIV
					ymm9 = _mm256_rcp_ps(ymm9);
#else
					ymm9 = _mm256_div_ps(ymm15, ymm9);
#endif	

					ymm9 = _mm256_sub_ps(ymm15, ymm9);
					ymm9 = _mm256_or_ps(ymm9, ymm8);

					//-----------------------------

					ymm3 = _mm256_max_ps(ymm3, ymm12);
					ymm9 = _mm256_max_ps(ymm9, ymm13);

					ymm13 = _mm256_load_ps(buff + REG_SIZE);
					ymm6 = _mm256_permute2f128_ps(ymm3, ymm9, 32);
					ymm8 = _mm256_permute2f128_ps(ymm9, ymm3, 19);

					ymm2 = _mm256_shuffle_ps(ymm6, ymm8, 221);
					ymm3 = _mm256_shuffle_ps(ymm6, ymm8, 136);
					ymm0 = _mm256_max_ps(ymm2, ymm3);

					//-----------------------------

					ymm0 = _mm256_mul_ps(ymm0, ymm5);
					ymm0 = _mm256_add_ps(ymm0, ymm4);

					//-----------------------------

					ymm3 = _mm256_and_ps(ymm14, ymm0);

					ymm3 = _mm256_add_ps(ymm15, ymm3);
					ymm4 = _mm256_mul_ps(ymm0, ymm0);

					ymm3 = _mm256_add_ps(ymm3, ymm4);
					ymm8 = _mm256_mul_ps(ymm4, ymm4);

					ymm8 = _mm256_mul_ps(ymm8, ymm7);
					ymm3 = _mm256_add_ps(ymm3, ymm8);
					ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm3 = _mm256_rcp_ps(ymm3);
#else
					ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

					ymm3 = _mm256_sub_ps(ymm15, ymm3);
					ymm3 = _mm256_or_ps(ymm3, ymm2);

					//-----------------------------

					ymm3 = _mm256_mul_ps(ymm3, ymm13);

					_mm256_store_ps(pDst, ymm3);
					pDst += REG_SIZE;
				}
				j2++;
			}
		}
		void CNNPP::max1_tanh_tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale)
		{
			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm7 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm1 = _mm256_broadcast_ss(conv_b);
			const __m256 ymm5 = _mm256_broadcast_ss(subs_w);
			const __m256 ymm4 = _mm256_broadcast_ss(subs_b);
			const __m256 ymm11 = _mm256_broadcast_ss(scale);

			src_size_l = src_size_l >> 1;
			size_t j2 = 0;
			for (size_t j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + j2 * dst_size_l;

				for (size_t i = 0; i < src_size_l; i += REG_SIZE)
				{
					__m256 ymm3 = _mm256_load_ps(pSrc0 + i);
					__m256 ymm12 = _mm256_load_ps(pSrc1 + i);
					__m256 ymm0 = _mm256_max_ps(ymm3, ymm12);

					//-----------------------------

					ymm0 = _mm256_add_ps(ymm0, ymm1);
					ymm3 = _mm256_and_ps(ymm14, ymm0);

					ymm3 = _mm256_add_ps(ymm15, ymm3);
					__m256 ymm6 = _mm256_mul_ps(ymm0, ymm0);

					ymm3 = _mm256_add_ps(ymm3, ymm6);
					__m256 ymm8 = _mm256_mul_ps(ymm6, ymm6);

					ymm8 = _mm256_mul_ps(ymm8, ymm7);
					ymm3 = _mm256_add_ps(ymm3, ymm8);
					__m256 ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm3 = _mm256_rcp_ps(ymm3);
#else
					ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

					ymm3 = _mm256_sub_ps(ymm15, ymm3);
					ymm0 = _mm256_or_ps(ymm3, ymm2);

					//-----------------------------

					ymm0 = _mm256_mul_ps(ymm0, ymm5);
					ymm0 = _mm256_add_ps(ymm0, ymm4);

					//-----------------------------

					ymm3 = _mm256_and_ps(ymm14, ymm0);

					ymm3 = _mm256_add_ps(ymm15, ymm3);
					ymm6 = _mm256_mul_ps(ymm0, ymm0);

					ymm3 = _mm256_add_ps(ymm3, ymm6);
					ymm8 = _mm256_mul_ps(ymm6, ymm6);

					ymm8 = _mm256_mul_ps(ymm8, ymm7);
					ymm3 = _mm256_add_ps(ymm3, ymm8);
					ymm2 = _mm256_andnot_ps(ymm14, ymm0);

#ifdef USE_FAST_DIV
					ymm3 = _mm256_rcp_ps(ymm3);
#else
					ymm3 = _mm256_div_ps(ymm15, ymm3);
#endif	

					ymm3 = _mm256_sub_ps(ymm15, ymm3);
					ymm3 = _mm256_or_ps(ymm3, ymm2);

					//-----------------------------

					ymm3 = _mm256_mul_ps(ymm3, ymm11);

					_mm256_store_ps(pDst, ymm3);
					pDst += REG_SIZE;
				}
				j2++;
			}
		}
	}

#endif
}