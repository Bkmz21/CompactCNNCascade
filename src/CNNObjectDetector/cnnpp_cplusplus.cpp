/*
*	Copyright (c) 2018, Ilya Kalinovskiy
*	All rights reserved.
*
*	This is an implementation of the algorithm described in the following paper:
*		I.A. Kalinovskiy, V.G. Spitsyn,
*		Compact Convolutional Neural Network Cascade for Face Detection,
*		http://arxiv.org/fabsf/1508.01292.
*
*	Redistribution and use of this program as source code or in binary form, with or without modifications, are permitted provided that the following conditions are met:
*		1. Redistributions may not be sold, nor may they be used in a commercial product or activity without prior permission from the copyright holder (contact him at kua_21@mail.ru).
*		2. Redistributions may not be used for military purposes.
*		3. Any published work which utilizes this program shall include the reference to the paper available at http://arxiv.org/fabsf/1508.01292
*		4. Redistributions must retain the above copyright notice and the reference to the algorithm on which the implementation is based on, this list of conditions and the following disclaimer.
*
*	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
*	IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


#include "cnnpp_cplusplus.h"
#include <cmath>
#include <stdio.h>
#include <algorithm>


//========================================================================================================


namespace NeuralNetworksLib
{
#if !defined(USE_SSE) && !defined(USE_AVX) && !defined(USE_ASM)

	namespace SIMD
	{
		void CNNPP::conv_3x3(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 2;
			if (H == 0) H = src_size_h - 2;

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				int i = 0;
				for (; i <= L - 8; i += 8)
				{
					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0++ + 2) * *(kernel + 2)
						+ *pSrc1 * *(kernel + 3) + *(pSrc1 + 1) * *(kernel + 4) + *(pSrc1++ + 2) * *(kernel + 5)
						+ *pSrc2 * *(kernel + 6) + *(pSrc2 + 1) * *(kernel + 7) + *(pSrc2++ + 2) * *(kernel + 8);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0++ + 2) * *(kernel + 2)
						+ *pSrc1 * *(kernel + 3) + *(pSrc1 + 1) * *(kernel + 4) + *(pSrc1++ + 2) * *(kernel + 5)
						+ *pSrc2 * *(kernel + 6) + *(pSrc2 + 1) * *(kernel + 7) + *(pSrc2++ + 2) * *(kernel + 8);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0++ + 2) * *(kernel + 2)
						+ *pSrc1 * *(kernel + 3) + *(pSrc1 + 1) * *(kernel + 4) + *(pSrc1++ + 2) * *(kernel + 5)
						+ *pSrc2 * *(kernel + 6) + *(pSrc2 + 1) * *(kernel + 7) + *(pSrc2++ + 2) * *(kernel + 8);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0++ + 2) * *(kernel + 2)
						+ *pSrc1 * *(kernel + 3) + *(pSrc1 + 1) * *(kernel + 4) + *(pSrc1++ + 2) * *(kernel + 5)
						+ *pSrc2 * *(kernel + 6) + *(pSrc2 + 1) * *(kernel + 7) + *(pSrc2++ + 2) * *(kernel + 8);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0++ + 2) * *(kernel + 2)
						+ *pSrc1 * *(kernel + 3) + *(pSrc1 + 1) * *(kernel + 4) + *(pSrc1++ + 2) * *(kernel + 5)
						+ *pSrc2 * *(kernel + 6) + *(pSrc2 + 1) * *(kernel + 7) + *(pSrc2++ + 2) * *(kernel + 8);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0++ + 2) * *(kernel + 2)
						+ *pSrc1 * *(kernel + 3) + *(pSrc1 + 1) * *(kernel + 4) + *(pSrc1++ + 2) * *(kernel + 5)
						+ *pSrc2 * *(kernel + 6) + *(pSrc2 + 1) * *(kernel + 7) + *(pSrc2++ + 2) * *(kernel + 8);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0++ + 2) * *(kernel + 2)
						+ *pSrc1 * *(kernel + 3) + *(pSrc1 + 1) * *(kernel + 4) + *(pSrc1++ + 2) * *(kernel + 5)
						+ *pSrc2 * *(kernel + 6) + *(pSrc2 + 1) * *(kernel + 7) + *(pSrc2++ + 2) * *(kernel + 8);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0++ + 2) * *(kernel + 2)
						+ *pSrc1 * *(kernel + 3) + *(pSrc1 + 1) * *(kernel + 4) + *(pSrc1++ + 2) * *(kernel + 5)
						+ *pSrc2 * *(kernel + 6) + *(pSrc2 + 1) * *(kernel + 7) + *(pSrc2++ + 2) * *(kernel + 8);
				}

				for (; i < L; ++i)
				{
					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0++ + 2) * *(kernel + 2)
						+ *pSrc1 * *(kernel + 3) + *(pSrc1 + 1) * *(kernel + 4) + *(pSrc1++ + 2) * *(kernel + 5)
						+ *pSrc2 * *(kernel + 6) + *(pSrc2 + 1) * *(kernel + 7) + *(pSrc2++ + 2) * *(kernel + 8);
				}
			}
		}
		void CNNPP::conv_4x4(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				int i = 0;
				for (; i <= L - 8; i += 8)
				{
					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0++ + 3) * *(kernel + 3)
						+ *pSrc1 * *(kernel + 4) + *(pSrc1 + 1) * *(kernel + 5) + *(pSrc1 + 2) * *(kernel + 6) + *(pSrc1++ + 3) * *(kernel + 7)
						+ *pSrc2 * *(kernel + 8) + *(pSrc2 + 1) * *(kernel + 9) + *(pSrc2 + 2) * *(kernel + 10) + *(pSrc2++ + 3) * *(kernel + 11)
						+ *pSrc3 * *(kernel + 12) + *(pSrc3 + 1) * *(kernel + 13) + *(pSrc3 + 2) * *(kernel + 14) + *(pSrc3++ + 3) * *(kernel + 15);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0++ + 3) * *(kernel + 3)
						+ *pSrc1 * *(kernel + 4) + *(pSrc1 + 1) * *(kernel + 5) + *(pSrc1 + 2) * *(kernel + 6) + *(pSrc1++ + 3) * *(kernel + 7)
						+ *pSrc2 * *(kernel + 8) + *(pSrc2 + 1) * *(kernel + 9) + *(pSrc2 + 2) * *(kernel + 10) + *(pSrc2++ + 3) * *(kernel + 11)
						+ *pSrc3 * *(kernel + 12) + *(pSrc3 + 1) * *(kernel + 13) + *(pSrc3 + 2) * *(kernel + 14) + *(pSrc3++ + 3) * *(kernel + 15);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0++ + 3) * *(kernel + 3)
						+ *pSrc1 * *(kernel + 4) + *(pSrc1 + 1) * *(kernel + 5) + *(pSrc1 + 2) * *(kernel + 6) + *(pSrc1++ + 3) * *(kernel + 7)
						+ *pSrc2 * *(kernel + 8) + *(pSrc2 + 1) * *(kernel + 9) + *(pSrc2 + 2) * *(kernel + 10) + *(pSrc2++ + 3) * *(kernel + 11)
						+ *pSrc3 * *(kernel + 12) + *(pSrc3 + 1) * *(kernel + 13) + *(pSrc3 + 2) * *(kernel + 14) + *(pSrc3++ + 3) * *(kernel + 15);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0++ + 3) * *(kernel + 3)
						+ *pSrc1 * *(kernel + 4) + *(pSrc1 + 1) * *(kernel + 5) + *(pSrc1 + 2) * *(kernel + 6) + *(pSrc1++ + 3) * *(kernel + 7)
						+ *pSrc2 * *(kernel + 8) + *(pSrc2 + 1) * *(kernel + 9) + *(pSrc2 + 2) * *(kernel + 10) + *(pSrc2++ + 3) * *(kernel + 11)
						+ *pSrc3 * *(kernel + 12) + *(pSrc3 + 1) * *(kernel + 13) + *(pSrc3 + 2) * *(kernel + 14) + *(pSrc3++ + 3) * *(kernel + 15);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0++ + 3) * *(kernel + 3)
						+ *pSrc1 * *(kernel + 4) + *(pSrc1 + 1) * *(kernel + 5) + *(pSrc1 + 2) * *(kernel + 6) + *(pSrc1++ + 3) * *(kernel + 7)
						+ *pSrc2 * *(kernel + 8) + *(pSrc2 + 1) * *(kernel + 9) + *(pSrc2 + 2) * *(kernel + 10) + *(pSrc2++ + 3) * *(kernel + 11)
						+ *pSrc3 * *(kernel + 12) + *(pSrc3 + 1) * *(kernel + 13) + *(pSrc3 + 2) * *(kernel + 14) + *(pSrc3++ + 3) * *(kernel + 15);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0++ + 3) * *(kernel + 3)
						+ *pSrc1 * *(kernel + 4) + *(pSrc1 + 1) * *(kernel + 5) + *(pSrc1 + 2) * *(kernel + 6) + *(pSrc1++ + 3) * *(kernel + 7)
						+ *pSrc2 * *(kernel + 8) + *(pSrc2 + 1) * *(kernel + 9) + *(pSrc2 + 2) * *(kernel + 10) + *(pSrc2++ + 3) * *(kernel + 11)
						+ *pSrc3 * *(kernel + 12) + *(pSrc3 + 1) * *(kernel + 13) + *(pSrc3 + 2) * *(kernel + 14) + *(pSrc3++ + 3) * *(kernel + 15);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0++ + 3) * *(kernel + 3)
						+ *pSrc1 * *(kernel + 4) + *(pSrc1 + 1) * *(kernel + 5) + *(pSrc1 + 2) * *(kernel + 6) + *(pSrc1++ + 3) * *(kernel + 7)
						+ *pSrc2 * *(kernel + 8) + *(pSrc2 + 1) * *(kernel + 9) + *(pSrc2 + 2) * *(kernel + 10) + *(pSrc2++ + 3) * *(kernel + 11)
						+ *pSrc3 * *(kernel + 12) + *(pSrc3 + 1) * *(kernel + 13) + *(pSrc3 + 2) * *(kernel + 14) + *(pSrc3++ + 3) * *(kernel + 15);

					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0++ + 3) * *(kernel + 3)
						+ *pSrc1 * *(kernel + 4) + *(pSrc1 + 1) * *(kernel + 5) + *(pSrc1 + 2) * *(kernel + 6) + *(pSrc1++ + 3) * *(kernel + 7)
						+ *pSrc2 * *(kernel + 8) + *(pSrc2 + 1) * *(kernel + 9) + *(pSrc2 + 2) * *(kernel + 10) + *(pSrc2++ + 3) * *(kernel + 11)
						+ *pSrc3 * *(kernel + 12) + *(pSrc3 + 1) * *(kernel + 13) + *(pSrc3 + 2) * *(kernel + 14) + *(pSrc3++ + 3) * *(kernel + 15);
				}

				for (; i < L; ++i)
				{
					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0++ + 3) * *(kernel + 3)
						+ *pSrc1 * *(kernel + 4) + *(pSrc1 + 1) * *(kernel + 5) + *(pSrc1 + 2) * *(kernel + 6) + *(pSrc1++ + 3) * *(kernel + 7)
						+ *pSrc2 * *(kernel + 8) + *(pSrc2 + 1) * *(kernel + 9) + *(pSrc2 + 2) * *(kernel + 10) + *(pSrc2++ + 3) * *(kernel + 11)
						+ *pSrc3 * *(kernel + 12) + *(pSrc3 + 1) * *(kernel + 13) + *(pSrc3 + 2) * *(kernel + 14) + *(pSrc3++ + 3) * *(kernel + 15);
				}
			}
		}
		void CNNPP::conv_5x4(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 4;

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
					*(pDst) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0++ + 3) * *(kernel + 3);
					//if (i + j == 0) printf("%f\n", *(pDst));
					*(pDst) += *pSrc1 * *(kernel + 4) + *(pSrc1 + 1) * *(kernel + 5) + *(pSrc1 + 2) * *(kernel + 6) + *(pSrc1++ + 3) * *(kernel + 7);
					//if (i + j == 0) printf("%f\n", *(pDst));
					*(pDst) += *pSrc2 * *(kernel + 8) + *(pSrc2 + 1) * *(kernel + 9) + *(pSrc2 + 2) * *(kernel + 10) + *(pSrc2++ + 3) * *(kernel + 11);
					//if (i + j == 0) printf("%f\n", *(pDst));
					*(pDst) += *pSrc3 * *(kernel + 12) + *(pSrc3 + 1) * *(kernel + 13) + *(pSrc3 + 2) * *(kernel + 14) + *(pSrc3++ + 3) * *(kernel + 15);
					//if (i + j == 0) printf("%f\n", *(pDst));
					*(pDst) += *pSrc4 * *(kernel + 16) + *(pSrc4 + 1) * *(kernel + 17) + *(pSrc4 + 2) * *(kernel + 18) + *(pSrc4++ + 3) * *(kernel + 19);
					//if (i + j == 0) printf("%f\n", *(pDst));
					pDst++;
				}
			}
		}
		void CNNPP::conv_5x5(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 4;
			if (H == 0) H = src_size_h - 4;

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
					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0 + 3) * *(kernel + 3) + *(pSrc0++ + 4) * *(kernel + 4)
						+ *pSrc1 * *(kernel + 5) + *(pSrc1 + 1) * *(kernel + 6) + *(pSrc1 + 2) * *(kernel + 7) + *(pSrc1 + 3) * *(kernel + 8) + *(pSrc1++ + 4) * *(kernel + 9)
						+ *pSrc2 * *(kernel + 10) + *(pSrc2 + 1) * *(kernel + 11) + *(pSrc2 + 2) * *(kernel + 12) + *(pSrc2 + 3) * *(kernel + 13) + *(pSrc2++ + 4) * *(kernel + 14)
						+ *pSrc3 * *(kernel + 15) + *(pSrc3 + 1) * *(kernel + 16) + *(pSrc3 + 2) * *(kernel + 17) + *(pSrc3 + 3) * *(kernel + 18) + *(pSrc3++ + 4) * *(kernel + 19)
						+ *pSrc4 * *(kernel + 20) + *(pSrc4 + 1) * *(kernel + 21) + *(pSrc4 + 2) * *(kernel + 22) + *(pSrc4 + 3) * *(kernel + 23) + *(pSrc4++ + 4) * *(kernel + 24);
				}
			}
		}
		void CNNPP::conv_6x5(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 4;
			if (H == 0) H = src_size_h - 5;

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
					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0 + 3) * *(kernel + 3) + *(pSrc0++ + 4) * *(kernel + 4)
						+ *pSrc1 * *(kernel + 5) + *(pSrc1 + 1) * *(kernel + 6) + *(pSrc1 + 2) * *(kernel + 7) + *(pSrc1 + 3) * *(kernel + 8) + *(pSrc1++ + 4) * *(kernel + 9)
						+ *pSrc2 * *(kernel + 10) + *(pSrc2 + 1) * *(kernel + 11) + *(pSrc2 + 2) * *(kernel + 12) + *(pSrc2 + 3) * *(kernel + 13) + *(pSrc2++ + 4) * *(kernel + 14)
						+ *pSrc3 * *(kernel + 15) + *(pSrc3 + 1) * *(kernel + 16) + *(pSrc3 + 2) * *(kernel + 17) + *(pSrc3 + 3) * *(kernel + 18) + *(pSrc3++ + 4) * *(kernel + 19)
						+ *pSrc4 * *(kernel + 20) + *(pSrc4 + 1) * *(kernel + 21) + *(pSrc4 + 2) * *(kernel + 22) + *(pSrc4 + 3) * *(kernel + 23) + *(pSrc4++ + 4) * *(kernel + 24)
						+ *pSrc5 * *(kernel + 25) + *(pSrc5 + 1) * *(kernel + 26) + *(pSrc5 + 2) * *(kernel + 27) + *(pSrc5 + 3) * *(kernel + 28) + *(pSrc5++ + 4) * *(kernel + 29);
				}
			}
		}
		void CNNPP::conv_6x6(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 5;
			if (H == 0) H = src_size_h - 5;

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
					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0 + 3) * *(kernel + 3) + *(pSrc0 + 4) * *(kernel + 4) + *(pSrc0++ + 5) * *(kernel + 5)
						+ *pSrc1 * *(kernel + 6) + *(pSrc1 + 1) * *(kernel + 7) + *(pSrc1 + 2) * *(kernel + 8) + *(pSrc1 + 3) * *(kernel + 9) + *(pSrc1 + 4) * *(kernel + 10) + *(pSrc0++ + 5) * *(kernel + 11)
						+ *pSrc2 * *(kernel + 12) + *(pSrc2 + 1) * *(kernel + 13) + *(pSrc2 + 2) * *(kernel + 14) + *(pSrc2 + 3) * *(kernel + 15) + *(pSrc2 + 4) * *(kernel + 16) + *(pSrc0++ + 5) * *(kernel + 17)
						+ *pSrc3 * *(kernel + 18) + *(pSrc3 + 1) * *(kernel + 19) + *(pSrc3 + 2) * *(kernel + 20) + *(pSrc3 + 3) * *(kernel + 21) + *(pSrc3 + 4) * *(kernel + 22) + *(pSrc0++ + 5) * *(kernel + 23)
						+ *pSrc4 * *(kernel + 24) + *(pSrc4 + 1) * *(kernel + 25) + *(pSrc4 + 2) * *(kernel + 26) + *(pSrc4 + 3) * *(kernel + 27) + *(pSrc4 + 4) * *(kernel + 28) + *(pSrc0++ + 5) * *(kernel + 29)
						+ *pSrc5 * *(kernel + 30) + *(pSrc5 + 1) * *(kernel + 31) + *(pSrc5 + 2) * *(kernel + 32) + *(pSrc5 + 3) * *(kernel + 33) + *(pSrc5 + 4) * *(kernel + 34) + *(pSrc0++ + 5) * *(kernel + 35);
				}
			}
		}
		void CNNPP::conv_7x7(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 6;
			if (H == 0) H = src_size_h - 6;

			for (size_t j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pSrc4 = src + (j + 4) * src_size_l;
				float* __restrict pSrc5 = src + (j + 5) * src_size_l;
				float* __restrict pSrc6 = src + (j + 6) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				for (size_t i = 0; i < L; ++i)
				{
					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0 + 3) * *(kernel + 3) + *(pSrc0 + 4) * *(kernel + 4) + *(pSrc0 + 5) * *(kernel + 5) + *(pSrc0++ + 6) * *(kernel + 6)
						+ *pSrc1 * *(kernel + 7) + *(pSrc1 + 1) * *(kernel + 8) + *(pSrc1 + 2) * *(kernel + 9) + *(pSrc1 + 3) * *(kernel + 10) + *(pSrc1 + 4) * *(kernel + 11) + *(pSrc1 + 5) * *(kernel + 12) + *(pSrc1++ + 6) * *(kernel + 13)
						+ *pSrc2 * *(kernel + 14) + *(pSrc2 + 1) * *(kernel + 15) + *(pSrc2 + 2) * *(kernel + 16) + *(pSrc2 + 3) * *(kernel + 17) + *(pSrc2 + 4) * *(kernel + 18) + *(pSrc2 + 5) * *(kernel + 19) + *(pSrc2++ + 6) * *(kernel + 20)
						+ *pSrc3 * *(kernel + 21) + *(pSrc3 + 1) * *(kernel + 22) + *(pSrc3 + 2) * *(kernel + 23) + *(pSrc3 + 3) * *(kernel + 24) + *(pSrc3 + 4) * *(kernel + 25) + *(pSrc3 + 5) * *(kernel + 26) + *(pSrc3++ + 6) * *(kernel + 27)
						+ *pSrc4 * *(kernel + 28) + *(pSrc4 + 1) * *(kernel + 29) + *(pSrc4 + 2) * *(kernel + 30) + *(pSrc4 + 3) * *(kernel + 31) + *(pSrc4 + 4) * *(kernel + 32) + *(pSrc4 + 5) * *(kernel + 33) + *(pSrc4++ + 6) * *(kernel + 34)
						+ *pSrc5 * *(kernel + 35) + *(pSrc5 + 1) * *(kernel + 36) + *(pSrc5 + 2) * *(kernel + 37) + *(pSrc5 + 3) * *(kernel + 38) + *(pSrc5 + 4) * *(kernel + 39) + *(pSrc5 + 5) * *(kernel + 40) + *(pSrc5++ + 6) * *(kernel + 41)
						+ *pSrc6 * *(kernel + 42) + *(pSrc6 + 1) * *(kernel + 43) + *(pSrc6 + 2) * *(kernel + 44) + *(pSrc6 + 3) * *(kernel + 45) + *(pSrc6 + 4) * *(kernel + 46) + *(pSrc6 + 5) * *(kernel + 47) + *(pSrc6++ + 6) * *(kernel + 48);
				}
			}
		}
		void CNNPP::conv_8x7(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 6;
			if (H == 0) H = src_size_h - 7;

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
					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0 + 3) * *(kernel + 3) + *(pSrc0 + 4) * *(kernel + 4) + *(pSrc0 + 5) * *(kernel + 5) + *(pSrc0++ + 6) * *(kernel + 6)
						+ *pSrc1 * *(kernel + 7) + *(pSrc1 + 1) * *(kernel + 8) + *(pSrc1 + 2) * *(kernel + 9) + *(pSrc1 + 3) * *(kernel + 10) + *(pSrc1 + 4) * *(kernel + 11) + *(pSrc1 + 5) * *(kernel + 12) + *(pSrc1++ + 6) * *(kernel + 13)
						+ *pSrc2 * *(kernel + 14) + *(pSrc2 + 1) * *(kernel + 15) + *(pSrc2 + 2) * *(kernel + 16) + *(pSrc2 + 3) * *(kernel + 17) + *(pSrc2 + 4) * *(kernel + 18) + *(pSrc2 + 5) * *(kernel + 19) + *(pSrc2++ + 6) * *(kernel + 20)
						+ *pSrc3 * *(kernel + 21) + *(pSrc3 + 1) * *(kernel + 22) + *(pSrc3 + 2) * *(kernel + 23) + *(pSrc3 + 3) * *(kernel + 24) + *(pSrc3 + 4) * *(kernel + 25) + *(pSrc3 + 5) * *(kernel + 26) + *(pSrc3++ + 6) * *(kernel + 27)
						+ *pSrc4 * *(kernel + 28) + *(pSrc4 + 1) * *(kernel + 29) + *(pSrc4 + 2) * *(kernel + 30) + *(pSrc4 + 3) * *(kernel + 31) + *(pSrc4 + 4) * *(kernel + 32) + *(pSrc4 + 5) * *(kernel + 33) + *(pSrc4++ + 6) * *(kernel + 34)
						+ *pSrc5 * *(kernel + 35) + *(pSrc5 + 1) * *(kernel + 36) + *(pSrc5 + 2) * *(kernel + 37) + *(pSrc5 + 3) * *(kernel + 38) + *(pSrc5 + 4) * *(kernel + 39) + *(pSrc5 + 5) * *(kernel + 40) + *(pSrc5++ + 6) * *(kernel + 41)
						+ *pSrc6 * *(kernel + 42) + *(pSrc6 + 1) * *(kernel + 43) + *(pSrc6 + 2) * *(kernel + 44) + *(pSrc6 + 3) * *(kernel + 45) + *(pSrc6 + 4) * *(kernel + 46) + *(pSrc6 + 5) * *(kernel + 47) + *(pSrc6++ + 6) * *(kernel + 48)
						+ *pSrc7 * *(kernel + 49) + *(pSrc7 + 1) * *(kernel + 50) + *(pSrc7 + 2) * *(kernel + 51) + *(pSrc7 + 3) * *(kernel + 52) + *(pSrc7 + 4) * *(kernel + 53) + *(pSrc7 + 5) * *(kernel + 54) + *(pSrc7++ + 6) * *(kernel + 55);
				}
			}
		}
		void CNNPP::conv_8x8(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 7;
			if (H == 0) H = src_size_h - 7;

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
					*(pDst++) = *pSrc0 * *kernel + *(pSrc0 + 1) * *(kernel + 1) + *(pSrc0 + 2) * *(kernel + 2) + *(pSrc0 + 3) * *(kernel + 3) + *(pSrc0 + 4) * *(kernel + 4) + *(pSrc0 + 5) * *(kernel + 5) + *(pSrc0 + 6) * *(kernel + 6) + *(pSrc0++ + 7) * *(kernel + 7)
						+ *pSrc1 * *(kernel + 8) + *(pSrc1 + 1) * *(kernel + 9) + *(pSrc1 + 2) * *(kernel + 10) + *(pSrc1 + 3) * *(kernel + 11) + *(pSrc1 + 4) * *(kernel + 12) + *(pSrc1 + 5) * *(kernel + 13) + *(pSrc1 + 6) * *(kernel + 14) + *(pSrc1++ + 7) * *(kernel + 15)
						+ *pSrc2 * *(kernel + 16) + *(pSrc2 + 1) * *(kernel + 17) + *(pSrc2 + 2) * *(kernel + 18) + *(pSrc2 + 3) * *(kernel + 19) + *(pSrc2 + 4) * *(kernel + 20) + *(pSrc2 + 5) * *(kernel + 21) + *(pSrc2 + 6) * *(kernel + 22) + *(pSrc2++ + 7) * *(kernel + 23)
						+ *pSrc3 * *(kernel + 24) + *(pSrc3 + 1) * *(kernel + 25) + *(pSrc3 + 2) * *(kernel + 26) + *(pSrc3 + 3) * *(kernel + 27) + *(pSrc3 + 4) * *(kernel + 28) + *(pSrc3 + 5) * *(kernel + 29) + *(pSrc3 + 6) * *(kernel + 30) + *(pSrc3++ + 7) * *(kernel + 31)
						+ *pSrc4 * *(kernel + 32) + *(pSrc4 + 1) * *(kernel + 33) + *(pSrc4 + 2) * *(kernel + 34) + *(pSrc4 + 3) * *(kernel + 35) + *(pSrc4 + 4) * *(kernel + 36) + *(pSrc4 + 5) * *(kernel + 37) + *(pSrc4 + 6) * *(kernel + 38) + *(pSrc4++ + 7) * *(kernel + 39)
						+ *pSrc5 * *(kernel + 40) + *(pSrc5 + 1) * *(kernel + 41) + *(pSrc5 + 2) * *(kernel + 42) + *(pSrc5 + 3) * *(kernel + 43) + *(pSrc5 + 4) * *(kernel + 44) + *(pSrc5 + 5) * *(kernel + 45) + *(pSrc5 + 6) * *(kernel + 46) + *(pSrc5++ + 7) * *(kernel + 47)
						+ *pSrc6 * *(kernel + 48) + *(pSrc6 + 1) * *(kernel + 49) + *(pSrc6 + 2) * *(kernel + 50) + *(pSrc6 + 3) * *(kernel + 51) + *(pSrc6 + 4) * *(kernel + 52) + *(pSrc6 + 5) * *(kernel + 53) + *(pSrc6 + 6) * *(kernel + 54) + *(pSrc6++ + 7) * *(kernel + 55)
						+ *pSrc7 * *(kernel + 56) + *(pSrc7 + 1) * *(kernel + 57) + *(pSrc7 + 2) * *(kernel + 58) + *(pSrc7 + 3) * *(kernel + 59) + *(pSrc7 + 4) * *(kernel + 60) + *(pSrc7 + 5) * *(kernel + 61) + *(pSrc7 + 6) * *(kernel + 62) + *(pSrc7++ + 7) * *(kernel + 63);
				}
			}
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
					int t = 0;
					for (size_t y = 0; y < 11; ++y)
					{
						for (size_t x = 0; x < 10; ++x)
						{
							d += *(pSrc_[y] + x) * *(kernel + t++);
						}
					}
					*(pDst++) = d;
				}
			}
		}
		void CNNPP::conv_11x11(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H)
		{
			if (L == 0) L = src_size_l - 10;
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
					int t = 0;
					for (size_t y = 0; y < 11; ++y)
					{
						for (size_t x = 0; x < 11; ++x)
						{
							d += *(pSrc_[y] + x) * *(kernel + t++);
						}
					}
					*(pDst++) = d;
				}
			}
		}

		void CNNPP::tanh_avr_tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale)
		{
			int  j2 = 0;
			for (size_t j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + j2 * dst_size_l;

				for (size_t i = 0; i < src_size_l; i += 2)
				{
					float c1 = *(pSrc0++) + *conv_b;
#ifdef USE_FAST_TANH				
					float sgn = 1.f;
					if (c1 < 0.f) sgn = -1.f;
					//(int&)sgn |= ((int&)c1&  0x80000000);
					c1 = sgn * (1.f - 1.f / (1.f + fabsf(c1) + c1 * c1 + tanh_a * c1 * c1 * c1 * c1));
#else
					c1 = tanhf(c1);
#endif

					float c2 = *(pSrc0++) + *conv_b;
#ifdef USE_FAST_TANH
					sgn = 1.f;
					if (c2 < 0.f) sgn = -1.f;
					c2 = sgn * (1.f - 1.f / (1.f + fabsf(c2) + c2 * c2 + tanh_a * c2 * c2 * c2 * c2));
#else
					c2 = tanhf(c2);
#endif

					float c3 = *(pSrc1++) + *conv_b;
#ifdef USE_FAST_TANH
					sgn = 1.f;
					if (c3 < 0.f) sgn = -1.f;
					c3 = sgn * (1.f - 1.f / (1.f + fabsf(c3) + c3 * c3 + tanh_a * c3 * c3 * c3 * c3));
#else
					c3 = tanhf(c3);
#endif

					float c4 = *(pSrc1++) + *conv_b;
#ifdef USE_FAST_TANH
					sgn = 1.f;
					if (c4 < 0.f) sgn = -1.f;
					c4 = sgn * (1.f - 1.f / (1.f + fabsf(c4) + c4 * c4 + tanh_a * c4 * c4 * c4 * c4));
#else
					c4 = tanhf(c4);
#endif

					float avr = *subs_w * (c1 + c2 + c3 + c4) + *subs_b;
#ifdef USE_FAST_TANH			
					sgn = 1.f;
					if (avr < 0.f) sgn = -1.f;
					avr = sgn * (1.f - 1.f / (1.f + fabsf(avr) + avr * avr + tanh_a * avr * avr * avr * avr));
#else
					avr = tanhf(avr);
#endif

					*(pDst++) = *scale * avr;
				}
				j2++;
			}
		}
		void CNNPP::max_tanh_tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale)
		{
			int  j2 = 0;
			for (size_t j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + j2 * dst_size_l;

				for (size_t i = 0; i < src_size_l; i += 2)
				{
					float c1 = *(pSrc0++) + *conv_b;
#ifdef USE_FAST_TANH				
					float sgn = 1.f;
					if (c1 < 0.f) sgn = -1.f;
					//(int&)sgn |= ((int&)c1 & 0x80000000);
					c1 = sgn * (1.f - 1.f / (1.f + fabsf(c1) + c1 * c1 + tanh_a * c1 * c1 * c1 * c1));
#else
					c1 = tanhf(c1);
#endif

					float c2 = *(pSrc0++) + *conv_b;
#ifdef USE_FAST_TANH
					sgn = 1.f;
					if (c2 < 0.f) sgn = -1.f;
					c2 = sgn * (1.f - 1.f / (1.f + fabsf(c2) + c2 * c2 + tanh_a * c2 * c2 * c2 * c2));
#else
					c2 = tanhf(c2);
#endif

					float c3 = *(pSrc1++) + *conv_b;
#ifdef USE_FAST_TANH
					sgn = 1.f;
					if (c3 < 0.f) sgn = -1.f;
					c3 = sgn * (1.f - 1.f / (1.f + fabsf(c3) + c3 * c3 + tanh_a * c3 * c3 * c3 * c3));
#else
					c3 = tanhf(c3);
#endif

					float c4 = *(pSrc1++) + *conv_b;
#ifdef USE_FAST_TANH
					sgn = 1.f;
					if (c4 < 0.f) sgn = -1.f;
					c4 = sgn * (1.f - 1.f / (1.f + fabsf(c4) + c4 * c4 + tanh_a * c4 * c4 * c4 * c4));
#else
					c4 = tanhf(c4);
#endif

					float avr = *subs_w * fmaxf(fmaxf(c1, c2), fmaxf(c3, c4)) + *subs_b;
#ifdef USE_FAST_TANH			
					sgn = 1.f;
					if (avr < 0.f) sgn = -1.f;
					avr = sgn * (1.f - 1.f / (1.f + fabsf(avr) + avr * avr + tanh_a * avr * avr * avr * avr));
#else
					avr = tanhf(avr);
#endif

					*(pDst++) = *scale * avr;
				}
				j2++;
			}
		}
		void CNNPP::max_tanh_bn(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict bn_w, float* __restrict bn_b, float* __restrict scale)
		{
			int  j2 = 0;
			for (size_t j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + j2 * dst_size_l;

				for (size_t i = 0; i < src_size_l; i += 2)
				{
					float c1 = *(pSrc0++) + *conv_b;
#ifdef USE_FAST_TANH				
					float sgn = 1.f;
					if (c1 < 0.f) sgn = -1.f;
					c1 = sgn * (1.f - 1.f / (1.f + fabsf(c1) + c1 * c1 + tanh_a * c1 * c1 * c1 * c1));
#else
					c1 = tanhf(c1);
#endif

					float c2 = *(pSrc0++) + *conv_b;
#ifdef USE_FAST_TANH
					sgn = 1.f;
					if (c2 < 0.f) sgn = -1.f;
					c2 = sgn * (1.f - 1.f / (1.f + fabsf(c2) + c2 * c2 + tanh_a * c2 * c2 * c2 * c2));
#else
					c2 = tanhf(c2);
#endif

					float c3 = *(pSrc1++) + *conv_b;
#ifdef USE_FAST_TANH
					sgn = 1.f;
					if (c3 < 0.f) sgn = -1.f;
					c3 = sgn * (1.f - 1.f / (1.f + fabsf(c3) + c3 * c3 + tanh_a * c3 * c3 * c3 * c3));
#else
					c3 = tanhf(c3);
#endif

					float c4 = *(pSrc1++) + *conv_b;
#ifdef USE_FAST_TANH
					sgn = 1.f;
					if (c4 < 0.f) sgn = -1.f;
					c4 = sgn * (1.f - 1.f / (1.f + fabsf(c4) + c4 * c4 + tanh_a * c4 * c4 * c4 * c4));
#else
					c4 = tanhf(c4);
#endif

					float avr = fmaxf(fmaxf(c1, c2), fmaxf(c3, c4));
					//avr = (*scale * avr -* bn_m) / sqrtf(*bn_v);
					//avr = *bn_s * avr + *bn_b;
					avr = *bn_w * avr + *bn_b;

					*(pDst++) = avr;
				}
				j2++;
			}
		}

		void CNNPP::lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b)
		{
			int  j2 = 0;
			for (size_t j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + j2 * dst_size_l;

				for (size_t i = 0; i < src_size_l; i += 2)
				{
					float c1 = *(pSrc0++) + *conv_b;
					c1 = *lrelu_w1 * c1 + *lrelu_w2 * fmaxf(0.f, c1) + *bn_b;

					float c2 = *(pSrc0++) + *conv_b;
					c2 = *lrelu_w1 * c2 + *lrelu_w2 * fmaxf(0.f, c2) + *bn_b;

					float c3 = *(pSrc1++) + *conv_b;
					c3 = *lrelu_w1 * c3 + *lrelu_w2 * fmaxf(0.f, c3) + *bn_b;

					float c4 = *(pSrc1++) + *conv_b;
					c4 = *lrelu_w1 * c4 + *lrelu_w2 * fmaxf(0.f, c4) + *bn_b;

					float pool = fmaxf(fmaxf(c1, c2), fmaxf(c3, c4));

					*(pDst++) = pool;
				}
				j2++;
			}
		}
		void CNNPP::lrelu_bn(float* __restrict dst, float* __restrict src, int size_, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b)
		{
			float* __restrict pSrc = src;
			float* __restrict pDst = dst;
			for (size_t i = 0; i < size_; i += 4)
			{
				float c1 = *(pSrc++) + *conv_b;
				c1 = *lrelu_w1 * c1 + *lrelu_w2 * fmaxf(0.f, c1) + *bn_b;
				*(pDst++) = c1;

				float c2 = *(pSrc++) + *conv_b;
				c2 = *lrelu_w1 * c2 + *lrelu_w2 * fmaxf(0.f, c2) + *bn_b;
				*(pDst++) = c2;

				float c3 = *(pSrc++) + *conv_b;
				c3 = *lrelu_w1 * c3 + *lrelu_w2 * fmaxf(0.f, c3) + *bn_b;
				*(pDst++) = c3;

				float c4 = *(pSrc++) + *conv_b;
				c4 = *lrelu_w1 * c4 + *lrelu_w2 * fmaxf(0.f, c4) + *bn_b;
				*(pDst++) = c4;
			}
		}
		void CNNPP::mulCN_add_tanhW(int N, float* __restrict dst, float** __restrict src_N, int size_, float* __restrict hl_w_N, float* __restrict hl_b, float* __restrict tanh_w, float* __restrict bn_w, float* __restrict bn_b)
		{
			float** __restrict pSrc = new float*[N];
			for (size_t j = 0; j < N; ++j)
			{
				pSrc[j] = src_N[j];
			}
			float* __restrict pDst = dst;
			for (size_t i = 0; i < size_; ++i)
			{
				float c1 = *hl_b;
				for (size_t j = 0; j < N; ++j)
				{
					c1 += *(pSrc[j]++) * hl_w_N[j];
				}

				c1 *= *tanh_w;
				float sgn = 1.f;
				if (c1 < 0.f) sgn = -1.f;
				c1 = sgn * (1.f - 1.f / (1.f + fabsf(c1) + c1 * c1 + tanh_a * c1 * c1 * c1 * c1));
				c1 = 0.5f * c1 + 0.5f;

				*(pDst++) = *bn_w * c1 + *bn_b;
			}

			delete[] pSrc;
		}
		void CNNPP::tanhW(float* dst, float* src, int size_, float* __restrict snn_ol_b, float* __restrict tanh_w, float* __restrict scale)
		{
			float* pSrc = src;
			float* pDst = dst;
			for (size_t i = 0; i < size_; ++i)
			{
				float c1 = *(pSrc++) + *snn_ol_b;
				c1 *= *tanh_w;
				float sgn = 1.f;
				if (c1 < 0.f) sgn = -1.f;
				c1 = sgn * (1.f - 1.f / (1.f + fabsf(c1) + c1 * c1 + tanh_a * c1 * c1 * c1 * c1));
				
				if (*scale == 0.f)
					*(pDst++) = 0.5f * c1 + 0.5f;
				else
					*(pDst++) = *scale * c1;
			}
		}

		void CNNPP::tanh_tanh_2tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale, float* __restrict snn_hl_w0, float* __restrict snn_hl_b0, float* __restrict snn_hl_w1, float* __restrict snn_hl_b1, float* __restrict snn_ol_w0, float* __restrict snn_ol_w1)
		{
			float* __restrict pSrc = src;
			float* __restrict pDst = dst;
			for (size_t i = 0; i < size_; ++i)
			{
				float c = *(pSrc++) + *conv_b;
#ifdef USE_FAST_TANH			
				float sgn = 1.f;
				if (c < 0.f) sgn = -1.f;
				c = sgn * (1.f - 1.f / (1.f + fabsf(c) + c * c + tanh_a * c * c * c * c));
#else
				c = tanhf(c);
#endif

				c = *subs_w * c + *subs_b;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c < 0.f) sgn = -1.f;
				c = sgn * (1.f - 1.f / (1.f + fabsf(c) + c * c + tanh_a * c * c * c * c));
#else
				c = tanhf(c);
#endif

				float c_1 = *scale * *snn_hl_w0 * c + *snn_hl_b0;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c_1 < 0.f) sgn = -1.f;
				c_1 = sgn * (1.f - 1.f / (1.f + fabsf(c_1) + c_1 * c_1 + tanh_a * c_1 * c_1 * c_1 * c_1));
#else
				c_1 = tanhf(c_1);
#endif

				float c_2 = *scale * *snn_hl_w1 * c + *snn_hl_b1;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c_2 < 0.f) sgn = -1.f;
				c_2 = sgn * (1.f - 1.f / (1.f + fabsf(c_2) + c_2 * c_2 + tanh_a * c_2 * c_2 * c_2 * c_2));
#else
				c_2 = tanhf(c_2);
#endif

				*(pDst++) = *scale * (*snn_ol_w0 * c_1 + *snn_ol_w1 * c_2);
			}
		}
		void CNNPP::tanh_bn_2tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict conv_b, float* __restrict bn_w, float* __restrict bn_b, float* __restrict scale, float* __restrict snn_hl_w0, float* __restrict snn_hl_b0, float* __restrict snn_hl_w1, float* __restrict snn_hl_b1, float* __restrict snn_ol_w0, float* __restrict snn_ol_w1)
		{
			float* __restrict pSrc = src;
			float* __restrict pDst = dst;
			for (size_t i = 0; i < size_; ++i)
			{
				float c = *(pSrc++) + *conv_b;
#ifdef USE_FAST_TANH			
				float sgn = 1.f;
				if (c < 0.f) sgn = -1.f;
				c = sgn * (1.f - 1.f / (1.f + fabsf(c) + c * c + tanh_a * c * c * c * c));
#else
				c = tanhf(c);
#endif

				//c = (*scale * c -* bn_m) / sqrtf(*bn_v);
				//c = *bn_s * c + *bn_b;
				c = *bn_w * c + *bn_b;

				float c_1 = *snn_hl_w0 * c + *snn_hl_b0;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c_1 < 0.f) sgn = -1.f;
				c_1 = sgn * (1.f - 1.f / (1.f + fabsf(c_1) + c_1 * c_1 + tanh_a * c_1 * c_1 * c_1 * c_1));
#else
				c_1 = tanhf(c_1);
#endif

				float c_2 = * snn_hl_w1 * c + *snn_hl_b1;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c_2 < 0.f) sgn = -1.f;
				c_2 = sgn * (1.f - 1.f / (1.f + fabsf(c_2) + c_2 * c_2 + tanh_a * c_2 * c_2 * c_2 * c_2));
#else
				c_2 = tanhf(c_2);
#endif

				*(pDst++) = *scale * (*snn_ol_w0 * c_1 + *snn_ol_w1 * c_2);
			}
		}

		void CNNPP::tanh_tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale)
		{
			float* __restrict pSrc = src;
			float* __restrict pDst = dst;
			for (size_t i = 0; i < size_; i += 4)
			{
				float c1 = *(pSrc++) + *conv_b;
#ifdef USE_FAST_TANH
				float sgn = 1.f;
				if (c1 < 0.f) sgn = -1.f;
				c1 = sgn * (1.f - 1.f / (1.f + fabsf(c1) + c1 * c1 + tanh_a * c1 * c1 * c1 * c1));
#else
				c1 = tanhf(c1);
#endif

				c1 = *subs_w * c1 + *subs_b;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c1 < 0.f) sgn = -1.f;
				c1 = sgn * (1.f - 1.f / (1.f + fabsf(c1) + c1 * c1 + tanh_a * c1 * c1 * c1 * c1));
#else
				c1 = tanhf(c1);
#endif

				*(pDst++) = *scale * c1;


				float c2 = *(pSrc++) + *conv_b;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c2 < 0.f) sgn = -1.f;
				c2 = sgn * (1.f - 1.f / (1.f + fabsf(c2) + c2 * c2 + tanh_a * c2 * c2 * c2 * c2));
#else
				c2 = tanhf(c2);
#endif

				c2 = *subs_w * c2 + *subs_b;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c2 < 0.f) sgn = -1.f;
				c2 = sgn * (1.f - 1.f / (1.f + fabsf(c2) + c2 * c2 + tanh_a * c2 * c2 * c2 * c2));
#else
				c2 = tanhf(c2);
#endif

				*(pDst++) = *scale * c2;


				float c3 = *(pSrc++) + *conv_b;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c3 < 0.f) sgn = -1.f;
				c3 = sgn * (1.f - 1.f / (1.f + fabsf(c3) + c3 * c3 + tanh_a * c3 * c3 * c3 * c3));
#else
				c3 = tanhf(c3);
#endif

				c3 = *subs_w * c3 + *subs_b;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c3 < 0.f) sgn = -1.f;
				c3 = sgn * (1.f - 1.f / (1.f + fabsf(c3) + c3 * c3 + tanh_a * c3 * c3 * c3 * c3));
#else
				c3 = tanhf(c3);
#endif

				*(pDst++) = *scale * c3;


				float c4 = *(pSrc++) + *conv_b;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c4 < 0.f) sgn = -1.f;
				c4 = sgn * (1.f - 1.f / (1.f + fabsf(c4) + c4 * c4 + tanh_a * c4 * c4 * c4 * c4));
#else
				c4 = tanhf(c4);
#endif

				c4 = *subs_w * c4 + *subs_b;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c4 < 0.f) sgn = -1.f;
				c4 = sgn * (1.f - 1.f / (1.f + fabsf(c4) + c4 * c4 + tanh_a * c4 * c4 * c4 * c4));
#else
				c4 = tanhf(c4);
#endif

				*(pDst++) = *scale * c4;
			}
		}
		void CNNPP::tanh(float* __restrict dst, float* __restrict src, int size_, float* __restrict snn_ol_b, float* __restrict scale)
		{
			float* __restrict pSrc = src;
			float* __restrict pDst = dst;
			for (size_t i = 0; i < size_; i += 4)
			{
				float c1 = *(pSrc++) + *snn_ol_b;
#ifdef USE_FAST_TANH
				float sgn = 1.f;
				if (c1 < 0.f) sgn = -1.f;
				c1 = sgn * (1.f - 1.f / (1.f + fabsf(c1) + c1 * c1 + tanh_a * c1 * c1 * c1 * c1));
#else
				c1 = tanhf(c1);
#endif
				*(pDst++) = *scale * c1;

				float c2 = *(pSrc++) + *snn_ol_b;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c2 < 0.f) sgn = -1.f;
				c2 = sgn * (1.f - 1.f / (1.f + fabsf(c2) + c2 * c2 + tanh_a * c2 * c2 * c2 * c2));
#else
				c2 = tanhf(c2);
#endif		
				*(pDst++) = *scale * c2;

				float c3 = *(pSrc++) + *snn_ol_b;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c3 < 0.f) sgn = -1.f;
				c3 = sgn * (1.f - 1.f / (1.f + fabsf(c3) + c3 * c3 + tanh_a * c3 * c3 * c3 * c3));
#else
				c3 = tanhf(c3);
#endif	
				*(pDst++) = *scale * c3;

				float c4 = *(pSrc++) + *snn_ol_b;
#ifdef USE_FAST_TANH
				sgn = 1.f;
				if (c4 < 0.f) sgn = -1.f;
				c4 = sgn * (1.f - 1.f / (1.f + fabsf(c4) + c4 * c4 + tanh_a * c4 * c4 * c4 * c4));
#else
				c4 = tanhf(c4);
#endif
				*(pDst++) = *scale * c4;
			}
		}

		void CNNPP::add(float* __restrict dst, float* __restrict src1, float* __restrict src2, int size_)
		{
			float* __restrict pSrc1 = src1;
			float* __restrict pSrc2 = src2;
			float* __restrict pDst = dst;
			for (size_t i = 0; i < size_; i += 8)
			{
				*(pDst++) = *(pSrc1++) + *(pSrc2++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++);
			}
		}
		void CNNPP::add2(float* __restrict dst, float* __restrict src1, float* __restrict src2, float* __restrict src3, int size_)
		{
			float* __restrict pSrc1 = src1;
			float* __restrict pSrc2 = src2;
			float* __restrict pSrc3 = src3;
			float* __restrict pDst = dst;
			for (size_t i = 0; i < size_; i += 8)
			{
				*(pDst++) = *(pSrc1++) + *(pSrc2++) + *(pSrc3++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++) + *(pSrc3++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++) + *(pSrc3++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++) + *(pSrc3++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++) + *(pSrc3++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++) + *(pSrc3++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++) + *(pSrc3++);
				*(pDst++) = *(pSrc1++) + *(pSrc2++) + *(pSrc3++);
			}
		}

		void CNNPP::mulC(float* dst, float* src_mulC, int size_, float* __restrict snn_ol_w)
		{
			float* pSrc_mulC = src_mulC;
			float* pDst = dst;
			for (size_t i = 0; i < size_; i += 8)
			{
				*(pDst++) = *(pSrc_mulC++) * *snn_ol_w;
				*(pDst++) = *(pSrc_mulC++) * *snn_ol_w;
				*(pDst++) = *(pSrc_mulC++) * *snn_ol_w;
				*(pDst++) = *(pSrc_mulC++) * *snn_ol_w;
				*(pDst++) = *(pSrc_mulC++) * *snn_ol_w;
				*(pDst++) = *(pSrc_mulC++) * *snn_ol_w;
				*(pDst++) = *(pSrc_mulC++) * *snn_ol_w;
				*(pDst++) = *(pSrc_mulC++) * *snn_ol_w;
			}
		}
		void CNNPP::mulC1_add(float* dst, float* src1_mulC, float* src2, int size_, float* __restrict snn_hl_w)
		{
			float* pSrc1_mulC = src1_mulC;
			float* pSrc2 = src2;
			float* pDst = dst;
			for (size_t i = 0; i < size_; i += 8)
			{
				*(pDst++) = *(pSrc1_mulC++) * *snn_hl_w + *(pSrc2++);
				*(pDst++) = *(pSrc1_mulC++) * *snn_hl_w + *(pSrc2++);
				*(pDst++) = *(pSrc1_mulC++) * *snn_hl_w + *(pSrc2++);
				*(pDst++) = *(pSrc1_mulC++) * *snn_hl_w + *(pSrc2++);
				*(pDst++) = *(pSrc1_mulC++) * *snn_hl_w + *(pSrc2++);
				*(pDst++) = *(pSrc1_mulC++) * *snn_hl_w + *(pSrc2++);
				*(pDst++) = *(pSrc1_mulC++) * *snn_hl_w + *(pSrc2++);
				*(pDst++) = *(pSrc1_mulC++) * *snn_hl_w + *(pSrc2++);
			}
		}
		void CNNPP::mulC2_add(float* __restrict dst, float* __restrict src1_mulC0, float* __restrict src2_mulC1, int size_, float* __restrict snn_hl_w0, float* __restrict snn_hl_w1)
		{
			float* __restrict pSrc1_mulC0 = src1_mulC0;
			float* __restrict pSrc2_mulC1 = src2_mulC1;
			float* __restrict pDst = dst;
			for (size_t i = 0; i < size_; i += 8)
			{
				*(pDst++) = *(pSrc1_mulC0++) * *snn_hl_w0 + *(pSrc2_mulC1++) * *snn_hl_w1;
				*(pDst++) = *(pSrc1_mulC0++) * *snn_hl_w0 + *(pSrc2_mulC1++) * *snn_hl_w1;
				*(pDst++) = *(pSrc1_mulC0++) * *snn_hl_w0 + *(pSrc2_mulC1++) * *snn_hl_w1;
				*(pDst++) = *(pSrc1_mulC0++) * *snn_hl_w0 + *(pSrc2_mulC1++) * *snn_hl_w1;
				*(pDst++) = *(pSrc1_mulC0++) * *snn_hl_w0 + *(pSrc2_mulC1++) * *snn_hl_w1;
				*(pDst++) = *(pSrc1_mulC0++) * *snn_hl_w0 + *(pSrc2_mulC1++) * *snn_hl_w1;
				*(pDst++) = *(pSrc1_mulC0++) * *snn_hl_w0 + *(pSrc2_mulC1++) * *snn_hl_w1;
				*(pDst++) = *(pSrc1_mulC0++) * *snn_hl_w0 + *(pSrc2_mulC1++) * *snn_hl_w1;
			}
		}

		void CNNPP::mulC24_add_tanh(float* __restrict dst, float* __restrict* src, int size_, float* __restrict snn_hl_w, float* __restrict snn_hl_b, float* __restrict scale, float* __restrict snn_ol_w)
		{
			float* __restrict pDst = dst;
			for (size_t i = 0; i < size_; ++i)
			{
				float c = *snn_hl_b
					+ src[0][i] * *snn_hl_w
					+ src[1][i] * *(snn_hl_w + 1)
					+ src[2][i] * *(snn_hl_w + 2)
					+ src[3][i] * *(snn_hl_w + 3)
					+ src[4][i] * *(snn_hl_w + 4)
					+ src[5][i] * *(snn_hl_w + 5)
					+ src[6][i] * *(snn_hl_w + 6)
					+ src[7][i] * *(snn_hl_w + 7)
					+ src[8][i] * *(snn_hl_w + 8)
					+ src[9][i] * *(snn_hl_w + 9)
					+ src[10][i] * *(snn_hl_w + 10)
					+ src[11][i] * *(snn_hl_w + 11)
					+ src[12][i] * *(snn_hl_w + 12)
					+ src[13][i] * *(snn_hl_w + 13)
					+ src[14][i] * *(snn_hl_w + 14)
					+ src[15][i] * *(snn_hl_w + 15)
					+ src[16][i] * *(snn_hl_w + 16)
					+ src[17][i] * *(snn_hl_w + 17)
					+ src[18][i] * *(snn_hl_w + 18)
					+ src[19][i] * *(snn_hl_w + 19)
					+ src[20][i] * *(snn_hl_w + 20)
					+ src[21][i] * *(snn_hl_w + 21)
					+ src[22][i] * *(snn_hl_w + 22)
					+ src[23][i] * *(snn_hl_w + 23);

#ifdef USE_FAST_TANH
				float sgn = 1.f;
				if (c < 0.f) sgn = -1.f;
				c = sgn * (1.f - 1.f / (1.f + fabsf(c) + c * c + tanh_a * c * c * c * c));
#else
				c = tanhf(c);
#endif

				*(pDst++) = *scale * c * *snn_ol_w;
			}
		}
	}

#endif
}