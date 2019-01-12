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

#include "cnnpp_simd_avx_v2.h"
#include <immintrin.h>

//#define USE_IACA
#ifdef USE_IACA
#	include <intrin.h>
#	include <iacaMarks.h>
#	ifdef __INTEL_COMPILER 
#		define IACA__START IACA_START
#		define IACA__END IACA_END
#	else
#		define IACA__START IACA_VC64_START
#		define IACA__END IACA_VC64_END
#	endif
#else
#	define IACA__START
#	define IACA__END
#endif


//========================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_AVX

	namespace SIMD
	{

#ifndef USE_FMA

		#define conv_block(k, id)													\
				ymm_d2 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 177);					\
				ymm_d2 = _mm256_add_ps(ymm_d1, ymm_d2);								\
				ymm_shf = _mm256_shuffle_ps(ymm_d1, ymm_d1, 27);					\
				ymm_ml = _mm256_permute2f128_ps(ymm_shf, ymm_shf, 1);				\
				ymm_shf = _mm256_blend_ps(ymm_shf, ymm_ml, 24);						\
				ymm_d1 = _mm256_blend_ps(ymm_shf, _mm256_setzero_ps(), 129);		\
				ymm_d1 = _mm256_add_ps(ymm_d1, ymm_d2);								\
				ymm_ml = _mm256_mul_ps(ymm_d1, ymm_k_1_##id[k]);					\
				sum_1 = _mm256_add_ps(ymm_ml, sum_1);								\
				ymm_ml = _mm256_mul_ps(ymm_d1, ymm_k_2_##id[k]);					\
				sum_2 = _mm256_add_ps(ymm_ml, sum_2);					

		void CNNPP_v2::conv_4x4(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;

			const __m256 ymm_k11 = _mm256_load_ps(kernel + 0 * REG_SIZE);
			const __m256 ymm_k12 = _mm256_load_ps(kernel + 1 * REG_SIZE);
			const __m256 ymm_k13 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			const __m256 ymm_k14 = _mm256_load_ps(kernel + 3 * REG_SIZE);

			const __m256 ymm_k21 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			const __m256 ymm_k22 = _mm256_load_ps(kernel + 5 * REG_SIZE);
			const __m256 ymm_k23 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			const __m256 ymm_k24 = _mm256_load_ps(kernel + 7 * REG_SIZE);

			const __m256 ymm_k31 = _mm256_load_ps(kernel + 8 * REG_SIZE);
			const __m256 ymm_k32 = _mm256_load_ps(kernel + 9 * REG_SIZE);
			const __m256 ymm_k33 = _mm256_load_ps(kernel + 10 * REG_SIZE);
			const __m256 ymm_k34 = _mm256_load_ps(kernel + 11 * REG_SIZE);

			const __m256 ymm_k41 = _mm256_load_ps(kernel + 12 * REG_SIZE);
			const __m256 ymm_k42 = _mm256_load_ps(kernel + 13 * REG_SIZE);
			const __m256 ymm_k43 = _mm256_load_ps(kernel + 14 * REG_SIZE);
			const __m256 ymm_k44 = _mm256_load_ps(kernel + 15 * REG_SIZE);

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i <= L - 2; i += 2)
				{
					//0
					__m256 ymm_d = _mm256_broadcast_ss(pSrc0 + 0);
					__m256 sum = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 128));

					ymm_d = _mm256_broadcast_ss(pSrc0 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k12);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k13);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k14);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 4);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 24));
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc0 += 2;

					//1
					ymm_d = _mm256_broadcast_ss(pSrc1 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k22);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k23);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k24);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 4);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 24));
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc1 += 2;

					//2
					ymm_d = _mm256_broadcast_ss(pSrc2 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k32);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k33);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k34);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 4);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 24));
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc2 += 2;

					//3
					ymm_d = _mm256_broadcast_ss(pSrc3 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k42);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k43);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k44);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 4);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 24));
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc3 += 2;


					ymm_d = _mm256_permute2f128_ps(sum, sum, 1);
					sum = _mm256_max_ps(sum, ymm_d);

					_mm_store_ps(pDst, _mm256_extractf128_ps(sum, 0));
					pDst += REG_SIZE / 2;
				}
				IACA__END

				if (L & 1)
				{
					//0
					__m256 ymm_d = _mm256_broadcast_ss(pSrc0 + 0);
					__m256 sum = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 128));

					ymm_d = _mm256_broadcast_ss(pSrc0 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k12);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k13);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k14);
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc0 += 2;

					//1
					ymm_d = _mm256_broadcast_ss(pSrc1 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k22);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k23);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k24);
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc1 += 2;

					//2
					ymm_d = _mm256_broadcast_ss(pSrc2 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k32);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k33);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k34);
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc2 += 2;

					//3
					ymm_d = _mm256_broadcast_ss(pSrc3 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k42);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k43);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k44);
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc3 += 2;

					_mm_store_ps(pDst, _mm256_extractf128_ps(sum, 0));
				}
			}
		}
		void CNNPP_v2::conv_3x3(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 2;
			if (H == 0) H = src_size_h - 2;

			const __m256 ymm_k11 = _mm256_load_ps(kernel + 0 * REG_SIZE);
			const __m256 ymm_k12 = _mm256_load_ps(kernel + 1 * REG_SIZE);
			const __m256 ymm_k13 = _mm256_load_ps(kernel + 2 * REG_SIZE);

			const __m256 ymm_k21 = _mm256_load_ps(kernel + 3 * REG_SIZE);
			const __m256 ymm_k22 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			__m256 ymm_k23 = _mm256_load_ps(kernel + 5 * REG_SIZE);

			const __m256 ymm_k31 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			const __m256 ymm_k32 = _mm256_load_ps(kernel + 7 * REG_SIZE);
			const __m256 ymm_k33 = _mm256_load_ps(kernel + 8 * REG_SIZE);


			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i <= L - 2; i += 2)
				{
					//0
					__m256 ymm_d1 = _mm256_load_ps(pSrc0 + 0);
					__m256 ymm_d2 = _mm256_load_ps(pSrc0 + REG_SIZE);
					pSrc0 += REG_SIZE;

					__m256 ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					__m256 ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					__m256 sum_1 = _mm256_mul_ps(ymm_d3, ymm_k11);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k12), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k13), sum_1);

					__m256 sum_2 = _mm256_mul_ps(ymm_d1, ymm_k11);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k12), sum_2);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k13), sum_2);

					//1
					ymm_d1 = _mm256_load_ps(pSrc1 + 0);
					ymm_d2 = _mm256_load_ps(pSrc1 + REG_SIZE);
					pSrc1 += REG_SIZE;

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d3, ymm_k21), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k22), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k23), sum_1);

					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k21), sum_2);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k22), sum_2);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k23), sum_2);

					//2
					ymm_d1 = _mm256_load_ps(pSrc2 + 0);
					ymm_d2 = _mm256_load_ps(pSrc2 + REG_SIZE);
					pSrc2 += REG_SIZE;

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d3, ymm_k31), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k32), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k33), sum_1);

					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k31), sum_2);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k32), sum_2);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k33), sum_2);

					sum_1 = _mm256_max_ps(sum_1, sum_2);

					ymm_d2 = _mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(sum_1), _mm256_castps_pd(sum_1), 6));
					sum_1 = _mm256_permute2f128_ps(ymm_d2, ymm_d2, 1);
					sum_1 = _mm256_blend_ps(sum_1, ymm_d2, 51);

					_mm256_store_ps(pDst, sum_1);
					pDst += REG_SIZE;
				}
				IACA__END

				if (L & 1)
				{
					//0
					__m256 ymm_d1 = _mm256_load_ps(pSrc0 + 0);
					__m256 ymm_d2 = _mm256_load_ps(pSrc0 + REG_SIZE);

					__m256 ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					__m256 ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					__m256 sum_1 = _mm256_mul_ps(ymm_d1, ymm_k11);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k12), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d3, ymm_k13), sum_1);

					//1
					ymm_d1 = _mm256_load_ps(pSrc1 + 0);
					ymm_d2 = _mm256_load_ps(pSrc1 + REG_SIZE);

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k21), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k22), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d3, ymm_k23), sum_1);

					//2
					ymm_d1 = _mm256_load_ps(pSrc2 + 0);
					ymm_d2 = _mm256_load_ps(pSrc2 + REG_SIZE);

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k31), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k32), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d3, ymm_k33), sum_1);

					ymm_d2 = _mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(sum_1), _mm256_castps_pd(sum_1), 6));
					sum_1 = _mm256_permute2f128_ps(ymm_d2, ymm_d2, 1);
					sum_1 = _mm256_blend_ps(sum_1, ymm_d2, 51);

					_mm256_store_ps(pDst, sum_1);
				}
			}
		}
		void CNNPP_v2::conv_6x5(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 4;
			if (H == 0) H = src_size_h - 5;

			__m256 ymm_k_1_0[6];
			__m256 ymm_k_1_1[6];
			__m256 ymm_k_1_2[6];
			__m256 ymm_k_1_3[6];
			__m256 ymm_k_1_4[6];

			for (size_t k = 0; k < 6; ++k)
			{
				ymm_k_1_0[k] = _mm256_load_ps(kernel + (0 + 5 * k) * REG_SIZE);
				ymm_k_1_1[k] = _mm256_load_ps(kernel + (1 + 5 * k) * REG_SIZE);
				ymm_k_1_2[k] = _mm256_load_ps(kernel + (2 + 5 * k) * REG_SIZE);
				ymm_k_1_3[k] = _mm256_load_ps(kernel + (3 + 5 * k) * REG_SIZE);
				ymm_k_1_4[k] = _mm256_load_ps(kernel + (4 + 5 * k) * REG_SIZE);
			}

			__m256 ymm_k_2_0[6];
			__m256 ymm_k_2_1[6];
			__m256 ymm_k_2_2[6];
			__m256 ymm_k_2_3[6];
			__m256 ymm_k_2_4[6];

			kernel += 30 * REG_SIZE;
			for (size_t k = 0; k < 6; ++k)
			{
				ymm_k_2_0[k] = _mm256_load_ps(kernel + (0 + 5 * k) * REG_SIZE);
				ymm_k_2_1[k] = _mm256_load_ps(kernel + (1 + 5 * k) * REG_SIZE);
				ymm_k_2_2[k] = _mm256_load_ps(kernel + (2 + 5 * k) * REG_SIZE);
				ymm_k_2_3[k] = _mm256_load_ps(kernel + (3 + 5 * k) * REG_SIZE);
				ymm_k_2_4[k] = _mm256_load_ps(kernel + (4 + 5 * k) * REG_SIZE);
			}

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc = src + j * src_size_l;
				float* __restrict pDst_1 = dst + j * dst_size_l;
				float* __restrict pDst_2 = dst + j * dst_size_l + dst_size_l / 2;

				IACA__START
				for (size_t i = 0; i < L; ++i)
				{
					__m256 sum_1 = _mm256_setzero_ps();
					__m256 sum_2 = _mm256_setzero_ps();

					#pragma unroll
					for (size_t k = 0; k < 6; ++k)
					{
						float* __restrict pSrc_temp = pSrc + k * src_size_l;
						__m256 ymm_d1, ymm_d2, ymm_ml, ymm_shf;

						//0
						ymm_d1 = _mm256_load_ps(pSrc_temp);
						pSrc_temp += REG_SIZE;
						conv_block(k, 0);

						//1
						ymm_d1 = _mm256_load_ps(pSrc_temp);
						pSrc_temp += REG_SIZE;
						conv_block(k, 1);

						//2
						ymm_d1 = _mm256_load_ps(pSrc_temp);
						pSrc_temp += REG_SIZE;
						conv_block(k, 2);

						//3
						ymm_d1 = _mm256_load_ps(pSrc_temp);
						pSrc_temp += REG_SIZE;
						conv_block(k, 3);

						//4
						ymm_d1 = _mm256_load_ps(pSrc_temp);
						pSrc_temp += REG_SIZE;
						conv_block(k, 4);
					}
					pSrc += REG_SIZE;

					_mm256_store_ps(pDst_1, sum_1);
					pDst_1 += REG_SIZE;
					_mm256_store_ps(pDst_2, sum_2);
					pDst_2 += REG_SIZE;
				}
				IACA__END
			}
		}

		void CNNPP_v2::conv_4x4_lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;

			const __m256 ymm_k11 = _mm256_load_ps(kernel + 0 * REG_SIZE);
			const __m256 ymm_k12 = _mm256_load_ps(kernel + 1 * REG_SIZE);
			const __m256 ymm_k13 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			const __m256 ymm_k14 = _mm256_load_ps(kernel + 3 * REG_SIZE);

			const __m256 ymm_k21 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			const __m256 ymm_k22 = _mm256_load_ps(kernel + 5 * REG_SIZE);
			const __m256 ymm_k23 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			const __m256 ymm_k24 = _mm256_load_ps(kernel + 7 * REG_SIZE);

			const __m256 ymm_k31 = _mm256_load_ps(kernel + 8 * REG_SIZE);
			const __m256 ymm_k32 = _mm256_load_ps(kernel + 9 * REG_SIZE);
			const __m256 ymm_k33 = _mm256_load_ps(kernel + 10 * REG_SIZE);
			const __m256 ymm_k34 = _mm256_load_ps(kernel + 11 * REG_SIZE);

			const __m256 ymm_k41 = _mm256_load_ps(kernel + 12 * REG_SIZE);
			const __m256 ymm_k42 = _mm256_load_ps(kernel + 13 * REG_SIZE);
			const __m256 ymm_k43 = _mm256_load_ps(kernel + 14 * REG_SIZE);
			const __m256 ymm_k44 = _mm256_load_ps(kernel + 15 * REG_SIZE);

			const __m256 ymm_conv_b = _mm256_load_ps(conv_b);
			const __m256 ymm_lrelu_w1 = _mm256_load_ps(lrelu_w1);
			const __m256 ymm_lrelu_w2 = _mm256_load_ps(lrelu_w2);
			const __m256 ymm_bn_b = _mm256_load_ps(bn_b);
			const __m256 ymm_zero = _mm256_setzero_ps();

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i <= L - 2; i += 2)
				{
					//0
					__m256 ymm_d = _mm256_broadcast_ss(pSrc0 + 0);
					__m256 sum = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 128));

					ymm_d = _mm256_broadcast_ss(pSrc0 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k12);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k13);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k14);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 4);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 24));
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc0 += 2;

					//1
					ymm_d = _mm256_broadcast_ss(pSrc1 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k22);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k23);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k24);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 4);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 24));
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc1 += 2;

					//2
					ymm_d = _mm256_broadcast_ss(pSrc2 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k32);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k33);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k34);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 4);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 24));
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc2 += 2;

					//3
					ymm_d = _mm256_broadcast_ss(pSrc3 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k42);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k43);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k44);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 4);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 24));
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc3 += 2;

					//-----------------------------

					sum = _mm256_add_ps(sum, ymm_conv_b);
					ymm_d = _mm256_max_ps(sum, ymm_zero);

					sum = _mm256_mul_ps(sum, ymm_lrelu_w1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_lrelu_w2);

					sum = _mm256_add_ps(sum, ymm_d);
					sum = _mm256_add_ps(sum, ymm_bn_b);

					//-----------------------------

					ymm_d = _mm256_permute2f128_ps(sum, sum, 1);
					sum = _mm256_max_ps(sum, ymm_d);

					_mm_store_ps(pDst, _mm256_extractf128_ps(sum, 0));
					pDst += REG_SIZE / 2;
				}
				IACA__END

				if (L & 1)
				{
					//0
					__m256 ymm_d = _mm256_broadcast_ss(pSrc0 + 0);
					__m256 sum = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 128));

					ymm_d = _mm256_broadcast_ss(pSrc0 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k12);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k13);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k14);
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc0 += 2;

					//1
					ymm_d = _mm256_broadcast_ss(pSrc1 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k22);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k23);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k24);
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc1 += 2;

					//2
					ymm_d = _mm256_broadcast_ss(pSrc2 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k32);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k33);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k34);
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc2 += 2;

					//3
					ymm_d = _mm256_broadcast_ss(pSrc3 + 0);
					ymm_d = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 128));
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k42);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 2);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k43);
					sum = _mm256_add_ps(ymm_d, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 3);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_k44);
					sum = _mm256_add_ps(ymm_d, sum);
					pSrc3 += 2;

					//-----------------------------

					sum = _mm256_add_ps(sum, ymm_conv_b);
					ymm_d = _mm256_max_ps(sum, ymm_zero);

					sum = _mm256_mul_ps(sum, ymm_lrelu_w1);
					ymm_d = _mm256_mul_ps(ymm_d, ymm_lrelu_w2);

					sum = _mm256_add_ps(sum, ymm_d);
					sum = _mm256_add_ps(sum, ymm_bn_b);

					//-----------------------------

					_mm_store_ps(pDst, _mm256_extractf128_ps(sum, 0));
				}
			}
		}
		void CNNPP_v2::conv_3x3_lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 2;
			if (H == 0) H = src_size_h - 2;

			const __m256 ymm_k11 = _mm256_load_ps(kernel + 0 * REG_SIZE);
			const __m256 ymm_k12 = _mm256_load_ps(kernel + 1 * REG_SIZE);
			const __m256 ymm_k13 = _mm256_load_ps(kernel + 2 * REG_SIZE);

			const __m256 ymm_k21 = _mm256_load_ps(kernel + 3 * REG_SIZE);
			const __m256 ymm_k22 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			const __m256 ymm_k23 = _mm256_load_ps(kernel + 5 * REG_SIZE);

			const __m256 ymm_k31 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			const __m256 ymm_k32 = _mm256_load_ps(kernel + 7 * REG_SIZE);
			const __m256 ymm_k33 = _mm256_load_ps(kernel + 8 * REG_SIZE);

			const __m256 ymm_conv_b = _mm256_load_ps(conv_b);
			const __m256 ymm_lrelu_w1 = _mm256_load_ps(lrelu_w1);
			const __m256 ymm_lrelu_w2 = _mm256_load_ps(lrelu_w2);
			const __m256 ymm_bn_b = _mm256_load_ps(bn_b);
			const __m256 ymm_zero = _mm256_setzero_ps();

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + (j << 1) * src_size_l;
				float* __restrict pSrc0_2 = src + ((j << 1) + 1) * src_size_l;

				float* __restrict pSrc1 = src + ((j << 1) + 2) * src_size_l;
				float* __restrict pSrc1_2 = src + ((j << 1) + 3) * src_size_l;

				float* __restrict pSrc2 = src + ((j << 1) + 4) * src_size_l;
				float* __restrict pSrc2_2 = src + ((j << 1) + 5) * src_size_l;

				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i <= L - 2; i += 2)
				{
					//0
					__m256 ymm_d1 = _mm256_load_ps(pSrc0 + 0);
					__m256 ymm_d2 = _mm256_load_ps(pSrc0 + REG_SIZE);
					pSrc0 += REG_SIZE;

					//-----------------------------

					__m256 ymm_d1_2 = _mm256_load_ps(pSrc0_2 + 0);
					__m256 ymm_d2_2 = _mm256_load_ps(pSrc0_2 + REG_SIZE);
					pSrc0_2 += REG_SIZE;

					ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d1_2);
					ymm_d2 = _mm256_max_ps(ymm_d2, ymm_d2_2);

					//-----------------------------

					__m256 ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					__m256 ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					__m256 sum_1 = _mm256_mul_ps(ymm_d3, ymm_k11);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k12), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k13), sum_1);

					__m256 sum_2 = _mm256_mul_ps(ymm_d1, ymm_k11);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k12), sum_2);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k13), sum_2);

					//1
					ymm_d1 = _mm256_load_ps(pSrc1 + 0);
					ymm_d2 = _mm256_load_ps(pSrc1 + REG_SIZE);
					pSrc1 += REG_SIZE;

					//-----------------------------

					ymm_d1_2 = _mm256_load_ps(pSrc1_2 + 0);
					ymm_d2_2 = _mm256_load_ps(pSrc1_2 + REG_SIZE);
					pSrc1_2 += REG_SIZE;

					ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d1_2);
					ymm_d2 = _mm256_max_ps(ymm_d2, ymm_d2_2);

					//-----------------------------

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d3, ymm_k21), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k22), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k23), sum_1);

					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k21), sum_2);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k22), sum_2);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k23), sum_2);

					//2
					ymm_d1 = _mm256_load_ps(pSrc2 + 0);
					ymm_d2 = _mm256_load_ps(pSrc2 + REG_SIZE);
					pSrc2 += REG_SIZE;

					//-----------------------------

					ymm_d1_2 = _mm256_load_ps(pSrc2_2 + 0);
					ymm_d2_2 = _mm256_load_ps(pSrc2_2 + REG_SIZE);
					pSrc2_2 += REG_SIZE;

					ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d1_2);
					ymm_d2 = _mm256_max_ps(ymm_d2, ymm_d2_2);

					//-----------------------------

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d3, ymm_k31), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k32), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k33), sum_1);

					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k31), sum_2);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d4, ymm_k32), sum_2);
					sum_2 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k33), sum_2);

					//-----------------------------

					sum_1 = _mm256_add_ps(sum_1, ymm_conv_b);
					ymm_d1 = _mm256_max_ps(sum_1, ymm_zero);

					sum_1 = _mm256_mul_ps(sum_1, ymm_lrelu_w1);
					ymm_d1 = _mm256_mul_ps(ymm_d1, ymm_lrelu_w2);

					sum_1 = _mm256_add_ps(sum_1, ymm_d1);
					sum_1 = _mm256_add_ps(sum_1, ymm_bn_b);

					sum_2 = _mm256_add_ps(sum_2, ymm_conv_b);
					ymm_d2 = _mm256_max_ps(sum_2, ymm_zero);

					sum_2 = _mm256_mul_ps(sum_2, ymm_lrelu_w1);
					ymm_d2 = _mm256_mul_ps(ymm_d2, ymm_lrelu_w2);

					sum_2 = _mm256_add_ps(sum_2, ymm_d2);
					sum_2 = _mm256_add_ps(sum_2, ymm_bn_b);

					//-----------------------------

					sum_1 = _mm256_max_ps(sum_1, sum_2);

					ymm_d2 = _mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(sum_1), _mm256_castps_pd(sum_1), 6));
					sum_1 = _mm256_permute2f128_ps(ymm_d2, ymm_d2, 1);
					sum_1 = _mm256_blend_ps(sum_1, ymm_d2, 51);

					_mm256_store_ps(pDst, sum_1);
					pDst += REG_SIZE;
				}
				IACA__END

				if (L & 1)
				{
					//0
					__m256 ymm_d1 = _mm256_load_ps(pSrc0 + 0);
					__m256 ymm_d2 = _mm256_load_ps(pSrc0 + REG_SIZE);

					__m256 ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					__m256 ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					__m256 sum_1 = _mm256_mul_ps(ymm_d1, ymm_k11);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k12), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d3, ymm_k13), sum_1);

					//1
					ymm_d1 = _mm256_load_ps(pSrc1 + 0);
					ymm_d2 = _mm256_load_ps(pSrc1 + REG_SIZE);

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k21), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k22), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d3, ymm_k23), sum_1);

					//2
					ymm_d1 = _mm256_load_ps(pSrc2 + 0);
					ymm_d2 = _mm256_load_ps(pSrc2 + REG_SIZE);

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d1, ymm_k31), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d2, ymm_k32), sum_1);
					sum_1 = _mm256_add_ps(_mm256_mul_ps(ymm_d3, ymm_k33), sum_1);

					//-----------------------------

					sum_1 = _mm256_add_ps(sum_1, ymm_conv_b);
					ymm_d1 = _mm256_max_ps(sum_1, ymm_zero);

					sum_1 = _mm256_mul_ps(sum_1, ymm_lrelu_w1);
					ymm_d1 = _mm256_mul_ps(ymm_d1, ymm_lrelu_w2);

					sum_1 = _mm256_add_ps(sum_1, ymm_d1);
					sum_1 = _mm256_add_ps(sum_1, ymm_bn_b);

					//-----------------------------

					ymm_d2 = _mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(sum_1), _mm256_castps_pd(sum_1), 6));
					sum_1 = _mm256_permute2f128_ps(ymm_d2, ymm_d2, 1);
					sum_1 = _mm256_blend_ps(sum_1, ymm_d2, 51);

					_mm256_store_ps(pDst, sum_1);
				}
			}
		}
		void CNNPP_v2::conv_5x4_lrelu_bn(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 4;
			if (H == 0) H = src_size_h - 5;

			__m256 ymm_k_1_0[5];
			__m256 ymm_k_1_1[5];
			__m256 ymm_k_1_2[5];
			__m256 ymm_k_1_3[5];

			for (size_t k = 0; k < 5; ++k)
			{
				ymm_k_1_0[k] = _mm256_load_ps(kernel + (0 + 4 * k) * REG_SIZE);
				ymm_k_1_1[k] = _mm256_load_ps(kernel + (1 + 4 * k) * REG_SIZE);
				ymm_k_1_2[k] = _mm256_load_ps(kernel + (2 + 4 * k) * REG_SIZE);
				ymm_k_1_3[k] = _mm256_load_ps(kernel + (3 + 4 * k) * REG_SIZE);
			}

			__m256 ymm_k_2_0[5];
			__m256 ymm_k_2_1[5];
			__m256 ymm_k_2_2[5];
			__m256 ymm_k_2_3[5];

			kernel += 5 * 4 * REG_SIZE;
			for (size_t k = 0; k < 5; ++k)
			{
				ymm_k_2_0[k] = _mm256_load_ps(kernel + (0 + 4 * k) * REG_SIZE);
				ymm_k_2_1[k] = _mm256_load_ps(kernel + (1 + 4 * k) * REG_SIZE);
				ymm_k_2_2[k] = _mm256_load_ps(kernel + (2 + 4 * k) * REG_SIZE);
				ymm_k_2_3[k] = _mm256_load_ps(kernel + (3 + 4 * k) * REG_SIZE);
			}

			const __m256 ymm_conv_b_1 = _mm256_load_ps(conv_b);
			const __m256 ymm_conv_b_2 = _mm256_load_ps(conv_b + REG_SIZE);
			const __m256 ymm_lrelu_w1_1 = _mm256_load_ps(lrelu_w1);
			const __m256 ymm_lrelu_w1_2 = _mm256_load_ps(lrelu_w1 + REG_SIZE);
			const __m256 ymm_lrelu_w2_1 = _mm256_load_ps(lrelu_w2);
			const __m256 ymm_lrelu_w2_2 = _mm256_load_ps(lrelu_w2 + REG_SIZE);
			const __m256 ymm_bn_b_1 = _mm256_load_ps(bn_b);
			const __m256 ymm_bn_b_2 = _mm256_load_ps(bn_b + REG_SIZE);
			const __m256 ymm_zero = _mm256_setzero_ps();

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc = src + (j << 1) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i < L; ++i)
				{
					__m256 sum_1 = _mm256_setzero_ps();
					__m256 sum_2 = _mm256_setzero_ps();

					#pragma unroll
					for (size_t k = 0; k < 5; ++k)
					{
						float* __restrict pSrc_temp_1 = pSrc + (k << 1) * src_size_l;
						float* __restrict pSrc_temp_2 = pSrc + ((k << 1) + 1) * src_size_l;
						__m256 ymm_d1, ymm_d2, ymm_ml, ymm_shf;

						//0
						ymm_d1 = _mm256_load_ps(pSrc_temp_1);
						ymm_d2 = _mm256_load_ps(pSrc_temp_2);
						pSrc_temp_1 += REG_SIZE;
						pSrc_temp_2 += REG_SIZE;
						ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d2);
						conv_block(k, 0);

						//1
						ymm_d1 = _mm256_load_ps(pSrc_temp_1);
						ymm_d2 = _mm256_load_ps(pSrc_temp_2);
						pSrc_temp_1 += REG_SIZE;
						pSrc_temp_2 += REG_SIZE;
						ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d2);
						conv_block(k, 1);

						//2
						ymm_d1 = _mm256_load_ps(pSrc_temp_1);
						ymm_d2 = _mm256_load_ps(pSrc_temp_2);
						pSrc_temp_1 += REG_SIZE;
						pSrc_temp_2 += REG_SIZE;
						ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d2);
						conv_block(k, 2);

						//3
						ymm_d1 = _mm256_load_ps(pSrc_temp_1);
						ymm_d2 = _mm256_load_ps(pSrc_temp_2);
						pSrc_temp_1 += REG_SIZE;
						pSrc_temp_2 += REG_SIZE;
						ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d2);
						conv_block(k, 3);
					}
					pSrc += REG_SIZE;

					//-----------------------------

					sum_1 = _mm256_add_ps(sum_1, ymm_conv_b_1);
					__m256 ymm_d1 = _mm256_max_ps(sum_1, ymm_zero);

					sum_1 = _mm256_mul_ps(sum_1, ymm_lrelu_w1_1);
					ymm_d1 = _mm256_mul_ps(ymm_d1, ymm_lrelu_w2_1);

					sum_1 = _mm256_add_ps(sum_1, ymm_d1);
					sum_1 = _mm256_add_ps(sum_1, ymm_bn_b_1);

					sum_2 = _mm256_add_ps(sum_2, ymm_conv_b_2);
					__m256 ymm_d2 = _mm256_max_ps(sum_2, ymm_zero);

					sum_2 = _mm256_mul_ps(sum_2, ymm_lrelu_w1_2);
					ymm_d2 = _mm256_mul_ps(ymm_d2, ymm_lrelu_w2_2);

					sum_2 = _mm256_add_ps(sum_2, ymm_d2);
					sum_2 = _mm256_add_ps(sum_2, ymm_bn_b_2);

					//-----------------------------

					_mm256_store_ps(pDst, sum_1);
					_mm256_store_ps(pDst + REG_SIZE, sum_2);
					pDst += 2 * REG_SIZE;
				}
				IACA__END
			}
		}

#else

		#define conv_block(k, id)												    \
				ymm_d2 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 177);					\
				ymm_d2 = _mm256_add_ps(ymm_d1, ymm_d2);								\
				ymm_d1 = _mm256_permutevar8x32_ps(ymm_d1, ymm_mask_temp);			\
				ymm_d1 = _mm256_blend_ps(ymm_d1, _mm256_setzero_ps(), 129);			\
				ymm_d1 = _mm256_add_ps(ymm_d1, ymm_d2);								\
				sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k_1_##id[k], sum_1);			\
				sum_2 = _mm256_fmadd_ps(ymm_d1, ymm_k_2_##id[k], sum_2);					

		void CNNPP_v2::conv_4x4(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;

			__m256 ymm_k11 = _mm256_load_ps(kernel + 0 * REG_SIZE);
			__m256 ymm_k12 = _mm256_load_ps(kernel + 1 * REG_SIZE);
			__m256 ymm_k13 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			__m256 ymm_k14 = _mm256_load_ps(kernel + 3 * REG_SIZE);
	
			__m256 ymm_k21 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			__m256 ymm_k22 = _mm256_load_ps(kernel + 5 * REG_SIZE);
			__m256 ymm_k23 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			__m256 ymm_k24 = _mm256_load_ps(kernel + 7 * REG_SIZE);

			__m256 ymm_k31 = _mm256_load_ps(kernel + 8 * REG_SIZE);
			__m256 ymm_k32 = _mm256_load_ps(kernel + 9 * REG_SIZE);
			__m256 ymm_k33 = _mm256_load_ps(kernel + 10 * REG_SIZE);
			__m256 ymm_k34 = _mm256_load_ps(kernel + 11 * REG_SIZE);

			__m256 ymm_k41 = _mm256_load_ps(kernel + 12 * REG_SIZE);
			__m256 ymm_k42 = _mm256_load_ps(kernel + 13 * REG_SIZE);
			__m256 ymm_k43 = _mm256_load_ps(kernel + 14 * REG_SIZE);
			__m256 ymm_k44 = _mm256_load_ps(kernel + 15 * REG_SIZE);

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;
	
				IACA__START
				for (size_t i = 0; i <= L - 2; i += 2)
				{
					//0
					__m256 ymm_d = _mm256_broadcast_ss(pSrc0 + 0);
					__m256 sum = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 128));

					ymm_d = _mm256_broadcast_ss(pSrc0 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k12, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k13, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k14, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 4);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 24), sum);
					pSrc0 += 2;

					//1
					ymm_d = _mm256_broadcast_ss(pSrc1 + 0);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 128), sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k22, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k23, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k24, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 4);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 24), sum);
					pSrc1 += 2;

					//2
					ymm_d = _mm256_broadcast_ss(pSrc2 + 0);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 128), sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k32, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k33, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k34, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 4);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 24), sum);
					pSrc2 += 2;

					//3
					ymm_d = _mm256_broadcast_ss(pSrc3 + 0);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 128), sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k42, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k43, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k44, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 4);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 24), sum);
					pSrc3 += 2;

					ymm_d = _mm256_permute2f128_ps(sum, sum, 1);
					sum = _mm256_max_ps(sum, ymm_d);

					_mm_store_ps(pDst, _mm256_extractf128_ps(sum, 0));
					pDst += REG_SIZE / 2;
				}
				IACA__END

				if (L & 1)
				{
					//0
					__m256 ymm_d = _mm256_broadcast_ss(pSrc0 + 0);
					__m256 sum = _mm256_mul_ps(ymm_d, ymm_k11);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k12, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k13, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k14, sum);

					//1
					ymm_d = _mm256_broadcast_ss(pSrc1 + 0);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k21, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k22, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k23, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k24, sum);

					//2
					ymm_d = _mm256_broadcast_ss(pSrc2 + 0);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k31, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k32, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k33, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k34, sum);

					//3
					ymm_d = _mm256_broadcast_ss(pSrc3 + 0);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k41, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k42, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k43, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k44, sum);

					_mm_store_ps(pDst, _mm256_extractf128_ps(sum, 0));
				}
			}
		}
		void CNNPP_v2::conv_3x3(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 2;
			if (H == 0) H = src_size_h - 2;

			__m256 ymm_k11 = _mm256_load_ps(kernel + 0 * REG_SIZE);
			__m256 ymm_k12 = _mm256_load_ps(kernel + 1 * REG_SIZE);
			__m256 ymm_k13 = _mm256_load_ps(kernel + 2 * REG_SIZE);

			__m256 ymm_k21 = _mm256_load_ps(kernel + 3 * REG_SIZE);
			__m256 ymm_k22 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			__m256 ymm_k23 = _mm256_load_ps(kernel + 5 * REG_SIZE);

			__m256 ymm_k31 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			__m256 ymm_k32 = _mm256_load_ps(kernel + 7 * REG_SIZE);
			__m256 ymm_k33 = _mm256_load_ps(kernel + 8 * REG_SIZE);

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i <= L - 2; i += 2)
				{
					//0
					__m256 ymm_d1 = _mm256_load_ps(pSrc0 + 0);
					__m256 ymm_d2 = _mm256_load_ps(pSrc0 + REG_SIZE);
					pSrc0 += REG_SIZE;

					__m256 ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					__m256 ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					__m256 sum_1 = _mm256_mul_ps(ymm_d3, ymm_k11);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k12, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d4, ymm_k13, sum_1);

					__m256 sum_2 = _mm256_mul_ps(ymm_d1, ymm_k11);
					sum_2 = _mm256_fmadd_ps(ymm_d4, ymm_k12, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_k13, sum_2);

					//1
					ymm_d1 = _mm256_load_ps(pSrc1 + 0);
					ymm_d2 = _mm256_load_ps(pSrc1 + REG_SIZE);
					pSrc1 += REG_SIZE;

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k21, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k22, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d4, ymm_k23, sum_1);

					sum_2 = _mm256_fmadd_ps(ymm_d1, ymm_k21, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d4, ymm_k22, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_k23, sum_2);

					//2
					ymm_d1 = _mm256_load_ps(pSrc2 + 0);
					ymm_d2 = _mm256_load_ps(pSrc2 + REG_SIZE);
					pSrc2 += REG_SIZE;

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k31, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k32, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d4, ymm_k33, sum_1);

					sum_2 = _mm256_fmadd_ps(ymm_d1, ymm_k31, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d4, ymm_k32, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_k33, sum_2);

					sum_1 = _mm256_max_ps(sum_1, sum_2);
					sum_1 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(sum_1), 120));

					_mm256_store_ps(pDst, sum_1);
					pDst += REG_SIZE;
				}
				IACA__END

				if (L & 1)
				{
					//0
					__m256 ymm_d1 = _mm256_load_ps(pSrc0 + 0);
					__m256 ymm_d2 = _mm256_load_ps(pSrc0 + REG_SIZE);

					__m256 ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					__m256 ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					__m256 sum_1 = _mm256_mul_ps(ymm_d1, ymm_k11);
					sum_1 = _mm256_fmadd_ps(ymm_d2, ymm_k12, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k13, sum_1);

					//1
					ymm_d1 = _mm256_load_ps(pSrc1 + 0);
					ymm_d2 = _mm256_load_ps(pSrc1 + REG_SIZE);

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k21, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d2, ymm_k22, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k23, sum_1);

					//2
					ymm_d1 = _mm256_load_ps(pSrc2 + 0);
					ymm_d2 = _mm256_load_ps(pSrc2 + REG_SIZE);

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k31, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d2, ymm_k32, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k33, sum_1);

					sum_1 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(sum_1), 120));

					_mm256_store_ps(pDst, sum_1);
				}
			}
		}
		void CNNPP_v2::conv_6x5(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, size_t L, size_t H, int num_threads)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { 0, 2, 1, 4, 3, 6, 5, 7 };

			if (L == 0) L = src_size_l - 4;
			if (H == 0) H = src_size_h - 5;

			const __m256i ymm_mask_temp = _mm256_load_si256((__m256i*)set1_mask);

			__m256 ymm_k_1_0[6];
			__m256 ymm_k_1_1[6];
			__m256 ymm_k_1_2[6];
			__m256 ymm_k_1_3[6];
			__m256 ymm_k_1_4[6];

			for (size_t k = 0; k < 6; ++k)
			{
				ymm_k_1_0[k] = _mm256_load_ps(kernel + (0 + 5 * k) * REG_SIZE);
				ymm_k_1_1[k] = _mm256_load_ps(kernel + (1 + 5 * k) * REG_SIZE);
				ymm_k_1_2[k] = _mm256_load_ps(kernel + (2 + 5 * k) * REG_SIZE);
				ymm_k_1_3[k] = _mm256_load_ps(kernel + (3 + 5 * k) * REG_SIZE);
				ymm_k_1_4[k] = _mm256_load_ps(kernel + (4 + 5 * k) * REG_SIZE);
			}

			__m256 ymm_k_2_0[6];
			__m256 ymm_k_2_1[6];
			__m256 ymm_k_2_2[6];
			__m256 ymm_k_2_3[6];
			__m256 ymm_k_2_4[6];

			kernel += 30 * REG_SIZE;
			for (size_t k = 0; k < 6; ++k)
			{
				ymm_k_2_0[k] = _mm256_load_ps(kernel + (0 + 5 * k) * REG_SIZE);
				ymm_k_2_1[k] = _mm256_load_ps(kernel + (1 + 5 * k) * REG_SIZE);
				ymm_k_2_2[k] = _mm256_load_ps(kernel + (2 + 5 * k) * REG_SIZE);
				ymm_k_2_3[k] = _mm256_load_ps(kernel + (3 + 5 * k) * REG_SIZE);
				ymm_k_2_4[k] = _mm256_load_ps(kernel + (4 + 5 * k) * REG_SIZE);
			}

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc = src + j * src_size_l;
				float* __restrict pDst_1 = dst + j * dst_size_l;
				float* __restrict pDst_2 = dst + j * dst_size_l + dst_size_l / 2;

				IACA__START
				for (size_t i = 0; i < L; ++i)
				{
					__m256 sum_1 = _mm256_setzero_ps();
					__m256 sum_2 = _mm256_setzero_ps();

					#pragma unroll
					for (size_t k = 0; k < 6; ++k)
					{
						float* __restrict pSrc_temp = pSrc + k * src_size_l;
						__m256 ymm_d1, ymm_d2;

						//0
						ymm_d1 = _mm256_load_ps(pSrc_temp);
						pSrc_temp += REG_SIZE;
						conv_block(k, 0);

						//1
						ymm_d1 = _mm256_load_ps(pSrc_temp);
						pSrc_temp += REG_SIZE;
						conv_block(k, 1);

						//2
						ymm_d1 = _mm256_load_ps(pSrc_temp);
						pSrc_temp += REG_SIZE;
						conv_block(k, 2);

						//3
						ymm_d1 = _mm256_load_ps(pSrc_temp);
						pSrc_temp += REG_SIZE;
						conv_block(k, 3);

						//4
						ymm_d1 = _mm256_load_ps(pSrc_temp);
						pSrc_temp += REG_SIZE;
						conv_block(k, 4);
					}
					pSrc += REG_SIZE;

					_mm256_store_ps(pDst_1, sum_1);
					pDst_1 += REG_SIZE;
					_mm256_store_ps(pDst_2, sum_2);
					pDst_2 += REG_SIZE;
				}
				IACA__END
			}
		}

#define MAX_POOL_2X2
#ifndef MAX_POOL_2X2
		void CNNPP_v2::conv_4x4_lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;

			const __m256 ymm_k11 = _mm256_load_ps(kernel + 0 * REG_SIZE);
			const __m256 ymm_k12 = _mm256_load_ps(kernel + 1 * REG_SIZE);
			const __m256 ymm_k13 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			const __m256 ymm_k14 = _mm256_load_ps(kernel + 3 * REG_SIZE);

			const __m256 ymm_k21 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			const __m256 ymm_k22 = _mm256_load_ps(kernel + 5 * REG_SIZE);
			const __m256 ymm_k23 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			const __m256 ymm_k24 = _mm256_load_ps(kernel + 7 * REG_SIZE);

			const __m256 ymm_k31 = _mm256_load_ps(kernel + 8 * REG_SIZE);
			const __m256 ymm_k32 = _mm256_load_ps(kernel + 9 * REG_SIZE);
			const __m256 ymm_k33 = _mm256_load_ps(kernel + 10 * REG_SIZE);
			const __m256 ymm_k34 = _mm256_load_ps(kernel + 11 * REG_SIZE);

			const __m256 ymm_k41 = _mm256_load_ps(kernel + 12 * REG_SIZE);
			const __m256 ymm_k42 = _mm256_load_ps(kernel + 13 * REG_SIZE);
			const __m256 ymm_k43 = _mm256_load_ps(kernel + 14 * REG_SIZE);
			const __m256 ymm_k44 = _mm256_load_ps(kernel + 15 * REG_SIZE);

			const __m256 ymm_conv_b = _mm256_load_ps(conv_b);
			const __m256 ymm_lrelu_w1 = _mm256_load_ps(lrelu_w1);
			const __m256 ymm_lrelu_w2 = _mm256_load_ps(lrelu_w2);
			const __m256 ymm_bn_b = _mm256_load_ps(bn_b);
			const __m256 ymm_zero = _mm256_setzero_ps();

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				for (size_t i = 0; i <= L - 2; i += 2)
				{
					//0
					__m256 ymm_d = _mm256_broadcast_ss(pSrc0 + 0);
					__m256 sum = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 128));

					ymm_d = _mm256_broadcast_ss(pSrc0 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k12, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k13, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k14, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 4);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 24), sum);
					pSrc0 += 2;

					//1
					ymm_d = _mm256_broadcast_ss(pSrc1 + 0);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 128), sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k22, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k23, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k24, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 4);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 24), sum);
					pSrc1 += 2;

					//2
					ymm_d = _mm256_broadcast_ss(pSrc2 + 0);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 128), sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k32, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k33, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k34, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 4);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 24), sum);
					pSrc2 += 2;

					//3
					ymm_d = _mm256_broadcast_ss(pSrc3 + 0);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 128), sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k42, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k43, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k44, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 4);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 24), sum);
					pSrc3 += 2;

					//-----------------------------

					sum = _mm256_add_ps(sum, ymm_conv_b);
					ymm_d = _mm256_max_ps(sum, ymm_zero);

					sum = _mm256_fmadd_ps(sum, ymm_lrelu_w1, ymm_bn_b);
					sum = _mm256_fmadd_ps(ymm_d, ymm_lrelu_w2, sum);

					//-----------------------------

					ymm_d = _mm256_permute2f128_ps(sum, sum, 1);
					sum = _mm256_max_ps(sum, ymm_d);

					_mm_store_ps(pDst, _mm256_extractf128_ps(sum, 0));
					pDst += REG_SIZE / 2;
				}

				if (L & 1)
				{
					//0
					__m256 ymm_d = _mm256_broadcast_ss(pSrc0 + 0);
					__m256 sum = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 128));

					ymm_d = _mm256_broadcast_ss(pSrc0 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k12, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k13, sum);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k14, sum);
					pSrc0 += 2;

					//1
					ymm_d = _mm256_broadcast_ss(pSrc1 + 0);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 128), sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k22, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k23, sum);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k24, sum);
					pSrc1 += 2;

					//2
					ymm_d = _mm256_broadcast_ss(pSrc2 + 0);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 128), sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k32, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k33, sum);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k34, sum);
					pSrc2 += 2;

					//3
					ymm_d = _mm256_broadcast_ss(pSrc3 + 0);
					sum = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 128), sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 1);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k42, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 2);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k43, sum);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 3);
					sum = _mm256_fmadd_ps(ymm_d, ymm_k44, sum);
					pSrc3 += 2;

					//-----------------------------

					sum = _mm256_add_ps(sum, ymm_conv_b);
					ymm_d = _mm256_max_ps(sum, ymm_zero);

					sum = _mm256_fmadd_ps(sum, ymm_lrelu_w1, ymm_bn_b);
					sum = _mm256_fmadd_ps(ymm_d, ymm_lrelu_w2, sum);

					//-----------------------------

					_mm_store_ps(pDst, _mm256_extractf128_ps(sum, 0));
				}
			}
		}
		void CNNPP_v2::conv_3x3_lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 2;
			if (H == 0) H = src_size_h - 2;

			__m256 ymm_k11 = _mm256_load_ps(kernel + 0 * REG_SIZE);
			__m256 ymm_k12 = _mm256_load_ps(kernel + 1 * REG_SIZE);
			__m256 ymm_k13 = _mm256_load_ps(kernel + 2 * REG_SIZE);

			__m256 ymm_k21 = _mm256_load_ps(kernel + 3 * REG_SIZE);
			__m256 ymm_k22 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			__m256 ymm_k23 = _mm256_load_ps(kernel + 5 * REG_SIZE);

			__m256 ymm_k31 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			__m256 ymm_k32 = _mm256_load_ps(kernel + 7 * REG_SIZE);
			__m256 ymm_k33 = _mm256_load_ps(kernel + 8 * REG_SIZE);

			const __m256 ymm_conv_b = _mm256_load_ps(conv_b);
			const __m256 ymm_lrelu_w1 = _mm256_load_ps(lrelu_w1);
			const __m256 ymm_lrelu_w2 = _mm256_load_ps(lrelu_w2);
			const __m256 ymm_bn_b = _mm256_load_ps(bn_b);
			const __m256 ymm_zero = _mm256_setzero_ps();

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + (j << 1) * src_size_l;
				float* __restrict pSrc0_2 = src + ((j << 1) + 1) * src_size_l;

				float* __restrict pSrc1 = src + ((j << 1) + 2) * src_size_l;
				float* __restrict pSrc1_2 = src + ((j << 1) + 3) * src_size_l;

				float* __restrict pSrc2 = src + ((j << 1) + 4) * src_size_l;
				float* __restrict pSrc2_2 = src + ((j << 1) + 5) * src_size_l;

				float* __restrict pDst = dst + j * dst_size_l;

				for (size_t i = 0; i <= L - 2; i += 2)
				{
					//0
					__m256 ymm_d1 = _mm256_load_ps(pSrc0 + 0);
					__m256 ymm_d2 = _mm256_load_ps(pSrc0 + REG_SIZE);
					pSrc0 += REG_SIZE;

					//-----------------------------

					__m256 ymm_d1_2 = _mm256_load_ps(pSrc0_2 + 0);
					__m256 ymm_d2_2 = _mm256_load_ps(pSrc0_2 + REG_SIZE);
					pSrc0_2 += REG_SIZE;

					ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d1_2);
					ymm_d2 = _mm256_max_ps(ymm_d2, ymm_d2_2);

					//-----------------------------

					__m256 ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					__m256 ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					__m256 sum_1 = _mm256_mul_ps(ymm_d3, ymm_k11);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k12, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d4, ymm_k13, sum_1);

					__m256 sum_2 = _mm256_mul_ps(ymm_d1, ymm_k11);
					sum_2 = _mm256_fmadd_ps(ymm_d4, ymm_k12, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_k13, sum_2);

					//1
					ymm_d1 = _mm256_load_ps(pSrc1 + 0);
					ymm_d2 = _mm256_load_ps(pSrc1 + REG_SIZE);
					pSrc1 += REG_SIZE;

					//-----------------------------

					ymm_d1_2 = _mm256_load_ps(pSrc1_2 + 0);
					ymm_d2_2 = _mm256_load_ps(pSrc1_2 + REG_SIZE);
					pSrc1_2 += REG_SIZE;

					ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d1_2);
					ymm_d2 = _mm256_max_ps(ymm_d2, ymm_d2_2);

					//-----------------------------

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k21, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k22, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d4, ymm_k23, sum_1);

					sum_2 = _mm256_fmadd_ps(ymm_d1, ymm_k21, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d4, ymm_k22, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_k23, sum_2);

					//2
					ymm_d1 = _mm256_load_ps(pSrc2 + 0);
					ymm_d2 = _mm256_load_ps(pSrc2 + REG_SIZE);
					pSrc2 += REG_SIZE;

					//-----------------------------

					ymm_d1_2 = _mm256_load_ps(pSrc2_2 + 0);
					ymm_d2_2 = _mm256_load_ps(pSrc2_2 + REG_SIZE);
					pSrc2_2 += REG_SIZE;

					ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d1_2);
					ymm_d2 = _mm256_max_ps(ymm_d2, ymm_d2_2);

					//-----------------------------

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k31, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k32, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d4, ymm_k33, sum_1);

					sum_2 = _mm256_fmadd_ps(ymm_d1, ymm_k31, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d4, ymm_k32, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_k33, sum_2);

					//-----------------------------

					sum_1 = _mm256_add_ps(sum_1, ymm_conv_b);
					ymm_d1 = _mm256_max_ps(sum_1, ymm_zero);

					sum_1 = _mm256_fmadd_ps(sum_1, ymm_lrelu_w1, ymm_bn_b);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_lrelu_w2, sum_1);

					sum_2 = _mm256_add_ps(sum_2, ymm_conv_b);
					ymm_d2 = _mm256_max_ps(sum_2, ymm_zero);

					sum_2 = _mm256_fmadd_ps(sum_2, ymm_lrelu_w1, ymm_bn_b);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_lrelu_w2, sum_2);

					//-----------------------------

					sum_1 = _mm256_max_ps(sum_1, sum_2);

					ymm_d2 = _mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(sum_1), _mm256_castps_pd(sum_1), 6));
					sum_1 = _mm256_permute2f128_ps(ymm_d2, ymm_d2, 1);
					sum_1 = _mm256_blend_ps(sum_1, ymm_d2, 51);

					_mm256_store_ps(pDst, sum_1);
					pDst += REG_SIZE;
				}

				if (L & 1)
				{
					//0
					__m256 ymm_d1 = _mm256_load_ps(pSrc0 + 0);
					__m256 ymm_d2 = _mm256_load_ps(pSrc0 + REG_SIZE);

					__m256 ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					__m256 ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					__m256 sum_1 = _mm256_mul_ps(ymm_d1, ymm_k11);
					sum_1 = _mm256_fmadd_ps(ymm_d2, ymm_k12, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k13, sum_1);

					//1
					ymm_d1 = _mm256_load_ps(pSrc1 + 0);
					ymm_d2 = _mm256_load_ps(pSrc1 + REG_SIZE);

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k21, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d2, ymm_k22, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k23, sum_1);

					//2
					ymm_d1 = _mm256_load_ps(pSrc2 + 0);
					ymm_d2 = _mm256_load_ps(pSrc2 + REG_SIZE);

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k31, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d2, ymm_k32, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k33, sum_1);

					//-----------------------------

					sum_1 = _mm256_add_ps(sum_1, ymm_conv_b);
					ymm_d1 = _mm256_max_ps(sum_1, ymm_zero);

					sum_1 = _mm256_fmadd_ps(sum_1, ymm_lrelu_w1, ymm_bn_b);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_lrelu_w2, sum_1);

					//-----------------------------

					ymm_d2 = _mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(sum_1), _mm256_castps_pd(sum_1), 6));
					sum_1 = _mm256_permute2f128_ps(ymm_d2, ymm_d2, 1);
					sum_1 = _mm256_blend_ps(sum_1, ymm_d2, 51);

					_mm256_store_ps(pDst, sum_1);
				}
			}
		}
#else
		void CNNPP_v2::conv_4x4_lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;
			if (H % 2 != 0) H--;

			const __m256 ymm_k11 = _mm256_load_ps(kernel + 0 * REG_SIZE);
			const __m256 ymm_k12 = _mm256_load_ps(kernel + 1 * REG_SIZE);
			const __m256 ymm_k13 = _mm256_load_ps(kernel + 2 * REG_SIZE);
			const __m256 ymm_k14 = _mm256_load_ps(kernel + 3 * REG_SIZE);

			const __m256 ymm_k21 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			const __m256 ymm_k22 = _mm256_load_ps(kernel + 5 * REG_SIZE);
			const __m256 ymm_k23 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			const __m256 ymm_k24 = _mm256_load_ps(kernel + 7 * REG_SIZE);

			const __m256 ymm_k31 = _mm256_load_ps(kernel + 8 * REG_SIZE);
			const __m256 ymm_k32 = _mm256_load_ps(kernel + 9 * REG_SIZE);
			const __m256 ymm_k33 = _mm256_load_ps(kernel + 10 * REG_SIZE);
			const __m256 ymm_k34 = _mm256_load_ps(kernel + 11 * REG_SIZE);

			const __m256 ymm_k41 = _mm256_load_ps(kernel + 12 * REG_SIZE);
			const __m256 ymm_k42 = _mm256_load_ps(kernel + 13 * REG_SIZE);
			const __m256 ymm_k43 = _mm256_load_ps(kernel + 14 * REG_SIZE);
			const __m256 ymm_k44 = _mm256_load_ps(kernel + 15 * REG_SIZE);

			const __m256 ymm_conv_b = _mm256_load_ps(conv_b);
			const __m256 ymm_lrelu_w1 = _mm256_load_ps(lrelu_w1);
			const __m256 ymm_lrelu_w2 = _mm256_load_ps(lrelu_w2);
			const __m256 ymm_bn_b = _mm256_load_ps(bn_b);
			const __m256 ymm_zero = _mm256_setzero_ps();

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pSrc4 = src + (j + 4) * src_size_l;
				float* __restrict pDst = dst + (j >> 1) * dst_size_l;

				IACA__START
				for (size_t i = 0; i <= L - 2; i += 2)
				{
					//0
					__m256 ymm_d = _mm256_broadcast_ss(pSrc0 + 0);
					__m256 sum_1 = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 128));

					ymm_d = _mm256_broadcast_ss(pSrc0 + 1);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k12, sum_1);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 2);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k13, sum_1);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 3);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k14, sum_1);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 4);
					sum_1 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 24), sum_1);
					pSrc0 += 2;

					//1
					ymm_d = _mm256_broadcast_ss(pSrc1 + 0);
					sum_1 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 128), sum_1);
					__m256 sum_2 = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 128));

					ymm_d = _mm256_broadcast_ss(pSrc1 + 1);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k22, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k12, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 2);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k23, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k13, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 3);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k24, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k14, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 4);
					sum_1 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 24), sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 24), sum_2);
					pSrc1 += 2;

					//2
					ymm_d = _mm256_broadcast_ss(pSrc2 + 0);
					sum_1 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 128), sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 128), sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 1);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k32, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k22, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 2);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k33, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k23, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 3);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k34, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k24, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 4);
					sum_1 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 24), sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 24), sum_2);
					pSrc2 += 2;

					//3
					ymm_d = _mm256_broadcast_ss(pSrc3 + 0);
					sum_1 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 128), sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 128), sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 1);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k42, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k32, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 2);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k43, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k33, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 3);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k44, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k34, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 4);
					sum_1 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 24), sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 24), sum_2);
					pSrc3 += 2;

					//4
					ymm_d = _mm256_broadcast_ss(pSrc4 + 0);
					sum_2 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 128), sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc4 + 1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k42, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc4 + 2);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k43, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc4 + 3);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k44, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc4 + 4);
					sum_2 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 24), sum_2);
					pSrc4 += 2;

					//-----------------------------

					sum_1 = _mm256_add_ps(sum_1, ymm_conv_b);
					ymm_d = _mm256_max_ps(sum_1, ymm_zero);

					sum_1 = _mm256_fmadd_ps(sum_1, ymm_lrelu_w1, ymm_bn_b);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_lrelu_w2, sum_1);

					sum_2 = _mm256_add_ps(sum_2, ymm_conv_b);
					ymm_d = _mm256_max_ps(sum_2, ymm_zero);

					sum_2 = _mm256_fmadd_ps(sum_2, ymm_lrelu_w1, ymm_bn_b);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_lrelu_w2, sum_2);

					//-----------------------------

					sum_1 = _mm256_max_ps(sum_1, sum_2);

					ymm_d = _mm256_permute2f128_ps(sum_1, sum_1, 1);
					sum_1 = _mm256_max_ps(sum_1, ymm_d);

#ifndef USE_HF
					_mm_store_ps(pDst, _mm256_extractf128_ps(sum_1, 0));
					pDst += REG_SIZE / 2;
#else					
					//_mm_storel_epi64((__m128i*)pDst, _mm256_cvtps_ph(sum_1, 0));
					_mm_storeu_si128((__m128i*)pDst, _mm256_cvtps_ph(sum_1, 0));
					pDst += REG_SIZE / 4;
#endif
				}
				IACA__END

				if (L & 1)
				{
					//0
					__m256 ymm_d = _mm256_broadcast_ss(pSrc0 + 0);
					__m256 sum_1 = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 128));

					ymm_d = _mm256_broadcast_ss(pSrc0 + 1);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k12, sum_1);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 2);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k13, sum_1);

					ymm_d = _mm256_broadcast_ss(pSrc0 + 3);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k14, sum_1);
					pSrc0 += 2;

					//1
					ymm_d = _mm256_broadcast_ss(pSrc1 + 0);
					sum_1 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 128), sum_1);
					__m256 sum_2 = _mm256_mul_ps(ymm_d, _mm256_permute2f128_ps(ymm_k11, ymm_k11, 128));

					ymm_d = _mm256_broadcast_ss(pSrc1 + 1);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k22, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k12, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 2);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k23, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k13, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc1 + 3);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k24, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k14, sum_2);
					pSrc1 += 2;

					//2
					ymm_d = _mm256_broadcast_ss(pSrc2 + 0);
					sum_1 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 128), sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k21, ymm_k21, 128), sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 1);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k32, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k22, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 2);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k33, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k23, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc2 + 3);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k34, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k24, sum_2);
					pSrc2 += 2;

					//3
					ymm_d = _mm256_broadcast_ss(pSrc3 + 0);
					sum_1 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 128), sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k31, ymm_k31, 128), sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 1);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k42, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k32, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 2);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k43, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k33, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc3 + 3);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_k44, sum_1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k34, sum_2);
					pSrc3 += 2;

					//4
					ymm_d = _mm256_broadcast_ss(pSrc4 + 0);
					sum_2 = _mm256_fmadd_ps(ymm_d, _mm256_permute2f128_ps(ymm_k41, ymm_k41, 128), sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc4 + 1);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k42, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc4 + 2);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k43, sum_2);

					ymm_d = _mm256_broadcast_ss(pSrc4 + 3);
					sum_2 = _mm256_fmadd_ps(ymm_d, ymm_k44, sum_2);
					pSrc3 += 2;

					//-----------------------------

					sum_1 = _mm256_add_ps(sum_1, ymm_conv_b);
					ymm_d = _mm256_max_ps(sum_1, ymm_zero);

					sum_1 = _mm256_fmadd_ps(sum_1, ymm_lrelu_w1, ymm_bn_b);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_lrelu_w2, sum_1);

					sum_2 = _mm256_add_ps(sum_2, ymm_conv_b);
					ymm_d = _mm256_max_ps(sum_2, ymm_zero);

					sum_2 = _mm256_fmadd_ps(sum_2, ymm_lrelu_w1, ymm_bn_b);
					sum_1 = _mm256_fmadd_ps(ymm_d, ymm_lrelu_w2, sum_2);

					//-----------------------------

					sum_1 = _mm256_max_ps(sum_1, sum_2);

#ifndef USE_HF
					_mm_store_ps(pDst, _mm256_extractf128_ps(sum_1, 0));
#else
					//_mm_storel_epi64((__m128i*)pDst, _mm256_cvtps_ph(sum_1, 0));
					_mm_storeu_si128((__m128i*)pDst, _mm256_cvtps_ph(sum_1, 0));
#endif
				}
			}
		}
		void CNNPP_v2::conv_3x3_lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 2;
			if (H == 0) H = src_size_h - 2;

			__m256 ymm_k11 = _mm256_load_ps(kernel + 0 * REG_SIZE);
			__m256 ymm_k12 = _mm256_load_ps(kernel + 1 * REG_SIZE);
			__m256 ymm_k13 = _mm256_load_ps(kernel + 2 * REG_SIZE);

			__m256 ymm_k21 = _mm256_load_ps(kernel + 3 * REG_SIZE);
			__m256 ymm_k22 = _mm256_load_ps(kernel + 4 * REG_SIZE);
			__m256 ymm_k23 = _mm256_load_ps(kernel + 5 * REG_SIZE);

			__m256 ymm_k31 = _mm256_load_ps(kernel + 6 * REG_SIZE);
			__m256 ymm_k32 = _mm256_load_ps(kernel + 7 * REG_SIZE);
			__m256 ymm_k33 = _mm256_load_ps(kernel + 8 * REG_SIZE);

			const __m256 ymm_conv_b = _mm256_load_ps(conv_b);
			const __m256 ymm_lrelu_w1 = _mm256_load_ps(lrelu_w1);
			const __m256 ymm_lrelu_w2 = _mm256_load_ps(lrelu_w2);
			const __m256 ymm_bn_b = _mm256_load_ps(bn_b);
			const __m256 ymm_zero = _mm256_setzero_ps();

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;

				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i <= L - 2; i += 2)
				{
					//0
#ifndef USE_HF
					__m256 ymm_d1 = _mm256_load_ps(pSrc0 + 0);
					__m256 ymm_d2 = _mm256_load_ps(pSrc0 + REG_SIZE);
					pSrc0 += REG_SIZE;
#else
					__m256i load_si = _mm256_loadu_si256((__m256i*)pSrc0);
					__m256 ymm_d1 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 0));
					__m256 ymm_d2 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 1));
					pSrc0 += REG_SIZE / 2;
#endif

					__m256 ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					__m256 ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					__m256 sum_1 = _mm256_mul_ps(ymm_d3, ymm_k11);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k12, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d4, ymm_k13, sum_1);

					__m256 sum_2 = _mm256_mul_ps(ymm_d1, ymm_k11);
					sum_2 = _mm256_fmadd_ps(ymm_d4, ymm_k12, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_k13, sum_2);

					//1
#ifndef USE_HF
					ymm_d1 = _mm256_load_ps(pSrc1 + 0);
					ymm_d2 = _mm256_load_ps(pSrc1 + REG_SIZE);
					pSrc1 += REG_SIZE;
#else
					load_si = _mm256_loadu_si256((__m256i*)pSrc1);
					ymm_d1 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 0));
					ymm_d2 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 1));
					pSrc1 += REG_SIZE / 2;
#endif

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k21, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k22, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d4, ymm_k23, sum_1);

					sum_2 = _mm256_fmadd_ps(ymm_d1, ymm_k21, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d4, ymm_k22, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_k23, sum_2);

					//2
#ifndef USE_HF
					ymm_d1 = _mm256_load_ps(pSrc2 + 0);
					ymm_d2 = _mm256_load_ps(pSrc2 + REG_SIZE);
					pSrc2 += REG_SIZE;
#else
					load_si = _mm256_loadu_si256((__m256i*)pSrc2);
					ymm_d1 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 0));
					ymm_d2 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 1));
					pSrc2 += REG_SIZE / 2;
#endif

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 250);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k31, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k32, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d4, ymm_k33, sum_1);

					sum_2 = _mm256_fmadd_ps(ymm_d1, ymm_k31, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d4, ymm_k32, sum_2);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_k33, sum_2);

					//-----------------------------

					sum_1 = _mm256_add_ps(sum_1, ymm_conv_b);
					ymm_d1 = _mm256_max_ps(sum_1, ymm_zero);

					sum_1 = _mm256_fmadd_ps(sum_1, ymm_lrelu_w1, ymm_bn_b);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_lrelu_w2, sum_1);

					sum_2 = _mm256_add_ps(sum_2, ymm_conv_b);
					ymm_d2 = _mm256_max_ps(sum_2, ymm_zero);

					sum_2 = _mm256_fmadd_ps(sum_2, ymm_lrelu_w1, ymm_bn_b);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_lrelu_w2, sum_2);

					//-----------------------------

					sum_1 = _mm256_max_ps(sum_1, sum_2);

					ymm_d2 = _mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(sum_1), _mm256_castps_pd(sum_1), 6));
					sum_1 = _mm256_permute2f128_ps(ymm_d2, ymm_d2, 1);
					sum_1 = _mm256_blend_ps(sum_1, ymm_d2, 51);

#ifndef USE_HF
					_mm256_store_ps(pDst, sum_1);
					pDst += REG_SIZE;
#else
					_mm_store_si128((__m128i*)pDst, _mm256_cvtps_ph(sum_1, 0));
					pDst += REG_SIZE / 2;
#endif
				}
				IACA__END

				if (L & 1)
				{
					//0
#ifndef USE_HF
					__m256 ymm_d1 = _mm256_load_ps(pSrc0 + 0);
					__m256 ymm_d2 = _mm256_load_ps(pSrc0 + REG_SIZE);
#else
					__m256i load_si = _mm256_loadu_si256((__m256i*)(pSrc0 + 0));
					__m256 ymm_d1 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 0));
					__m256 ymm_d2 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 1));
#endif

					__m256 ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					__m256 ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					__m256 sum_1 = _mm256_mul_ps(ymm_d1, ymm_k11);
					sum_1 = _mm256_fmadd_ps(ymm_d2, ymm_k12, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k13, sum_1);

					//1
#ifndef USE_HF
					ymm_d1 = _mm256_load_ps(pSrc1 + 0);
					ymm_d2 = _mm256_load_ps(pSrc1 + REG_SIZE);
#else
					load_si = _mm256_loadu_si256((__m256i*)(pSrc1 + 0));
					ymm_d1 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 0));
					ymm_d2 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 1));
#endif

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k21, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d2, ymm_k22, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k23, sum_1);

					//2
#ifndef USE_HF
					ymm_d1 = _mm256_load_ps(pSrc2 + 0);
					ymm_d2 = _mm256_load_ps(pSrc2 + REG_SIZE);
#else
					load_si = _mm256_loadu_si256((__m256i*)(pSrc2 + 0));
					ymm_d1 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 0));
					ymm_d2 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 1));
#endif

					ymm_d3 = _mm256_hadd_ps(ymm_d1, ymm_d2);

					ymm_d4 = _mm256_shuffle_ps(ymm_d1, ymm_d2, 153);
					ymm_d4 = _mm256_shuffle_ps(ymm_d4, ymm_d4, 177);
					ymm_d4 = _mm256_add_ps(ymm_d3, ymm_d4);

					ymm_d1 = _mm256_permute2f128_ps(ymm_d3, ymm_d4, 32);
					ymm_d2 = _mm256_permute2f128_ps(ymm_d4, ymm_d3, 19);

					ymm_d3 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 80);
					ymm_d1 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 80);
					ymm_d2 = _mm256_shuffle_ps(ymm_d2, ymm_d2, 250);

					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k31, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d2, ymm_k32, sum_1);
					sum_1 = _mm256_fmadd_ps(ymm_d3, ymm_k33, sum_1);

					//-----------------------------

					sum_1 = _mm256_add_ps(sum_1, ymm_conv_b);
					ymm_d1 = _mm256_max_ps(sum_1, ymm_zero);

					sum_1 = _mm256_fmadd_ps(sum_1, ymm_lrelu_w1, ymm_bn_b);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_lrelu_w2, sum_1);

					//-----------------------------

					ymm_d2 = _mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(sum_1), _mm256_castps_pd(sum_1), 6));
					sum_1 = _mm256_permute2f128_ps(ymm_d2, ymm_d2, 1);
					sum_1 = _mm256_blend_ps(sum_1, ymm_d2, 51);

#ifndef USE_HF
					_mm256_store_ps(pDst, sum_1);
#else
					_mm_store_si128((__m128i*)pDst, _mm256_cvtps_ph(sum_1, 0));
#endif
				}
			}
		}
#endif

		void CNNPP_v2::conv_5x4_lrelu_bn(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { 0, 2, 1, 4, 3, 6, 5, 7 };

			if (L == 0) L = src_size_l - 4;
			if (H == 0) H = src_size_h - 5;

			const __m256i ymm_mask_temp = _mm256_load_si256((__m256i*)set1_mask);

			__m256 ymm_k_1_0[5];
			__m256 ymm_k_1_1[5];
			__m256 ymm_k_1_2[5];
			__m256 ymm_k_1_3[5];

			for (size_t k = 0; k < 5; ++k)
			{
				ymm_k_1_0[k] = _mm256_load_ps(kernel + (0 + 4 * k) * REG_SIZE);
				ymm_k_1_1[k] = _mm256_load_ps(kernel + (1 + 4 * k) * REG_SIZE);
				ymm_k_1_2[k] = _mm256_load_ps(kernel + (2 + 4 * k) * REG_SIZE);
				ymm_k_1_3[k] = _mm256_load_ps(kernel + (3 + 4 * k) * REG_SIZE);
			}

			__m256 ymm_k_2_0[5];
			__m256 ymm_k_2_1[5];
			__m256 ymm_k_2_2[5];
			__m256 ymm_k_2_3[5];

			kernel += 5 * 4 * REG_SIZE;
			for (size_t k = 0; k < 5; ++k)
			{
				ymm_k_2_0[k] = _mm256_load_ps(kernel + (0 + 4 * k) * REG_SIZE);
				ymm_k_2_1[k] = _mm256_load_ps(kernel + (1 + 4 * k) * REG_SIZE);
				ymm_k_2_2[k] = _mm256_load_ps(kernel + (2 + 4 * k) * REG_SIZE);
				ymm_k_2_3[k] = _mm256_load_ps(kernel + (3 + 4 * k) * REG_SIZE);
			}

			const __m256 ymm_conv_b_1 = _mm256_load_ps(conv_b);
			const __m256 ymm_conv_b_2 = _mm256_load_ps(conv_b + REG_SIZE);
			const __m256 ymm_lrelu_w1_1 = _mm256_load_ps(lrelu_w1);
			const __m256 ymm_lrelu_w1_2 = _mm256_load_ps(lrelu_w1 + REG_SIZE);
			const __m256 ymm_lrelu_w2_1 = _mm256_load_ps(lrelu_w2);
			const __m256 ymm_lrelu_w2_2 = _mm256_load_ps(lrelu_w2 + REG_SIZE);
			const __m256 ymm_bn_b_1 = _mm256_load_ps(bn_b);
			const __m256 ymm_bn_b_2 = _mm256_load_ps(bn_b + REG_SIZE);
			const __m256 ymm_zero = _mm256_setzero_ps();

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc = src + (j << 1) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i < L; ++i)
				{
					__m256 sum_1 = _mm256_setzero_ps();
					__m256 sum_2 = _mm256_setzero_ps();

					#pragma unroll
					for (size_t k = 0; k < 5; ++k)
					{
						float* __restrict pSrc_temp_1 = pSrc + (k << 1) * src_size_l;
						float* __restrict pSrc_temp_2 = pSrc + ((k << 1) + 1) * src_size_l;
						__m256 ymm_d1, ymm_d2, ymm_ml, ymm_shf;

						//0
#ifndef USE_HF
						ymm_d1 = _mm256_load_ps(pSrc_temp_1);
						ymm_d2 = _mm256_load_ps(pSrc_temp_2);
						pSrc_temp_1 += REG_SIZE;
						pSrc_temp_2 += REG_SIZE;
#else
						ymm_d1 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)pSrc_temp_1));
						ymm_d2 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)pSrc_temp_2));
						pSrc_temp_1 += REG_SIZE / 2;
						pSrc_temp_2 += REG_SIZE / 2;
#endif
						ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d2);
						conv_block(k, 0);

						//1
#ifndef USE_HF
						ymm_d1 = _mm256_load_ps(pSrc_temp_1);
						ymm_d2 = _mm256_load_ps(pSrc_temp_2);
						pSrc_temp_1 += REG_SIZE;
						pSrc_temp_2 += REG_SIZE;
#else
						ymm_d1 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)pSrc_temp_1));
						ymm_d2 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)pSrc_temp_2));
						pSrc_temp_1 += REG_SIZE / 2;
						pSrc_temp_2 += REG_SIZE / 2;
#endif
						ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d2);
						conv_block(k, 1);

						//2
#ifndef USE_HF
						ymm_d1 = _mm256_load_ps(pSrc_temp_1);
						ymm_d2 = _mm256_load_ps(pSrc_temp_2);
						pSrc_temp_1 += REG_SIZE;
						pSrc_temp_2 += REG_SIZE;
#else
						ymm_d1 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)pSrc_temp_1));
						ymm_d2 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)pSrc_temp_2));
						pSrc_temp_1 += REG_SIZE / 2;
						pSrc_temp_2 += REG_SIZE / 2;
#endif
						ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d2);
						conv_block(k, 2);

						//3
#ifndef USE_HF
						ymm_d1 = _mm256_load_ps(pSrc_temp_1);
						ymm_d2 = _mm256_load_ps(pSrc_temp_2);
						pSrc_temp_1 += REG_SIZE;
						pSrc_temp_2 += REG_SIZE;
#else
						ymm_d1 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)pSrc_temp_1));
						ymm_d2 = _mm256_cvtph_ps(_mm_load_si128((__m128i*)pSrc_temp_2));
						pSrc_temp_1 += REG_SIZE / 2;
						pSrc_temp_2 += REG_SIZE / 2;
#endif
						ymm_d1 = _mm256_max_ps(ymm_d1, ymm_d2);
						conv_block(k, 3);
					}
#ifndef USE_HF
					pSrc += REG_SIZE;
#else
					pSrc += REG_SIZE / 2;
#endif

					//-----------------------------

					sum_1 = _mm256_add_ps(sum_1, ymm_conv_b_1);
					__m256 ymm_d1 = _mm256_max_ps(sum_1, ymm_zero);

					sum_1 = _mm256_fmadd_ps(sum_1, ymm_lrelu_w1_1, ymm_bn_b_1);
					sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_lrelu_w2_1, sum_1);

					sum_2 = _mm256_add_ps(sum_2, ymm_conv_b_2);
					__m256 ymm_d2 = _mm256_max_ps(sum_2, ymm_zero);

					sum_2 = _mm256_fmadd_ps(sum_2, ymm_lrelu_w1_2, ymm_bn_b_2);
					sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_lrelu_w2_2, sum_2);

					//-----------------------------

#ifndef USE_HF
					_mm256_store_ps(pDst, sum_1);
					_mm256_store_ps(pDst + REG_SIZE, sum_2);
					pDst += 2 * REG_SIZE;
#else
					__m256i store_si256 = _mm256_set_m128i(_mm256_cvtps_ph(sum_2, 0), _mm256_cvtps_ph(sum_1, 0));
					_mm256_storeu_si256((__m256i*)pDst, store_si256);
					pDst += REG_SIZE;
#endif
				}
				IACA__END
			}
		}

#endif

		void CNNPP_v2::max_tanh_tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale, int num_threads)
		{
			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm7 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm1 = _mm256_load_ps(conv_b);
			const __m256 ymm5 = _mm256_load_ps(subs_w);
			const __m256 ymm4 = _mm256_load_ps(subs_b);
			const __m256 ymm11 = _mm256_broadcast_ss(scale);

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < src_size_h; j += 2)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pDst = dst + (j / 2) * dst_size_l;

				IACA__START
				for (size_t i = 0; i < src_size_l; i += REG_SIZE)
				{
					__m256 ymm3 = _mm256_load_ps(pSrc0 + i);
					__m256 ymm12 = _mm256_load_ps(pSrc1 + i);

					//-----------------------------

					__m256 ymm0 = _mm256_max_ps(ymm3, ymm12);

					//-----------------------------

					ymm0 = _mm256_add_ps(ymm0, ymm1);
					ymm3 = _mm256_and_ps(ymm14, ymm0);

					ymm3 = _mm256_add_ps(ymm15, ymm3);
					__m256 ymm6 = _mm256_mul_ps(ymm0, ymm0);

					ymm3 = _mm256_add_ps(ymm3, ymm6);
					__m256 ymm8 = _mm256_mul_ps(ymm6, ymm6);

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
				IACA__END
			}
		}
		void CNNPP_v2::tanh_tanh_2tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict conv_b, float* __restrict subs_w, float* __restrict subs_b, float* __restrict scale, float* __restrict snn_hl_w0, float* __restrict snn_hl_b0, float* __restrict snn_hl_w1, float* __restrict snn_hl_b1, float* __restrict snn_ol_w0, float* __restrict snn_ol_w1, int num_threads)
		{
			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm13 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm12 = _mm256_load_ps(conv_b);
			const __m256 ymm11 = _mm256_load_ps(subs_w);
			const __m256 ymm10 = _mm256_load_ps(subs_b);
			const __m256 ymm9 = _mm256_broadcast_ss(scale);

			const __m256 buff_0 = _mm256_mul_ps(_mm256_load_ps(snn_hl_w0), ymm9);
			const __m256 buff_1 = _mm256_load_ps(snn_hl_b0);
			const __m256 buff_2 = _mm256_mul_ps(_mm256_load_ps(snn_hl_w1), ymm9);
			const __m256 buff_3 = _mm256_load_ps(snn_hl_b1);
			const __m256 buff_4 = _mm256_mul_ps(_mm256_load_ps(snn_ol_w0), ymm9);
			const __m256 buff_5 = _mm256_mul_ps(_mm256_load_ps(snn_ol_w1), ymm9);

			const size_t L = src_size_l >> 1; // div on 2
			
			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < src_size_h; ++j)
			{
				float* __restrict pSrc = src + j * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i < L; i += REG_SIZE)
				{
					__m256 ymm0 = _mm256_load_ps(pSrc);
					pSrc += REG_SIZE;

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
					__m256 ymm6 = _mm256_or_ps(ymm3, ymm2);

#ifdef USE_FMA
					ymm0 = _mm256_fmadd_ps(ymm6, buff_0, buff_1);
#else
					ymm0 = _mm256_mul_ps(ymm6, buff_0);
					ymm0 = _mm256_add_ps(ymm0, buff_1);
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
					__m256 ymm7 = _mm256_or_ps(ymm3, ymm2);

#ifdef USE_FMA
					ymm0 = _mm256_fmadd_ps(ymm6, buff_2, buff_3);
#else
					ymm0 = _mm256_mul_ps(ymm6, buff_2);
					ymm0 = _mm256_add_ps(ymm0, buff_3);
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
					__m256 ymm8 = _mm256_or_ps(ymm3, ymm2);

					ymm7 = _mm256_mul_ps(ymm7, buff_4);

#ifdef USE_FMA
					ymm8 = _mm256_fmadd_ps(ymm8, buff_5, ymm7);
#else
					ymm8 = _mm256_mul_ps(ymm8, buff_5);
					ymm8 = _mm256_add_ps(ymm8, ymm7);
#endif

					ymm7 = _mm256_permute2f128_ps(ymm8, ymm8, 19);
					ymm7 = _mm256_add_ps(ymm8, ymm7);
					
					ymm8 = _mm256_permute_ps(ymm7, 14);
					ymm7 = _mm256_add_ps(ymm7, ymm8);
					ymm8 = _mm256_permute_ps(ymm7, 1);
					ymm7 = _mm256_add_ps(ymm7, ymm8);

					_mm_store_ss(pDst, _mm256_extractf128_ps(ymm7, 0));
					pDst += 2;
				}
				IACA__END
			}
		}
		void CNNPP_v2::tanh(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict snn_ol_b, float* __restrict scale, int num_threads)
		{
			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm13 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm12 = _mm256_broadcast_ss(snn_ol_b);
			const __m256 ymm11 = _mm256_broadcast_ss(scale);

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < src_size_h; ++j)
			{
				float* __restrict pSrc = src + j * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i < src_size_l; i += 2 * REG_SIZE)
				{
					__m256 ymm0_0 = _mm256_load_ps(pSrc);
					__m256 ymm0_1 = _mm256_load_ps(pSrc + REG_SIZE);
					pSrc += 2 * REG_SIZE;

					__m256 ymm0 = _mm256_hadd_ps(ymm0_0, ymm0_1);

#ifdef USE_FMA
					ymm0 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(ymm0), 216));
#else
					ymm0_0 = _mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(ymm0), _mm256_castps_pd(ymm0), 5));
					ymm0_0 = _mm256_permute2f128_ps(ymm0_0, ymm0_0, 1);
					ymm0 = _mm256_blend_ps(ymm0, ymm0_0, 60);
#endif

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

					_mm256_store_ps(pDst, ymm3);
					pDst += REG_SIZE;
				}
				IACA__END
			}
		}

		void CNNPP_v2::mulCN_add_tanhW_add(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float** __restrict snn_hl_w, float** __restrict snn_hl_b, float* __restrict snn_tanh_w, float* __restrict snn_bn_w, float* __restrict snn_bn_b, float** __restrict snn_ol_w, size_t L, size_t H, int num_threads)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { 1, 2, 3, 4, 5, 6, 7, 0 };
			const __m256i ymm_mask_temp = _mm256_load_si256((__m256i*)set1_mask);

			__m256 ymm_hl_w[4][8];
			for (size_t k = 0; k < 4; ++k)
			{
				for (size_t i = 0; i < 8; ++i)
				{
					ymm_hl_w[k][i] = _mm256_load_ps(snn_hl_w[k] + i * REG_SIZE);
				}
			}

			__m256 ymm_hl_b[4][2];
			for (size_t k = 0; k < 4; ++k)
			{
				for (size_t i = 0; i < 2; ++i)
				{
					ymm_hl_b[k][i] = _mm256_load_ps(snn_hl_b[k] + i * REG_SIZE);
				}
			}

			__m256 ymm_tanh_w[2];
			for (size_t i = 0; i < 2; ++i)
			{
				ymm_tanh_w[i] = _mm256_load_ps(snn_tanh_w + i * REG_SIZE);
			}

			__m256 ymm_bn_w[4];
			for (size_t i = 0; i < 4; ++i)
			{
				ymm_bn_w[i] = _mm256_broadcast_ss(&snn_bn_w[i]);
			}

			__m256 ymm_bn_b[4];
			for (size_t i = 0; i < 4; ++i)
			{
				ymm_bn_b[i] = _mm256_broadcast_ss(&snn_bn_b[i]);
			}

			__m256 ymm_ol_w[4][2];
			for (size_t k = 0; k < 4; ++k)
			{
				for (size_t i = 0; i < 2; ++i)
				{
					ymm_ol_w[k][i] = _mm256_load_ps(snn_ol_w[k] + i * REG_SIZE);
				}
			}

			const float scale = 0.5f;
			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm13 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm_scale = _mm256_broadcast_ss(&scale);

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc = src + j * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i < L; ++i)
				{
#ifndef USE_HF
					__m256 ymm_0 = _mm256_load_ps(pSrc);
					__m256 ymm_1 = _mm256_load_ps(pSrc + REG_SIZE);
					pSrc += 2 * REG_SIZE;
#else
					__m256i load_si = _mm256_loadu_si256((__m256i*)pSrc);
					__m256 ymm_0 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 0));
					__m256 ymm_1 = _mm256_cvtph_ps(_mm256_extracti128_si256(load_si, 1));
					pSrc += REG_SIZE;
#endif

					__m256 ymm_sum = _mm256_xor_ps(ymm_0, ymm_0);
					for (size_t k = 0; k < 4; ++k)
					{
#ifdef USE_FMA
						__m256 ymm_0_w, ymm_1_w;
						__m256 ymm_s0 = _mm256_fmadd_ps(ymm_0, ymm_hl_w[k][0], _mm256_mul_ps(ymm_1, ymm_hl_w[k][1]));

						__m256 ymm_0_shf = _mm256_permutevar8x32_ps(ymm_0, ymm_mask_temp);
						__m256 ymm_s1 = _mm256_fmadd_ps(ymm_0_shf, ymm_hl_w[k][3], _mm256_mul_ps(ymm_1, ymm_hl_w[k][2]));
#else
						__m256 ymm_0_w = _mm256_mul_ps(ymm_0, ymm_hl_w[k][0]);
						__m256 ymm_1_w = _mm256_mul_ps(ymm_1, ymm_hl_w[k][1]);
						__m256 ymm_s0 = _mm256_add_ps(ymm_0_w, ymm_1_w);

						__m256 ymm_0_shf = _mm256_shuffle_ps(ymm_0, ymm_0, 57);
						ymm_0_shf = _mm256_blend_ps(ymm_0_shf, _mm256_permute2f128_ps(ymm_0_shf, ymm_0_shf, 1), 136);

						ymm_0_w = _mm256_mul_ps(ymm_0_shf, ymm_hl_w[k][3]);
						ymm_1_w = _mm256_mul_ps(ymm_1, ymm_hl_w[k][2]);
						__m256 ymm_s1 = _mm256_add_ps(ymm_0_w, ymm_1_w);
#endif

						ymm_s0 = _mm256_hadd_ps(ymm_s0, ymm_s1);

#ifdef USE_FMA
						__m256 ymm_1_shf = _mm256_permutevar8x32_ps(ymm_1, ymm_mask_temp);
						ymm_s1 = _mm256_fmadd_ps(ymm_0_shf, ymm_hl_w[k][4], _mm256_mul_ps(ymm_1_shf, ymm_hl_w[k][5]));

						ymm_0_shf = _mm256_permutevar8x32_ps(ymm_0_shf, ymm_mask_temp);
						ymm_0_shf = _mm256_fmadd_ps(ymm_0_shf, ymm_hl_w[k][7], _mm256_mul_ps(ymm_1_shf, ymm_hl_w[k][6]));
#else
						__m256 ymm_1_shf = _mm256_shuffle_ps(ymm_1, ymm_1, 57);
						ymm_1_shf = _mm256_blend_ps(ymm_1_shf, _mm256_permute2f128_ps(ymm_1_shf, ymm_1_shf, 1), 136);

						ymm_0_w = _mm256_mul_ps(ymm_0_shf, ymm_hl_w[k][4]);
						ymm_1_w = _mm256_mul_ps(ymm_1_shf, ymm_hl_w[k][5]);
						ymm_s1 = _mm256_add_ps(ymm_0_w, ymm_1_w);

						ymm_0_shf = _mm256_shuffle_ps(ymm_0_shf, ymm_0_shf, 57);
						ymm_0_shf = _mm256_blend_ps(ymm_0_shf, _mm256_permute2f128_ps(ymm_0_shf, ymm_0_shf, 1), 136);

						ymm_0_w = _mm256_mul_ps(ymm_0_shf, ymm_hl_w[k][7]);
						ymm_1_w = _mm256_mul_ps(ymm_1_shf, ymm_hl_w[k][6]);
						ymm_0_shf = _mm256_add_ps(ymm_0_w, ymm_1_w);
#endif

						ymm_s1 = _mm256_hadd_ps(ymm_s1, ymm_0_shf);

						ymm_s0 = _mm256_add_ps(ymm_s0, ymm_hl_b[k][0]);
						ymm_s1 = _mm256_add_ps(ymm_s1, ymm_hl_b[k][1]);

						ymm_s0 = _mm256_mul_ps(ymm_s0, ymm_tanh_w[0]);
						ymm_s1 = _mm256_mul_ps(ymm_s1, ymm_tanh_w[1]);

						//------------------------------

						ymm_0_w = _mm256_and_ps(ymm14, ymm_s0);
						ymm_1_w = _mm256_add_ps(ymm15, ymm_0_w);
						ymm_0_shf = _mm256_mul_ps(ymm_s0, ymm_s0);

						ymm_1_w = _mm256_add_ps(ymm_1_w, ymm_0_shf);
						ymm_0_shf = _mm256_mul_ps(ymm_0_shf, ymm_0_shf);

#ifdef USE_FMA
						ymm_1_w = _mm256_fmadd_ps(ymm_0_shf, ymm13, ymm_1_w);
#else	
						ymm_0_shf = _mm256_mul_ps(ymm_0_shf, ymm13);
						ymm_1_w = _mm256_add_ps(ymm_1_w, ymm_0_shf);
#endif

						ymm_0_shf = _mm256_andnot_ps(ymm14, ymm_s0);

#ifdef USE_FAST_DIV
						ymm_1_w = _mm256_rcp_ps(ymm_1_w);
#else
						ymm_1_w = _mm256_div_ps(ymm15, ymm_1_w);
#endif	

						ymm_1_w = _mm256_sub_ps(ymm15, ymm_1_w);
						ymm_s0 = _mm256_or_ps(ymm_1_w, ymm_0_shf);

						//------------------------------

						ymm_0_w = _mm256_and_ps(ymm14, ymm_s1);
						ymm_1_w = _mm256_add_ps(ymm15, ymm_0_w);
						ymm_0_shf = _mm256_mul_ps(ymm_s1, ymm_s1);

						ymm_1_w = _mm256_add_ps(ymm_1_w, ymm_0_shf);
						ymm_0_shf = _mm256_mul_ps(ymm_0_shf, ymm_0_shf);

#ifdef USE_FMA
						ymm_1_w = _mm256_fmadd_ps(ymm_0_shf, ymm13, ymm_1_w);
#else	
						ymm_0_shf = _mm256_mul_ps(ymm_0_shf, ymm13);
						ymm_1_w = _mm256_add_ps(ymm_1_w, ymm_0_shf);
#endif

						ymm_0_shf = _mm256_andnot_ps(ymm14, ymm_s1);

#ifdef USE_FAST_DIV
						ymm_1_w = _mm256_rcp_ps(ymm_1_w);
#else
						ymm_1_w = _mm256_div_ps(ymm15, ymm_1_w);
#endif	

						ymm_1_w = _mm256_sub_ps(ymm15, ymm_1_w);
						ymm_s1 = _mm256_or_ps(ymm_1_w, ymm_0_shf);

						//------------------------------

#ifdef USE_FMA
						ymm_s0 = _mm256_fmadd_ps(ymm_s0, ymm_scale, ymm_scale);
						ymm_s0 = _mm256_fmadd_ps(ymm_s0, ymm_bn_w[k], ymm_bn_b[k]);

						ymm_s1 = _mm256_fmadd_ps(ymm_s1, ymm_scale, ymm_scale);
						ymm_s1 = _mm256_fmadd_ps(ymm_s1, ymm_bn_w[k], ymm_bn_b[k]);

						//------------------------------

						ymm_sum = _mm256_fmadd_ps(ymm_s0, ymm_ol_w[k][0], ymm_sum);
						ymm_sum = _mm256_fmadd_ps(ymm_s1, ymm_ol_w[k][1], ymm_sum);
#else
						ymm_s0 = _mm256_mul_ps(ymm_s0, ymm_scale);
						ymm_s0 = _mm256_add_ps(ymm_s0, ymm_scale);

						ymm_s0 = _mm256_mul_ps(ymm_s0, ymm_bn_w[k]);
						ymm_s0 = _mm256_add_ps(ymm_s0, ymm_bn_b[k]);

						ymm_s1 = _mm256_mul_ps(ymm_s1, ymm_scale);
						ymm_s1 = _mm256_add_ps(ymm_s1, ymm_scale);

						ymm_s1 = _mm256_mul_ps(ymm_s1, ymm_bn_w[k]);
						ymm_s1 = _mm256_add_ps(ymm_s1, ymm_bn_b[k]);

						//------------------------------

						ymm_s0 = _mm256_mul_ps(ymm_s0, ymm_ol_w[k][0]);
						ymm_s1 = _mm256_mul_ps(ymm_s1, ymm_ol_w[k][1]);

						ymm_sum = _mm256_add_ps(ymm_sum, ymm_s0);
						ymm_sum = _mm256_add_ps(ymm_sum, ymm_s1);
#endif
					}

					//------------------------------

					ymm_1 = _mm256_permute2f128_ps(ymm_sum, ymm_sum, 19);
					ymm_1 = _mm256_add_ps(ymm_sum, ymm_1);

					ymm_0 = _mm256_permute_ps(ymm_1, 14);
					ymm_1 = _mm256_add_ps(ymm_1, ymm_0);
					ymm_0 = _mm256_permute_ps(ymm_1, 1);
					ymm_1 = _mm256_add_ps(ymm_1, ymm_0);

					//------------------------------

					_mm_store_ss(pDst++, _mm256_extractf128_ps(ymm_1, 0));
				}
				IACA__END
			}
		}
		void CNNPP_v2::tanhW(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict snn_ol_b, float* __restrict tanh_w, float* __restrict scale, size_t L, size_t H, int num_threads)
		{
			const __m256 ymm14 = _mm256_broadcast_ss((float*)&abs_mask);
			const __m256 ymm15 = _mm256_broadcast_ss(&one);
			const __m256 ymm13 = _mm256_broadcast_ss(&tanh_a);
			const __m256 ymm12 = _mm256_broadcast_ss(snn_ol_b);
			const __m256 ymm11 = _mm256_broadcast_ss(scale);
			const __m256 ymm10 = _mm256_broadcast_ss(tanh_w);

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc = src + j * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i < L; i += REG_SIZE)
				{
					__m256 ymm0 = _mm256_load_ps(pSrc);
					pSrc += REG_SIZE;

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

					ymm3 = _mm256_mul_ps(ymm3, ymm11);

					_mm256_store_ps(pDst, ymm3);
					pDst += REG_SIZE;
				}
				IACA__END
			}
		}

}

#endif
}