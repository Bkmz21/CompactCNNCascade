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

#include "cnnpp_simd_avx_v3.h"
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
#ifdef USE_FIXED_POINT

	namespace SIMD
	{
		#define FP2FxP(m256, ymm_toFxP) _mm256_cvtps_epi32(_mm256_mul_ps(m256, ymm_toFxP))
		#define FxP_squeeze(m256i) _mm256_permute4x64_epi64(_mm256_hadd_epi16(m256i, m256i), 216) 
		#define FxP2FP(m256i, ymm_toFP, imm) _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(m256i, imm))), ymm_toFP)

		#define conv_block(k, id)												    \
				ymm_s1 = _mm256_shuffle_epi32(ymm_d1, 78);							\
				ymm_d1 = _mm256_add_epi16(ymm_d1, ymm_s1);							\
				ymm_s2 = _mm256_shuffle_epi8(ymm_s1, ymm_mask1);					\
				ymm_d1 = _mm256_add_epi16(ymm_d1, ymm_s2);							\
				ymm_s1 = _mm256_mulhrs_epi16(ymm_d1, ymm_k_1_##id[k]);				\
				sum_1 = _mm256_add_epi16(sum_1, ymm_s1);							\
				ymm_s2 = _mm256_mulhrs_epi16(ymm_d1, ymm_k_2_##id[k]);				\
				sum_2 = _mm256_add_epi16(sum_2, ymm_s2);					

		void CNNPP_v3::conv_4x4_lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 3;
			if (H == 0) H = src_size_h - 3;
			if (H & 1) H--;

			const __m256i ymm_mask1 = { 0, 1, 0, 1, 0, 1, 0, 1, 128, 128, 128, 128, 128, 128, 128, 128, 4, 5, 4, 5, 4, 5, 4, 5, 128, 128, 128, 128, 128, 128, 128, 128 };
			const __m256i ymm_mask2 = { 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7 };
			const __m256i ymm_mask3 = { 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9 };
			const __m256i ymm_mask4 = { 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11 };
			const __m256i ymm_mask5 = { 128, 128, 128, 128, 128, 128, 128, 128, 8, 9, 8, 9, 8, 9, 8, 9, 128, 128, 128, 128, 128, 128, 128, 128, 12, 13, 12, 13, 12, 13, 12, 13 };

			fct = 1.f;
			const __m256 ymm_toFxP_data = _mm256_set1_ps(toFxP / scale_data);
			const __m256 ymm_toFxP_kernel = _mm256_set1_ps(toFxP / skt1);
			const __m256 ymm_toFxP_conv_b = _mm256_set1_ps(toFxP / (fct * 2.f * scale_data * skt1));
			const __m256 ymm_toFxP_lrelu_w = _mm256_set1_ps(toFxP / skt2);
			const __m256 ymm_toFxP_bn_b = _mm256_set1_ps(toFxP / (fct * 2.f * 2.f * scale_data * skt1 * skt2));

			//const __m256 ymm_toFP_data = _mm256_set1_ps(fct * toFP * 2.f * scale_data * skt1);
			//const __m256 ymm_toFP_data2 = _mm256_set1_ps(fct * toFP * 2.f * 2.f * scale_data * skt1 * skt2);
			fct *= 2.f * 2.f * scale_data * skt1 * skt2;

			__m256i ymm_temp;

			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 0 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k11 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 1 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k12 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 2 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k13 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 3 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k14 = FxP_squeeze(ymm_temp);

			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 4 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k21 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 5 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k22 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 6 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k23 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 7 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k24 = FxP_squeeze(ymm_temp);

			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 8 * REG_SIZE), ymm_toFxP_kernel);  const __m256i ymm_k31 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 9 * REG_SIZE), ymm_toFxP_kernel);  const __m256i ymm_k32 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 10 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k33 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 11 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k34 = FxP_squeeze(ymm_temp);

			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 12 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k41 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 13 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k42 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 14 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k43 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 15 * REG_SIZE), ymm_toFxP_kernel); const __m256i ymm_k44 = FxP_squeeze(ymm_temp);

			ymm_temp = FP2FxP(_mm256_load_ps(conv_b), ymm_toFxP_conv_b);	const __m256i ymm_conv_b = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(lrelu_w1), ymm_toFxP_lrelu_w);	const __m256i ymm_lrelu_w1 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(lrelu_w2), ymm_toFxP_lrelu_w);	const __m256i ymm_lrelu_w2 = FxP_squeeze(ymm_temp);
			ymm_temp = FP2FxP(_mm256_load_ps(bn_b), ymm_toFxP_bn_b);		const __m256i ymm_bn_b = FxP_squeeze(ymm_temp);

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; j += 2)
			{
				float* __restrict pSrc0 = src + (j + 0) * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;
				float* __restrict pSrc3 = src + (j + 3) * src_size_l;
				float* __restrict pSrc4 = src + (j + 4) * src_size_l;
				float* __restrict pDst = dst + (j >> 1) * dst_size_l;

				IACA__START;
				for (size_t i = 0; i < L; i += 4)
				{
					//0
					__m256i ymm_data = FP2FxP(_mm256_loadu_ps(pSrc0), ymm_toFxP_data);
					ymm_data = FxP_squeeze(ymm_data);
					pSrc0 += 4;

					__m256i ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask1);
					__m256i sum_1 = _mm256_mulhrs_epi16(ymm_d, ymm_k11);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask2);
					__m256i ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k12);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask3);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k13);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask4);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k14);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask5);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k11);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);

					//1
					ymm_data = FP2FxP(_mm256_loadu_ps(pSrc1), ymm_toFxP_data);
					ymm_data = FxP_squeeze(ymm_data);
					pSrc1 += 4;

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask1);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k21);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					__m256i sum_2 = _mm256_mulhrs_epi16(ymm_d, ymm_k11);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask2);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k22);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k12);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask3);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k23);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k13);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask4);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k24);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k14);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask5);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k21);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k11);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					//2
					ymm_data = FP2FxP(_mm256_loadu_ps(pSrc2), ymm_toFxP_data);
					ymm_data = FxP_squeeze(ymm_data);
					pSrc2 += 4;

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask1);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k31);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k21);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask2);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k32);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k22);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask3);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k33);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k23);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask4);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k34);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k24);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask5);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k31);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k21);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					//3
					ymm_data = FP2FxP(_mm256_loadu_ps(pSrc3), ymm_toFxP_data);
					ymm_data = FxP_squeeze(ymm_data);
					pSrc3 += 4;

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask1);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k41);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k31);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask2);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k42);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k32);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask3);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k43);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k33);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask4);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k44);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k34);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask5);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k41);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k31);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					//4
					ymm_data = FP2FxP(_mm256_loadu_ps(pSrc4), ymm_toFxP_data);
					ymm_data = FxP_squeeze(ymm_data);
					pSrc4 += 4;

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask1);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k41);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask2);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k42);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask3);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k43);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask4);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k44);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					ymm_d = _mm256_shuffle_epi8(ymm_data, ymm_mask5);
					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_k41);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					//-----------------------------

					sum_1 = _mm256_add_epi16(sum_1, ymm_conv_b);
					ymm_d = _mm256_max_epi16(sum_1, _mm256_setzero_si256());

					ymm_m = _mm256_mulhrs_epi16(sum_1, ymm_lrelu_w1);
					sum_1 = _mm256_add_epi16(ymm_bn_b, ymm_m);

					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_lrelu_w2);
					sum_1 = _mm256_add_epi16(sum_1, ymm_m);

					sum_2 = _mm256_add_epi16(sum_2, ymm_conv_b);
					ymm_d = _mm256_max_epi16(sum_2, _mm256_setzero_si256());

					ymm_m = _mm256_mulhrs_epi16(sum_2, ymm_lrelu_w1);
					sum_2 = _mm256_add_epi16(ymm_bn_b, ymm_m);

					ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_lrelu_w2);
					sum_2 = _mm256_add_epi16(sum_2, ymm_m);

					//-----------------------------

					sum_1 = _mm256_max_epi16(sum_1, sum_2);
					sum_1 = _mm256_max_epi16(_mm256_permute4x64_epi64(sum_1, 8), _mm256_permute4x64_epi64(sum_1, 13));

					_mm_storeu_si128((__m128i*)pDst, _mm256_extracti128_si256(sum_1, 0));
					pDst += REG_SIZE / 2;

					//_mm_storeu_si128((__m128i*)pDst, _mm256_cvtps_ph(FxP2FP(sum_1, ymm_toFP_data2, 0), 0));
					//pDst += REG_SIZE / 2;
				}
				IACA__END
			}
		}
		void CNNPP_v3::conv_3x3_lrelu_bn_max(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 2;
			if (H == 0) H = src_size_h - 2;

			ALIGN(ALIGN_DEF) const int set1_mask[8] = { 0, 4, 6, 2, 1, 5, 7, 3 };
			const __m256i ymm_mask0 = _mm256_load_si256((__m256i*)set1_mask);

			const __m256i ymm_mask1 = { 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13 };
			const __m256i ymm_mask2 = { 128, 128, 4, 5, 2, 3, 128, 128, 128, 128, 12, 13, 10, 11, 128, 128, 128, 128, 4, 5, 2, 3, 128, 128, 128, 128, 12, 13, 10, 11, 128, 128 };
			const __m256i ymm_mask3 = { 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15 };
			const __m256i ymm_mask4 = { 0, 1, 8, 9, 12, 13, 4, 5, 2, 3, 10, 11, 14, 15, 6, 7, 0, 1, 8, 9, 12, 13, 4, 5, 2, 3, 10, 11, 14, 15, 6, 7 };

			const __m256 ymm_toFxP_data = _mm256_set1_ps(toFxP / scale_data);
			const __m256 ymm_toFxP_kernel = _mm256_set1_ps(toFxP / skt3);
			const __m256 ymm_toFxP_conv_b = _mm256_set1_ps(toFxP / (fct * 2.f * skt3));
			const __m256 ymm_toFxP_lrelu_w = _mm256_set1_ps(toFxP / (skt4));
			const __m256 ymm_toFxP_bn_b = _mm256_set1_ps(toFxP / (fct * 2.f * 2.f * skt3 * skt4));

			//const __m256 ymm_toFP_data = _mm256_set1_ps(fct * toFP * 2.f * skt3);
			//const __m256 ymm_toFP_data2 = _mm256_set1_ps(fct * toFP * 2.f * 2.f * skt3 * skt4);
			fct *= 2.f * 2.f * skt3 * skt4;

			__m256i ymm_temp;
			IACA__START
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 0 * REG_SIZE), ymm_toFxP_kernel); __m256i ymm_k1 = _mm256_permutevar8x32_epi32(ymm_temp, ymm_mask0);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 1 * REG_SIZE), ymm_toFxP_kernel); __m256i ymm_k2 = _mm256_permutevar8x32_epi32(ymm_temp, ymm_mask0);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 2 * REG_SIZE), ymm_toFxP_kernel); __m256i ymm_k3 = _mm256_permutevar8x32_epi32(ymm_temp, ymm_mask0);
			__m256i ymm_p1 = _mm256_hadd_epi16(ymm_k1, ymm_k2);
			__m256i ymm_p2 = _mm256_hadd_epi16(ymm_k3, _mm256_setzero_si256());
			const __m256i ymm_kp311 = _mm256_shuffle_epi8(_mm256_permute2x128_si256(ymm_p1, ymm_p2, 32), ymm_mask3);
			const __m256i ymm_kp312 = _mm256_shuffle_epi8(_mm256_permute2x128_si256(ymm_p1, ymm_p2, 49), ymm_mask3);
			ymm_temp = _mm256_shuffle_epi8(ymm_kp311, ymm_mask1);
			const __m256i ymm_kp313 = _mm256_blend_epi16(ymm_temp, _mm256_permute2x128_si256(ymm_temp, ymm_temp, 1), 85);
			ymm_temp = _mm256_shuffle_epi8(ymm_kp312, ymm_mask1);
			const __m256i ymm_kp314 = _mm256_blend_epi16(ymm_temp, _mm256_permute2x128_si256(ymm_temp, ymm_temp, 1), 85);

			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 3 * REG_SIZE), ymm_toFxP_kernel); ymm_k1 = _mm256_permutevar8x32_epi32(ymm_temp, ymm_mask0);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 4 * REG_SIZE), ymm_toFxP_kernel); ymm_k2 = _mm256_permutevar8x32_epi32(ymm_temp, ymm_mask0);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 5 * REG_SIZE), ymm_toFxP_kernel); ymm_k3 = _mm256_permutevar8x32_epi32(ymm_temp, ymm_mask0);
			ymm_p1 = _mm256_hadd_epi16(ymm_k1, ymm_k2);
			ymm_p2 = _mm256_hadd_epi16(ymm_k3, _mm256_setzero_si256());
			const __m256i ymm_kp321 = _mm256_shuffle_epi8(_mm256_permute2x128_si256(ymm_p1, ymm_p2, 32), ymm_mask3);
			const __m256i ymm_kp322 = _mm256_shuffle_epi8(_mm256_permute2x128_si256(ymm_p1, ymm_p2, 49), ymm_mask3);
			ymm_temp = _mm256_shuffle_epi8(ymm_kp321, ymm_mask1);
			const __m256i ymm_kp323 = _mm256_blend_epi16(ymm_temp, _mm256_permute2x128_si256(ymm_temp, ymm_temp, 1), 85);
			ymm_temp = _mm256_shuffle_epi8(ymm_kp322, ymm_mask1);
			const __m256i ymm_kp324 = _mm256_blend_epi16(ymm_temp, _mm256_permute2x128_si256(ymm_temp, ymm_temp, 1), 85);

			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 6 * REG_SIZE), ymm_toFxP_kernel); ymm_k1 = _mm256_permutevar8x32_epi32(ymm_temp, ymm_mask0);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 7 * REG_SIZE), ymm_toFxP_kernel); ymm_k2 = _mm256_permutevar8x32_epi32(ymm_temp, ymm_mask0);
			ymm_temp = FP2FxP(_mm256_load_ps(kernel + 8 * REG_SIZE), ymm_toFxP_kernel); ymm_k3 = _mm256_permutevar8x32_epi32(ymm_temp, ymm_mask0);
			ymm_p1 = _mm256_hadd_epi16(ymm_k1, ymm_k2);
			ymm_p2 = _mm256_hadd_epi16(ymm_k3, _mm256_setzero_si256());
			const __m256i ymm_kp331 = _mm256_shuffle_epi8(_mm256_permute2x128_si256(ymm_p1, ymm_p2, 32), ymm_mask3);
			const __m256i ymm_kp332 = _mm256_shuffle_epi8(_mm256_permute2x128_si256(ymm_p1, ymm_p2, 49), ymm_mask3);
			ymm_temp = _mm256_shuffle_epi8(ymm_kp331, ymm_mask1);
			const __m256i ymm_kp333 = _mm256_blend_epi16(ymm_temp, _mm256_permute2x128_si256(ymm_temp, ymm_temp, 1), 85);
			ymm_temp = _mm256_shuffle_epi8(ymm_kp332, ymm_mask1);
			const __m256i ymm_kp334 = _mm256_blend_epi16(ymm_temp, _mm256_permute2x128_si256(ymm_temp, ymm_temp, 1), 85);

			ymm_temp = FP2FxP(_mm256_load_ps(conv_b), ymm_toFxP_conv_b);	const __m256i ymm_conv_b = _mm256_shuffle_epi8(FxP_squeeze(ymm_temp), ymm_mask4);
			ymm_temp = FP2FxP(_mm256_load_ps(lrelu_w1), ymm_toFxP_lrelu_w);	const __m256i ymm_lrelu_w1 = _mm256_shuffle_epi8(FxP_squeeze(ymm_temp), ymm_mask4);
			ymm_temp = FP2FxP(_mm256_load_ps(lrelu_w2), ymm_toFxP_lrelu_w);	const __m256i ymm_lrelu_w2 = _mm256_shuffle_epi8(FxP_squeeze(ymm_temp), ymm_mask4);
			ymm_temp = FP2FxP(_mm256_load_ps(bn_b), ymm_toFxP_bn_b);		const __m256i ymm_bn_b = _mm256_shuffle_epi8(FxP_squeeze(ymm_temp), ymm_mask4);
			IACA__END

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src + j * src_size_l;
				float* __restrict pSrc1 = src + (j + 1) * src_size_l;
				float* __restrict pSrc2 = src + (j + 2) * src_size_l;

				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
				for (size_t i = 0; i < L; i += 4)
				{
					//0
					__m256i ymm_d1 = _mm256_loadu_si256((__m256i*)pSrc0);
					__m256i ymm_d2 = _mm256_loadu_si256((__m256i*)(pSrc0 + REG_SIZE / 2));
					pSrc0 += REG_SIZE;

					__m256i ymm_s1 = _mm256_shuffle_epi8(ymm_d1, ymm_mask1);
					__m256i ymm_a1 = _mm256_add_epi16(ymm_d1, ymm_s1);
					ymm_s1 = _mm256_shuffle_epi8(ymm_d1, ymm_mask2);
					ymm_a1 = _mm256_add_epi16(ymm_a1, ymm_s1);
					ymm_a1 = _mm256_shuffle_epi8(ymm_a1, ymm_mask3);

					__m256i ymm_s2 = _mm256_shuffle_epi8(ymm_d2, ymm_mask1);
					__m256i ymm_a2 = _mm256_add_epi16(ymm_d2, ymm_s2);
					ymm_s2 = _mm256_shuffle_epi8(ymm_d2, ymm_mask2);
					ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_s2);
					ymm_a2 = _mm256_shuffle_epi8(ymm_a2, ymm_mask3);

					__m256i ymm_r11 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp311);
					__m256i ymm_r12 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp312);
					__m256i ymm_r13 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp311);
					__m256i ymm_r14 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp312);

					__m256i ymm_r21 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp313);
					__m256i ymm_r22 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp314);
					__m256i ymm_r23 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp313);
					__m256i ymm_r24 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp314);

					//1
					ymm_d1 = _mm256_loadu_si256((__m256i*)pSrc1);
					ymm_d2 = _mm256_loadu_si256((__m256i*)(pSrc1 + REG_SIZE / 2));
					pSrc1 += REG_SIZE;

					ymm_s1 = _mm256_shuffle_epi8(ymm_d1, ymm_mask1);
					ymm_a1 = _mm256_add_epi16(ymm_d1, ymm_s1);
					ymm_s1 = _mm256_shuffle_epi8(ymm_d1, ymm_mask2);
					ymm_a1 = _mm256_add_epi16(ymm_a1, ymm_s1);
					ymm_a1 = _mm256_shuffle_epi8(ymm_a1, ymm_mask3);

					ymm_s2 = _mm256_shuffle_epi8(ymm_d2, ymm_mask1);
					ymm_a2 = _mm256_add_epi16(ymm_d2, ymm_s2);
					ymm_s2 = _mm256_shuffle_epi8(ymm_d2, ymm_mask2);
					ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_s2);
					ymm_a2 = _mm256_shuffle_epi8(ymm_a2, ymm_mask3);

					ymm_s1 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp321);
					ymm_r11 = _mm256_add_epi16(ymm_r11, ymm_s1);
					ymm_s2 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp322);
					ymm_r12 = _mm256_add_epi16(ymm_r12, ymm_s2);
					ymm_s1 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp321);
					ymm_r13 = _mm256_add_epi16(ymm_r13, ymm_s1);
					ymm_s2 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp322);
					ymm_r14 = _mm256_add_epi16(ymm_r14, ymm_s2);

					ymm_s1 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp323);
					ymm_r21 = _mm256_add_epi16(ymm_r21, ymm_s1);
					ymm_s2 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp324);
					ymm_r22 = _mm256_add_epi16(ymm_r22, ymm_s2);
					ymm_s1 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp323);
					ymm_r23 = _mm256_add_epi16(ymm_r23, ymm_s1);
					ymm_s2 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp324);
					ymm_r24 = _mm256_add_epi16(ymm_r24, ymm_s2);

					//2
					ymm_d1 = _mm256_loadu_si256((__m256i*)pSrc2);
					ymm_d2 = _mm256_loadu_si256((__m256i*)(pSrc2 + REG_SIZE / 2));
					pSrc2 += REG_SIZE;

					ymm_s1 = _mm256_shuffle_epi8(ymm_d1, ymm_mask1);
					ymm_a1 = _mm256_add_epi16(ymm_d1, ymm_s1);
					ymm_s1 = _mm256_shuffle_epi8(ymm_d1, ymm_mask2);
					ymm_a1 = _mm256_add_epi16(ymm_a1, ymm_s1);
					ymm_a1 = _mm256_shuffle_epi8(ymm_a1, ymm_mask3);

					ymm_s2 = _mm256_shuffle_epi8(ymm_d2, ymm_mask1);
					ymm_a2 = _mm256_add_epi16(ymm_d2, ymm_s2);
					ymm_s2 = _mm256_shuffle_epi8(ymm_d2, ymm_mask2);
					ymm_a2 = _mm256_add_epi16(ymm_a2, ymm_s2);
					ymm_a2 = _mm256_shuffle_epi8(ymm_a2, ymm_mask3);

					ymm_s1 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp331);
					ymm_r11 = _mm256_add_epi16(ymm_r11, ymm_s1);
					ymm_s2 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp332);
					ymm_r12 = _mm256_add_epi16(ymm_r12, ymm_s2);
					ymm_s1 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp331);
					ymm_r13 = _mm256_add_epi16(ymm_r13, ymm_s1);
					ymm_s2 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp332);
					ymm_r14 = _mm256_add_epi16(ymm_r14, ymm_s2);
					
					ymm_s1 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp333);
					ymm_r21 = _mm256_add_epi16(ymm_r21, ymm_s1);
					ymm_s2 = _mm256_mulhrs_epi16(ymm_a1, ymm_kp334);
					ymm_r22 = _mm256_add_epi16(ymm_r22, ymm_s2);
					ymm_s1 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp333);
					ymm_r23 = _mm256_add_epi16(ymm_r23, ymm_s1);
					ymm_s2 = _mm256_mulhrs_epi16(ymm_a2, ymm_kp334);
					ymm_r24 = _mm256_add_epi16(ymm_r24, ymm_s2);

					//-----------------------------

					ymm_r11 = _mm256_hadd_epi16(ymm_r11, ymm_r12);
					ymm_r13 = _mm256_hadd_epi16(ymm_r13, ymm_r14);

					ymm_s1 = _mm256_permute2x128_si256(ymm_r11, ymm_r13, 32);
					ymm_s2 = _mm256_permute2x128_si256(ymm_r13, ymm_r11, 19);

					ymm_r11 = _mm256_add_epi16(ymm_s1, ymm_s2);

#if 1
					ymm_r11 = _mm256_add_epi16(ymm_r11, ymm_conv_b);	
					ymm_s1 = _mm256_max_epi16(ymm_r11, _mm256_setzero_si256());

					ymm_s2 = _mm256_mulhrs_epi16(ymm_r11, ymm_lrelu_w1);		
					ymm_r11 = _mm256_add_epi16(ymm_bn_b, ymm_s2);
					
					ymm_s2 = _mm256_mulhrs_epi16(ymm_s1, ymm_lrelu_w2);
					ymm_r11 = _mm256_add_epi16(ymm_r11, ymm_s2);
#else
					__m256i ymm_r11_0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ymm_r11, 0));
					__m256i ymm_r11_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ymm_r11, 1));

					ymm_r11_0 = _mm256_add_epi32(ymm_r11_0, ymm_conv_b);
					ymm_r11_1 = _mm256_add_epi32(ymm_r11_1, ymm_conv_b);

					__m256i ymm_s1_0 = _mm256_max_epi32(ymm_r11_0, _mm256_setzero_si256());
					__m256i ymm_s1_1 = _mm256_max_epi32(ymm_r11_1, _mm256_setzero_si256());

					__m256i ymm_s2_0 = _mm256_mul_epi32(ymm_r11_0, ymm_lrelu_w1);
					__m256i ymm_s2_1 = _mm256_mul_epi32(ymm_r11_1, ymm_lrelu_w1);
					ymm_s2_0 = _mm256_srai_epi32(ymm_s2_0, 14);
					ymm_s2_1 = _mm256_srai_epi32(ymm_s2_1, 14);

					ymm_r11_0 = _mm256_add_epi32(ymm_bn_b, ymm_s2_0);
					ymm_r11_1 = _mm256_add_epi32(ymm_bn_b, ymm_s2_1);

					ymm_s2_0 = _mm256_mul_epi32(ymm_s1_0, ymm_lrelu_w2);
					ymm_s2_1 = _mm256_mul_epi32(ymm_s1_1, ymm_lrelu_w2);
					ymm_s2_0 = _mm256_srai_epi32(ymm_s2_0, 14);
					ymm_s2_1 = _mm256_srai_epi32(ymm_s2_1, 14);

					ymm_r11_0 = _mm256_add_epi32(ymm_r11_0, ymm_s2_0);
					ymm_r11_1 = _mm256_add_epi32(ymm_r11_1, ymm_s2_1);

					ymm_r11 = _mm256_permute4x64_epi64(_mm256_hadd_epi16(ymm_r11_0, ymm_r11_1), 216);
#endif

					//-----------------------------

					ymm_r21 = _mm256_hadd_epi16(ymm_r21, ymm_r22);
					ymm_r23 = _mm256_hadd_epi16(ymm_r23, ymm_r24);

					ymm_s1 = _mm256_permute2x128_si256(ymm_r21, ymm_r23, 32);
					ymm_s2 = _mm256_permute2x128_si256(ymm_r23, ymm_r21, 19);
					
					ymm_r21 = _mm256_add_epi16(ymm_s1, ymm_s2);

#if 1
					ymm_r21 = _mm256_add_epi16(ymm_r21, ymm_conv_b);
					ymm_s1 = _mm256_max_epi16(ymm_r21, _mm256_setzero_si256());

					ymm_s2 = _mm256_mulhrs_epi16(ymm_r21, ymm_lrelu_w1);			
					ymm_r21 = _mm256_add_epi16(ymm_bn_b, ymm_s2);

					ymm_s2 = _mm256_mulhrs_epi16(ymm_s1, ymm_lrelu_w2);
					ymm_r21 = _mm256_add_epi16(ymm_r21, ymm_s2);
#else
					ymm_r11_0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ymm_r21, 0));
					ymm_r11_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ymm_r21, 1));

					ymm_r11_0 = _mm256_add_epi32(ymm_r11_0, ymm_conv_b);
					ymm_r11_1 = _mm256_add_epi32(ymm_r11_1, ymm_conv_b);

					ymm_s1_0 = _mm256_max_epi32(ymm_r11_0, _mm256_setzero_si256());
					ymm_s1_1 = _mm256_max_epi32(ymm_r11_1, _mm256_setzero_si256());
					
					ymm_s2_0 = _mm256_mul_epi32(ymm_r11_0, ymm_lrelu_w1);
					ymm_s2_1 = _mm256_mul_epi32(ymm_r11_1, ymm_lrelu_w1);
					ymm_s2_0 = _mm256_srai_epi32(ymm_s2_0, 14);
					ymm_s2_1 = _mm256_srai_epi32(ymm_s2_1, 14);

					ymm_r11_0 = _mm256_add_epi32(ymm_bn_b, ymm_s2_0);
					ymm_r11_1 = _mm256_add_epi32(ymm_bn_b, ymm_s2_1);

					ymm_s2_0 = _mm256_mul_epi32(ymm_s1_0, ymm_lrelu_w2);
					ymm_s2_1 = _mm256_mul_epi32(ymm_s1_1, ymm_lrelu_w2);
					ymm_s2_0 = _mm256_srai_epi32(ymm_s2_0, 14);
					ymm_s2_1 = _mm256_srai_epi32(ymm_s2_1, 14);

					ymm_r11_0 = _mm256_add_epi32(ymm_r11_0, ymm_s2_0);
					ymm_r11_1 = _mm256_add_epi32(ymm_r11_1, ymm_s2_1);

					ymm_r21 = _mm256_permute4x64_epi64(_mm256_hadd_epi16(ymm_r11_0, ymm_r11_1), 216);
#endif

					//-----------------------------

					ymm_r11 = _mm256_max_epi16(ymm_r11, ymm_r21);

					_mm256_storeu_si256((__m256i*)pDst, ymm_r11);
					pDst += REG_SIZE;

					//const __m256i ymm_mask = { 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15 };
					//ymm_r11 = _mm256_shuffle_epi8(ymm_r11, ymm_mask);
					//_mm_storeu_si128((__m128i*)pDst, _mm256_cvtps_ph(FxP2FP(ymm_r11, ymm_toFP_data2, 0), 0));
					//pDst += REG_SIZE / 2;
					//_mm_storeu_si128((__m128i*)pDst, _mm256_cvtps_ph(FxP2FP(ymm_r11, ymm_toFP_data2, 1), 0));
					//pDst += REG_SIZE / 2;
				}
				IACA__END
			}
		}
		void CNNPP_v3::conv_5x4_lrelu_bn(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
		{
			if (L == 0) L = src_size_l - 4;
			if (H == 0) H = src_size_h - 5;

			const __m256i ymm_mask0 = { 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 };
			const __m256i ymm_mask1 = { 128, 128, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 128, 128, 128, 128, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 128, 128 };
			const __m256i ymm_mask3 = { 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15 };
			
			const __m256 ymm_toFxP_data = _mm256_set1_ps(toFxP / scale_data);
			const __m256 ymm_toFxP_kernel = _mm256_set1_ps(toFxP / skt5);
			const __m256 ymm_toFxP_conv_b = _mm256_set1_ps(toFxP / (fct * 2.f * skt5));
			const __m256 ymm_toFxP_lrelu_w = _mm256_set1_ps(toFxP / skt6);
			const __m256 ymm_toFxP_bn_b = _mm256_set1_ps(toFxP / (fct * 2.f * 2.f * skt5 * skt6));

			//const __m256 ymm_toFP_data = _mm256_set1_ps(fct * toFP * 2.f * skt5);
			//const __m256 ymm_toFP_data2 = _mm256_set1_ps(fct * toFP * 2.f * 2.f * skt5 * skt6);
			fct *= 2.f * 2.f * skt5 * skt6;

			__m256i ymm_k_1_0[5];
			__m256i ymm_k_1_1[5];
			__m256i ymm_k_1_2[5];
			__m256i ymm_k_1_3[5];

			for (size_t k = 0; k < 5; ++k)
			{
				ymm_k_1_0[k] = _mm256_shuffle_epi8(FxP_squeeze(FP2FxP(_mm256_load_ps(kernel + (0 + 4 * k) * REG_SIZE), ymm_toFxP_kernel)), ymm_mask0);
				ymm_k_1_1[k] = _mm256_shuffle_epi8(FxP_squeeze(FP2FxP(_mm256_load_ps(kernel + (1 + 4 * k) * REG_SIZE), ymm_toFxP_kernel)), ymm_mask0);
				ymm_k_1_2[k] = _mm256_shuffle_epi8(FxP_squeeze(FP2FxP(_mm256_load_ps(kernel + (2 + 4 * k) * REG_SIZE), ymm_toFxP_kernel)), ymm_mask0);
				ymm_k_1_3[k] = _mm256_shuffle_epi8(FxP_squeeze(FP2FxP(_mm256_load_ps(kernel + (3 + 4 * k) * REG_SIZE), ymm_toFxP_kernel)), ymm_mask0);
			}

			__m256i ymm_k_2_0[5];
			__m256i ymm_k_2_1[5];
			__m256i ymm_k_2_2[5];
			__m256i ymm_k_2_3[5];

			kernel += 5 * 4 * REG_SIZE;
			for (size_t k = 0; k < 5; ++k)
			{
				ymm_k_2_0[k] = _mm256_shuffle_epi8(FxP_squeeze(FP2FxP(_mm256_load_ps(kernel + (0 + 4 * k) * REG_SIZE), ymm_toFxP_kernel)), ymm_mask0);
				ymm_k_2_1[k] = _mm256_shuffle_epi8(FxP_squeeze(FP2FxP(_mm256_load_ps(kernel + (1 + 4 * k) * REG_SIZE), ymm_toFxP_kernel)), ymm_mask0);
				ymm_k_2_2[k] = _mm256_shuffle_epi8(FxP_squeeze(FP2FxP(_mm256_load_ps(kernel + (2 + 4 * k) * REG_SIZE), ymm_toFxP_kernel)), ymm_mask0);
				ymm_k_2_3[k] = _mm256_shuffle_epi8(FxP_squeeze(FP2FxP(_mm256_load_ps(kernel + (3 + 4 * k) * REG_SIZE), ymm_toFxP_kernel)), ymm_mask0);
			}

			const __m256 _ymm_conv_b_1 = _mm256_load_ps(conv_b);
			const __m256 _ymm_conv_b_2 = _mm256_load_ps(conv_b + REG_SIZE);
			const __m256 _ymm_lrelu_w1_1 = _mm256_load_ps(lrelu_w1);
			const __m256 _ymm_lrelu_w1_2 = _mm256_load_ps(lrelu_w1 + REG_SIZE);
			const __m256 _ymm_lrelu_w2_1 = _mm256_load_ps(lrelu_w2);
			const __m256 _ymm_lrelu_w2_2 = _mm256_load_ps(lrelu_w2 + REG_SIZE);
			const __m256 _ymm_bn_b_1 = _mm256_load_ps(bn_b);
			const __m256 _ymm_bn_b_2 = _mm256_load_ps(bn_b + REG_SIZE);

			const __m256i ymm_conv_b_1 = FxP_squeeze(FP2FxP(_ymm_conv_b_1, ymm_toFxP_conv_b));
			const __m256i ymm_conv_b_2 = FxP_squeeze(FP2FxP(_ymm_conv_b_2, ymm_toFxP_conv_b));
			const __m256i ymm_lrelu_w1_1 = FxP_squeeze(FP2FxP(_ymm_lrelu_w1_1, ymm_toFxP_lrelu_w));
			const __m256i ymm_lrelu_w1_2 = FxP_squeeze(FP2FxP(_ymm_lrelu_w1_2, ymm_toFxP_lrelu_w));
			const __m256i ymm_lrelu_w2_1 = FxP_squeeze(FP2FxP(_ymm_lrelu_w2_1, ymm_toFxP_lrelu_w));
			const __m256i ymm_lrelu_w2_2 = FxP_squeeze(FP2FxP(_ymm_lrelu_w2_2, ymm_toFxP_lrelu_w));
			const __m256i ymm_bn_b_1 = FxP_squeeze(FP2FxP(_ymm_bn_b_1, ymm_toFxP_bn_b));
			const __m256i ymm_bn_b_2 = FxP_squeeze(FP2FxP(_ymm_bn_b_2, ymm_toFxP_bn_b));

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc = src + (j << 1) * src_size_l;
				float* __restrict pDst = dst + j * dst_size_l;

				IACA__START
					for (size_t i = 0; i < L; i += 2)
					{
						__m256i sum_1 = _mm256_setzero_si256();
						__m256i sum_2 = _mm256_setzero_si256();

#pragma unroll
						for (size_t k = 0; k < 5; ++k)
						{
							float* __restrict pSrc_temp_1 = pSrc + (k << 1) * src_size_l;
							float* __restrict pSrc_temp_2 = pSrc + ((k << 1) + 1) * src_size_l;
							__m256i ymm_s1, ymm_s2;

							//0
							__m256i ymm_d1 = _mm256_loadu_si256((__m256i*)pSrc_temp_1);
							__m256i ymm_d2 = _mm256_loadu_si256((__m256i*)pSrc_temp_2);
							pSrc_temp_1 += REG_SIZE / 2;
							pSrc_temp_2 += REG_SIZE / 2;
							ymm_d1 = _mm256_max_epi16(ymm_d1, ymm_d2);
							conv_block(k, 0);

							//1
							ymm_d1 = _mm256_loadu_si256((__m256i*)pSrc_temp_1);
							ymm_d2 = _mm256_loadu_si256((__m256i*)pSrc_temp_2);
							pSrc_temp_1 += REG_SIZE / 2;
							pSrc_temp_2 += REG_SIZE / 2;
							ymm_d1 = _mm256_max_epi16(ymm_d1, ymm_d2);
							conv_block(k, 1);

							//2
							ymm_d1 = _mm256_loadu_si256((__m256i*)pSrc_temp_1);
							ymm_d2 = _mm256_loadu_si256((__m256i*)pSrc_temp_2);
							pSrc_temp_1 += REG_SIZE / 2;
							pSrc_temp_2 += REG_SIZE / 2;
							ymm_d1 = _mm256_max_epi16(ymm_d1, ymm_d2);
							conv_block(k, 2);

							//3
							ymm_d1 = _mm256_loadu_si256((__m256i*)pSrc_temp_1);
							ymm_d2 = _mm256_loadu_si256((__m256i*)pSrc_temp_2);
							pSrc_temp_1 += REG_SIZE / 2;
							pSrc_temp_2 += REG_SIZE / 2;
							ymm_d1 = _mm256_max_epi16(ymm_d1, ymm_d2);
							conv_block(k, 3);
						}
						pSrc += REG_SIZE;

						//-----------------------------

#if 0
						sum_1 = _mm256_shuffle_epi8(sum_1, ymm_mask4);
						sum_2 = _mm256_shuffle_epi8(sum_2, ymm_mask4);

						__m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_1, 0)));
						__m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_1, 1)));

						q1 = _mm256_fmadd_ps(q1, ymm_toFP_data, ymm_conv_b_1);
						q2 = _mm256_fmadd_ps(q2, ymm_toFP_data, ymm_conv_b_1);

						__m256 ymm_d1 = _mm256_max_ps(q1, _mm256_setzero_ps());
						__m256 ymm_d2 = _mm256_max_ps(q2, _mm256_setzero_ps());

						q1 = _mm256_fmadd_ps(q1, ymm_lrelu_w1_1, ymm_bn_b_1);
						q2 = _mm256_fmadd_ps(q2, ymm_lrelu_w1_1, ymm_bn_b_1);

						q1 = _mm256_fmadd_ps(ymm_d1, ymm_lrelu_w2_1, q1);
						q2 = _mm256_fmadd_ps(ymm_d2, ymm_lrelu_w2_1, q2);

						_mm256_storeu_ps(pDst, q1);
						_mm256_storeu_ps(pDst + 2 * REG_SIZE, q2);

						__m256 q3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_2, 0)));
						__m256 q4 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_2, 1)));

						q3 = _mm256_fmadd_ps(q3, ymm_toFP_data, ymm_conv_b_2);
						q4 = _mm256_fmadd_ps(q4, ymm_toFP_data, ymm_conv_b_2);

						ymm_d1 = _mm256_max_ps(q3, _mm256_setzero_ps());
						ymm_d2 = _mm256_max_ps(q4, _mm256_setzero_ps());

						q3 = _mm256_fmadd_ps(q3, ymm_lrelu_w1_2, ymm_bn_b_2);
						q4 = _mm256_fmadd_ps(q4, ymm_lrelu_w1_2, ymm_bn_b_2);

						q3 = _mm256_fmadd_ps(ymm_d1, ymm_lrelu_w2_2, q3);
						q4 = _mm256_fmadd_ps(ymm_d2, ymm_lrelu_w2_2, q4);

						_mm256_storeu_ps(pDst + REG_SIZE, q3);
						_mm256_storeu_ps(pDst + 3 * REG_SIZE, q4);

						pDst += 4 * REG_SIZE;
#else
						sum_1 = _mm256_shuffle_epi8(sum_1, ymm_mask3);
						sum_1 = _mm256_add_epi16(sum_1, ymm_conv_b_1);

						__m256i ymm_d = _mm256_max_epi16(sum_1, _mm256_setzero_si256());
						__m256i ymm_m = _mm256_mulhrs_epi16(sum_1, ymm_lrelu_w1_1); //loss of accuracy!
						sum_1 = _mm256_add_epi16(ymm_bn_b_1, ymm_m);

						ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_lrelu_w2_1);
						sum_1 = _mm256_add_epi16(sum_1, ymm_m);
						_mm256_storeu_si256((__m256i*)pDst, sum_1);
						pDst += REG_SIZE;

						sum_2 = _mm256_shuffle_epi8(sum_2, ymm_mask3);
						sum_2 = _mm256_add_epi16(sum_2, ymm_conv_b_2);

						ymm_d = _mm256_max_epi16(sum_2, _mm256_setzero_si256());
						ymm_m = _mm256_mulhrs_epi16(sum_2, ymm_lrelu_w1_2);
						sum_2 = _mm256_add_epi16(ymm_bn_b_2, ymm_m);

						ymm_m = _mm256_mulhrs_epi16(ymm_d, ymm_lrelu_w2_2);
						sum_2 = _mm256_add_epi16(sum_2, ymm_m);
						_mm256_storeu_si256((__m256i*)pDst, sum_2);
						pDst += REG_SIZE;

						//-----------------------------

						//__m256 q1 = FxP2FP(sum_1, ymm_toFP_data2, 0);
						//__m256 q2 = FxP2FP(sum_1, ymm_toFP_data2, 1);
						//__m256 q3 = FxP2FP(sum_2, ymm_toFP_data2, 0);
						//__m256 q4 = FxP2FP(sum_2, ymm_toFP_data2, 1);

						//_mm256_storeu_ps(pDst, q1);
						//_mm256_storeu_ps(pDst + 2 * REG_SIZE, q2);

						//_mm256_storeu_ps(pDst + REG_SIZE, q3);
						//_mm256_storeu_ps(pDst + 3 * REG_SIZE, q4);

						//pDst += 4 * REG_SIZE;
#endif
					}
				IACA__END
			}
		}
		void CNNPP_v3::mulCN_add_tanhW_add(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float** __restrict snn_hl_w, float** __restrict snn_hl_b, float* __restrict snn_tanh_w, float* __restrict snn_bn_w, float* __restrict snn_bn_b, float** __restrict snn_ol_w, size_t L, size_t H, int num_threads)
		{
			ALIGN(ALIGN_DEF) const int set1_mask[8] = { 1, 2, 3, 4, 5, 6, 7, 0 };
			const __m256i ymm_mask_temp = _mm256_load_si256((__m256i*)set1_mask);

			const __m256 ymm_toFP_data = _mm256_set1_ps(fct * toFP);

			__m256 ymm_hl_w[4][8];
			for (size_t k = 0; k < 4; ++k)
			{
				for (size_t i = 0; i < 8; ++i)
				{
					ymm_hl_w[k][i] = _mm256_load_ps(snn_hl_w[k] + i * REG_SIZE);
					ymm_hl_w[k][i] = _mm256_mul_ps(ymm_hl_w[k][i], ymm_toFP_data);
				}
			}

			__m256 ymm_tanh_w[2];
			for (size_t i = 0; i < 2; ++i)
			{
				ymm_tanh_w[i] = _mm256_load_ps(snn_tanh_w + i * REG_SIZE);
			}

			__m256 ymm_hl_b[4][2];
			for (size_t k = 0; k < 4; ++k)
			{
				for (size_t i = 0; i < 2; ++i)
				{
					ymm_hl_b[k][i] = _mm256_load_ps(snn_hl_b[k] + i * REG_SIZE);
					ymm_hl_b[k][i] = _mm256_mul_ps(ymm_hl_b[k][i], ymm_tanh_w[i]);
				}
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
					__m256 ymm_0, ymm_1;
					if (i & 1)
					{
						ymm_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)(pSrc + REG_SIZE / 2))));
						ymm_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)(pSrc + 3 * REG_SIZE / 2))));
						pSrc += 2 * REG_SIZE;
					}
					else
					{
						ymm_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)pSrc)));
						ymm_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)(pSrc + REG_SIZE))));
					}

					//__m256 ymm_0 = _mm256_loadu_ps(pSrc);
					//__m256 ymm_1 = _mm256_loadu_ps(pSrc + REG_SIZE);
					//pSrc += 2 * REG_SIZE;

					__m256 ymm_sum = _mm256_setzero_ps();
					for (size_t k = 0; k < 4; ++k)
					{
						__m256 ymm_0_w, ymm_1_w;
						__m256 ymm_s0 = _mm256_fmadd_ps(ymm_0, ymm_hl_w[k][0], _mm256_mul_ps(ymm_1, ymm_hl_w[k][1]));

						__m256 ymm_0_shf = _mm256_permutevar8x32_ps(ymm_0, ymm_mask_temp);
						__m256 ymm_s1 = _mm256_fmadd_ps(ymm_0_shf, ymm_hl_w[k][3], _mm256_mul_ps(ymm_1, ymm_hl_w[k][2]));

						ymm_s0 = _mm256_hadd_ps(ymm_s0, ymm_s1);

						__m256 ymm_1_shf = _mm256_permutevar8x32_ps(ymm_1, ymm_mask_temp);
						ymm_s1 = _mm256_fmadd_ps(ymm_0_shf, ymm_hl_w[k][4], _mm256_mul_ps(ymm_1_shf, ymm_hl_w[k][5]));

						ymm_0_shf = _mm256_permutevar8x32_ps(ymm_0_shf, ymm_mask_temp);
						ymm_0_shf = _mm256_fmadd_ps(ymm_0_shf, ymm_hl_w[k][7], _mm256_mul_ps(ymm_1_shf, ymm_hl_w[k][6]));

						ymm_s1 = _mm256_hadd_ps(ymm_s1, ymm_0_shf);

						ymm_s0 = _mm256_fmadd_ps(ymm_s0, ymm_tanh_w[0], ymm_hl_b[k][0]);
						ymm_s1 = _mm256_fmadd_ps(ymm_s1, ymm_tanh_w[1], ymm_hl_b[k][1]);

						//------------------------------

						ymm_0_w = _mm256_and_ps(ymm14, ymm_s0);
						ymm_1_w = _mm256_add_ps(ymm15, ymm_0_w);
						ymm_0_shf = _mm256_mul_ps(ymm_s0, ymm_s0);

						ymm_1_w = _mm256_add_ps(ymm_1_w, ymm_0_shf);
						ymm_0_shf = _mm256_mul_ps(ymm_0_shf, ymm_0_shf);

						ymm_1_w = _mm256_fmadd_ps(ymm_0_shf, ymm13, ymm_1_w);

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

						ymm_1_w = _mm256_fmadd_ps(ymm_0_shf, ymm13, ymm_1_w);

						ymm_0_shf = _mm256_andnot_ps(ymm14, ymm_s1);

#ifdef USE_FAST_DIV
						ymm_1_w = _mm256_rcp_ps(ymm_1_w);
#else
						ymm_1_w = _mm256_div_ps(ymm15, ymm_1_w);
#endif	

						ymm_1_w = _mm256_sub_ps(ymm15, ymm_1_w);
						ymm_s1 = _mm256_or_ps(ymm_1_w, ymm_0_shf);

						//------------------------------

						ymm_s0 = _mm256_fmadd_ps(ymm_s0, ymm_scale, ymm_scale);
						ymm_s0 = _mm256_fmadd_ps(ymm_s0, ymm_bn_w[k], ymm_bn_b[k]);

						ymm_s1 = _mm256_fmadd_ps(ymm_s1, ymm_scale, ymm_scale);
						ymm_s1 = _mm256_fmadd_ps(ymm_s1, ymm_bn_w[k], ymm_bn_b[k]);

						//------------------------------

						ymm_sum = _mm256_fmadd_ps(ymm_s0, ymm_ol_w[k][0], ymm_sum);
						ymm_sum = _mm256_fmadd_ps(ymm_s1, ymm_ol_w[k][1], ymm_sum);
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
		void CNNPP_v3::tanhW(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict snn_ol_b, float* __restrict tanh_w, float* __restrict scale, size_t L, size_t H, int num_threads)
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

					ymm3 = _mm256_fmadd_ps(ymm5, ymm13, ymm3);

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

#if 0
		inline void print_mm(__m256 ymm, std::string str = "")
		{
			printf("%s ", str.c_str());
			float buff[8];
			_mm256_storeu_ps(buff, ymm);
			for (int i = 0; i < 8; ++i) printf("%f ", buff[i]);
			printf("\n");
		}
		inline void print_mm(__m256i ymm, std::string str = "")
		{
			printf("%s ", str.c_str());
			short int buff[16];
			_mm256_storeu_si256((__m256i*)buff, ymm);
			for (int i = 0; i < 16; ++i) printf("%d ", buff[i]);
			printf("\n");
		}

		__m256i add32(__m256i a, __m256i b, int u)
		{
			__m256i ymm_r11_0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 0));
			__m256i ymm_r11_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 1));

			ymm_r11_0 = _mm256_slli_epi32(ymm_r11_0, u);
			ymm_r11_1 = _mm256_slli_epi32(ymm_r11_1, u);

			ymm_r11_0 = _mm256_add_epi32(ymm_r11_0, b);
			ymm_r11_1 = _mm256_add_epi32(ymm_r11_1, b);

			ymm_r11_0 = _mm256_srai_epi32(ymm_r11_0, u);
			ymm_r11_1 = _mm256_srai_epi32(ymm_r11_1, u);

			return _mm256_permute4x64_epi64(_mm256_hadd_epi16(ymm_r11_0, ymm_r11_1), 216);
		}
		__m256i mul32(__m256i a, __m256i b, int u)
		{
			__m256i ymm_r11_0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 0));
			__m256i ymm_r11_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 1));

			ymm_r11_0 = _mm256_slli_epi32(ymm_r11_0, u);
			ymm_r11_1 = _mm256_slli_epi32(ymm_r11_1, u);

			ymm_r11_0 = _mm256_mul_epi32(ymm_r11_0, b);
			ymm_r11_1 = _mm256_mul_epi32(ymm_r11_1, b);

			ymm_r11_0 = _mm256_srai_epi32(ymm_r11_0, u + 14);
			ymm_r11_1 = _mm256_srai_epi32(ymm_r11_1, u + 14);

			return _mm256_permute4x64_epi64(_mm256_hadd_epi16(ymm_r11_0, ymm_r11_1), 216);
		}

		#undef conv_block
		#define conv_block(k, id)												    \
				ymm_d2 = _mm256_shuffle_ps(ymm_d1, ymm_d1, 177);					\
				ymm_d2 = _mm256_add_ps(ymm_d1, ymm_d2);								\
				ymm_d1 = _mm256_permutevar8x32_ps(ymm_d1, ymm_mask_temp);			\
				ymm_d1 = _mm256_blend_ps(ymm_d1, _mm256_setzero_ps(), 129);			\
				ymm_d1 = _mm256_add_ps(ymm_d1, ymm_d2);								\
				sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_k_1_##id[k], sum_1);			\
				sum_2 = _mm256_fmadd_ps(ymm_d1, ymm_k_2_##id[k], sum_2);					

		void CNNPP_v3::conv_4x4_lrelu_bn_max_old(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
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
						for (size_t i = 0; i < L; i += 2)
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

							//if (i == 0 && j == 0) print_mm(sum_1, "old sum_1");
							//if (i == 0 && j == 0) print_mm(ymm_conv_b, "old ymm_conv_b");

							sum_1 = _mm256_add_ps(sum_1, ymm_conv_b);
							//if (i == 0 && j == 0) print_mm(sum_1, "old sum_1");

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

							//if (i == 0 && j == 0) print_mm(sum_1, "old sum_1");

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
		void CNNPP_v3::conv_3x3_lrelu_bn_max_old(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
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
						for (size_t i = 0; i < L; i += 2)
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
							//if (j == 0 && i == 0) print_mm(sum_1, "old3 sum_1");
							//if (j == 0 && i == 0) print_mm(ymm_conv_b, "old3 ymm_conv_b");
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

							//if (j == 0 && i == 0)
							//{
							//	print_mm(sum_1, "old3 sum_1");
							//}
							//system("pause");

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
		void CNNPP_v3::conv_5x4_lrelu_bn_old(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float* __restrict kernel, float* __restrict conv_b, float* __restrict lrelu_w1, float* __restrict lrelu_w2, float* __restrict bn_w, float* __restrict bn_b, size_t L, size_t H, int num_threads)
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
								//if (j == 0 && i == 0) print_mm(ymm_d1);
								conv_block(k, 0);
								//if (j == 0 && i == 0) print_mm(sum_1);
								//if (j == 0 && i == 0) print_mm(ymm_k_1_0[0]);
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
								//if (j == 0 && i == 0) print_mm(sum_1);

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
								//if (j == 0 && i == 0) print_mm(sum_1);

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
								//if (j == 0 && i == 0) print_mm(sum_1);
							}
#ifndef USE_HF
							pSrc += REG_SIZE;
#else
							pSrc += REG_SIZE / 2;
#endif
							//if (j == 0 && i == 0) print_mm(sum_1);
							//-----------------------------


							//	if (j == 0 && i == 0) print_mm(sum_1, "qq");
							sum_1 = _mm256_add_ps(sum_1, ymm_conv_b_1);
							//if (j == 0 && i == 0) print_mm(sum_1, "qq sum_1");
							//if (j == 0 && i == 0) print_mm(sum_2, "qq sum_2");

							__m256 ymm_d1 = _mm256_max_ps(sum_1, ymm_zero);
							//		if (j == 0 && i == 0) print_mm(ymm_d1);
							//		if (j == 0 && i == 0) print_mm(_mm256_mul_ps(sum_1, ymm_lrelu_w1_1));
							sum_1 = _mm256_fmadd_ps(sum_1, ymm_lrelu_w1_1, ymm_bn_b_1);
							//		if (j == 0 && i == 0) print_mm(sum_1);
							//		if (j == 0 && i == 0) print_mm(_mm256_mul_ps(ymm_d1, ymm_lrelu_w2_1));
							sum_1 = _mm256_fmadd_ps(ymm_d1, ymm_lrelu_w2_1, sum_1);
							//		if (j == 0 && i == 0) print_mm(sum_1);
							sum_2 = _mm256_add_ps(sum_2, ymm_conv_b_2);
							__m256 ymm_d2 = _mm256_max_ps(sum_2, ymm_zero);

							sum_2 = _mm256_fmadd_ps(sum_2, ymm_lrelu_w1_2, ymm_bn_b_2);
							sum_2 = _mm256_fmadd_ps(ymm_d2, ymm_lrelu_w2_2, sum_2);

							//-----------------------------

							if (j == 0 && i == 0)
							{
								//print_mm(sum_1);
								//printf("\n");
								//print_mm(sum_2);
							}

#ifndef USE_HF
							_mm256_store_ps(pDst, sum_1);
							_mm256_store_ps(pDst + REG_SIZE, sum_2);
							pDst += 2 * REG_SIZE;
#else
							__m256i store_si256 = _mm256_set_m128i(_mm256_cvtps_ph(sum_2, 0), _mm256_cvtps_ph(sum_1, 0));
							_mm256_storeu_si256((__m256i*)pDst, store_si256);
							pDst += REG_SIZE;

							//if (j == 0 && i == 0) print_mm(sum_1, "old 5");
							//system("pause");
#endif
						}
					IACA__END
				}
		}
		void CNNPP_v3::mulCN_add_tanhW_add_old(float* __restrict dst, int dst_size_l, float* __restrict src, int src_size_l, int src_size_h, float** __restrict snn_hl_w, float** __restrict snn_hl_b, float* __restrict snn_tanh_w, float* __restrict snn_bn_w, float* __restrict snn_bn_b, float** __restrict snn_ol_w, size_t L, size_t H, int num_threads)
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
#endif

}

#endif
}