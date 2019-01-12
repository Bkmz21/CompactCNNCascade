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


#include "image_proc.h"

#ifdef USE_OMP
#	include <omp.h>
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{

	namespace SIMD
	{
		void ImageConverter::Img8uToImg32f(Image_32f& img_32f, Image_8u& img_8u, int num_threads)
		{
#if defined(USE_SSE) || defined(USE_AVX)
			ALIGN(ALIGN_SSE) const uchar_ set1[16] = { 0, 128, 128, 128, 1, 128, 128, 128, 2, 128, 128, 128, 3, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set2[16] = { 4, 128, 128, 128, 5, 128, 128, 128, 6, 128, 128, 128, 7, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set3[16] = { 8, 128, 128, 128, 9, 128, 128, 128, 10, 128, 128, 128, 11, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set4[16] = { 12, 128, 128, 128, 13, 128, 128, 128, 14, 128, 128, 128, 15, 128, 128, 128 };

			const __m128i xmm11i = _mm_load_si128((__m128i*)set1);
			const __m128i xmm12i = _mm_load_si128((__m128i*)set2);
			const __m128i xmm13i = _mm_load_si128((__m128i*)set3);
			const __m128i xmm14i = _mm_load_si128((__m128i*)set4);
#endif

			OMP_PRAGMA(omp parallel for num_threads(num_threads) schedule(static))
			for (int j = 0; j < img_8u.height; ++j)
			{
				const int imgc_y_offset = j * img_8u.widthStep;
				const int imgg_y_offset = j * img_32f.widthStep;

				uchar_* pSrc = img_8u.data + imgc_y_offset;
				float* pDst = img_32f.data + imgg_y_offset;

				int i = 0;

#if defined(USE_SSE) || (defined(USE_AVX) && !defined(USE_AVX2))
				for (; i <= img_8u.width - 16; i += 16)
				{
					__m128i xmm3i = _mm_loadu_si128((__m128i*)pSrc);
					pSrc += 16;

					__m128i xmm5i = _mm_shuffle_epi8(xmm3i, xmm11i);
					__m128 xmm5 = _mm_cvtepi32_ps(xmm5i);
					__m128i xmm6i = _mm_shuffle_epi8(xmm3i, xmm12i);
					__m128 xmm6 = _mm_cvtepi32_ps(xmm6i);
					__m128i xmm7i = _mm_shuffle_epi8(xmm3i, xmm13i);
					__m128 xmm7 = _mm_cvtepi32_ps(xmm7i);
					__m128i xmm8i = _mm_shuffle_epi8(xmm3i, xmm14i);
					__m128 xmm8 = _mm_cvtepi32_ps(xmm8i);

					_mm_storeu_ps(pDst, xmm5);
					_mm_storeu_ps(pDst + 4, xmm6);
					_mm_storeu_ps(pDst + 8, xmm7);
					_mm_storeu_ps(pDst + 12, xmm8);
					pDst += 16;
				}
#endif

#if defined(USE_AVX2)
				for (; i <= img_8u.width - 16; i += 16)
				{
					__m128i xmmi = _mm_loadu_si128((__m128i*)pSrc);
					pSrc += 16;

					__m256 ymmf0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(xmmi));
					__m256 ymmf1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(xmmi, 78)));

					_mm256_storeu_ps(pDst, ymmf0);
					_mm256_storeu_ps(pDst + 8, ymmf1);
					pDst += 16;
				}
#endif

				for (; i < img_8u.width; ++i)
				{
					*(pDst++) = float(*(pSrc++));
				}
			}
		}
		void ImageConverter::Img8uBGRToImg32fGRAY(Image_32f& img_gray, Image_8u& img_color, int num_threads)
		{
			ALIGN(ALIGN_SSE) const float w[4] = { 0.114f, 0.587f, 0.299f, 0.0f };

#if defined(USE_SSE) || (defined(USE_AVX) && !defined(USE_AVX2))
			ALIGN(ALIGN_SSE) const uchar_ set1[16] = { 0, 128, 128, 128, 1, 128, 128, 128, 2, 128, 128, 128, 128, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set2[16] = { 3, 128, 128, 128, 4, 128, 128, 128, 5, 128, 128, 128, 128, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set3[16] = { 6, 128, 128, 128, 7, 128, 128, 128, 8, 128, 128, 128, 128, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set4[16] = { 9, 128, 128, 128, 10, 128, 128, 128, 11, 128, 128, 128, 128, 128, 128, 128 };

			const __m128 xmm0 = _mm_load_ss(&w[0]);
			const __m128 xmm1 = _mm_load_ss(&w[1]);
			const __m128 xmm2 = _mm_load_ss(&w[2]);

			const __m128 xmm10 = _mm_load_ps(w);
			const __m128i xmm11i = _mm_load_si128((__m128i*)set1);
			const __m128i xmm12i = _mm_load_si128((__m128i*)set2);
			const __m128i xmm13i = _mm_load_si128((__m128i*)set3);
			const __m128i xmm14i = _mm_load_si128((__m128i*)set4);
#endif

#if defined(USE_AVX2)
			const __m128i xmm_mask = { 0, 1, 3, 4, 6, 7, 9, 10, 2, 128, 5, 128, 8, 128, 11, 128 };
			const __m256 ymm_w1 = { 0.114f, 0.587f, 0.114f, 0.587f, 0.114f, 0.587f, 0.114f, 0.587f };
			const __m256 ymm_w2 = { 0.299f, 0.0f, 0.299f, 0.0f, 0.299f, 0.0f, 0.299f, 0.0f };
#endif

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < img_color.height; ++j)
			{
				const int imgc_y_offset = j * img_color.widthStep;
				const int imgg_y_offset = j * img_gray.widthStep;

				uchar_* pSrc = img_color.data + imgc_y_offset;
				float* pDst = img_gray.data + imgg_y_offset;

				int i = 0;

#if defined(USE_SSE) || (defined(USE_AVX) && !defined(USE_AVX2))
				for (; i <= img_color.width - 4; i += 4)
				{
					__m128i xmm3i = _mm_loadu_si128((__m128i*)pSrc);
					pSrc += 3 * 4;

					__m128i xmm5i = _mm_shuffle_epi8(xmm3i, xmm11i);
					__m128 xmm5 = _mm_cvtepi32_ps(xmm5i);
					xmm5 = _mm_mul_ps(xmm5, xmm10);

					__m128i xmm6i = _mm_shuffle_epi8(xmm3i, xmm12i);
					__m128 xmm6 = _mm_cvtepi32_ps(xmm6i);
					xmm6 = _mm_mul_ps(xmm6, xmm10);
					xmm5 = _mm_hadd_ps(xmm5, xmm6);

					xmm6i = _mm_shuffle_epi8(xmm3i, xmm13i);
					xmm6 = _mm_cvtepi32_ps(xmm6i);
					xmm6 = _mm_mul_ps(xmm6, xmm10);

					__m128i xmm7i = _mm_shuffle_epi8(xmm3i, xmm14i);
					__m128 xmm7 = _mm_cvtepi32_ps(xmm7i);
					xmm7 = _mm_mul_ps(xmm7, xmm10);
					xmm6 = _mm_hadd_ps(xmm6, xmm7);

					xmm5 = _mm_hadd_ps(xmm5, xmm6);

					_mm_storeu_ps(pDst, xmm5);
					pDst += 4;
				}
#endif

#if defined(USE_AVX2)
				for (; i <= img_color.width - 4; i += 4)
				{
					__m128i xmm3i = _mm_loadu_si128((__m128i*)pSrc);
					pSrc += 3 * 4;

					xmm3i = _mm_shuffle_epi8(xmm3i, xmm_mask);
					__m256 ymm3i_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(xmm3i));
					__m256 ymm3i_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(xmm3i, 78)));
			
					ymm3i_1 = _mm256_mul_ps(ymm3i_1, ymm_w2);
					ymm3i_0 = _mm256_fmadd_ps(ymm3i_0, ymm_w1, ymm3i_1);

					ymm3i_0 = _mm256_hadd_ps(ymm3i_0, ymm3i_0);
					ymm3i_0 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(ymm3i_0), 216));

					_mm_storeu_ps(pDst, _mm256_extractf128_ps(ymm3i_0, 0));
					pDst += 4;
				}
#endif

				for (; i < img_color.width; ++i)
				{
					const float B = float(*(pSrc++));
					const float G = float(*(pSrc++));
					const float R = float(*(pSrc++));
					*(pDst++) = w[0] * B + w[1] * G + w[2] * R;
				}
			}
		}
		void ImageConverter::Img8uBGRAToImg32fGRAY(Image_32f& img_gray, Image_8u& img_color, int num_threads)
		{
#if defined(USE_SSE) || defined(USE_AVX)
			ALIGN(ALIGN_SSE) const float w[4] = { 0.114f, 0.587f, 0.299f, 0.0f };

			ALIGN(ALIGN_SSE) const uchar_ set1[16] = { 0, 128, 128, 128, 1, 128, 128, 128, 2, 128, 128, 128, 128, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set2[16] = { 4, 128, 128, 128, 5, 128, 128, 128, 6, 128, 128, 128, 128, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set3[16] = { 8, 128, 128, 128, 9, 128, 128, 128, 10, 128, 128, 128, 128, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set4[16] = { 12, 128, 128, 128, 13, 128, 128, 128, 14, 128, 128, 128, 128, 128, 128, 128 };

			const __m128 xmm0 = _mm_load_ss(&w[0]);
			const __m128 xmm1 = _mm_load_ss(&w[1]);
			const __m128 xmm2 = _mm_load_ss(&w[2]);

			const __m128 xmm10 = _mm_load_ps(w);
			const __m128i xmm11i = _mm_load_si128((__m128i*)set1);
			const __m128i xmm12i = _mm_load_si128((__m128i*)set2);
			const __m128i xmm13i = _mm_load_si128((__m128i*)set3);
			const __m128i xmm14i = _mm_load_si128((__m128i*)set4);
#endif

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < img_color.height; ++j)
			{
				const int imgc_y_offset = j * img_color.widthStep;
				const int imgg_y_offset = j * img_gray.widthStep;

				uchar_* pSrc = img_color.data + imgc_y_offset;
				float* pDst = img_gray.data + imgg_y_offset;

				int i = 0;
#if defined(USE_SSE) || defined(USE_AVX)
				for (; i <= img_color.width - 4; i += 4)
				{
					__m128i xmm3i = _mm_loadu_si128((__m128i*)pSrc);
					pSrc += 4 * 4;

					__m128i xmm5i = _mm_shuffle_epi8(xmm3i, xmm11i);
					__m128 xmm5 = _mm_cvtepi32_ps(xmm5i);
					xmm5 = _mm_mul_ps(xmm5, xmm10);

					__m128i xmm6i = _mm_shuffle_epi8(xmm3i, xmm12i);
					__m128 xmm6 = _mm_cvtepi32_ps(xmm6i);
					xmm6 = _mm_mul_ps(xmm6, xmm10);
					xmm5 = _mm_hadd_ps(xmm5, xmm6);

					xmm6i = _mm_shuffle_epi8(xmm3i, xmm13i);
					xmm6 = _mm_cvtepi32_ps(xmm6i);
					xmm6 = _mm_mul_ps(xmm6, xmm10);

					__m128i xmm7i = _mm_shuffle_epi8(xmm3i, xmm14i);
					__m128 xmm7 = _mm_cvtepi32_ps(xmm7i);
					xmm7 = _mm_mul_ps(xmm7, xmm10);
					xmm6 = _mm_hadd_ps(xmm6, xmm7);

					xmm5 = _mm_hadd_ps(xmm5, xmm6);

					_mm_storeu_ps(pDst, xmm5);
					pDst += 4;
				}

				for (; i < img_color.width; ++i)
				{
					__m128 xmm3 = _mm_cvtsi32_ss(xmm0,* (pSrc++));
					xmm3 = _mm_mul_ss(xmm3, xmm0);
					__m128 xmm4 = _mm_cvtsi32_ss(xmm0,* (pSrc++));
#ifndef USE_FMA
					xmm4 = _mm_mul_ss(xmm4, xmm1);
					xmm3 = _mm_add_ss(xmm3, xmm4);
#else
					xmm3 = _mm_fmadd_ss(xmm4, xmm1, xmm3);
#endif
					__m128 xmm5 = _mm_cvtsi32_ss(xmm0,* (pSrc++));
#ifndef USE_FMA
					xmm5 = _mm_mul_ss(xmm5, xmm2);
					xmm3 = _mm_add_ss(xmm3, xmm5);
#else
					xmm3 = _mm_fmadd_ss(xmm5, xmm2, xmm3);
#endif
					_mm_store_ss(pDst++, xmm3);
					pSrc++;
				}
#else
				for (; i < img_color.width; ++i)
				{
					const float B = float(*(pSrc++));
					const float G = float(*(pSrc++));
					const float R = float(*(pSrc++));
					pSrc++;
					*(pDst++) = 0.114f * B + 0.587f * G + 0.299f * R;
				}
#endif
			}
		}

		int ImageConverter::Img8uToImg32fGRAY(Image_32f& img_32f, Image_8u& img_8u, int num_threads)
		{
			switch (img_8u.nChannel)
			{
			case 1:
				Img8uToImg32f(img_32f, img_8u, num_threads);
				break;
			case 3:
				Img8uBGRToImg32fGRAY(img_32f, img_8u, num_threads);
				break;
			case 4:
				Img8uBGRAToImg32fGRAY(img_32f, img_8u, num_threads);
				break;
			default:
				return -1;
			}

			return 0;
		}
		int ImageConverter::Img8uToImg32fGRAY_blur(Image_32f& img_32f, Image_8u& img_8u, const float* kernel_col, const float* kernel_row, int num_threads)
		{
			Img8uToImg32fGRAY(img_32f, img_8u, num_threads);

			rowFilter3_32f(img_32f, img_32f, kernel_row, num_threads);
			colFilter3_32f(img_32f, img_32f, kernel_col, num_threads);

			return 0;
		}

		void ImageConverter::FloatToUChar(Image_8u& dst, Image_32f& src, const Rect& roi)
		{
#if defined(USE_SSE) || defined(USE_AVX)
			ALIGN(ALIGN_SSE) const uchar_ set_byte[16] = { 0, 4, 8, 12, 128, 128, 128, 128, 2, 128, 128, 128, 128, 128, 128, 128 };

			const __m128i xmm2 = _mm_load_si128((__m128i*)set_byte);
			for (int j = 0; j < roi.height; ++j)
			{
				float* pSrc = src.data + (roi.y + j) * src.widthStep + roi.x;
				uchar_* pDst = dst.data + j * dst.widthStep;
				float* pfDst = (float*)pDst;

				int i = 0;
				for (; i <= roi.width - 4; i += 4)
				{
					__m128 xmm0 = _mm_loadu_ps(pSrc + i);
					__m128i xmm1 = _mm_cvtps_epi32(xmm0);
					xmm1 = _mm_shuffle_epi8(xmm1, xmm2);
					__m128 xmm1f = _mm_castsi128_ps(xmm1);
					_mm_store_ss(pfDst++, xmm1f);
				}

				pDst += i;
				for (; i < roi.width; ++i)
				{
					__m128 xmm0 = _mm_load_ss(pSrc + i);
					*pDst++ = (uchar_)_mm_cvt_ss2si(xmm0);
				}
			}
#else
			for (int j = 0; j < roi.height; ++j)
			{
				float* pSrc = src.data + (roi.y + j) * src.widthStep + roi.x;
				uchar_* pDst = dst.data + j * dst.widthStep;

				for (int i = 0; i < roi.width; ++i)
				{
					*pDst++ = static_cast<uchar_>(*pSrc++ + 0.5f);
				}
			}
#endif
		}

		void ImageConverter::UCharToFloat(Image_32f& dst, Image_8u& src, int offset)
		{
#if defined(USE_SSE) || defined(USE_AVX)
			ALIGN(ALIGN_SSE) const uchar_ set1[16] = { 0, 128, 128, 128, 1, 128, 128, 128, 2, 128, 128, 128, 3, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set2[16] = { 4, 128, 128, 128, 5, 128, 128, 128, 6, 128, 128, 128, 7, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set3[16] = { 8, 128, 128, 128, 9, 128, 128, 128, 10, 128, 128, 128, 11, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set4[16] = { 12, 128, 128, 128, 13, 128, 128, 128, 14, 128, 128, 128, 15, 128, 128, 128 };

			const __m128i xmm11i = _mm_load_si128((__m128i*)set1);
			const __m128i xmm12i = _mm_load_si128((__m128i*)set2);
			const __m128i xmm13i = _mm_load_si128((__m128i*)set3);
			const __m128i xmm14i = _mm_load_si128((__m128i*)set4);

			for (int j = 0; j < dst.height; ++j)
			{
				uchar_* pSrc = src.data + j * src.widthStep + offset;
				float* pDst = dst.data + j * dst.widthStep;

				int i = 0;
				for (; i <= dst.width - 16; i += 16)
				{
					__m128i xmm3i = _mm_loadu_si128((__m128i*)pSrc);
					pSrc += 16;

					__m128i xmm5i = _mm_shuffle_epi8(xmm3i, xmm11i);
					__m128 xmm5 = _mm_cvtepi32_ps(xmm5i);
					__m128i xmm6i = _mm_shuffle_epi8(xmm3i, xmm12i);
					__m128 xmm6 = _mm_cvtepi32_ps(xmm6i);
					__m128i xmm7i = _mm_shuffle_epi8(xmm3i, xmm13i);
					__m128 xmm7 = _mm_cvtepi32_ps(xmm7i);
					__m128i xmm8i = _mm_shuffle_epi8(xmm3i, xmm14i);
					__m128 xmm8 = _mm_cvtepi32_ps(xmm8i);

					_mm_storeu_ps(pDst, xmm5);
					_mm_storeu_ps(pDst + 4, xmm6);
					_mm_storeu_ps(pDst + 8, xmm7);
					_mm_storeu_ps(pDst + 12, xmm8);
					pDst += 16;
				}

				for (; i < dst.width; ++i)
				{
					*pDst++ = static_cast<float>(*pSrc++);
				}
			}
#else
			for (int j = 0; j < dst.height; ++j)
			{
				uchar_* pSrc = src.data + j * src.widthStep + offset;
				float* pDst = dst.data + j * dst.widthStep;

				for (int i = 0; i < dst.width; ++i)
				{
					*pDst++ = static_cast<float>(*pSrc++);
				}
			}
#endif
		}
		void ImageConverter::UCharToFloat_inv(Image_32f& dst, Image_8u& src)
		{
#if defined(USE_SSE) || defined(USE_AVX)
			ALIGN(ALIGN_SSE) const uchar_ set1[16] = { 0, 128, 128, 128, 1, 128, 128, 128, 2, 128, 128, 128, 3, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set2[16] = { 4, 128, 128, 128, 5, 128, 128, 128, 6, 128, 128, 128, 7, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set3[16] = { 8, 128, 128, 128, 9, 128, 128, 128, 10, 128, 128, 128, 11, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set4[16] = { 12, 128, 128, 128, 13, 128, 128, 128, 14, 128, 128, 128, 15, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set_inv[16] = { 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };

			const __m128i xmm11i = _mm_load_si128((__m128i*)set1);
			const __m128i xmm12i = _mm_load_si128((__m128i*)set2);
			const __m128i xmm13i = _mm_load_si128((__m128i*)set3);
			const __m128i xmm14i = _mm_load_si128((__m128i*)set4);
			const __m128i xmm15i = _mm_load_si128((__m128i*)set_inv);

			const int offset = dst.widthStep - dst.width;
			for (int j = 0; j < dst.height; ++j)
			{
				uchar_* pSrc = src.data + j * src.widthStep;
				float* pDst = dst.data + (j + 1) * dst.widthStep - offset;

				int i = 0;
				for (; i <= dst.width - 16; i += 16)
				{
					__m128i xmm3i = _mm_loadu_si128((__m128i*)pSrc);
					pSrc += 16;

					xmm3i = _mm_shuffle_epi8(xmm3i, xmm15i);
					__m128i xmm5i = _mm_shuffle_epi8(xmm3i, xmm11i);
					__m128 xmm5 = _mm_cvtepi32_ps(xmm5i);
					__m128i xmm6i = _mm_shuffle_epi8(xmm3i, xmm12i);
					__m128 xmm6 = _mm_cvtepi32_ps(xmm6i);
					__m128i xmm7i = _mm_shuffle_epi8(xmm3i, xmm13i);
					__m128 xmm7 = _mm_cvtepi32_ps(xmm7i);
					__m128i xmm8i = _mm_shuffle_epi8(xmm3i, xmm14i);
					__m128 xmm8 = _mm_cvtepi32_ps(xmm8i);

					_mm_storeu_ps(pDst - 16, xmm5);
					_mm_storeu_ps(pDst - 12, xmm6);
					_mm_storeu_ps(pDst - 8, xmm7);
					_mm_storeu_ps(pDst - 4, xmm8);
					pDst -= 16;
				}

				for (; i < dst.width; ++i)
				{
					*(--pDst) = static_cast<float>(*pSrc++);
				}
			}
#else
			const int offset = dst.widthStep - dst.width;
			for (int j = 0; j < dst.height; ++j)
			{
				uchar_* pSrc = src.data + j * src.widthStep;
				float* pDst = dst.data + (j + 1) * dst.widthStep - offset;

				for (int i = 0; i < dst.width; ++i)
				{
					*(--pDst) = static_cast<float>(*pSrc++);
				}
			}
#endif
		}
		void ImageConverter::UCharToFloat_add_rnd(Image_32f& dst, Image_8u& src, TmpImage<char>& rnd_matrix, int offset)
		{
#if defined(USE_SSE) || defined(USE_AVX)
			ALIGN(ALIGN_SSE) const uchar_ set1[16] = { 0, 128, 128, 128, 1, 128, 128, 128, 2, 128, 128, 128, 3, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set2[16] = { 4, 128, 128, 128, 5, 128, 128, 128, 6, 128, 128, 128, 7, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set3[16] = { 8, 128, 128, 128, 9, 128, 128, 128, 10, 128, 128, 128, 11, 128, 128, 128 };
			ALIGN(ALIGN_SSE) const uchar_ set4[16] = { 12, 128, 128, 128, 13, 128, 128, 128, 14, 128, 128, 128, 15, 128, 128, 128 };

			const __m128i xmm11i = _mm_load_si128((__m128i*)set1);
			const __m128i xmm12i = _mm_load_si128((__m128i*)set2);
			const __m128i xmm13i = _mm_load_si128((__m128i*)set3);
			const __m128i xmm14i = _mm_load_si128((__m128i*)set4);

			for (int j = 0; j < dst.height; ++j)
			{
				uchar_* pSrc = src.data + j * src.widthStep + offset;
				char* pRndM = rnd_matrix.data + j * rnd_matrix.widthStep + offset;
				float* pDst = dst.data + j * dst.widthStep;

				int i = 0;
				for (; i <= dst.width - 16; i += 16)
				{
					__m128i xmm3i = _mm_loadu_si128((__m128i*)pSrc);
					pSrc += 16;

					__m128i xmm0i = _mm_loadu_si128((__m128i*)pRndM);
					xmm3i = _mm_add_epi8(xmm3i, xmm0i);
					pRndM += 16;

					__m128i xmm5i = _mm_shuffle_epi8(xmm3i, xmm11i);
					__m128 xmm5 = _mm_cvtepi32_ps(xmm5i);
					__m128i xmm6i = _mm_shuffle_epi8(xmm3i, xmm12i);
					__m128 xmm6 = _mm_cvtepi32_ps(xmm6i);
					__m128i xmm7i = _mm_shuffle_epi8(xmm3i, xmm13i);
					__m128 xmm7 = _mm_cvtepi32_ps(xmm7i);
					__m128i xmm8i = _mm_shuffle_epi8(xmm3i, xmm14i);
					__m128 xmm8 = _mm_cvtepi32_ps(xmm8i);

					_mm_storeu_ps(pDst, xmm5);
					_mm_storeu_ps(pDst + 4, xmm6);
					_mm_storeu_ps(pDst + 8, xmm7);
					_mm_storeu_ps(pDst + 12, xmm8);
					pDst += 16;
				}

				for (; i < dst.width; ++i)
				{
					*pDst++ = static_cast<float>(uchar_(*pSrc++ +* pRndM++));
				}
			}
#else
			for (int j = 0; j < dst.height; ++j)
			{
				uchar_* pSrc = src.data + j * src.widthStep + offset;
				char* pRndM = rnd_matrix.data + j * rnd_matrix.widthStep + offset;
				float* pDst = dst.data + j * dst.widthStep;

				for (int i = 0; i < dst.width; ++i)
				{
					*pDst++ = static_cast<float>(uchar_(*pSrc++ +* pRndM++));
				}
			}
#endif
		}

		void colFilter3_32f(Image_32f& dst, Image_32f& src, const float* kernel, int num_threads)
		{
			const int L = src.width;
			const int H = src.height - 2;

			const int srcStep = src.widthStep;
			const int dstStep = dst.widthStep;

#if defined(USE_AVX)
			const __m256 ymm13 = _mm256_broadcast_ss(&kernel[0]);
			const __m256 ymm14 = _mm256_broadcast_ss(&kernel[1]);
			const __m256 ymm15 = _mm256_broadcast_ss(&kernel[2]);
#else
#	ifdef USE_SSE
			const __m128 ymm13 = _mm_broadcast_ss(&kernel[0]);
			const __m128 ymm14 = _mm_broadcast_ss(&kernel[1]);
			const __m128 ymm15 = _mm_broadcast_ss(&kernel[2]);
#	endif
#endif

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc0 = src.data + j * srcStep;
				float* __restrict pSrc1 = src.data + (j + 1) * srcStep;
				float* __restrict pSrc2 = src.data + (j + 2) * srcStep;
				float* pDst = dst.data + j * dstStep;

				int i = 0;
#if defined(USE_SSE) || defined(USE_AVX)
				for (; i <= L - REG_SIZE; i += REG_SIZE)
				{
#if defined(USE_AVX)
					__m256 ymm0 = _mm256_loadu_ps(pSrc0);
					__m256 ymm1 = _mm256_loadu_ps(pSrc1);
					__m256 ymm2 = _mm256_loadu_ps(pSrc2);
					pSrc0 += REG_SIZE;
					pSrc1 += REG_SIZE;
					pSrc2 += REG_SIZE;

#	ifndef USE_FMA
					ymm0 = _mm256_mul_ps(ymm0, ymm13);
					ymm1 = _mm256_mul_ps(ymm1, ymm14);
					ymm0 = _mm256_add_ps(ymm0, ymm1);
					ymm2 = _mm256_mul_ps(ymm2, ymm15);
					ymm0 = _mm256_add_ps(ymm0, ymm2);
#	else
					ymm0 = _mm256_mul_ps(ymm0, ymm13);
					ymm0 = _mm256_fmadd_ps(ymm1, ymm14, ymm0);
					ymm0 = _mm256_fmadd_ps(ymm2, ymm15, ymm0);
#	endif

					_mm256_storeu_ps(pDst, ymm0);
					pDst += REG_SIZE;
#else
#	ifdef USE_SSE
					__m128 ymm0 = _mm_loadu_ps(pSrc0);
					__m128 ymm1 = _mm_loadu_ps(pSrc1);
					__m128 ymm2 = _mm_loadu_ps(pSrc2);
					pSrc0 += REG_SIZE;
					pSrc1 += REG_SIZE;
					pSrc2 += REG_SIZE;

#		ifndef USE_FMA
					ymm0 = _mm_mul_ps(ymm0, ymm13);
					ymm1 = _mm_mul_ps(ymm1, ymm14);
					ymm0 = _mm_add_ps(ymm0, ymm1);
					ymm2 = _mm_mul_ps(ymm2, ymm15);
					ymm0 = _mm_add_ps(ymm0, ymm2);
#		else
					ymm0 = _mm_mul_ps(ymm0, ymm13);
					ymm0 = _mm_fmadd_ps(ymm1, ymm14, ymm0);
					ymm0 = _mm_fmadd_ps(ymm2, ymm15, ymm0);
#		endif

					_mm_storeu_ps(pDst, ymm0);
					pDst += REG_SIZE;
#	endif
#endif
				}
#endif

				for (; i < L; ++i)
				{
					*(pDst++) = *(pSrc0++)* * kernel +* (pSrc1++)* * (kernel + 1) +* (pSrc2++)* * (kernel + 2);
				}
			}
		}
		void rowFilter3_32f(Image_32f& dst, Image_32f& src, const float* kernel, int num_threads)
		{
			const int L = src.width + 2 < src.widthStep ? src.width : src.width - 2;
			const int H = src.height;

			const int srcStep = src.widthStep;
			const int dstStep = dst.widthStep;

#if defined(USE_AVX)
			const __m256 ymm15 = _mm256_load_ps(kernel);
#else
#	ifdef USE_SSE
			const __m128 ymm15 = _mm_load_ps(kernel);
#	endif
#endif

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int j = 0; j < H; ++j)
			{
				float* __restrict pSrc = src.data + j * srcStep;
				float* pDst = dst.data + j * dstStep;

				int i = 0;
#if defined(USE_SSE) || defined(USE_AVX)
				for (; i <= L - REG_SIZE; i += REG_SIZE)
				{
#if defined(USE_AVX)
					__m256 ymm0 = _mm256_loadu_ps(pSrc);
					__m256 ymm1 = _mm256_loadu_ps(pSrc + 1);
					__m256 ymm2 = _mm256_loadu_ps(pSrc + 2);
					__m256 ymm3 = _mm256_loadu_ps(pSrc + 3);
					pSrc += REG_SIZE;

					ymm0 = _mm256_dp_ps(ymm0, ymm15, 241);
					ymm1 = _mm256_dp_ps(ymm1, ymm15, 242);
					ymm2 = _mm256_dp_ps(ymm2, ymm15, 244);
					ymm3 = _mm256_dp_ps(ymm3, ymm15, 248);

					ymm0 = _mm256_blend_ps(ymm0, ymm1, 34);
					ymm0 = _mm256_blend_ps(ymm0, ymm2, 68);
					ymm0 = _mm256_blend_ps(ymm0, ymm3, 136);

					_mm256_storeu_ps(pDst, ymm0);
					pDst += REG_SIZE;
#else
#	ifdef USE_SSE
					__m128 ymm0 = _mm_loadu_ps(pSrc);
					__m128 ymm1 = _mm_loadu_ps(pSrc + 1);
					__m128 ymm2 = _mm_loadu_ps(pSrc + 2);
					__m128 ymm3 = _mm_loadu_ps(pSrc + 3);
					pSrc += REG_SIZE;

					ymm0 = _mm_dp_ps(ymm0, ymm15, 241);
					ymm1 = _mm_dp_ps(ymm1, ymm15, 242);
					ymm2 = _mm_dp_ps(ymm2, ymm15, 244);
					ymm3 = _mm_dp_ps(ymm3, ymm15, 248);

					ymm0 = _mm_blend_ps(ymm0, ymm1, 34);
					ymm0 = _mm_blend_ps(ymm0, ymm2, 68);
					ymm0 = _mm_blend_ps(ymm0, ymm3, 136);

					_mm_storeu_ps(pDst, ymm0);
					pDst += REG_SIZE;
#	endif
#endif
				}
#endif

				for (; i < L; ++i)
				{
					*(pDst++) = *pSrc* * kernel +* (pSrc + 1)* * (kernel + 1) +* (pSrc++ + 2)* * (kernel + 2);
				}
			}
		}

		void equalizeImage(Image_8u& img)
		{
			ALIGN(ALIGN_SSE) int hist[256];
			for (int k = 0; k < 256; k += 4)
			{
#if defined(USE_SSE) || defined(USE_AVX)
				__m128i ymm0 = _mm_load_si128((__m128i*)(hist + k));
				ymm0 = _mm_xor_si128(ymm0, ymm0);
				_mm_store_si128((__m128i*)(hist + k), ymm0);
#else
				hist[k] = 0;
				hist[k + 1] = 0;
				hist[k + 2] = 0;
				hist[k + 3] = 0;
#endif
			}

			for (int j = 0; j < img.height; ++j)
			{
				uchar_* pData = img.data + j * img.widthStep;
				for (int i = 0; i < img.width; ++i)
				{
					hist[*pData++]++;
				}
			}

			ALIGN(ALIGN_SSE) int cf_hist[256];
			cf_hist[0] = hist[0];
			for (int k = 1; k < 256; k += 5)
			{
				cf_hist[k] = cf_hist[k - 1] + hist[k];
				cf_hist[k + 1] = cf_hist[k + 1 - 1] + hist[k + 1];
				cf_hist[k + 2] = cf_hist[k + 2 - 1] + hist[k + 2];
				cf_hist[k + 3] = cf_hist[k + 3 - 1] + hist[k + 3];
				cf_hist[k + 4] = cf_hist[k + 4 - 1] + hist[k + 4];
			}

			const float b_min_f = (float)cf_hist[0];
			const float b_max_f = (float)cf_hist[255];
			if (b_min_f != b_max_f)
			{
				const float delt_b_f = 255.f / (b_max_f - b_min_f);

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX2)
				const __m128 ymm0f = _mm_set1_ps(b_min_f);
				const __m128 ymm2f = _mm_set1_ps(delt_b_f);
#else
				float ymm0f[4], ymm2f[4];
				for (int i = 0; i < 4; ++i) {
					ymm0f[i] = b_min_f;
					ymm2f[i] = delt_b_f;
				}
#endif
				for (int k = 0; k < 256; k += 4)
				{
#if defined(USE_SSE) || defined(USE_AVX)
					__m128i ymm3 = _mm_load_si128((__m128i*)(cf_hist + k));
					__m128 ymm3f = _mm_cvtepi32_ps(ymm3);
					ymm3f = _mm_sub_ps(ymm3f, ymm0f);
					ymm3f = _mm_mul_ps(ymm3f, ymm2f);
					ymm3 = _mm_cvtps_epi32(ymm3f);
					_mm_store_si128((__m128i*)(cf_hist + k), ymm3);
#else
					cf_hist[k] = int((float(cf_hist[k]) - b_min_f) * delt_b_f + 0.5f);
					cf_hist[k + 1] = int((float(cf_hist[k + 1]) - b_min_f) * delt_b_f + 0.5f);
					cf_hist[k + 2] = int((float(cf_hist[k + 2]) - b_min_f) * delt_b_f + 0.5f);
					cf_hist[k + 3] = int((float(cf_hist[k + 3]) - b_min_f) * delt_b_f + 0.5f);
#endif
				}

				for (int j = 0; j < img.height; ++j)
				{
					uchar_* pData = img.data + j * img.widthStep;
					for (int i = 0; i < img.width; ++i)
					{
						*pData = static_cast<uchar_>(cf_hist[*pData]);
						pData++;
					}
				}
			}
		}

		/*
		inline void ImgRGB8uToImgSplit32f(Image_32f* img_32f, Image_8u& img_8u, int num_threads = 1)
		{
		#ifdef USE_OMP
		#pragma omp parallel for num_threads(num_threads)
		#endif
		for (int t = 0; t < 3; ++t)
		#ifdef USE_CVCORE_PPL
		Core::Parallel::For(0, 3, [&](int t)
		#endif
		{
		for (int j = 0; j < img_8u.height; ++j)
		{
		const int imgc_y_offset = j * img_8u.widthStep;
		const int imgg_y_offset = j * img_32f[t].widthStep;

		uchar_* pSrc = img_8u.data + imgc_y_offset;
		float* pDst = img_32f[t].data + imgg_y_offset;

		for (int i = 0; i < img_8u.width; ++i)
		{
		*(pDst + i) = float(*(pSrc + 3 * i + t));
		}
		}
		}
		#ifdef USE_CVCORE_PPL
		);
		#endif
		}
		*/

		/*
		void Img8uBGRToImg32fGRAY_asm(float* gray_data, uchar_* color_data, int64_ _size)
		{
		static float w[4] = { 0.114f, 0.587f, 0.299f, 0.0f };

		static char set1[16] = { 0, 128, 128, 128, 1, 128, 128, 128, 2, 128, 128, 128, 128, 128, 128, 128 };
		static char set2[16] = { 3, 128, 128, 128, 4, 128, 128, 128, 5, 128, 128, 128, 128, 128, 128, 128 };
		static char set3[16] = { 6, 128, 128, 128, 7, 128, 128, 128, 8, 128, 128, 128, 128, 128, 128, 128 };
		static char set4[16] = { 9, 128, 128, 128, 10, 128, 128, 128, 11, 128, 128, 128, 128, 128, 128, 128 };

		int64_ j = (int)ceilf((float)_size / 4.0f) - 1;

		_asm
		{
		mov r9, j;

		mov r10, qword ptr[color_data];
		mov r11, qword ptr[gray_data];

		movups xmm10, w;
		movups xmm11, set1;
		movups xmm12, set2;
		movups xmm13, set3;
		movups xmm14, set4;

		LOOP1:
		movups xmm3, xmmword ptr[r10];

		movaps xmm5, xmm3;
		pshufb xmm5, xmm11;
		cvtdq2ps xmm5, xmm5;
		mulps xmm5, xmm10;

		movaps xmm6, xmm3;
		pshufb xmm6, xmm12;
		cvtdq2ps xmm6, xmm6;
		mulps xmm6, xmm10;
		haddps xmm5, xmm6;


		movaps xmm6, xmm3;
		pshufb xmm6, xmm13;
		cvtdq2ps xmm6, xmm6;
		mulps xmm6, xmm10;

		movaps xmm7, xmm3;
		pshufb xmm7, xmm14;
		cvtdq2ps xmm7, xmm7;
		mulps xmm7, xmm10;
		haddps xmm6, xmm7;

		haddps xmm5, xmm6;

		movups[r11], xmm5;

		add r10, 12;
		add r11, 16;

		dec r9;
		jne LOOP1;
		}

		int64_ k = _size - 4 * j;
		if (k > 0)
		{
		static float a = 0.114f;
		static float b = 0.587f;
		static float c = 0.299f;

		_size -= k;

		_asm
		{
		movss xmm0, a;
		movss xmm1, b;
		movss xmm2, c;

		mov r9, _size;
		mov r12, k;

		mov r10, qword ptr[color_data];
		mov r11, qword ptr[gray_data];
		mov r14, r9;
		imul r14, 0x3;
		add r10, r14;
		mov r14, r9;
		imul r14, 0x4;
		add r11, r14;

		xor r13, r13;
		xor r14, r14;
		xor r15, r15;

		LOOP2:
		mov r13B, byte ptr[r10];
		mov r14B, byte ptr[r10 + 1];
		mov r15B, byte ptr[r10 + 2];

		cvtsi2ss xmm3, r13;
		mulss xmm3, xmm0;
		cvtsi2ss xmm4, r14;
		mulss xmm4, xmm1;
		addss xmm3, xmm4;
		cvtsi2ss xmm5, r15;
		mulss xmm5, xmm2;
		addss xmm3, xmm5;

		movss[r11], xmm3;

		add r10, 3;
		add r11, 4;

		dec r12;
		jne LOOP2;
		}
		}
		}
		*/

		/*
		inline void FloatToUChar_add_rnd(uchar_* uc_data, float* f_data, __int64 _size, float* rnd_matrix)
		{
		__int64 j = (int)ceilf((float)_size / 4.0f) - 1;

		static char set_byte[16] = { 0, 4, 8, 12, 128, 128, 128, 128, 2, 128, 128, 128, 128, 128, 128, 128 };

		_asm
		{
		mov r9, j;

		mov r10, qword ptr[f_data];
		mov r11, qword ptr[uc_data];
		mov r8, qword ptr[rnd_matrix];

		movups xmm2, set_byte;

		LOOP1:
		movups xmm0, xmmword ptr[r10];
		movups xmm5, xmmword ptr[r8];

		addps xmm0, xmm5;
		cvtps2dq xmm1, xmm0;
		pshufb xmm1, xmm2;
		movss word ptr[r11], xmm1;

		add r10, 16;
		add r11, 4;
		add r8, 16;

		dec r9;
		jne LOOP1;
		}

		__int64 k = _size - 4 * j;
		if (k > 0)
		{
		_size -= k;

		_asm
		{
		mov r9, _size;
		mov r12, k;

		mov r10, qword ptr[f_data];
		mov r11, qword ptr[uc_data];
		mov r8, qword ptr[rnd_matrix];

		mov r14, r9;
		add r11, r14;
		mov r14, r9;
		imul r14, 0x4;
		add r10, r14;
		add r8, r14;

		xor r13, r13;

		LOOP2:
		movss xmm0, xmmword ptr[r10];
		movss xmm5, xmmword ptr[r8];

		addss xmm0, xmm5;
		cvtss2si r13, xmm0;
		mov byte ptr[r11], r13B;

		add r10, 4;
		add r11, 1;
		add r8, 4;

		dec r12;
		jne LOOP2;
		}
		}
		}
		inline void FloatToUChar(uchar_* uc_data, float* f_data, __int64 _size)
		{
		__int64 j = (int)ceilf((float)_size / 4.0f) - 1;

		static char set_byte[16] = { 0, 4, 8, 12, 128, 128, 128, 128, 2, 128, 128, 128, 128, 128, 128, 128 };

		_asm
		{
		mov r9, j;

		mov r10, qword ptr[f_data];
		mov r11, qword ptr[uc_data];

		movups xmm2, set_byte;

		LOOP1:
		movups xmm0, xmmword ptr[r10];

		cvtps2dq xmm1, xmm0;
		pshufb xmm1, xmm2;
		movss word ptr[r11], xmm1;

		add r10, 16;
		add r11, 4;
		add r8, 16;

		dec r9;
		jne LOOP1;
		}

		__int64 k = _size - 4 * j;
		if (k > 0)
		{
		_size -= k;

		_asm
		{
		mov r9, _size;
		mov r12, k;

		mov r10, qword ptr[f_data];
		mov r11, qword ptr[uc_data];

		mov r14, r9;
		add r11, r14;
		mov r14, r9;
		imul r14, 0x4;
		add r10, r14;

		xor r13, r13;

		LOOP2:
		movss xmm0, xmmword ptr[r10];

		cvtss2si r13, xmm0;
		mov byte ptr[r11], r13B;

		add r10, 4;
		add r11, 1;
		add r8, 4;

		dec r12;
		jne LOOP2;
		}
		}
		}
		inline void UCharToFloat(float* f_data, uchar_* uc_data, __int64 _size)
		{
		static char set1[16] = { 0, 128, 128, 128, 1, 128, 128, 128, 2, 128, 128, 128, 3, 128, 128, 128 };
		static char set2[16] = { 4, 128, 128, 128, 5, 128, 128, 128, 6, 128, 128, 128, 7, 128, 128, 128 };
		static char set3[16] = { 8, 128, 128, 128, 9, 128, 128, 128, 10, 128, 128, 128, 11, 128, 128, 128 };
		static char set4[16] = { 12, 128, 128, 128, 13, 128, 128, 128, 14, 128, 128, 128, 15, 128, 128, 128 };

		__int64 j = (int)ceilf((float)_size / 16.0f) - 1;

		_asm
		{
		mov r9, j;

		mov r10, qword ptr[uc_data];
		mov r11, qword ptr[f_data];

		movups xmm11, set1;
		movups xmm12, set2;
		movups xmm13, set3;
		movups xmm14, set4;

		LOOP1:
		movups xmm3, xmmword ptr[r10];

		movaps xmm5, xmm3;
		pshufb xmm5, xmm11;
		cvtdq2ps xmm5, xmm5;

		movaps xmm6, xmm3;
		pshufb xmm6, xmm12;
		cvtdq2ps xmm6, xmm6;

		movaps xmm7, xmm3;
		pshufb xmm7, xmm13;
		cvtdq2ps xmm7, xmm7;

		movaps xmm8, xmm3;
		pshufb xmm8, xmm14;
		cvtdq2ps xmm8, xmm8;

		movups[r11], xmm5;
		movups[r11 + 16], xmm6;
		movups[r11 + 32], xmm7;
		movups[r11 + 48], xmm8;

		add r10, 16;
		add r11, 64;

		dec r9;
		jne LOOP1;
		}

		__int64 k = _size - 16 * j;
		if (k > 0)
		{
		_size -= k;

		_asm
		{
		mov r9, _size;
		mov r12, k;

		mov r10, qword ptr[uc_data];
		mov r11, qword ptr[f_data];
		mov r14, r9;
		add r10, r14;
		mov r14, r9;
		imul r14, 0x4;
		add r11, r14;

		xor r13, r13;

		LOOP2:
		mov r13B, byte ptr[r10];

		cvtsi2ss xmm3, r13;
		movss[r11], xmm3;

		add r10, 1;
		add r11, 4;

		dec r12;
		jne LOOP2;
		}
		}
		}
		inline void UCharToFloat_inv(float* f_data, uchar_* uc_data, __int64 _size)
		{
		static char set1[16] = { 0, 128, 128, 128, 1, 128, 128, 128, 2, 128, 128, 128, 3, 128, 128, 128 };
		static char set2[16] = { 4, 128, 128, 128, 5, 128, 128, 128, 6, 128, 128, 128, 7, 128, 128, 128 };
		static char set3[16] = { 8, 128, 128, 128, 9, 128, 128, 128, 10, 128, 128, 128, 11, 128, 128, 128 };
		static char set4[16] = { 12, 128, 128, 128, 13, 128, 128, 128, 14, 128, 128, 128, 15, 128, 128, 128 };
		static char set_inv[16] = { 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };

		__int64 j = (int)ceilf((float)_size / 16.0f) - 1;

		_asm
		{
		mov r9, j;

		mov r10, qword ptr[uc_data];
		mov r11, qword ptr[f_data];

		movups xmm11, set1;
		movups xmm12, set2;
		movups xmm13, set3;
		movups xmm14, set4;
		movups xmm15, set_inv;

		LOOP1:
		sub r11, 64;

		movups xmm3, xmmword ptr[r10];
		pshufb xmm3, xmm15;

		movaps xmm5, xmm3;
		pshufb xmm5, xmm11;
		cvtdq2ps xmm5, xmm5;

		movaps xmm6, xmm3;
		pshufb xmm6, xmm12;
		cvtdq2ps xmm6, xmm6;

		movaps xmm7, xmm3;
		pshufb xmm7, xmm13;
		cvtdq2ps xmm7, xmm7;

		movaps xmm8, xmm3;
		pshufb xmm8, xmm14;
		cvtdq2ps xmm8, xmm8;

		movups[r11], xmm5;
		movups[r11 + 16], xmm6;
		movups[r11 + 32], xmm7;
		movups[r11 + 48], xmm8;

		add r10, 16;

		dec r9;
		jne LOOP1;
		}

		__int64 t = 16 * j;
		__int64 k = _size - t;
		if (k > 0)
		{
		_size -= k;

		_asm
		{
		mov r9, _size;
		mov r12, k;
		mov r13, t;

		mov r10, qword ptr[uc_data];
		mov r11, qword ptr[f_data];
		mov r14, r9;
		add r10, r14;
		mov r14, r13;
		imul r14, 0x4;
		sub r11, r14;

		xor r13, r13;

		LOOP2:
		sub r11, 4;

		mov r13B, byte ptr[r10];

		cvtsi2ss xmm3, r13;
		movss[r11], xmm3;

		add r10, 1;

		dec r12;
		jne LOOP2;
		}
		}
		}
		*/
	}

}