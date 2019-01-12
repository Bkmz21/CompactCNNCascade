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


#include "image_resize.h"

#include <cmath>

#if defined(USE_SSE) || defined(USE_AVX)
#	include <immintrin.h>
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{

	namespace SIMD
	{
		ImageResizer::ImageResizer() { }
		ImageResizer::ImageResizer(Size _dst_img_size, Size _src_img_size)
		{
			src_img_size = _src_img_size;
			dst_img_size = _dst_img_size;
			preprocessing();
		}

		void ImageResizer::preprocessing()
		{
			uint_ ix, iy;
			uint_ dst_img_width = dst_img_size.width;
			uint_ dst_img_height = dst_img_size.height;

			const uint_ QUANT_BIT = 12;	//8~12
			const uint_ QUANT = (1 << QUANT_BIT);
			const uint_ QUANT_BIT2 = 2 * QUANT_BIT;
			const float QUANT_BIT2_f32 = powf(2.0f, (float)QUANT_BIT2);

			const uint_ sclx = (src_img_size.width << QUANT_BIT) / dst_img_size.width + 1;
			const uint_ scly = (src_img_size.height << QUANT_BIT) / dst_img_size.height + 1;

			// allocate small-array on stack
			if (pyLine.isEmpty()) pyLine = Array_32u(dst_img_height, ALIGN_SSE);
			if (ayLUT.isEmpty()) ayLUT = Array_32f(dst_img_height * 2, ALIGN_SSE);

			// make LUT for y-axis
			uint_* p_pyLine = pyLine();
			float* p_ayLUT = ayLUT();
			for (iy = 0; iy < (uint_)dst_img_height; ++iy)
			{
				const uint_ y = iy * scly;
				const uint_ py = (y >> QUANT_BIT);
				const float fy = float(y - (py << QUANT_BIT));
				const float cy = float(QUANT - fy);

				*p_pyLine++ = MIN(py, src_img_size.height - 1);
				*p_ayLUT++ = fy / QUANT_BIT2_f32;
				*p_ayLUT++ = cy / QUANT_BIT2_f32;
			}

			// allocate small-array on stack
			if (pxLine.isEmpty()) pxLine = Array_32u(dst_img_width, ALIGN_SSE);
			if (axLUT.isEmpty()) axLUT = Array_32f(dst_img_width * 2, ALIGN_SSE);

			// make LUT for x-axis
			uint_* p_pxLine = pxLine();
			float* p_axLUT = axLUT();
			float* p_axLUT2 = axLUT() + dst_img_width;
			for (ix = 0; ix < (uint_)dst_img_width; ++ix)
			{
				const uint_ x = ix * sclx;
				const uint_ px = (x >> QUANT_BIT);
				const float fx = float(x - (px << QUANT_BIT));
				const float cx = float(QUANT - fx);
	
				*p_pxLine++ = MIN(px, src_img_size.width - 2);
				*p_axLUT++ = fx;
				*p_axLUT2++ = cx;
			}

//#if defined(USE_SSE) || (defined(USE_AVX) && !defined(USE_AVX2))
//			//only for SSE implementation
//			p_pxLine = pxLine();
//			p_axLUT = axLUT();
//			p_axLUT2 = axLUT() + dst_img_width;
//			for (ix = 0; ix <= (uint_)dst_img_width - 8; ix += 4)
//			{
//				//if (*(p_pxLine + 7) + 1 >= (uint_)src_img_size.width) continue;
//				//p_pxLine += 4;
//
//				__m128 ymm_fx = _mm_loadu_ps(p_axLUT);
//				__m128 ymm_cx = _mm_loadu_ps(p_axLUT2);
//
//				ymm_fx = _mm_shuffle_ps(ymm_fx, ymm_fx, 216);
//				ymm_cx = _mm_shuffle_ps(ymm_cx, ymm_cx, 216);
//
//				_mm_storeu_ps(p_axLUT, ymm_fx);
//				_mm_storeu_ps(p_axLUT2, ymm_cx);
//
//				p_axLUT += 4;
//				p_axLUT2 += 4;
//			}
//#endif
		}
		void ImageResizer::clear()
		{
			pyLine.clear();
			pxLine.clear();
			ayLUT.clear();
			axLUT.clear();
		}
		inline void ImageResizer::checkSize(const Size& _dst_img_size, const Size& _src_img_size)
		{
			if (src_img_size.width != _src_img_size.width || src_img_size.height != _src_img_size.height
			 || dst_img_size.width != _dst_img_size.width || dst_img_size.height != _dst_img_size.height)
			{
				src_img_size.width = _src_img_size.width;
				src_img_size.height = _src_img_size.height;

				if (dst_img_size.width < _dst_img_size.width || dst_img_size.height < _dst_img_size.height)
				{
					clear();
				}

				dst_img_size.width = _dst_img_size.width;
				dst_img_size.height = _dst_img_size.height;

				preprocessing();
			}
		}

		void ImageResizer::NearestNeighborInterpolation(Image_8u& dst, Image_8u& src, int num_threads)
		{
#ifndef USE_OMP
			uint_* p_pyLine = pyLine();
#endif

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int iy = 0; iy < dst.height; ++iy)
			{
#ifndef USE_OMP
				const uint_ py = *p_pyLine++;
#else
				const uint_ py = pyLine[iy];
#endif

				//if (py >= (uint_)src.height) continue;

				const uchar_* __restrict pSrc0 = src.data + py * src.widthStep;
				uchar_* __restrict pDst = dst.data + iy * dst.widthStep;

				uint_* p_pxLine = pxLine();

				int ix = 0;
				for (; ix <= dst.width - 8; ix += 8)
				{
					//if (*(p_pxLine + 7) + 1> (uint_)src.width) continue;

					*(pDst + 0) = pSrc0[*(p_pxLine + 0)];
					*(pDst + 1) = pSrc0[*(p_pxLine + 1)];
					*(pDst + 2) = pSrc0[*(p_pxLine + 2)];
					*(pDst + 3) = pSrc0[*(p_pxLine + 3)];
					*(pDst + 4) = pSrc0[*(p_pxLine + 4)];
					*(pDst + 5) = pSrc0[*(p_pxLine + 5)];
					*(pDst + 6) = pSrc0[*(p_pxLine + 6)];
					*(pDst + 7) = pSrc0[*(p_pxLine + 7)];
					p_pxLine += 8;
					pDst += 8;
				}

				for (; ix < dst.width; ++ix)
				{
					const uint_ px = *p_pxLine++;
					//if (px >= (uint_)src.width) continue;
					*pDst++ = pSrc0[px /*+ 1*/];
				}
			}
		}
		void ImageResizer::BilinearInterpolation(Image_8u& dst, Image_8u& src, int num_threads)
		{
#ifndef USE_OMP
			uint_* p_pyLine = pyLine();
			float* p_ayLUT = ayLUT();
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX2)
			__m128i ymm_mask = { 0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15 };
#else
			uchar_ ymm_mask[16] = { 0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15 };
#endif

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int iy = 0; iy < dst.height; ++iy)
			{
#ifndef USE_OMP
				uint_ py = *p_pyLine++;
#else
				uint_ py = pyLine[iy];
#endif

				if (py + 1 >= (uint_)src.height) py--;
				//if (py + 1 >= (uint_)src.height) continue;

				const uchar_* __restrict pSrc0 = src.data + py * src.widthStep;
				const uchar_* __restrict pSrc1 = src.data + (py + 1) * src.widthStep;
				uchar_* __restrict pDst = dst.data + iy * dst.widthStep;

				uint_* p_pxLine = pxLine();
				float* __restrict p_axLUT = axLUT();
				float* __restrict p_axLUT2 = axLUT() + dst.width;

#ifndef USE_OMP
				const float fy = *p_ayLUT++;
				const float cy = *p_ayLUT++;
#else
				const float fy = ayLUT[iy << 1];
				const float cy = ayLUT[(iy << 1) + 1];
#endif

				int ix = 0;

#if defined(USE_SSE) || defined(USE_AVX)
				const __m128 xmm_fy = _mm_set1_ps(fy);
				const __m128 xmm_cy = _mm_set1_ps(cy);
				for (; ix <= dst.width - 8; ix += 4)
				{
					//if (*(p_pxLine + 7) + 1 >= (uint_)src.width) continue;

					const __m128 xmm_fx = _mm_loadu_ps(p_axLUT);
					p_axLUT += 4;
					const __m128 xmm_cx = _mm_loadu_ps(p_axLUT2);
					p_axLUT2 += 4;

					const uint_ px1 = *(p_pxLine + 0);
					const uint_ px2 = *(p_pxLine + 1);
					const uint_ px3 = *(p_pxLine + 2);
					const uint_ px4 = *(p_pxLine + 3);
					p_pxLine += 4;

					//__m128i xmm81 = _mm_set_epi8(pSrc1[px4 + 1], pSrc1[px2 + 1], pSrc1[px3 + 1], pSrc1[px1 + 1],
					//							pSrc1[px4], pSrc1[px2], pSrc1[px3], pSrc1[px1],
					//							pSrc0[px4 + 1], pSrc0[px2 + 1], pSrc0[px3 + 1], pSrc0[px1 + 1],
					//							pSrc0[px4], pSrc0[px2], pSrc0[px3], pSrc0[px1]);

					__m128i xmm8 = _mm_set_epi16(*(short*)(pSrc1 + px4), *(short*)(pSrc1 + px3), *(short*)(pSrc1 + px2), *(short*)(pSrc1 + px1),
												 *(short*)(pSrc0 + px4), *(short*)(pSrc0 + px3), *(short*)(pSrc0 + px2), *(short*)(pSrc0 + px1));

					xmm8 = _mm_shuffle_epi8(xmm8, ymm_mask);

					__m128 xmm_d1 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(xmm8));
					xmm_d1 = _mm_mul_ps(xmm_d1, xmm_cx);

					__m128 xmm_d2 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi32(xmm8, 1)));
#ifdef USE_FMA
					xmm_d1 = _mm_fmadd_ps(xmm_d2, xmm_fx, xmm_d1);
#else
					xmm_d2 = _mm_mul_ps(xmm_d2, xmm_fx);
					xmm_d1 = _mm_add_ps(xmm_d1, xmm_d2);
#endif
					xmm_d1 = _mm_mul_ps(xmm_d1, xmm_cy);


					__m128 xmm_d3 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi32(xmm8, 2)));
					xmm_d3 = _mm_mul_ps(xmm_d3, xmm_cx);

					__m128 xmm_d4 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi32(xmm8, 3)));
#ifdef USE_FMA
					xmm_d3 = _mm_fmadd_ps(xmm_d4, xmm_fx, xmm_d3);
#else
					xmm_d4 = _mm_mul_ps(xmm_d4, xmm_fx);
					xmm_d3 = _mm_add_ps(xmm_d3, xmm_d4);
#endif

#ifdef USE_FMA
					xmm_d1 = _mm_fmadd_ps(xmm_d3, xmm_fy, xmm_d1);
#else
					xmm_d3 = _mm_mul_ps(xmm_d3, xmm_fy);
					xmm_d1 = _mm_add_ps(xmm_d1, xmm_d3);
#endif

					//xmm8 = _mm_shuffle_epi8(_mm_cvtps_epi32(xmm_d1), _mm_insert_epi32(xmm8, 201590784, 0));
					xmm8 = _mm_shuffle_epi8(_mm_cvtps_epi32(xmm_d1), _mm_insert_epi32(xmm8, 201851904, 0));
					
					_mm_store_ss((float*)pDst, _mm_castsi128_ps(xmm8));
					pDst += 4;
				}
#endif

				for (; ix < dst.width; ++ix)
				{
					const uint_ px = *p_pxLine++;

					//if (px + 1 >= (uint_)src.width) px--;
					//if (px + 1 >= (uint_)src.width) continue;

					const float fx = *p_axLUT++;
					const float cx = *p_axLUT2++;

					const float p0 = pSrc0[px];
					const float p1 = pSrc0[px + 1];
					const float p2 = pSrc1[px];
					const float p3 = pSrc1[px + 1];

					const float outv = (p0 * cx + p1 * fx) * cy + (p2 * cx + p3 * fx) * fy;
					*pDst++ = (uchar_)outv;
				}
			}
		}

		void ImageResizer::NearestNeighborInterpolation(Image_32f& dst, Image_32f& src, int num_threads)
		{
#ifndef USE_OMP
			uint_* p_pyLine = pyLine();
#endif

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int iy = 0; iy < dst.height; ++iy)
			{
#ifndef USE_OMP
				const uint_ py = *p_pyLine++;
#else
				const uint_ py = pyLine[iy];
#endif

				//if (py >= (uint_)src.height) continue;

				const float* __restrict pSrc0 = (float*)(src.data + py * src.widthStep);
				float* __restrict pDst = (float*)(dst.data + iy * dst.widthStep);

				uint_* p_pxLine = pxLine();

				int ix = 0;

#if defined(USE_SSE) || (defined(USE_AVX) && !defined(USE_AVX2))
				for (; ix <= dst.width - 8; ix += 4)
				{
					//if (*(p_pxLine + 7) >= (uint_)src_img_size.width) continue;

					__m128 ymm_p1 = _mm_setzero_ps();

					__m128 ymm_pS00, ymm_pS01;
					ymm_pS00 = _mm_loadl_pi(ymm_p1, (__m64*)&(pSrc0[*p_pxLine++]));
					ymm_pS00 = _mm_loadh_pi(ymm_pS00, (__m64*)&pSrc0[*p_pxLine++]);
					ymm_pS01 = _mm_loadl_pi(ymm_p1, (__m64*)&pSrc0[*p_pxLine++]);
					ymm_pS01 = _mm_loadh_pi(ymm_pS01, (__m64*)&pSrc0[*p_pxLine++]);

					ymm_p1 = _mm_shuffle_ps(ymm_pS00, ymm_pS01, 136);

					_mm_storeu_ps(pDst, ymm_p1);
					pDst += 4;
				}
#endif

#if defined(USE_AVX2)
				for (; ix <= dst.width - 8; ix += 8)
				{
					//if *(p_pxLine + 7) >= (uint_)src_img_size.width) continue;

					const __m256i vindex = _mm256_loadu_si256((__m256i*)p_pxLine);
					const __m256 ymm_p0 = _mm256_i32gather_ps(pSrc0, vindex, 4);
					_mm256_storeu_ps(pDst, ymm_p0);
					p_pxLine += 8;
					pDst += 8;
				}
#endif

				for (; ix < dst.width; ++ix)
				{
					const uint_ px = *p_pxLine++;
					//if (px >= (uint_)src.width) continue;
					*pDst++ = pSrc0[px /*+ 1*/];
				}
			}
		}
		void ImageResizer::BilinearInterpolation(Image_32f& dst, Image_32f& src, int num_threads)
		{
#ifndef USE_OMP
			uint_* p_pyLine = pyLine();
			float* p_ayLUT = ayLUT();
#endif

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int iy = 0; iy < dst.height; ++iy)
			{
#ifndef USE_OMP
				uint_ py = *p_pyLine++;
#else
				uint_ py = pyLine[iy];
#endif

				if (py + 1 >= (uint_)src.height) py--;
				//if (py + 1 >= (uint_)src.height) continue;
	
				const float* __restrict pSrc0 = (float*)(src.data + py * src.widthStep);
				const float* __restrict pSrc1 = (float*)(src.data + (py + 1) * src.widthStep);
				float* __restrict pDst = (float*)(dst.data + iy * dst.widthStep);

				uint_* p_pxLine = pxLine();
				float* __restrict p_axLUT = axLUT();
				float* __restrict p_axLUT2 = axLUT() + dst.width;

#ifndef USE_OMP
				const float fy = *p_ayLUT++;
				const float cy = *p_ayLUT++;
#else
				const float fy = ayLUT[iy << 1];
				const float cy = ayLUT[(iy << 1) + 1];
#endif

				int ix = 0;

#if defined(USE_SSE) || (defined(USE_AVX) && !defined(USE_AVX2))
				const __m128 ymm_fy = _mm_set1_ps(fy);
				const __m128 ymm_cy = _mm_set1_ps(cy);
				for (; ix <= dst.width - 8; ix += 4)
				{
					//if (*(p_pxLine + 7) + 1 >= (uint_)src.width) continue;

					__m128 ymm_fx = _mm_loadu_ps(p_axLUT);
					__m128 ymm_cx = _mm_loadu_ps(p_axLUT2);
					ymm_fx = _mm_shuffle_ps(ymm_fx, ymm_fx, 216);
					ymm_cx = _mm_shuffle_ps(ymm_cx, ymm_cx, 216);
					p_axLUT += 4;
					p_axLUT2 += 4;

					__m128 ymm_pS00, ymm_pS10;
					ymm_pS00 = _mm_loadl_pi(ymm_fy, (__m64*)&pSrc0[*p_pxLine]);
					ymm_pS10 = _mm_loadl_pi(ymm_fy, (__m64*)&pSrc1[*p_pxLine++]);

					__m128 ymm_pS01, ymm_pS11;
					ymm_pS01 = _mm_loadl_pi(ymm_fy, (__m64*)&pSrc0[*p_pxLine]);
					ymm_pS11 = _mm_loadl_pi(ymm_fy, (__m64*)&pSrc1[*p_pxLine++]);

					ymm_pS00 = _mm_loadh_pi(ymm_pS00, (__m64*)&pSrc0[*p_pxLine]);
					ymm_pS10 = _mm_loadh_pi(ymm_pS10, (__m64*)&pSrc1[*p_pxLine++]);

					ymm_pS01 = _mm_loadh_pi(ymm_pS01, (__m64*)&pSrc0[*p_pxLine]);
					ymm_pS11 = _mm_loadh_pi(ymm_pS11, (__m64*)&pSrc1[*p_pxLine++]);

					// Calculate the weighted sum of pixels
					//float outv = (p0 * cx + p1 * fx) * cy + (p2 * cx + p3 * fx) * fy;				
					__m128 ymm_p0 = _mm_shuffle_ps(ymm_pS00, ymm_pS01, 136);
					ymm_p0 = _mm_mul_ps(ymm_p0, ymm_cx);

					__m128 ymm_p1 = _mm_shuffle_ps(ymm_pS00, ymm_pS01, 221);
					ymm_p1 = _mm_mul_ps(ymm_p1, ymm_fx);

					__m128 ymm_p2 = _mm_shuffle_ps(ymm_pS10, ymm_pS11, 136);
					ymm_p2 = _mm_mul_ps(ymm_p2, ymm_cx);

					__m128 ymm_p3 = _mm_shuffle_ps(ymm_pS10, ymm_pS11, 221);
					ymm_p3 = _mm_mul_ps(ymm_p3, ymm_fx);

					ymm_p2 = _mm_add_ps(ymm_p2, ymm_p3);
					ymm_p2 = _mm_mul_ps(ymm_p2, ymm_fy);

					ymm_p0 = _mm_add_ps(ymm_p0, ymm_p1);

					ymm_p0 = _mm_mul_ps(ymm_p0, ymm_cy);
					ymm_p0 = _mm_add_ps(ymm_p0, ymm_p2);

					ymm_p0 = _mm_shuffle_ps(ymm_p0, ymm_p0, 216);

					_mm_storeu_ps(pDst, ymm_p0);
					pDst += 4;
				}
#endif

#if defined(USE_AVX2)
				const __m256 ymm_fy = _mm256_broadcast_ss(&fy);
				const __m256 ymm_cy = _mm256_broadcast_ss(&cy);
				for (; ix <= dst.width - 8; ix += 8)
				{
					//if (*(p_pxLine + 7) + 1 >= (uint_)src.width) continue;

					const __m256 ymm_fx = _mm256_loadu_ps(p_axLUT);
					p_axLUT += 8;
					const __m256 ymm_cx = _mm256_loadu_ps(p_axLUT2);
					p_axLUT2 += 8;

					__m256i vindex = _mm256_loadu_si256((__m256i*)p_pxLine);
					__m256 ymm_p0 = _mm256_i32gather_ps(pSrc0, vindex, 4);
					__m256 ymm_p2 = _mm256_i32gather_ps(pSrc1, vindex, 4);
					vindex = _mm256_add_epi32(vindex, _mm256_set1_epi32(1));
					__m256 ymm_p1 = _mm256_i32gather_ps(pSrc0, vindex, 4);
					__m256 ymm_p3 = _mm256_i32gather_ps(pSrc1, vindex, 4);
					p_pxLine += 8;

					//float outv = (p0 * cx + p1 * fx) * cy + (p2 * cx + p3 * fx) * fy;				
					ymm_p0 = _mm256_mul_ps(ymm_p0, ymm_cx);
					ymm_p2 = _mm256_mul_ps(ymm_p2, ymm_cx);

					ymm_p0 = _mm256_fmadd_ps(ymm_p1, ymm_fx, ymm_p0);
					ymm_p2 = _mm256_fmadd_ps(ymm_p3, ymm_fx, ymm_p2);

					ymm_p0 = _mm256_mul_ps(ymm_p0, ymm_cy);
					ymm_p0 = _mm256_fmadd_ps(ymm_p2, ymm_fy, ymm_p0);

					_mm256_storeu_ps(pDst, ymm_p0);
					pDst += 8;
				}
#endif

				for (; ix < dst.width; ++ix)
				{
					const uint_ px = *p_pxLine++;
					
					//if (px + 1 >= (uint_)src.width) px--;
					//if (px + 1 >= (uint_)src.width) continue;

					const float fx = *p_axLUT++;
					const float cx = *p_axLUT2++;

					const float p0 = pSrc0[px];
					const float p1 = pSrc0[px + 1];
					const float p2 = pSrc1[px];
					const float p3 = pSrc1[px + 1];

					const float outv = (p0 * cx + p1 * fx) * cy + (p2 * cx + p3 * fx) * fy;
					*pDst++ = outv;
				}
			}
		}

		void ImageResizer::FastImageResize(Image_8u& dst, Image_8u& src, const int type_resize, int num_threads)
		{
			if (dst.width == src.width && dst.height == src.height)
			{
				dst.copyData(src);
				return;
			}

			checkSize(dst.getSize(), src.getSize());

			switch (type_resize)
			{
			default:
			case 0:
				NearestNeighborInterpolation(dst, src, num_threads);
				break;

			case 1:
				BilinearInterpolation(dst, src, num_threads);
			}
		}
		void ImageResizer::FastImageResize(Image_32f& dst, Image_32f& src, const int type_resize, int num_threads)
		{
			if (dst.width == src.width && dst.height == src.height)
			{
				dst.copyData(src);
				return;
			}

			checkSize(dst.getSize(), src.getSize());

			switch (type_resize)
			{
			default:
			case 0:
				NearestNeighborInterpolation(dst, src, num_threads);
				break;

			case 1:
				BilinearInterpolation(dst, src, num_threads);
			}
		}
		void ImageResizer::getLineIndexes(uint_*& _pxLine, uint_*& _pyLine, const Size& _dst_img_size, const Size& _src_img_size)
		{
			checkSize(_dst_img_size, _src_img_size);
			_pxLine = pxLine();
			_pyLine = pyLine();
		}
	}

}
