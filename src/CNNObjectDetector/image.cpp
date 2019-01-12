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


#include "image.h"
#if defined(USE_SSE) || defined(USE_AVX)
#	include <immintrin.h>
#endif


//========================================================================================================


namespace NeuralNetworksLib
{

	namespace SIMD
	{
		void* mm_malloc(size_t size, size_t align)
		{
#if defined(_MSC_VER)
			return _mm_malloc(size, align);
#else
            void* memptr = NULL;
			posix_memalign(&memptr, align, size);
			return memptr;
#endif
		}

		void mm_free(void* p)
		{
#if defined(_MSC_VER)
			return _mm_free(p);
#else
			return free(p);
#endif
		}

		void mm_erase(void* p, int size)
		{
			unsigned char* pData = (unsigned char*)(p);

			int i = 0;
#if defined(USE_SSE) || defined(USE_AVX)
			for (; i <= size - 16; i += 16)
			{
				__m128i ymm0 = _mm_loadu_si128((__m128i*)(pData));
				ymm0 = _mm_xor_si128(ymm0, ymm0);
				_mm_storeu_si128((__m128i*)(pData), ymm0);
				pData += 16;
			}
#endif

			for (; i < size; ++i)
			{
				*pData++ = 0;
			}
		}
	}

}
