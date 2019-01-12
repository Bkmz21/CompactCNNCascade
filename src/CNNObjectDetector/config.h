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


//================================================================================================================================================


namespace NeuralNetworksLib
{

	//#define USE_CNTK_MODELS
	//#define USE_RESOURCE

	//#define USE_SSE
	//#define USE_AVX
	//#define USE_AVX2

	//#define USE_CUDA
	//#define USE_CL

	//#define USE_OMP
	//#define MAX_NUM_THREADS 4

	//#define PROFILE_DETECTOR
	//#define PROFILE_CNN_SIMD
	//#define PROFILE_CNN_CUDA
	//#define PROFILE_CNN_CL

	//#define CHECK_TEST


	//---------------------------------------------------------------------------


	#ifdef USE_AVX2
	#	define USE_AVX
	#	define USE_FMA
	#	ifdef USE_CNTK_MODELS
	#		define USE_HF
	#		define USE_FIXED_POINT
	#	endif
	#endif

	#define USE_FAST_TANH

	#ifndef CHECK_TEST
	#   define USE_FAST_DIV
	#endif

	//Legacy	
	//#define USE_CUDA_PTX_TANH
	//#define USE_CUDA_INLINE_WEIGHTS

	typedef unsigned char uchar_;
	typedef unsigned int uint_;

	#define ALIGN_SSE 16
	#define ALIGN_AVX 32
    #if defined(_MSC_VER)
    #   define ALIGN(_ALIGN) __declspec(align(_ALIGN))
    #else
    #   define ALIGN(_ALIGN) __attribute__ ((aligned(_ALIGN)))
    #endif

	#if defined(USE_AVX) || (!defined(USE_AVX) && !defined(USE_SSE))
	#	define REG_SIZE 8
	#	define ALIGN_DEF ALIGN_AVX
	#else
	#	define REG_SIZE 4
	#	define ALIGN_DEF ALIGN_SSE
	#endif

	#ifdef USE_OMP
	#	define OMP_PRAGMA(pragma) __pragma (pragma)
	#	define OMP_RUNTIME(func) func;
	#else
	#	define OMP_PRAGMA(pragma)
	#	define OMP_RUNTIME(func)
	#endif

	#ifdef USE_CUDA
	#	define CUDA_CODE(func) func
	#else
	#	define CUDA_CODE(func)
	#endif

	#ifdef USE_CL
	#	define CL_CODE(func) func
	#else
	#	define CL_CODE(func)
	#endif

	#if defined(USE_CUDA) || defined(USE_CL)
	#	define GPU_ONLY(func) func
	#else
	#	define GPU_ONLY(func)
	#endif

	#ifdef PROFILE_DETECTOR
	#	define PROFILE_TIMER(timer, acc, func)		\
			timer->start();							\
			func									\
			acc += timer->get(1000.);			
	#	define PROFILE_COUNTER_INC(counter) counter++;
	#	define PROFILE_COUNTER_ADD(counter, val) counter += val;
	#else
	#	define PROFILE_TIMER(timer, acc, func) func
	#	define PROFILE_COUNTER_INC(counter)
	#	define PROFILE_COUNTER_ADD(counter, val)
	#endif

	#if defined(USE_AVX) && defined(USE_SSE)						\
	 || defined(USE_AVX2) && defined(USE_SSE)						\
	 || defined(USE_FIXED_POINT) && !defined(USE_AVX2)				\
	 || defined(USE_CUDA) && defined(USE_CL)						\
	 || defined(USE_CUDA) && !defined(USE_CNTK_MODELS)				\
	 || defined(USE_CNTK_MODELS) && defined(USE_SSE)				\
	 || defined(USE_CNTK_MODELS) && defined(CHECK_TEST)
	#	error Incorrect flags!
	#endif  

}