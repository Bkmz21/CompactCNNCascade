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

#ifdef USE_CUDA
#	include <cuda_runtime.h>
#endif

#ifdef USE_CL
#	include "CL\opencl.h"
#endif

#include <chrono>


//================================================================================================================================================


namespace NeuralNetworksLib
{

	class Timer
	{
	private:
		std::chrono::steady_clock::time_point* counters;

	public:
		Timer(int = 1, bool = false);
		~Timer(void);

		void start(int = 0);
		double get(double = 1., int = 0);
		void print(double = 1., int = 0);
	};

	//----------------------------------------------------------

#ifdef USE_CUDA
	namespace CUDA
	{
		class Timer
		{
		private:
			cudaStream_t stream = 0;
			cudaEvent_t beginEvent;
			cudaEvent_t endEvent;

		public:
			Timer(bool = false, cudaStream_t = 0);
			~Timer(void);

			void setStream(cudaStream_t);
			void start();
			double get(double = 1.);
			void print(double = 1.);
		};
	}
#endif

	//----------------------------------------------------------

#ifdef USE_CL
	namespace CL
	{
		class Timer
		{
		private:
			cl_command_queue queue;
			std::chrono::steady_clock::time_point counters;

		public:
			Timer(cl_command_queue _queue = NULL, bool = false);
			~Timer(void);

			void setQueue(cl_command_queue);
			void start();
			double get(double = 1.);
			void print(double = 1.);
		};
	}
#endif

}