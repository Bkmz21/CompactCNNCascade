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


#include "timer.h"
#include <Windows.h>
#include <stdio.h>


//================================================================================================================================================


namespace NeuralNetworksLib
{

	Timer::Timer(int n, bool _start)
	{
		this->counters = new  unsigned __int64[n];
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

		if (_start)
		{
			for (int i = 0; i < n; ++i)
			{
				start(i);
			}
		}
	}
	Timer::~Timer(void)
	{
		delete[] this->counters;
	}

	void Timer::start(int counter) 
	{
		QueryPerformanceCounter((LARGE_INTEGER*)&this->counters[counter]);
	} 
	double Timer::get(double multiply, int counter) 
	{
		unsigned __int64 end;
		QueryPerformanceCounter((LARGE_INTEGER*)&end);
		return (double(end - this->counters[counter]) / freq) * multiply;
	}
	void Timer::print(double multiply, int counter)
	{
		printf("timer %d = %7.3f ", counter, get(multiply, counter));
		if (multiply == 1)		 printf("s");
		if (multiply == 1000)	 printf("ms");
		if (multiply == 1000000) printf("us");
		printf("\n");
	}

	//----------------------------------------------------------

#ifdef USE_CUDA
	namespace CUDA
	{
		Timer::Timer(bool _start, cudaStream_t _stream)
		{
 			cudaEventCreate(&beginEvent);
			cudaEventCreate(&endEvent);
			setStream(_stream);
			if (_start) start();
		}
		Timer::~Timer(void)
		{
			stream = 0;
			cudaEventDestroy(beginEvent);
			cudaEventDestroy(endEvent);
		}
		
		void Timer::setStream(cudaStream_t _stream)
		{
			stream = _stream;
		}
		void Timer::start()
		{
			cudaEventRecord(beginEvent, stream);
		}
		double Timer::get(double multiply)
		{
			cudaEventRecord(endEvent, stream);
			cudaEventSynchronize(endEvent);
			float timeValue;
			cudaEventElapsedTime(&timeValue, beginEvent, endEvent);
			return double(timeValue) / 1000. * multiply;
		}
		void Timer::print(double multiply)
		{
			printf("cuda timer = %7.3f ", get(multiply));
			if (multiply == 1)		 printf("s");
			if (multiply == 1000)	 printf("ms");
			if (multiply == 1000000) printf("us");
			printf("\n");
		}
	}
#endif

	//----------------------------------------------------------

#ifdef USE_CL
	namespace CL
	{
		Timer::Timer(cl_command_queue _queue, bool _start)
		{
			queue = _queue;
			QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

			if (_start) start();
		}
		Timer::~Timer(void) { }

		void Timer::setQueue(cl_command_queue _queue)
		{
			queue = _queue;
		}
		void Timer::start()
		{
			clFinish(queue);
			QueryPerformanceCounter((LARGE_INTEGER*)&this->counters);
		}
		double Timer::get(double multiply)
		{
			clFinish(queue);
			unsigned __int64 end;
			QueryPerformanceCounter((LARGE_INTEGER*)&end);
			return (double(end - this->counters) / freq) * multiply;
		}
		void Timer::print(double multiply)
		{
			printf("cl timer = %7.3f ", get(multiply));
			if (multiply == 1)		 printf("s");
			if (multiply == 1000)	 printf("ms");
			if (multiply == 1000000) printf("us");
			printf("\n");
		}
	}
#endif

}
