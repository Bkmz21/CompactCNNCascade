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
#include "type.h"
#include "image.h"

#ifdef USE_CL
#	include <CL\opencl.h>
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_CL

	namespace CL
	{
		#define getPosImCLHost(ImCL, x, y) ImCL.dataHost[(y) * ImCL.widthStepHost + (x)]
		#define getPosImCLDevice(ImCL, x, y) ImCL.dataDevice[(y) * ImCL.widthStepDevice + (x)]

		#ifdef _DEBUG
			#define clERR(func)																													\
			do {																																\
				cl_int error = func;																											\
				if (error != CL_SUCCESS)															  											\
				{																																\
					/*printf("OpenCL Error: line %d of file \"%s\" <%s> = %s\n", __LINE__, __FILE__, __FUNCTION__, (int)error);*/					\
				}																																\
			} while (false)
		#else
			#define clERR(func) func
		#endif

		template <typename type, const int pinned_mem>
		class TmpImage
		{
		private:
			enum clMemcpyKind
			{
				clMemcpyHostToDevice = 1,
				clMemcpyDeviceToHost = 2
			};

			cl_context context = 0;
			cl_command_queue queue = 0;
			const int dev_align = 128;

			void clCopyData(int kind, cl_event* _event);
			void clCopyDataAsync(int kind, cl_event* _event);

		public:
			int width = 0;
			int height = 0;
			int size = 0;
			int widthStepHost = 0;
			int heightStepHost = 0;
			int widthStepDevice = 0;
			int heightStepDevice = 0;
			int nChannel = 0;
			type* dataHost = NULL;
			cl_mem dataDevice = 0;
			int alignDataHost = 0;
			bool sharingDataHost = false;
			bool sharingDataDevice = false;
			int offsetDevice = 0;

			TmpImage();
			TmpImage(const TmpImage& img);
			TmpImage(cl_context _context, cl_command_queue _queue, int _width, int _height, int _nChannel = 1, bool _stride = false, int _alignHost = 0);
			TmpImage(cl_context _context, cl_command_queue _queue, int _width, int _height, int _widthStepHost, int _widthStepDevice, int _alignHost);
			TmpImage(cl_context _context, cl_command_queue _queue, int _width, int _height, int _widthStepHost, int _widthStepDevice, int _heightStepDevice, int _alignHost);
			TmpImage(cl_context _context, cl_command_queue _queue, int _width, int _height, int _nChannel, type* _DataHost, int _widthStep = 0);
			TmpImage(cl_context _context, cl_command_queue _queue, int _width, int _height, int _nChannel, type* _DataHost, cl_mem _DataDevice, int _widthStepHost, int _widthStepDevice, int _offsetDevice = 0);
			~TmpImage() { clear(); }
			void copyData(TmpImage<type, pinned_mem>* src, bool async = false, cl_event* _event = NULL);
			void copyData(SIMD::TmpImage<type>* src, bool async = false, cl_event* _event = NULL);
			void copyDataHost(SIMD::TmpImage<type>* src);
			void copyDataDevice(SIMD::TmpImage<type>* src, bool async = false, cl_event* _event = NULL);
			void copyDataDeviceToDevice(TmpImage* src, cl_event* _event = NULL);
			void updateDataHost(bool async = false, cl_event* _event = NULL);
			void updateDataDevice(bool async = false, cl_event* _event = NULL);
			void clone(const TmpImage& img);
			void erase();
			void clear();
			inline bool isEmpty() const { return dataHost == 0 && dataDevice == 0; }
			inline void setSize(const Size& size) { width = size.width; height = size.height; }
			inline Size getSize() const { return Size(width, height); }
			TmpImage<type, pinned_mem>& operator=(const TmpImage<type, pinned_mem>& _img);
		};
		typedef TmpImage<cl_uchar, 0> Image_8u;
		//typedef TmpImage<cl_uchar, 1> Image_8u_pinned;
		//typedef TmpImage<cl_uchar, 2> Image_8u_mapped;
		typedef TmpImage<cl_float, 0> Image_32f;
		//typedef TmpImage<cl_float, 1> Image_32f_pinned;
		//typedef TmpImage<cl_float, 2> Image_32f_mapped;

		//----------------------------------------------------------

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::clCopyData(int kind, cl_event* _event)
		{
			if (widthStepHost == widthStepDevice)
			{
				size_t size_buffer = 0;
				if (!pinned_mem)
				{
					size_buffer = widthStepHost * heightStepHost * sizeof(type);
				}
				else
				{
					size_buffer = widthStepDevice * heightStepDevice * sizeof(type);
				}

				if (kind == clMemcpyHostToDevice)
				{
					clERR(clEnqueueWriteBuffer(queue, dataDevice, CL_TRUE, 0, size_buffer, dataHost, 0, NULL, _event));
				}
				if (kind == clMemcpyDeviceToHost)
				{
					clERR(clEnqueueReadBuffer(queue, dataDevice, CL_TRUE, 0, size_buffer, dataHost, 0, NULL, _event));
				}
			}
			else
			{
				printf("error cl memory\n");
				system("pause");
			}

			//else
			//{
			//	if (kind == clMemcpyHostToDevice)
			//	{
			//		size_t size_buffer = width * sizeof(type);
			//		clERR(cudaMemcpy2D(dataDevice, widthStepDevice * sizeof(type), dataHost, widthStepHost * sizeof(type), size_buffer, height, (cudaMemcpyKind)kind));
			//	}
			//	if (kind == clMemcpyDeviceToHost)
			//	{
			//		size_t size_buffer = width * sizeof(type);
			//		clERR(cudaMemcpy2D(dataHost, widthStepHost * sizeof(type), dataDevice, widthStepDevice * sizeof(type), size_buffer, height, (cudaMemcpyKind)kind));
			//	}
			//}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::clCopyDataAsync(int kind, cl_event* _event)
		{
			if (widthStepHost == widthStepDevice)
			{
				size_t size_buffer = 0;
				if (!pinned_mem)
				{
					size_buffer = widthStepHost * heightStepHost * sizeof(type);
				}
				else
				{
					size_buffer = widthStepDevice * heightStepDevice * sizeof(type);
				}

				if (kind == clMemcpyHostToDevice)
				{
					clERR(clEnqueueWriteBuffer(queue, dataDevice, CL_FALSE, 0, size_buffer, dataHost, 0, NULL, _event));
				}
				if (kind == clMemcpyDeviceToHost)
				{
					clERR(clEnqueueReadBuffer(queue, dataDevice, CL_FALSE, 0, size_buffer, dataHost, 0, NULL, _event));
				}
			}
			else
			{
				printf("error cl memory\n");
				system("pause");
			}

			//else
			//{
			//	if (kind == clMemcpyHostToDevice)
			//	{
			//		size_t size_buffer = width * sizeof(type);
			//		clERR(cudaMemcpy2DAsync(dataDevice, widthStepDevice * sizeof(type), dataHost, widthStepHost * sizeof(type), size_buffer, height, (cudaMemcpyKind)kind, cuda_stream));
			//	}
			//	if (kind == clMemcpyDeviceToHost)
			//	{
			//		size_t size_buffer = width * sizeof(type);
			//		clERR(cudaMemcpy2DAsync(dataHost, widthStepHost * sizeof(type), dataDevice, widthStepDevice * sizeof(type), size_buffer, height, (cudaMemcpyKind)kind, cuda_stream));
			//	}
			//}
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage()
		{
			context = 0;
			queue = 0;
			width = 0;
			height = 0;
			size = 0;
			widthStepHost = 0;
			heightStepHost = 0;
			widthStepDevice = 0;
			heightStepDevice = 0;
			nChannel = 1;
			dataHost = 0;
			dataDevice = 0;
			alignDataHost = 4;
			sharingDataHost = false;
			sharingDataDevice = false;
			offsetDevice = 0;
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage(const TmpImage<type, pinned_mem>& _img) :
			context(_img.context),
		    queue(_img.queue),
			width(_img.width),
			height(_img.height),
			size(_img.size),
			widthStepHost(_img.widthStepHost),
			heightStepHost(_img.heightStepHost),
			widthStepDevice(_img.widthStepDevice),
			heightStepDevice(_img.heightStepDevice),
			nChannel(_img.nChannel),
			dataHost(_img.dataHost),
			dataDevice(_img.dataDevice),
			alignDataHost(_img.alignDataHost),
			sharingDataHost(_img.sharingDataHost),
			sharingDataDevice(_img.sharingDataDevice),
			offsetDevice(_img.offsetDevice)
		{ }

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage(cl_context _context, cl_command_queue _queue, int _width, int _height, int _nChannel, bool _stride, int _alignHost)
		{
			context = _context;
			queue = _queue;

			width = _width;
			height = _height;
			size = width * height;
			widthStepHost = width * _nChannel;
			nChannel = _nChannel;
			alignDataHost = roundUpMul(_alignHost > 0 ? _alignHost : sizeof(type), sizeof(type));
			sharingDataHost = false;
			sharingDataDevice = false;
			if (_stride)
			{
				widthStepHost = roundUpMul(widthStepHost, alignDataHost / sizeof(type));
			}

			if (!pinned_mem)
			{
				size_t size_buffer = widthStepHost * height * sizeof(type);
				dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);

				//if (_stride)
				//{
				//	widthStepDevice = width * _nChannel;
				//	widthStepDevice = roundUpMul(widthStepDevice, dev_align / sizeof(type));
				//	size_t size_buffer = widthStepDevice * height * sizeof(type);
				//
				//	cl_int err = CL_INVALID_VALUE;
				//	dataDevice = clCreateBuffer(context, CL_MEM_READ_WRITE, size_buffer, NULL, &err);
				//	clERR(err);
				//}
				//else
				{
					cl_int err = CL_INVALID_VALUE;
					dataDevice = clCreateBuffer(context, CL_MEM_READ_WRITE, size_buffer, NULL, &err);
					clERR(err);
					widthStepDevice = widthStepHost;
				}
			}
			//else
			//{
			//	if (_stride)
			//	{
			//		widthStepHost = roundUpMul(widthStepHost, dev_align / sizeof(type));
			//	}
			//	widthStepDevice = widthStepHost;
			//	size_t size_buffer = widthStepDevice * height * sizeof(type);

			//	if (pinned_mem == 2)
			//	{
			//		dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
			//		clERR(cudaHostRegister(dataHost, size_buffer, cudaHostRegisterMapped));
			//		clERR(cudaHostGetDevicePointer(&dataDevice, dataHost, 0));
			//	}
			//	else
			//	{
			//		dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
			//		clERR(cudaHostRegister(dataHost, size_buffer, 0));
			//		clERR(cudaMalloc((void**)&dataDevice, size_buffer));
			//	}
			//}

			heightStepHost = height;
			heightStepDevice = height;
			offsetDevice = 0;
			erase();
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage(cl_context _context, cl_command_queue _queue, int _width, int _height, int _widthStepHost, int _widthStepDevice, int _alignHost)
		{
			_widthStepHost = _widthStepDevice;

			context = _context;
			queue = _queue;

			width = _width;
			height = _height;
			size = width * height;
			widthStepHost = _widthStepHost;
			widthStepDevice = _widthStepDevice;
			nChannel = 1;
			alignDataHost = _alignHost;
			sharingDataHost = false;
			sharingDataDevice = false;

			size_t size_buffer = 0;

			if (!pinned_mem)
			{
				size_buffer = widthStepHost * height * sizeof(type);
				dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);

				size_buffer = widthStepDevice * height * sizeof(type);
				cl_int err = CL_INVALID_VALUE;
				dataDevice = clCreateBuffer(context, CL_MEM_READ_WRITE, size_buffer, NULL, &err);
				clERR(err);
			}
			//else
			//{
			//	widthStepHost = widthStepDevice;
			//	size_buffer = widthStepDevice * height * sizeof(type);

			//	if (pinned_mem == 2)
			//	{
			//		dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
			//		clERR(cudaHostRegister(dataHost, size_buffer, cudaHostRegisterMapped));
			//		clERR(cudaHostGetDevicePointer(&dataDevice, dataHost, 0));
			//	}
			//	else
			//	{
			//		dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
			//		clERR(cudaHostRegister(dataHost, size_buffer, 0));
			//		clERR(cudaMalloc((void**)&dataDevice, size_buffer));
			//	}
			//}

			heightStepHost = height;
			heightStepDevice = height;
			offsetDevice = 0;
			erase();
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage(cl_context _context, cl_command_queue _queue, int _width, int _height, int _widthStepHost, int _widthStepDevice, int _heightStepDivice, int _alignHost)
		{
			_widthStepHost = _widthStepDevice;

			context = _context;
			queue = _queue;

			width = _width;
			height = _height;
			size = width * height;
			widthStepHost = _widthStepHost;
			heightStepHost = _heightStepDivice;
			widthStepDevice = _widthStepDevice;
			heightStepDevice = _heightStepDivice;
			nChannel = 1;
			alignDataHost = _alignHost;
			sharingDataHost = false;
			sharingDataDevice = false;

			size_t size_buffer = 0;

			if (!pinned_mem)
			{
				size_buffer = widthStepHost * heightStepHost * sizeof(type);
				dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);

				size_buffer = widthStepDevice * heightStepDevice * sizeof(type);
				cl_int err = CL_INVALID_VALUE;
				dataDevice = clCreateBuffer(context, CL_MEM_READ_WRITE, size_buffer, NULL, &err);
				clERR(err);
			}
			//else
			//{
			//	widthStepHost = widthStepDevice;
			//	size_buffer = widthStepDevice * heightStepDevice * sizeof(type);

			//	if (pinned_mem == 2)
			//	{
			//		dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
			//		clERR(cudaHostRegister(dataHost, size_buffer, cudaHostRegisterMapped));
			//		clERR(cudaHostGetDevicePointer(&dataDevice, dataHost, 0));
			//	}
			//	else
			//	{
			//		dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
			//		clERR(cudaHostRegister(dataHost, size_buffer, 0));
			//		clERR(cudaMalloc((void**)&dataDevice, size_buffer));
			//	}
			//}

			offsetDevice = 0;
			erase();
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage(cl_context _context, cl_command_queue _queue, int _width, int _height, int _nChannel, type* _DataHost, int _widthStep)
		{
			context = _context;
			queue = _queue;

			width = _width;
			height = _height;
			size = width * height;
			widthStepHost = _widthStep;
			if (widthStepHost == 0) widthStepHost = width * _nChannel;
			heightStepHost = height;
			widthStepDevice = widthStepHost;
			heightStepDevice = height;
			nChannel = _nChannel;
			dataHost = _DataHost;
			sharingDataHost = true;
			sharingDataDevice = false;

			size_t size_buffer = widthStepDevice * heightStepHost * sizeof(type);

			if (!pinned_mem)
			{
				cl_int err = CL_INVALID_VALUE;
				dataDevice = clCreateBuffer(context, CL_MEM_READ_WRITE, size_buffer, NULL, &err);
				clERR(err);

				clCopyData(clMemcpyHostToDevice, NULL);
			}
			//else
			//{
			//	if (pinned_mem == 2)
			//	{
			//		clERR(cudaHostRegister(dataHost, size_buffer, cudaHostRegisterMapped));
			//		clERR(cudaHostGetDevicePointer(&dataDevice, dataHost, 0));
			//	}
			//	else
			//	{
			//		clERR(cudaMalloc((void**)&dataDevice, size_buffer));
			//	}
			//}

			offsetDevice = 0;
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage(cl_context _context, cl_command_queue _queue, int _width, int _height, int _nChannel, type* _DataHost, cl_mem _DataDevice, int _widthStepHost, int _widthStepDevice, int _offsetDevice)
		{
			context = _context;
			queue = _queue;

			width = _width;
			height = _height;
			size = width * height;
			widthStepHost = _widthStepHost;
			if (widthStepHost == 0) widthStepHost = width * _nChannel;
			heightStepHost = height;
			widthStepDevice = _widthStepDevice;
			heightStepDevice = height;
			nChannel = _nChannel;
			dataHost = _DataHost;
			dataDevice = _DataDevice;
			sharingDataHost = true;
			sharingDataDevice = true;
			offsetDevice = _offsetDevice;

			size_t size_buffer = widthStepDevice * heightStepHost * sizeof(type);

			//if (pinned_mem == 1)
			//{
			//	clERR(cudaHostRegister(dataHost, size_buffer, cudaHostRegisterMapped));
			//}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::copyData(TmpImage<type, pinned_mem>* src, bool async, cl_event* _event)
		{
			const size_t _size = MIN(src->width * src->nChannel, widthStepHost) * sizeof(type);
			for (int j = 0; j < src->height; ++j)
			{
				memcpy(dataHost + j * widthStepHost, src->dataHost + j * src->widthStepHost, _size);
			}

			if (!async)
			{
				clCopyData(clMemcpyHostToDevice, _event);
			}
			else
			{
				clCopyDataAsync(clMemcpyHostToDevice, _event);
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::copyData(SIMD::TmpImage<type>* src, bool async, cl_event* _event)
		{
			if (this->width == src->width && this->widthStepHost == src->widthStep)
			{
				const size_t _size = src->widthStep * src->height * sizeof(type);
				memcpy(dataHost, src->data, _size);
			}
			else
			{
				const size_t _size = MIN(src->width * src->nChannel, widthStepHost) * sizeof(type);
				for (int j = 0; j < src->height; ++j)
				{
					memcpy(dataHost + j * widthStepHost, src->data + j * src->widthStep, _size);
				}
			}

			if (!async)
			{
				clCopyData(clMemcpyHostToDevice, _event);
			}
			else
			{
				clCopyDataAsync(clMemcpyHostToDevice, _event);
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::copyDataHost(SIMD::TmpImage<type>* src)
		{
			const size_t _size = min(src->width * src->nChannel, widthStepHost) * sizeof(type);
			for (int j = 0; j < src->height; ++j)
			{
				memcpy(dataHost + j * widthStepHost, src->data + j * src->widthStep, _size);
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::copyDataDevice(SIMD::TmpImage<type>* src, bool async, cl_event* _event)
		{
			if (!async)
			{
				clCopyData(clMemcpyHostToDevice, _event);
			}
			else
			{
				clCopyDataAsync(clMemcpyHostToDevice, _event);
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::copyDataDeviceToDevice(TmpImage* src, cl_event* _event = NULL)
		{
			const size_t src_origin[3] = { (size_t)src->offsetDevice, 0, 0 };
			const size_t dst_origin[3] = { (size_t)offsetDevice, 0, 0 };
			const size_t region[3] = { (size_t)(src->nChannel * src->width), (size_t)src->height, 1 };

			clERR(clEnqueueCopyBufferRect(
				queue,
				src->dataDevice,
				dataDevice,
				src_origin,
				dst_origin,
				region,
				src->widthStepDevice * sizeof(type),
				0,
				widthStepDevice * sizeof(type),
				0,
				0,
				NULL,
				_event));
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::updateDataHost(bool async, cl_event* _event)
		{
			if (!async)
			{
				clCopyData(clMemcpyDeviceToHost, _event);
			}
			else
			{
				clCopyDataAsync(clMemcpyDeviceToHost, _event);
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::updateDataDevice(bool async, cl_event* _event)
		{
			if (!async)
			{
				clCopyData(clMemcpyHostToDevice, _event);
			}
			else
			{
				clCopyDataAsync(clMemcpyHostToDevice, _event);
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::clone(const TmpImage& img)
		{
			context = img.context;
			queue = img.queue;

			width = img.width;
			height = img.height;
			size = img.size;
			widthStepHost = img.widthStepHost;
			heightStepHost = img.heightStepHost;
			widthStepDevice = img.widthStepDevice;
			heightStepDevice = img.heightStepDevice;
			nChannel = img.nChannel;
			dataHost = img.dataHost;
			dataDevice = img.dataDevice;
			alignDataHost = img.alignDataHost;
			sharingDataHost = true;
			sharingDataDevice = true;
			offsetDevice = img.offsetDevice;
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::erase()
		{
			int size_buffer;

			if (!pinned_mem)
			{
				size_buffer = widthStepHost * height * sizeof(type);
				SIMD::mm_erase(dataHost, size_buffer);

				clCopyData(clMemcpyHostToDevice, NULL);
			}
			//else
			//{
			//	size_buffer = widthStepDevice * heightStepDevice * sizeof(type);

			//	if (pinned_mem == 2)
			//	{
			//		SIMD::mm_erase(dataHost, size_buffer);
			//	}
			//	else
			//	{
			//		clERR(cudaMemset(dataDevice, 0, size_buffer));
			//	}
			//}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::clear()
		{
			if (!pinned_mem)
			{
				if (dataDevice != 0 && !sharingDataDevice) clERR(clReleaseMemObject(dataDevice));
			}
			//else
			//{
			//	if (dataHost != 0) clERR(cudaHostUnregister(dataHost));

			//	if (pinned_mem == 1)
			//	{
			//		if (dataDevice != 0 && !sharingDataDevice) clERR(clReleaseMemObject(dataDevice));
			//	}
			//}

			context = 0;
			queue = 0;

			width = 0;
			height = 0;
			size = 0;
			widthStepHost = 0;
			heightStepHost = 0;
			nChannel = 0;
			alignDataHost = 4;
			if (dataHost != 0 && !sharingDataHost) SIMD::mm_free(dataHost);
			dataHost = 0;
			sharingDataHost = false;

			dataDevice = 0;
			sharingDataDevice = false;
			widthStepDevice = 0;
			heightStepDevice = 0;
			offsetDevice = 0;
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>& TmpImage<type, pinned_mem>::operator=(const TmpImage<type, pinned_mem>& _img)
		{
			if (this == &_img) return *this;
			if (!isEmpty()) clear();

			context = _img.context;
			queue = _img.queue;

			width = _img.width;
			height = _img.height;
			size = _img.size;
			widthStepHost = _img.widthStepHost;
			heightStepHost = _img.heightStepHost;
			widthStepDevice = _img.widthStepDevice;
			heightStepDevice = _img.heightStepDevice;
			nChannel = _img.nChannel;
			dataHost = _img.dataHost;
			dataDevice = _img.dataDevice;
			alignDataHost = _img.alignDataHost;
			sharingDataHost = _img.sharingDataHost;
			sharingDataDevice = _img.sharingDataDevice;
			offsetDevice = _img.offsetDevice;

			TmpImage<type, pinned_mem> *pimg = const_cast<TmpImage<type, pinned_mem>*>(&_img);
			pimg->context = 0;
			pimg->queue = 0;
			pimg->width = 0;
			pimg->height = 0;
			pimg->size = 0;
			pimg->widthStepHost = 0;
			pimg->heightStepHost = 0;
			pimg->widthStepDevice = 0;
			pimg->heightStepDevice = 0;
			pimg->nChannel = 1;
			pimg->dataHost = 0;
			pimg->dataDevice = 0;
			pimg->alignDataHost = 4;
			pimg->sharingDataHost = false;
			pimg->sharingDataDevice = false;
			pimg->offsetDevice = 0;

			return *this;
		}
	}

#endif
}
