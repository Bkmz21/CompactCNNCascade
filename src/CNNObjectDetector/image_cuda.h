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

#ifdef USE_CUDA
#	include <cuda_runtime.h>
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_CUDA

	namespace CUDA
	{
		#define getPosImCUHost(ImCU, x, y) ImCU.dataHost[(y) * ImCU.widthStepHost + (x)]
		#define getPosImCUDevice(ImCU, x, y) ImCU.dataDevice[(y) * ImCU.widthStepDevice + (x)]

		#ifdef _DEBUG
			#define cuERR(func)																													\
			do {																																\
				cudaError_t error = func;																										\
				if (error != 0)															  														\
				{																																\
					printf("CUDA Error: line %d of file \"%s\" <%s> = %s\n", __LINE__, __FILE__, __FUNCTION__, cudaGetErrorString(error));		\
				}																																\
			} while (false)
		#else
			#define cuERR(func) func
		#endif

		template <typename type, const int pinned_mem>
		class TmpImage
		{
		private:
			void cudaCopyData(int kind, bool update_part = false);
			void cudaCopyDataAsync(int kind, cudaStream_t cuda_stream = 0, bool update_part = false);

			const int dev_align = 128;

		public:
			int width = 0;
			int height = 0;
			int size = 0;
			int widthStepHost = 0;
			int widthStepDevice = 0;
			int heightStepDevice = 0;
			int nChannel = 0;
			type* dataHost = NULL;
			type* dataDevice = NULL;
			int alignDataHost = 0;
			bool sharingDataHost = false;
			bool sharingDataDevice = false;
			int offsetDevice = 0;

			TmpImage();
			TmpImage(const TmpImage& img);
			TmpImage(int _width, int _height, int _nChannel = 1, bool _stride = false, int _alignHost = 0);
			TmpImage(int _width, int _height, int _widthStepHost, int _widthStepDevice, int _alignHost);
			TmpImage(int _width, int _height, int _widthStepHost, int _widthStepDevice, int _heightStepDevice, int _alignHost);
			TmpImage(int _width, int _height, int _nChannel, type* _DataHost, int _widthStep = 0);
			TmpImage(int _width, int _height, int _nChannel, type* _DataHost, type* _DataDevice, int _widthStepHost, int _widthStepDevice, int _offsetDevice = 0);
			~TmpImage() { clear(); }
			void copyData(TmpImage<type, pinned_mem>* src, bool async = false, cudaStream_t cuda_stream = 0);
			void copyData(SIMD::TmpImage<type>* src, bool async = false, cudaStream_t cuda_stream = 0);
			void copyDataHost(SIMD::TmpImage<type>* src);
			void copyDataDevice(SIMD::TmpImage<type>* src, bool async = false, cudaStream_t cuda_stream = 0);
			void copyDataDeviceToDevice(TmpImage* src, bool async = false, cudaStream_t cuda_stream = 0);
			void updateDataHost(bool async = false, cudaStream_t cuda_stream = 0, bool update_part = false);
			void updateDataDevice(bool async = false, cudaStream_t cuda_stream = 0, bool update_part = false);
			void clone(const TmpImage& img);
			void erase();
			void clear();
			inline bool isEmpty() const { return dataHost == 0 && dataDevice == 0; }
			inline void setSize(const Size& size) { width = size.width; height = size.height; }
			inline Size getSize() const { return Size(width, height); }
			TmpImage<type, pinned_mem>& operator=(TmpImage<type, pinned_mem>& img);
		};
		typedef TmpImage<uchar_, 0> Image_8u;
		typedef TmpImage<uchar_, 1> Image_8u_pinned;
		typedef TmpImage<uchar_, 2> Image_8u_mapped;
		typedef TmpImage<float, 0> Image_32f;
		typedef TmpImage<float, 1> Image_32f_pinned;
		typedef TmpImage<float, 2> Image_32f_mapped;

		//----------------------------------------------------------

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::cudaCopyData(int kind, bool update_part)
		{
			if (widthStepHost == widthStepDevice && !update_part)
			{
				size_t size_buffer = 0;
				if (!pinned_mem)
				{
					size_buffer = widthStepHost * height * sizeof(type);
				}
				else
				{
					size_buffer = widthStepDevice * heightStepDevice * sizeof(type);
				}

				if (kind == cudaMemcpyHostToDevice)
				{
					cuERR(cudaMemcpy(dataDevice, dataHost, size_buffer, (cudaMemcpyKind)kind));
				}
				if (kind == cudaMemcpyDeviceToHost)
				{
					cuERR(cudaMemcpy(dataHost, dataDevice, size_buffer, (cudaMemcpyKind)kind));
				}
			}
			else
			{
				if (kind == cudaMemcpyHostToDevice)
				{
					size_t size_buffer = nChannel * width * sizeof(type);
					cuERR(cudaMemcpy2D(
						dataDevice, 
						widthStepDevice * sizeof(type),
						dataHost, 
						widthStepHost * sizeof(type), 
						size_buffer, 
						height, 
						(cudaMemcpyKind)kind));
				}
				if (kind == cudaMemcpyDeviceToHost)
				{
					size_t size_buffer = nChannel * width * sizeof(type);
					cuERR(cudaMemcpy2D(
						dataHost, 
						widthStepHost * sizeof(type), 
						dataDevice,
						widthStepDevice * sizeof(type), 
						size_buffer, 
						height, 
						(cudaMemcpyKind)kind));
				}
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::cudaCopyDataAsync(int kind, cudaStream_t cuda_stream, bool update_part)
		{
			if (widthStepHost == widthStepDevice && !update_part)
			{
				size_t size_buffer = 0;
				if (!pinned_mem)
				{
					size_buffer = widthStepHost * height * sizeof(type);
				}
				else
				{
					size_buffer = widthStepDevice * heightStepDevice * sizeof(type);
				}

				if (kind == cudaMemcpyHostToDevice)
				{
					cuERR(cudaMemcpyAsync(dataDevice, dataHost, size_buffer, (cudaMemcpyKind)kind, cuda_stream));
				}
				if (kind == cudaMemcpyDeviceToHost)
				{
					cuERR(cudaMemcpyAsync(dataHost, dataDevice, size_buffer, (cudaMemcpyKind)kind, cuda_stream));
				}
			}
			else
			{
				if (kind == cudaMemcpyHostToDevice)
				{
					size_t size_buffer = nChannel * width * sizeof(type);
					cuERR(cudaMemcpy2DAsync(
						dataDevice, 
						widthStepDevice * sizeof(type), 
						dataHost,
						widthStepHost * sizeof(type), 
						size_buffer, 
						height, 
						(cudaMemcpyKind)kind, 
						cuda_stream));
				}
				if (kind == cudaMemcpyDeviceToHost)
				{
					size_t size_buffer = nChannel * width * sizeof(type);
					cuERR(cudaMemcpy2DAsync(
						dataHost,
						widthStepHost * sizeof(type), 
						dataDevice, 
						widthStepDevice * sizeof(type), 
						size_buffer, 
						height, 
						(cudaMemcpyKind)kind, 
						cuda_stream));
				}
			}
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage()
		{
			width = 0;
			height = 0;
			size = 0;
			widthStepHost = 0;
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
			width(_img.width),
			height(_img.height),
			size(_img.size),
			widthStepHost(_img.widthStepHost),
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
		TmpImage<type, pinned_mem>::TmpImage(int _width, int _height, int _nChannel, bool _stride, int _alignHost)
		{
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

				if (_stride)
				{
					size_t pitch;
					size_t size_buffer = width * sizeof(type);
					cuERR(cudaMallocPitch((void**)&dataDevice, &pitch, size_buffer, height));
					widthStepDevice = static_cast<int>(pitch / sizeof(type));
				}
				else
				{
					cuERR(cudaMalloc((void**)&dataDevice, size_buffer));
					widthStepDevice = widthStepHost;
				}
			}
			else
			{
				if (_stride)
				{
					widthStepHost = roundUpMul(widthStepHost, dev_align / sizeof(type));
				}
				widthStepDevice = widthStepHost;
				size_t size_buffer = widthStepDevice * height * sizeof(type);

				if (pinned_mem == 2)
				{
					dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
					cuERR(cudaHostRegister(dataHost, size_buffer, cudaHostRegisterMapped));
					cuERR(cudaHostGetDevicePointer(&dataDevice, dataHost, 0));
				}
				else
				{
					dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
					cuERR(cudaHostRegister(dataHost, size_buffer, 0));
					cuERR(cudaMalloc((void**)&dataDevice, size_buffer));
				}
			}

			heightStepDevice = height;
			offsetDevice = 0;
			erase();
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage(int _width, int _height, int _widthStepHost, int _widthStepDevice, int _alignHost)
		{
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
				cuERR(cudaMalloc((void**)&dataDevice, size_buffer));
			}
			else
			{
				widthStepHost = widthStepDevice;
				size_buffer = widthStepDevice * height * sizeof(type);

				if (pinned_mem == 2)
				{
					dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
					cuERR(cudaHostRegister(dataHost, size_buffer, cudaHostRegisterMapped));
					cuERR(cudaHostGetDevicePointer(&dataDevice, dataHost, 0));
				}
				else
				{
					dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
					cuERR(cudaHostRegister(dataHost, size_buffer, 0));
					cuERR(cudaMalloc((void**)&dataDevice, size_buffer));
				}
			}

			heightStepDevice = height;
			offsetDevice = 0;
			erase();
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage(int _width, int _height, int _widthStepHost, int _widthStepDevice, int _heightStepDivice, int _alignHost)
		{
			width = _width;
			height = _height;
			size = width * height;
			widthStepHost = _widthStepHost;
			widthStepDevice = _widthStepDevice;
			heightStepDevice = _heightStepDivice;
			nChannel = 1;
			alignDataHost = _alignHost;
			sharingDataHost = false;
			sharingDataDevice = false;

			size_t size_buffer = 0;

			if (!pinned_mem)
			{
				size_buffer = widthStepHost * height * sizeof(type);
				dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);

				size_buffer = widthStepDevice * heightStepDevice * sizeof(type);
				cuERR(cudaMalloc((void**)&dataDevice, size_buffer));
			}
			else
			{
				widthStepHost = widthStepDevice;
				size_buffer = widthStepDevice * heightStepDevice * sizeof(type);

				if (pinned_mem == 2)
				{
					dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
					cuERR(cudaHostRegister(dataHost, size_buffer, cudaHostRegisterMapped));
					cuERR(cudaHostGetDevicePointer(&dataDevice, dataHost, 0));
				}
				else
				{
					dataHost = (type*)SIMD::mm_malloc(size_buffer, alignDataHost);
					cuERR(cudaHostRegister(dataHost, size_buffer, 0));
					cuERR(cudaMalloc((void**)&dataDevice, size_buffer));
				}
			}

			offsetDevice = 0;
			erase();
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage(int _width, int _height, int _nChannel, type* _DataHost, int _widthStep)
		{
			width = _width;
			height = _height;
			size = width * height;
			widthStepHost = _widthStep;
			if (widthStepHost == 0) widthStepHost = width * _nChannel;
			widthStepDevice = widthStepHost;
			heightStepDevice = height;
			nChannel = _nChannel;
			dataHost = _DataHost;
			sharingDataHost = true;
			sharingDataDevice = false;

			size_t size_buffer = widthStepDevice * height * sizeof(type);

			if (!pinned_mem)
			{
				cuERR(cudaMalloc((void**)&dataDevice, size_buffer));

				cudaCopyData(cudaMemcpyHostToDevice);
			}
			else
			{
				if (pinned_mem == 2)
				{
					cuERR(cudaHostRegister(dataHost, size_buffer, cudaHostRegisterMapped));
					cuERR(cudaHostGetDevicePointer(&dataDevice, dataHost, 0));
				}
				else
				{
					cuERR(cudaMalloc((void**)&dataDevice, size_buffer));
				}
			}

			offsetDevice = 0;
		}

		template <typename type, const int pinned_mem>
		TmpImage<type, pinned_mem>::TmpImage(int _width, int _height, int _nChannel, type* _DataHost, type* _DataDevice, int _widthStepHost, int _widthStepDevice, int _offsetDevice)
		{
			width = _width;
			height = _height;
			size = width * height;
			widthStepHost = _widthStepHost;
			if (widthStepHost == 0) widthStepHost = width * _nChannel;
			widthStepDevice = _widthStepDevice;
			heightStepDevice = height;
			nChannel = _nChannel;
			dataHost = _DataHost;
			dataDevice = _DataDevice;
			sharingDataHost = true;
			sharingDataDevice = true;
			offsetDevice = _offsetDevice;

			size_t size_buffer = widthStepDevice * height * sizeof(type);

			if (pinned_mem == 1)
			{
				cuERR(cudaHostRegister(dataHost, size_buffer, cudaHostRegisterMapped));
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::copyData(TmpImage<type, pinned_mem>* src, bool async, cudaStream_t cuda_stream)
		{
			const size_t _size = MIN(src->width * src->nChannel, widthStepHost) * sizeof(type);
			for (int j = 0; j < src->height; ++j)
			{
				memcpy(dataHost + j * widthStepHost, src->dataHost + j * src->widthStepHost, _size);
			}

			if (!async)
			{
				cudaCopyData(cudaMemcpyHostToDevice);
			}
			else
			{
				cudaCopyDataAsync(cudaMemcpyHostToDevice, cuda_stream);
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::copyData(SIMD::TmpImage<type>* src, bool async, cudaStream_t cuda_stream)
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
				cudaCopyData(cudaMemcpyHostToDevice);
			}
			else
			{
				cudaCopyDataAsync(cudaMemcpyHostToDevice, cuda_stream);
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
		void TmpImage<type, pinned_mem>::copyDataDevice(SIMD::TmpImage<type>* src, bool async, cudaStream_t cuda_stream)
		{
			if (!async)
			{
				cudaCopyData(cudaMemcpyHostToDevice);
			}
			else
			{
				cudaCopyDataAsync(cudaMemcpyHostToDevice, cuda_stream);
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::copyDataDeviceToDevice(TmpImage* src, bool async, cudaStream_t cuda_stream)
		{
			if (!async)
			{
				cuERR(cudaMemcpy2D(
					dataDevice + offsetDevice,
					widthStepDevice * sizeof(type),
					src->dataDevice + src->offsetDevice,
					src->widthStepHost * sizeof(type),
					src->nChannel * src->width * sizeof(type),
					src->height,
					cudaMemcpyDeviceToDevice));
			}
			else
			{
				cuERR(cudaMemcpy2DAsync(
					dataDevice + offsetDevice,
					widthStepDevice * sizeof(type),
					src->dataDevice + src->offsetDevice,
					src->widthStepHost * sizeof(type),
					src->nChannel * src->width * sizeof(type),
					src->height,
					cudaMemcpyDeviceToDevice,
					cuda_stream));
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::updateDataHost(bool async, cudaStream_t cuda_stream, bool update_part)
		{
			if (!async)
			{
				cudaCopyData(cudaMemcpyDeviceToHost, update_part);
			}
			else
			{
				cudaCopyDataAsync(cudaMemcpyDeviceToHost, cuda_stream, update_part);
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::updateDataDevice(bool async, cudaStream_t cuda_stream, bool update_part)
		{
			if (!async)
			{
				cudaCopyData(cudaMemcpyHostToDevice, update_part);
			}
			else
			{
				cudaCopyDataAsync(cudaMemcpyHostToDevice, cuda_stream, update_part);
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::clone(const TmpImage& img)
		{
			width = img.width;
			height = img.height;
			size = img.size;
			widthStepHost = img.widthStepHost;
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
			size_t size_buffer;

			if (!pinned_mem)
			{
				size_buffer = widthStepHost * height * sizeof(type);
				SIMD::mm_erase(dataHost, (int)size_buffer);

				size_buffer = widthStepDevice * heightStepDevice * sizeof(type);
				cuERR(cudaMemset(dataDevice, 0, size_buffer));
			}
			else
			{
				size_buffer = widthStepDevice * heightStepDevice * sizeof(type);

				if (pinned_mem == 2)
				{
					SIMD::mm_erase(dataHost, (int)size_buffer);
				}
				else
				{
					cuERR(cudaMemset(dataDevice, 0, size_buffer));
				}
			}
		}

		template <typename type, const int pinned_mem>
		void TmpImage<type, pinned_mem>::clear()
		{
			if (!pinned_mem)
			{
				if (dataDevice != 0 && !sharingDataDevice) cuERR(cudaFree(dataDevice));
			}
			else
			{
				if (dataHost != 0) cuERR(cudaHostUnregister(dataHost));

				if (pinned_mem == 1)
				{
					if (dataDevice != 0 && !sharingDataDevice) cuERR(cudaFree(dataDevice));
				}
			}

			width = 0;
			height = 0;
			size = 0;
			widthStepHost = 0;
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
		TmpImage<type, pinned_mem>& TmpImage<type, pinned_mem>::operator=(TmpImage<type, pinned_mem>& _img)
		{
			if (this == &_img) return *this;
			if (!isEmpty()) clear();

			width = _img.width;
			height = _img.height;
			size = _img.size;
			widthStepHost = _img.widthStepHost;
			widthStepDevice = _img.widthStepDevice;
			heightStepDevice = _img.heightStepDevice;
			nChannel = _img.nChannel;
			dataHost = _img.dataHost;
			dataDevice = _img.dataDevice;
			alignDataHost = _img.alignDataHost;
			sharingDataHost = _img.sharingDataHost;
			sharingDataDevice = _img.sharingDataDevice;
			offsetDevice = _img.offsetDevice;

			_img.width = 0;
			_img.height = 0;
			_img.size = 0;
			_img.widthStepHost = 0;
			_img.widthStepDevice = 0;
			_img.heightStepDevice = 0;
			_img.nChannel = 1;
			_img.dataHost = 0;
			_img.dataDevice = 0;
			_img.alignDataHost = 4;
			_img.sharingDataHost = false;
			_img.sharingDataDevice = false;
			_img.offsetDevice = 0;

			return *this;
		}
	}

#endif
}
