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
#include <memory.h>
#include <algorithm>


//================================================================================================================================================


namespace NeuralNetworksLib
{
	namespace SIMD
	{
#define getPosImage(Image, x, y) Image.data[(y) * Image.widthStep + (x)]

		void* mm_malloc(size_t size, size_t align);
		void  mm_free(void* p);
		void  mm_erase(void* p, int size);

		template <typename type>
		class TmpImage
		{
		public:
			int width = 0;
			int height = 0;
			int size = 0;
			int widthStep = 0;
			int nChannel = 0;
			type* data = nullptr;
			int alignData = 0;
			bool sharingData = false;		
			struct Indices
			{
				uint_* px = nullptr;
				uint_* py = nullptr;
			} indices;

			TmpImage();
			TmpImage(const TmpImage<type>& _img);
			TmpImage(int _width, int _height, int _align = 0, bool _stride = false);
			TmpImage(int _width, int _height, int _widthStep, int _align);
			TmpImage(int _width, int _height, int _nChannel, int _align, bool _stride);
			TmpImage(int _width, int _height, int _nChannel, type* _data, int _widthStep = 0);
			~TmpImage() { clear(); }
			void copyData(TmpImage<type>& _src);
			void copyData(int _width, int _height, type* _data, int _widthStep = 0);
			void clone(const TmpImage& _img);
			void erase();
			void clear();
			inline bool isEmpty() const { return data == 0; }
			inline void setSize(const Size& size) { width = size.width; height = size.height; }
			inline Size getSize() const { return Size(width, height); }
			TmpImage<type>& operator=(const TmpImage<type>& _img);
			inline type* operator()(size_t offset = 0) const;
			inline type& operator[](size_t idx);
		};
		typedef TmpImage<uchar_> Image_8u;
		typedef TmpImage<float> Image_32f;


		//----------------------------------------------------------


		template <typename type>
		TmpImage<type>::TmpImage()
		{
			width = 0;
			height = 0;
			size = 0;
			widthStep = 0;
			nChannel = 1;
			data = 0;
			alignData = 4;
			sharingData = false;
		}

		template <typename type>
		TmpImage<type>::TmpImage(const TmpImage<type>& _img) :
			width(_img.width),
			height(_img.height),
			size(_img.size),
			widthStep(_img.widthStep),
			nChannel(_img.nChannel),
			data(_img.data),
			alignData(_img.alignData),
			sharingData(_img.sharingData),
			indices(_img.indices)
		{ }

		template <typename type>
		TmpImage<type>::TmpImage(int _width, int _height, int _align, bool _stride)
		{
			width = _width;
			height = _height;
			size = width * height;
			nChannel = 1;
			widthStep = width * nChannel;
			alignData = roundUpMul(_align > 0 ? _align : sizeof(type), sizeof(type));
			sharingData = false;
			if (_stride)
			{
				widthStep += roundUpMul(width, alignData / sizeof(type)) - width;
			}

			size_t size_buffer = widthStep * height * sizeof(type);
			data = (type*)mm_malloc(size_buffer, alignData);

			erase();
		}

		template <typename type>
		TmpImage<type>::TmpImage(int _width, int _height, int _widthStep, int _align)
		{
			width = _width;
			height = _height;
			size = width * height;
			nChannel = 1;
			widthStep = _widthStep;
			alignData = _align;
			sharingData = false;

			size_t size_buffer = widthStep * height * sizeof(type);
			data = (type*)mm_malloc(size_buffer, alignData);

			erase();
		}

		template <typename type>
		TmpImage<type>::TmpImage(int _width, int _height, int _nChannel, int _align, bool _stride)
		{
			width = _width;
			height = _height;
			size = width * height;
			nChannel = _nChannel;
			widthStep = width * nChannel;
			alignData = roundUpMul(_align > 0 ? _align : sizeof(type), sizeof(type));
			sharingData = false;
			if (_stride)
			{
				widthStep += roundUpMul(width, alignData / sizeof(type)) - width;
			}

			size_t size_buffer = widthStep * height * sizeof(type);
			data = (type*)mm_malloc(size_buffer, alignData);

			erase();
		}

		template <typename type>
		TmpImage<type>::TmpImage(int _width, int _height, int _nChannel, type* _data, int _widthStep)
		{
			width = _width;
			height = _height;
			size = width * height;
			nChannel = _nChannel;
			widthStep = _widthStep;
			if (widthStep == 0) widthStep = width;
			data = _data;
			alignData = 4;
			sharingData = true;
		}

		template <typename type>
		void TmpImage<type>::copyData(TmpImage<type>& _src)
		{
			const size_t _size = _src.width * sizeof(type);
			for (int j = 0; j < _src.height; ++j)
			{
				memcpy(data + j * widthStep, _src.data + j * _src.widthStep, _size);
			}
		}

		template <typename type>
		void TmpImage<type>::copyData(int _width, int _height, type* _data, int _widthStep)
		{
			if (_widthStep == 0) _widthStep = _width;
			const int _size = _width * sizeof(type);
			for (int j = 0; j < _height; ++j)
			{
				memcpy(data + j * widthStep, _data + j * _widthStep, _size);
			}
		}

		template <typename type>
		void TmpImage<type>::clone(const TmpImage& _img)
		{
			width = _img.width;
			height = _img.height;
			size = _img.size;
			widthStep = _img.widthStep;
			nChannel = _img.nChannel;
			data = _img.data;
			alignData = _img.alignData;
			sharingData = true;
			indices = _img.indices;
		}

		template <typename type>
		void TmpImage<type>::erase()
		{
			int size_buffer = widthStep * height * sizeof(type);
			mm_erase(data, size_buffer);
		}

		template <typename type>
		void TmpImage<type>::clear()
		{
			width = 0;
			height = 0;
			size = 0;
			widthStep = 0;
			nChannel = 0;
			alignData = 4;
			if (data != 0 && !sharingData) mm_free(data);
			data = 0;
			sharingData = false;
			indices = Indices();
		}

		template <typename type>
		TmpImage<type>& TmpImage<type>::operator=(const TmpImage<type>& _img)
		{
			if (this == &_img) return *this;
			if (!isEmpty()) clear();

			width = _img.width;
			height = _img.height;
			size = _img.size;
			widthStep = _img.widthStep;
			nChannel = _img.nChannel;
			data = _img.data;
			alignData = _img.alignData;
			sharingData = _img.sharingData;
			indices = _img.indices;

			TmpImage<type>* pimg = const_cast<TmpImage<type>*>(&_img);
			pimg->width = 0;
			pimg->height = 0;
			pimg->size = 0;
			pimg->widthStep = 0;
			pimg->nChannel = 1;
			pimg->data = 0;
			pimg->alignData = 4;
			pimg->sharingData = false;
			pimg->indices = Indices();

			return *this;
		}

		template <typename type>
		type* TmpImage<type>::operator()(size_t offset) const
		{
			return data + offset;
		}

		template <typename type>
		type& TmpImage<type>::operator[](size_t idx)
		{
			return data[idx];
		}

		//----------------------------------------------------------

		template <typename type>
		class TmpArray : public TmpImage<type>
		{
		public:
			TmpArray() : TmpImage<type>() { }
			TmpArray(int _size, int _align = 0) : TmpImage<type>(_size, 1, _align) { }     
			TmpArray<type>& operator=(const TmpArray<type>& _img)
			{ 
				*((TmpImage<type>*)this) = (TmpImage<type>&)_img;
				return *this;
			};
		};
		typedef TmpArray<uchar_> Array_8u;
		typedef TmpArray<uint_> Array_32u;
		typedef TmpArray<float> Array_32f;

		//----------------------------------------------------------

		template <typename type>
		class TmpRef
		{
		private:
			type** ref = NULL;

		public:
			TmpRef() : ref(0) { }
			TmpRef(TmpArray<type>* _data, size_t _count)
			{
				ref = new type*[_count];
				for (size_t i = 0; i < _count; ++i)
				{
					ref[i] = _data[i]();
				}
			}
			TmpRef(const TmpRef<type>& _ref) : ref(_ref.ref) { }
			~TmpRef() { clear(); }
			bool isEmpty() const { return ref == 0; }
			void clear()
			{
				if (ref != 0)
				{
					delete[] ref;
					ref = 0;
				}
			}
			TmpRef<type>& operator=(const TmpRef<type>& _ref)
			{
				if (this == &_ref) return *this;
				if (!isEmpty()) clear();

				ref = _ref.ref;
				
				TmpRef<type>* pref = const_cast<TmpRef<type>*>(&_ref);
				pref->ref = 0;

				return *this;
			}
			inline type** operator()(size_t offset = 0) { return ref + offset; }
		};
		typedef TmpRef<float> Array_32f_ref;

	}
}
