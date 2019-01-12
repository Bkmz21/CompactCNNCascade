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
#include "image.h"


//================================================================================================================================================


namespace NeuralNetworksLib
{
	namespace SIMD
	{
		//only one channel Image_32f
		class ImageResizer
		{
		private:
			Array_32u pyLine;
			Array_32u pxLine;
			Array_32f ayLUT;
			Array_32f axLUT;

			Size src_img_size;
			Size dst_img_size;

			void preprocessing();
			void clear();
			inline void checkSize(const Size& _dst_img_size, const Size& _src_img_size);

			// fast bilinear interpolation for image resize by fixed-point + LUT optimization
			// only support one channel image
			void NearestNeighborInterpolation(Image_8u& dst, Image_8u& src, int num_threads = 1);
			void BilinearInterpolation(Image_8u& dst, Image_8u& src, int num_threads = 1);

			void NearestNeighborInterpolation(Image_32f& dst, Image_32f& src, int num_threads = 1);
			void BilinearInterpolation(Image_32f& dst, Image_32f& src, int num_threads = 1);

		public:
			ImageResizer();
			ImageResizer(Size _dst_img_size, Size _src_img_size);
			~ImageResizer() { clear(); }

			void FastImageResize(Image_8u& dst, Image_8u& src, const int type_resize, int num_threads = 1);
			void FastImageResize(Image_32f& dst, Image_32f& src, const int type_resize, int num_threads = 1);
			void getLineIndexes(uint_*& _pxLine, uint_*& _pyLine, const Size& _dst_img_size, const Size& _src_img_size);
		};
	}
}