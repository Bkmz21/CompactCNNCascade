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
#include <algorithm>
#include <cmath>


//================================================================================================================================================


namespace NeuralNetworksLib
{
	#define MIN(a, b)  ((a) > (b) ? (b) : (a))
	#define MAX(a, b)  ((a) < (b) ? (b) : (a))
	//#define MIN(a, b)  (std::min((a), (b)))
	//#define MAX(a, b)  (std::max((a), (b)))

	#define roundUp(x, k) ((int(x) - 1) / int(k) + 1)
	#define roundUpMul(x, k) int(k) * ((int(x) - 1) / int(k) + 1)
	#define addRoundUpMul(x, k) int(k) * ((int(x) + int(k) - 1) / int(k) + 1)

	#define blockCount(size_data, size_block) ((size_data - 1) / (size_block) + 1)

	#define FB_READ(FNAME, VAL) FNAME.read((char*)&(VAL), sizeof(VAL))
	#define FB_WRITE(FNAME, VAL) FNAME.write((const char*)&(VAL), sizeof(VAL))

	struct Size
	{
		int width = 0;
		int height = 0;

		Size() { }
		Size(int _width, int _height)
		{
			width = _width;
			height = _height;
		}

		inline const Size operator+(const int offset)
		{
			return Size(width + offset, height + offset);
		}
		inline const Size operator*(const float scale)
		{
			return Size(int((float)width * scale), int((float)height * scale));
		}
	};

	struct Size2d
	{
		int	cols = 0;
		int rows = 0;
		int step = 0;
		int size = 0;

		Size2d() { }
		Size2d(int _cols, int _rows, int _step = 0)
		{
			cols = _cols;
			rows = _rows;
			if (_step == 0)
				step = _cols;
			else
				step = _step;
			size = rows * step;
		}
	};

	struct Point
	{
		int x = 0;
		int y = 0;

		Point() { }
		Point(int _x, int _y)
		{
			x = _x;
			y = _y;
		}
	};

	struct Rect
	{
		int x = 0;
		int y = 0;
		int width = 0;
		int height = 0;

		int x2 = 0;
		int y2 = 0;
		int cx = 0;
		int cy = 0;

		Rect() { }
		Rect(int _x, int _y, int _width, int _height)
		{
			x = _x;
			y = _y;
			width = _width;
			height = _height;

			x2 = x + width;
			y2 = y + height;
			cx = (x2 + x) >> 1;
			cy = (y2 + y) >> 1;
		}

		int area()
		{
			return width * height;
		}
		int intersects(Rect& rect)
		{
			if (x > rect.x2 || rect.x > x2 || y > rect.y2 || rect.y > y2)
			{
				return 0;
			}

			const int X1 = MAX(x, rect.x);
			const int X2 = MIN(x2, rect.x2);

			const int Y1 = MAX(y, rect.y);
			const int Y2 = MIN(y2, rect.y2);

			return (X2 - X1) * (Y2 - Y1);
		}
		float overlap(Rect& rect)
		{
			if (x > rect.x2 || rect.x > x2 || y > rect.y2 || rect.y > y2)
			{
				return 0.f;
			}

			const int X0 = MIN(x, rect.x);
			const int X1 = MAX(x, rect.x);
			const int X2 = MIN(x2, rect.x2);
			const int X3 = MAX(x2, rect.x2);

			const int Y0 = MIN(y, rect.y);
			const int Y1 = MAX(y, rect.y);
			const int Y2 = MIN(y2, rect.y2);
			const int Y3 = MAX(y2, rect.y2);

			const float S_union = float((X3 - X0) * (Y3 - Y0));
			const float S_intersection = float((X2 - X1) * (Y2 - Y1));

			return S_intersection / S_union;
		}
		float dist(Rect& rect)
		{
			return sqrtf(static_cast<float>((cx - rect.cx)*(cx - rect.cx) + (cy - rect.cy)*(cy - rect.cy)));
		}

		inline const Rect operator+(int offset)
		{
			if (offset > 0 && 2 * offset >= MAX(width, height))
			{
				offset = MAX(width, height) >> 2;
			}

			return Rect(x + offset, y + offset, width - 2 * offset, height - 2 * offset);
		}
		inline const Rect operator*(float scale)
		{
			scale = 1.f - scale;
			int scale_x = int(float(width) * scale);
			int scale_y = int(float(height) * scale);

			return Rect(x + (scale_x >> 1), y + (scale_y >> 1), width - scale_x, height - scale_y);
		}
	};

}