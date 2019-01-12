///////////////////////////////////////////////////
///source code: http://habrahabr.ru/post/136225 ///
///////////////////////////////////////////////////


#pragma once

#include "type.h"
#include <vector>


//========================================================================================================


namespace NeuralNetworksLib
{
	class Packing2D
	{
	private:
		int STRIPH = 1;
		int STRIPW = 1280;

		class Level
		{
		public:
			Level(int b, int h, int f, int w, int stripW, int stripH) : bottom(b),
				height(h),
				floor(f),
				initW(w),
				ceiling(0),
				STRIPW(stripW),
				STRIPH(stripH){}

			const Rect put(const Rect& rect, bool f = true, bool leftJustified = true);
			bool ceilingFeasible(const Rect& rect, const std::vector<Rect> existing);
			bool floorFeasible(const Rect& rect);
			int getSpace(bool f = true);

			int bottom;
			int height;
			int floor;
			int initW;
			int ceiling;
			int STRIPW;
			int STRIPH;
		};

	public:
		Packing2D() { }
		~Packing2D() { }

		float packing(std::vector<Rect>* packed, Size* packed_size, std::vector<float>& scales, Size& max_size, int strip_width = 0);
	};

}