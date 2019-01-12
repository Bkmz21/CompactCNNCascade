///////////////////////////////////////////////////
///source code: http://habrahabr.ru/post/136225 ///
///////////////////////////////////////////////////


#include "packing_2D.h"
//#include <opencv2/opencv.hpp>


//================================================================================================================================================


namespace NeuralNetworksLib
{

	float Packing2D::packing(std::vector<Rect>* packed, Size* packed_size, std::vector<float>& scales, Size& max_size, int strip_width)
	{
		if (strip_width > 0)
		{
			STRIPW = strip_width;
		}

		packed->clear();

		std::vector<Rect> unpacked;
		for (int scl = 0; scl < (int)scales.size(); ++scl)
		{
			Size img_resize = max_size * scales[scl];
			unpacked.push_back(Rect(0, 0, roundUpMul(img_resize.width, 4), roundUpMul(img_resize.height, 4)));
		}

		std::vector<Level> levels;
		Level level(0, unpacked[0].height, 0, unpacked[0].width, STRIPW, STRIPH);

		packed->push_back(level.put(unpacked[0]));
		levels.push_back(level);

		for (int i = 1; i < unpacked.size(); i++)
		{
			int found = -1;
			int min = STRIPW;
			for (int j = 0; j < levels.size(); j++)
			{
				if (levels[j].floorFeasible(unpacked[i]))
				{
					if (levels[j].getSpace() < min)
					{
						found = j;
						min = levels[j].getSpace();
					}
				}
			}
			if (found > -1)
			{ // floor-pack on existing level
				packed->push_back(levels[found].put(unpacked[i]));
			}
			else
			{
				int found = -1;
				int min = STRIPW;
				for (int j = 0; j < levels.size(); j++)
				{
					if (levels[j].ceilingFeasible(unpacked[i],* packed))
					{
						if (levels[j].getSpace(false) < min)
						{
							found = j;
							min = levels[j].getSpace(false);
						}
					}
				}
				if (found > -1)
				{ // ceiling-pack on existing level
					packed->push_back(levels[found].put(unpacked[i], false));
				}
				else
				{ // a new level
					Level newLevel(levels[levels.size() - 1].bottom + levels[levels.size() - 1].height, unpacked[i].height, 0, unpacked[i].width, STRIPW, STRIPH);
					packed->push_back(newLevel.put(unpacked[i]));
					levels.push_back(newLevel);
				}
			}
		}

		int max_x = 0;
		int max_y = 0;
		int area = 0;
		for (auto it = packed->begin(); it != packed->end(); ++it)
		{
			it->y = abs(it->y);
			it->y2 = abs(it->y2);
			it->cy = abs(it->cy);
			if (it->y > it->y2)
			{
				std::swap(it->y, it->y2);
			}
			area += it->width * it->height;
			max_x = MAX(it->x2, max_x);
			max_y = MAX(it->y2, max_y);
		}

		packed_size->width = max_x;
		packed_size->height = max_y;
		float area_ratio = float(area) / float(max_x * max_y);

		//display
		//cv::Mat img = cv::Mat(cv::Size(max_x, max_y), CV_8UC3);
		//cv::RNG rng = cv::RNG(0xffffffff);
		//for (int i = 0; i < packed->size(); ++i)
		//{
		//	cv::rectangle(img, cv::Point((*packed)[i].x, (*packed)[i].y), cv::Point((*packed)[i].x2, (*packed)[i].y2), cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), cv::FILLED, 8, 0);
		//}
		//cv::namedWindow("rect", cv::WINDOW_NORMAL);
		//cv::imshow("rect", img);
		//cv::imwrite("packing.png", img);
		//cv::waitKey(100);

		return area_ratio;
	}

	const Rect Packing2D::Level::put(const Rect& rect, bool f, bool leftJustified)
	{
		Rect newRect;

		if (f)
		{
			if (leftJustified)
			{
				newRect = Rect(floor,
					STRIPH - (bottom + rect.height + 1),
					rect.width,
					rect.height);
			}
			else
			{
				// 'ceiling' is used for right-justified rectangles packed on the floor
				newRect = Rect(STRIPW - (ceiling + rect.width),
					STRIPH - (bottom + rect.height + 1),
					rect.width,
					rect.height);
				ceiling += rect.width;
			}
			floor += rect.width;
		}
		else
		{
			newRect = Rect(STRIPW - (ceiling + rect.width),
				STRIPH - (bottom + height + 1),
				rect.width,
				rect.height);
			ceiling += rect.width;
		}

		return newRect;
	}
	bool Packing2D::Level::ceilingFeasible(const Rect& rect, const std::vector<Rect> existing)
	{
		Rect testRect;
		testRect = Rect(STRIPW - (ceiling + rect.width),
			STRIPH - (bottom + height + 1),
			rect.width,
			rect.height);

		bool intersected = false;
		for (int i = 0; i < existing.size(); i++)
		{
			Rect rect = existing[i];
			if (testRect.intersects(rect) > 0)
			{
				intersected = true;
				break;
			}
		}
		bool fit = rect.width <= (STRIPW - ceiling - initW);
		return fit && !intersected;
	}
	bool Packing2D::Level::floorFeasible(const Rect& rect)
	{
		return rect.width <= (STRIPW - floor);
	}
	int Packing2D::Level::getSpace(bool f)
	{
		if (f)
		{
			return STRIPW - floor;
		}
		else
		{
			return STRIPW - ceiling - initW;
		}
	}

}