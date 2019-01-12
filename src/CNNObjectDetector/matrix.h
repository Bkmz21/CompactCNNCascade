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

#include <vector>


//================================================================================================================================================


namespace NeuralNetworksLib
{

	namespace Legacy
	{
		template <typename type>
		class Matrix
		{
		protected:
			std::vector<type>* v;
			int cols;
			int rows;

		public:
			Matrix() { v = NULL; rows = 0; cols = 0; }
			Matrix(int m, int n);
			~Matrix() { }

			std::vector<type>& operator [](int i) { return v[i]; }

			int row_count()  { return rows; }
			int col_count()  { return cols; }
			type**& GetPtr() { return v; }

			void clear();
		};


		template <typename type>
		class ArrayMatrix : public std::vector < Matrix<type> >
		{
		public:
			ArrayMatrix() : std::vector<Matrix<type>>() { }
			ArrayMatrix(int rows, int cols, int depth);
			~ArrayMatrix() { }

			void clear();
		};


		//----------------------------------------------------------


		template <typename type>
		Matrix<type>::Matrix(int m, int n)
		{
			rows = m;
			cols = n;
			v = new std::vector<type>[rows];
			for (int i = 0; i < rows; ++i)
			{
				v[i] = std::vector<type>(cols);
			}
		}

		template <typename type>
		void Matrix<type>::clear()
		{
			if (v != 0)
			{
				for (int i = 0; i < rows; ++i)
				{
					v[i].clear();
				}
				delete[] v;
				v = NULL;
				rows = 0;
				cols = 0;
			}
		}


		template <typename type>
		ArrayMatrix<type>::ArrayMatrix(int rows, int cols, int depth)
		{
			this->reserve(depth);
			for (int i = 0; i < depth; ++i)
			{
				this->push_back(Matrix<type>(rows, cols));
			}
		}

		template <typename type>
		void ArrayMatrix<type>::clear()
		{
			for (int i = 0; i < (int)this->size(); ++i)
			{
				(*this)[i].clear();
			}
			this->erase(this->begin(), this->end());
		}
	}

}

