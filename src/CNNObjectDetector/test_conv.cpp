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


#include "image.h"
#ifdef USE_FIXED_POINT
#	include "cnnpp_simd_avx_v3.h"
#else
#	include "cnnpp_simd_avx_v2.h"
#endif
#include "timer.h"

#include <iostream>

using namespace NeuralNetworksLib;


//================================================================================================================================================


int main(int argc, char* argv[])
{
	int R;

	std::cout << "Enter image size: ";
	std::cin >> R;

	if (R <= 0)
		R = 156;
	printf("R = %d\n", R);

	Size2d input_size(4 * R, 2 * R);
	SIMD::Array_32f input_buff(input_size.size, ALIGN_DEF);
	for (int j = 0; j < input_size.rows; ++j) {
		for (int i = 0; i < input_size.cols; ++i) {
			input_buff.data[j * input_size.step + i] = i % 4;
		}
	}

	Size2d output_size(8 * R / 2, R);
	SIMD::Array_32f output_buff(input_size.size, ALIGN_DEF);

	SIMD::Array_32f kernels(8 * 3 * 3, ALIGN_DEF), bias(8, ALIGN_DEF);
	SIMD::Array_32f lrelu_w1(8, ALIGN_DEF), lrelu_w2(8, ALIGN_DEF);
	SIMD::Array_32f bn_w(8, ALIGN_DEF), bn_b(8, ALIGN_DEF);
	for (int i = 0; i < kernels.size; ++i) {
		kernels.data[i] = 1.f;
	}
	for (int i = 0; i < bias.size; ++i) {
		bias.data[i] = 1.f;
		lrelu_w1.data[i] = 1.f;
		lrelu_w2.data[i] = 1.f;
		bn_w.data[i] = 1.f;
		bn_b.data[i] = 0.f;
	}

#ifdef USE_FIXED_POINT
	SIMD::CNNPP_v3 cnnpp;
#else
	SIMD::CNNPP_v2 cnnpp;
#endif

	Timer timer(1, true);
	for (int i = 0; i < 10000; ++i)
	cnnpp.conv_3x3_lrelu_bn_max(
		output_buff.data, output_size.step, 
		input_buff.data, input_size.step, input_size.rows, 
		kernels.data, bias.data, 
		lrelu_w1.data, lrelu_w2.data, 
		bn_w.data, bn_b.data, 
		R - 2, R - 2);
	timer.print(0.1);

	for (int j = 0; j < std::min(800, output_size.rows); ++j) {
		for (int i = 0; i < std::min(800, output_size.cols); ++i) {
			printf("%1.0f ", output_buff.data[j * output_size.step + i]);
		}
		printf("\n");
	}

	std::system("pause");
}