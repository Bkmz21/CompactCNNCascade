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


//================================================================================================================================================


typedef struct tag_iSize 
{ 
	int cols;
	int rows;
	int step;
	int size;
} Size2d;

float tanhf(const float x)
{
	float sgn = 1.f;
	if (x < 0.f) sgn = -1.f;
	return sgn * (1.f - 1.f / (1.f + fabs(x) + x * x + 1.41645f * x * x * x * x));
}

__kernel void conv_4x4x4_max_tanh_tanh_cl(
										/*0*/__global float* dst_0,					
										/*1*/__global float* dst_1,
										/*2*/__global float* dst_2,
										/*3*/__global float* dst_3,
										/*4*/int dst_step,
										/*5*/__global float* src_L1,
										/*6*/int src_cols,
										/*7*/int src_rows,
										/*8*/int src_step,
										/*9*/__constant float* kernels_L1_0,
										/*10*/__constant float* kernels_L1_1,
										/*11*/__constant float* kernels_L1_2,
										/*12*/__constant float* kernels_L1_3,
										/*13*/__constant float* conv_b_L1,
										/*14*/__constant float* subs_w_L1,
										/*15*/__constant float* subs_b_L1,
										/*16*/__constant float* scale_L1
										)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	const int dst_offset = j * dst_step + i;

	i *= 2;
	j *= 2;
	if (i + 1 >= src_cols - 3 || j + 1 >= src_rows - 3) return;
	const int src_offset = j * src_step + i;

	float temp[10];

	src_L1 += src_offset;
	temp[0 * 5 + 0] = *(src_L1 + 0);
	temp[0 * 5 + 1] = *(src_L1 + 1);
	temp[0 * 5 + 2] = *(src_L1 + 2);
	temp[0 * 5 + 3] = *(src_L1 + 3);
	temp[0 * 5 + 4] = *(src_L1 + 4);

	src_L1 += src_step;
	temp[1 * 5 + 0] = *(src_L1 + 0);
	temp[1 * 5 + 1] = *(src_L1 + 1);
	temp[1 * 5 + 2] = *(src_L1 + 2);
	temp[1 * 5 + 3] = *(src_L1 + 3);
	temp[1 * 5 + 4] = *(src_L1 + 4);

	float c[4 * 4];
	{
		c[4 * 0 + 0] = temp[0 * 5 + 0] * kernels_L1_0[0]
			+ temp[0 * 5 + 1] * kernels_L1_0[1]
			+ temp[0 * 5 + 2] * kernels_L1_0[2]
			+ temp[0 * 5 + 3] * kernels_L1_0[3];

		c[4 * 0 + 1] = temp[0 * 5 + 1] * kernels_L1_0[0]
			+ temp[0 * 5 + 2] * kernels_L1_0[1]
			+ temp[0 * 5 + 3] * kernels_L1_0[2]
			+ temp[0 * 5 + 4] * kernels_L1_0[3];

		c[4 * 0 + 2] = temp[1 * 5 + 0] * kernels_L1_0[0]
			+ temp[1 * 5 + 1] * kernels_L1_0[1]
			+ temp[1 * 5 + 2] * kernels_L1_0[2]
			+ temp[1 * 5 + 3] * kernels_L1_0[3];

		c[4 * 0 + 3] = temp[1 * 5 + 1] * kernels_L1_0[0]
			+ temp[1 * 5 + 2] * kernels_L1_0[1]
			+ temp[1 * 5 + 3] * kernels_L1_0[2]
			+ temp[1 * 5 + 4] * kernels_L1_0[3];
	}
	{
		c[4 * 1 + 0] = temp[0 * 5 + 0] * kernels_L1_1[0]
			+ temp[0 * 5 + 1] * kernels_L1_1[1]
			+ temp[0 * 5 + 2] * kernels_L1_1[2]
			+ temp[0 * 5 + 3] * kernels_L1_1[3];

		c[4 * 1 + 1] = temp[0 * 5 + 1] * kernels_L1_1[0]
			+ temp[0 * 5 + 2] * kernels_L1_1[1]
			+ temp[0 * 5 + 3] * kernels_L1_1[2]
			+ temp[0 * 5 + 4] * kernels_L1_1[3];

		c[4 * 1 + 2] = temp[1 * 5 + 0] * kernels_L1_1[0]
			+ temp[1 * 5 + 1] * kernels_L1_1[1]
			+ temp[1 * 5 + 2] * kernels_L1_1[2]
			+ temp[1 * 5 + 3] * kernels_L1_1[3];

		c[4 * 1 + 3] = temp[1 * 5 + 1] * kernels_L1_1[0]
			+ temp[1 * 5 + 2] * kernels_L1_1[1]
			+ temp[1 * 5 + 3] * kernels_L1_1[2]
			+ temp[1 * 5 + 4] * kernels_L1_1[3];
	}
	{
		c[4 * 2 + 0] = temp[0 * 5 + 0] * kernels_L1_2[0]
			+ temp[0 * 5 + 1] * kernels_L1_2[1]
			+ temp[0 * 5 + 2] * kernels_L1_2[2]
			+ temp[0 * 5 + 3] * kernels_L1_2[3];

		c[4 * 2 + 1] = temp[0 * 5 + 1] * kernels_L1_2[0]
			+ temp[0 * 5 + 2] * kernels_L1_2[1]
			+ temp[0 * 5 + 3] * kernels_L1_2[2]
			+ temp[0 * 5 + 4] * kernels_L1_2[3];

		c[4 * 2 + 2] = temp[1 * 5 + 0] * kernels_L1_2[0]
			+ temp[1 * 5 + 1] * kernels_L1_2[1]
			+ temp[1 * 5 + 2] * kernels_L1_2[2]
			+ temp[1 * 5 + 3] * kernels_L1_2[3];

		c[4 * 2 + 3] = temp[1 * 5 + 1] * kernels_L1_2[0]
			+ temp[1 * 5 + 2] * kernels_L1_2[1]
			+ temp[1 * 5 + 3] * kernels_L1_2[2]
			+ temp[1 * 5 + 4] * kernels_L1_2[3];
	}
	{
		c[4 * 3 + 0] = temp[0 * 5 + 0] * kernels_L1_3[0]
			+ temp[0 * 5 + 1] * kernels_L1_3[1]
			+ temp[0 * 5 + 2] * kernels_L1_3[2]
			+ temp[0 * 5 + 3] * kernels_L1_3[3];

		c[4 * 3 + 1] = temp[0 * 5 + 1] * kernels_L1_3[0]
			+ temp[0 * 5 + 2] * kernels_L1_3[1]
			+ temp[0 * 5 + 3] * kernels_L1_3[2]
			+ temp[0 * 5 + 4] * kernels_L1_3[3];

		c[4 * 3 + 2] = temp[1 * 5 + 0] * kernels_L1_3[0]
			+ temp[1 * 5 + 1] * kernels_L1_3[1]
			+ temp[1 * 5 + 2] * kernels_L1_3[2]
			+ temp[1 * 5 + 3] * kernels_L1_3[3];

		c[4 * 3 + 3] = temp[1 * 5 + 1] * kernels_L1_3[0]
			+ temp[1 * 5 + 2] * kernels_L1_3[1]
			+ temp[1 * 5 + 3] * kernels_L1_3[2]
			+ temp[1 * 5 + 4] * kernels_L1_3[3];
	}

	src_L1 += src_step;
	temp[0 * 5 + 0] = *(src_L1 + 0);
	temp[0 * 5 + 1] = *(src_L1 + 1);
	temp[0 * 5 + 2] = *(src_L1 + 2);
	temp[0 * 5 + 3] = *(src_L1 + 3);
	temp[0 * 5 + 4] = *(src_L1 + 4);

	{
		c[4 * 0 + 0] += temp[1 * 5 + 0] * kernels_L1_0[4]
			+ temp[1 * 5 + 1] * kernels_L1_0[5]
			+ temp[1 * 5 + 2] * kernels_L1_0[6]
			+ temp[1 * 5 + 3] * kernels_L1_0[7];

		c[4 * 0 + 1] += temp[1 * 5 + 1] * kernels_L1_0[4]
			+ temp[1 * 5 + 2] * kernels_L1_0[5]
			+ temp[1 * 5 + 3] * kernels_L1_0[6]
			+ temp[1 * 5 + 4] * kernels_L1_0[7];

		c[4 * 0 + 2] += temp[0 * 5 + 0] * kernels_L1_0[4]
			+ temp[0 * 5 + 1] * kernels_L1_0[5]
			+ temp[0 * 5 + 2] * kernels_L1_0[6]
			+ temp[0 * 5 + 3] * kernels_L1_0[7];

		c[4 * 0 + 3] += temp[0 * 5 + 1] * kernels_L1_0[4]
			+ temp[0 * 5 + 2] * kernels_L1_0[5]
			+ temp[0 * 5 + 3] * kernels_L1_0[6]
			+ temp[0 * 5 + 4] * kernels_L1_0[7];
	}
	{
		c[4 * 1 + 0] += temp[1 * 5 + 0] * kernels_L1_1[4]
			+ temp[1 * 5 + 1] * kernels_L1_1[5]
			+ temp[1 * 5 + 2] * kernels_L1_1[6]
			+ temp[1 * 5 + 3] * kernels_L1_1[7];

		c[4 * 1 + 1] += temp[1 * 5 + 1] * kernels_L1_1[4]
			+ temp[1 * 5 + 2] * kernels_L1_1[5]
			+ temp[1 * 5 + 3] * kernels_L1_1[6]
			+ temp[1 * 5 + 4] * kernels_L1_1[7];

		c[4 * 1 + 2] += temp[0 * 5 + 0] * kernels_L1_1[4]
			+ temp[0 * 5 + 1] * kernels_L1_1[5]
			+ temp[0 * 5 + 2] * kernels_L1_1[6]
			+ temp[0 * 5 + 3] * kernels_L1_1[7];

		c[4 * 1 + 3] += temp[0 * 5 + 1] * kernels_L1_1[4]
			+ temp[0 * 5 + 2] * kernels_L1_1[5]
			+ temp[0 * 5 + 3] * kernels_L1_1[6]
			+ temp[0 * 5 + 4] * kernels_L1_1[7];
	}
	{
		c[4 * 2 + 0] += temp[1 * 5 + 0] * kernels_L1_2[4]
			+ temp[1 * 5 + 1] * kernels_L1_2[5]
			+ temp[1 * 5 + 2] * kernels_L1_2[6]
			+ temp[1 * 5 + 3] * kernels_L1_2[7];

		c[4 * 2 + 1] += temp[1 * 5 + 1] * kernels_L1_2[4]
			+ temp[1 * 5 + 2] * kernels_L1_2[5]
			+ temp[1 * 5 + 3] * kernels_L1_2[6]
			+ temp[1 * 5 + 4] * kernels_L1_2[7];

		c[4 * 2 + 2] += temp[0 * 5 + 0] * kernels_L1_2[4]
			+ temp[0 * 5 + 1] * kernels_L1_2[5]
			+ temp[0 * 5 + 2] * kernels_L1_2[6]
			+ temp[0 * 5 + 3] * kernels_L1_2[7];

		c[4 * 2 + 3] += temp[0 * 5 + 1] * kernels_L1_2[4]
			+ temp[0 * 5 + 2] * kernels_L1_2[5]
			+ temp[0 * 5 + 3] * kernels_L1_2[6]
			+ temp[0 * 5 + 4] * kernels_L1_2[7];
	}
	{
		c[4 * 3 + 0] += temp[1 * 5 + 0] * kernels_L1_3[4]
			+ temp[1 * 5 + 1] * kernels_L1_3[5]
			+ temp[1 * 5 + 2] * kernels_L1_3[6]
			+ temp[1 * 5 + 3] * kernels_L1_3[7];

		c[4 * 3 + 1] += temp[1 * 5 + 1] * kernels_L1_3[4]
			+ temp[1 * 5 + 2] * kernels_L1_3[5]
			+ temp[1 * 5 + 3] * kernels_L1_3[6]
			+ temp[1 * 5 + 4] * kernels_L1_3[7];

		c[4 * 3 + 2] += temp[0 * 5 + 0] * kernels_L1_3[4]
			+ temp[0 * 5 + 1] * kernels_L1_3[5]
			+ temp[0 * 5 + 2] * kernels_L1_3[6]
			+ temp[0 * 5 + 3] * kernels_L1_3[7];

		c[4 * 3 + 3] += temp[0 * 5 + 1] * kernels_L1_3[4]
			+ temp[0 * 5 + 2] * kernels_L1_3[5]
			+ temp[0 * 5 + 3] * kernels_L1_3[6]
			+ temp[0 * 5 + 4] * kernels_L1_3[7];
	}

	src_L1 += src_step;
	temp[1 * 5 + 0] = *(src_L1 + 0);
	temp[1 * 5 + 1] = *(src_L1 + 1);
	temp[1 * 5 + 2] = *(src_L1 + 2);
	temp[1 * 5 + 3] = *(src_L1 + 3);
	temp[1 * 5 + 4] = *(src_L1 + 4);

	{
		c[4 * 0 + 0] += temp[0 * 5 + 0] * kernels_L1_0[8]
			+ temp[0 * 5 + 1] * kernels_L1_0[9]
			+ temp[0 * 5 + 2] * kernels_L1_0[10]
			+ temp[0 * 5 + 3] * kernels_L1_0[11];

		c[4 * 0 + 1] += temp[0 * 5 + 1] * kernels_L1_0[8]
			+ temp[0 * 5 + 2] * kernels_L1_0[9]
			+ temp[0 * 5 + 3] * kernels_L1_0[10]
			+ temp[0 * 5 + 4] * kernels_L1_0[11];

		c[4 * 0 + 2] += temp[1 * 5 + 0] * kernels_L1_0[8]
			+ temp[1 * 5 + 1] * kernels_L1_0[9]
			+ temp[1 * 5 + 2] * kernels_L1_0[10]
			+ temp[1 * 5 + 3] * kernels_L1_0[11];

		c[4 * 0 + 3] += temp[1 * 5 + 1] * kernels_L1_0[8]
			+ temp[1 * 5 + 2] * kernels_L1_0[9]
			+ temp[1 * 5 + 3] * kernels_L1_0[10]
			+ temp[1 * 5 + 4] * kernels_L1_0[11];
	}
	{
		c[4 * 1 + 0] += temp[0 * 5 + 0] * kernels_L1_1[8]
			+ temp[0 * 5 + 1] * kernels_L1_1[9]
			+ temp[0 * 5 + 2] * kernels_L1_1[10]
			+ temp[0 * 5 + 3] * kernels_L1_1[11];

		c[4 * 1 + 1] += temp[0 * 5 + 1] * kernels_L1_1[8]
			+ temp[0 * 5 + 2] * kernels_L1_1[9]
			+ temp[0 * 5 + 3] * kernels_L1_1[10]
			+ temp[0 * 5 + 4] * kernels_L1_1[11];

		c[4 * 1 + 2] += temp[1 * 5 + 0] * kernels_L1_1[8]
			+ temp[1 * 5 + 1] * kernels_L1_1[9]
			+ temp[1 * 5 + 2] * kernels_L1_1[10]
			+ temp[1 * 5 + 3] * kernels_L1_1[11];

		c[4 * 1 + 3] += temp[1 * 5 + 1] * kernels_L1_1[8]
			+ temp[1 * 5 + 2] * kernels_L1_1[9]
			+ temp[1 * 5 + 3] * kernels_L1_1[10]
			+ temp[1 * 5 + 4] * kernels_L1_1[11];
	}
	{
		c[4 * 2 + 0] += temp[0 * 5 + 0] * kernels_L1_2[8]
			+ temp[0 * 5 + 1] * kernels_L1_2[9]
			+ temp[0 * 5 + 2] * kernels_L1_2[10]
			+ temp[0 * 5 + 3] * kernels_L1_2[11];

		c[4 * 2 + 1] += temp[0 * 5 + 1] * kernels_L1_2[8]
			+ temp[0 * 5 + 2] * kernels_L1_2[9]
			+ temp[0 * 5 + 3] * kernels_L1_2[10]
			+ temp[0 * 5 + 4] * kernels_L1_2[11];

		c[4 * 2 + 2] += temp[1 * 5 + 0] * kernels_L1_2[8]
			+ temp[1 * 5 + 1] * kernels_L1_2[9]
			+ temp[1 * 5 + 2] * kernels_L1_2[10]
			+ temp[1 * 5 + 3] * kernels_L1_2[11];

		c[4 * 2 + 3] += temp[1 * 5 + 1] * kernels_L1_2[8]
			+ temp[1 * 5 + 2] * kernels_L1_2[9]
			+ temp[1 * 5 + 3] * kernels_L1_2[10]
			+ temp[1 * 5 + 4] * kernels_L1_2[11];
	}
	{
		c[4 * 3 + 0] += temp[0 * 5 + 0] * kernels_L1_3[8]
			+ temp[0 * 5 + 1] * kernels_L1_3[9]
			+ temp[0 * 5 + 2] * kernels_L1_3[10]
			+ temp[0 * 5 + 3] * kernels_L1_3[11];
		
		c[4 * 3 + 1] += temp[0 * 5 + 1] * kernels_L1_3[8]
			+ temp[0 * 5 + 2] * kernels_L1_3[9]
			+ temp[0 * 5 + 3] * kernels_L1_3[10]
			+ temp[0 * 5 + 4] * kernels_L1_3[11];

		c[4 * 3 + 2] += temp[1 * 5 + 0] * kernels_L1_3[8]
			+ temp[1 * 5 + 1] * kernels_L1_3[9]
			+ temp[1 * 5 + 2] * kernels_L1_3[10]
			+ temp[1 * 5 + 3] * kernels_L1_3[11];

		c[4 * 3 + 3] += temp[1 * 5 + 1] * kernels_L1_3[8]
			+ temp[1 * 5 + 2] * kernels_L1_3[9]
			+ temp[1 * 5 + 3] * kernels_L1_3[10]
			+ temp[1 * 5 + 4] * kernels_L1_3[11];
	}

	src_L1 += src_step;
	temp[0 * 5 + 0] = *(src_L1 + 0);
	temp[0 * 5 + 1] = *(src_L1 + 1);
	temp[0 * 5 + 2] = *(src_L1 + 2);
	temp[0 * 5 + 3] = *(src_L1 + 3);
	temp[0 * 5 + 4] = *(src_L1 + 4);

	{
		c[4 * 0 + 0] += temp[1 * 5 + 0] * kernels_L1_0[12]
			+ temp[1 * 5 + 1] * kernels_L1_0[13]
			+ temp[1 * 5 + 2] * kernels_L1_0[14]
			+ temp[1 * 5 + 3] * kernels_L1_0[15];

		c[4 * 0 + 1] += temp[1 * 5 + 1] * kernels_L1_0[12]
			+ temp[1 * 5 + 2] * kernels_L1_0[13]
			+ temp[1 * 5 + 3] * kernels_L1_0[14]
			+ temp[1 * 5 + 4] * kernels_L1_0[15];

		c[4 * 0 + 2] += temp[0 * 5 + 0] * kernels_L1_0[12]
			+ temp[0 * 5 + 1] * kernels_L1_0[13]
			+ temp[0 * 5 + 2] * kernels_L1_0[14]
			+ temp[0 * 5 + 3] * kernels_L1_0[15];

		c[4 * 0 + 3] += temp[0 * 5 + 1] * kernels_L1_0[12]
			+ temp[0 * 5 + 2] * kernels_L1_0[13]
			+ temp[0 * 5 + 3] * kernels_L1_0[14]
			+ temp[0 * 5 + 4] * kernels_L1_0[15];

		float res = fmax(fmax(c[4 * 0 + 0], c[4 * 0 + 1]), fmax(c[4 * 0 + 2], c[4 * 0 + 3])) + conv_b_L1[0];
		res = tanhf(res);

		res = res * subs_w_L1[0] + subs_b_L1[0];
		res = tanhf(res);

		*(dst_0 + dst_offset) = scale_L1[0] * res;
	}
	{
		c[4 * 1 + 0] += temp[1 * 5 + 0] * kernels_L1_1[12]
			+ temp[1 * 5 + 1] * kernels_L1_1[13]
			+ temp[1 * 5 + 2] * kernels_L1_1[14]
			+ temp[1 * 5 + 3] * kernels_L1_1[15];

		c[4 * 1 + 1] += temp[1 * 5 + 1] * kernels_L1_1[12]
			+ temp[1 * 5 + 2] * kernels_L1_1[13]
			+ temp[1 * 5 + 3] * kernels_L1_1[14]
			+ temp[1 * 5 + 4] * kernels_L1_1[15];

		c[4 * 1 + 2] += temp[0 * 5 + 0] * kernels_L1_1[12]
			+ temp[0 * 5 + 1] * kernels_L1_1[13]
			+ temp[0 * 5 + 2] * kernels_L1_1[14]
			+ temp[0 * 5 + 3] * kernels_L1_1[15];

		c[4 * 1 + 3] += temp[0 * 5 + 1] * kernels_L1_1[12]
			+ temp[0 * 5 + 2] * kernels_L1_1[13]
			+ temp[0 * 5 + 3] * kernels_L1_1[14]
			+ temp[0 * 5 + 4] * kernels_L1_1[15];

		float res = fmax(fmax(c[4 * 1 + 0], c[4 * 1 + 1]), fmax(c[4 * 1 + 2], c[4 * 1 + 3])) + conv_b_L1[1];
		res = tanhf(res);
		
		res = res * subs_w_L1[1] + subs_b_L1[1];
		res = tanhf(res);

		*(dst_1 + dst_offset) = scale_L1[0] * res;
	}
	{
		c[4 * 2 + 0] += temp[1 * 5 + 0] * kernels_L1_2[12]
			+ temp[1 * 5 + 1] * kernels_L1_2[13]
			+ temp[1 * 5 + 2] * kernels_L1_2[14]
			+ temp[1 * 5 + 3] * kernels_L1_2[15];

		c[4 * 2 + 1] += temp[1 * 5 + 1] * kernels_L1_2[12]
			+ temp[1 * 5 + 2] * kernels_L1_2[13]
			+ temp[1 * 5 + 3] * kernels_L1_2[14]
			+ temp[1 * 5 + 4] * kernels_L1_2[15];

		c[4 * 2 + 2] += temp[0 * 5 + 0] * kernels_L1_2[12]
			+ temp[0 * 5 + 1] * kernels_L1_2[13]
			+ temp[0 * 5 + 2] * kernels_L1_2[14]
			+ temp[0 * 5 + 3] * kernels_L1_2[15];

		c[4 * 2 + 3] += temp[0 * 5 + 1] * kernels_L1_2[12]
			+ temp[0 * 5 + 2] * kernels_L1_2[13]
			+ temp[0 * 5 + 3] * kernels_L1_2[14]
			+ temp[0 * 5 + 4] * kernels_L1_2[15];

		float res = fmax(fmax(c[4 * 2 + 0], c[4 * 2 + 1]), fmax(c[4 * 2 + 2], c[4 * 2 + 3])) + conv_b_L1[2];
		res = tanhf(res);

		res = res * subs_w_L1[2] + subs_b_L1[2];
		res = tanhf(res);

		*(dst_2 + dst_offset) = scale_L1[0] * res;
	}
	{
		c[4 * 3 + 0] += temp[1 * 5 + 0] * kernels_L1_3[12]
			+ temp[1 * 5 + 1] * kernels_L1_3[13]
			+ temp[1 * 5 + 2] * kernels_L1_3[14]
			+ temp[1 * 5 + 3] * kernels_L1_3[15];

		c[4 * 3 + 1] += temp[1 * 5 + 1] * kernels_L1_3[12]
			+ temp[1 * 5 + 2] * kernels_L1_3[13]
			+ temp[1 * 5 + 3] * kernels_L1_3[14]
			+ temp[1 * 5 + 4] * kernels_L1_3[15];

		c[4 * 3 + 2] += temp[0 * 5 + 0] * kernels_L1_3[12]
			+ temp[0 * 5 + 1] * kernels_L1_3[13]
			+ temp[0 * 5 + 2] * kernels_L1_3[14]
			+ temp[0 * 5 + 3] * kernels_L1_3[15];

		c[4 * 3 + 3] += temp[0 * 5 + 1] * kernels_L1_3[12]
			+ temp[0 * 5 + 2] * kernels_L1_3[13]
			+ temp[0 * 5 + 3] * kernels_L1_3[14]
			+ temp[0 * 5 + 4] * kernels_L1_3[15];

		float res = fmax(fmax(c[4 * 3 + 0], c[4 * 3 + 1]), fmax(c[4 * 3 + 2], c[4 * 3 + 3])) + conv_b_L1[3];
		res = tanhf(res);

		res = res * subs_w_L1[3] + subs_b_L1[3];
		res = tanhf(res);

		*(dst_3 + dst_offset) = scale_L1[0] * res;
	}
}

__kernel void add_2_3_conv_4x3x3_max_tanh_tanh_L_cl(
													/*0*/__global float* dst_0,
													/*1*/__global float* dst_1,
													/*2*/__global float* dst_2,
													/*3*/__global float* dst_3,
													/*4*/int dst_step,
													/*5*/__global float* src_L2_1,
													/*6*/__global float* src_L2_2,
													/*7*/__global float* src_L2_3,
													/*8*/int src_cols,
													/*9*/int src_rows,
													/*10*/int src_step,
													/*11*/__constant float* kernels_L2_0,
													/*12*/__constant float* kernels_L2_1,
													/*13*/__constant float* kernels_L2_2,
													/*14*/__constant float* kernels_L2_3,
													/*15*/__constant float* conv_b_L2,
													/*16*/__constant float* subs_w_L2,
													/*17*/__constant float* subs_b_L2,
													/*18*/__constant float* scale_L2
													)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	const int dst_offset = j * dst_step + i;

	i *= 2;
	j *= 2;
	if (i + 1 >= src_cols - 2 || j + 1 >= src_rows - 2) return;
	const int src_offset = j * src_step + i;
	
	float temp1[8];

	src_L2_1 += src_offset;
	temp1[0 * 4 + 0] = *(src_L2_1 + 0);
	temp1[0 * 4 + 1] = *(src_L2_1 + 1);
	temp1[0 * 4 + 2] = *(src_L2_1 + 2);
	temp1[0 * 4 + 3] = *(src_L2_1 + 3);

	src_L2_1 += src_step;
	temp1[1 * 4 + 0] = *(src_L2_1 + 0);
	temp1[1 * 4 + 1] = *(src_L2_1 + 1);
	temp1[1 * 4 + 2] = *(src_L2_1 + 2);
	temp1[1 * 4 + 3] = *(src_L2_1 + 3);

	src_L2_2 += src_offset;
	temp1[0 * 4 + 0] += *(src_L2_2 + 0);
	temp1[0 * 4 + 1] += *(src_L2_2 + 1);
	temp1[0 * 4 + 2] += *(src_L2_2 + 2);
	temp1[0 * 4 + 3] += *(src_L2_2 + 3);

	src_L2_2 += src_step;
	temp1[1 * 4 + 0] += *(src_L2_2 + 0);
	temp1[1 * 4 + 1] += *(src_L2_2 + 1);
	temp1[1 * 4 + 2] += *(src_L2_2 + 2);
	temp1[1 * 4 + 3] += *(src_L2_2 + 3);

	float temp2[8];

	src_L2_3 += src_offset;
	temp2[0 * 4 + 0] = temp1[0 * 4 + 0] + *(src_L2_3 + 0);
	temp2[0 * 4 + 1] = temp1[0 * 4 + 1] + *(src_L2_3 + 1);
	temp2[0 * 4 + 2] = temp1[0 * 4 + 2] + *(src_L2_3 + 2);
	temp2[0 * 4 + 3] = temp1[0 * 4 + 3] + *(src_L2_3 + 3);
	
	src_L2_3 += src_step;
	temp2[1 * 4 + 0] = temp1[1 * 4 + 0] + *(src_L2_3 + 0);
	temp2[1 * 4 + 1] = temp1[1 * 4 + 1] + *(src_L2_3 + 1);
	temp2[1 * 4 + 2] = temp1[1 * 4 + 2] + *(src_L2_3 + 2);
	temp2[1 * 4 + 3] = temp1[1 * 4 + 3] + *(src_L2_3 + 3);

	float c[4 * 4];
	{
		c[4 * 0 + 0] = temp1[0 * 4 + 0] * kernels_L2_0[0]
			+ temp1[0 * 4 + 1] * kernels_L2_0[1]
			+ temp1[0 * 4 + 2] * kernels_L2_0[2];

		c[4 * 0 + 1] = temp1[0 * 4 + 1] * kernels_L2_0[0]
			+ temp1[0 * 4 + 2] * kernels_L2_0[1]
			+ temp1[0 * 4 + 3] * kernels_L2_0[2];

		c[4 * 0 + 2] = temp1[1 * 4 + 0] * kernels_L2_0[0]
			+ temp1[1 * 4 + 1] * kernels_L2_0[1]
			+ temp1[1 * 4 + 2] * kernels_L2_0[2];

		c[4 * 0 + 3] = temp1[1 * 4 + 1] * kernels_L2_0[0]
			+ temp1[1 * 4 + 2] * kernels_L2_0[1]
			+ temp1[1 * 4 + 3] * kernels_L2_0[2];
	}
	{
		c[4 * 1 + 0] = temp1[0 * 4 + 0] * kernels_L2_1[0]
			+ temp1[0 * 4 + 1] * kernels_L2_1[1]
			+ temp1[0 * 4 + 2] * kernels_L2_1[2];

		c[4 * 1 + 1] = temp1[0 * 4 + 1] * kernels_L2_1[0]
			+ temp1[0 * 4 + 2] * kernels_L2_1[1]
			+ temp1[0 * 4 + 3] * kernels_L2_1[2];

		c[4 * 1 + 2] = temp1[1 * 4 + 0] * kernels_L2_1[0]
			+ temp1[1 * 4 + 1] * kernels_L2_1[1]
			+ temp1[1 * 4 + 2] * kernels_L2_1[2];

		c[4 * 1 + 3] = temp1[1 * 4 + 1] * kernels_L2_1[0]
			+ temp1[1 * 4 + 2] * kernels_L2_1[1]
			+ temp1[1 * 4 + 3] * kernels_L2_1[2];
	}
	{
		c[4 * 2 + 0] = temp2[0 * 4 + 0] * kernels_L2_2[0]
			+ temp2[0 * 4 + 1] * kernels_L2_2[1]
			+ temp2[0 * 4 + 2] * kernels_L2_2[2];

		c[4 * 2 + 1] = temp2[0 * 4 + 1] * kernels_L2_2[0]
			+ temp2[0 * 4 + 2] * kernels_L2_2[1]
			+ temp2[0 * 4 + 3] * kernels_L2_2[2];

		c[4 * 2 + 2] = temp2[1 * 4 + 0] * kernels_L2_2[0]
			+ temp2[1 * 4 + 1] * kernels_L2_2[1]
			+ temp2[1 * 4 + 2] * kernels_L2_2[2];

		c[4 * 2 + 3] = temp2[1 * 4 + 1] * kernels_L2_2[0]
			+ temp2[1 * 4 + 2] * kernels_L2_2[1]
			+ temp2[1 * 4 + 3] * kernels_L2_2[2];
	}
	{
		c[4 * 3 + 0] = temp2[0 * 4 + 0] * kernels_L2_3[0]
			+ temp2[0 * 4 + 1] * kernels_L2_3[1]
			+ temp2[0 * 4 + 2] * kernels_L2_3[2];

		c[4 * 3 + 1] = temp2[0 * 4 + 1] * kernels_L2_3[0]
			+ temp2[0 * 4 + 2] * kernels_L2_3[1]
			+ temp2[0 * 4 + 3] * kernels_L2_3[2];

		c[4 * 3 + 2] = temp2[1 * 4 + 0] * kernels_L2_3[0]
			+ temp2[1 * 4 + 1] * kernels_L2_3[1]
			+ temp2[1 * 4 + 2] * kernels_L2_3[2];

		c[4 * 3 + 3] = temp2[1 * 4 + 1] * kernels_L2_3[0]
			+ temp2[1 * 4 + 2] * kernels_L2_3[1]
			+ temp2[1 * 4 + 3] * kernels_L2_3[2];

	}
	
	src_L2_1 += src_step;
	temp1[0 * 4 + 0] = *(src_L2_1 + 0);
	temp1[0 * 4 + 1] = *(src_L2_1 + 1);
	temp1[0 * 4 + 2] = *(src_L2_1 + 2);
	temp1[0 * 4 + 3] = *(src_L2_1 + 3);

	src_L2_2 += src_step;
	temp1[0 * 4 + 0] += *(src_L2_2 + 0);
	temp1[0 * 4 + 1] += *(src_L2_2 + 1);
	temp1[0 * 4 + 2] += *(src_L2_2 + 2);
	temp1[0 * 4 + 3] += *(src_L2_2 + 3);

	src_L2_3 += src_step;
	temp2[0 * 4 + 0] = temp1[0 * 4 + 0] + *(src_L2_3 + 0);
	temp2[0 * 4 + 1] = temp1[0 * 4 + 1] + *(src_L2_3 + 1);
	temp2[0 * 4 + 2] = temp1[0 * 4 + 2] + *(src_L2_3 + 2);
	temp2[0 * 4 + 3] = temp1[0 * 4 + 3] + *(src_L2_3 + 3);

	{
		c[4 * 0 + 0] += temp1[1 * 4 + 0] * kernels_L2_0[3]
			+ temp1[1 * 4 + 1] * kernels_L2_0[4]
			+ temp1[1 * 4 + 2] * kernels_L2_0[5];

		c[4 * 0 + 1] += temp1[1 * 4 + 1] * kernels_L2_0[3]
			+ temp1[1 * 4 + 2] * kernels_L2_0[4]
			+ temp1[1 * 4 + 3] * kernels_L2_0[5];

		c[4 * 0 + 2] += temp1[0 * 4 + 0] * kernels_L2_0[3]
			+ temp1[0 * 4 + 1] * kernels_L2_0[4]
			+ temp1[0 * 4 + 2] * kernels_L2_0[5];

		c[4 * 0 + 3] += temp1[0 * 4 + 1] * kernels_L2_0[3]
			+ temp1[0 * 4 + 2] * kernels_L2_0[4]
			+ temp1[0 * 4 + 3] * kernels_L2_0[5];
	}
	{
		c[4 * 1 + 0] += temp1[1 * 4 + 0] * kernels_L2_1[3]
			+ temp1[1 * 4 + 1] * kernels_L2_1[4]
			+ temp1[1 * 4 + 2] * kernels_L2_1[5];

		c[4 * 1 + 1] += temp1[1 * 4 + 1] * kernels_L2_1[3]
			+ temp1[1 * 4 + 2] * kernels_L2_1[4]
			+ temp1[1 * 4 + 3] * kernels_L2_1[5];

		c[4 * 1 + 2] += temp1[0 * 4 + 0] * kernels_L2_1[3]
			+ temp1[0 * 4 + 1] * kernels_L2_1[4]
			+ temp1[0 * 4 + 2] * kernels_L2_1[5];

		c[4 * 1 + 3] += temp1[0 * 4 + 1] * kernels_L2_1[3]
			+ temp1[0 * 4 + 2] * kernels_L2_1[4]
			+ temp1[0 * 4 + 3] * kernels_L2_1[5];
	}
	{
		c[4 * 2 + 0] += temp2[1 * 4 + 0] * kernels_L2_2[3]
			+ temp2[1 * 4 + 1] * kernels_L2_2[4]
			+ temp2[1 * 4 + 2] * kernels_L2_2[5];

		c[4 * 2 + 1] += temp2[1 * 4 + 1] * kernels_L2_2[3]
			+ temp2[1 * 4 + 2] * kernels_L2_2[4]
			+ temp2[1 * 4 + 3] * kernels_L2_2[5];

		c[4 * 2 + 2] += temp2[0 * 4 + 0] * kernels_L2_2[3]
			+ temp2[0 * 4 + 1] * kernels_L2_2[4]
			+ temp2[0 * 4 + 2] * kernels_L2_2[5];

		c[4 * 2 + 3] += temp2[0 * 4 + 1] * kernels_L2_2[3]
			+ temp2[0 * 4 + 2] * kernels_L2_2[4]
			+ temp2[0 * 4 + 3] * kernels_L2_2[5];
	}
	{
		c[4 * 3 + 0] += temp2[1 * 4 + 0] * kernels_L2_3[3]
			+ temp2[1 * 4 + 1] * kernels_L2_3[4]
			+ temp2[1 * 4 + 2] * kernels_L2_3[5];

		c[4 * 3 + 1] += temp2[1 * 4 + 1] * kernels_L2_3[3]
			+ temp2[1 * 4 + 2] * kernels_L2_3[4]
			+ temp2[1 * 4 + 3] * kernels_L2_3[5];

		c[4 * 3 + 2] += temp2[0 * 4 + 0] * kernels_L2_3[3]
			+ temp2[0 * 4 + 1] * kernels_L2_3[4]
			+ temp2[0 * 4 + 2] * kernels_L2_3[5];

		c[4 * 3 + 3] += temp2[0 * 4 + 1] * kernels_L2_3[3]
			+ temp2[0 * 4 + 2] * kernels_L2_3[4]
			+ temp2[0 * 4 + 3] * kernels_L2_3[5];
	}

	src_L2_1 += src_step;
	temp1[1 * 4 + 0] = *(src_L2_1 + 0);
	temp1[1 * 4 + 1] = *(src_L2_1 + 1);
	temp1[1 * 4 + 2] = *(src_L2_1 + 2);
	temp1[1 * 4 + 3] = *(src_L2_1 + 3);

	src_L2_2 += src_step;
	temp1[1 * 4 + 0] += *(src_L2_2 + 0);
	temp1[1 * 4 + 1] += *(src_L2_2 + 1);
	temp1[1 * 4 + 2] += *(src_L2_2 + 2);
	temp1[1 * 4 + 3] += *(src_L2_2 + 3);

	src_L2_3 += src_step;
	temp2[1 * 4 + 0] = temp1[1 * 4 + 0] + *(src_L2_3 + 0);
	temp2[1 * 4 + 1] = temp1[1 * 4 + 1] + *(src_L2_3 + 1);
	temp2[1 * 4 + 2] = temp1[1 * 4 + 2] + *(src_L2_3 + 2);
	temp2[1 * 4 + 3] = temp1[1 * 4 + 3] + *(src_L2_3 + 3);

	{
		c[4 * 0 + 0] += temp1[0 * 4 + 0] * kernels_L2_0[6]
			+ temp1[0 * 4 + 1] * kernels_L2_0[7]
			+ temp1[0 * 4 + 2] * kernels_L2_0[8];

		c[4 * 0 + 1] += temp1[0 * 4 + 1] * kernels_L2_0[6]
			+ temp1[0 * 4 + 2] * kernels_L2_0[7]
			+ temp1[0 * 4 + 3] * kernels_L2_0[8];

		c[4 * 0 + 2] += temp1[1 * 4 + 0] * kernels_L2_0[6]
			+ temp1[1 * 4 + 1] * kernels_L2_0[7]
			+ temp1[1 * 4 + 2] * kernels_L2_0[8];

		c[4 * 0 + 3] += temp1[1 * 4 + 1] * kernels_L2_0[6]
			+ temp1[1 * 4 + 2] * kernels_L2_0[7]
			+ temp1[1 * 4 + 3] * kernels_L2_0[8];

		float res = fmax(fmax(c[4 * 0 + 0], c[4 * 0 + 1]), fmax(c[4 * 0 + 2], c[4 * 0 + 3])) + conv_b_L2[0];
		res = tanhf(res);

		res = res * subs_w_L2[0] + subs_b_L2[0];
		res = tanhf(res);

		*(dst_0 + dst_offset) = scale_L2[0] * res;
	}
	{
		c[4 * 1 + 0] += temp1[0 * 4 + 0] * kernels_L2_1[6]
			+ temp1[0 * 4 + 1] * kernels_L2_1[7]
			+ temp1[0 * 4 + 2] * kernels_L2_1[8];

		c[4 * 1 + 1] += temp1[0 * 4 + 1] * kernels_L2_1[6]
			+ temp1[0 * 4 + 2] * kernels_L2_1[7]
			+ temp1[0 * 4 + 3] * kernels_L2_1[8];

		c[4 * 1 + 2] += temp1[1 * 4 + 0] * kernels_L2_1[6]
			+ temp1[1 * 4 + 1] * kernels_L2_1[7]
			+ temp1[1 * 4 + 2] * kernels_L2_1[8];

		c[4 * 1 + 3] += temp1[1 * 4 + 1] * kernels_L2_1[6]
			+ temp1[1 * 4 + 2] * kernels_L2_1[7]
			+ temp1[1 * 4 + 3] * kernels_L2_1[8];

		float res = fmax(fmax(c[4 * 1 + 0], c[4 * 1 + 1]), fmax(c[4 * 1 + 2], c[4 * 1 + 3])) + conv_b_L2[1];
		res = tanhf(res);

		res = res * subs_w_L2[1] + subs_b_L2[1];
		res = tanhf(res);

		*(dst_1 + dst_offset) = scale_L2[0] * res;
	}
	{
		c[4 * 2 + 0] += temp2[0 * 4 + 0] * kernels_L2_2[6]
			+ temp2[0 * 4 + 1] * kernels_L2_2[7]
			+ temp2[0 * 4 + 2] * kernels_L2_2[8];

		c[4 * 2 + 1] += temp2[0 * 4 + 1] * kernels_L2_2[6]
			+ temp2[0 * 4 + 2] * kernels_L2_2[7]
			+ temp2[0 * 4 + 3] * kernels_L2_2[8];

		c[4 * 2 + 2] += temp2[1 * 4 + 0] * kernels_L2_2[6]
			+ temp2[1 * 4 + 1] * kernels_L2_2[7]
			+ temp2[1 * 4 + 2] * kernels_L2_2[8];

		c[4 * 2 + 3] += temp2[1 * 4 + 1] * kernels_L2_2[6]
			+ temp2[1 * 4 + 2] * kernels_L2_2[7]
			+ temp2[1 * 4 + 3] * kernels_L2_2[8];

		float res = fmax(fmax(c[4 * 2 + 0], c[4 * 2 + 1]), fmax(c[4 * 2 + 2], c[4 * 2 + 3])) + conv_b_L2[2];
		res = tanhf(res);

		res = res * subs_w_L2[2] + subs_b_L2[2];
		res = tanhf(res);

		*(dst_2 + dst_offset) = scale_L2[0] * res;
	}
	{
		c[4 * 3 + 0] += temp2[0 * 4 + 0] * kernels_L2_3[6]
			+ temp2[0 * 4 + 1] * kernels_L2_3[7]
			+ temp2[0 * 4 + 2] * kernels_L2_3[8];

		c[4 * 3 + 1] += temp2[0 * 4 + 1] * kernels_L2_3[6]
			+ temp2[0 * 4 + 2] * kernels_L2_3[7]
			+ temp2[0 * 4 + 3] * kernels_L2_3[8];

		c[4 * 3 + 2] += temp2[1 * 4 + 0] * kernels_L2_3[6]
			+ temp2[1 * 4 + 1] * kernels_L2_3[7]
			+ temp2[1 * 4 + 2] * kernels_L2_3[8];

		c[4 * 3 + 3] += temp2[1 * 4 + 1] * kernels_L2_3[6]
			+ temp2[1 * 4 + 2] * kernels_L2_3[7]
			+ temp2[1 * 4 + 3] * kernels_L2_3[8];

		float res = fmax(fmax(c[4 * 3 + 0], c[4 * 3 + 1]), fmax(c[4 * 3 + 2], c[4 * 3 + 3])) + conv_b_L2[3];
		res = tanhf(res);

		res = res * subs_w_L2[3] + subs_b_L2[3];
		res = tanhf(res);

		*(dst_3 + dst_offset) = scale_L2[0] * res;
	}
}
__kernel void add_2_3_conv_4x3x3_max_tanh_tanh_R_cl(
													/*0*/__global float* dst_4,
													/*1*/__global float* dst_5,
													/*2*/__global float* dst_6,
													/*3*/__global float* dst_7,
													/*4*/int dst_step,
													/*5*/__global float* src_L2_2,
													/*6*/__global float* src_L2_3,
													/*7*/__global float* src_L2_4,
													/*8*/int src_cols,
													/*9*/int src_rows,
													/*10*/int src_step,
													/*11*/__constant float* kernels_L2_4,
													/*12*/__constant float* kernels_L2_5,
													/*13*/__constant float* kernels_L2_6,
													/*14*/__constant float* kernels_L2_7,
													/*15*/__constant float* conv_b_L2,
													/*16*/__constant float* subs_w_L2,
													/*17*/__constant float* subs_b_L2,
													/*18*/__constant float* scale_L2
													)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	const int dst_offset = j * dst_step + i;

	i *= 2;
	j *= 2;
	if (i + 1 >= src_cols - 2 || j + 1 >= src_rows - 2) return;
	const int src_offset = j * src_step + i;

	float temp1[8];

	src_L2_4 += src_offset;
	temp1[0 * 4 + 0] = *(src_L2_4 + 0);
	temp1[0 * 4 + 1] = *(src_L2_4 + 1);
	temp1[0 * 4 + 2] = *(src_L2_4 + 2);
	temp1[0 * 4 + 3] = *(src_L2_4 + 3);
	
	src_L2_4 += src_step;
	temp1[1 * 4 + 0] = *(src_L2_4 + 0);
	temp1[1 * 4 + 1] = *(src_L2_4 + 1);
	temp1[1 * 4 + 2] = *(src_L2_4 + 2);
	temp1[1 * 4 + 3] = *(src_L2_4 + 3);
	
	src_L2_3 += src_offset;
	temp1[0 * 4 + 0] += *(src_L2_3 + 0);
	temp1[0 * 4 + 1] += *(src_L2_3 + 1);
	temp1[0 * 4 + 2] += *(src_L2_3 + 2);
	temp1[0 * 4 + 3] += *(src_L2_3 + 3);
	
	src_L2_3 += src_step;
	temp1[1 * 4 + 0] += *(src_L2_3 + 0);
	temp1[1 * 4 + 1] += *(src_L2_3 + 1);
	temp1[1 * 4 + 2] += *(src_L2_3 + 2);
	temp1[1 * 4 + 3] += *(src_L2_3 + 3);

	float temp2[8];
	
	src_L2_2 += src_offset;
	temp2[0 * 4 + 0] = temp1[0 * 4 + 0] + *(src_L2_2 + 0);
	temp2[0 * 4 + 1] = temp1[0 * 4 + 1] + *(src_L2_2 + 1);
	temp2[0 * 4 + 2] = temp1[0 * 4 + 2] + *(src_L2_2 + 2);
	temp2[0 * 4 + 3] = temp1[0 * 4 + 3] + *(src_L2_2 + 3);
	
	src_L2_2 += src_step;
	temp2[1 * 4 + 0] = temp1[1 * 4 + 0] + *(src_L2_2 + 0);
	temp2[1 * 4 + 1] = temp1[1 * 4 + 1] + *(src_L2_2 + 1);
	temp2[1 * 4 + 2] = temp1[1 * 4 + 2] + *(src_L2_2 + 2);
	temp2[1 * 4 + 3] = temp1[1 * 4 + 3] + *(src_L2_2 + 3);

	float c[4 * 4];
	{
		c[4 * 0 + 0] = temp1[0 * 4 + 0] * kernels_L2_7[0]
			+ temp1[0 * 4 + 1] * kernels_L2_7[1]
			+ temp1[0 * 4 + 2] * kernels_L2_7[2];

		c[4 * 0 + 1] = temp1[0 * 4 + 1] * kernels_L2_7[0]
			+ temp1[0 * 4 + 2] * kernels_L2_7[1]
			+ temp1[0 * 4 + 3] * kernels_L2_7[2];

		c[4 * 0 + 2] = temp1[1 * 4 + 0] * kernels_L2_7[0]
			+ temp1[1 * 4 + 1] * kernels_L2_7[1]
			+ temp1[1 * 4 + 2] * kernels_L2_7[2];

		c[4 * 0 + 3] = temp1[1 * 4 + 1] * kernels_L2_7[0]
			+ temp1[1 * 4 + 2] * kernels_L2_7[1]
			+ temp1[1 * 4 + 3] * kernels_L2_7[2];
	}
	{
		c[4 * 1 + 0] = temp1[0 * 4 + 0] * kernels_L2_6[0]
			+ temp1[0 * 4 + 1] * kernels_L2_6[1]
			+ temp1[0 * 4 + 2] * kernels_L2_6[2];

		c[4 * 1 + 1] = temp1[0 * 4 + 1] * kernels_L2_6[0]
			+ temp1[0 * 4 + 2] * kernels_L2_6[1]
			+ temp1[0 * 4 + 3] * kernels_L2_6[2];

		c[4 * 1 + 2] = temp1[1 * 4 + 0] * kernels_L2_6[0]
			+ temp1[1 * 4 + 1] * kernels_L2_6[1]
			+ temp1[1 * 4 + 2] * kernels_L2_6[2];

		c[4 * 1 + 3] = temp1[1 * 4 + 1] * kernels_L2_6[0]
			+ temp1[1 * 4 + 2] * kernels_L2_6[1]
			+ temp1[1 * 4 + 3] * kernels_L2_6[2];
	}
	{
		c[4 * 2 + 0] = temp2[0 * 4 + 0] * kernels_L2_5[0]
			+ temp2[0 * 4 + 1] * kernels_L2_5[1]
			+ temp2[0 * 4 + 2] * kernels_L2_5[2];

		c[4 * 2 + 1] = temp2[0 * 4 + 1] * kernels_L2_5[0]
			+ temp2[0 * 4 + 2] * kernels_L2_5[1]
			+ temp2[0 * 4 + 3] * kernels_L2_5[2];

		c[4 * 2 + 2] = temp2[1 * 4 + 0] * kernels_L2_5[0]
			+ temp2[1 * 4 + 1] * kernels_L2_5[1]
			+ temp2[1 * 4 + 2] * kernels_L2_5[2];

		c[4 * 2 + 3] = temp2[1 * 4 + 1] * kernels_L2_5[0]
			+ temp2[1 * 4 + 2] * kernels_L2_5[1]
			+ temp2[1 * 4 + 3] * kernels_L2_5[2];
	}
	{
		c[4 * 3 + 0] = temp2[0 * 4 + 0] * kernels_L2_4[0]
			+ temp2[0 * 4 + 1] * kernels_L2_4[1]
			+ temp2[0 * 4 + 2] * kernels_L2_4[2];

		c[4 * 3 + 1] = temp2[0 * 4 + 1] * kernels_L2_4[0]
			+ temp2[0 * 4 + 2] * kernels_L2_4[1]
			+ temp2[0 * 4 + 3] * kernels_L2_4[2];

		c[4 * 3 + 2] = temp2[1 * 4 + 0] * kernels_L2_4[0]
			+ temp2[1 * 4 + 1] * kernels_L2_4[1]
			+ temp2[1 * 4 + 2] * kernels_L2_4[2];

		c[4 * 3 + 3] = temp2[1 * 4 + 1] * kernels_L2_4[0]
			+ temp2[1 * 4 + 2] * kernels_L2_4[1]
			+ temp2[1 * 4 + 3] * kernels_L2_4[2];
	}

	src_L2_4 += src_step;
	temp1[0 * 4 + 0] = *(src_L2_4 + 0);
	temp1[0 * 4 + 1] = *(src_L2_4 + 1);
	temp1[0 * 4 + 2] = *(src_L2_4 + 2);
	temp1[0 * 4 + 3] = *(src_L2_4 + 3);

	src_L2_3 += src_step;
	temp1[0 * 4 + 0] += *(src_L2_3 + 0);
	temp1[0 * 4 + 1] += *(src_L2_3 + 1);
	temp1[0 * 4 + 2] += *(src_L2_3 + 2);
	temp1[0 * 4 + 3] += *(src_L2_3 + 3);

	src_L2_2 += src_step;
	temp2[0 * 4 + 0] = temp1[0 * 4 + 0] + *(src_L2_2 + 0);
	temp2[0 * 4 + 1] = temp1[0 * 4 + 1] + *(src_L2_2 + 1);
	temp2[0 * 4 + 2] = temp1[0 * 4 + 2] + *(src_L2_2 + 2);
	temp2[0 * 4 + 3] = temp1[0 * 4 + 3] + *(src_L2_2 + 3);

	{
		c[4 * 0 + 0] += temp1[1 * 4 + 0] * kernels_L2_7[3]
			+ temp1[1 * 4 + 1] * kernels_L2_7[4]
			+ temp1[1 * 4 + 2] * kernels_L2_7[5];
	
		c[4 * 0 + 1] += temp1[1 * 4 + 1] * kernels_L2_7[3]
			+ temp1[1 * 4 + 2] * kernels_L2_7[4]
			+ temp1[1 * 4 + 3] * kernels_L2_7[5];

		c[4 * 0 + 2] += temp1[0 * 4 + 0] * kernels_L2_7[3]
			+ temp1[0 * 4 + 1] * kernels_L2_7[4]
			+ temp1[0 * 4 + 2] * kernels_L2_7[5];

		c[4 * 0 + 3] += temp1[0 * 4 + 1] * kernels_L2_7[3]
			+ temp1[0 * 4 + 2] * kernels_L2_7[4]
			+ temp1[0 * 4 + 3] * kernels_L2_7[5];
	}
	{
		c[4 * 1 + 0] += temp1[1 * 4 + 0] * kernels_L2_6[3]
			+ temp1[1 * 4 + 1] * kernels_L2_6[4]
			+ temp1[1 * 4 + 2] * kernels_L2_6[5];

		c[4 * 1 + 1] += temp1[1 * 4 + 1] * kernels_L2_6[3]
			+ temp1[1 * 4 + 2] * kernels_L2_6[4]
			+ temp1[1 * 4 + 3] * kernels_L2_6[5];

		c[4 * 1 + 2] += temp1[0 * 4 + 0] * kernels_L2_6[3]
			+ temp1[0 * 4 + 1] * kernels_L2_6[4]
			+ temp1[0 * 4 + 2] * kernels_L2_6[5];

		c[4 * 1 + 3] += temp1[0 * 4 + 1] * kernels_L2_6[3]
			+ temp1[0 * 4 + 2] * kernels_L2_6[4]
			+ temp1[0 * 4 + 3] * kernels_L2_6[5];
	}
	{
		c[4 * 2 + 0] += temp2[1 * 4 + 0] * kernels_L2_5[3]
			+ temp2[1 * 4 + 1] * kernels_L2_5[4]
			+ temp2[1 * 4 + 2] * kernels_L2_5[5];

		c[4 * 2 + 1] += temp2[1 * 4 + 1] * kernels_L2_5[3]
			+ temp2[1 * 4 + 2] * kernels_L2_5[4]
			+ temp2[1 * 4 + 3] * kernels_L2_5[5];

		c[4 * 2 + 2] += temp2[0 * 4 + 0] * kernels_L2_5[3]
			+ temp2[0 * 4 + 1] * kernels_L2_5[4]
			+ temp2[0 * 4 + 2] * kernels_L2_5[5];

		c[4 * 2 + 3] += temp2[0 * 4 + 1] * kernels_L2_5[3]
			+ temp2[0 * 4 + 2] * kernels_L2_5[4]
			+ temp2[0 * 4 + 3] * kernels_L2_5[5];
	}
	{
		c[4 * 3 + 0] += temp2[1 * 4 + 0] * kernels_L2_4[3]
			+ temp2[1 * 4 + 1] * kernels_L2_4[4]
			+ temp2[1 * 4 + 2] * kernels_L2_4[5];

		c[4 * 3 + 1] += temp2[1 * 4 + 1] * kernels_L2_4[3]
			+ temp2[1 * 4 + 2] * kernels_L2_4[4]
			+ temp2[1 * 4 + 3] * kernels_L2_4[5];

		c[4 * 3 + 2] += temp2[0 * 4 + 0] * kernels_L2_4[3]
			+ temp2[0 * 4 + 1] * kernels_L2_4[4]
			+ temp2[0 * 4 + 2] * kernels_L2_4[5];

		c[4 * 3 + 3] += temp2[0 * 4 + 1] * kernels_L2_4[3]
			+ temp2[0 * 4 + 2] * kernels_L2_4[4]
			+ temp2[0 * 4 + 3] * kernels_L2_4[5];
	}

	src_L2_4 += src_step;
	temp1[1 * 4 + 0] = *(src_L2_4 + 0);
	temp1[1 * 4 + 1] = *(src_L2_4 + 1);
	temp1[1 * 4 + 2] = *(src_L2_4 + 2);
	temp1[1 * 4 + 3] = *(src_L2_4 + 3);
	
	src_L2_3 += src_step;
	temp1[1 * 4 + 0] += *(src_L2_3 + 0);
	temp1[1 * 4 + 1] += *(src_L2_3 + 1);
	temp1[1 * 4 + 2] += *(src_L2_3 + 2);
	temp1[1 * 4 + 3] += *(src_L2_3 + 3);

	src_L2_2 += src_step;
	temp2[1 * 4 + 0] = temp1[1 * 4 + 0] + *(src_L2_2 + 0);
	temp2[1 * 4 + 1] = temp1[1 * 4 + 1] + *(src_L2_2 + 1);
	temp2[1 * 4 + 2] = temp1[1 * 4 + 2] + *(src_L2_2 + 2);
	temp2[1 * 4 + 3] = temp1[1 * 4 + 3] + *(src_L2_2 + 3);

	{
		c[4 * 0 + 0] += temp1[0 * 4 + 0] * kernels_L2_7[6]
			+ temp1[0 * 4 + 1] * kernels_L2_7[7]
			+ temp1[0 * 4 + 2] * kernels_L2_7[8];

		c[4 * 0 + 1] += temp1[0 * 4 + 1] * kernels_L2_7[6]
			+ temp1[0 * 4 + 2] * kernels_L2_7[7]
			+ temp1[0 * 4 + 3] * kernels_L2_7[8];

		c[4 * 0 + 2] += temp1[1 * 4 + 0] * kernels_L2_7[6]
			+ temp1[1 * 4 + 1] * kernels_L2_7[7]
			+ temp1[1 * 4 + 2] * kernels_L2_7[8];

		c[4 * 0 + 3] += temp1[1 * 4 + 1] * kernels_L2_7[6]
			+ temp1[1 * 4 + 2] * kernels_L2_7[7]
			+ temp1[1 * 4 + 3] * kernels_L2_7[8];

		float res = fmax(fmax(c[4 * 0 + 0], c[4 * 0 + 1]), fmax(c[4 * 0 + 2], c[4 * 0 + 3])) + conv_b_L2[7];
		res = tanhf(res);

		res = res * subs_w_L2[7] + subs_b_L2[7];
		res = tanhf(res);

		*(dst_7 + dst_offset) = scale_L2[0] * res;
	}
	{
		c[4 * 1 + 0] += temp1[0 * 4 + 0] * kernels_L2_6[6]
			+ temp1[0 * 4 + 1] * kernels_L2_6[7]
			+ temp1[0 * 4 + 2] * kernels_L2_6[8];

		c[4 * 1 + 1] += temp1[0 * 4 + 1] * kernels_L2_6[6]
			+ temp1[0 * 4 + 2] * kernels_L2_6[7]
			+ temp1[0 * 4 + 3] * kernels_L2_6[8];

		c[4 * 1 + 2] += temp1[1 * 4 + 0] * kernels_L2_6[6]
			+ temp1[1 * 4 + 1] * kernels_L2_6[7]
			+ temp1[1 * 4 + 2] * kernels_L2_6[8];

		c[4 * 1 + 3] += temp1[1 * 4 + 1] * kernels_L2_6[6]
			+ temp1[1 * 4 + 2] * kernels_L2_6[7]
			+ temp1[1 * 4 + 3] * kernels_L2_6[8];

		float res = fmax(fmax(c[4 * 1 + 0], c[4 * 1 + 1]), fmax(c[4 * 1 + 2], c[4 * 1 + 3])) + conv_b_L2[6];
		res = tanhf(res);

		res = res * subs_w_L2[6] + subs_b_L2[6];
		res = tanhf(res);

		*(dst_6 + dst_offset) = scale_L2[0] * res;
	}
	{
		c[4 * 2 + 0] += temp2[0 * 4 + 0] * kernels_L2_5[6]
			+ temp2[0 * 4 + 1] * kernels_L2_5[7]
			+ temp2[0 * 4 + 2] * kernels_L2_5[8];
		
		c[4 * 2 + 1] += temp2[0 * 4 + 1] * kernels_L2_5[6]
			+ temp2[0 * 4 + 2] * kernels_L2_5[7]
			+ temp2[0 * 4 + 3] * kernels_L2_5[8];

		c[4 * 2 + 2] += temp2[1 * 4 + 0] * kernels_L2_5[6]
			+ temp2[1 * 4 + 1] * kernels_L2_5[7]
			+ temp2[1 * 4 + 2] * kernels_L2_5[8];

		c[4 * 2 + 3] += temp2[1 * 4 + 1] * kernels_L2_5[6]
			+ temp2[1 * 4 + 2] * kernels_L2_5[7]
			+ temp2[1 * 4 + 3] * kernels_L2_5[8];

		float res = fmax(fmax(c[4 * 2 + 0], c[4 * 2 + 1]), fmax(c[4 * 2 + 2], c[4 * 2 + 3])) + conv_b_L2[5];
		res = tanhf(res);

		res = res * subs_w_L2[5] + subs_b_L2[5];
		res = tanhf(res);

		*(dst_5 + dst_offset) = scale_L2[0] * res;
	}
	{
		c[4 * 3 + 0] += temp2[0 * 4 + 0] * kernels_L2_4[6]
			+ temp2[0 * 4 + 1] * kernels_L2_4[7]
			+ temp2[0 * 4 + 2] * kernels_L2_4[8];

		c[4 * 3 + 1] += temp2[0 * 4 + 1] * kernels_L2_4[6]
			+ temp2[0 * 4 + 2] * kernels_L2_4[7]
			+ temp2[0 * 4 + 3] * kernels_L2_4[8];

		c[4 * 3 + 2] += temp2[1 * 4 + 0] * kernels_L2_4[6]
			+ temp2[1 * 4 + 1] * kernels_L2_4[7]
			+ temp2[1 * 4 + 2] * kernels_L2_4[8];

		c[4 * 3 + 3] += temp2[1 * 4 + 1] * kernels_L2_4[6]
			+ temp2[1 * 4 + 2] * kernels_L2_4[7]
			+ temp2[1 * 4 + 3] * kernels_L2_4[8];

		float res = fmax(fmax(c[4 * 3 + 0], c[4 * 3 + 1]), fmax(c[4 * 3 + 2], c[4 * 3 + 3])) + conv_b_L2[4];
		res = tanhf(res);

		res = res * subs_w_L2[4] + subs_b_L2[4];
		res = tanhf(res);

		*(dst_4 + dst_offset) = scale_L2[0] * res;
	}
}

__kernel void add_2_3_conv_4x6x5_L_cl(
									/*0*/__global float* dst_0,
									/*1*/__global float* dst_1,
									/*2*/__global float* dst_2,
									/*3*/__global float* dst_3,
									/*4*/int dst_step,
									/*5*/__global float* src_L3_1,
									/*6*/__global float* src_L3_2,
									/*7*/__global float* src_L3_3,
									/*8*/int src_cols,
									/*9*/int src_rows,
									/*10*/int src_step,
									/*11*/__constant float* kernels_L3_0,
									/*12*/__constant float* kernels_L3_1,
									/*13*/__constant float* kernels_L3_2,
									/*14*/__constant float* kernels_L3_3
									)
{
	const int i = 2 * get_global_id(0);
	const int j = 2 * get_global_id(1);

	if (i + 1 >= src_cols - 4 || j + 1 >= src_rows - 5) return;

	const int src_offset = j * src_step + i;
	const int dst_offset = j * dst_step + i;

	float temp1[12];

	src_L3_1 += src_offset;
	src_L3_2 += src_offset;
	temp1[0 * 6 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[0 * 6 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[0 * 6 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[0 * 6 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[0 * 6 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);
	temp1[0 * 6 + 5] = *(src_L3_1 + 5) + *(src_L3_2 + 5);

	src_L3_1 += src_step;
	src_L3_2 += src_step;
	temp1[1 * 6 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[1 * 6 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[1 * 6 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[1 * 6 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[1 * 6 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);
	temp1[1 * 6 + 5] = *(src_L3_1 + 5) + *(src_L3_2 + 5);

	float temp2[12];

	src_L3_3 += src_offset;
	temp2[0 * 6 + 0] = temp1[0 * 6 + 0] + *(src_L3_3 + 0);
	temp2[0 * 6 + 1] = temp1[0 * 6 + 1] + *(src_L3_3 + 1);
	temp2[0 * 6 + 2] = temp1[0 * 6 + 2] + *(src_L3_3 + 2);
	temp2[0 * 6 + 3] = temp1[0 * 6 + 3] + *(src_L3_3 + 3);
	temp2[0 * 6 + 4] = temp1[0 * 6 + 4] + *(src_L3_3 + 4);
	temp2[0 * 6 + 5] = temp1[0 * 6 + 5] + *(src_L3_3 + 5);

	src_L3_3 += src_step;
	temp2[1 * 6 + 0] = temp1[1 * 6 + 0] + *(src_L3_3 + 0);
	temp2[1 * 6 + 1] = temp1[1 * 6 + 1] + *(src_L3_3 + 1);
	temp2[1 * 6 + 2] = temp1[1 * 6 + 2] + *(src_L3_3 + 2);
	temp2[1 * 6 + 3] = temp1[1 * 6 + 3] + *(src_L3_3 + 3);
	temp2[1 * 6 + 4] = temp1[1 * 6 + 4] + *(src_L3_3 + 4);
	temp2[1 * 6 + 5] = temp1[1 * 6 + 5] + *(src_L3_3 + 5);

	float c[4 * 4];
	{
		c[4 * 0 + 0] = temp1[0 * 6 + 0] * kernels_L3_0[0]
			+ temp1[0 * 6 + 1] * kernels_L3_0[1]
			+ temp1[0 * 6 + 2] * kernels_L3_0[2]
			+ temp1[0 * 6 + 3] * kernels_L3_0[3]
			+ temp1[0 * 6 + 4] * kernels_L3_0[4];

		c[4 * 0 + 1] = temp1[0 * 6 + 1] * kernels_L3_0[0]
			+ temp1[0 * 6 + 2] * kernels_L3_0[1]
			+ temp1[0 * 6 + 3] * kernels_L3_0[2]
			+ temp1[0 * 6 + 4] * kernels_L3_0[3]
			+ temp1[0 * 6 + 5] * kernels_L3_0[4];

		c[4 * 0 + 2] = temp1[1 * 6 + 0] * kernels_L3_0[0]
			+ temp1[1 * 6 + 1] * kernels_L3_0[1]
			+ temp1[1 * 6 + 2] * kernels_L3_0[2]
			+ temp1[1 * 6 + 3] * kernels_L3_0[3]
			+ temp1[1 * 6 + 4] * kernels_L3_0[4];

		c[4 * 0 + 3] = temp1[1 * 6 + 1] * kernels_L3_0[0]
			+ temp1[1 * 6 + 2] * kernels_L3_0[1]
			+ temp1[1 * 6 + 3] * kernels_L3_0[2]
			+ temp1[1 * 6 + 4] * kernels_L3_0[3]
			+ temp1[1 * 6 + 5] * kernels_L3_0[4];
	}
	{
		c[4 * 1 + 0] = temp1[0 * 6 + 0] * kernels_L3_1[0]
			+ temp1[0 * 6 + 1] * kernels_L3_1[1]
			+ temp1[0 * 6 + 2] * kernels_L3_1[2]
			+ temp1[0 * 6 + 3] * kernels_L3_1[3]
			+ temp1[0 * 6 + 4] * kernels_L3_1[4];

		c[4 * 1 + 1] = temp1[0 * 6 + 1] * kernels_L3_1[0]
			+ temp1[0 * 6 + 2] * kernels_L3_1[1]
			+ temp1[0 * 6 + 3] * kernels_L3_1[2]
			+ temp1[0 * 6 + 4] * kernels_L3_1[3]
			+ temp1[0 * 6 + 5] * kernels_L3_1[4];

		c[4 * 1 + 2] = temp1[1 * 6 + 0] * kernels_L3_1[0]
			+ temp1[1 * 6 + 1] * kernels_L3_1[1]
			+ temp1[1 * 6 + 2] * kernels_L3_1[2]
			+ temp1[1 * 6 + 3] * kernels_L3_1[3]
			+ temp1[1 * 6 + 4] * kernels_L3_1[4];

		c[4 * 1 + 3] = temp1[1 * 6 + 1] * kernels_L3_1[0]
			+ temp1[1 * 6 + 2] * kernels_L3_1[1]
			+ temp1[1 * 6 + 3] * kernels_L3_1[2]
			+ temp1[1 * 6 + 4] * kernels_L3_1[3]
			+ temp1[1 * 6 + 5] * kernels_L3_1[4];
	}
	{
		c[4 * 2 + 0] = temp2[0 * 6 + 0] * kernels_L3_2[0]
			+ temp2[0 * 6 + 1] * kernels_L3_2[1]
			+ temp2[0 * 6 + 2] * kernels_L3_2[2]
			+ temp2[0 * 6 + 3] * kernels_L3_2[3]
			+ temp2[0 * 6 + 4] * kernels_L3_2[4];

		c[4 * 2 + 1] = temp2[0 * 6 + 1] * kernels_L3_2[0]
			+ temp2[0 * 6 + 2] * kernels_L3_2[1]
			+ temp2[0 * 6 + 3] * kernels_L3_2[2]
			+ temp2[0 * 6 + 4] * kernels_L3_2[3]
			+ temp2[0 * 6 + 5] * kernels_L3_2[4];

		c[4 * 2 + 2] = temp2[1 * 6 + 0] * kernels_L3_2[0]
			+ temp2[1 * 6 + 1] * kernels_L3_2[1]
			+ temp2[1 * 6 + 2] * kernels_L3_2[2]
			+ temp2[1 * 6 + 3] * kernels_L3_2[3]
			+ temp2[1 * 6 + 4] * kernels_L3_2[4];

		c[4 * 2 + 3] = temp2[1 * 6 + 1] * kernels_L3_2[0]
			+ temp2[1 * 6 + 2] * kernels_L3_2[1]
			+ temp2[1 * 6 + 3] * kernels_L3_2[2]
			+ temp2[1 * 6 + 4] * kernels_L3_2[3]
			+ temp2[1 * 6 + 5] * kernels_L3_2[4];
	}
	{
		c[4 * 3 + 0] = temp2[0 * 6 + 0] * kernels_L3_3[0]
			+ temp2[0 * 6 + 1] * kernels_L3_3[1]
			+ temp2[0 * 6 + 2] * kernels_L3_3[2]
			+ temp2[0 * 6 + 3] * kernels_L3_3[3]
			+ temp2[0 * 6 + 4] * kernels_L3_3[4];

		c[4 * 3 + 1] = temp2[0 * 6 + 1] * kernels_L3_3[0]
			+ temp2[0 * 6 + 2] * kernels_L3_3[1]
			+ temp2[0 * 6 + 3] * kernels_L3_3[2]
			+ temp2[0 * 6 + 4] * kernels_L3_3[3]
			+ temp2[0 * 6 + 5] * kernels_L3_3[4];

		c[4 * 3 + 2] = temp2[1 * 6 + 0] * kernels_L3_3[0]
			+ temp2[1 * 6 + 1] * kernels_L3_3[1]
			+ temp2[1 * 6 + 2] * kernels_L3_3[2]
			+ temp2[1 * 6 + 3] * kernels_L3_3[3]
			+ temp2[1 * 6 + 4] * kernels_L3_3[4];

		c[4 * 3 + 3] = temp2[1 * 6 + 1] * kernels_L3_3[0]
			+ temp2[1 * 6 + 2] * kernels_L3_3[1]
			+ temp2[1 * 6 + 3] * kernels_L3_3[2]
			+ temp2[1 * 6 + 4] * kernels_L3_3[3]
			+ temp2[1 * 6 + 5] * kernels_L3_3[4];
	}

	src_L3_1 += src_step;
	src_L3_2 += src_step;
	temp1[0 * 6 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[0 * 6 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[0 * 6 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[0 * 6 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[0 * 6 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);
	temp1[0 * 6 + 5] = *(src_L3_1 + 5) + *(src_L3_2 + 5);

	src_L3_3 += src_step;
	temp2[0 * 6 + 0] = temp1[0 * 6 + 0] + *(src_L3_3 + 0);
	temp2[0 * 6 + 1] = temp1[0 * 6 + 1] + *(src_L3_3 + 1);
	temp2[0 * 6 + 2] = temp1[0 * 6 + 2] + *(src_L3_3 + 2);
	temp2[0 * 6 + 3] = temp1[0 * 6 + 3] + *(src_L3_3 + 3);
	temp2[0 * 6 + 4] = temp1[0 * 6 + 4] + *(src_L3_3 + 4);
	temp2[0 * 6 + 5] = temp1[0 * 6 + 5] + *(src_L3_3 + 5);

	{
		c[4 * 0 + 0] += temp1[1 * 6 + 0] * kernels_L3_0[5]
			+ temp1[1 * 6 + 1] * kernels_L3_0[6]
			+ temp1[1 * 6 + 2] * kernels_L3_0[7]
			+ temp1[1 * 6 + 3] * kernels_L3_0[8]
			+ temp1[1 * 6 + 4] * kernels_L3_0[9];

		c[4 * 0 + 1] += temp1[1 * 6 + 1] * kernels_L3_0[5]
			+ temp1[1 * 6 + 2] * kernels_L3_0[6]
			+ temp1[1 * 6 + 3] * kernels_L3_0[7]
			+ temp1[1 * 6 + 4] * kernels_L3_0[8]
			+ temp1[1 * 6 + 5] * kernels_L3_0[9];

		c[4 * 0 + 2] += temp1[0 * 6 + 0] * kernels_L3_0[5]
			+ temp1[0 * 6 + 1] * kernels_L3_0[6]
			+ temp1[0 * 6 + 2] * kernels_L3_0[7]
			+ temp1[0 * 6 + 3] * kernels_L3_0[8]
			+ temp1[0 * 6 + 4] * kernels_L3_0[9];

		c[4 * 0 + 3] += temp1[0 * 6 + 1] * kernels_L3_0[5]
			+ temp1[0 * 6 + 2] * kernels_L3_0[6]
			+ temp1[0 * 6 + 3] * kernels_L3_0[7]
			+ temp1[0 * 6 + 4] * kernels_L3_0[8]
			+ temp1[0 * 6 + 5] * kernels_L3_0[9];
	}
	{
		c[4 * 1 + 0] += temp1[1 * 6 + 0] * kernels_L3_1[5]
			+ temp1[1 * 6 + 1] * kernels_L3_1[6]
			+ temp1[1 * 6 + 2] * kernels_L3_1[7]
			+ temp1[1 * 6 + 3] * kernels_L3_1[8]
			+ temp1[1 * 6 + 4] * kernels_L3_1[9];

		c[4 * 1 + 1] += temp1[1 * 6 + 1] * kernels_L3_1[5]
			+ temp1[1 * 6 + 2] * kernels_L3_1[6]
			+ temp1[1 * 6 + 3] * kernels_L3_1[7]
			+ temp1[1 * 6 + 4] * kernels_L3_1[8]
			+ temp1[1 * 6 + 5] * kernels_L3_1[9];

		c[4 * 1 + 2] += temp1[0 * 6 + 0] * kernels_L3_1[5]
			+ temp1[0 * 6 + 1] * kernels_L3_1[6]
			+ temp1[0 * 6 + 2] * kernels_L3_1[7]
			+ temp1[0 * 6 + 3] * kernels_L3_1[8]
			+ temp1[0 * 6 + 4] * kernels_L3_1[9];

		c[4 * 1 + 3] += temp1[0 * 6 + 1] * kernels_L3_1[5]
			+ temp1[0 * 6 + 2] * kernels_L3_1[6]
			+ temp1[0 * 6 + 3] * kernels_L3_1[7]
			+ temp1[0 * 6 + 4] * kernels_L3_1[8]
			+ temp1[0 * 6 + 5] * kernels_L3_1[9];
	}
	{
		c[4 * 2 + 0] += temp2[1 * 6 + 0] * kernels_L3_2[5]
			+ temp2[1 * 6 + 1] * kernels_L3_2[6]
			+ temp2[1 * 6 + 2] * kernels_L3_2[7]
			+ temp2[1 * 6 + 3] * kernels_L3_2[8]
			+ temp2[1 * 6 + 4] * kernels_L3_2[9];

		c[4 * 2 + 1] += temp2[1 * 6 + 1] * kernels_L3_2[5]
			+ temp2[1 * 6 + 2] * kernels_L3_2[6]
			+ temp2[1 * 6 + 3] * kernels_L3_2[7]
			+ temp2[1 * 6 + 4] * kernels_L3_2[8]
			+ temp2[1 * 6 + 5] * kernels_L3_2[9];

		c[4 * 2 + 2] += temp2[0 * 6 + 0] * kernels_L3_2[5]
			+ temp2[0 * 6 + 1] * kernels_L3_2[6]
			+ temp2[0 * 6 + 2] * kernels_L3_2[7]
			+ temp2[0 * 6 + 3] * kernels_L3_2[8]
			+ temp2[0 * 6 + 4] * kernels_L3_2[9];

		c[4 * 2 + 3] += temp2[0 * 6 + 1] * kernels_L3_2[5]
			+ temp2[0 * 6 + 2] * kernels_L3_2[6]
			+ temp2[0 * 6 + 3] * kernels_L3_2[7]
			+ temp2[0 * 6 + 4] * kernels_L3_2[8]
			+ temp2[0 * 6 + 5] * kernels_L3_2[9];
	}
	{
		c[4 * 3 + 0] += temp2[1 * 6 + 0] * kernels_L3_3[5]
			+ temp2[1 * 6 + 1] * kernels_L3_3[6]
			+ temp2[1 * 6 + 2] * kernels_L3_3[7]
			+ temp2[1 * 6 + 3] * kernels_L3_3[8]
			+ temp2[1 * 6 + 4] * kernels_L3_3[9];

		c[4 * 3 + 1] += temp2[1 * 6 + 1] * kernels_L3_3[5]
			+ temp2[1 * 6 + 2] * kernels_L3_3[6]
			+ temp2[1 * 6 + 3] * kernels_L3_3[7]
			+ temp2[1 * 6 + 4] * kernels_L3_3[8]
			+ temp2[1 * 6 + 5] * kernels_L3_3[9];

		c[4 * 3 + 2] += temp2[0 * 6 + 0] * kernels_L3_3[5]
			+ temp2[0 * 6 + 1] * kernels_L3_3[6]
			+ temp2[0 * 6 + 2] * kernels_L3_3[7]
			+ temp2[0 * 6 + 3] * kernels_L3_3[8]
			+ temp2[0 * 6 + 4] * kernels_L3_3[9];

		c[4 * 3 + 3] += temp2[0 * 6 + 1] * kernels_L3_3[5]
			+ temp2[0 * 6 + 2] * kernels_L3_3[6]
			+ temp2[0 * 6 + 3] * kernels_L3_3[7]
			+ temp2[0 * 6 + 4] * kernels_L3_3[8]
			+ temp2[0 * 6 + 5] * kernels_L3_3[9];
	}

	src_L3_1 += src_step;
	src_L3_2 += src_step;
	temp1[1 * 6 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[1 * 6 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[1 * 6 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[1 * 6 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[1 * 6 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);
	temp1[1 * 6 + 5] = *(src_L3_1 + 5) + *(src_L3_2 + 5);

	src_L3_3 += src_step;
	temp2[1 * 6 + 0] = temp1[1 * 6 + 0] + *(src_L3_3 + 0);
	temp2[1 * 6 + 1] = temp1[1 * 6 + 1] + *(src_L3_3 + 1);
	temp2[1 * 6 + 2] = temp1[1 * 6 + 2] + *(src_L3_3 + 2);
	temp2[1 * 6 + 3] = temp1[1 * 6 + 3] + *(src_L3_3 + 3);
	temp2[1 * 6 + 4] = temp1[1 * 6 + 4] + *(src_L3_3 + 4);
	temp2[1 * 6 + 5] = temp1[1 * 6 + 5] + *(src_L3_3 + 5);

	{
		c[4 * 0 + 0] += temp1[0 * 6 + 0] * kernels_L3_0[10]
			+ temp1[0 * 6 + 1] * kernels_L3_0[11]
			+ temp1[0 * 6 + 2] * kernels_L3_0[12]
			+ temp1[0 * 6 + 3] * kernels_L3_0[13]
			+ temp1[0 * 6 + 4] * kernels_L3_0[14];

		c[4 * 0 + 1] += temp1[0 * 6 + 1] * kernels_L3_0[10]
			+ temp1[0 * 6 + 2] * kernels_L3_0[11]
			+ temp1[0 * 6 + 3] * kernels_L3_0[12]
			+ temp1[0 * 6 + 4] * kernels_L3_0[13]
			+ temp1[0 * 6 + 5] * kernels_L3_0[14];

		c[4 * 0 + 2] += temp1[1 * 6 + 0] * kernels_L3_0[10]
			+ temp1[1 * 6 + 1] * kernels_L3_0[11]
			+ temp1[1 * 6 + 2] * kernels_L3_0[12]
			+ temp1[1 * 6 + 3] * kernels_L3_0[13]
			+ temp1[1 * 6 + 4] * kernels_L3_0[14];

		c[4 * 0 + 3] += temp1[1 * 6 + 1] * kernels_L3_0[10]
			+ temp1[1 * 6 + 2] * kernels_L3_0[11]
			+ temp1[1 * 6 + 3] * kernels_L3_0[12]
			+ temp1[1 * 6 + 4] * kernels_L3_0[13]
			+ temp1[1 * 6 + 5] * kernels_L3_0[14];
	}
	{
		c[4 * 1 + 0] += temp1[0 * 6 + 0] * kernels_L3_1[10]
			+ temp1[0 * 6 + 1] * kernels_L3_1[11]
			+ temp1[0 * 6 + 2] * kernels_L3_1[12]
			+ temp1[0 * 6 + 3] * kernels_L3_1[13]
			+ temp1[0 * 6 + 4] * kernels_L3_1[14];

		c[4 * 1 + 1] += temp1[0 * 6 + 1] * kernels_L3_1[10]
			+ temp1[0 * 6 + 2] * kernels_L3_1[11]
			+ temp1[0 * 6 + 3] * kernels_L3_1[12]
			+ temp1[0 * 6 + 4] * kernels_L3_1[13]
			+ temp1[0 * 6 + 5] * kernels_L3_1[14];

		c[4 * 1 + 2] += temp1[1 * 6 + 0] * kernels_L3_1[10]
			+ temp1[1 * 6 + 1] * kernels_L3_1[11]
			+ temp1[1 * 6 + 2] * kernels_L3_1[12]
			+ temp1[1 * 6 + 3] * kernels_L3_1[13]
			+ temp1[1 * 6 + 4] * kernels_L3_1[14];

		c[4 * 1 + 3] += temp1[1 * 6 + 1] * kernels_L3_1[10]
			+ temp1[1 * 6 + 2] * kernels_L3_1[11]
			+ temp1[1 * 6 + 3] * kernels_L3_1[12]
			+ temp1[1 * 6 + 4] * kernels_L3_1[13]
			+ temp1[1 * 6 + 5] * kernels_L3_1[14];
	}
	{
		c[4 * 2 + 0] += temp2[0 * 6 + 0] * kernels_L3_2[10]
			+ temp2[0 * 6 + 1] * kernels_L3_2[11]
			+ temp2[0 * 6 + 2] * kernels_L3_2[12]
			+ temp2[0 * 6 + 3] * kernels_L3_2[13]
			+ temp2[0 * 6 + 4] * kernels_L3_2[14];

		c[4 * 2 + 1] += temp2[0 * 6 + 1] * kernels_L3_2[10]
			+ temp2[0 * 6 + 2] * kernels_L3_2[11]
			+ temp2[0 * 6 + 3] * kernels_L3_2[12]
			+ temp2[0 * 6 + 4] * kernels_L3_2[13]
			+ temp2[0 * 6 + 5] * kernels_L3_2[14];

		c[4 * 2 + 2] += temp2[1 * 6 + 0] * kernels_L3_2[10]
			+ temp2[1 * 6 + 1] * kernels_L3_2[11]
			+ temp2[1 * 6 + 2] * kernels_L3_2[12]
			+ temp2[1 * 6 + 3] * kernels_L3_2[13]
			+ temp2[1 * 6 + 4] * kernels_L3_2[14];

		c[4 * 2 + 3] += temp2[1 * 6 + 1] * kernels_L3_2[10]
			+ temp2[1 * 6 + 2] * kernels_L3_2[11]
			+ temp2[1 * 6 + 3] * kernels_L3_2[12]
			+ temp2[1 * 6 + 4] * kernels_L3_2[13]
			+ temp2[1 * 6 + 5] * kernels_L3_2[14];
	}
	{
		c[4 * 3 + 0] += temp2[0 * 6 + 0] * kernels_L3_3[10]
			+ temp2[0 * 6 + 1] * kernels_L3_3[11]
			+ temp2[0 * 6 + 2] * kernels_L3_3[12]
			+ temp2[0 * 6 + 3] * kernels_L3_3[13]
			+ temp2[0 * 6 + 4] * kernels_L3_3[14];

		c[4 * 3 + 1] += temp2[0 * 6 + 1] * kernels_L3_3[10]
			+ temp2[0 * 6 + 2] * kernels_L3_3[11]
			+ temp2[0 * 6 + 3] * kernels_L3_3[12]
			+ temp2[0 * 6 + 4] * kernels_L3_3[13]
			+ temp2[0 * 6 + 5] * kernels_L3_3[14];

		c[4 * 3 + 2] += temp2[1 * 6 + 0] * kernels_L3_3[10]
			+ temp2[1 * 6 + 1] * kernels_L3_3[11]
			+ temp2[1 * 6 + 2] * kernels_L3_3[12]
			+ temp2[1 * 6 + 3] * kernels_L3_3[13]
			+ temp2[1 * 6 + 4] * kernels_L3_3[14];

		c[4 * 3 + 3] += temp2[1 * 6 + 1] * kernels_L3_3[10]
			+ temp2[1 * 6 + 2] * kernels_L3_3[11]
			+ temp2[1 * 6 + 3] * kernels_L3_3[12]
			+ temp2[1 * 6 + 4] * kernels_L3_3[13]
			+ temp2[1 * 6 + 5] * kernels_L3_3[14];
	}

	src_L3_1 += src_step;
	src_L3_2 += src_step;
	temp1[0 * 6 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[0 * 6 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[0 * 6 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[0 * 6 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[0 * 6 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);
	temp1[0 * 6 + 5] = *(src_L3_1 + 5) + *(src_L3_2 + 5);

	src_L3_3 += src_step;
	temp2[0 * 6 + 0] = temp1[0 * 6 + 0] + *(src_L3_3 + 0);
	temp2[0 * 6 + 1] = temp1[0 * 6 + 1] + *(src_L3_3 + 1);
	temp2[0 * 6 + 2] = temp1[0 * 6 + 2] + *(src_L3_3 + 2);
	temp2[0 * 6 + 3] = temp1[0 * 6 + 3] + *(src_L3_3 + 3);
	temp2[0 * 6 + 4] = temp1[0 * 6 + 4] + *(src_L3_3 + 4);
	temp2[0 * 6 + 5] = temp1[0 * 6 + 5] + *(src_L3_3 + 5);

	{
		c[4 * 0 + 0] += temp1[1 * 6 + 0] * kernels_L3_0[15]
			+ temp1[1 * 6 + 1] * kernels_L3_0[16]
			+ temp1[1 * 6 + 2] * kernels_L3_0[17]
			+ temp1[1 * 6 + 3] * kernels_L3_0[18]
			+ temp1[1 * 6 + 4] * kernels_L3_0[19];

		c[4 * 0 + 1] += temp1[1 * 6 + 1] * kernels_L3_0[15]
			+ temp1[1 * 6 + 2] * kernels_L3_0[16]
			+ temp1[1 * 6 + 3] * kernels_L3_0[17]
			+ temp1[1 * 6 + 4] * kernels_L3_0[18]
			+ temp1[1 * 6 + 5] * kernels_L3_0[19];

		c[4 * 0 + 2] += temp1[0 * 6 + 0] * kernels_L3_0[15]
			+ temp1[0 * 6 + 1] * kernels_L3_0[16]
			+ temp1[0 * 6 + 2] * kernels_L3_0[17]
			+ temp1[0 * 6 + 3] * kernels_L3_0[18]
			+ temp1[0 * 6 + 4] * kernels_L3_0[19];

		c[4 * 0 + 3] += temp1[0 * 6 + 1] * kernels_L3_0[15]
			+ temp1[0 * 6 + 2] * kernels_L3_0[16]
			+ temp1[0 * 6 + 3] * kernels_L3_0[17]
			+ temp1[0 * 6 + 4] * kernels_L3_0[18]
			+ temp1[0 * 6 + 5] * kernels_L3_0[19];
	}
	{
		c[4 * 1 + 0] += temp1[1 * 6 + 0] * kernels_L3_1[15]
			+ temp1[1 * 6 + 1] * kernels_L3_1[16]
			+ temp1[1 * 6 + 2] * kernels_L3_1[17]
			+ temp1[1 * 6 + 3] * kernels_L3_1[18]
			+ temp1[1 * 6 + 4] * kernels_L3_1[19];

		c[4 * 1 + 1] += temp1[1 * 6 + 1] * kernels_L3_1[15]
			+ temp1[1 * 6 + 2] * kernels_L3_1[16]
			+ temp1[1 * 6 + 3] * kernels_L3_1[17]
			+ temp1[1 * 6 + 4] * kernels_L3_1[18]
			+ temp1[1 * 6 + 5] * kernels_L3_1[19];

		c[4 * 1 + 2] += temp1[0 * 6 + 0] * kernels_L3_1[15]
			+ temp1[0 * 6 + 1] * kernels_L3_1[16]
			+ temp1[0 * 6 + 2] * kernels_L3_1[17]
			+ temp1[0 * 6 + 3] * kernels_L3_1[18]
			+ temp1[0 * 6 + 4] * kernels_L3_1[19];

		c[4 * 1 + 3] += temp1[0 * 6 + 1] * kernels_L3_1[15]
			+ temp1[0 * 6 + 2] * kernels_L3_1[16]
			+ temp1[0 * 6 + 3] * kernels_L3_1[17]
			+ temp1[0 * 6 + 4] * kernels_L3_1[18]
			+ temp1[0 * 6 + 5] * kernels_L3_1[19];
	}
	{
		c[4 * 2 + 0] += temp2[1 * 6 + 0] * kernels_L3_2[15]
			+ temp2[1 * 6 + 1] * kernels_L3_2[16]
			+ temp2[1 * 6 + 2] * kernels_L3_2[17]
			+ temp2[1 * 6 + 3] * kernels_L3_2[18]
			+ temp2[1 * 6 + 4] * kernels_L3_2[19];

		c[4 * 2 + 1] += temp2[1 * 6 + 1] * kernels_L3_2[15]
			+ temp2[1 * 6 + 2] * kernels_L3_2[16]
			+ temp2[1 * 6 + 3] * kernels_L3_2[17]
			+ temp2[1 * 6 + 4] * kernels_L3_2[18]
			+ temp2[1 * 6 + 5] * kernels_L3_2[19];

		c[4 * 2 + 2] += temp2[0 * 6 + 0] * kernels_L3_2[15]
			+ temp2[0 * 6 + 1] * kernels_L3_2[16]
			+ temp2[0 * 6 + 2] * kernels_L3_2[17]
			+ temp2[0 * 6 + 3] * kernels_L3_2[18]
			+ temp2[0 * 6 + 4] * kernels_L3_2[19];

		c[4 * 2 + 3] += temp2[0 * 6 + 1] * kernels_L3_2[15]
			+ temp2[0 * 6 + 2] * kernels_L3_2[16]
			+ temp2[0 * 6 + 3] * kernels_L3_2[17]
			+ temp2[0 * 6 + 4] * kernels_L3_2[18]
			+ temp2[0 * 6 + 5] * kernels_L3_2[19];
	}
	{
		c[4 * 3 + 0] += temp2[1 * 6 + 0] * kernels_L3_3[15]
			+ temp2[1 * 6 + 1] * kernels_L3_3[16]
			+ temp2[1 * 6 + 2] * kernels_L3_3[17]
			+ temp2[1 * 6 + 3] * kernels_L3_3[18]
			+ temp2[1 * 6 + 4] * kernels_L3_3[19];

		c[4 * 3 + 1] += temp2[1 * 6 + 1] * kernels_L3_3[15]
			+ temp2[1 * 6 + 2] * kernels_L3_3[16]
			+ temp2[1 * 6 + 3] * kernels_L3_3[17]
			+ temp2[1 * 6 + 4] * kernels_L3_3[18]
			+ temp2[1 * 6 + 5] * kernels_L3_3[19];

		c[4 * 3 + 2] += temp2[0 * 6 + 0] * kernels_L3_3[15]
			+ temp2[0 * 6 + 1] * kernels_L3_3[16]
			+ temp2[0 * 6 + 2] * kernels_L3_3[17]
			+ temp2[0 * 6 + 3] * kernels_L3_3[18]
			+ temp2[0 * 6 + 4] * kernels_L3_3[19];

		c[4 * 3 + 3] += temp2[0 * 6 + 1] * kernels_L3_3[15]
			+ temp2[0 * 6 + 2] * kernels_L3_3[16]
			+ temp2[0 * 6 + 3] * kernels_L3_3[17]
			+ temp2[0 * 6 + 4] * kernels_L3_3[18]
			+ temp2[0 * 6 + 5] * kernels_L3_3[19];
	}

	src_L3_1 += src_step;
	src_L3_2 += src_step;
	temp1[1 * 6 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[1 * 6 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[1 * 6 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[1 * 6 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[1 * 6 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);
	temp1[1 * 6 + 5] = *(src_L3_1 + 5) + *(src_L3_2 + 5);

	src_L3_3 += src_step;
	temp2[1 * 6 + 0] = temp1[1 * 6 + 0] + *(src_L3_3 + 0);
	temp2[1 * 6 + 1] = temp1[1 * 6 + 1] + *(src_L3_3 + 1);
	temp2[1 * 6 + 2] = temp1[1 * 6 + 2] + *(src_L3_3 + 2);
	temp2[1 * 6 + 3] = temp1[1 * 6 + 3] + *(src_L3_3 + 3);
	temp2[1 * 6 + 4] = temp1[1 * 6 + 4] + *(src_L3_3 + 4);
	temp2[1 * 6 + 5] = temp1[1 * 6 + 5] + *(src_L3_3 + 5);

	{
		c[4 * 0 + 0] += temp1[0 * 6 + 0] * kernels_L3_0[20]
			+ temp1[0 * 6 + 1] * kernels_L3_0[21]
			+ temp1[0 * 6 + 2] * kernels_L3_0[22]
			+ temp1[0 * 6 + 3] * kernels_L3_0[23]
			+ temp1[0 * 6 + 4] * kernels_L3_0[24];

		c[4 * 0 + 1] += temp1[0 * 6 + 1] * kernels_L3_0[20]
			+ temp1[0 * 6 + 2] * kernels_L3_0[21]
			+ temp1[0 * 6 + 3] * kernels_L3_0[22]
			+ temp1[0 * 6 + 4] * kernels_L3_0[23]
			+ temp1[0 * 6 + 5] * kernels_L3_0[24];

		c[4 * 0 + 2] += temp1[1 * 6 + 0] * kernels_L3_0[20]
			+ temp1[1 * 6 + 1] * kernels_L3_0[21]
			+ temp1[1 * 6 + 2] * kernels_L3_0[22]
			+ temp1[1 * 6 + 3] * kernels_L3_0[23]
			+ temp1[1 * 6 + 4] * kernels_L3_0[24];

		c[4 * 0 + 3] += temp1[1 * 6 + 1] * kernels_L3_0[20]
			+ temp1[1 * 6 + 2] * kernels_L3_0[21]
			+ temp1[1 * 6 + 3] * kernels_L3_0[22]
			+ temp1[1 * 6 + 4] * kernels_L3_0[23]
			+ temp1[1 * 6 + 5] * kernels_L3_0[24];
	}
	{
		c[4 * 1 + 0] += temp1[0 * 6 + 0] * kernels_L3_1[20]
			+ temp1[0 * 6 + 1] * kernels_L3_1[21]
			+ temp1[0 * 6 + 2] * kernels_L3_1[22]
			+ temp1[0 * 6 + 3] * kernels_L3_1[23]
			+ temp1[0 * 6 + 4] * kernels_L3_1[24];

		c[4 * 1 + 1] += temp1[0 * 6 + 1] * kernels_L3_1[20]
			+ temp1[0 * 6 + 2] * kernels_L3_1[21]
			+ temp1[0 * 6 + 3] * kernels_L3_1[22]
			+ temp1[0 * 6 + 4] * kernels_L3_1[23]
			+ temp1[0 * 6 + 5] * kernels_L3_1[24];

		c[4 * 1 + 2] += temp1[1 * 6 + 0] * kernels_L3_1[20]
			+ temp1[1 * 6 + 1] * kernels_L3_1[21]
			+ temp1[1 * 6 + 2] * kernels_L3_1[22]
			+ temp1[1 * 6 + 3] * kernels_L3_1[23]
			+ temp1[1 * 6 + 4] * kernels_L3_1[24];

		c[4 * 1 + 3] += temp1[1 * 6 + 1] * kernels_L3_1[20]
			+ temp1[1 * 6 + 2] * kernels_L3_1[21]
			+ temp1[1 * 6 + 3] * kernels_L3_1[22]
			+ temp1[1 * 6 + 4] * kernels_L3_1[23]
			+ temp1[1 * 6 + 5] * kernels_L3_1[24];
	}
	{
		c[4 * 2 + 0] += temp2[0 * 6 + 0] * kernels_L3_2[20]
			+ temp2[0 * 6 + 1] * kernels_L3_2[21]
			+ temp2[0 * 6 + 2] * kernels_L3_2[22]
			+ temp2[0 * 6 + 3] * kernels_L3_2[23]
			+ temp2[0 * 6 + 4] * kernels_L3_2[24];

		c[4 * 2 + 1] += temp2[0 * 6 + 1] * kernels_L3_2[20]
			+ temp2[0 * 6 + 2] * kernels_L3_2[21]
			+ temp2[0 * 6 + 3] * kernels_L3_2[22]
			+ temp2[0 * 6 + 4] * kernels_L3_2[23]
			+ temp2[0 * 6 + 5] * kernels_L3_2[24];

		c[4 * 2 + 2] += temp2[1 * 6 + 0] * kernels_L3_2[20]
			+ temp2[1 * 6 + 1] * kernels_L3_2[21]
			+ temp2[1 * 6 + 2] * kernels_L3_2[22]
			+ temp2[1 * 6 + 3] * kernels_L3_2[23]
			+ temp2[1 * 6 + 4] * kernels_L3_2[24];

		c[4 * 2 + 3] += temp2[1 * 6 + 1] * kernels_L3_2[20]
			+ temp2[1 * 6 + 2] * kernels_L3_2[21]
			+ temp2[1 * 6 + 3] * kernels_L3_2[22]
			+ temp2[1 * 6 + 4] * kernels_L3_2[23]
			+ temp2[1 * 6 + 5] * kernels_L3_2[24];
	}
	{
		c[4 * 3 + 0] += temp2[0 * 6 + 0] * kernels_L3_3[20]
			+ temp2[0 * 6 + 1] * kernels_L3_3[21]
			+ temp2[0 * 6 + 2] * kernels_L3_3[22]
			+ temp2[0 * 6 + 3] * kernels_L3_3[23]
			+ temp2[0 * 6 + 4] * kernels_L3_3[24];

		c[4 * 3 + 1] += temp2[0 * 6 + 1] * kernels_L3_3[20]
			+ temp2[0 * 6 + 2] * kernels_L3_3[21]
			+ temp2[0 * 6 + 3] * kernels_L3_3[22]
			+ temp2[0 * 6 + 4] * kernels_L3_3[23]
			+ temp2[0 * 6 + 5] * kernels_L3_3[24];

		c[4 * 3 + 2] += temp2[1 * 6 + 0] * kernels_L3_3[20]
			+ temp2[1 * 6 + 1] * kernels_L3_3[21]
			+ temp2[1 * 6 + 2] * kernels_L3_3[22]
			+ temp2[1 * 6 + 3] * kernels_L3_3[23]
			+ temp2[1 * 6 + 4] * kernels_L3_3[24];

		c[4 * 3 + 3] += temp2[1 * 6 + 1] * kernels_L3_3[20]
			+ temp2[1 * 6 + 2] * kernels_L3_3[21]
			+ temp2[1 * 6 + 3] * kernels_L3_3[22]
			+ temp2[1 * 6 + 4] * kernels_L3_3[23]
			+ temp2[1 * 6 + 5] * kernels_L3_3[24];
	}

	src_L3_1 += src_step;
	src_L3_2 += src_step;
	temp1[0 * 6 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[0 * 6 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[0 * 6 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[0 * 6 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[0 * 6 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);
	temp1[0 * 6 + 5] = *(src_L3_1 + 5) + *(src_L3_2 + 5);

	src_L3_3 += src_step;
	temp2[0 * 6 + 0] = temp1[0 * 6 + 0] + *(src_L3_3 + 0);
	temp2[0 * 6 + 1] = temp1[0 * 6 + 1] + *(src_L3_3 + 1);
	temp2[0 * 6 + 2] = temp1[0 * 6 + 2] + *(src_L3_3 + 2);
	temp2[0 * 6 + 3] = temp1[0 * 6 + 3] + *(src_L3_3 + 3);
	temp2[0 * 6 + 4] = temp1[0 * 6 + 4] + *(src_L3_3 + 4);
	temp2[0 * 6 + 5] = temp1[0 * 6 + 5] + *(src_L3_3 + 5);

	{
		c[4 * 0 + 0] += temp1[1 * 6 + 0] * kernels_L3_0[25]
			+ temp1[1 * 6 + 1] * kernels_L3_0[26]
			+ temp1[1 * 6 + 2] * kernels_L3_0[27]
			+ temp1[1 * 6 + 3] * kernels_L3_0[28]
			+ temp1[1 * 6 + 4] * kernels_L3_0[29];

		c[4 * 0 + 1] += temp1[1 * 6 + 1] * kernels_L3_0[25]
			+ temp1[1 * 6 + 2] * kernels_L3_0[26]
			+ temp1[1 * 6 + 3] * kernels_L3_0[27]
			+ temp1[1 * 6 + 4] * kernels_L3_0[28]
			+ temp1[1 * 6 + 5] * kernels_L3_0[29];

		c[4 * 0 + 2] += temp1[0 * 6 + 0] * kernels_L3_0[25]
			+ temp1[0 * 6 + 1] * kernels_L3_0[26]
			+ temp1[0 * 6 + 2] * kernels_L3_0[27]
			+ temp1[0 * 6 + 3] * kernels_L3_0[28]
			+ temp1[0 * 6 + 4] * kernels_L3_0[29];

		c[4 * 0 + 3] += temp1[0 * 6 + 1] * kernels_L3_0[25]
			+ temp1[0 * 6 + 2] * kernels_L3_0[26]
			+ temp1[0 * 6 + 3] * kernels_L3_0[27]
			+ temp1[0 * 6 + 4] * kernels_L3_0[28]
			+ temp1[0 * 6 + 5] * kernels_L3_0[29];

		dst_0 += dst_offset;
		*(dst_0 + 0) = c[4 * 0 + 0];
		*(dst_0 + 1) = c[4 * 0 + 1];
		dst_0 += dst_step;
		*(dst_0 + 0) = c[4 * 0 + 2];
		*(dst_0 + 1) = c[4 * 0 + 3];
	}
	{
		c[4 * 1 + 0] += temp1[1 * 6 + 0] * kernels_L3_1[25]
			+ temp1[1 * 6 + 1] * kernels_L3_1[26]
			+ temp1[1 * 6 + 2] * kernels_L3_1[27]
			+ temp1[1 * 6 + 3] * kernels_L3_1[28]
			+ temp1[1 * 6 + 4] * kernels_L3_1[29];

		c[4 * 1 + 1] += temp1[1 * 6 + 1] * kernels_L3_1[25]
			+ temp1[1 * 6 + 2] * kernels_L3_1[26]
			+ temp1[1 * 6 + 3] * kernels_L3_1[27]
			+ temp1[1 * 6 + 4] * kernels_L3_1[28]
			+ temp1[1 * 6 + 5] * kernels_L3_1[29];

		c[4 * 1 + 2] += temp1[0 * 6 + 0] * kernels_L3_1[25]
			+ temp1[0 * 6 + 1] * kernels_L3_1[26]
			+ temp1[0 * 6 + 2] * kernels_L3_1[27]
			+ temp1[0 * 6 + 3] * kernels_L3_1[28]
			+ temp1[0 * 6 + 4] * kernels_L3_1[29];

		c[4 * 1 + 3] += temp1[0 * 6 + 1] * kernels_L3_1[25]
			+ temp1[0 * 6 + 2] * kernels_L3_1[26]
			+ temp1[0 * 6 + 3] * kernels_L3_1[27]
			+ temp1[0 * 6 + 4] * kernels_L3_1[28]
			+ temp1[0 * 6 + 5] * kernels_L3_1[29];

		dst_1 += dst_offset;
		*(dst_1 + 0) = c[4 * 1 + 0];
		*(dst_1 + 1) = c[4 * 1 + 1];
		dst_1 += dst_step;
		*(dst_1 + 0) = c[4 * 1 + 2];
		*(dst_1 + 1) = c[4 * 1 + 3];
	}
	{
		c[4 * 2 + 0] += temp2[1 * 6 + 0] * kernels_L3_2[25]
			+ temp2[1 * 6 + 1] * kernels_L3_2[26]
			+ temp2[1 * 6 + 2] * kernels_L3_2[27]
			+ temp2[1 * 6 + 3] * kernels_L3_2[28]
			+ temp2[1 * 6 + 4] * kernels_L3_2[29];

		c[4 * 2 + 1] += temp2[1 * 6 + 1] * kernels_L3_2[25]
			+ temp2[1 * 6 + 2] * kernels_L3_2[26]
			+ temp2[1 * 6 + 3] * kernels_L3_2[27]
			+ temp2[1 * 6 + 4] * kernels_L3_2[28]
			+ temp2[1 * 6 + 5] * kernels_L3_2[29];

		c[4 * 2 + 2] += temp2[0 * 6 + 0] * kernels_L3_2[25]
			+ temp2[0 * 6 + 1] * kernels_L3_2[26]
			+ temp2[0 * 6 + 2] * kernels_L3_2[27]
			+ temp2[0 * 6 + 3] * kernels_L3_2[28]
			+ temp2[0 * 6 + 4] * kernels_L3_2[29];

		c[4 * 2 + 3] += temp2[0 * 6 + 1] * kernels_L3_2[25]
			+ temp2[0 * 6 + 2] * kernels_L3_2[26]
			+ temp2[0 * 6 + 3] * kernels_L3_2[27]
			+ temp2[0 * 6 + 4] * kernels_L3_2[28]
			+ temp2[0 * 6 + 5] * kernels_L3_2[29];

		dst_2 += dst_offset;
		*(dst_2 + 0) = c[4 * 2 + 0];
		*(dst_2 + 1) = c[4 * 2 + 1];
		dst_2 += dst_step;
		*(dst_2 + 0) = c[4 * 2 + 2];
		*(dst_2 + 1) = c[4 * 2 + 3];
	}
	{
		c[4 * 3 + 0] += temp2[1 * 6 + 0] * kernels_L3_3[25]
			+ temp2[1 * 6 + 1] * kernels_L3_3[26]
			+ temp2[1 * 6 + 2] * kernels_L3_3[27]
			+ temp2[1 * 6 + 3] * kernels_L3_3[28]
			+ temp2[1 * 6 + 4] * kernels_L3_3[29];

		c[4 * 3 + 1] += temp2[1 * 6 + 1] * kernels_L3_3[25]
			+ temp2[1 * 6 + 2] * kernels_L3_3[26]
			+ temp2[1 * 6 + 3] * kernels_L3_3[27]
			+ temp2[1 * 6 + 4] * kernels_L3_3[28]
			+ temp2[1 * 6 + 5] * kernels_L3_3[29];

		c[4 * 3 + 2] += temp2[0 * 6 + 0] * kernels_L3_3[25]
			+ temp2[0 * 6 + 1] * kernels_L3_3[26]
			+ temp2[0 * 6 + 2] * kernels_L3_3[27]
			+ temp2[0 * 6 + 3] * kernels_L3_3[28]
			+ temp2[0 * 6 + 4] * kernels_L3_3[29];

		c[4 * 3 + 3] += temp2[0 * 6 + 1] * kernels_L3_3[25]
			+ temp2[0 * 6 + 2] * kernels_L3_3[26]
			+ temp2[0 * 6 + 3] * kernels_L3_3[27]
			+ temp2[0 * 6 + 4] * kernels_L3_3[28]
			+ temp2[0 * 6 + 5] * kernels_L3_3[29];

		dst_3 += dst_offset;
		*(dst_3 + 0) = c[4 * 3 + 0];
		*(dst_3 + 1) = c[4 * 3 + 1];
		dst_3 += dst_step;
		*(dst_3 + 0) = c[4 * 3 + 2];
		*(dst_3 + 1) = c[4 * 3 + 3];
	}
}
__kernel void add_3_conv_4x6x5_1_cl(
									/*0*/__global float* dst_4,
									/*1*/__global float* dst_5,
									/*2*/__global float* dst_6,
									/*3*/__global float* dst_7,
									/*4*/int dst_step,
									/*5*/__global float* src_L3_2,
									/*6*/__global float* src_L3_3,
									/*7*/__global float* src_L3_4,
									/*8*/__global float* src_L3_5,
									/*9*/int src_cols,
									/*10*/int src_rows,
									/*11*/int src_step,
									/*12*/__constant float* kernels_L3_4,
									/*13*/__constant float* kernels_L3_5,
									/*14*/__constant float* kernels_L3_6,
									/*15*/__constant float* kernels_L3_7
									)

{
	const int i = 2 * get_global_id(0);
	const int j = 2 * get_global_id(1);

	if (i + 1 >= src_cols - 4 || j + 1 >= src_rows - 5) return;

	const int src_offset = j * src_step + i;
	const int dst_offset = j * dst_step + i;

	float temp1[12];

	src_L3_3 += src_offset;
	src_L3_4 += src_offset;
	temp1[0 * 6 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[0 * 6 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[0 * 6 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[0 * 6 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[0 * 6 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);
	temp1[0 * 6 + 5] = *(src_L3_3 + 5) + *(src_L3_4 + 5);

	src_L3_3 += src_step;
	src_L3_4 += src_step;
	temp1[1 * 6 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[1 * 6 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[1 * 6 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[1 * 6 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[1 * 6 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);
	temp1[1 * 6 + 5] = *(src_L3_3 + 5) + *(src_L3_4 + 5);

	float temp2[12];

	src_L3_5 += src_offset;
	temp2[0 * 6 + 0] = temp1[0 * 6 + 0] + *(src_L3_5 + 0);
	temp2[0 * 6 + 1] = temp1[0 * 6 + 1] + *(src_L3_5 + 1);
	temp2[0 * 6 + 2] = temp1[0 * 6 + 2] + *(src_L3_5 + 2);
	temp2[0 * 6 + 3] = temp1[0 * 6 + 3] + *(src_L3_5 + 3);
	temp2[0 * 6 + 4] = temp1[0 * 6 + 4] + *(src_L3_5 + 4);
	temp2[0 * 6 + 5] = temp1[0 * 6 + 5] + *(src_L3_5 + 5);

	src_L3_5 += src_step;
	temp2[1 * 6 + 0] = temp1[1 * 6 + 0] + *(src_L3_5 + 0);
	temp2[1 * 6 + 1] = temp1[1 * 6 + 1] + *(src_L3_5 + 1);
	temp2[1 * 6 + 2] = temp1[1 * 6 + 2] + *(src_L3_5 + 2);
	temp2[1 * 6 + 3] = temp1[1 * 6 + 3] + *(src_L3_5 + 3);
	temp2[1 * 6 + 4] = temp1[1 * 6 + 4] + *(src_L3_5 + 4);
	temp2[1 * 6 + 5] = temp1[1 * 6 + 5] + *(src_L3_5 + 5);


	src_L3_2 += src_offset;
	temp1[0 * 6 + 0] += *(src_L3_2 + 0);
	temp1[0 * 6 + 1] += *(src_L3_2 + 1);
	temp1[0 * 6 + 2] += *(src_L3_2 + 2);
	temp1[0 * 6 + 3] += *(src_L3_2 + 3);
	temp1[0 * 6 + 4] += *(src_L3_2 + 4);
	temp1[0 * 6 + 5] += *(src_L3_2 + 5);
	
	src_L3_2 += src_step;
	temp1[1 * 6 + 0] += *(src_L3_2 + 0);
	temp1[1 * 6 + 1] += *(src_L3_2 + 1);
	temp1[1 * 6 + 2] += *(src_L3_2 + 2);
	temp1[1 * 6 + 3] += *(src_L3_2 + 3);
	temp1[1 * 6 + 4] += *(src_L3_2 + 4);
	temp1[1 * 6 + 5] += *(src_L3_2 + 5);
	
	float c[4 * 4];
	{
		c[4 * 0 + 0] = temp1[0 * 6 + 0] * kernels_L3_4[0]
			+ temp1[0 * 6 + 1] * kernels_L3_4[1]
			+ temp1[0 * 6 + 2] * kernels_L3_4[2]
			+ temp1[0 * 6 + 3] * kernels_L3_4[3]
			+ temp1[0 * 6 + 4] * kernels_L3_4[4];

		c[4 * 0 + 1] = temp1[0 * 6 + 1] * kernels_L3_4[0]
			+ temp1[0 * 6 + 2] * kernels_L3_4[1]
			+ temp1[0 * 6 + 3] * kernels_L3_4[2]
			+ temp1[0 * 6 + 4] * kernels_L3_4[3]
			+ temp1[0 * 6 + 5] * kernels_L3_4[4];

		c[4 * 0 + 2] = temp1[1 * 6 + 0] * kernels_L3_4[0]
			+ temp1[1 * 6 + 1] * kernels_L3_4[1]
			+ temp1[1 * 6 + 2] * kernels_L3_4[2]
			+ temp1[1 * 6 + 3] * kernels_L3_4[3]
			+ temp1[1 * 6 + 4] * kernels_L3_4[4];

		c[4 * 0 + 3] = temp1[1 * 6 + 1] * kernels_L3_4[0]
			+ temp1[1 * 6 + 2] * kernels_L3_4[1]
			+ temp1[1 * 6 + 3] * kernels_L3_4[2]
			+ temp1[1 * 6 + 4] * kernels_L3_4[3]
			+ temp1[1 * 6 + 5] * kernels_L3_4[4];
	}
	{
		c[4 * 1 + 0] = temp1[0 * 6 + 0] * kernels_L3_5[0]
			+ temp1[0 * 6 + 1] * kernels_L3_5[1]
			+ temp1[0 * 6 + 2] * kernels_L3_5[2]
			+ temp1[0 * 6 + 3] * kernels_L3_5[3]
			+ temp1[0 * 6 + 4] * kernels_L3_5[4];

		c[4 * 1 + 1] = temp1[0 * 6 + 1] * kernels_L3_5[0]
			+ temp1[0 * 6 + 2] * kernels_L3_5[1]
			+ temp1[0 * 6 + 3] * kernels_L3_5[2]
			+ temp1[0 * 6 + 4] * kernels_L3_5[3]
			+ temp1[0 * 6 + 5] * kernels_L3_5[4];

		c[4 * 1 + 2] = temp1[1 * 6 + 0] * kernels_L3_5[0]
			+ temp1[1 * 6 + 1] * kernels_L3_5[1]
			+ temp1[1 * 6 + 2] * kernels_L3_5[2]
			+ temp1[1 * 6 + 3] * kernels_L3_5[3]
			+ temp1[1 * 6 + 4] * kernels_L3_5[4];

		c[4 * 1 + 3] = temp1[1 * 6 + 1] * kernels_L3_5[0]
			+ temp1[1 * 6 + 2] * kernels_L3_5[1]
			+ temp1[1 * 6 + 3] * kernels_L3_5[2]
			+ temp1[1 * 6 + 4] * kernels_L3_5[3]
			+ temp1[1 * 6 + 5] * kernels_L3_5[4];
	}
	{
		c[4 * 2 + 0] = temp2[0 * 6 + 0] * kernels_L3_6[0]
			+ temp2[0 * 6 + 1] * kernels_L3_6[1]
			+ temp2[0 * 6 + 2] * kernels_L3_6[2]
			+ temp2[0 * 6 + 3] * kernels_L3_6[3]
			+ temp2[0 * 6 + 4] * kernels_L3_6[4];

		c[4 * 2 + 1] = temp2[0 * 6 + 1] * kernels_L3_6[0]
			+ temp2[0 * 6 + 2] * kernels_L3_6[1]
			+ temp2[0 * 6 + 3] * kernels_L3_6[2]
			+ temp2[0 * 6 + 4] * kernels_L3_6[3]
			+ temp2[0 * 6 + 5] * kernels_L3_6[4];

		c[4 * 2 + 2] = temp2[1 * 6 + 0] * kernels_L3_6[0]
			+ temp2[1 * 6 + 1] * kernels_L3_6[1]
			+ temp2[1 * 6 + 2] * kernels_L3_6[2]
			+ temp2[1 * 6 + 3] * kernels_L3_6[3]
			+ temp2[1 * 6 + 4] * kernels_L3_6[4];

		c[4 * 2 + 3] = temp2[1 * 6 + 1] * kernels_L3_6[0]
			+ temp2[1 * 6 + 2] * kernels_L3_6[1]
			+ temp2[1 * 6 + 3] * kernels_L3_6[2]
			+ temp2[1 * 6 + 4] * kernels_L3_6[3]
			+ temp2[1 * 6 + 5] * kernels_L3_6[4];
	}
	{
		c[4 * 3 + 0] = temp2[0 * 6 + 0] * kernels_L3_7[0]
			+ temp2[0 * 6 + 1] * kernels_L3_7[1]
			+ temp2[0 * 6 + 2] * kernels_L3_7[2]
			+ temp2[0 * 6 + 3] * kernels_L3_7[3]
			+ temp2[0 * 6 + 4] * kernels_L3_7[4];

		c[4 * 3 + 1] = temp2[0 * 6 + 1] * kernels_L3_7[0]
			+ temp2[0 * 6 + 2] * kernels_L3_7[1]
			+ temp2[0 * 6 + 3] * kernels_L3_7[2]
			+ temp2[0 * 6 + 4] * kernels_L3_7[3]
			+ temp2[0 * 6 + 5] * kernels_L3_7[4];

		c[4 * 3 + 2] = temp2[1 * 6 + 0] * kernels_L3_7[0]
			+ temp2[1 * 6 + 1] * kernels_L3_7[1]
			+ temp2[1 * 6 + 2] * kernels_L3_7[2]
			+ temp2[1 * 6 + 3] * kernels_L3_7[3]
			+ temp2[1 * 6 + 4] * kernels_L3_7[4];

		c[4 * 3 + 3] = temp2[1 * 6 + 1] * kernels_L3_7[0]
			+ temp2[1 * 6 + 2] * kernels_L3_7[1]
			+ temp2[1 * 6 + 3] * kernels_L3_7[2]
			+ temp2[1 * 6 + 4] * kernels_L3_7[3]
			+ temp2[1 * 6 + 5] * kernels_L3_7[4];
	}

	src_L3_3 += src_step;
	src_L3_4 += src_step;
	temp1[0 * 6 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[0 * 6 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[0 * 6 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[0 * 6 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[0 * 6 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);
	temp1[0 * 6 + 5] = *(src_L3_3 + 5) + *(src_L3_4 + 5);

	src_L3_5 += src_step;
	temp2[0 * 6 + 0] = temp1[0 * 6 + 0] + *(src_L3_5 + 0);
	temp2[0 * 6 + 1] = temp1[0 * 6 + 1] + *(src_L3_5 + 1);
	temp2[0 * 6 + 2] = temp1[0 * 6 + 2] + *(src_L3_5 + 2);
	temp2[0 * 6 + 3] = temp1[0 * 6 + 3] + *(src_L3_5 + 3);
	temp2[0 * 6 + 4] = temp1[0 * 6 + 4] + *(src_L3_5 + 4);
	temp2[0 * 6 + 5] = temp1[0 * 6 + 5] + *(src_L3_5 + 5);

	src_L3_2 += src_step;
	temp1[0 * 6 + 0] += *(src_L3_2 + 0);
	temp1[0 * 6 + 1] += *(src_L3_2 + 1);
	temp1[0 * 6 + 2] += *(src_L3_2 + 2);
	temp1[0 * 6 + 3] += *(src_L3_2 + 3);
	temp1[0 * 6 + 4] += *(src_L3_2 + 4);
	temp1[0 * 6 + 5] += *(src_L3_2 + 5);

	{
		c[4 * 0 + 0] += temp1[1 * 6 + 0] * kernels_L3_4[5]
			+ temp1[1 * 6 + 1] * kernels_L3_4[6]
			+ temp1[1 * 6 + 2] * kernels_L3_4[7]
			+ temp1[1 * 6 + 3] * kernels_L3_4[8]
			+ temp1[1 * 6 + 4] * kernels_L3_4[9];

		c[4 * 0 + 1] += temp1[1 * 6 + 1] * kernels_L3_4[5]
			+ temp1[1 * 6 + 2] * kernels_L3_4[6]
			+ temp1[1 * 6 + 3] * kernels_L3_4[7]
			+ temp1[1 * 6 + 4] * kernels_L3_4[8]
			+ temp1[1 * 6 + 5] * kernels_L3_4[9];

		c[4 * 0 + 2] += temp1[0 * 6 + 0] * kernels_L3_4[5]
			+ temp1[0 * 6 + 1] * kernels_L3_4[6]
			+ temp1[0 * 6 + 2] * kernels_L3_4[7]
			+ temp1[0 * 6 + 3] * kernels_L3_4[8]
			+ temp1[0 * 6 + 4] * kernels_L3_4[9];

		c[4 * 0 + 3] += temp1[0 * 6 + 1] * kernels_L3_4[5]
			+ temp1[0 * 6 + 2] * kernels_L3_4[6]
			+ temp1[0 * 6 + 3] * kernels_L3_4[7]
			+ temp1[0 * 6 + 4] * kernels_L3_4[8]
			+ temp1[0 * 6 + 5] * kernels_L3_4[9];
	}
	{
		c[4 * 1 + 0] += temp1[1 * 6 + 0] * kernels_L3_5[5]
			+ temp1[1 * 6 + 1] * kernels_L3_5[6]
			+ temp1[1 * 6 + 2] * kernels_L3_5[7]
			+ temp1[1 * 6 + 3] * kernels_L3_5[8]
			+ temp1[1 * 6 + 4] * kernels_L3_5[9];

		c[4 * 1 + 1] += temp1[1 * 6 + 1] * kernels_L3_5[5]
			+ temp1[1 * 6 + 2] * kernels_L3_5[6]
			+ temp1[1 * 6 + 3] * kernels_L3_5[7]
			+ temp1[1 * 6 + 4] * kernels_L3_5[8]
			+ temp1[1 * 6 + 5] * kernels_L3_5[9];

		c[4 * 1 + 2] += temp1[0 * 6 + 0] * kernels_L3_5[5]
			+ temp1[0 * 6 + 1] * kernels_L3_5[6]
			+ temp1[0 * 6 + 2] * kernels_L3_5[7]
			+ temp1[0 * 6 + 3] * kernels_L3_5[8]
			+ temp1[0 * 6 + 4] * kernels_L3_5[9];

		c[4 * 1 + 3] += temp1[0 * 6 + 1] * kernels_L3_5[5]
			+ temp1[0 * 6 + 2] * kernels_L3_5[6]
			+ temp1[0 * 6 + 3] * kernels_L3_5[7]
			+ temp1[0 * 6 + 4] * kernels_L3_5[8]
			+ temp1[0 * 6 + 5] * kernels_L3_5[9];
	}
	{
		c[4 * 2 + 0] += temp2[1 * 6 + 0] * kernels_L3_6[5]
			+ temp2[1 * 6 + 1] * kernels_L3_6[6]
			+ temp2[1 * 6 + 2] * kernels_L3_6[7]
			+ temp2[1 * 6 + 3] * kernels_L3_6[8]
			+ temp2[1 * 6 + 4] * kernels_L3_6[9];

		c[4 * 2 + 1] += temp2[1 * 6 + 1] * kernels_L3_6[5]
			+ temp2[1 * 6 + 2] * kernels_L3_6[6]
			+ temp2[1 * 6 + 3] * kernels_L3_6[7]
			+ temp2[1 * 6 + 4] * kernels_L3_6[8]
			+ temp2[1 * 6 + 5] * kernels_L3_6[9];

		c[4 * 2 + 2] += temp2[0 * 6 + 0] * kernels_L3_6[5]
			+ temp2[0 * 6 + 1] * kernels_L3_6[6]
			+ temp2[0 * 6 + 2] * kernels_L3_6[7]
			+ temp2[0 * 6 + 3] * kernels_L3_6[8]
			+ temp2[0 * 6 + 4] * kernels_L3_6[9];

		c[4 * 2 + 3] += temp2[0 * 6 + 1] * kernels_L3_6[5]
			+ temp2[0 * 6 + 2] * kernels_L3_6[6]
			+ temp2[0 * 6 + 3] * kernels_L3_6[7]
			+ temp2[0 * 6 + 4] * kernels_L3_6[8]
			+ temp2[0 * 6 + 5] * kernels_L3_6[9];
	}
	{
		c[4 * 3 + 0] += temp2[1 * 6 + 0] * kernels_L3_7[5]
			+ temp2[1 * 6 + 1] * kernels_L3_7[6]
			+ temp2[1 * 6 + 2] * kernels_L3_7[7]
			+ temp2[1 * 6 + 3] * kernels_L3_7[8]
			+ temp2[1 * 6 + 4] * kernels_L3_7[9];

		c[4 * 3 + 1] += temp2[1 * 6 + 1] * kernels_L3_7[5]
			+ temp2[1 * 6 + 2] * kernels_L3_7[6]
			+ temp2[1 * 6 + 3] * kernels_L3_7[7]
			+ temp2[1 * 6 + 4] * kernels_L3_7[8]
			+ temp2[1 * 6 + 5] * kernels_L3_7[9];

		c[4 * 3 + 2] += temp2[0 * 6 + 0] * kernels_L3_7[5]
			+ temp2[0 * 6 + 1] * kernels_L3_7[6]
			+ temp2[0 * 6 + 2] * kernels_L3_7[7]
			+ temp2[0 * 6 + 3] * kernels_L3_7[8]
			+ temp2[0 * 6 + 4] * kernels_L3_7[9];

		c[4 * 3 + 3] += temp2[0 * 6 + 1] * kernels_L3_7[5]
			+ temp2[0 * 6 + 2] * kernels_L3_7[6]
			+ temp2[0 * 6 + 3] * kernels_L3_7[7]
			+ temp2[0 * 6 + 4] * kernels_L3_7[8]
			+ temp2[0 * 6 + 5] * kernels_L3_7[9];
	}

	src_L3_3 += src_step;
	src_L3_4 += src_step;
	temp1[1 * 6 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[1 * 6 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[1 * 6 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[1 * 6 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[1 * 6 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);
	temp1[1 * 6 + 5] = *(src_L3_3 + 5) + *(src_L3_4 + 5);

	src_L3_5 += src_step;
	temp2[1 * 6 + 0] = temp1[1 * 6 + 0] + *(src_L3_5 + 0);
	temp2[1 * 6 + 1] = temp1[1 * 6 + 1] + *(src_L3_5 + 1);
	temp2[1 * 6 + 2] = temp1[1 * 6 + 2] + *(src_L3_5 + 2);
	temp2[1 * 6 + 3] = temp1[1 * 6 + 3] + *(src_L3_5 + 3);
	temp2[1 * 6 + 4] = temp1[1 * 6 + 4] + *(src_L3_5 + 4);
	temp2[1 * 6 + 5] = temp1[1 * 6 + 5] + *(src_L3_5 + 5);

	src_L3_2 += src_step;
	temp1[1 * 6 + 0] += *(src_L3_2 + 0);
	temp1[1 * 6 + 1] += *(src_L3_2 + 1);
	temp1[1 * 6 + 2] += *(src_L3_2 + 2);
	temp1[1 * 6 + 3] += *(src_L3_2 + 3);
	temp1[1 * 6 + 4] += *(src_L3_2 + 4);
	temp1[1 * 6 + 5] += *(src_L3_2 + 5);

	{
		c[4 * 0 + 0] += temp1[0 * 6 + 0] * kernels_L3_4[10]
			+ temp1[0 * 6 + 1] * kernels_L3_4[11]
			+ temp1[0 * 6 + 2] * kernels_L3_4[12]
			+ temp1[0 * 6 + 3] * kernels_L3_4[13]
			+ temp1[0 * 6 + 4] * kernels_L3_4[14];

		c[4 * 0 + 1] += temp1[0 * 6 + 1] * kernels_L3_4[10]
			+ temp1[0 * 6 + 2] * kernels_L3_4[11]
			+ temp1[0 * 6 + 3] * kernels_L3_4[12]
			+ temp1[0 * 6 + 4] * kernels_L3_4[13]
			+ temp1[0 * 6 + 5] * kernels_L3_4[14];

		c[4 * 0 + 2] += temp1[1 * 6 + 0] * kernels_L3_4[10]
			+ temp1[1 * 6 + 1] * kernels_L3_4[11]
			+ temp1[1 * 6 + 2] * kernels_L3_4[12]
			+ temp1[1 * 6 + 3] * kernels_L3_4[13]
			+ temp1[1 * 6 + 4] * kernels_L3_4[14];

		c[4 * 0 + 3] += temp1[1 * 6 + 1] * kernels_L3_4[10]
			+ temp1[1 * 6 + 2] * kernels_L3_4[11]
			+ temp1[1 * 6 + 3] * kernels_L3_4[12]
			+ temp1[1 * 6 + 4] * kernels_L3_4[13]
			+ temp1[1 * 6 + 5] * kernels_L3_4[14];
	}
	{
		c[4 * 1 + 0] += temp1[0 * 6 + 0] * kernels_L3_5[10]
			+ temp1[0 * 6 + 1] * kernels_L3_5[11]
			+ temp1[0 * 6 + 2] * kernels_L3_5[12]
			+ temp1[0 * 6 + 3] * kernels_L3_5[13]
			+ temp1[0 * 6 + 4] * kernels_L3_5[14];

		c[4 * 1 + 1] += temp1[0 * 6 + 1] * kernels_L3_5[10]
			+ temp1[0 * 6 + 2] * kernels_L3_5[11]
			+ temp1[0 * 6 + 3] * kernels_L3_5[12]
			+ temp1[0 * 6 + 4] * kernels_L3_5[13]
			+ temp1[0 * 6 + 5] * kernels_L3_5[14];

		c[4 * 1 + 2] += temp1[1 * 6 + 0] * kernels_L3_5[10]
			+ temp1[1 * 6 + 1] * kernels_L3_5[11]
			+ temp1[1 * 6 + 2] * kernels_L3_5[12]
			+ temp1[1 * 6 + 3] * kernels_L3_5[13]
			+ temp1[1 * 6 + 4] * kernels_L3_5[14];

		c[4 * 1 + 3] += temp1[1 * 6 + 1] * kernels_L3_5[10]
			+ temp1[1 * 6 + 2] * kernels_L3_5[11]
			+ temp1[1 * 6 + 3] * kernels_L3_5[12]
			+ temp1[1 * 6 + 4] * kernels_L3_5[13]
			+ temp1[1 * 6 + 5] * kernels_L3_5[14];
	}
	{
		c[4 * 2 + 0] += temp2[0 * 6 + 0] * kernels_L3_6[10]
			+ temp2[0 * 6 + 1] * kernels_L3_6[11]
			+ temp2[0 * 6 + 2] * kernels_L3_6[12]
			+ temp2[0 * 6 + 3] * kernels_L3_6[13]
			+ temp2[0 * 6 + 4] * kernels_L3_6[14];

		c[4 * 2 + 1] += temp2[0 * 6 + 1] * kernels_L3_6[10]
			+ temp2[0 * 6 + 2] * kernels_L3_6[11]
			+ temp2[0 * 6 + 3] * kernels_L3_6[12]
			+ temp2[0 * 6 + 4] * kernels_L3_6[13]
			+ temp2[0 * 6 + 5] * kernels_L3_6[14];

		c[4 * 2 + 2] += temp2[1 * 6 + 0] * kernels_L3_6[10]
			+ temp2[1 * 6 + 1] * kernels_L3_6[11]
			+ temp2[1 * 6 + 2] * kernels_L3_6[12]
			+ temp2[1 * 6 + 3] * kernels_L3_6[13]
			+ temp2[1 * 6 + 4] * kernels_L3_6[14];

		c[4 * 2 + 3] += temp2[1 * 6 + 1] * kernels_L3_6[10]
			+ temp2[1 * 6 + 2] * kernels_L3_6[11]
			+ temp2[1 * 6 + 3] * kernels_L3_6[12]
			+ temp2[1 * 6 + 4] * kernels_L3_6[13]
			+ temp2[1 * 6 + 5] * kernels_L3_6[14];
	}
	{
		c[4 * 3 + 0] += temp2[0 * 6 + 0] * kernels_L3_7[10]
			+ temp2[0 * 6 + 1] * kernels_L3_7[11]
			+ temp2[0 * 6 + 2] * kernels_L3_7[12]
			+ temp2[0 * 6 + 3] * kernels_L3_7[13]
			+ temp2[0 * 6 + 4] * kernels_L3_7[14];

		c[4 * 3 + 1] += temp2[0 * 6 + 1] * kernels_L3_7[10]
			+ temp2[0 * 6 + 2] * kernels_L3_7[11]
			+ temp2[0 * 6 + 3] * kernels_L3_7[12]
			+ temp2[0 * 6 + 4] * kernels_L3_7[13]
			+ temp2[0 * 6 + 5] * kernels_L3_7[14];

		c[4 * 3 + 2] += temp2[1 * 6 + 0] * kernels_L3_7[10]
			+ temp2[1 * 6 + 1] * kernels_L3_7[11]
			+ temp2[1 * 6 + 2] * kernels_L3_7[12]
			+ temp2[1 * 6 + 3] * kernels_L3_7[13]
			+ temp2[1 * 6 + 4] * kernels_L3_7[14];

		c[4 * 3 + 3] += temp2[1 * 6 + 1] * kernels_L3_7[10]
			+ temp2[1 * 6 + 2] * kernels_L3_7[11]
			+ temp2[1 * 6 + 3] * kernels_L3_7[12]
			+ temp2[1 * 6 + 4] * kernels_L3_7[13]
			+ temp2[1 * 6 + 5] * kernels_L3_7[14];
	}

	src_L3_3 += src_step;
	src_L3_4 += src_step;
	temp1[0 * 6 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[0 * 6 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[0 * 6 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[0 * 6 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[0 * 6 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);
	temp1[0 * 6 + 5] = *(src_L3_3 + 5) + *(src_L3_4 + 5);

	src_L3_5 += src_step;
	temp2[0 * 6 + 0] = temp1[0 * 6 + 0] + *(src_L3_5 + 0);
	temp2[0 * 6 + 1] = temp1[0 * 6 + 1] + *(src_L3_5 + 1);
	temp2[0 * 6 + 2] = temp1[0 * 6 + 2] + *(src_L3_5 + 2);
	temp2[0 * 6 + 3] = temp1[0 * 6 + 3] + *(src_L3_5 + 3);
	temp2[0 * 6 + 4] = temp1[0 * 6 + 4] + *(src_L3_5 + 4);
	temp2[0 * 6 + 5] = temp1[0 * 6 + 5] + *(src_L3_5 + 5);

	src_L3_2 += src_step;
	temp1[0 * 6 + 0] += *(src_L3_2 + 0);
	temp1[0 * 6 + 1] += *(src_L3_2 + 1);
	temp1[0 * 6 + 2] += *(src_L3_2 + 2);
	temp1[0 * 6 + 3] += *(src_L3_2 + 3);
	temp1[0 * 6 + 4] += *(src_L3_2 + 4);
	temp1[0 * 6 + 5] += *(src_L3_2 + 5);

	{
		c[4 * 0 + 0] += temp1[1 * 6 + 0] * kernels_L3_4[15]
			+ temp1[1 * 6 + 1] * kernels_L3_4[16]
			+ temp1[1 * 6 + 2] * kernels_L3_4[17]
			+ temp1[1 * 6 + 3] * kernels_L3_4[18]
			+ temp1[1 * 6 + 4] * kernels_L3_4[19];

		c[4 * 0 + 1] += temp1[1 * 6 + 1] * kernels_L3_4[15]
			+ temp1[1 * 6 + 2] * kernels_L3_4[16]
			+ temp1[1 * 6 + 3] * kernels_L3_4[17]
			+ temp1[1 * 6 + 4] * kernels_L3_4[18]
			+ temp1[1 * 6 + 5] * kernels_L3_4[19];

		c[4 * 0 + 2] += temp1[0 * 6 + 0] * kernels_L3_4[15]
			+ temp1[0 * 6 + 1] * kernels_L3_4[16]
			+ temp1[0 * 6 + 2] * kernels_L3_4[17]
			+ temp1[0 * 6 + 3] * kernels_L3_4[18]
			+ temp1[0 * 6 + 4] * kernels_L3_4[19];

		c[4 * 0 + 3] += temp1[0 * 6 + 1] * kernels_L3_4[15]
			+ temp1[0 * 6 + 2] * kernels_L3_4[16]
			+ temp1[0 * 6 + 3] * kernels_L3_4[17]
			+ temp1[0 * 6 + 4] * kernels_L3_4[18]
			+ temp1[0 * 6 + 5] * kernels_L3_4[19];
	}
	{
		c[4 * 1 + 0] += temp1[1 * 6 + 0] * kernels_L3_5[15]
			+ temp1[1 * 6 + 1] * kernels_L3_5[16]
			+ temp1[1 * 6 + 2] * kernels_L3_5[17]
			+ temp1[1 * 6 + 3] * kernels_L3_5[18]
			+ temp1[1 * 6 + 4] * kernels_L3_5[19];

		c[4 * 1 + 1] += temp1[1 * 6 + 1] * kernels_L3_5[15]
			+ temp1[1 * 6 + 2] * kernels_L3_5[16]
			+ temp1[1 * 6 + 3] * kernels_L3_5[17]
			+ temp1[1 * 6 + 4] * kernels_L3_5[18]
			+ temp1[1 * 6 + 5] * kernels_L3_5[19];

		c[4 * 1 + 2] += temp1[0 * 6 + 0] * kernels_L3_5[15]
			+ temp1[0 * 6 + 1] * kernels_L3_5[16]
			+ temp1[0 * 6 + 2] * kernels_L3_5[17]
			+ temp1[0 * 6 + 3] * kernels_L3_5[18]
			+ temp1[0 * 6 + 4] * kernels_L3_5[19];

		c[4 * 1 + 3] += temp1[0 * 6 + 1] * kernels_L3_5[15]
			+ temp1[0 * 6 + 2] * kernels_L3_5[16]
			+ temp1[0 * 6 + 3] * kernels_L3_5[17]
			+ temp1[0 * 6 + 4] * kernels_L3_5[18]
			+ temp1[0 * 6 + 5] * kernels_L3_5[19];
	}
	{
		c[4 * 2 + 0] += temp2[1 * 6 + 0] * kernels_L3_6[15]
			+ temp2[1 * 6 + 1] * kernels_L3_6[16]
			+ temp2[1 * 6 + 2] * kernels_L3_6[17]
			+ temp2[1 * 6 + 3] * kernels_L3_6[18]
			+ temp2[1 * 6 + 4] * kernels_L3_6[19];

		c[4 * 2 + 1] += temp2[1 * 6 + 1] * kernels_L3_6[15]
			+ temp2[1 * 6 + 2] * kernels_L3_6[16]
			+ temp2[1 * 6 + 3] * kernels_L3_6[17]
			+ temp2[1 * 6 + 4] * kernels_L3_6[18]
			+ temp2[1 * 6 + 5] * kernels_L3_6[19];

		c[4 * 2 + 2] += temp2[0 * 6 + 0] * kernels_L3_6[15]
			+ temp2[0 * 6 + 1] * kernels_L3_6[16]
			+ temp2[0 * 6 + 2] * kernels_L3_6[17]
			+ temp2[0 * 6 + 3] * kernels_L3_6[18]
			+ temp2[0 * 6 + 4] * kernels_L3_6[19];

		c[4 * 2 + 3] += temp2[0 * 6 + 1] * kernels_L3_6[15]
			+ temp2[0 * 6 + 2] * kernels_L3_6[16]
			+ temp2[0 * 6 + 3] * kernels_L3_6[17]
			+ temp2[0 * 6 + 4] * kernels_L3_6[18]
			+ temp2[0 * 6 + 5] * kernels_L3_6[19];
	}
	{
		c[4 * 3 + 0] += temp2[1 * 6 + 0] * kernels_L3_7[15]
			+ temp2[1 * 6 + 1] * kernels_L3_7[16]
			+ temp2[1 * 6 + 2] * kernels_L3_7[17]
			+ temp2[1 * 6 + 3] * kernels_L3_7[18]
			+ temp2[1 * 6 + 4] * kernels_L3_7[19];

		c[4 * 3 + 1] += temp2[1 * 6 + 1] * kernels_L3_7[15]
			+ temp2[1 * 6 + 2] * kernels_L3_7[16]
			+ temp2[1 * 6 + 3] * kernels_L3_7[17]
			+ temp2[1 * 6 + 4] * kernels_L3_7[18]
			+ temp2[1 * 6 + 5] * kernels_L3_7[19];

		c[4 * 3 + 2] += temp2[0 * 6 + 0] * kernels_L3_7[15]
			+ temp2[0 * 6 + 1] * kernels_L3_7[16]
			+ temp2[0 * 6 + 2] * kernels_L3_7[17]
			+ temp2[0 * 6 + 3] * kernels_L3_7[18]
			+ temp2[0 * 6 + 4] * kernels_L3_7[19];

		c[4 * 3 + 3] += temp2[0 * 6 + 1] * kernels_L3_7[15]
			+ temp2[0 * 6 + 2] * kernels_L3_7[16]
			+ temp2[0 * 6 + 3] * kernels_L3_7[17]
			+ temp2[0 * 6 + 4] * kernels_L3_7[18]
			+ temp2[0 * 6 + 5] * kernels_L3_7[19];
	}

	src_L3_3 += src_step;
	src_L3_4 += src_step;
	temp1[1 * 6 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[1 * 6 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[1 * 6 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[1 * 6 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[1 * 6 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);
	temp1[1 * 6 + 5] = *(src_L3_3 + 5) + *(src_L3_4 + 5);

	src_L3_5 += src_step;
	temp2[1 * 6 + 0] = temp1[1 * 6 + 0] + *(src_L3_5 + 0);
	temp2[1 * 6 + 1] = temp1[1 * 6 + 1] + *(src_L3_5 + 1);
	temp2[1 * 6 + 2] = temp1[1 * 6 + 2] + *(src_L3_5 + 2);
	temp2[1 * 6 + 3] = temp1[1 * 6 + 3] + *(src_L3_5 + 3);
	temp2[1 * 6 + 4] = temp1[1 * 6 + 4] + *(src_L3_5 + 4);
	temp2[1 * 6 + 5] = temp1[1 * 6 + 5] + *(src_L3_5 + 5);

	src_L3_2 += src_step;
	temp1[1 * 6 + 0] += *(src_L3_2 + 0);
	temp1[1 * 6 + 1] += *(src_L3_2 + 1);
	temp1[1 * 6 + 2] += *(src_L3_2 + 2);
	temp1[1 * 6 + 3] += *(src_L3_2 + 3);
	temp1[1 * 6 + 4] += *(src_L3_2 + 4);
	temp1[1 * 6 + 5] += *(src_L3_2 + 5);

	{
		c[4 * 0 + 0] += temp1[0 * 6 + 0] * kernels_L3_4[20]
			+ temp1[0 * 6 + 1] * kernels_L3_4[21]
			+ temp1[0 * 6 + 2] * kernels_L3_4[22]
			+ temp1[0 * 6 + 3] * kernels_L3_4[23]
			+ temp1[0 * 6 + 4] * kernels_L3_4[24];

		c[4 * 0 + 1] += temp1[0 * 6 + 1] * kernels_L3_4[20]
			+ temp1[0 * 6 + 2] * kernels_L3_4[21]
			+ temp1[0 * 6 + 3] * kernels_L3_4[22]
			+ temp1[0 * 6 + 4] * kernels_L3_4[23]
			+ temp1[0 * 6 + 5] * kernels_L3_4[24];

		c[4 * 0 + 2] += temp1[1 * 6 + 0] * kernels_L3_4[20]
			+ temp1[1 * 6 + 1] * kernels_L3_4[21]
			+ temp1[1 * 6 + 2] * kernels_L3_4[22]
			+ temp1[1 * 6 + 3] * kernels_L3_4[23]
			+ temp1[1 * 6 + 4] * kernels_L3_4[24];

		c[4 * 0 + 3] += temp1[1 * 6 + 1] * kernels_L3_4[20]
			+ temp1[1 * 6 + 2] * kernels_L3_4[21]
			+ temp1[1 * 6 + 3] * kernels_L3_4[22]
			+ temp1[1 * 6 + 4] * kernels_L3_4[23]
			+ temp1[1 * 6 + 5] * kernels_L3_4[24];
	}
	{
		c[4 * 1 + 0] += temp1[0 * 6 + 0] * kernels_L3_5[20]
			+ temp1[0 * 6 + 1] * kernels_L3_5[21]
			+ temp1[0 * 6 + 2] * kernels_L3_5[22]
			+ temp1[0 * 6 + 3] * kernels_L3_5[23]
			+ temp1[0 * 6 + 4] * kernels_L3_5[24];

		c[4 * 1 + 1] += temp1[0 * 6 + 1] * kernels_L3_5[20]
			+ temp1[0 * 6 + 2] * kernels_L3_5[21]
			+ temp1[0 * 6 + 3] * kernels_L3_5[22]
			+ temp1[0 * 6 + 4] * kernels_L3_5[23]
			+ temp1[0 * 6 + 5] * kernels_L3_5[24];

		c[4 * 1 + 2] += temp1[1 * 6 + 0] * kernels_L3_5[20]
			+ temp1[1 * 6 + 1] * kernels_L3_5[21]
			+ temp1[1 * 6 + 2] * kernels_L3_5[22]
			+ temp1[1 * 6 + 3] * kernels_L3_5[23]
			+ temp1[1 * 6 + 4] * kernels_L3_5[24];

		c[4 * 1 + 3] += temp1[1 * 6 + 1] * kernels_L3_5[20]
			+ temp1[1 * 6 + 2] * kernels_L3_5[21]
			+ temp1[1 * 6 + 3] * kernels_L3_5[22]
			+ temp1[1 * 6 + 4] * kernels_L3_5[23]
			+ temp1[1 * 6 + 5] * kernels_L3_5[24];
	}
	{
		c[4 * 2 + 0] += temp2[0 * 6 + 0] * kernels_L3_6[20]
			+ temp2[0 * 6 + 1] * kernels_L3_6[21]
			+ temp2[0 * 6 + 2] * kernels_L3_6[22]
			+ temp2[0 * 6 + 3] * kernels_L3_6[23]
			+ temp2[0 * 6 + 4] * kernels_L3_6[24];

		c[4 * 2 + 1] += temp2[0 * 6 + 1] * kernels_L3_6[20]
			+ temp2[0 * 6 + 2] * kernels_L3_6[21]
			+ temp2[0 * 6 + 3] * kernels_L3_6[22]
			+ temp2[0 * 6 + 4] * kernels_L3_6[23]
			+ temp2[0 * 6 + 5] * kernels_L3_6[24];

		c[4 * 2 + 2] += temp2[1 * 6 + 0] * kernels_L3_6[20]
			+ temp2[1 * 6 + 1] * kernels_L3_6[21]
			+ temp2[1 * 6 + 2] * kernels_L3_6[22]
			+ temp2[1 * 6 + 3] * kernels_L3_6[23]
			+ temp2[1 * 6 + 4] * kernels_L3_6[24];

		c[4 * 2 + 3] += temp2[1 * 6 + 1] * kernels_L3_6[20]
			+ temp2[1 * 6 + 2] * kernels_L3_6[21]
			+ temp2[1 * 6 + 3] * kernels_L3_6[22]
			+ temp2[1 * 6 + 4] * kernels_L3_6[23]
			+ temp2[1 * 6 + 5] * kernels_L3_6[24];
	}
	{
		c[4 * 3 + 0] += temp2[0 * 6 + 0] * kernels_L3_7[20]
			+ temp2[0 * 6 + 1] * kernels_L3_7[21]
			+ temp2[0 * 6 + 2] * kernels_L3_7[22]
			+ temp2[0 * 6 + 3] * kernels_L3_7[23]
			+ temp2[0 * 6 + 4] * kernels_L3_7[24];

		c[4 * 3 + 1] += temp2[0 * 6 + 1] * kernels_L3_7[20]
			+ temp2[0 * 6 + 2] * kernels_L3_7[21]
			+ temp2[0 * 6 + 3] * kernels_L3_7[22]
			+ temp2[0 * 6 + 4] * kernels_L3_7[23]
			+ temp2[0 * 6 + 5] * kernels_L3_7[24];

		c[4 * 3 + 2] += temp2[1 * 6 + 0] * kernels_L3_7[20]
			+ temp2[1 * 6 + 1] * kernels_L3_7[21]
			+ temp2[1 * 6 + 2] * kernels_L3_7[22]
			+ temp2[1 * 6 + 3] * kernels_L3_7[23]
			+ temp2[1 * 6 + 4] * kernels_L3_7[24];

		c[4 * 3 + 3] += temp2[1 * 6 + 1] * kernels_L3_7[20]
			+ temp2[1 * 6 + 2] * kernels_L3_7[21]
			+ temp2[1 * 6 + 3] * kernels_L3_7[22]
			+ temp2[1 * 6 + 4] * kernels_L3_7[23]
			+ temp2[1 * 6 + 5] * kernels_L3_7[24];
	}

	src_L3_3 += src_step;
	src_L3_4 += src_step;
	temp1[0 * 6 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[0 * 6 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[0 * 6 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[0 * 6 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[0 * 6 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);
	temp1[0 * 6 + 5] = *(src_L3_3 + 5) + *(src_L3_4 + 5);

	src_L3_5 += src_step;
	temp2[0 * 6 + 0] = temp1[0 * 6 + 0] + *(src_L3_5 + 0);
	temp2[0 * 6 + 1] = temp1[0 * 6 + 1] + *(src_L3_5 + 1);
	temp2[0 * 6 + 2] = temp1[0 * 6 + 2] + *(src_L3_5 + 2);
	temp2[0 * 6 + 3] = temp1[0 * 6 + 3] + *(src_L3_5 + 3);
	temp2[0 * 6 + 4] = temp1[0 * 6 + 4] + *(src_L3_5 + 4);
	temp2[0 * 6 + 5] = temp1[0 * 6 + 5] + *(src_L3_5 + 5);

	src_L3_2 += src_step;
	temp1[0 * 6 + 0] += *(src_L3_2 + 0);
	temp1[0 * 6 + 1] += *(src_L3_2 + 1);
	temp1[0 * 6 + 2] += *(src_L3_2 + 2);
	temp1[0 * 6 + 3] += *(src_L3_2 + 3);
	temp1[0 * 6 + 4] += *(src_L3_2 + 4);
	temp1[0 * 6 + 5] += *(src_L3_2 + 5);

	{
		c[4 * 0 + 0] += temp1[1 * 6 + 0] * kernels_L3_4[25]
			+ temp1[1 * 6 + 1] * kernels_L3_4[26]
			+ temp1[1 * 6 + 2] * kernels_L3_4[27]
			+ temp1[1 * 6 + 3] * kernels_L3_4[28]
			+ temp1[1 * 6 + 4] * kernels_L3_4[29];

		c[4 * 0 + 1] += temp1[1 * 6 + 1] * kernels_L3_4[25]
			+ temp1[1 * 6 + 2] * kernels_L3_4[26]
			+ temp1[1 * 6 + 3] * kernels_L3_4[27]
			+ temp1[1 * 6 + 4] * kernels_L3_4[28]
			+ temp1[1 * 6 + 5] * kernels_L3_4[29];

		c[4 * 0 + 2] += temp1[0 * 6 + 0] * kernels_L3_4[25]
			+ temp1[0 * 6 + 1] * kernels_L3_4[26]
			+ temp1[0 * 6 + 2] * kernels_L3_4[27]
			+ temp1[0 * 6 + 3] * kernels_L3_4[28]
			+ temp1[0 * 6 + 4] * kernels_L3_4[29];

		c[4 * 0 + 3] += temp1[0 * 6 + 1] * kernels_L3_4[25]
			+ temp1[0 * 6 + 2] * kernels_L3_4[26]
			+ temp1[0 * 6 + 3] * kernels_L3_4[27]
			+ temp1[0 * 6 + 4] * kernels_L3_4[28]
			+ temp1[0 * 6 + 5] * kernels_L3_4[29];

		dst_4 += dst_offset;
		*(dst_4 + 0) = c[4 * 0 + 0];
		*(dst_4 + 1) = c[4 * 0 + 1];
		dst_4 += dst_step;
		*(dst_4 + 0) = c[4 * 0 + 2];
		*(dst_4 + 1) = c[4 * 0 + 3];
	}
	{
		c[4 * 1 + 0] += temp1[1 * 6 + 0] * kernels_L3_5[25]
			+ temp1[1 * 6 + 1] * kernels_L3_5[26]
			+ temp1[1 * 6 + 2] * kernels_L3_5[27]
			+ temp1[1 * 6 + 3] * kernels_L3_5[28]
			+ temp1[1 * 6 + 4] * kernels_L3_5[29];

		c[4 * 1 + 1] += temp1[1 * 6 + 1] * kernels_L3_5[25]
			+ temp1[1 * 6 + 2] * kernels_L3_5[26]
			+ temp1[1 * 6 + 3] * kernels_L3_5[27]
			+ temp1[1 * 6 + 4] * kernels_L3_5[28]
			+ temp1[1 * 6 + 5] * kernels_L3_5[29];

		c[4 * 1 + 2] += temp1[0 * 6 + 0] * kernels_L3_5[25]
			+ temp1[0 * 6 + 1] * kernels_L3_5[26]
			+ temp1[0 * 6 + 2] * kernels_L3_5[27]
			+ temp1[0 * 6 + 3] * kernels_L3_5[28]
			+ temp1[0 * 6 + 4] * kernels_L3_5[29];

		c[4 * 1 + 3] += temp1[0 * 6 + 1] * kernels_L3_5[25]
			+ temp1[0 * 6 + 2] * kernels_L3_5[26]
			+ temp1[0 * 6 + 3] * kernels_L3_5[27]
			+ temp1[0 * 6 + 4] * kernels_L3_5[28]
			+ temp1[0 * 6 + 5] * kernels_L3_5[29];

		dst_5 += dst_offset;
		*(dst_5 + 0) = c[4 * 1 + 0];
		*(dst_5 + 1) = c[4 * 1 + 1];
		dst_5 += dst_step;
		*(dst_5 + 0) = c[4 * 1 + 2];
		*(dst_5 + 1) = c[4 * 1 + 3];
	}
	{
		c[4 * 2 + 0] += temp2[1 * 6 + 0] * kernels_L3_6[25]
			+ temp2[1 * 6 + 1] * kernels_L3_6[26]
			+ temp2[1 * 6 + 2] * kernels_L3_6[27]
			+ temp2[1 * 6 + 3] * kernels_L3_6[28]
			+ temp2[1 * 6 + 4] * kernels_L3_6[29];

		c[4 * 2 + 1] += temp2[1 * 6 + 1] * kernels_L3_6[25]
			+ temp2[1 * 6 + 2] * kernels_L3_6[26]
			+ temp2[1 * 6 + 3] * kernels_L3_6[27]
			+ temp2[1 * 6 + 4] * kernels_L3_6[28]
			+ temp2[1 * 6 + 5] * kernels_L3_6[29];

		c[4 * 2 + 2] += temp2[0 * 6 + 0] * kernels_L3_6[25]
			+ temp2[0 * 6 + 1] * kernels_L3_6[26]
			+ temp2[0 * 6 + 2] * kernels_L3_6[27]
			+ temp2[0 * 6 + 3] * kernels_L3_6[28]
			+ temp2[0 * 6 + 4] * kernels_L3_6[29];

		c[4 * 2 + 3] += temp2[0 * 6 + 1] * kernels_L3_6[25]
			+ temp2[0 * 6 + 2] * kernels_L3_6[26]
			+ temp2[0 * 6 + 3] * kernels_L3_6[27]
			+ temp2[0 * 6 + 4] * kernels_L3_6[28]
			+ temp2[0 * 6 + 5] * kernels_L3_6[29];

		dst_6 += dst_offset;
		*(dst_6 + 0) = c[4 * 2 + 0];
		*(dst_6 + 1) = c[4 * 2 + 1];
		dst_6 += dst_step;
		*(dst_6 + 0) = c[4 * 2 + 2];
		*(dst_6 + 1) = c[4 * 2 + 3];
	}
	{
		c[4 * 3 + 0] += temp2[1 * 6 + 0] * kernels_L3_7[25]
			+ temp2[1 * 6 + 1] * kernels_L3_7[26]
			+ temp2[1 * 6 + 2] * kernels_L3_7[27]
			+ temp2[1 * 6 + 3] * kernels_L3_7[28]
			+ temp2[1 * 6 + 4] * kernels_L3_7[29];

		c[4 * 3 + 1] += temp2[1 * 6 + 1] * kernels_L3_7[25]
			+ temp2[1 * 6 + 2] * kernels_L3_7[26]
			+ temp2[1 * 6 + 3] * kernels_L3_7[27]
			+ temp2[1 * 6 + 4] * kernels_L3_7[28]
			+ temp2[1 * 6 + 5] * kernels_L3_7[29];

		c[4 * 3 + 2] += temp2[0 * 6 + 0] * kernels_L3_7[25]
			+ temp2[0 * 6 + 1] * kernels_L3_7[26]
			+ temp2[0 * 6 + 2] * kernels_L3_7[27]
			+ temp2[0 * 6 + 3] * kernels_L3_7[28]
			+ temp2[0 * 6 + 4] * kernels_L3_7[29];

		c[4 * 3 + 3] += temp2[0 * 6 + 1] * kernels_L3_7[25]
			+ temp2[0 * 6 + 2] * kernels_L3_7[26]
			+ temp2[0 * 6 + 3] * kernels_L3_7[27]
			+ temp2[0 * 6 + 4] * kernels_L3_7[28]
			+ temp2[0 * 6 + 5] * kernels_L3_7[29];

		dst_7 += dst_offset;
		*(dst_7 + 0) = c[4 * 3 + 0];
		*(dst_7 + 1) = c[4 * 3 + 1];
		dst_7 += dst_step;
		*(dst_7 + 0) = c[4 * 3 + 2];
		*(dst_7 + 1) = c[4 * 3 + 3];
	}
}

__kernel void add_16_tanh_tanh_2tanh_tanh_texmem_cl(
													/*0*/__global float* dst,
													/*1*/int dst_cols,
													/*2*/int dst_rows,
													/*3*/int dst_step,
													/*4*/__global float* src_HL_1,
													/*5*/__global float* src_HL_2,
													/*6*/__global float* src_HL_3,
													/*7*/__global float* src_HL_4,
													/*8*/__global float* src_HL_5,
													/*9*/__global float* src_HL_6,
													/*10*/__global float* src_HL_7,
													/*11*/__global float* src_HL_8,
													/*12*/__global float* src_HL_9,
													/*13*/__global float* src_HL_10,
													/*14*/__global float* src_HL_11,
													/*15*/__global float* src_HL_12,
													/*16*/__global float* src_HL_13,
													/*17*/__global float* src_HL_14,
													/*18*/__global float* src_HL_15,
													/*19*/__global float* src_HL_16,
													/*20*/int src_cols,
													/*21*/int src_rows,
													/*22*/int src_step,
													/*23*/__constant float* conv_b_L3,
													/*24*/__constant float* subs_w_L3,
													/*25*/__constant float* subs_b_L3,
													/*26*/__constant float* scale_L3,
													/*27*/__constant float* hl_w,
													/*28*/__constant float* hl_b,
													/*29*/__constant float* ol_w,
													/*30*/__constant float* ol_b,
													/*31*/__constant float* scale_HL
													)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);

	if (i >= dst_cols || j >= dst_rows) return;

	const int src_offset = j * src_step + i;
	const int dst_offset = j * dst_step + i;

	float res = ol_b[0];
	{
		float c = *(src_HL_1 + src_offset);

		c += conv_b_L3[0];
		c = tanhf(c);

		c = c * subs_w_L3[0] + subs_b_L3[0];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[0] + hl_b[0]);
		const float d2 = tanhf(c * hl_w[0 + 1] + hl_b[0 + 1]);

		c = d1 * ol_w[0] + d2 * ol_w[0 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_2 + src_offset);

		c += conv_b_L3[1];
		c = tanhf(c);

		c = c * subs_w_L3[1] + subs_b_L3[1];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[2] + hl_b[2]);
		const float d2 = tanhf(c * hl_w[2 + 1] + hl_b[2 + 1]);

		c = d1 * ol_w[2] + d2 * ol_w[2 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_3 + src_offset);

		c += conv_b_L3[2];
		c = tanhf(c);

		c = c * subs_w_L3[2] + subs_b_L3[2];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[4] + hl_b[4]);
		const float d2 = tanhf(c * hl_w[4 + 1] + hl_b[4 + 1]);

		c = d1 * ol_w[4] + d2 * ol_w[4 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_4 + src_offset);

		c += conv_b_L3[3];
		c = tanhf(c);

		c = c * subs_w_L3[3] + subs_b_L3[3];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[6] + hl_b[6]);
		const float d2 = tanhf(c * hl_w[6 + 1] + hl_b[6 + 1]);

		c = d1 * ol_w[6] + d2 * ol_w[6 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_5 + src_offset);

		c += conv_b_L3[4];
		c = tanhf(c);
		
		c = c * subs_w_L3[4] + subs_b_L3[4];
		c = tanhf(c);
		
		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[8] + hl_b[8]);
		const float d2 = tanhf(c * hl_w[8 + 1] + hl_b[8 + 1]);

		c = d1 * ol_w[8] + d2 * ol_w[8 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_6 + src_offset);

		c += conv_b_L3[5];
		c = tanhf(c);

		c = c * subs_w_L3[5] + subs_b_L3[5];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[10] + hl_b[10]);
		const float d2 = tanhf(c * hl_w[10 + 1] + hl_b[10 + 1]);

		c = d1 * ol_w[10] + d2 * ol_w[10 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_7 + src_offset);

		c += conv_b_L3[6];
		c = tanhf(c);

		c = c * subs_w_L3[6] + subs_b_L3[6];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[12] + hl_b[12]);
		const float d2 = tanhf(c * hl_w[12 + 1] + hl_b[12 + 1]);

		c = d1 * ol_w[12] + d2 * ol_w[12 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_8 + src_offset);

		c += conv_b_L3[7];
		c = tanhf(c);

		c = c * subs_w_L3[7] + subs_b_L3[7];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[14] + hl_b[14]);
		const float d2 = tanhf(c * hl_w[14 + 1] + hl_b[14 + 1]);

		c = d1 * ol_w[14] + d2 * ol_w[14 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_9 + src_offset);

		c += conv_b_L3[8];
		c = tanhf(c);

		c = c * subs_w_L3[8] + subs_b_L3[8];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[16] + hl_b[16]);
		const float d2 = tanhf(c * hl_w[16 + 1] + hl_b[16 + 1]);

		c = d1 * ol_w[16] + d2 * ol_w[16 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_10 + src_offset);

		c += conv_b_L3[9];
		c = tanhf(c);

		c = c * subs_w_L3[9] + subs_b_L3[9];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[18] + hl_b[18]);
		const float d2 = tanhf(c * hl_w[18 + 1] + hl_b[18 + 1]);

		c = d1 * ol_w[18] + d2 * ol_w[18 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_11 + src_offset);

		c += conv_b_L3[10];
		c = tanhf(c);

		c = c * subs_w_L3[10] + subs_b_L3[10];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[20] + hl_b[20]);
		const float d2 = tanhf(c * hl_w[20 + 1] + hl_b[20 + 1]);

		c = d1 * ol_w[20] + d2 * ol_w[20 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_12 + src_offset);

		c += conv_b_L3[11];
		c = tanhf(c);

		c = c * subs_w_L3[11] + subs_b_L3[11];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[22] + hl_b[22]);
		const float d2 = tanhf(c * hl_w[22 + 1] + hl_b[22 + 1]);

		c = d1 * ol_w[22] + d2 * ol_w[22 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_13 + src_offset);

		c += conv_b_L3[12];
		c = tanhf(c);

		c = c * subs_w_L3[12] + subs_b_L3[12];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[24] + hl_b[24]);
		const float d2 = tanhf(c * hl_w[24 + 1] + hl_b[24 + 1]);

		c = d1 * ol_w[24] + d2 * ol_w[24 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_14 + src_offset);

		c += conv_b_L3[13];
		c = tanhf(c);

		c = c * subs_w_L3[13] + subs_b_L3[13];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[26] + hl_b[26]);
		const float d2 = tanhf(c * hl_w[26 + 1] + hl_b[26 + 1]);

		c = d1 * ol_w[26] + d2 * ol_w[26 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_15 + src_offset);

		c += conv_b_L3[14];
		c = tanhf(c);

		c = c * subs_w_L3[14] + subs_b_L3[14];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[28] + hl_b[28]);
		const float d2 = tanhf(c * hl_w[28 + 1] + hl_b[28 + 1]);

		c = d1 * ol_w[28] + d2 * ol_w[28 + 1];
		res += c * scale_HL[0];
	}
	{
		float c = *(src_HL_16 + src_offset);

		c += conv_b_L3[15];
		c = tanhf(c);

		c = c * subs_w_L3[15] + subs_b_L3[15];
		c = tanhf(c);

		c *= scale_L3[0];
		const float d1 = tanhf(c * hl_w[30] + hl_b[30]);
		const float d2 = tanhf(c * hl_w[30 + 1] + hl_b[30 + 1]);

		c = d1 * ol_w[30] + d2 * ol_w[30 + 1];
		res += c * scale_HL[0];
	}

	res = tanhf(res);

	*(dst + dst_offset) = scale_HL[0] * res;
}