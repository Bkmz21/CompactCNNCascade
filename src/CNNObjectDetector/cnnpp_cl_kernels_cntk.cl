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

#define block_4x4_4x4(y, x, t, p, l, k)									\
	c[4 * y + x] += temp[t * 5 + 0 + p] * kernels_L1_##l[0 + 4 * k]		\
		+ temp[t * 5 + 1 + p] * kernels_L1_##l[1 + 4 * k]				\
		+ temp[t * 5 + 2 + p] * kernels_L1_##l[2 + 4 * k]				\
		+ temp[t * 5 + 3 + p] * kernels_L1_##l[3 + 4 * k];

#define block_4x4_3x3(y, x, t, p, l, k, u)								\
	c[4 * y + x] += temp##u[t * 4 + 0 + p] * kernels_L2_##l[0 + 3 * k]	\
		+ temp##u[t * 4 + 1 + p] * kernels_L2_##l[1 + 3 * k]			\
		+ temp##u[t * 4 + 2 + p] * kernels_L2_##l[2 + 3 * k];

#define block_4x4_4x5(y, x, t, p, l, k, u)								\
	c[4 * y + x] += temp##u[t * 5 + 0 + p] * kernels_L3_##l[0 + 4 * k]	\
		+ temp##u[t * 5 + 1 + p] * kernels_L3_##l[1 + 4 * k]			\
		+ temp##u[t * 5 + 2 + p] * kernels_L3_##l[2 + 4 * k]			\
		+ temp##u[t * 5 + 3 + p] * kernels_L3_##l[3 + 4 * k];

#define b_relu_bn_max(y, L, idx)																							\
	c[4 * y + 0] += conv_b_L##L[idx];																						\
	const float res_0 = lrelu_w1_L##L[idx] * c[4 * y + 0] + lrelu_w2_L##L[idx] * fmax(0.f, c[4 * y + 0]) + bn_b_L##L[idx];	\
	c[4 * y + 1] += conv_b_L##L[idx];																						\
	const float res_1 = lrelu_w1_L##L[idx] * c[4 * y + 1] + lrelu_w2_L##L[idx] * fmax(0.f, c[4 * y + 1]) + bn_b_L##L[idx];	\
	c[4 * y + 2] += conv_b_L##L[idx];																						\
	const float res_2 = lrelu_w1_L##L[idx] * c[4 * y + 2] + lrelu_w2_L##L[idx] * fmax(0.f, c[4 * y + 2]) + bn_b_L##L[idx];	\
	c[4 * y + 3] += conv_b_L##L[idx];																						\
	const float res_3 = lrelu_w1_L##L[idx] * c[4 * y + 3] + lrelu_w2_L##L[idx] * fmax(0.f, c[4 * y + 3]) + bn_b_L##L[idx];	\
	*(dst_##idx + dst_offset) = fmax(fmax(res_0, res_1), fmax(res_2, res_3));

#define b_relu_bn(y, L, idx)																								\
	c[4 * y + 0] += conv_b_L##L[idx];																						\
	const float res_0 = lrelu_w1_L##L[idx] * c[4 * y + 0] + lrelu_w2_L##L[idx] * fmax(0.f, c[4 * y + 0]) + bn_b_L##L[idx];	\
	c[4 * y + 1] += conv_b_L##L[idx];																						\
	const float res_1 = lrelu_w1_L##L[idx] * c[4 * y + 1] + lrelu_w2_L##L[idx] * fmax(0.f, c[4 * y + 1]) + bn_b_L##L[idx];	\
	c[4 * y + 2] += conv_b_L##L[idx];																						\
	const float res_2 = lrelu_w1_L##L[idx] * c[4 * y + 2] + lrelu_w2_L##L[idx] * fmax(0.f, c[4 * y + 2]) + bn_b_L##L[idx];	\
	c[4 * y + 3] += conv_b_L##L[idx];																						\
	const float res_3 = lrelu_w1_L##L[idx] * c[4 * y + 3] + lrelu_w2_L##L[idx] * fmax(0.f, c[4 * y + 3]) + bn_b_L##L[idx];

#define hl_layer(p, idx0, idx1, idx2, idx3)														\
{																								\
	const float c0 = *(src_HL_##idx0 + src_offset);												\
	const float c1 = *(src_HL_##idx1 + src_offset);												\
	const float c2 = *(src_HL_##idx2 + src_offset);												\
	const float c3 = *(src_HL_##idx3 + src_offset);												\
	for (int t = 0; t < surf_hl_scale; ++t)														\
	{																							\
		float s = c0 * hl_w[(p * surf_hl_scale + t) * surf_hl_connect + 0]						\
			+ c1 * hl_w[(p * surf_hl_scale + t) * surf_hl_connect + 1]							\
			+ c2 * hl_w[(p * surf_hl_scale + t) * surf_hl_connect + 2]							\
			+ c3 * hl_w[(p * surf_hl_scale + t) * surf_hl_connect + 3]							\
			+ hl_b[p * surf_hl_scale + t];														\
		s *= hl_tanh_w[p];																		\
		s = tanhf(s);																			\
		s = 0.5f * s + 0.5f;																	\
		s = hl_bn_w[t] * s + hl_bn_b[t];														\
		res += s * ol_w[index_output * surf_hl_size + p + t * (surf_hl_size / surf_hl_scale)];	\
	}																							\
}																								


__kernel void conv_4x4x4_lrelu_bn_max_cl(
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
										/*14*/__constant float* lrelu_w1_L1,
										/*15*/__constant float* lrelu_w2_L1,
										/*16*/__constant float* bn_b_L1
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
	for (int i = 0; i < 16; i += 4)
	{
		c[i + 0] = 0.f;
		c[i + 1] = 0.f;
		c[i + 2] = 0.f;
		c[i + 3] = 0.f;
	}

	{
		block_4x4_4x4(0, 0, 0, 0, 0, 0)
		block_4x4_4x4(0, 1, 0, 1, 0, 0)
		block_4x4_4x4(0, 2, 1, 0, 0, 0)
		block_4x4_4x4(0, 3, 1, 1, 0, 0)

		//c[4 * 0 + 0] = temp[0 * 5 + 0] * kernels_L1_0[0]
		//	+ temp[0 * 5 + 1] * kernels_L1_0[1]
		//	+ temp[0 * 5 + 2] * kernels_L1_0[2]
		//	+ temp[0 * 5 + 3] * kernels_L1_0[3];

		//c[4 * 0 + 1] = temp[0 * 5 + 1] * kernels_L1_0[0]
		//	+ temp[0 * 5 + 2] * kernels_L1_0[1]
		//	+ temp[0 * 5 + 3] * kernels_L1_0[2]
		//	+ temp[0 * 5 + 4] * kernels_L1_0[3];

		//c[4 * 0 + 2] = temp[1 * 5 + 0] * kernels_L1_0[0]
		//	+ temp[1 * 5 + 1] * kernels_L1_0[1]
		//	+ temp[1 * 5 + 2] * kernels_L1_0[2]
		//	+ temp[1 * 5 + 3] * kernels_L1_0[3];

		//c[4 * 0 + 3] = temp[1 * 5 + 1] * kernels_L1_0[0]
		//	+ temp[1 * 5 + 2] * kernels_L1_0[1]
		//	+ temp[1 * 5 + 3] * kernels_L1_0[2]
		//	+ temp[1 * 5 + 4] * kernels_L1_0[3];
	}
	{
		block_4x4_4x4(1, 0, 0, 0, 1, 0)
		block_4x4_4x4(1, 1, 0, 1, 1, 0)
		block_4x4_4x4(1, 2, 1, 0, 1, 0)
		block_4x4_4x4(1, 3, 1, 1, 1, 0)

		//c[4 * 1 + 0] = temp[0 * 5 + 0] * kernels_L1_1[0]
		//	+ temp[0 * 5 + 1] * kernels_L1_1[1]
		//	+ temp[0 * 5 + 2] * kernels_L1_1[2]
		//	+ temp[0 * 5 + 3] * kernels_L1_1[3];

		//c[4 * 1 + 1] = temp[0 * 5 + 1] * kernels_L1_1[0]
		//	+ temp[0 * 5 + 2] * kernels_L1_1[1]
		//	+ temp[0 * 5 + 3] * kernels_L1_1[2]
		//	+ temp[0 * 5 + 4] * kernels_L1_1[3];

		//c[4 * 1 + 2] = temp[1 * 5 + 0] * kernels_L1_1[0]
		//	+ temp[1 * 5 + 1] * kernels_L1_1[1]
		//	+ temp[1 * 5 + 2] * kernels_L1_1[2]
		//	+ temp[1 * 5 + 3] * kernels_L1_1[3];

		//c[4 * 1 + 3] = temp[1 * 5 + 1] * kernels_L1_1[0]
		//	+ temp[1 * 5 + 2] * kernels_L1_1[1]
		//	+ temp[1 * 5 + 3] * kernels_L1_1[2]
		//	+ temp[1 * 5 + 4] * kernels_L1_1[3];
	}
	{
		block_4x4_4x4(2, 0, 0, 0, 2, 0)
		block_4x4_4x4(2, 1, 0, 1, 2, 0)
		block_4x4_4x4(2, 2, 1, 0, 2, 0)
		block_4x4_4x4(2, 3, 1, 1, 2, 0)

		//c[4 * 2 + 0] = temp[0 * 5 + 0] * kernels_L1_2[0]
		//	+ temp[0 * 5 + 1] * kernels_L1_2[1]
		//	+ temp[0 * 5 + 2] * kernels_L1_2[2]
		//	+ temp[0 * 5 + 3] * kernels_L1_2[3];

		//c[4 * 2 + 1] = temp[0 * 5 + 1] * kernels_L1_2[0]
		//	+ temp[0 * 5 + 2] * kernels_L1_2[1]
		//	+ temp[0 * 5 + 3] * kernels_L1_2[2]
		//	+ temp[0 * 5 + 4] * kernels_L1_2[3];

		//c[4 * 2 + 2] = temp[1 * 5 + 0] * kernels_L1_2[0]
		//	+ temp[1 * 5 + 1] * kernels_L1_2[1]
		//	+ temp[1 * 5 + 2] * kernels_L1_2[2]
		//	+ temp[1 * 5 + 3] * kernels_L1_2[3];

		//c[4 * 2 + 3] = temp[1 * 5 + 1] * kernels_L1_2[0]
		//	+ temp[1 * 5 + 2] * kernels_L1_2[1]
		//	+ temp[1 * 5 + 3] * kernels_L1_2[2]
		//	+ temp[1 * 5 + 4] * kernels_L1_2[3];
	}
	{
		block_4x4_4x4(3, 0, 0, 0, 3, 0)
		block_4x4_4x4(3, 1, 0, 1, 3, 0)
		block_4x4_4x4(3, 2, 1, 0, 3, 0)
		block_4x4_4x4(3, 3, 1, 1, 3, 0)

		//c[4 * 3 + 0] = temp[0 * 5 + 0] * kernels_L1_3[0]
		//	+ temp[0 * 5 + 1] * kernels_L1_3[1]
		//	+ temp[0 * 5 + 2] * kernels_L1_3[2]
		//	+ temp[0 * 5 + 3] * kernels_L1_3[3];

		//c[4 * 3 + 1] = temp[0 * 5 + 1] * kernels_L1_3[0]
		//	+ temp[0 * 5 + 2] * kernels_L1_3[1]
		//	+ temp[0 * 5 + 3] * kernels_L1_3[2]
		//	+ temp[0 * 5 + 4] * kernels_L1_3[3];

		//c[4 * 3 + 2] = temp[1 * 5 + 0] * kernels_L1_3[0]
		//	+ temp[1 * 5 + 1] * kernels_L1_3[1]
		//	+ temp[1 * 5 + 2] * kernels_L1_3[2]
		//	+ temp[1 * 5 + 3] * kernels_L1_3[3];

		//c[4 * 3 + 3] = temp[1 * 5 + 1] * kernels_L1_3[0]
		//	+ temp[1 * 5 + 2] * kernels_L1_3[1]
		//	+ temp[1 * 5 + 3] * kernels_L1_3[2]
		//	+ temp[1 * 5 + 4] * kernels_L1_3[3];
	}

	src_L1 += src_step;
	temp[0 * 5 + 0] = *(src_L1 + 0);
	temp[0 * 5 + 1] = *(src_L1 + 1);
	temp[0 * 5 + 2] = *(src_L1 + 2);
	temp[0 * 5 + 3] = *(src_L1 + 3);
	temp[0 * 5 + 4] = *(src_L1 + 4);

	{
		block_4x4_4x4(0, 0, 1, 0, 0, 1)
		block_4x4_4x4(0, 1, 1, 1, 0, 1)
		block_4x4_4x4(0, 2, 0, 0, 0, 1)
		block_4x4_4x4(0, 3, 0, 1, 0, 1)

		//c[4 * 0 + 0] += temp[1 * 5 + 0] * kernels_L1_0[4]
		//	+ temp[1 * 5 + 1] * kernels_L1_0[5]
		//	+ temp[1 * 5 + 2] * kernels_L1_0[6]
		//	+ temp[1 * 5 + 3] * kernels_L1_0[7];

		//c[4 * 0 + 1] += temp[1 * 5 + 1] * kernels_L1_0[4]
		//	+ temp[1 * 5 + 2] * kernels_L1_0[5]
		//	+ temp[1 * 5 + 3] * kernels_L1_0[6]
		//	+ temp[1 * 5 + 4] * kernels_L1_0[7];

		//c[4 * 0 + 2] += temp[0 * 5 + 0] * kernels_L1_0[4]
		//	+ temp[0 * 5 + 1] * kernels_L1_0[5]
		//	+ temp[0 * 5 + 2] * kernels_L1_0[6]
		//	+ temp[0 * 5 + 3] * kernels_L1_0[7];

		//c[4 * 0 + 3] += temp[0 * 5 + 1] * kernels_L1_0[4]
		//	+ temp[0 * 5 + 2] * kernels_L1_0[5]
		//	+ temp[0 * 5 + 3] * kernels_L1_0[6]
		//	+ temp[0 * 5 + 4] * kernels_L1_0[7];
	}
	{
		block_4x4_4x4(1, 0, 1, 0, 1, 1)
		block_4x4_4x4(1, 1, 1, 1, 1, 1)
		block_4x4_4x4(1, 2, 0, 0, 1, 1)
		block_4x4_4x4(1, 3, 0, 1, 1, 1)

		//c[4 * 1 + 0] += temp[1 * 5 + 0] * kernels_L1_1[4]
		//	+ temp[1 * 5 + 1] * kernels_L1_1[5]
		//	+ temp[1 * 5 + 2] * kernels_L1_1[6]
		//	+ temp[1 * 5 + 3] * kernels_L1_1[7];

		//c[4 * 1 + 1] += temp[1 * 5 + 1] * kernels_L1_1[4]
		//	+ temp[1 * 5 + 2] * kernels_L1_1[5]
		//	+ temp[1 * 5 + 3] * kernels_L1_1[6]
		//	+ temp[1 * 5 + 4] * kernels_L1_1[7];

		//c[4 * 1 + 2] += temp[0 * 5 + 0] * kernels_L1_1[4]
		//	+ temp[0 * 5 + 1] * kernels_L1_1[5]
		//	+ temp[0 * 5 + 2] * kernels_L1_1[6]
		//	+ temp[0 * 5 + 3] * kernels_L1_1[7];

		//c[4 * 1 + 3] += temp[0 * 5 + 1] * kernels_L1_1[4]
		//	+ temp[0 * 5 + 2] * kernels_L1_1[5]
		//	+ temp[0 * 5 + 3] * kernels_L1_1[6]
		//	+ temp[0 * 5 + 4] * kernels_L1_1[7];
	}
	{
		block_4x4_4x4(2, 0, 1, 0, 2, 1)
		block_4x4_4x4(2, 1, 1, 1, 2, 1)
		block_4x4_4x4(2, 2, 0, 0, 2, 1)
		block_4x4_4x4(2, 3, 0, 1, 2, 1)

		//c[4 * 2 + 0] += temp[1 * 5 + 0] * kernels_L1_2[4]
		//	+ temp[1 * 5 + 1] * kernels_L1_2[5]
		//	+ temp[1 * 5 + 2] * kernels_L1_2[6]
		//	+ temp[1 * 5 + 3] * kernels_L1_2[7];

		//c[4 * 2 + 1] += temp[1 * 5 + 1] * kernels_L1_2[4]
		//	+ temp[1 * 5 + 2] * kernels_L1_2[5]
		//	+ temp[1 * 5 + 3] * kernels_L1_2[6]
		//	+ temp[1 * 5 + 4] * kernels_L1_2[7];

		//c[4 * 2 + 2] += temp[0 * 5 + 0] * kernels_L1_2[4]
		//	+ temp[0 * 5 + 1] * kernels_L1_2[5]
		//	+ temp[0 * 5 + 2] * kernels_L1_2[6]
		//	+ temp[0 * 5 + 3] * kernels_L1_2[7];

		//c[4 * 2 + 3] += temp[0 * 5 + 1] * kernels_L1_2[4]
		//	+ temp[0 * 5 + 2] * kernels_L1_2[5]
		//	+ temp[0 * 5 + 3] * kernels_L1_2[6]
		//	+ temp[0 * 5 + 4] * kernels_L1_2[7];
	}
	{
		block_4x4_4x4(3, 0, 1, 0, 3, 1)
		block_4x4_4x4(3, 1, 1, 1, 3, 1)
		block_4x4_4x4(3, 2, 0, 0, 3, 1)
		block_4x4_4x4(3, 3, 0, 1, 3, 1)

		//c[4 * 3 + 0] += temp[1 * 5 + 0] * kernels_L1_3[4]
		//	+ temp[1 * 5 + 1] * kernels_L1_3[5]
		//	+ temp[1 * 5 + 2] * kernels_L1_3[6]
		//	+ temp[1 * 5 + 3] * kernels_L1_3[7];

		//c[4 * 3 + 1] += temp[1 * 5 + 1] * kernels_L1_3[4]
		//	+ temp[1 * 5 + 2] * kernels_L1_3[5]
		//	+ temp[1 * 5 + 3] * kernels_L1_3[6]
		//	+ temp[1 * 5 + 4] * kernels_L1_3[7];

		//c[4 * 3 + 2] += temp[0 * 5 + 0] * kernels_L1_3[4]
		//	+ temp[0 * 5 + 1] * kernels_L1_3[5]
		//	+ temp[0 * 5 + 2] * kernels_L1_3[6]
		//	+ temp[0 * 5 + 3] * kernels_L1_3[7];

		//c[4 * 3 + 3] += temp[0 * 5 + 1] * kernels_L1_3[4]
		//	+ temp[0 * 5 + 2] * kernels_L1_3[5]
		//	+ temp[0 * 5 + 3] * kernels_L1_3[6]
		//	+ temp[0 * 5 + 4] * kernels_L1_3[7];
	}

	src_L1 += src_step;
	temp[1 * 5 + 0] = *(src_L1 + 0);
	temp[1 * 5 + 1] = *(src_L1 + 1);
	temp[1 * 5 + 2] = *(src_L1 + 2);
	temp[1 * 5 + 3] = *(src_L1 + 3);
	temp[1 * 5 + 4] = *(src_L1 + 4);

	{
		block_4x4_4x4(0, 0, 0, 0, 0, 2)
		block_4x4_4x4(0, 1, 0, 1, 0, 2)
		block_4x4_4x4(0, 2, 1, 0, 0, 2)
		block_4x4_4x4(0, 3, 1, 1, 0, 2)

		//c[4 * 0 + 0] += temp[0 * 5 + 0] * kernels_L1_0[8]
		//	+ temp[0 * 5 + 1] * kernels_L1_0[9]
		//	+ temp[0 * 5 + 2] * kernels_L1_0[10]
		//	+ temp[0 * 5 + 3] * kernels_L1_0[11];

		//c[4 * 0 + 1] += temp[0 * 5 + 1] * kernels_L1_0[8]
		//	+ temp[0 * 5 + 2] * kernels_L1_0[9]
		//	+ temp[0 * 5 + 3] * kernels_L1_0[10]
		//	+ temp[0 * 5 + 4] * kernels_L1_0[11];

		//c[4 * 0 + 2] += temp[1 * 5 + 0] * kernels_L1_0[8]
		//	+ temp[1 * 5 + 1] * kernels_L1_0[9]
		//	+ temp[1 * 5 + 2] * kernels_L1_0[10]
		//	+ temp[1 * 5 + 3] * kernels_L1_0[11];

		//c[4 * 0 + 3] += temp[1 * 5 + 1] * kernels_L1_0[8]
		//	+ temp[1 * 5 + 2] * kernels_L1_0[9]
		//	+ temp[1 * 5 + 3] * kernels_L1_0[10]
		//	+ temp[1 * 5 + 4] * kernels_L1_0[11];
	}
	{
		block_4x4_4x4(1, 0, 0, 0, 1, 2)
		block_4x4_4x4(1, 1, 0, 1, 1, 2)
		block_4x4_4x4(1, 2, 1, 0, 1, 2)
		block_4x4_4x4(1, 3, 1, 1, 1, 2)

		//c[4 * 1 + 0] += temp[0 * 5 + 0] * kernels_L1_1[8]
		//	+ temp[0 * 5 + 1] * kernels_L1_1[9]
		//	+ temp[0 * 5 + 2] * kernels_L1_1[10]
		//	+ temp[0 * 5 + 3] * kernels_L1_1[11];

		//c[4 * 1 + 1] += temp[0 * 5 + 1] * kernels_L1_1[8]
		//	+ temp[0 * 5 + 2] * kernels_L1_1[9]
		//	+ temp[0 * 5 + 3] * kernels_L1_1[10]
		//	+ temp[0 * 5 + 4] * kernels_L1_1[11];

		//c[4 * 1 + 2] += temp[1 * 5 + 0] * kernels_L1_1[8]
		//	+ temp[1 * 5 + 1] * kernels_L1_1[9]
		//	+ temp[1 * 5 + 2] * kernels_L1_1[10]
		//	+ temp[1 * 5 + 3] * kernels_L1_1[11];

		//c[4 * 1 + 3] += temp[1 * 5 + 1] * kernels_L1_1[8]
		//	+ temp[1 * 5 + 2] * kernels_L1_1[9]
		//	+ temp[1 * 5 + 3] * kernels_L1_1[10]
		//	+ temp[1 * 5 + 4] * kernels_L1_1[11];
	}
	{
		block_4x4_4x4(2, 0, 0, 0, 2, 2)
		block_4x4_4x4(2, 1, 0, 1, 2, 2)
		block_4x4_4x4(2, 2, 1, 0, 2, 2)
		block_4x4_4x4(2, 3, 1, 1, 2, 2)

		//c[4 * 2 + 0] += temp[0 * 5 + 0] * kernels_L1_2[8]
		//	+ temp[0 * 5 + 1] * kernels_L1_2[9]
		//	+ temp[0 * 5 + 2] * kernels_L1_2[10]
		//	+ temp[0 * 5 + 3] * kernels_L1_2[11];

		//c[4 * 2 + 1] += temp[0 * 5 + 1] * kernels_L1_2[8]
		//	+ temp[0 * 5 + 2] * kernels_L1_2[9]
		//	+ temp[0 * 5 + 3] * kernels_L1_2[10]
		//	+ temp[0 * 5 + 4] * kernels_L1_2[11];

		//c[4 * 2 + 2] += temp[1 * 5 + 0] * kernels_L1_2[8]
		//	+ temp[1 * 5 + 1] * kernels_L1_2[9]
		//	+ temp[1 * 5 + 2] * kernels_L1_2[10]
		//	+ temp[1 * 5 + 3] * kernels_L1_2[11];

		//c[4 * 2 + 3] += temp[1 * 5 + 1] * kernels_L1_2[8]
		//	+ temp[1 * 5 + 2] * kernels_L1_2[9]
		//	+ temp[1 * 5 + 3] * kernels_L1_2[10]
		//	+ temp[1 * 5 + 4] * kernels_L1_2[11];
	}
	{
		block_4x4_4x4(3, 0, 0, 0, 3, 2)
		block_4x4_4x4(3, 1, 0, 1, 3, 2)
		block_4x4_4x4(3, 2, 1, 0, 3, 2)
		block_4x4_4x4(3, 3, 1, 1, 3, 2)

		//c[4 * 3 + 0] += temp[0 * 5 + 0] * kernels_L1_3[8]
		//	+ temp[0 * 5 + 1] * kernels_L1_3[9]
		//	+ temp[0 * 5 + 2] * kernels_L1_3[10]
		//	+ temp[0 * 5 + 3] * kernels_L1_3[11];
		//
		//c[4 * 3 + 1] += temp[0 * 5 + 1] * kernels_L1_3[8]
		//	+ temp[0 * 5 + 2] * kernels_L1_3[9]
		//	+ temp[0 * 5 + 3] * kernels_L1_3[10]
		//	+ temp[0 * 5 + 4] * kernels_L1_3[11];

		//c[4 * 3 + 2] += temp[1 * 5 + 0] * kernels_L1_3[8]
		//	+ temp[1 * 5 + 1] * kernels_L1_3[9]
		//	+ temp[1 * 5 + 2] * kernels_L1_3[10]
		//	+ temp[1 * 5 + 3] * kernels_L1_3[11];

		//c[4 * 3 + 3] += temp[1 * 5 + 1] * kernels_L1_3[8]
		//	+ temp[1 * 5 + 2] * kernels_L1_3[9]
		//	+ temp[1 * 5 + 3] * kernels_L1_3[10]
		//	+ temp[1 * 5 + 4] * kernels_L1_3[11];
	}

	src_L1 += src_step;
	temp[0 * 5 + 0] = *(src_L1 + 0);
	temp[0 * 5 + 1] = *(src_L1 + 1);
	temp[0 * 5 + 2] = *(src_L1 + 2);
	temp[0 * 5 + 3] = *(src_L1 + 3);
	temp[0 * 5 + 4] = *(src_L1 + 4);

	{
		block_4x4_4x4(0, 0, 1, 0, 0, 3)
		block_4x4_4x4(0, 1, 1, 1, 0, 3)
		block_4x4_4x4(0, 2, 0, 0, 0, 3)
		block_4x4_4x4(0, 3, 0, 1, 0, 3)

		//c[4 * 0 + 0] += temp[1 * 5 + 0] * kernels_L1_0[12]
		//	+ temp[1 * 5 + 1] * kernels_L1_0[13]
		//	+ temp[1 * 5 + 2] * kernels_L1_0[14]
		//	+ temp[1 * 5 + 3] * kernels_L1_0[15];

		//c[4 * 0 + 1] += temp[1 * 5 + 1] * kernels_L1_0[12]
		//	+ temp[1 * 5 + 2] * kernels_L1_0[13]
		//	+ temp[1 * 5 + 3] * kernels_L1_0[14]
		//	+ temp[1 * 5 + 4] * kernels_L1_0[15];

		//c[4 * 0 + 2] += temp[0 * 5 + 0] * kernels_L1_0[12]
		//	+ temp[0 * 5 + 1] * kernels_L1_0[13]
		//	+ temp[0 * 5 + 2] * kernels_L1_0[14]
		//	+ temp[0 * 5 + 3] * kernels_L1_0[15];

		//c[4 * 0 + 3] += temp[0 * 5 + 1] * kernels_L1_0[12]
		//	+ temp[0 * 5 + 2] * kernels_L1_0[13]
		//	+ temp[0 * 5 + 3] * kernels_L1_0[14]
		//	+ temp[0 * 5 + 4] * kernels_L1_0[15];

		b_relu_bn_max(0, 1, 0)
	}
	{
		block_4x4_4x4(1, 0, 1, 0, 1, 3)
		block_4x4_4x4(1, 1, 1, 1, 1, 3)
		block_4x4_4x4(1, 2, 0, 0, 1, 3)
		block_4x4_4x4(1, 3, 0, 1, 1, 3)

		//c[4 * 1 + 0] += temp[1 * 5 + 0] * kernels_L1_1[12]
		//	+ temp[1 * 5 + 1] * kernels_L1_1[13]
		//	+ temp[1 * 5 + 2] * kernels_L1_1[14]
		//	+ temp[1 * 5 + 3] * kernels_L1_1[15];

		//c[4 * 1 + 1] += temp[1 * 5 + 1] * kernels_L1_1[12]
		//	+ temp[1 * 5 + 2] * kernels_L1_1[13]
		//	+ temp[1 * 5 + 3] * kernels_L1_1[14]
		//	+ temp[1 * 5 + 4] * kernels_L1_1[15];

		//c[4 * 1 + 2] += temp[0 * 5 + 0] * kernels_L1_1[12]
		//	+ temp[0 * 5 + 1] * kernels_L1_1[13]
		//	+ temp[0 * 5 + 2] * kernels_L1_1[14]
		//	+ temp[0 * 5 + 3] * kernels_L1_1[15];

		//c[4 * 1 + 3] += temp[0 * 5 + 1] * kernels_L1_1[12]
		//	+ temp[0 * 5 + 2] * kernels_L1_1[13]
		//	+ temp[0 * 5 + 3] * kernels_L1_1[14]
		//	+ temp[0 * 5 + 4] * kernels_L1_1[15];

		b_relu_bn_max(1, 1, 1)
	}
	{
		block_4x4_4x4(2, 0, 1, 0, 2, 3)
		block_4x4_4x4(2, 1, 1, 1, 2, 3)
		block_4x4_4x4(2, 2, 0, 0, 2, 3)
		block_4x4_4x4(2, 3, 0, 1, 2, 3)

		//c[4 * 2 + 0] += temp[1 * 5 + 0] * kernels_L1_2[12]
		//	+ temp[1 * 5 + 1] * kernels_L1_2[13]
		//	+ temp[1 * 5 + 2] * kernels_L1_2[14]
		//	+ temp[1 * 5 + 3] * kernels_L1_2[15];

		//c[4 * 2 + 1] += temp[1 * 5 + 1] * kernels_L1_2[12]
		//	+ temp[1 * 5 + 2] * kernels_L1_2[13]
		//	+ temp[1 * 5 + 3] * kernels_L1_2[14]
		//	+ temp[1 * 5 + 4] * kernels_L1_2[15];

		//c[4 * 2 + 2] += temp[0 * 5 + 0] * kernels_L1_2[12]
		//	+ temp[0 * 5 + 1] * kernels_L1_2[13]
		//	+ temp[0 * 5 + 2] * kernels_L1_2[14]
		//	+ temp[0 * 5 + 3] * kernels_L1_2[15];

		//c[4 * 2 + 3] += temp[0 * 5 + 1] * kernels_L1_2[12]
		//	+ temp[0 * 5 + 2] * kernels_L1_2[13]
		//	+ temp[0 * 5 + 3] * kernels_L1_2[14]
		//	+ temp[0 * 5 + 4] * kernels_L1_2[15];

		b_relu_bn_max(2, 1, 2)
	}
	{
		block_4x4_4x4(3, 0, 1, 0, 3, 3)
		block_4x4_4x4(3, 1, 1, 1, 3, 3)
		block_4x4_4x4(3, 2, 0, 0, 3, 3)
		block_4x4_4x4(3, 3, 0, 1, 3, 3)

		//c[4 * 3 + 0] += temp[1 * 5 + 0] * kernels_L1_3[12]
		//	+ temp[1 * 5 + 1] * kernels_L1_3[13]
		//	+ temp[1 * 5 + 2] * kernels_L1_3[14]
		//	+ temp[1 * 5 + 3] * kernels_L1_3[15];

		//c[4 * 3 + 1] += temp[1 * 5 + 1] * kernels_L1_3[12]
		//	+ temp[1 * 5 + 2] * kernels_L1_3[13]
		//	+ temp[1 * 5 + 3] * kernels_L1_3[14]
		//	+ temp[1 * 5 + 4] * kernels_L1_3[15];

		//c[4 * 3 + 2] += temp[0 * 5 + 0] * kernels_L1_3[12]
		//	+ temp[0 * 5 + 1] * kernels_L1_3[13]
		//	+ temp[0 * 5 + 2] * kernels_L1_3[14]
		//	+ temp[0 * 5 + 3] * kernels_L1_3[15];

		//c[4 * 3 + 3] += temp[0 * 5 + 1] * kernels_L1_3[12]
		//	+ temp[0 * 5 + 2] * kernels_L1_3[13]
		//	+ temp[0 * 5 + 3] * kernels_L1_3[14]
		//	+ temp[0 * 5 + 4] * kernels_L1_3[15];

		b_relu_bn_max(3, 1, 3)
	}
}

__kernel void add_2_3_conv_4x3x3_lrelu_bn_max_L_cl(
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
													/*16*/__constant float* lrelu_w1_L2,
													/*17*/__constant float* lrelu_w2_L2,
													/*18*/__constant float* bn_b_L2
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
	for (int i = 0; i < 16; i += 4)
	{
		c[i + 0] = 0.f;
		c[i + 1] = 0.f;
		c[i + 2] = 0.f;
		c[i + 3] = 0.f;
	}

	{
		block_4x4_3x3(0, 0, 0, 0, 0, 0, 1)
		block_4x4_3x3(0, 1, 0, 1, 0, 0, 1)
		block_4x4_3x3(0, 2, 1, 0, 0, 0, 1)
		block_4x4_3x3(0, 3, 1, 1, 0, 0, 1)

		//c[4 * 0 + 0] = temp1[0 * 4 + 0] * kernels_L2_0[0]
		//	+ temp1[0 * 4 + 1] * kernels_L2_0[1]
		//	+ temp1[0 * 4 + 2] * kernels_L2_0[2];

		//c[4 * 0 + 1] = temp1[0 * 4 + 1] * kernels_L2_0[0]
		//	+ temp1[0 * 4 + 2] * kernels_L2_0[1]
		//	+ temp1[0 * 4 + 3] * kernels_L2_0[2];

		//c[4 * 0 + 2] = temp1[1 * 4 + 0] * kernels_L2_0[0]
		//	+ temp1[1 * 4 + 1] * kernels_L2_0[1]
		//	+ temp1[1 * 4 + 2] * kernels_L2_0[2];

		//c[4 * 0 + 3] = temp1[1 * 4 + 1] * kernels_L2_0[0]
		//	+ temp1[1 * 4 + 2] * kernels_L2_0[1]
		//	+ temp1[1 * 4 + 3] * kernels_L2_0[2];
	}
	{
		block_4x4_3x3(1, 0, 0, 0, 1, 0, 1)
		block_4x4_3x3(1, 1, 0, 1, 1, 0, 1)
		block_4x4_3x3(1, 2, 1, 0, 1, 0, 1)
		block_4x4_3x3(1, 3, 1, 1, 1, 0, 1)

		//c[4 * 1 + 0] = temp1[0 * 4 + 0] * kernels_L2_1[0]
		//	+ temp1[0 * 4 + 1] * kernels_L2_1[1]
		//	+ temp1[0 * 4 + 2] * kernels_L2_1[2];

		//c[4 * 1 + 1] = temp1[0 * 4 + 1] * kernels_L2_1[0]
		//	+ temp1[0 * 4 + 2] * kernels_L2_1[1]
		//	+ temp1[0 * 4 + 3] * kernels_L2_1[2];

		//c[4 * 1 + 2] = temp1[1 * 4 + 0] * kernels_L2_1[0]
		//	+ temp1[1 * 4 + 1] * kernels_L2_1[1]
		//	+ temp1[1 * 4 + 2] * kernels_L2_1[2];

		//c[4 * 1 + 3] = temp1[1 * 4 + 1] * kernels_L2_1[0]
		//	+ temp1[1 * 4 + 2] * kernels_L2_1[1]
		//	+ temp1[1 * 4 + 3] * kernels_L2_1[2];
	}
	{
		block_4x4_3x3(2, 0, 0, 0, 2, 0, 2)
		block_4x4_3x3(2, 1, 0, 1, 2, 0, 2)
		block_4x4_3x3(2, 2, 1, 0, 2, 0, 2)
		block_4x4_3x3(2, 3, 1, 1, 2, 0, 2)
		
		//c[4 * 2 + 0] = temp2[0 * 4 + 0] * kernels_L2_2[0]
		//	+ temp2[0 * 4 + 1] * kernels_L2_2[1]
		//	+ temp2[0 * 4 + 2] * kernels_L2_2[2];

		//c[4 * 2 + 1] = temp2[0 * 4 + 1] * kernels_L2_2[0]
		//	+ temp2[0 * 4 + 2] * kernels_L2_2[1]
		//	+ temp2[0 * 4 + 3] * kernels_L2_2[2];

		//c[4 * 2 + 2] = temp2[1 * 4 + 0] * kernels_L2_2[0]
		//	+ temp2[1 * 4 + 1] * kernels_L2_2[1]
		//	+ temp2[1 * 4 + 2] * kernels_L2_2[2];

		//c[4 * 2 + 3] = temp2[1 * 4 + 1] * kernels_L2_2[0]
		//	+ temp2[1 * 4 + 2] * kernels_L2_2[1]
		//	+ temp2[1 * 4 + 3] * kernels_L2_2[2];
	}
	{
		block_4x4_3x3(3, 0, 0, 0, 3, 0, 2)
		block_4x4_3x3(3, 1, 0, 1, 3, 0, 2)
		block_4x4_3x3(3, 2, 1, 0, 3, 0, 2)
		block_4x4_3x3(3, 3, 1, 1, 3, 0, 2)

		//c[4 * 3 + 0] = temp2[0 * 4 + 0] * kernels_L2_3[0]
		//	+ temp2[0 * 4 + 1] * kernels_L2_3[1]
		//	+ temp2[0 * 4 + 2] * kernels_L2_3[2];

		//c[4 * 3 + 1] = temp2[0 * 4 + 1] * kernels_L2_3[0]
		//	+ temp2[0 * 4 + 2] * kernels_L2_3[1]
		//	+ temp2[0 * 4 + 3] * kernels_L2_3[2];

		//c[4 * 3 + 2] = temp2[1 * 4 + 0] * kernels_L2_3[0]
		//	+ temp2[1 * 4 + 1] * kernels_L2_3[1]
		//	+ temp2[1 * 4 + 2] * kernels_L2_3[2];

		//c[4 * 3 + 3] = temp2[1 * 4 + 1] * kernels_L2_3[0]
		//	+ temp2[1 * 4 + 2] * kernels_L2_3[1]
		//	+ temp2[1 * 4 + 3] * kernels_L2_3[2];

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
		block_4x4_3x3(0, 0, 1, 0, 0, 1, 1)
		block_4x4_3x3(0, 1, 1, 1, 0, 1, 1)
		block_4x4_3x3(0, 2, 0, 0, 0, 1, 1)
		block_4x4_3x3(0, 3, 0, 1, 0, 1, 1)

		//c[4 * 0 + 0] += temp1[1 * 4 + 0] * kernels_L2_0[3]
		//	+ temp1[1 * 4 + 1] * kernels_L2_0[4]
		//	+ temp1[1 * 4 + 2] * kernels_L2_0[5];

		//c[4 * 0 + 1] += temp1[1 * 4 + 1] * kernels_L2_0[3]
		//	+ temp1[1 * 4 + 2] * kernels_L2_0[4]
		//	+ temp1[1 * 4 + 3] * kernels_L2_0[5];

		//c[4 * 0 + 2] += temp1[0 * 4 + 0] * kernels_L2_0[3]
		//	+ temp1[0 * 4 + 1] * kernels_L2_0[4]
		//	+ temp1[0 * 4 + 2] * kernels_L2_0[5];

		//c[4 * 0 + 3] += temp1[0 * 4 + 1] * kernels_L2_0[3]
		//	+ temp1[0 * 4 + 2] * kernels_L2_0[4]
		//	+ temp1[0 * 4 + 3] * kernels_L2_0[5];
	}
	{
		block_4x4_3x3(1, 0, 1, 0, 1, 1, 1)
		block_4x4_3x3(1, 1, 1, 1, 1, 1, 1)
		block_4x4_3x3(1, 2, 0, 0, 1, 1, 1)
		block_4x4_3x3(1, 3, 0, 1, 1, 1, 1)

		//c[4 * 1 + 0] += temp1[1 * 4 + 0] * kernels_L2_1[3]
		//	+ temp1[1 * 4 + 1] * kernels_L2_1[4]
		//	+ temp1[1 * 4 + 2] * kernels_L2_1[5];

		//c[4 * 1 + 1] += temp1[1 * 4 + 1] * kernels_L2_1[3]
		//	+ temp1[1 * 4 + 2] * kernels_L2_1[4]
		//	+ temp1[1 * 4 + 3] * kernels_L2_1[5];

		//c[4 * 1 + 2] += temp1[0 * 4 + 0] * kernels_L2_1[3]
		//	+ temp1[0 * 4 + 1] * kernels_L2_1[4]
		//	+ temp1[0 * 4 + 2] * kernels_L2_1[5];

		//c[4 * 1 + 3] += temp1[0 * 4 + 1] * kernels_L2_1[3]
		//	+ temp1[0 * 4 + 2] * kernels_L2_1[4]
		//	+ temp1[0 * 4 + 3] * kernels_L2_1[5];
	}
	{
		block_4x4_3x3(2, 0, 1, 0, 2, 1, 2)
		block_4x4_3x3(2, 1, 1, 1, 2, 1, 2)
		block_4x4_3x3(2, 2, 0, 0, 2, 1, 2)
		block_4x4_3x3(2, 3, 0, 1, 2, 1, 2)

		//c[4 * 2 + 0] += temp2[1 * 4 + 0] * kernels_L2_2[3]
		//	+ temp2[1 * 4 + 1] * kernels_L2_2[4]
		//	+ temp2[1 * 4 + 2] * kernels_L2_2[5];

		//c[4 * 2 + 1] += temp2[1 * 4 + 1] * kernels_L2_2[3]
		//	+ temp2[1 * 4 + 2] * kernels_L2_2[4]
		//	+ temp2[1 * 4 + 3] * kernels_L2_2[5];

		//c[4 * 2 + 2] += temp2[0 * 4 + 0] * kernels_L2_2[3]
		//	+ temp2[0 * 4 + 1] * kernels_L2_2[4]
		//	+ temp2[0 * 4 + 2] * kernels_L2_2[5];

		//c[4 * 2 + 3] += temp2[0 * 4 + 1] * kernels_L2_2[3]
		//	+ temp2[0 * 4 + 2] * kernels_L2_2[4]
		//	+ temp2[0 * 4 + 3] * kernels_L2_2[5];
	}
	{
		block_4x4_3x3(3, 0, 1, 0, 3, 1, 2)
		block_4x4_3x3(3, 1, 1, 1, 3, 1, 2)
		block_4x4_3x3(3, 2, 0, 0, 3, 1, 2)
		block_4x4_3x3(3, 3, 0, 1, 3, 1, 2)

		//c[4 * 3 + 0] += temp2[1 * 4 + 0] * kernels_L2_3[3]
		//	+ temp2[1 * 4 + 1] * kernels_L2_3[4]
		//	+ temp2[1 * 4 + 2] * kernels_L2_3[5];

		//c[4 * 3 + 1] += temp2[1 * 4 + 1] * kernels_L2_3[3]
		//	+ temp2[1 * 4 + 2] * kernels_L2_3[4]
		//	+ temp2[1 * 4 + 3] * kernels_L2_3[5];

		//c[4 * 3 + 2] += temp2[0 * 4 + 0] * kernels_L2_3[3]
		//	+ temp2[0 * 4 + 1] * kernels_L2_3[4]
		//	+ temp2[0 * 4 + 2] * kernels_L2_3[5];

		//c[4 * 3 + 3] += temp2[0 * 4 + 1] * kernels_L2_3[3]
		//	+ temp2[0 * 4 + 2] * kernels_L2_3[4]
		//	+ temp2[0 * 4 + 3] * kernels_L2_3[5];
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
		block_4x4_3x3(0, 0, 0, 0, 0, 2, 1)
		block_4x4_3x3(0, 1, 0, 1, 0, 2, 1)
		block_4x4_3x3(0, 2, 1, 0, 0, 2, 1)
		block_4x4_3x3(0, 3, 1, 1, 0, 2, 1)

		//c[4 * 0 + 0] += temp1[0 * 4 + 0] * kernels_L2_0[6]
		//	+ temp1[0 * 4 + 1] * kernels_L2_0[7]
		//	+ temp1[0 * 4 + 2] * kernels_L2_0[8];

		//c[4 * 0 + 1] += temp1[0 * 4 + 1] * kernels_L2_0[6]
		//	+ temp1[0 * 4 + 2] * kernels_L2_0[7]
		//	+ temp1[0 * 4 + 3] * kernels_L2_0[8];

		//c[4 * 0 + 2] += temp1[1 * 4 + 0] * kernels_L2_0[6]
		//	+ temp1[1 * 4 + 1] * kernels_L2_0[7]
		//	+ temp1[1 * 4 + 2] * kernels_L2_0[8];

		//c[4 * 0 + 3] += temp1[1 * 4 + 1] * kernels_L2_0[6]
		//	+ temp1[1 * 4 + 2] * kernels_L2_0[7]
		//	+ temp1[1 * 4 + 3] * kernels_L2_0[8];

		b_relu_bn_max(0, 2, 0)
	}
	{
		block_4x4_3x3(1, 0, 0, 0, 1, 2, 1)
		block_4x4_3x3(1, 1, 0, 1, 1, 2, 1)
		block_4x4_3x3(1, 2, 1, 0, 1, 2, 1)
		block_4x4_3x3(1, 3, 1, 1, 1, 2, 1)

		//c[4 * 1 + 0] += temp1[0 * 4 + 0] * kernels_L2_1[6]
		//	+ temp1[0 * 4 + 1] * kernels_L2_1[7]
		//	+ temp1[0 * 4 + 2] * kernels_L2_1[8];

		//c[4 * 1 + 1] += temp1[0 * 4 + 1] * kernels_L2_1[6]
		//	+ temp1[0 * 4 + 2] * kernels_L2_1[7]
		//	+ temp1[0 * 4 + 3] * kernels_L2_1[8];

		//c[4 * 1 + 2] += temp1[1 * 4 + 0] * kernels_L2_1[6]
		//	+ temp1[1 * 4 + 1] * kernels_L2_1[7]
		//	+ temp1[1 * 4 + 2] * kernels_L2_1[8];

		//c[4 * 1 + 3] += temp1[1 * 4 + 1] * kernels_L2_1[6]
		//	+ temp1[1 * 4 + 2] * kernels_L2_1[7]
		//	+ temp1[1 * 4 + 3] * kernels_L2_1[8];

		b_relu_bn_max(1, 2, 1)
	}
	{
		block_4x4_3x3(2, 0, 0, 0, 2, 2, 2)
		block_4x4_3x3(2, 1, 0, 1, 2, 2, 2)
		block_4x4_3x3(2, 2, 1, 0, 2, 2, 2)
		block_4x4_3x3(2, 3, 1, 1, 2, 2, 2)

		//c[4 * 2 + 0] += temp2[0 * 4 + 0] * kernels_L2_2[6]
		//	+ temp2[0 * 4 + 1] * kernels_L2_2[7]
		//	+ temp2[0 * 4 + 2] * kernels_L2_2[8];

		//c[4 * 2 + 1] += temp2[0 * 4 + 1] * kernels_L2_2[6]
		//	+ temp2[0 * 4 + 2] * kernels_L2_2[7]
		//	+ temp2[0 * 4 + 3] * kernels_L2_2[8];

		//c[4 * 2 + 2] += temp2[1 * 4 + 0] * kernels_L2_2[6]
		//	+ temp2[1 * 4 + 1] * kernels_L2_2[7]
		//	+ temp2[1 * 4 + 2] * kernels_L2_2[8];

		//c[4 * 2 + 3] += temp2[1 * 4 + 1] * kernels_L2_2[6]
		//	+ temp2[1 * 4 + 2] * kernels_L2_2[7]
		//	+ temp2[1 * 4 + 3] * kernels_L2_2[8];

		b_relu_bn_max(2, 2, 2)
	}
	{
		block_4x4_3x3(3, 0, 0, 0, 3, 2, 2)
		block_4x4_3x3(3, 1, 0, 1, 3, 2, 2)
		block_4x4_3x3(3, 2, 1, 0, 3, 2, 2)
		block_4x4_3x3(3, 3, 1, 1, 3, 2, 2)

		//c[4 * 3 + 0] += temp2[0 * 4 + 0] * kernels_L2_3[6]
		//	+ temp2[0 * 4 + 1] * kernels_L2_3[7]
		//	+ temp2[0 * 4 + 2] * kernels_L2_3[8];

		//c[4 * 3 + 1] += temp2[0 * 4 + 1] * kernels_L2_3[6]
		//	+ temp2[0 * 4 + 2] * kernels_L2_3[7]
		//	+ temp2[0 * 4 + 3] * kernels_L2_3[8];

		//c[4 * 3 + 2] += temp2[1 * 4 + 0] * kernels_L2_3[6]
		//	+ temp2[1 * 4 + 1] * kernels_L2_3[7]
		//	+ temp2[1 * 4 + 2] * kernels_L2_3[8];

		//c[4 * 3 + 3] += temp2[1 * 4 + 1] * kernels_L2_3[6]
		//	+ temp2[1 * 4 + 2] * kernels_L2_3[7]
		//	+ temp2[1 * 4 + 3] * kernels_L2_3[8];

		b_relu_bn_max(3, 2, 3)
	}
}
__kernel void add_2_3_conv_4x3x3_lrelu_bn_max_R_cl(
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
													/*16*/__constant float* lrelu_w1_L2,
													/*17*/__constant float* lrelu_w2_L2,
													/*18*/__constant float* bn_b_L2
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
	for (int i = 0; i < 16; i += 4)
	{
		c[i + 0] = 0.f;
		c[i + 1] = 0.f;
		c[i + 2] = 0.f;
		c[i + 3] = 0.f;
	}

	{
		block_4x4_3x3(0, 0, 0, 0, 7, 0, 1)
		block_4x4_3x3(0, 1, 0, 1, 7, 0, 1)
		block_4x4_3x3(0, 2, 1, 0, 7, 0, 1)
		block_4x4_3x3(0, 3, 1, 1, 7, 0, 1)

		//c[4 * 0 + 0] = temp1[0 * 4 + 0] * kernels_L2_7[0]
		//	+ temp1[0 * 4 + 1] * kernels_L2_7[1]
		//	+ temp1[0 * 4 + 2] * kernels_L2_7[2];

		//c[4 * 0 + 1] = temp1[0 * 4 + 1] * kernels_L2_7[0]
		//	+ temp1[0 * 4 + 2] * kernels_L2_7[1]
		//	+ temp1[0 * 4 + 3] * kernels_L2_7[2];

		//c[4 * 0 + 2] = temp1[1 * 4 + 0] * kernels_L2_7[0]
		//	+ temp1[1 * 4 + 1] * kernels_L2_7[1]
		//	+ temp1[1 * 4 + 2] * kernels_L2_7[2];

		//c[4 * 0 + 3] = temp1[1 * 4 + 1] * kernels_L2_7[0]
		//	+ temp1[1 * 4 + 2] * kernels_L2_7[1]
		//	+ temp1[1 * 4 + 3] * kernels_L2_7[2];
	}
	{
		block_4x4_3x3(1, 0, 0, 0, 6, 0, 1)
		block_4x4_3x3(1, 1, 0, 1, 6, 0, 1)
		block_4x4_3x3(1, 2, 1, 0, 6, 0, 1)
		block_4x4_3x3(1, 3, 1, 1, 6, 0, 1)

		//c[4 * 1 + 0] = temp1[0 * 4 + 0] * kernels_L2_6[0]
		//	+ temp1[0 * 4 + 1] * kernels_L2_6[1]
		//	+ temp1[0 * 4 + 2] * kernels_L2_6[2];

		//c[4 * 1 + 1] = temp1[0 * 4 + 1] * kernels_L2_6[0]
		//	+ temp1[0 * 4 + 2] * kernels_L2_6[1]
		//	+ temp1[0 * 4 + 3] * kernels_L2_6[2];

		//c[4 * 1 + 2] = temp1[1 * 4 + 0] * kernels_L2_6[0]
		//	+ temp1[1 * 4 + 1] * kernels_L2_6[1]
		//	+ temp1[1 * 4 + 2] * kernels_L2_6[2];

		//c[4 * 1 + 3] = temp1[1 * 4 + 1] * kernels_L2_6[0]
		//	+ temp1[1 * 4 + 2] * kernels_L2_6[1]
		//	+ temp1[1 * 4 + 3] * kernels_L2_6[2];
	}
	{
		block_4x4_3x3(2, 0, 0, 0, 5, 0, 2)
		block_4x4_3x3(2, 1, 0, 1, 5, 0, 2)
		block_4x4_3x3(2, 2, 1, 0, 5, 0, 2)
		block_4x4_3x3(2, 3, 1, 1, 5, 0, 2)

		//c[4 * 2 + 0] = temp2[0 * 4 + 0] * kernels_L2_5[0]
		//	+ temp2[0 * 4 + 1] * kernels_L2_5[1]
		//	+ temp2[0 * 4 + 2] * kernels_L2_5[2];

		//c[4 * 2 + 1] = temp2[0 * 4 + 1] * kernels_L2_5[0]
		//	+ temp2[0 * 4 + 2] * kernels_L2_5[1]
		//	+ temp2[0 * 4 + 3] * kernels_L2_5[2];

		//c[4 * 2 + 2] = temp2[1 * 4 + 0] * kernels_L2_5[0]
		//	+ temp2[1 * 4 + 1] * kernels_L2_5[1]
		//	+ temp2[1 * 4 + 2] * kernels_L2_5[2];

		//c[4 * 2 + 3] = temp2[1 * 4 + 1] * kernels_L2_5[0]
		//	+ temp2[1 * 4 + 2] * kernels_L2_5[1]
		//	+ temp2[1 * 4 + 3] * kernels_L2_5[2];
	}
	{
		block_4x4_3x3(3, 0, 0, 0, 4, 0, 2)
		block_4x4_3x3(3, 1, 0, 1, 4, 0, 2)
		block_4x4_3x3(3, 2, 1, 0, 4, 0, 2)
		block_4x4_3x3(3, 3, 1, 1, 4, 0, 2)

		//c[4 * 3 + 0] = temp2[0 * 4 + 0] * kernels_L2_4[0]
		//	+ temp2[0 * 4 + 1] * kernels_L2_4[1]
		//	+ temp2[0 * 4 + 2] * kernels_L2_4[2];

		//c[4 * 3 + 1] = temp2[0 * 4 + 1] * kernels_L2_4[0]
		//	+ temp2[0 * 4 + 2] * kernels_L2_4[1]
		//	+ temp2[0 * 4 + 3] * kernels_L2_4[2];

		//c[4 * 3 + 2] = temp2[1 * 4 + 0] * kernels_L2_4[0]
		//	+ temp2[1 * 4 + 1] * kernels_L2_4[1]
		//	+ temp2[1 * 4 + 2] * kernels_L2_4[2];

		//c[4 * 3 + 3] = temp2[1 * 4 + 1] * kernels_L2_4[0]
		//	+ temp2[1 * 4 + 2] * kernels_L2_4[1]
		//	+ temp2[1 * 4 + 3] * kernels_L2_4[2];
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
		block_4x4_3x3(0, 0, 1, 0, 7, 1, 1)
		block_4x4_3x3(0, 1, 1, 1, 7, 1, 1)
		block_4x4_3x3(0, 2, 0, 0, 7, 1, 1)
		block_4x4_3x3(0, 3, 0, 1, 7, 1, 1)

		//c[4 * 0 + 0] += temp1[1 * 4 + 0] * kernels_L2_7[3]
		//	+ temp1[1 * 4 + 1] * kernels_L2_7[4]
		//	+ temp1[1 * 4 + 2] * kernels_L2_7[5];
	
		//c[4 * 0 + 1] += temp1[1 * 4 + 1] * kernels_L2_7[3]
		//	+ temp1[1 * 4 + 2] * kernels_L2_7[4]
		//	+ temp1[1 * 4 + 3] * kernels_L2_7[5];

		//c[4 * 0 + 2] += temp1[0 * 4 + 0] * kernels_L2_7[3]
		//	+ temp1[0 * 4 + 1] * kernels_L2_7[4]
		//	+ temp1[0 * 4 + 2] * kernels_L2_7[5];

		//c[4 * 0 + 3] += temp1[0 * 4 + 1] * kernels_L2_7[3]
		//	+ temp1[0 * 4 + 2] * kernels_L2_7[4]
		//	+ temp1[0 * 4 + 3] * kernels_L2_7[5];
	}
	{
		block_4x4_3x3(1, 0, 1, 0, 6, 1, 1)
		block_4x4_3x3(1, 1, 1, 1, 6, 1, 1)
		block_4x4_3x3(1, 2, 0, 0, 6, 1, 1)
		block_4x4_3x3(1, 3, 0, 1, 6, 1, 1)

		//c[4 * 1 + 0] += temp1[1 * 4 + 0] * kernels_L2_6[3]
		//	+ temp1[1 * 4 + 1] * kernels_L2_6[4]
		//	+ temp1[1 * 4 + 2] * kernels_L2_6[5];

		//c[4 * 1 + 1] += temp1[1 * 4 + 1] * kernels_L2_6[3]
		//	+ temp1[1 * 4 + 2] * kernels_L2_6[4]
		//	+ temp1[1 * 4 + 3] * kernels_L2_6[5];

		//c[4 * 1 + 2] += temp1[0 * 4 + 0] * kernels_L2_6[3]
		//	+ temp1[0 * 4 + 1] * kernels_L2_6[4]
		//	+ temp1[0 * 4 + 2] * kernels_L2_6[5];

		//c[4 * 1 + 3] += temp1[0 * 4 + 1] * kernels_L2_6[3]
		//	+ temp1[0 * 4 + 2] * kernels_L2_6[4]
		//	+ temp1[0 * 4 + 3] * kernels_L2_6[5];
	}
	{
		block_4x4_3x3(2, 0, 1, 0, 5, 1, 2)
		block_4x4_3x3(2, 1, 1, 1, 5, 1, 2)
		block_4x4_3x3(2, 2, 0, 0, 5, 1, 2)
		block_4x4_3x3(2, 3, 0, 1, 5, 1, 2)

		//c[4 * 2 + 0] += temp2[1 * 4 + 0] * kernels_L2_5[3]
		//	+ temp2[1 * 4 + 1] * kernels_L2_5[4]
		//	+ temp2[1 * 4 + 2] * kernels_L2_5[5];

		//c[4 * 2 + 1] += temp2[1 * 4 + 1] * kernels_L2_5[3]
		//	+ temp2[1 * 4 + 2] * kernels_L2_5[4]
		//	+ temp2[1 * 4 + 3] * kernels_L2_5[5];

		//c[4 * 2 + 2] += temp2[0 * 4 + 0] * kernels_L2_5[3]
		//	+ temp2[0 * 4 + 1] * kernels_L2_5[4]
		//	+ temp2[0 * 4 + 2] * kernels_L2_5[5];

		//c[4 * 2 + 3] += temp2[0 * 4 + 1] * kernels_L2_5[3]
		//	+ temp2[0 * 4 + 2] * kernels_L2_5[4]
		//	+ temp2[0 * 4 + 3] * kernels_L2_5[5];
	}
	{
		block_4x4_3x3(3, 0, 1, 0, 4, 1, 2)
		block_4x4_3x3(3, 1, 1, 1, 4, 1, 2)
		block_4x4_3x3(3, 2, 0, 0, 4, 1, 2)
		block_4x4_3x3(3, 3, 0, 1, 4, 1, 2)

		//c[4 * 3 + 0] += temp2[1 * 4 + 0] * kernels_L2_4[3]
		//	+ temp2[1 * 4 + 1] * kernels_L2_4[4]
		//	+ temp2[1 * 4 + 2] * kernels_L2_4[5];

		//c[4 * 3 + 1] += temp2[1 * 4 + 1] * kernels_L2_4[3]
		//	+ temp2[1 * 4 + 2] * kernels_L2_4[4]
		//	+ temp2[1 * 4 + 3] * kernels_L2_4[5];

		//c[4 * 3 + 2] += temp2[0 * 4 + 0] * kernels_L2_4[3]
		//	+ temp2[0 * 4 + 1] * kernels_L2_4[4]
		//	+ temp2[0 * 4 + 2] * kernels_L2_4[5];

		//c[4 * 3 + 3] += temp2[0 * 4 + 1] * kernels_L2_4[3]
		//	+ temp2[0 * 4 + 2] * kernels_L2_4[4]
		//	+ temp2[0 * 4 + 3] * kernels_L2_4[5];
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
		block_4x4_3x3(0, 0, 0, 0, 7, 2, 1)
		block_4x4_3x3(0, 1, 0, 1, 7, 2, 1)
		block_4x4_3x3(0, 2, 1, 0, 7, 2, 1)
		block_4x4_3x3(0, 3, 1, 1, 7, 2, 1)

		//c[4 * 0 + 0] += temp1[0 * 4 + 0] * kernels_L2_7[6]
		//	+ temp1[0 * 4 + 1] * kernels_L2_7[7]
		//	+ temp1[0 * 4 + 2] * kernels_L2_7[8];

		//c[4 * 0 + 1] += temp1[0 * 4 + 1] * kernels_L2_7[6]
		//	+ temp1[0 * 4 + 2] * kernels_L2_7[7]
		//	+ temp1[0 * 4 + 3] * kernels_L2_7[8];

		//c[4 * 0 + 2] += temp1[1 * 4 + 0] * kernels_L2_7[6]
		//	+ temp1[1 * 4 + 1] * kernels_L2_7[7]
		//	+ temp1[1 * 4 + 2] * kernels_L2_7[8];

		//c[4 * 0 + 3] += temp1[1 * 4 + 1] * kernels_L2_7[6]
		//	+ temp1[1 * 4 + 2] * kernels_L2_7[7]
		//	+ temp1[1 * 4 + 3] * kernels_L2_7[8];

		b_relu_bn_max(0, 2, 7)
	}
	{
		block_4x4_3x3(1, 0, 0, 0, 6, 2, 1)
		block_4x4_3x3(1, 1, 0, 1, 6, 2, 1)
		block_4x4_3x3(1, 2, 1, 0, 6, 2, 1)
		block_4x4_3x3(1, 3, 1, 1, 6, 2, 1)

		//c[4 * 1 + 0] += temp1[0 * 4 + 0] * kernels_L2_6[6]
		//	+ temp1[0 * 4 + 1] * kernels_L2_6[7]
		//	+ temp1[0 * 4 + 2] * kernels_L2_6[8];

		//c[4 * 1 + 1] += temp1[0 * 4 + 1] * kernels_L2_6[6]
		//	+ temp1[0 * 4 + 2] * kernels_L2_6[7]
		//	+ temp1[0 * 4 + 3] * kernels_L2_6[8];

		//c[4 * 1 + 2] += temp1[1 * 4 + 0] * kernels_L2_6[6]
		//	+ temp1[1 * 4 + 1] * kernels_L2_6[7]
		//	+ temp1[1 * 4 + 2] * kernels_L2_6[8];

		//c[4 * 1 + 3] += temp1[1 * 4 + 1] * kernels_L2_6[6]
		//	+ temp1[1 * 4 + 2] * kernels_L2_6[7]
		//	+ temp1[1 * 4 + 3] * kernels_L2_6[8];

		b_relu_bn_max(1, 2, 6)
	}
	{
		block_4x4_3x3(2, 0, 0, 0, 5, 2, 2)
		block_4x4_3x3(2, 1, 0, 1, 5, 2, 2)
		block_4x4_3x3(2, 2, 1, 0, 5, 2, 2)
		block_4x4_3x3(2, 3, 1, 1, 5, 2, 2)

		//c[4 * 2 + 0] += temp2[0 * 4 + 0] * kernels_L2_5[6]
		//	+ temp2[0 * 4 + 1] * kernels_L2_5[7]
		//	+ temp2[0 * 4 + 2] * kernels_L2_5[8];
		//
		//c[4 * 2 + 1] += temp2[0 * 4 + 1] * kernels_L2_5[6]
		//	+ temp2[0 * 4 + 2] * kernels_L2_5[7]
		//	+ temp2[0 * 4 + 3] * kernels_L2_5[8];

		//c[4 * 2 + 2] += temp2[1 * 4 + 0] * kernels_L2_5[6]
		//	+ temp2[1 * 4 + 1] * kernels_L2_5[7]
		//	+ temp2[1 * 4 + 2] * kernels_L2_5[8];

		//c[4 * 2 + 3] += temp2[1 * 4 + 1] * kernels_L2_5[6]
		//	+ temp2[1 * 4 + 2] * kernels_L2_5[7]
		//	+ temp2[1 * 4 + 3] * kernels_L2_5[8];

		b_relu_bn_max(2, 2, 5)
	}
	{
		block_4x4_3x3(3, 0, 0, 0, 4, 2, 2)
		block_4x4_3x3(3, 1, 0, 1, 4, 2, 2)
		block_4x4_3x3(3, 2, 1, 0, 4, 2, 2)
		block_4x4_3x3(3, 3, 1, 1, 4, 2, 2)

		//c[4 * 3 + 0] += temp2[0 * 4 + 0] * kernels_L2_4[6]
		//	+ temp2[0 * 4 + 1] * kernels_L2_4[7]
		//	+ temp2[0 * 4 + 2] * kernels_L2_4[8];

		//c[4 * 3 + 1] += temp2[0 * 4 + 1] * kernels_L2_4[6]
		//	+ temp2[0 * 4 + 2] * kernels_L2_4[7]
		//	+ temp2[0 * 4 + 3] * kernels_L2_4[8];

		//c[4 * 3 + 2] += temp2[1 * 4 + 0] * kernels_L2_4[6]
		//	+ temp2[1 * 4 + 1] * kernels_L2_4[7]
		//	+ temp2[1 * 4 + 2] * kernels_L2_4[8];

		//c[4 * 3 + 3] += temp2[1 * 4 + 1] * kernels_L2_4[6]
		//	+ temp2[1 * 4 + 2] * kernels_L2_4[7]
		//	+ temp2[1 * 4 + 3] * kernels_L2_4[8];

		b_relu_bn_max(3, 2, 4)
	}
}

__kernel void add_2_3_conv_4x5x4_L_cl(
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
									/*14*/__constant float* kernels_L3_3,
									/*15*/__constant float* conv_b_L3,
									/*16*/__constant float* lrelu_w1_L3,
									/*17*/__constant float* lrelu_w2_L3,
									/*18*/__constant float* bn_b_L3
									)
{
	const int i = 2 * get_global_id(0);
	const int j = 2 * get_global_id(1);

	if (i + 1 >= src_cols - 3 || j + 1 >= src_rows - 4) return;

	const int src_offset = j * src_step + i;
	const int dst_offset = j * dst_step + i;

	float temp1[10];

	src_L3_1 += src_offset;
	src_L3_2 += src_offset;
	temp1[0 * 5 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[0 * 5 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[0 * 5 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[0 * 5 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[0 * 5 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);

	src_L3_1 += src_step;
	src_L3_2 += src_step;
	temp1[1 * 5 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[1 * 5 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[1 * 5 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[1 * 5 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[1 * 5 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);
	
	float temp2[10];

	src_L3_3 += src_offset;
	temp2[0 * 5 + 0] = temp1[0 * 5 + 0] + *(src_L3_3 + 0);
	temp2[0 * 5 + 1] = temp1[0 * 5 + 1] + *(src_L3_3 + 1);
	temp2[0 * 5 + 2] = temp1[0 * 5 + 2] + *(src_L3_3 + 2);
	temp2[0 * 5 + 3] = temp1[0 * 5 + 3] + *(src_L3_3 + 3);
	temp2[0 * 5 + 4] = temp1[0 * 5 + 4] + *(src_L3_3 + 4);

	src_L3_3 += src_step;
	temp2[1 * 5 + 0] = temp1[1 * 5 + 0] + *(src_L3_3 + 0);
	temp2[1 * 5 + 1] = temp1[1 * 5 + 1] + *(src_L3_3 + 1);
	temp2[1 * 5 + 2] = temp1[1 * 5 + 2] + *(src_L3_3 + 2);
	temp2[1 * 5 + 3] = temp1[1 * 5 + 3] + *(src_L3_3 + 3);
	temp2[1 * 5 + 4] = temp1[1 * 5 + 4] + *(src_L3_3 + 4);

	float c[4 * 4];
	for (int i = 0; i < 16; i += 4)
	{
		c[i + 0] = 0.f;
		c[i + 1] = 0.f;
		c[i + 2] = 0.f;
		c[i + 3] = 0.f;
	}

	{
		block_4x4_4x5(0, 0, 0, 0, 0, 0, 1)
		block_4x4_4x5(0, 1, 0, 1, 0, 0, 1)
		block_4x4_4x5(0, 2, 1, 0, 0, 0, 1)
		block_4x4_4x5(0, 3, 1, 1, 0, 0, 1)

		//c[4 * 0 + 0] = temp1[0 * 5 + 0] * kernels_L3_0[0]
		//	+ temp1[0 * 5 + 1] * kernels_L3_0[1]
		//	+ temp1[0 * 5 + 2] * kernels_L3_0[2]
		//	+ temp1[0 * 5 + 3] * kernels_L3_0[3];

		//c[4 * 0 + 1] = temp1[0 * 5 + 1] * kernels_L3_0[0]
		//	+ temp1[0 * 5 + 2] * kernels_L3_0[1]
		//	+ temp1[0 * 5 + 3] * kernels_L3_0[2]
		//	+ temp1[0 * 5 + 4] * kernels_L3_0[3];

		//c[4 * 0 + 2] = temp1[1 * 5 + 0] * kernels_L3_0[0]
		//	+ temp1[1 * 5 + 1] * kernels_L3_0[1]
		//	+ temp1[1 * 5 + 2] * kernels_L3_0[2]
		//	+ temp1[1 * 5 + 3] * kernels_L3_0[3];

		//c[4 * 0 + 3] = temp1[1 * 5 + 1] * kernels_L3_0[0]
		//	+ temp1[1 * 5 + 2] * kernels_L3_0[1]
		//	+ temp1[1 * 5 + 3] * kernels_L3_0[2]
		//	+ temp1[1 * 5 + 4] * kernels_L3_0[3];
	}
	{
		block_4x4_4x5(1, 0, 0, 0, 1, 0, 1)
		block_4x4_4x5(1, 1, 0, 1, 1, 0, 1)
		block_4x4_4x5(1, 2, 1, 0, 1, 0, 1)
		block_4x4_4x5(1, 3, 1, 1, 1, 0, 1)

		//c[4 * 1 + 0] = temp1[0 * 5 + 0] * kernels_L3_1[0]
		//	+ temp1[0 * 5 + 1] * kernels_L3_1[1]
		//	+ temp1[0 * 5 + 2] * kernels_L3_1[2]
		//	+ temp1[0 * 5 + 3] * kernels_L3_1[3];

		//c[4 * 1 + 1] = temp1[0 * 5 + 1] * kernels_L3_1[0]
		//	+ temp1[0 * 5 + 2] * kernels_L3_1[1]
		//	+ temp1[0 * 5 + 3] * kernels_L3_1[2]
		//	+ temp1[0 * 5 + 4] * kernels_L3_1[3];

		//c[4 * 1 + 2] = temp1[1 * 5 + 0] * kernels_L3_1[0]
		//	+ temp1[1 * 5 + 1] * kernels_L3_1[1]
		//	+ temp1[1 * 5 + 2] * kernels_L3_1[2]
		//	+ temp1[1 * 5 + 3] * kernels_L3_1[3];

		//c[4 * 1 + 3] = temp1[1 * 5 + 1] * kernels_L3_1[0]
		//	+ temp1[1 * 5 + 2] * kernels_L3_1[1]
		//	+ temp1[1 * 5 + 3] * kernels_L3_1[2]
		//	+ temp1[1 * 5 + 4] * kernels_L3_1[3];
	}
	{
		block_4x4_4x5(2, 0, 0, 0, 2, 0, 2)
		block_4x4_4x5(2, 1, 0, 1, 2, 0, 2)
		block_4x4_4x5(2, 2, 1, 0, 2, 0, 2)
		block_4x4_4x5(2, 3, 1, 1, 2, 0, 2)

		//c[4 * 2 + 0] = temp2[0 * 5 + 0] * kernels_L3_2[0]
		//	+ temp2[0 * 5 + 1] * kernels_L3_2[1]
		//	+ temp2[0 * 5 + 2] * kernels_L3_2[2]
		//	+ temp2[0 * 5 + 3] * kernels_L3_2[3];
		
		//c[4 * 2 + 1] = temp2[0 * 5 + 1] * kernels_L3_2[0]
		//	+ temp2[0 * 5 + 2] * kernels_L3_2[1]
		//	+ temp2[0 * 5 + 3] * kernels_L3_2[2]
		//	+ temp2[0 * 5 + 4] * kernels_L3_2[3];

		//c[4 * 2 + 2] = temp2[1 * 5 + 0] * kernels_L3_2[0]
		//	+ temp2[1 * 5 + 1] * kernels_L3_2[1]
		//	+ temp2[1 * 5 + 2] * kernels_L3_2[2]
		//	+ temp2[1 * 5 + 3] * kernels_L3_2[3];

		//c[4 * 2 + 3] = temp2[1 * 5 + 1] * kernels_L3_2[0]
		//	+ temp2[1 * 5 + 2] * kernels_L3_2[1]
		//	+ temp2[1 * 5 + 3] * kernels_L3_2[2]
		//	+ temp2[1 * 5 + 4] * kernels_L3_2[3];
	}
	{
		block_4x4_4x5(3, 0, 0, 0, 3, 0, 2)
		block_4x4_4x5(3, 1, 0, 1, 3, 0, 2)
		block_4x4_4x5(3, 2, 1, 0, 3, 0, 2)
		block_4x4_4x5(3, 3, 1, 1, 3, 0, 2)

		//c[4 * 3 + 0] = temp2[0 * 5 + 0] * kernels_L3_3[0]
		//	+ temp2[0 * 5 + 1] * kernels_L3_3[1]
		//	+ temp2[0 * 5 + 2] * kernels_L3_3[2]
		//	+ temp2[0 * 5 + 3] * kernels_L3_3[3];

		//c[4 * 3 + 1] = temp2[0 * 5 + 1] * kernels_L3_3[0]
		//	+ temp2[0 * 5 + 2] * kernels_L3_3[1]
		//	+ temp2[0 * 5 + 3] * kernels_L3_3[2]
		//	+ temp2[0 * 5 + 4] * kernels_L3_3[3];

		//c[4 * 3 + 2] = temp2[1 * 5 + 0] * kernels_L3_3[0]
		//	+ temp2[1 * 5 + 1] * kernels_L3_3[1]
		//	+ temp2[1 * 5 + 2] * kernels_L3_3[2]
		//	+ temp2[1 * 5 + 3] * kernels_L3_3[3];

		//c[4 * 3 + 3] = temp2[1 * 5 + 1] * kernels_L3_3[0]
		//	+ temp2[1 * 5 + 2] * kernels_L3_3[1]
		//	+ temp2[1 * 5 + 3] * kernels_L3_3[2]
		//	+ temp2[1 * 5 + 4] * kernels_L3_3[3];
	}

	src_L3_1 += src_step;
	src_L3_2 += src_step;
	temp1[0 * 5 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[0 * 5 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[0 * 5 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[0 * 5 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[0 * 5 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);

	src_L3_3 += src_step;
	temp2[0 * 5 + 0] = temp1[0 * 5 + 0] + *(src_L3_3 + 0);
	temp2[0 * 5 + 1] = temp1[0 * 5 + 1] + *(src_L3_3 + 1);
	temp2[0 * 5 + 2] = temp1[0 * 5 + 2] + *(src_L3_3 + 2);
	temp2[0 * 5 + 3] = temp1[0 * 5 + 3] + *(src_L3_3 + 3);
	temp2[0 * 5 + 4] = temp1[0 * 5 + 4] + *(src_L3_3 + 4);

	{
		block_4x4_4x5(0, 0, 1, 0, 0, 1, 1)
		block_4x4_4x5(0, 1, 1, 1, 0, 1, 1)
		block_4x4_4x5(0, 2, 0, 0, 0, 1, 1)
		block_4x4_4x5(0, 3, 0, 1, 0, 1, 1)

		//c[4 * 0 + 0] += temp1[1 * 5 + 0] * kernels_L3_0[4]
		//	+ temp1[1 * 5 + 1] * kernels_L3_0[5]
		//	+ temp1[1 * 5 + 2] * kernels_L3_0[6]
		//	+ temp1[1 * 5 + 3] * kernels_L3_0[7];

		//c[4 * 0 + 1] += temp1[1 * 5 + 1] * kernels_L3_0[4]
		//	+ temp1[1 * 5 + 2] * kernels_L3_0[5]
		//	+ temp1[1 * 5 + 3] * kernels_L3_0[6]
		//	+ temp1[1 * 5 + 4] * kernels_L3_0[7];

		//c[4 * 0 + 2] += temp1[0 * 5 + 0] * kernels_L3_0[4]
		//	+ temp1[0 * 5 + 1] * kernels_L3_0[5]
		//	+ temp1[0 * 5 + 2] * kernels_L3_0[6]
		//	+ temp1[0 * 5 + 3] * kernels_L3_0[7];

		//c[4 * 0 + 3] += temp1[0 * 5 + 1] * kernels_L3_0[4]
		//	+ temp1[0 * 5 + 2] * kernels_L3_0[5]
		//	+ temp1[0 * 5 + 3] * kernels_L3_0[6]
		//	+ temp1[0 * 5 + 4] * kernels_L3_0[7];
	}
	{
		block_4x4_4x5(1, 0, 1, 0, 1, 1, 1)
		block_4x4_4x5(1, 1, 1, 1, 1, 1, 1)
		block_4x4_4x5(1, 2, 0, 0, 1, 1, 1)
		block_4x4_4x5(1, 3, 0, 1, 1, 1, 1)

		//c[4 * 1 + 0] += temp1[1 * 5 + 0] * kernels_L3_1[4]
		//	+ temp1[1 * 5 + 1] * kernels_L3_1[5]
		//	+ temp1[1 * 5 + 2] * kernels_L3_1[6]
		//	+ temp1[1 * 5 + 3] * kernels_L3_1[7];

		//c[4 * 1 + 1] += temp1[1 * 5 + 1] * kernels_L3_1[4]
		//	+ temp1[1 * 5 + 2] * kernels_L3_1[5]
		//	+ temp1[1 * 5 + 3] * kernels_L3_1[6]
		//	+ temp1[1 * 5 + 4] * kernels_L3_1[7];

		//c[4 * 1 + 2] += temp1[0 * 5 + 0] * kernels_L3_1[4]
		//	+ temp1[0 * 5 + 1] * kernels_L3_1[5]
		//	+ temp1[0 * 5 + 2] * kernels_L3_1[6]
		//	+ temp1[0 * 5 + 3] * kernels_L3_1[7];

		//c[4 * 1 + 3] += temp1[0 * 5 + 1] * kernels_L3_1[4]
		//	+ temp1[0 * 5 + 2] * kernels_L3_1[5]
		//	+ temp1[0 * 5 + 3] * kernels_L3_1[6]
		//	+ temp1[0 * 5 + 4] * kernels_L3_1[7];
	}
	{
		block_4x4_4x5(2, 0, 1, 0, 2, 1, 2)
		block_4x4_4x5(2, 1, 1, 1, 2, 1, 2)
		block_4x4_4x5(2, 2, 0, 0, 2, 1, 2)
		block_4x4_4x5(2, 3, 0, 1, 2, 1, 2)

		//c[4 * 2 + 0] += temp2[1 * 5 + 0] * kernels_L3_2[4]
		//	+ temp2[1 * 5 + 1] * kernels_L3_2[5]
		//	+ temp2[1 * 5 + 2] * kernels_L3_2[6]
		//	+ temp2[1 * 5 + 3] * kernels_L3_2[7];

		//c[4 * 2 + 1] += temp2[1 * 5 + 1] * kernels_L3_2[4]
		//	+ temp2[1 * 5 + 2] * kernels_L3_2[5]
		//	+ temp2[1 * 5 + 3] * kernels_L3_2[6]
		//	+ temp2[1 * 5 + 4] * kernels_L3_2[7];

		//c[4 * 2 + 2] += temp2[0 * 5 + 0] * kernels_L3_2[4]
		//	+ temp2[0 * 5 + 1] * kernels_L3_2[5]
		//	+ temp2[0 * 5 + 2] * kernels_L3_2[6]
		//	+ temp2[0 * 5 + 3] * kernels_L3_2[7];

		//c[4 * 2 + 3] += temp2[0 * 5 + 1] * kernels_L3_2[4]
		//	+ temp2[0 * 5 + 2] * kernels_L3_2[5]
		//	+ temp2[0 * 5 + 3] * kernels_L3_2[6]
		//	+ temp2[0 * 5 + 4] * kernels_L3_2[7];
	}
	{
		block_4x4_4x5(3, 0, 1, 0, 3, 1, 2)
		block_4x4_4x5(3, 1, 1, 1, 3, 1, 2)
		block_4x4_4x5(3, 2, 0, 0, 3, 1, 2)
		block_4x4_4x5(3, 3, 0, 1, 3, 1, 2)

		//c[4 * 3 + 0] += temp2[1 * 5 + 0] * kernels_L3_3[4]
		//	+ temp2[1 * 5 + 1] * kernels_L3_3[5]
		//	+ temp2[1 * 5 + 2] * kernels_L3_3[6]
		//	+ temp2[1 * 5 + 3] * kernels_L3_3[7];

		//c[4 * 3 + 1] += temp2[1 * 5 + 1] * kernels_L3_3[4]
		//	+ temp2[1 * 5 + 2] * kernels_L3_3[5]
		//	+ temp2[1 * 5 + 3] * kernels_L3_3[6]
		//	+ temp2[1 * 5 + 4] * kernels_L3_3[7];

		//c[4 * 3 + 2] += temp2[0 * 5 + 0] * kernels_L3_3[4]
		//	+ temp2[0 * 5 + 1] * kernels_L3_3[5]
		//	+ temp2[0 * 5 + 2] * kernels_L3_3[6]
		//	+ temp2[0 * 5 + 3] * kernels_L3_3[7];

		//c[4 * 3 + 3] += temp2[0 * 5 + 1] * kernels_L3_3[4]
		//	+ temp2[0 * 5 + 2] * kernels_L3_3[5]
		//	+ temp2[0 * 5 + 3] * kernels_L3_3[6]
		//	+ temp2[0 * 5 + 4] * kernels_L3_3[7];
	}

	src_L3_1 += src_step;
	src_L3_2 += src_step;
	temp1[1 * 5 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[1 * 5 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[1 * 5 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[1 * 5 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[1 * 5 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);

	src_L3_3 += src_step;
	temp2[1 * 5 + 0] = temp1[1 * 5 + 0] + *(src_L3_3 + 0);
	temp2[1 * 5 + 1] = temp1[1 * 5 + 1] + *(src_L3_3 + 1);
	temp2[1 * 5 + 2] = temp1[1 * 5 + 2] + *(src_L3_3 + 2);
	temp2[1 * 5 + 3] = temp1[1 * 5 + 3] + *(src_L3_3 + 3);
	temp2[1 * 5 + 4] = temp1[1 * 5 + 4] + *(src_L3_3 + 4);

	{
		block_4x4_4x5(0, 0, 0, 0, 0, 2, 1)
		block_4x4_4x5(0, 1, 0, 1, 0, 2, 1)
		block_4x4_4x5(0, 2, 1, 0, 0, 2, 1)
		block_4x4_4x5(0, 3, 1, 1, 0, 2, 1)

		//c[4 * 0 + 0] += temp1[0 * 5 + 0] * kernels_L3_0[8]
		//	+ temp1[0 * 5 + 1] * kernels_L3_0[9]
		//	+ temp1[0 * 5 + 2] * kernels_L3_0[10]
		//	+ temp1[0 * 5 + 3] * kernels_L3_0[11];

		//c[4 * 0 + 1] += temp1[0 * 5 + 1] * kernels_L3_0[8]
		//	+ temp1[0 * 5 + 2] * kernels_L3_0[9]
		//	+ temp1[0 * 5 + 3] * kernels_L3_0[10]
		//	+ temp1[0 * 5 + 4] * kernels_L3_0[11];

		//c[4 * 0 + 2] += temp1[1 * 5 + 0] * kernels_L3_0[8]
		//	+ temp1[1 * 5 + 1] * kernels_L3_0[9]
		//	+ temp1[1 * 5 + 2] * kernels_L3_0[10]
		//	+ temp1[1 * 5 + 3] * kernels_L3_0[11];

		//c[4 * 0 + 3] += temp1[1 * 5 + 1] * kernels_L3_0[8]
		//	+ temp1[1 * 5 + 2] * kernels_L3_0[9]
		//	+ temp1[1 * 5 + 3] * kernels_L3_0[10]
		//	+ temp1[1 * 5 + 4] * kernels_L3_0[11];
	}
	{
		block_4x4_4x5(1, 0, 0, 0, 1, 2, 1)
		block_4x4_4x5(1, 1, 0, 1, 1, 2, 1)
		block_4x4_4x5(1, 2, 1, 0, 1, 2, 1)
		block_4x4_4x5(1, 3, 1, 1, 1, 2, 1)

		//c[4 * 1 + 0] += temp1[0 * 5 + 0] * kernels_L3_1[8]
		//	+ temp1[0 * 5 + 1] * kernels_L3_1[9]
		//	+ temp1[0 * 5 + 2] * kernels_L3_1[10]
		//	+ temp1[0 * 5 + 3] * kernels_L3_1[11];

		//c[4 * 1 + 1] += temp1[0 * 5 + 1] * kernels_L3_1[8]
		//	+ temp1[0 * 5 + 2] * kernels_L3_1[9]
		//	+ temp1[0 * 5 + 3] * kernels_L3_1[10]
		//	+ temp1[0 * 5 + 4] * kernels_L3_1[11];

		//c[4 * 1 + 2] += temp1[1 * 5 + 0] * kernels_L3_1[8]
		//	+ temp1[1 * 5 + 1] * kernels_L3_1[9]
		//	+ temp1[1 * 5 + 2] * kernels_L3_1[10]
		//	+ temp1[1 * 5 + 3] * kernels_L3_1[11];

		//c[4 * 1 + 3] += temp1[1 * 5 + 1] * kernels_L3_1[8]
		//	+ temp1[1 * 5 + 2] * kernels_L3_1[9]
		//	+ temp1[1 * 5 + 3] * kernels_L3_1[10]
		//	+ temp1[1 * 5 + 4] * kernels_L3_1[11];
	}
	{
		block_4x4_4x5(2, 0, 0, 0, 2, 2, 2)
		block_4x4_4x5(2, 1, 0, 1, 2, 2, 2)
		block_4x4_4x5(2, 2, 1, 0, 2, 2, 2)
		block_4x4_4x5(2, 3, 1, 1, 2, 2, 2)

		//c[4 * 2 + 0] += temp2[0 * 5 + 0] * kernels_L3_2[8]
		//	+ temp2[0 * 5 + 1] * kernels_L3_2[9]
		//	+ temp2[0 * 5 + 2] * kernels_L3_2[10]
		//	+ temp2[0 * 5 + 3] * kernels_L3_2[11];

		//c[4 * 2 + 1] += temp2[0 * 5 + 1] * kernels_L3_2[8]
		//	+ temp2[0 * 5 + 2] * kernels_L3_2[9]
		//	+ temp2[0 * 5 + 3] * kernels_L3_2[10]
		//	+ temp2[0 * 5 + 4] * kernels_L3_2[11];

		//c[4 * 2 + 2] += temp2[1 * 5 + 0] * kernels_L3_2[8]
		//	+ temp2[1 * 5 + 1] * kernels_L3_2[9]
		//	+ temp2[1 * 5 + 2] * kernels_L3_2[10]
		//	+ temp2[1 * 5 + 3] * kernels_L3_2[11];

		//c[4 * 2 + 3] += temp2[1 * 5 + 1] * kernels_L3_2[8]
		//	+ temp2[1 * 5 + 2] * kernels_L3_2[9]
		//	+ temp2[1 * 5 + 3] * kernels_L3_2[10]
		//	+ temp2[1 * 5 + 4] * kernels_L3_2[11];
	}
	{
		block_4x4_4x5(3, 0, 0, 0, 3, 2, 2)
		block_4x4_4x5(3, 1, 0, 1, 3, 2, 2)
		block_4x4_4x5(3, 2, 1, 0, 3, 2, 2)
		block_4x4_4x5(3, 3, 1, 1, 3, 2, 2)

		//c[4 * 3 + 0] += temp2[0 * 5 + 0] * kernels_L3_3[8]
		//	+ temp2[0 * 5 + 1] * kernels_L3_3[9]
		//	+ temp2[0 * 5 + 2] * kernels_L3_3[10]
		//	+ temp2[0 * 5 + 3] * kernels_L3_3[11];

		//c[4 * 3 + 1] += temp2[0 * 5 + 1] * kernels_L3_3[8]
		//	+ temp2[0 * 5 + 2] * kernels_L3_3[9]
		//	+ temp2[0 * 5 + 3] * kernels_L3_3[10]
		//	+ temp2[0 * 5 + 4] * kernels_L3_3[11];

		//c[4 * 3 + 2] += temp2[1 * 5 + 0] * kernels_L3_3[8]
		//	+ temp2[1 * 5 + 1] * kernels_L3_3[9]
		//	+ temp2[1 * 5 + 2] * kernels_L3_3[10]
		//	+ temp2[1 * 5 + 3] * kernels_L3_3[11];

		//c[4 * 3 + 3] += temp2[1 * 5 + 1] * kernels_L3_3[8]
		//	+ temp2[1 * 5 + 2] * kernels_L3_3[9]
		//	+ temp2[1 * 5 + 3] * kernels_L3_3[10]
		//	+ temp2[1 * 5 + 4] * kernels_L3_3[11];
	}

	src_L3_1 += src_step;
	src_L3_2 += src_step;
	temp1[0 * 5 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[0 * 5 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[0 * 5 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[0 * 5 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[0 * 5 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);

	src_L3_3 += src_step;
	temp2[0 * 5 + 0] = temp1[0 * 5 + 0] + *(src_L3_3 + 0);
	temp2[0 * 5 + 1] = temp1[0 * 5 + 1] + *(src_L3_3 + 1);
	temp2[0 * 5 + 2] = temp1[0 * 5 + 2] + *(src_L3_3 + 2);
	temp2[0 * 5 + 3] = temp1[0 * 5 + 3] + *(src_L3_3 + 3);
	temp2[0 * 5 + 4] = temp1[0 * 5 + 4] + *(src_L3_3 + 4);

	{
		block_4x4_4x5(0, 0, 1, 0, 0, 3, 1)
		block_4x4_4x5(0, 1, 1, 1, 0, 3, 1)
		block_4x4_4x5(0, 2, 0, 0, 0, 3, 1)
		block_4x4_4x5(0, 3, 0, 1, 0, 3, 1)

		//c[4 * 0 + 0] += temp1[1 * 5 + 0] * kernels_L3_0[12]
		//	+ temp1[1 * 5 + 1] * kernels_L3_0[13]
		//	+ temp1[1 * 5 + 2] * kernels_L3_0[14]
		//	+ temp1[1 * 5 + 3] * kernels_L3_0[15];

		//c[4 * 0 + 1] += temp1[1 * 5 + 1] * kernels_L3_0[12]
		//	+ temp1[1 * 5 + 2] * kernels_L3_0[13]
		//	+ temp1[1 * 5 + 3] * kernels_L3_0[14]
		//	+ temp1[1 * 5 + 4] * kernels_L3_0[15];

		//c[4 * 0 + 2] += temp1[0 * 5 + 0] * kernels_L3_0[12]
		//	+ temp1[0 * 5 + 1] * kernels_L3_0[13]
		//	+ temp1[0 * 5 + 2] * kernels_L3_0[14]
		//	+ temp1[0 * 5 + 3] * kernels_L3_0[15];

		//c[4 * 0 + 3] += temp1[0 * 5 + 1] * kernels_L3_0[12]
		//	+ temp1[0 * 5 + 2] * kernels_L3_0[13]
		//	+ temp1[0 * 5 + 3] * kernels_L3_0[14]
		//	+ temp1[0 * 5 + 4] * kernels_L3_0[15];
	}
	{
		block_4x4_4x5(1, 0, 1, 0, 1, 3, 1)
		block_4x4_4x5(1, 1, 1, 1, 1, 3, 1)
		block_4x4_4x5(1, 2, 0, 0, 1, 3, 1)
		block_4x4_4x5(1, 3, 0, 1, 1, 3, 1)

		//c[4 * 1 + 0] += temp1[1 * 5 + 0] * kernels_L3_1[12]
		//	+ temp1[1 * 5 + 1] * kernels_L3_1[13]
		//	+ temp1[1 * 5 + 2] * kernels_L3_1[14]
		//	+ temp1[1 * 5 + 3] * kernels_L3_1[15];

		//c[4 * 1 + 1] += temp1[1 * 5 + 1] * kernels_L3_1[12]
		//	+ temp1[1 * 5 + 2] * kernels_L3_1[13]
		//	+ temp1[1 * 5 + 3] * kernels_L3_1[14]
		//	+ temp1[1 * 5 + 4] * kernels_L3_1[15];

		//c[4 * 1 + 2] += temp1[0 * 5 + 0] * kernels_L3_1[12]
		//	+ temp1[0 * 5 + 1] * kernels_L3_1[13]
		//	+ temp1[0 * 5 + 2] * kernels_L3_1[14]
		//	+ temp1[0 * 5 + 3] * kernels_L3_1[15];

		//c[4 * 1 + 3] += temp1[0 * 5 + 1] * kernels_L3_1[12]
		//	+ temp1[0 * 5 + 2] * kernels_L3_1[13]
		//	+ temp1[0 * 5 + 3] * kernels_L3_1[14]
		//	+ temp1[0 * 5 + 4] * kernels_L3_1[15];
	}
	{
		block_4x4_4x5(2, 0, 1, 0, 2, 3, 2)
		block_4x4_4x5(2, 1, 1, 1, 2, 3, 2)
		block_4x4_4x5(2, 2, 0, 0, 2, 3, 2)
		block_4x4_4x5(2, 3, 0, 1, 2, 3, 2)

		//c[4 * 2 + 0] += temp2[1 * 5 + 0] * kernels_L3_2[12]
		//	+ temp2[1 * 5 + 1] * kernels_L3_2[13]
		//	+ temp2[1 * 5 + 2] * kernels_L3_2[14]
		//	+ temp2[1 * 5 + 3] * kernels_L3_2[15];

		//c[4 * 2 + 1] += temp2[1 * 5 + 1] * kernels_L3_2[12]
		//	+ temp2[1 * 5 + 2] * kernels_L3_2[13]
		//	+ temp2[1 * 5 + 3] * kernels_L3_2[14]
		//	+ temp2[1 * 5 + 4] * kernels_L3_2[15];

		//c[4 * 2 + 2] += temp2[0 * 5 + 0] * kernels_L3_2[12]
		//	+ temp2[0 * 5 + 1] * kernels_L3_2[13]
		//	+ temp2[0 * 5 + 2] * kernels_L3_2[14]
		//	+ temp2[0 * 5 + 3] * kernels_L3_2[15];

		//c[4 * 2 + 3] += temp2[0 * 5 + 1] * kernels_L3_2[12]
		//	+ temp2[0 * 5 + 2] * kernels_L3_2[13]
		//	+ temp2[0 * 5 + 3] * kernels_L3_2[14]
		//	+ temp2[0 * 5 + 4] * kernels_L3_2[15];
	}
	{
		block_4x4_4x5(3, 0, 1, 0, 3, 3, 2)
		block_4x4_4x5(3, 1, 1, 1, 3, 3, 2)
		block_4x4_4x5(3, 2, 0, 0, 3, 3, 2)
		block_4x4_4x5(3, 3, 0, 1, 3, 3, 2)

		//c[4 * 3 + 0] += temp2[1 * 5 + 0] * kernels_L3_3[12]
		//	+ temp2[1 * 5 + 1] * kernels_L3_3[13]
		//	+ temp2[1 * 5 + 2] * kernels_L3_3[14]
		//	+ temp2[1 * 5 + 3] * kernels_L3_3[15];

		//c[4 * 3 + 1] += temp2[1 * 5 + 1] * kernels_L3_3[12]
		//	+ temp2[1 * 5 + 2] * kernels_L3_3[13]
		//	+ temp2[1 * 5 + 3] * kernels_L3_3[14]
		//	+ temp2[1 * 5 + 4] * kernels_L3_3[15];

		//c[4 * 3 + 2] += temp2[0 * 5 + 0] * kernels_L3_3[12]
		//	+ temp2[0 * 5 + 1] * kernels_L3_3[13]
		//	+ temp2[0 * 5 + 2] * kernels_L3_3[14]
		//	+ temp2[0 * 5 + 3] * kernels_L3_3[15];

		//c[4 * 3 + 3] += temp2[0 * 5 + 1] * kernels_L3_3[12]
		//	+ temp2[0 * 5 + 2] * kernels_L3_3[13]
		//	+ temp2[0 * 5 + 3] * kernels_L3_3[14]
		//	+ temp2[0 * 5 + 4] * kernels_L3_3[15];
	}

	src_L3_1 += src_step;
	src_L3_2 += src_step;
	temp1[1 * 5 + 0] = *(src_L3_1 + 0) + *(src_L3_2 + 0);
	temp1[1 * 5 + 1] = *(src_L3_1 + 1) + *(src_L3_2 + 1);
	temp1[1 * 5 + 2] = *(src_L3_1 + 2) + *(src_L3_2 + 2);
	temp1[1 * 5 + 3] = *(src_L3_1 + 3) + *(src_L3_2 + 3);
	temp1[1 * 5 + 4] = *(src_L3_1 + 4) + *(src_L3_2 + 4);

	src_L3_3 += src_step;
	temp2[1 * 5 + 0] = temp1[1 * 5 + 0] + *(src_L3_3 + 0);
	temp2[1 * 5 + 1] = temp1[1 * 5 + 1] + *(src_L3_3 + 1);
	temp2[1 * 5 + 2] = temp1[1 * 5 + 2] + *(src_L3_3 + 2);
	temp2[1 * 5 + 3] = temp1[1 * 5 + 3] + *(src_L3_3 + 3);
	temp2[1 * 5 + 4] = temp1[1 * 5 + 4] + *(src_L3_3 + 4);

	{
		block_4x4_4x5(0, 0, 0, 0, 0, 4, 1)
		block_4x4_4x5(0, 1, 0, 1, 0, 4, 1)
		block_4x4_4x5(0, 2, 1, 0, 0, 4, 1)
		block_4x4_4x5(0, 3, 1, 1, 0, 4, 1)

		//c[4 * 0 + 0] += temp1[0 * 5 + 0] * kernels_L3_0[16]
		//	+ temp1[0 * 5 + 1] * kernels_L3_0[17]
		//	+ temp1[0 * 5 + 2] * kernels_L3_0[18]
		//	+ temp1[0 * 5 + 3] * kernels_L3_0[19];

		//c[4 * 0 + 1] += temp1[0 * 5 + 1] * kernels_L3_0[16]
		//	+ temp1[0 * 5 + 2] * kernels_L3_0[17]
		//	+ temp1[0 * 5 + 3] * kernels_L3_0[18]
		//	+ temp1[0 * 5 + 4] * kernels_L3_0[19];

		//c[4 * 0 + 2] += temp1[1 * 5 + 0] * kernels_L3_0[16]
		//	+ temp1[1 * 5 + 1] * kernels_L3_0[17]
		//	+ temp1[1 * 5 + 2] * kernels_L3_0[18]
		//	+ temp1[1 * 5 + 3] * kernels_L3_0[19];

		//c[4 * 0 + 3] += temp1[1 * 5 + 1] * kernels_L3_0[16]
		//	+ temp1[1 * 5 + 2] * kernels_L3_0[17]
		//	+ temp1[1 * 5 + 3] * kernels_L3_0[18]
		//	+ temp1[1 * 5 + 4] * kernels_L3_0[19];

		b_relu_bn(0, 3, 0);

		dst_0 += dst_offset;
		*(dst_0 + 0) = res_0;
		*(dst_0 + 1) = res_1;
		dst_0 += dst_step;
		*(dst_0 + 0) = res_2;
		*(dst_0 + 1) = res_3;
	}
	{
		block_4x4_4x5(1, 0, 0, 0, 1, 4, 1)
		block_4x4_4x5(1, 1, 0, 1, 1, 4, 1)
		block_4x4_4x5(1, 2, 1, 0, 1, 4, 1)
		block_4x4_4x5(1, 3, 1, 1, 1, 4, 1)

		//c[4 * 1 + 0] += temp1[0 * 5 + 0] * kernels_L3_1[16]
		//	+ temp1[0 * 5 + 1] * kernels_L3_1[17]
		//	+ temp1[0 * 5 + 2] * kernels_L3_1[18]
		//	+ temp1[0 * 5 + 3] * kernels_L3_1[19];

		//c[4 * 1 + 1] += temp1[0 * 5 + 1] * kernels_L3_1[16]
		//	+ temp1[0 * 5 + 2] * kernels_L3_1[17]
		//	+ temp1[0 * 5 + 3] * kernels_L3_1[18]
		//	+ temp1[0 * 5 + 4] * kernels_L3_1[19];

		//c[4 * 1 + 2] += temp1[1 * 5 + 0] * kernels_L3_1[16]
		//	+ temp1[1 * 5 + 1] * kernels_L3_1[17]
		//	+ temp1[1 * 5 + 2] * kernels_L3_1[18]
		//	+ temp1[1 * 5 + 3] * kernels_L3_1[19];

		//c[4 * 1 + 3] += temp1[1 * 5 + 1] * kernels_L3_1[16]
		//	+ temp1[1 * 5 + 2] * kernels_L3_1[17]
		//	+ temp1[1 * 5 + 3] * kernels_L3_1[18]
		//	+ temp1[1 * 5 + 4] * kernels_L3_1[19];

		b_relu_bn(1, 3, 1);

		dst_1 += dst_offset;
		*(dst_1 + 0) = res_0;
		*(dst_1 + 1) = res_1;
		dst_1 += dst_step;
		*(dst_1 + 0) = res_2;
		*(dst_1 + 1) = res_3;
	}
	{
		block_4x4_4x5(2, 0, 0, 0, 2, 4, 2)
		block_4x4_4x5(2, 1, 0, 1, 2, 4, 2)
		block_4x4_4x5(2, 2, 1, 0, 2, 4, 2)
		block_4x4_4x5(2, 3, 1, 1, 2, 4, 2)

		//c[4 * 2 + 0] += temp2[0 * 5 + 0] * kernels_L3_2[16]
		//	+ temp2[0 * 5 + 1] * kernels_L3_2[17]
		//	+ temp2[0 * 5 + 2] * kernels_L3_2[18]
		//	+ temp2[0 * 5 + 3] * kernels_L3_2[19];

		//c[4 * 2 + 1] += temp2[0 * 5 + 1] * kernels_L3_2[16]
		//	+ temp2[0 * 5 + 2] * kernels_L3_2[17]
		//	+ temp2[0 * 5 + 3] * kernels_L3_2[18]
		//	+ temp2[0 * 5 + 4] * kernels_L3_2[19];

		//c[4 * 2 + 2] += temp2[1 * 5 + 0] * kernels_L3_2[16]
		//	+ temp2[1 * 5 + 1] * kernels_L3_2[17]
		//	+ temp2[1 * 5 + 2] * kernels_L3_2[18]
		//	+ temp2[1 * 5 + 3] * kernels_L3_2[19];

		//c[4 * 2 + 3] += temp2[1 * 5 + 1] * kernels_L3_2[16]
		//	+ temp2[1 * 5 + 2] * kernels_L3_2[17]
		//	+ temp2[1 * 5 + 3] * kernels_L3_2[18]
		//	+ temp2[1 * 5 + 4] * kernels_L3_2[19];

		b_relu_bn(2, 3, 2);

		dst_2 += dst_offset;
		*(dst_2 + 0) = res_0;
		*(dst_2 + 1) = res_1;
		dst_2 += dst_step;
		*(dst_2 + 0) = res_2;
		*(dst_2 + 1) = res_3;
	}
	{
		block_4x4_4x5(3, 0, 0, 0, 3, 4, 2)
		block_4x4_4x5(3, 1, 0, 1, 3, 4, 2)
		block_4x4_4x5(3, 2, 1, 0, 3, 4, 2)
		block_4x4_4x5(3, 3, 1, 1, 3, 4, 2)

		//c[4 * 3 + 0] += temp2[0 * 5 + 0] * kernels_L3_3[16]
		//	+ temp2[0 * 5 + 1] * kernels_L3_3[17]
		//	+ temp2[0 * 5 + 2] * kernels_L3_3[18]
		//	+ temp2[0 * 5 + 3] * kernels_L3_3[19];

		//c[4 * 3 + 1] += temp2[0 * 5 + 1] * kernels_L3_3[16]
		//	+ temp2[0 * 5 + 2] * kernels_L3_3[17]
		//	+ temp2[0 * 5 + 3] * kernels_L3_3[18]
		//	+ temp2[0 * 5 + 4] * kernels_L3_3[19];

		//c[4 * 3 + 2] += temp2[1 * 5 + 0] * kernels_L3_3[16]
		//	+ temp2[1 * 5 + 1] * kernels_L3_3[17]
		//	+ temp2[1 * 5 + 2] * kernels_L3_3[18]
		//	+ temp2[1 * 5 + 3] * kernels_L3_3[19];

		//c[4 * 3 + 3] += temp2[1 * 5 + 1] * kernels_L3_3[16]
		//	+ temp2[1 * 5 + 2] * kernels_L3_3[17]
		//	+ temp2[1 * 5 + 3] * kernels_L3_3[18]
		//	+ temp2[1 * 5 + 4] * kernels_L3_3[19];

		b_relu_bn(3, 3, 3);

		dst_3 += dst_offset;
		*(dst_3 + 0) = res_0;
		*(dst_3 + 1) = res_1;
		dst_3 += dst_step;
		*(dst_3 + 0) = res_2;
		*(dst_3 + 1) = res_3;
	}
}
__kernel void add_2_3_conv_4x5x4_1_cl(
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
									/*15*/__constant float* kernels_L3_7,
									/*16*/__constant float* conv_b_L3,
									/*17*/__constant float* lrelu_w1_L3,
									/*18*/__constant float* lrelu_w2_L3,
									/*19*/__constant float* bn_b_L3
									)
{
	const int i = 2 * get_global_id(0);
	const int j = 2 * get_global_id(1);

	if (i + 1 >= src_cols - 3 || j + 1 >= src_rows - 4) return;

	const int src_offset = j * src_step + i;
	const int dst_offset = j * dst_step + i;

	float temp1[10];

	src_L3_3 += src_offset;
	src_L3_4 += src_offset;
	temp1[0 * 5 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[0 * 5 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[0 * 5 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[0 * 5 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[0 * 5 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);

	src_L3_3 += src_step;
	src_L3_4 += src_step;
	temp1[1 * 5 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[1 * 5 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[1 * 5 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[1 * 5 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[1 * 5 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);

	float temp2[10];

	src_L3_5 += src_offset;
	temp2[0 * 5 + 0] = temp1[0 * 5 + 0] + *(src_L3_5 + 0);
	temp2[0 * 5 + 1] = temp1[0 * 5 + 1] + *(src_L3_5 + 1);
	temp2[0 * 5 + 2] = temp1[0 * 5 + 2] + *(src_L3_5 + 2);
	temp2[0 * 5 + 3] = temp1[0 * 5 + 3] + *(src_L3_5 + 3);
	temp2[0 * 5 + 4] = temp1[0 * 5 + 4] + *(src_L3_5 + 4);

	src_L3_5 += src_step;
	temp2[1 * 5 + 0] = temp1[1 * 5 + 0] + *(src_L3_5 + 0);
	temp2[1 * 5 + 1] = temp1[1 * 5 + 1] + *(src_L3_5 + 1);
	temp2[1 * 5 + 2] = temp1[1 * 5 + 2] + *(src_L3_5 + 2);
	temp2[1 * 5 + 3] = temp1[1 * 5 + 3] + *(src_L3_5 + 3);
	temp2[1 * 5 + 4] = temp1[1 * 5 + 4] + *(src_L3_5 + 4);


	src_L3_2 += src_offset;
	temp1[0 * 5 + 0] += *(src_L3_2 + 0);
	temp1[0 * 5 + 1] += *(src_L3_2 + 1);
	temp1[0 * 5 + 2] += *(src_L3_2 + 2);
	temp1[0 * 5 + 3] += *(src_L3_2 + 3);
	temp1[0 * 5 + 4] += *(src_L3_2 + 4);
	
	src_L3_2 += src_step;
	temp1[1 * 5 + 0] += *(src_L3_2 + 0);
	temp1[1 * 5 + 1] += *(src_L3_2 + 1);
	temp1[1 * 5 + 2] += *(src_L3_2 + 2);
	temp1[1 * 5 + 3] += *(src_L3_2 + 3);
	temp1[1 * 5 + 4] += *(src_L3_2 + 4);
	
	float c[4 * 4];
	for (int i = 0; i < 16; i += 4)
	{
		c[i + 0] = 0.f;
		c[i + 1] = 0.f;
		c[i + 2] = 0.f;
		c[i + 3] = 0.f;
	}

	{
		block_4x4_4x5(0, 0, 0, 0, 4, 0, 1)
		block_4x4_4x5(0, 1, 0, 1, 4, 0, 1)
		block_4x4_4x5(0, 2, 1, 0, 4, 0, 1)
		block_4x4_4x5(0, 3, 1, 1, 4, 0, 1)
	}
	{
		block_4x4_4x5(1, 0, 0, 0, 5, 0, 1)
		block_4x4_4x5(1, 1, 0, 1, 5, 0, 1)
		block_4x4_4x5(1, 2, 1, 0, 5, 0, 1)
		block_4x4_4x5(1, 3, 1, 1, 5, 0, 1)
	}
	{
		block_4x4_4x5(2, 0, 0, 0, 6, 0, 2)
		block_4x4_4x5(2, 1, 0, 1, 6, 0, 2)
		block_4x4_4x5(2, 2, 1, 0, 6, 0, 2)
		block_4x4_4x5(2, 3, 1, 1, 6, 0, 2)
	}
	{
		block_4x4_4x5(3, 0, 0, 0, 7, 0, 2)
		block_4x4_4x5(3, 1, 0, 1, 7, 0, 2)
		block_4x4_4x5(3, 2, 1, 0, 7, 0, 2)
		block_4x4_4x5(3, 3, 1, 1, 7, 0, 2)
	}

	src_L3_3 += src_step;
	src_L3_4 += src_step;
	temp1[0 * 5 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[0 * 5 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[0 * 5 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[0 * 5 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[0 * 5 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);

	src_L3_5 += src_step;
	temp2[0 * 5 + 0] = temp1[0 * 5 + 0] + *(src_L3_5 + 0);
	temp2[0 * 5 + 1] = temp1[0 * 5 + 1] + *(src_L3_5 + 1);
	temp2[0 * 5 + 2] = temp1[0 * 5 + 2] + *(src_L3_5 + 2);
	temp2[0 * 5 + 3] = temp1[0 * 5 + 3] + *(src_L3_5 + 3);
	temp2[0 * 5 + 4] = temp1[0 * 5 + 4] + *(src_L3_5 + 4);

	src_L3_2 += src_step;
	temp1[0 * 5 + 0] += *(src_L3_2 + 0);
	temp1[0 * 5 + 1] += *(src_L3_2 + 1);
	temp1[0 * 5 + 2] += *(src_L3_2 + 2);
	temp1[0 * 5 + 3] += *(src_L3_2 + 3);
	temp1[0 * 5 + 4] += *(src_L3_2 + 4);

	{
		block_4x4_4x5(0, 0, 1, 0, 4, 1, 1)
		block_4x4_4x5(0, 1, 1, 1, 4, 1, 1)
		block_4x4_4x5(0, 2, 0, 0, 4, 1, 1)
		block_4x4_4x5(0, 3, 0, 1, 4, 1, 1)
	}
	{
		block_4x4_4x5(1, 0, 1, 0, 5, 1, 1)
		block_4x4_4x5(1, 1, 1, 1, 5, 1, 1)
		block_4x4_4x5(1, 2, 0, 0, 5, 1, 1)
		block_4x4_4x5(1, 3, 0, 1, 5, 1, 1)
	}
	{
		block_4x4_4x5(2, 0, 1, 0, 6, 1, 2)
		block_4x4_4x5(2, 1, 1, 1, 6, 1, 2)
		block_4x4_4x5(2, 2, 0, 0, 6, 1, 2)
		block_4x4_4x5(2, 3, 0, 1, 6, 1, 2)
	}
	{
		block_4x4_4x5(3, 0, 1, 0, 7, 1, 2)
		block_4x4_4x5(3, 1, 1, 1, 7, 1, 2)
		block_4x4_4x5(3, 2, 0, 0, 7, 1, 2)
		block_4x4_4x5(3, 3, 0, 1, 7, 1, 2)
	}

	src_L3_3 += src_step;
	src_L3_4 += src_step;
	temp1[1 * 5 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[1 * 5 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[1 * 5 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[1 * 5 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[1 * 5 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);

	src_L3_5 += src_step;
	temp2[1 * 5 + 0] = temp1[1 * 5 + 0] + *(src_L3_5 + 0);
	temp2[1 * 5 + 1] = temp1[1 * 5 + 1] + *(src_L3_5 + 1);
	temp2[1 * 5 + 2] = temp1[1 * 5 + 2] + *(src_L3_5 + 2);
	temp2[1 * 5 + 3] = temp1[1 * 5 + 3] + *(src_L3_5 + 3);
	temp2[1 * 5 + 4] = temp1[1 * 5 + 4] + *(src_L3_5 + 4);

	src_L3_2 += src_step;
	temp1[1 * 5 + 0] += *(src_L3_2 + 0);
	temp1[1 * 5 + 1] += *(src_L3_2 + 1);
	temp1[1 * 5 + 2] += *(src_L3_2 + 2);
	temp1[1 * 5 + 3] += *(src_L3_2 + 3);
	temp1[1 * 5 + 4] += *(src_L3_2 + 4);
	temp1[1 * 5 + 5] += *(src_L3_2 + 5);

	{
		block_4x4_4x5(0, 0, 0, 0, 4, 2, 1)
		block_4x4_4x5(0, 1, 0, 1, 4, 2, 1)
		block_4x4_4x5(0, 2, 1, 0, 4, 2, 1)
		block_4x4_4x5(0, 3, 1, 1, 4, 2, 1)
	}
	{
		block_4x4_4x5(1, 0, 0, 0, 5, 2, 1)
		block_4x4_4x5(1, 1, 0, 1, 5, 2, 1)
		block_4x4_4x5(1, 2, 1, 0, 5, 2, 1)
		block_4x4_4x5(1, 3, 1, 1, 5, 2, 1)
	}
	{
		block_4x4_4x5(2, 0, 0, 0, 6, 2, 2)
		block_4x4_4x5(2, 1, 0, 1, 6, 2, 2)
		block_4x4_4x5(2, 2, 1, 0, 6, 2, 2)
		block_4x4_4x5(2, 3, 1, 1, 6, 2, 2)
	}
	{
		block_4x4_4x5(3, 0, 0, 0, 7, 2, 2)
		block_4x4_4x5(3, 1, 0, 1, 7, 2, 2)
		block_4x4_4x5(3, 2, 1, 0, 7, 2, 2)
		block_4x4_4x5(3, 3, 1, 1, 7, 2, 2)
	}

	src_L3_3 += src_step;
	src_L3_4 += src_step;
	temp1[0 * 5 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[0 * 5 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[0 * 5 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[0 * 5 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[0 * 5 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);

	src_L3_5 += src_step;
	temp2[0 * 5 + 0] = temp1[0 * 5 + 0] + *(src_L3_5 + 0);
	temp2[0 * 5 + 1] = temp1[0 * 5 + 1] + *(src_L3_5 + 1);
	temp2[0 * 5 + 2] = temp1[0 * 5 + 2] + *(src_L3_5 + 2);
	temp2[0 * 5 + 3] = temp1[0 * 5 + 3] + *(src_L3_5 + 3);
	temp2[0 * 5 + 4] = temp1[0 * 5 + 4] + *(src_L3_5 + 4);

	src_L3_2 += src_step;
	temp1[0 * 5 + 0] += *(src_L3_2 + 0);
	temp1[0 * 5 + 1] += *(src_L3_2 + 1);
	temp1[0 * 5 + 2] += *(src_L3_2 + 2);
	temp1[0 * 5 + 3] += *(src_L3_2 + 3);
	temp1[0 * 5 + 4] += *(src_L3_2 + 4);

	{
		block_4x4_4x5(0, 0, 1, 0, 4, 3, 1)
		block_4x4_4x5(0, 1, 1, 1, 4, 3, 1)
		block_4x4_4x5(0, 2, 0, 0, 4, 3, 1)
		block_4x4_4x5(0, 3, 0, 1, 4, 3, 1)
	}
	{
		block_4x4_4x5(1, 0, 1, 0, 5, 3, 1)
		block_4x4_4x5(1, 1, 1, 1, 5, 3, 1)
		block_4x4_4x5(1, 2, 0, 0, 5, 3, 1)
		block_4x4_4x5(1, 3, 0, 1, 5, 3, 1)
	}
	{
		block_4x4_4x5(2, 0, 1, 0, 6, 3, 2)
		block_4x4_4x5(2, 1, 1, 1, 6, 3, 2)
		block_4x4_4x5(2, 2, 0, 0, 6, 3, 2)
		block_4x4_4x5(2, 3, 0, 1, 6, 3, 2)
	}
	{
		block_4x4_4x5(3, 0, 1, 0, 7, 3, 2)
		block_4x4_4x5(3, 1, 1, 1, 7, 3, 2)
		block_4x4_4x5(3, 2, 0, 0, 7, 3, 2)
		block_4x4_4x5(3, 3, 0, 1, 7, 3, 2)
	}

	src_L3_3 += src_step;
	src_L3_4 += src_step;
	temp1[1 * 5 + 0] = *(src_L3_3 + 0) + *(src_L3_4 + 0);
	temp1[1 * 5 + 1] = *(src_L3_3 + 1) + *(src_L3_4 + 1);
	temp1[1 * 5 + 2] = *(src_L3_3 + 2) + *(src_L3_4 + 2);
	temp1[1 * 5 + 3] = *(src_L3_3 + 3) + *(src_L3_4 + 3);
	temp1[1 * 5 + 4] = *(src_L3_3 + 4) + *(src_L3_4 + 4);

	src_L3_5 += src_step;
	temp2[1 * 5 + 0] = temp1[1 * 5 + 0] + *(src_L3_5 + 0);
	temp2[1 * 5 + 1] = temp1[1 * 5 + 1] + *(src_L3_5 + 1);
	temp2[1 * 5 + 2] = temp1[1 * 5 + 2] + *(src_L3_5 + 2);
	temp2[1 * 5 + 3] = temp1[1 * 5 + 3] + *(src_L3_5 + 3);
	temp2[1 * 5 + 4] = temp1[1 * 5 + 4] + *(src_L3_5 + 4);

	src_L3_2 += src_step;
	temp1[1 * 5 + 0] += *(src_L3_2 + 0);
	temp1[1 * 5 + 1] += *(src_L3_2 + 1);
	temp1[1 * 5 + 2] += *(src_L3_2 + 2);
	temp1[1 * 5 + 3] += *(src_L3_2 + 3);
	temp1[1 * 5 + 4] += *(src_L3_2 + 4);

	{
		block_4x4_4x5(0, 0, 0, 0, 4, 4, 1)
		block_4x4_4x5(0, 1, 0, 1, 4, 4, 1)
		block_4x4_4x5(0, 2, 1, 0, 4, 4, 1)
		block_4x4_4x5(0, 3, 1, 1, 4, 4, 1)

		b_relu_bn(0, 3, 4);

		dst_4 += dst_offset;
		*(dst_4 + 0) = res_0;
		*(dst_4 + 1) = res_1;
		dst_4 += dst_step;
		*(dst_4 + 0) = res_2;
		*(dst_4 + 1) = res_3;
	}
	{
		block_4x4_4x5(1, 0, 0, 0, 5, 4, 1)
		block_4x4_4x5(1, 1, 0, 1, 5, 4, 1)
		block_4x4_4x5(1, 2, 1, 0, 5, 4, 1)
		block_4x4_4x5(1, 3, 1, 1, 5, 4, 1)
			
		b_relu_bn(1, 3, 5);

		dst_5 += dst_offset;
		*(dst_5 + 0) = res_0;
		*(dst_5 + 1) = res_1;
		dst_5 += dst_step;
		*(dst_5 + 0) = res_2;
		*(dst_5 + 1) = res_3;
	}
	{
		block_4x4_4x5(2, 0, 0, 0, 6, 4, 2)
		block_4x4_4x5(2, 1, 0, 1, 6, 4, 2)
		block_4x4_4x5(2, 2, 1, 0, 6, 4, 2)
		block_4x4_4x5(2, 3, 1, 1, 6, 4, 2)
			
		b_relu_bn(2, 3, 6);

		dst_6 += dst_offset;
		*(dst_6 + 0) = res_0;
		*(dst_6 + 1) = res_1;
		dst_6 += dst_step;
		*(dst_6 + 0) = res_2;
		*(dst_6 + 1) = res_3;
	}
	{
		block_4x4_4x5(3, 0, 0, 0, 7, 4, 2)
		block_4x4_4x5(3, 1, 0, 1, 7, 4, 2)
		block_4x4_4x5(3, 2, 1, 0, 7, 4, 2)
		block_4x4_4x5(3, 3, 1, 1, 7, 4, 2)
			
		b_relu_bn(3, 3, 7);

		dst_7 += dst_offset;
		*(dst_7 + 0) = res_0;
		*(dst_7 + 1) = res_1;
		dst_7 += dst_step;
		*(dst_7 + 0) = res_2;
		*(dst_7 + 1) = res_3;
	}
}

__kernel void lrelu_bn_add4_tanh_add52_tanh_texmem_cl(
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
													/*23*/__constant float* hl_w,
													/*24*/__constant float* hl_b,
													/*25*/__constant float* hl_tanh_w,
													/*26*/__constant float* hl_bn_w,
													/*27*/__constant float* hl_bn_b,
													/*28*/__constant float* ol_w,
													/*29*/__constant float* ol_b,
													/*30*/__constant float* ol_tanh_w,
													/*31*/__constant float* ol_scale,
													/*32*/int index_output
													)
{
	const int surf_hl_size = 52;
	const int surf_hl_connect = 4;
	const int surf_hl_scale = 4;

	const int i = get_global_id(0);
	const int j = get_global_id(1);

	if (i >= dst_cols || j >= dst_rows) return;

	const int src_offset = j * src_step + i;
	const int dst_offset = j * dst_step + i;

	float res = ol_b[0];
	hl_layer(0, 1, 2, 3, 4)
	hl_layer(1, 2, 3, 4, 5)
	hl_layer(2, 3, 4, 5, 6)
	hl_layer(3, 4, 5, 6, 7)
	hl_layer(4, 5, 6, 7, 8)
	hl_layer(5, 6, 7, 8, 9)
	hl_layer(6, 7, 8, 9, 10)
	hl_layer(7, 8, 9, 10, 11)
	hl_layer(8, 9, 10, 11, 12)
	hl_layer(9, 10, 11, 12, 13)
	hl_layer(10, 11, 12, 13, 14)
	hl_layer(11, 12, 13, 14, 15)
	hl_layer(12, 13, 14, 15, 16)

	res *= ol_tanh_w[0];
	res = tanhf(res);
	*(dst + dst_offset) = ol_scale[0] * res;
}