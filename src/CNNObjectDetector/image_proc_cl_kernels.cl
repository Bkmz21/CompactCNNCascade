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


__kernel void Img8uToImg32f_cl(									
							/*0*/__global float* dst,
							/*1*/int dst_step,
							/*2*/__global unsigned char* src,
							/*3*/int src_cols,
							/*4*/int src_rows,
							/*5*/int src_step
							)
{
	int i = 2 * get_global_id(0);
	int j = 2 * get_global_id(1);

	if (i + 1 >= src_cols) i--;
	if (j + 1 >= src_rows) j--;
	if (i + 1 >= src_cols || j + 1 >= src_rows) return;

	const int src_offset = j * src_step + i;
	const int dst_offset = j * dst_step + i;

	src += src_offset;
	dst += dst_offset;
	*(dst + 0) = (float)*(src + 0);
	*(dst + 1) = (float)*(src + 1);

	src += src_step;
	dst += dst_step;
	*(dst + 0) = (float)*(src + 0);
	*(dst + 1) = (float)*(src + 1);
}

__kernel void Img8uBGRToImg32fGRAY_cl(
									/*0*/__global float* dst,
									/*1*/int dst_step,
									/*2*/__global unsigned char* src,
									/*3*/int src_cols,
									/*4*/int src_rows,
									/*5*/int src_step
									)
{
	int i = 2 * get_global_id(0);
	int j = 2 * get_global_id(1);

	if (i + 1 >= src_cols) i--;
	if (j + 1 >= src_rows) j--;
	if (i + 1 >= src_cols || j + 1 >= src_rows) return;

	const int src_offset = j * src_step + 3 * i;
	const int dst_offset = j * dst_step + i;

	src += src_offset;
	dst += dst_offset;
	float B = (float)*(src + 0);
	float G = (float)*(src + 1);
	float R = (float)*(src + 2);
	*(dst + 0) = B * 0.114f + G * 0.587f + R * 0.299f;

	B = (float)*(src + 3 + 0);
	G = (float)*(src + 3 + 1);
	R = (float)*(src + 3 + 2);
	*(dst + 1) = B * 0.114f + G * 0.587f + R * 0.299f;

	src += src_step;
	dst += dst_step;
	B = (float)*(src + 0);
	G = (float)*(src + 1);
	R = (float)*(src + 2);
	*(dst + 0) = B * 0.114f + G * 0.587f + R * 0.299f;

	B = (float)*(src + 3 + 0);
	G = (float)*(src + 3 + 1);
	R = (float)*(src + 3 + 2);
	*(dst + 1) = B * 0.114f + G * 0.587f + R * 0.299f;
}

__kernel void Img8uBGRAToImg32fGRAY_cl(
									/*0*/__global float* dst,
									/*1*/int dst_step,
									/*2*/__global unsigned char* src,
									/*3*/int src_cols,
									/*4*/int src_rows,
									/*5*/int src_step
									)
{
	int i = 2 * get_global_id(0);
	int j = 2 * get_global_id(1);

	if (i + 1 >= src_cols) i--;
	if (j + 1 >= src_rows) j--;
	if (i + 1 >= src_cols || j + 1 >= src_rows) return;

	const int src_offset = j * src_step + 4 * i;
	const int dst_offset = j * dst_step + i;

	src += src_offset;
	dst += dst_offset;
	float B = (float)*(src + 0);
	float G = (float)*(src + 1);
	float R = (float)*(src + 2);
	*(dst + 0) = B * 0.114f + G * 0.587f + R * 0.299f;

	B = (float)*(src + 4 + 0);
	G = (float)*(src + 4 + 1);
	R = (float)*(src + 4 + 2);
	*(dst + 1) = B * 0.114f + G * 0.587f + R * 0.299f;

	src += src_step;
	dst += dst_step;
	B = (float)*(src + 0);
	G = (float)*(src + 1);
	R = (float)*(src + 2);
	*(dst + 0) = B * 0.114f + G * 0.587f + R * 0.299f;

	B = (float)*(src + 4 + 0);
	G = (float)*(src + 4 + 1);
	R = (float)*(src + 4 + 2);
	*(dst + 1) = B * 0.114f + G * 0.587f + R * 0.299f;
}

__kernel void ImgColBlur_cl(									
							/*0*/__global float* dst,
							/*1*/int dst_step,
							/*2*/__global float* src,
							/*3*/int src_step,
							/*4*/float ck0,
							/*5*/float ck1,
							/*6*/float ck2
							)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);

	const int src_offset = j * src_step + i;
	const int dst_offset = j * dst_step + i;

	src += src_offset;
	dst += dst_offset;

	*dst = *src * ck0 +* (src + src_step) * ck1 +* (src + 2 * src_step) * ck2;
}

__kernel void ImgRowBlur_cl(									
							/*0*/__global float* dst,
							/*1*/int dst_step,
							/*2*/__global float* src,
							/*3*/int src_step,
							/*4*/float rk0,
							/*5*/float rk1,
							/*6*/float rk2
							)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	if (i + 2 >= src_step) i = src_step - 3;

	const int src_offset = j * src_step + i;
	const int dst_offset = j * dst_step + i;

	src += src_offset;
	dst += dst_offset;

	*dst = *src++ * rk0 +* src++ * rk1 +* src * rk2;
}