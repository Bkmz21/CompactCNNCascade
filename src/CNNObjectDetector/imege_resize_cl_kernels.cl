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


__kernel void NearestNeighborInterpolation_cl(
											/*0*/__global float* dst,
											/*1*/int dst_step,
											/*2*/int _dst_offset,
											/*3*/__global float* src,
											/*4*/int src_cols,
											/*5*/int src_rows,
											/*6*/int src_step,
											/*7*/unsigned int sclx,
											/*8*/unsigned int scly,
											/*9*/unsigned int QUANT_BIT
											)
{
	const unsigned int ix = get_global_id(0);
	const unsigned int iy = get_global_id(1);

	// make LUT for y-axis
	const unsigned int x = ix * sclx;
	const unsigned int px = (x >> QUANT_BIT);

	const unsigned int y = iy * scly;
	const unsigned int py = (y >> QUANT_BIT);

	if (px >= src_cols || py >= src_rows) return;

	const unsigned int src_offset = py * src_step + px;
	const unsigned int dst_offset = _dst_offset + iy * dst_step + ix;

	*(dst + dst_offset) = *(src + src_offset);
}

__kernel void BilinearInterpolation_cl(											
									/*0*/__global float* dst,
									/*1*/int dst_step,
									/*2*/int _dst_offset,
									/*3*/__global float* src,
									/*4*/int src_cols,
									/*5*/int src_rows,
									/*6*/int src_step,
									/*7*/unsigned int sclx,
									/*8*/unsigned int scly,
									/*9*/unsigned int QUANT_BIT,
									/*10*/float QUANT_BIT_f32,
									/*11*/float QUANT_BIT2_f32
									)
{
	const unsigned int ix = get_global_id(0);
	const unsigned int iy = get_global_id(1);

	// make LUT for y-axis
	const unsigned int x = ix * sclx;
	unsigned int px = (x >> QUANT_BIT);
	if (px + 1 >= src_cols) px--;

	const unsigned int y = iy * scly;
	unsigned int py = (y >> QUANT_BIT);
	if (py + 1 >= src_rows) py--;

	if (px + 1 >= src_cols || py + 1 >= src_rows) return;

	const unsigned int src_offset = py * src_step + px;
	const unsigned int dst_offset = _dst_offset + iy * dst_step + ix;

	const float fx = (float)(x - (px << QUANT_BIT));
	const float cx = QUANT_BIT_f32 - fx;

	const float fy = (float)(y - (py << QUANT_BIT));
	const float cy = QUANT_BIT_f32 - fy;

	// four neighbor pixels
	src += src_offset;
	const float p0 = *(src + 0);
	const float p1 = *(src + 1);

	src += src_step;
	const float p2 = *(src + 0);
	const float p3 = *(src + 1);

	// Calculate the weighted sum of pixels
	const float outv = (p0 * cx + p1 * fx) * cy + (p2 * cx + p3 * fx) * fy;
	*(dst + dst_offset) = outv * QUANT_BIT2_f32;
}
