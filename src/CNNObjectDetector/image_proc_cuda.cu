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


#include "image_proc_cuda.h"


//================================================================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_CUDA

	namespace CUDA
	{
		struct SurfRef
		{
			float* surf;
			int cols;
			int rows;
			int step;
		};

		typedef texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> Texture;
		Texture tex;
		cudaChannelFormatDesc ip_tex_channelDesc;
	
		__global__ void Img8uToImg32f_cu(SurfRef surf_ref_dst, SurfRef surf_ref_src)
		{
			const int i = blockIdx.x * (2 * 16);
			const int j = blockIdx.y * (2 * 16);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int tx = i + z1;
			const int ty = j + 4 * ry;
			uchar_* ptr_src = (uchar_*)surf_ref_src.surf + ty * surf_ref_src.step + tx;
			float* ptr_dst = surf_ref_dst.surf + ty * surf_ref_dst.step + tx;

			if (tx < surf_ref_src.cols)
			{
				if (ty < surf_ref_src.rows)
				{
					*ptr_dst = (float)*ptr_src;

					if (ty + 1 < surf_ref_src.rows)
					{
						ptr_src += surf_ref_src.step;
						ptr_dst += surf_ref_dst.step;
						*ptr_dst = (float)*ptr_src;

						if (ty + 2 < surf_ref_src.rows)
						{
							ptr_src += surf_ref_src.step;
							ptr_dst += surf_ref_dst.step;
							*ptr_dst = (float)*ptr_src;

							if (ty + 3 < surf_ref_src.rows)
							{
								ptr_src += surf_ref_src.step;
								ptr_dst += surf_ref_dst.step;
								*ptr_dst = (float)*ptr_src;
							}
						}
					}
				}
			}
		}
		__global__ void Img8uBGRToImg32fGRAY_cu(SurfRef surf_ref_dst, SurfRef surf_ref_src)
		{
			const int i = blockIdx.x * (2 * 16);
			const int j = blockIdx.y * (2 * 16);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int tx = i + z1;
			const int ty = j + 4 * ry;
			uchar_* ptr_src = (uchar_*)surf_ref_src.surf + ty * surf_ref_src.step + 3 * tx;
			float* ptr_dst = surf_ref_dst.surf + ty * surf_ref_dst.step + tx;

			if (tx < surf_ref_src.cols)
			{
				if (ty < surf_ref_src.rows)
				{
					const float B = (float)*ptr_src;
					const float  G = (float)*(ptr_src + 1);
					const float  R = (float)*(ptr_src + 2);
					const float r1 = B * 0.114f + G * 0.587f + R * 0.299f;
					*ptr_dst = r1;

					if (ty + 1 < surf_ref_src.rows)
					{
						ptr_src += surf_ref_src.step;
						ptr_dst += surf_ref_dst.step;
						const float B = (float)*ptr_src;
						const float G = (float)*(ptr_src + 1);
						const float R = (float)*(ptr_src + 2);
						const float r2 = B * 0.114f + G * 0.587f + R * 0.299f;
						*ptr_dst = r2;

						if (ty + 2 < surf_ref_src.rows)
						{
							ptr_src += surf_ref_src.step;
							ptr_dst += surf_ref_dst.step;
							const float B = (float)*ptr_src;
							const float G = (float)*(ptr_src + 1);
							const float R = (float)*(ptr_src + 2);
							const float r3 = B * 0.114f + G * 0.587f + R * 0.299f;
							*ptr_dst = r3;

							if (ty + 3 < surf_ref_src.rows)
							{
								ptr_src += surf_ref_src.step;
								ptr_dst += surf_ref_dst.step;
								const float B = (float)*ptr_src;
								const float G = (float)*(ptr_src + 1);
								const float R = (float)*(ptr_src + 2);
								const float r4 = B * 0.114f + G * 0.587f + R * 0.299f;
								*ptr_dst = r4;
							}
						}
					}
				}
			}
		}
		__global__ void Img8uBGRAToImg32fGRAY_cu(SurfRef surf_ref_dst, SurfRef surf_ref_src)
		{
			const int i = blockIdx.x * (2 * 16);
			const int j = blockIdx.y * (2 * 16);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int tx = i + z1;
			const int ty = j + 4 * ry;
			uchar_* ptr_src = (uchar_*)surf_ref_src.surf + ty * surf_ref_src.step + 4 * tx;
			float* ptr_dst = surf_ref_dst.surf + ty * surf_ref_dst.step + tx;

			if (tx < surf_ref_src.cols)
			{
				if (ty < surf_ref_src.rows)
				{
					const float B = (float)*ptr_src;
					const float  G = (float)*(ptr_src + 1);
					const float  R = (float)*(ptr_src + 2);
					const float r1 = B * 0.114f + G * 0.587f + R * 0.299f;
					*ptr_dst = r1;

					if (ty + 1 < surf_ref_src.rows)
					{
						ptr_src += surf_ref_src.step;
						ptr_dst += surf_ref_dst.step;
						const float B = (float)*ptr_src;
						const float G = (float)*(ptr_src + 1);
						const float R = (float)*(ptr_src + 2);
						const float r2 = B * 0.114f + G * 0.587f + R * 0.299f;
						*ptr_dst = r2;

						if (ty + 2 < surf_ref_src.rows)
						{
							ptr_src += surf_ref_src.step;
							ptr_dst += surf_ref_dst.step;
							const float B = (float)*ptr_src;
							const float G = (float)*(ptr_src + 1);
							const float R = (float)*(ptr_src + 2);
							const float r3 = B * 0.114f + G * 0.587f + R * 0.299f;
							*ptr_dst = r3;

							if (ty + 3 < surf_ref_src.rows)
							{
								ptr_src += surf_ref_src.step;
								ptr_dst += surf_ref_dst.step;
								const float B = (float)*ptr_src;
								const float G = (float)*(ptr_src + 1);
								const float R = (float)*(ptr_src + 2);
								const float r4 = B * 0.114f + G * 0.587f + R * 0.299f;
								*ptr_dst = r4;
							}
						}
					}
				}
			}
		}

		__global__ void Img8uToImg32f_tex_cu(SurfRef surf_ref_dst)
		{
			const int i = blockIdx.x * (2 * 16);
			const int j = blockIdx.y * (2 * 16);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int sx = i + z1;
			const int sy = j + 4 * ry;
			const float tx = sx + 0.5f;
			const float ty = sy + 0.5f;

			float* ptr_dst = surf_ref_dst.surf + sy * surf_ref_dst.step + sx;
			*ptr_dst = 255.f * tex2D(tex, tx, ty);

			ptr_dst += surf_ref_dst.step;
			*ptr_dst = 255.f * tex2D(tex, tx, ty + 1.f);

			ptr_dst += surf_ref_dst.step;
			*ptr_dst = 255.f * tex2D(tex, tx, ty + 2.f);

			ptr_dst += surf_ref_dst.step;
			*ptr_dst = 255.f * tex2D(tex, tx, ty + 3.f);
		}
		__global__ void Img8uBGRToImg32fGRAY_tex_cu(SurfRef surf_ref_dst)
		{
			const int i = blockIdx.x * (2 * 16);
			const int j = blockIdx.y * (2 * 16);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int sx = i + z1;
			const int sy = j + 4 * ry;
			const float tx = 3 * sx + 0.5f;
			const float ty = sy + 0.5f;

			float* ptr_dst = surf_ref_dst.surf + sy * surf_ref_dst.step + sx;

			float B = tex2D(tex, tx, ty);
			float G = tex2D(tex, tx + 1.f, ty);
			float R = tex2D(tex, tx + 2.f, ty);
			*ptr_dst = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			ptr_dst += surf_ref_dst.step;
			B = tex2D(tex, tx, ty + 1.f);
			G = tex2D(tex, tx + 1.f, ty + 1.f);
			R = tex2D(tex, tx + 2.f, ty + 1.f);
			*ptr_dst = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			ptr_dst += surf_ref_dst.step;
			B = tex2D(tex, tx, ty + 2.f);
			G = tex2D(tex, tx + 1.f, ty + 2.f);
			R = tex2D(tex, tx + 2.f, ty + 2.f);
			*ptr_dst = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			ptr_dst += surf_ref_dst.step;
			B = tex2D(tex, tx, ty + 3.f);
			G = tex2D(tex, tx + 1.f, ty + 3.f);
			R = tex2D(tex, tx + 2.f, ty + 3.f);
			*ptr_dst = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;
		}
		__global__ void Img8uBGRAToImg32fGRAY_tex_cu(SurfRef surf_ref_dst)
		{
			const int i = blockIdx.x * (2 * 16);
			const int j = blockIdx.y * (2 * 16);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int sx = i + z1;
			const int sy = j + 4 * ry;
			const float tx = 4 * sx + 0.5f;
			const float ty = sy + 0.5f;

			float* ptr_dst = surf_ref_dst.surf + sy * surf_ref_dst.step + sx;

			float B = tex2D(tex, tx, ty);
			float G = tex2D(tex, tx + 1.f, ty);
			float R = tex2D(tex, tx + 2.f, ty);
			*ptr_dst = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			ptr_dst += surf_ref_dst.step;
			B = tex2D(tex, tx, ty + 1.f);
			G = tex2D(tex, tx + 1.f, ty + 1.f);
			R = tex2D(tex, tx + 2.f, ty + 1.f);
			*ptr_dst = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			ptr_dst += surf_ref_dst.step;
			B = tex2D(tex, tx, ty + 2.f);
			G = tex2D(tex, tx + 1.f, ty + 2.f);
			R = tex2D(tex, tx + 2.f, ty + 2.f);
			*ptr_dst = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			ptr_dst += surf_ref_dst.step;
			B = tex2D(tex, tx, ty + 3.f);
			G = tex2D(tex, tx + 1.f, ty + 3.f);
			R = tex2D(tex, tx + 2.f, ty + 3.f);
			*ptr_dst = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;
		}

		__global__ void Img8uToImg32f_blur_cu(SurfRef surf_ref_dst, SurfRef surf_ref_src, const float ck0, const float ck1, const float ck2, const float rk0, const float rk1, const float rk2)
		{
			__shared__ float temp1[(2 * 16) * (2 * 16)];
			__shared__ float temp2[(2 * 16) * (2 * 16)];

			const int i = blockIdx.x * (2 * 16 - 3 + 1);
			const int j = blockIdx.y * (2 * 16 - 3 + 1);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int z2 = (4 * ry) * (2 * 16) + z1;

			const int tx = i + z1;
			const int ty = j + 4 * ry;
			uchar_* ptr_src = (uchar_*)surf_ref_src.surf + ty * surf_ref_src.step + tx;

			if (tx < surf_ref_src.cols)
			{
				if (ty < surf_ref_src.rows)
				{
					temp1[z2 + 0 * (2 * 16)] = (float)*ptr_src;

					if (ty + 1 < surf_ref_src.rows)
					{
						ptr_src += surf_ref_src.step;
						temp1[z2 + 1 * (2 * 16)] = (float)*ptr_src;

						if (ty + 2 < surf_ref_src.rows)
						{
							ptr_src += surf_ref_src.step;
							temp1[z2 + 2 * (2 * 16)] = (float)*ptr_src;

							if (ty + 3 < surf_ref_src.rows)
							{
								ptr_src += surf_ref_src.step;
								temp1[z2 + 3 * (2 * 16)] = (float)*ptr_src;
							}
						}
					}
				}
			}

			__syncthreads();

			const int r = 2 * trdX + (2 * 16) * 2 * trdY;
			const float* ptr_temp1 = temp1 + r;
			float* ptr_temp2 = temp2 + r;

			if (trdY < 15)
			{
				*ptr_temp2 = *(ptr_temp1 + 0 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + 1) = *(ptr_temp1 + 0 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck2;

				*(ptr_temp2 + (2 * 16)) = *(ptr_temp1 + 1 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + (2 * 16) + 1) = *(ptr_temp1 + 1 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 1) * ck2;
			}

			__syncthreads();

			if (trdX < 15)
			{
				const float r1 = *(ptr_temp2 + 0 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk2;

				const float r2 = *(ptr_temp2 + 0 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 3) * rk2;

				const float r3 = *(ptr_temp2 + 1 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk2;

				const float r4 = *(ptr_temp2 + 1 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 3) * rk2;

				if (trdY < 15)
				{
					const int dx = i + 2 * trdX;
					const int dy = j + 2 * trdY;

					if (dy < surf_ref_dst.rows)
					{
						if (dx < surf_ref_dst.cols)
						{
							float* ptr_dst = surf_ref_dst.surf + dy * surf_ref_dst.step + dx;
							*ptr_dst = r1;

							if (dx + 1 < surf_ref_dst.cols)
							{
								*(ptr_dst + 1) = r2;
							}

							if (dy + 1 < surf_ref_dst.rows)
							{
								ptr_dst += surf_ref_dst.step;
								*ptr_dst = r3;

								if (dx + 1 < surf_ref_dst.cols)
								{
									*(ptr_dst + 1) = r4;
								}
							}
						}
					}
				}
			}

			__syncthreads();
		}
		__global__ void Img8uBGRToImg32fGRAY_blur_cu(SurfRef surf_ref_dst, SurfRef surf_ref_src, const float ck0, const float ck1, const float ck2, const float rk0, const float rk1, const float rk2)
		{
			__shared__ float temp1[(2 * 16) * (2 * 16)];
			__shared__ float temp2[(2 * 16) * (2 * 16)];

			const int i = blockIdx.x * (2 * 16 - 3 + 1);
			const int j = blockIdx.y * (2 * 16 - 3 + 1);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int z2 = (4 * ry) * (2 * 16) + z1;

			const int tx = i + z1;
			const int ty = j + 4 * ry;
			uchar_* ptr_src = (uchar_*)surf_ref_src.surf + ty * surf_ref_src.step + 3 * tx;

			if (tx < surf_ref_src.cols)
			{
				if (ty < surf_ref_src.rows)
				{
					const float B = (float)*ptr_src;
					const float G = (float)*(ptr_src + 1);
					const float R = (float)*(ptr_src + 2);
					temp1[z2 + 0 * (2 * 16)] = B * 0.114f + G * 0.587f + R * 0.299f;

					if (ty + 1 < surf_ref_src.rows)
					{
						ptr_src += surf_ref_src.step;
						const float B = (float)*ptr_src;
						const float G = (float)*(ptr_src + 1);
						const float R = (float)*(ptr_src + 2);
						temp1[z2 + 1 * (2 * 16)] = B * 0.114f + G * 0.587f + R * 0.299f;

						if (ty + 2 < surf_ref_src.rows)
						{
							ptr_src += surf_ref_src.step;
							const float B = (float)*ptr_src;
							const float G = (float)*(ptr_src + 1);
							const float R = (float)*(ptr_src + 2);
							temp1[z2 + 2 * (2 * 16)] = B * 0.114f + G * 0.587f + R * 0.299f;

							if (ty + 3 < surf_ref_src.rows)
							{
								ptr_src += surf_ref_src.step;
								const float B = (float)*ptr_src;
								const float G = (float)*(ptr_src + 1);
								const float R = (float)*(ptr_src + 2);
								temp1[z2 + 3 * (2 * 16)] = B * 0.114f + G * 0.587f + R * 0.299f;
							}
						}
					}
				}
			}

			__syncthreads();

			const int r = 2 * trdX + (2 * 16) * 2 * trdY;
			const float* ptr_temp1 = temp1 + r;
			float* ptr_temp2 = temp2 + r;

			if (trdY < 15)
			{
				*ptr_temp2 = *(ptr_temp1 + 0 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + 1) = *(ptr_temp1 + 0 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck2;

				*(ptr_temp2 + (2 * 16)) = *(ptr_temp1 + 1 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + (2 * 16) + 1) = *(ptr_temp1 + 1 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 1) * ck2;
			}

			__syncthreads();

			if (trdX < 15)
			{
				const float r1 = *(ptr_temp2 + 0 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk2;

				const float r2 = *(ptr_temp2 + 0 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 3) * rk2;

				const float r3 = *(ptr_temp2 + 1 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk2;

				const float r4 = *(ptr_temp2 + 1 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 3) * rk2;

				if (trdY < 15)
				{
					const int dx = i + 2 * trdX;
					const int dy = j + 2 * trdY;

					if (dy < surf_ref_dst.rows)
					{
						if (dx < surf_ref_dst.cols)
						{
							float* ptr_dst = surf_ref_dst.surf + dy * surf_ref_dst.step + dx;
							*ptr_dst = r1;

							if (dx + 1 < surf_ref_dst.cols)
							{
								*(ptr_dst + 1) = r2;
							}

							if (dy + 1 < surf_ref_dst.rows)
							{
								ptr_dst += surf_ref_dst.step;
								*ptr_dst = r3;

								if (dx + 1 < surf_ref_dst.cols)
								{
									*(ptr_dst + 1) = r4;
								}
							}
						}
					}
				}
			}

			__syncthreads();
		}
		__global__ void Img8uBGRAToImg32fGRAY_blur_cu(SurfRef surf_ref_dst, SurfRef surf_ref_src, const float ck0, const float ck1, const float ck2, const float rk0, const float rk1, const float rk2)
		{
			__shared__ float temp1[(2 * 16) * (2 * 16)];
			__shared__ float temp2[(2 * 16) * (2 * 16)];

			const int i = blockIdx.x * (2 * 16 - 3 + 1);
			const int j = blockIdx.y * (2 * 16 - 3 + 1);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int z2 = (4 * ry) * (2 * 16) + z1;

			const int tx = i + z1;
			const int ty = j + 4 * ry;
			uchar_* ptr_src = (uchar_*)surf_ref_src.surf + ty * surf_ref_src.step + 4 * tx;

			if (tx < surf_ref_src.cols)
			{
				if (ty < surf_ref_src.rows)
				{
					const float B = (float)*ptr_src;
					const float  G = (float)*(ptr_src + 1);
					const float  R = (float)*(ptr_src + 2);
					temp1[z2 + 0 * (2 * 16)] = B * 0.114f + G * 0.587f + R * 0.299f;

					if (ty + 1 < surf_ref_src.rows)
					{
						ptr_src += surf_ref_src.step;
						const float B = (float)*ptr_src;
						const float G = (float)*(ptr_src + 1);
						const float R = (float)*(ptr_src + 2);
						temp1[z2 + 1 * (2 * 16)] = B * 0.114f + G * 0.587f + R * 0.299f;

						if (ty + 2 < surf_ref_src.rows)
						{
							ptr_src += surf_ref_src.step;
							const float B = (float)*ptr_src;
							const float G = (float)*(ptr_src + 1);
							const float R = (float)*(ptr_src + 2);
							temp1[z2 + 2 * (2 * 16)] = B * 0.114f + G * 0.587f + R * 0.299f;

							if (ty + 3 < surf_ref_src.rows)
							{
								ptr_src += surf_ref_src.step;
								const float B = (float)*ptr_src;
								const float G = (float)*(ptr_src + 1);
								const float R = (float)*(ptr_src + 2);
								temp1[z2 + 3 * (2 * 16)] = B * 0.114f + G * 0.587f + R * 0.299f;
							}
						}
					}
				}
			}

			__syncthreads();

			const int r = 2 * trdX + (2 * 16) * 2 * trdY;
			const float* ptr_temp1 = temp1 + r;
			float* ptr_temp2 = temp2 + r;

			if (trdY < 15)
			{
				*ptr_temp2 = *(ptr_temp1 + 0 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + 1) = *(ptr_temp1 + 0 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck2;

				*(ptr_temp2 + (2 * 16)) = *(ptr_temp1 + 1 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + (2 * 16) + 1) = *(ptr_temp1 + 1 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 1) * ck2;
			}

			__syncthreads();

			if (trdX < 15)
			{
				const float r1 = *(ptr_temp2 + 0 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk2;

				const float r2 = *(ptr_temp2 + 0 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 3) * rk2;

				const float r3 = *(ptr_temp2 + 1 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk2;

				const float r4 = *(ptr_temp2 + 1 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 3) * rk2;

				if (trdY < 15)
				{
					const int dx = i + 2 * trdX;
					const int dy = j + 2 * trdY;

					if (dy < surf_ref_dst.rows)
					{
						if (dx < surf_ref_dst.cols)
						{
							float* ptr_dst = surf_ref_dst.surf + dy * surf_ref_dst.step + dx;
							*ptr_dst = r1;

							if (dx + 1 < surf_ref_dst.cols)
							{
								*(ptr_dst + 1) = r2;
							}

							if (dy + 1 < surf_ref_dst.rows)
							{
								ptr_dst += surf_ref_dst.step;
								*ptr_dst = r3;

								if (dx + 1 < surf_ref_dst.cols)
								{
									*(ptr_dst + 1) = r4;
								}
							}
						}
					}
				}
			}

			__syncthreads();
		}

		__global__ void Img8uToImg32f_blur_tex_cu(SurfRef surf_ref_dst, const float ck0, const float ck1, const float ck2, const float rk0, const float rk1, const float rk2)
		{
			__shared__ float temp1[(2 * 16) * (2 * 16)];
			__shared__ float temp2[(2 * 16) * (2 * 16)];

			const int i = blockIdx.x * (2 * 16 - 3 + 1);
			const int j = blockIdx.y * (2 * 16 - 3 + 1);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int z2 = (4 * ry) * (2 * 16) + z1;
			const float tx = (i + z1) + 0.5f;
			const float ty = j + 4 * ry + 0.5f;

			temp1[z2 + 0 * (2 * 16)] = 255.f * tex2D(tex, tx, ty);
			temp1[z2 + 1 * (2 * 16)] = 255.f * tex2D(tex, tx, ty + 1.f);
			temp1[z2 + 2 * (2 * 16)] = 255.f * tex2D(tex, tx, ty + 2.f);
			temp1[z2 + 3 * (2 * 16)] = 255.f * tex2D(tex, tx, ty + 3.f);

			__syncthreads();

			const int r = 2 * trdX + (2 * 16) * 2 * trdY;
			const float* ptr_temp1 = temp1 + r;
			float* ptr_temp2 = temp2 + r;

			if (trdY < 15)
			{
				*ptr_temp2 = *(ptr_temp1 + 0 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + 1) = *(ptr_temp1 + 0 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck2;

				*(ptr_temp2 + (2 * 16)) = *(ptr_temp1 + 1 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + (2 * 16) + 1) = *(ptr_temp1 + 1 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 1) * ck2;
			}

			__syncthreads();

			if (trdX < 15)
			{
				const float r1 = *(ptr_temp2 + 0 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk2;

				const float r2 = *(ptr_temp2 + 0 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 3) * rk2;

				const float r3 = *(ptr_temp2 + 1 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk2;

				const float r4 = *(ptr_temp2 + 1 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 3) * rk2;

				if (trdY < 15)
				{
					const int dx = i + 2 * trdX;
					const int dy = j + 2 * trdY;

					if (dy < surf_ref_dst.rows)
					{
						if (dx < surf_ref_dst.cols)
						{
							float* ptr_dst = surf_ref_dst.surf + dy * surf_ref_dst.step + dx;
							*ptr_dst = r1;

							if (dx + 1 < surf_ref_dst.cols)
							{
								*(ptr_dst + 1) = r2;
							}

							if (dy + 1 < surf_ref_dst.rows)
							{
								ptr_dst += surf_ref_dst.step;
								*ptr_dst = r3;

								if (dx + 1 < surf_ref_dst.cols)
								{
									*(ptr_dst + 1) = r4;
								}
							}
						}
					}
				}
			}

			__syncthreads();
		}
		__global__ void Img8uBGRToImg32fGRAY_blur_tex_cu(SurfRef surf_ref_dst, const float ck0, const float ck1, const float ck2, const float rk0, const float rk1, const float rk2)
		{
			__shared__ float temp1[(2 * 16) * (2 * 16)];
			__shared__ float temp2[(2 * 16) * (2 * 16)];

			const int i = blockIdx.x * (2 * 16 - 3 + 1);
			const int j = blockIdx.y * (2 * 16 - 3 + 1);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int z2 = (4 * ry) * (2 * 16) + z1;
			const float tx = 3 * (i + z1) + 0.5f;
			const float ty = j + 4 * ry + 0.5f;

			float B = tex2D(tex, tx, ty);
			float G = tex2D(tex, tx + 1.f, ty);
			float R = tex2D(tex, tx + 2.f, ty);
			temp1[z2 + 0 * (2 * 16)] = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			B = tex2D(tex, tx, ty + 1.f);
			G = tex2D(tex, tx + 1.f, ty + 1.f);
			R = tex2D(tex, tx + 2.f, ty + 1.f);
			temp1[z2 + 1 * (2 * 16)] = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			B = tex2D(tex, tx, ty + 2.f);
			G = tex2D(tex, tx + 1.f, ty + 2.f);
			R = tex2D(tex, tx + 2.f, ty + 2.f);
			temp1[z2 + 2 * (2 * 16)] = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			B = tex2D(tex, tx, ty + 3.f);
			G = tex2D(tex, tx + 1.f, ty + 3.f);
			R = tex2D(tex, tx + 2.f, ty + 3.f);
			temp1[z2 + 3 * (2 * 16)] = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			__syncthreads();

			const int r = 2 * trdX + (2 * 16) * 2 * trdY;
			const float* ptr_temp1 = temp1 + r;
			float* ptr_temp2 = temp2 + r;

			if (trdY < 15)
			{
				*ptr_temp2 = *(ptr_temp1 + 0 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + 1) = *(ptr_temp1 + 0 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck2;

				*(ptr_temp2 + (2 * 16)) = *(ptr_temp1 + 1 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + (2 * 16) + 1) = *(ptr_temp1 + 1 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 1) * ck2;
			}

			__syncthreads();

			if (trdX < 15)
			{
				const float r1 = *(ptr_temp2 + 0 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk2;

				const float r2 = *(ptr_temp2 + 0 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 3) * rk2;

				const float r3 = *(ptr_temp2 + 1 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk2;

				const float r4 = *(ptr_temp2 + 1 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 3) * rk2;

				if (trdY < 15)
				{
					const int dx = i + 2 * trdX;
					const int dy = j + 2 * trdY;

					if (dy < surf_ref_dst.rows)
					{
						if (dx < surf_ref_dst.cols)
						{
							float* ptr_dst = surf_ref_dst.surf + dy * surf_ref_dst.step + dx;
							*ptr_dst = r1;

							if (dx + 1 < surf_ref_dst.cols)
							{
								*(ptr_dst + 1) = r2;
							}

							if (dy + 1 < surf_ref_dst.rows)
							{
								ptr_dst += surf_ref_dst.step;
								*ptr_dst = r3;

								if (dx + 1 < surf_ref_dst.cols)
								{
									*(ptr_dst + 1) = r4;
								}
							}
						}
					}
				}
			}

			__syncthreads();
		}
		__global__ void Img8uBGRAToImg32fGRAY_blur_tex_cu(SurfRef surf_ref_dst, const float ck0, const float ck1, const float ck2, const float rk0, const float rk1, const float rk2)
		{
			__shared__ float temp1[(2 * 16) * (2 * 16)];
			__shared__ float temp2[(2 * 16) * (2 * 16)];

			const int i = blockIdx.x * (2 * 16 - 3 + 1);
			const int j = blockIdx.y * (2 * 16 - 3 + 1);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int z2 = (4 * ry) * (2 * 16) + z1;
			const float tx = 4 * (i + z1) + 0.5f;
			const float ty = j + 4 * ry + 0.5f;

			float B = tex2D(tex, tx, ty);
			float G = tex2D(tex, tx + 1.f, ty);
			float R = tex2D(tex, tx + 2.f, ty);
			temp1[z2 + 0 * (2 * 16)] = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			B = tex2D(tex, tx, ty + 1.f);
			G = tex2D(tex, tx + 1.f, ty + 1.f);
			R = tex2D(tex, tx + 2.f, ty + 1.f);
			temp1[z2 + 1 * (2 * 16)] = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			B = tex2D(tex, tx, ty + 2.f);
			G = tex2D(tex, tx + 1.f, ty + 2.f);
			R = tex2D(tex, tx + 2.f, ty + 2.f);
			temp1[z2 + 2 * (2 * 16)] = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			B = tex2D(tex, tx, ty + 3.f);
			G = tex2D(tex, tx + 1.f, ty + 3.f);
			R = tex2D(tex, tx + 2.f, ty + 3.f);
			temp1[z2 + 3 * (2 * 16)] = B * 0.114f * 255.f + G * 0.587f * 255.f + R * 0.299f * 255.f;

			__syncthreads();

			const int r = 2 * trdX + (2 * 16) * 2 * trdY;
			const float* ptr_temp1 = temp1 + r;
			float* ptr_temp2 = temp2 + r;

			if (trdY < 15)
			{
				*ptr_temp2 = *(ptr_temp1 + 0 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + 1) = *(ptr_temp1 + 0 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck2;

				*(ptr_temp2 + (2 * 16)) = *(ptr_temp1 + 1 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + (2 * 16) + 1) = *(ptr_temp1 + 1 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 1) * ck2;
			}

			__syncthreads();

			if (trdX < 15)
			{
				const float r1 = *(ptr_temp2 + 0 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk2;

				const float r2 = *(ptr_temp2 + 0 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 3) * rk2;

				const float r3 = *(ptr_temp2 + 1 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk2;

				const float r4 = *(ptr_temp2 + 1 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 3) * rk2;

				if (trdY < 15)
				{
					const int dx = i + 2 * trdX;
					const int dy = j + 2 * trdY;

					if (dy < surf_ref_dst.rows)
					{
						if (dx < surf_ref_dst.cols)
						{
							float* ptr_dst = surf_ref_dst.surf + dy * surf_ref_dst.step + dx;
							*ptr_dst = r1;

							if (dx + 1 < surf_ref_dst.cols)
							{
								*(ptr_dst + 1) = r2;
							}

							if (dy + 1 < surf_ref_dst.rows)
							{
								ptr_dst += surf_ref_dst.step;
								*ptr_dst = r3;

								if (dx + 1 < surf_ref_dst.cols)
								{
									*(ptr_dst + 1) = r4;
								}
							}
						}
					}
				}
			}

			__syncthreads();
		}

		__global__ void sepFilter3_cu(SurfRef surf_ref_dst, SurfRef surf_ref_src, const float ck0, const float ck1, const float ck2, const float rk0, const float rk1, const float rk2)
		{
			__shared__ float temp1[(2 * 16) * (2 * 16)];
			__shared__ float temp2[(2 * 16) * (2 * 16)];

			const int i = blockIdx.x * (2 * 16 - 3 + 1);
			const int j = blockIdx.y * (2 * 16 - 3 + 1);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);

			const int z1 = trdX + id_walf_w * 16;
			const int z2 = (4 * ry) * (2 * 16) + z1;
			//const float tx = i + z1 + 0.5f;
			//const float ty = j + 4 * ry + 0.5f;
			//temp1[z2 + 0 * (2 * 16)] = tex2D(tex, tx, ty);
			//temp1[z2 + 1 * (2 * 16)] = tex2D(tex, tx, ty + 1.f);
			//temp1[z2 + 2 * (2 * 16)] = tex2D(tex, tx, ty + 2.f);
			//temp1[z2 + 3 * (2 * 16)] = tex2D(tex, tx, ty + 3.f);

			const int tx = i + z1;
			const int ty = j + 4 * ry;
			float* ptr_src = surf_ref_src.surf + ty * surf_ref_src.step + tx;
			temp1[z2 + 0 * (2 * 16)] = *ptr_src;
			ptr_src += surf_ref_src.step;
			temp1[z2 + 1 * (2 * 16)] = *ptr_src;
			ptr_src += surf_ref_src.step;
			temp1[z2 + 2 * (2 * 16)] = *ptr_src;
			ptr_src += surf_ref_src.step;
			temp1[z2 + 3 * (2 * 16)] = *ptr_src;

			__syncthreads();

			const int r = 2 * trdX + (2 * 16) * 2 * trdY;
			const float* ptr_temp1 = temp1 + r;
			float* ptr_temp2 = temp2 + r;

			if (trdY < 15)
			{
				*ptr_temp2 = *(ptr_temp1 + 0 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + 1) = *(ptr_temp1 + 0 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 1 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck2;

				*(ptr_temp2 + (2 * 16)) = *(ptr_temp1 + 1 * (2 * 16) + 0) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 0) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 0) * ck2;

				*(ptr_temp2 + (2 * 16) + 1) = *(ptr_temp1 + 1 * (2 * 16) + 1) * ck0
					+ *(ptr_temp1 + 2 * (2 * 16) + 1) * ck1
					+ *(ptr_temp1 + 3 * (2 * 16) + 1) * ck2;
			}

			__syncthreads();

			if (trdX < 15)
			{
				const float r1 = *(ptr_temp2 + 0 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk2;

				const float r2 = *(ptr_temp2 + 0 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 0 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 0 * (2 * 16) + 3) * rk2;

				const float r3 = *(ptr_temp2 + 1 * (2 * 16) + 0) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 1) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk2;

				const float r4 = *(ptr_temp2 + 1 * (2 * 16) + 1) * rk0
					+ *(ptr_temp2 + 1 * (2 * 16) + 2) * rk1
					+ *(ptr_temp2 + 1 * (2 * 16) + 3) * rk2;

				if (trdY < 15)
				{
					float* ptr_dst = surf_ref_dst.surf + (j + 2 * trdY) * surf_ref_dst.step + i + 2 * trdX;
					*ptr_dst = r1;
					*(ptr_dst + 1) = r2;

					ptr_dst += surf_ref_dst.step;
					*ptr_dst = r3;
					*(ptr_dst + 1) = r4;
				}
			}

			__syncthreads();
		}
		__global__ void rowFilter3_cu(SurfRef surf_ref, const float k0, const float k1, const float k2)
		{
			const int i = blockIdx.x * 32;
			const int j = blockIdx.y * 32;

			const int px = i + threadIdx.x;
			const int py = j + threadIdx.y;

			const float tx = px + 0.5f;
			const float ty = py + 0.5f;

			surf_ref.surf[py * surf_ref.step + px] = k0 * tex2D(tex, tx, ty) + k1 * tex2D(tex, tx + 1, ty) + k2 * tex2D(tex, tx + 2, ty);
		}


		void ImageConverter::Init()
		{
			tex.addressMode[0] = cudaAddressModeBorder;
			tex.addressMode[1] = cudaAddressModeBorder;
			tex.filterMode = cudaFilterModeLinear;
			tex.normalized = false;
			ip_tex_channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		}

		void ImageConverter::SepFilter3(CUDA::Image_32f* dst, CUDA::Image_32f_mapped* src, const float* kernel_col, const float* kernel_row, cudaStream_t cuda_stream)
		{
			dim3 block(16, 16);
			dim3 grid(blockCount(src->width, 30), blockCount(src->height, 30));

			//tex.addressMode[0] = cudaAddressModeClamp;
			//tex.addressMode[1] = cudaAddressModeClamp;
			//tex.filterMode = cudaFilterModeLinear;
			//tex.normalized = false;
			//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

			SurfRef surf_ref_src;
			surf_ref_src.surf = src->dataDevice;
			surf_ref_src.step = src->widthStepDevice;

			SurfRef surf_ref_dst;
			surf_ref_dst.surf = dst->dataDevice;
			surf_ref_dst.step = dst->widthStepDevice;

			//cuERR(cudaBindTexture2D(NULL, &tex, src->dataDevice, &channelDesc, (size_t)src->widthStepDevice, (size_t)src->heightStepDevice, (size_t)src->widthStepDevice * 4));

			sepFilter3_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst, surf_ref_src, kernel_col[0], kernel_col[1], kernel_col[2], kernel_row[0], kernel_row[1], kernel_row[2]);

			//cuERR(cudaDeviceSynchronize());
			//cuERR(cudaUnbindTexture(&tex));
		}

		int ImageConverter::Img8uToImg32fGRAY(CUDA::Image_32f_pinned* dst, CUDA::Image_8u_pinned* src, cudaStream_t cuda_stream)
		{
			dim3 block(16, 16);
			dim3 grid(blockCount(src->width, 32), blockCount(src->height, 32));

			SurfRef surf_ref_src;
			surf_ref_src.surf = (float*)src->dataDevice;
			surf_ref_src.cols = src->width;
			surf_ref_src.rows = src->height;
			surf_ref_src.step = src->widthStepDevice;

			SurfRef surf_ref_dst;
			surf_ref_dst.surf = dst->dataDevice;
			surf_ref_dst.cols = dst->width;
			surf_ref_dst.rows = dst->height;
			surf_ref_dst.step = dst->widthStepDevice;

			switch (src->nChannel)
			{
			case 1:
				Img8uToImg32f_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst, surf_ref_src);
				break;

			case 3:
				Img8uBGRToImg32fGRAY_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst, surf_ref_src);
				break;

			case 4:
				Img8uBGRAToImg32fGRAY_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst, surf_ref_src);
				break;

			default:
				return -1;
			}

			return 0;
		}
		int ImageConverter::Img8uToImg32fGRAY_blur(CUDA::Image_32f_pinned* dst, CUDA::Image_8u_pinned* src, const float* kernel_col, const float* kernel_row, cudaStream_t cuda_stream)
		{
			dim3 block(16, 16);
			dim3 grid(blockCount(src->width, 30), blockCount(src->height, 30));

			SurfRef surf_ref_src;
			surf_ref_src.surf = (float*)src->dataDevice;
			surf_ref_src.cols = src->width;
			surf_ref_src.rows = src->height;
			surf_ref_src.step = src->widthStepDevice;

			SurfRef surf_ref_dst;
			surf_ref_dst.surf = dst->dataDevice;
			surf_ref_dst.cols = dst->width;
			surf_ref_dst.rows = dst->height;
			surf_ref_dst.step = dst->widthStepDevice;

			switch (src->nChannel)
			{
			case 1:
				Img8uToImg32f_blur_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst, surf_ref_src, kernel_col[0], kernel_col[1], kernel_col[2], kernel_row[0], kernel_row[1], kernel_row[2]);
				break;

			case 3:
				Img8uBGRToImg32fGRAY_blur_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst, surf_ref_src, kernel_col[0], kernel_col[1], kernel_col[2], kernel_row[0], kernel_row[1], kernel_row[2]);
				break;

			case 4:
				Img8uBGRAToImg32fGRAY_blur_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst, surf_ref_src, kernel_col[0], kernel_col[1], kernel_col[2], kernel_row[0], kernel_row[1], kernel_row[2]);
				break;

			default:
				return -1;
			}

			return 0;
		}

		int ImageConverter::Img8uToImg32fGRAY_tex(CUDA::Image_32f_pinned* dst, CUDA::Image_8u_pinned* src, cudaStream_t cuda_stream)
		{
			dim3 block(16, 16);
			dim3 grid(blockCount(src->width, 32), blockCount(src->height, 32));

			SurfRef surf_ref_dst;
			surf_ref_dst.surf = dst->dataDevice;
			surf_ref_dst.cols = dst->width;
			surf_ref_dst.rows = dst->height;
			surf_ref_dst.step = dst->widthStepDevice;

			cuERR(cudaBindTexture2D(NULL, &tex, (void*)src->dataDevice, &ip_tex_channelDesc, (size_t)(src->width * src->nChannel), (size_t)src->height, (size_t)src->widthStepDevice));

			switch (src->nChannel)
			{
			case 1:
				Img8uToImg32f_tex_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst);
				break;

			case 3:
				Img8uBGRToImg32fGRAY_tex_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst);
				break;

			case 4:
				Img8uBGRAToImg32fGRAY_tex_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst);
				break;

			default:
				return -1;
			}

			cuERR(cudaUnbindTexture(&tex));

			return 0;
		}
		int ImageConverter::Img8uToImg32fGRAY_blur_tex(CUDA::Image_32f_pinned* dst, CUDA::Image_8u_pinned* src, const float* kernel_col, const float* kernel_row, cudaStream_t cuda_stream)
		{
			dim3 block(16, 16);
			dim3 grid(blockCount(src->width, 30), blockCount(src->height, 30));

			SurfRef surf_ref_dst;
			surf_ref_dst.surf = dst->dataDevice;
			surf_ref_dst.cols = dst->width;
			surf_ref_dst.rows = dst->height;
			surf_ref_dst.step = dst->widthStepDevice;

			cuERR(cudaBindTexture2D(NULL, &tex, (void*)src->dataDevice, &ip_tex_channelDesc, (size_t)(src->width * src->nChannel), (size_t)src->height, (size_t)src->widthStepDevice));

			switch (src->nChannel)
			{
			case 1:
				Img8uToImg32f_blur_tex_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst, kernel_col[0], kernel_col[1], kernel_col[2], kernel_row[0], kernel_row[1], kernel_row[2]);
				break;

			case 3:
				Img8uBGRToImg32fGRAY_blur_tex_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst, kernel_col[0], kernel_col[1], kernel_col[2], kernel_row[0], kernel_row[1], kernel_row[2]);
				break;

			case 4:
				Img8uBGRAToImg32fGRAY_blur_tex_cu << <grid, block, 0, cuda_stream >> >(surf_ref_dst, kernel_col[0], kernel_col[1], kernel_col[2], kernel_row[0], kernel_row[1], kernel_row[2]);
				break;

			default:
				return -1;
			}

			cuERR(cudaUnbindTexture(&tex));

			return 0;
		}
	}

#endif
}