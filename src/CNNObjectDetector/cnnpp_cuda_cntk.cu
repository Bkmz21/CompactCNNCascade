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


#include "cnnpp_cuda_cntk.h"


//================================================================================================================================================


namespace NeuralNetworksLib
{
#if defined(USE_CUDA) && defined(USE_CNTK_MODELS) && !defined(USE_CUDA_INLINE_WEIGHTS)

	namespace CUDA
	{
		#define surf_L1_m1 4
		#define surf_L2_m1 8
		#define surf_L3_m1 16
		#define surf_hl_m1 52
		#define surf_hl_scale_m1 4
		#define surf_hl_connect_m1 4
		#define surf_ol_m1 3
		#define kernel_size_L1_m1 4*4
		#define kernel_size_L2_m1 3*3
		#define kernel_size_L3_m1 5*4
		
		#define surf_L1_m2 6
		#define surf_L2_m2 12
		#define surf_L3_m2 24
		#define surf_hl_m2 84
		#define surf_hl_scale_m2 4
		#define surf_hl_connect_m2 4
		#define surf_ol_m2 3
		#define kernel_size_L1_m2 4*4
		#define kernel_size_L2_m2 3*3
		#define kernel_size_L3_m2 8*7

		struct SurfRef
		{
			float* surf[surf_L3_m2];
			int cols = 0;
			int rows = 0;
			int step = 0;

			SurfRef()
			{
				for (int i = 0; i < surf_L3_m2; ++i)
				{
					surf[i] = NULL;
				}
			}

			SurfRef(CUDA::Image_32f* img, int out_surf_count)
			{
				for (int i = 0; i < out_surf_count; ++i)
				{
					surf[i] = img[i].dataDevice;
				}
				cols = img[0].width;
				rows = img[0].height;
				step = img[0].widthStepDevice;
			}

			~SurfRef()
			{
				for (int i = 0; i < surf_L3_m2; ++i)
				{
					surf[i] = NULL;
				}
				cols = 0;
				rows = 0;
				step = 0;
			}
		};
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		typedef texture<float, cudaTextureType2D, cudaReadModeElementType> Texture;


		//model 1
		__constant__ float kernels_L1_m1[surf_L1_m1][kernel_size_L1_m1];
		__constant__ float kernels_L2_m1[surf_L2_m1][kernel_size_L2_m1];
		__constant__ float kernels_L3_m1[surf_L3_m1][kernel_size_L3_m1];

		__constant__ float conv_b_L1_m1[surf_L1_m1];
		__constant__ float conv_b_L2_m1[surf_L2_m1];
		__constant__ float conv_b_L3_m1[surf_L3_m1];

		__constant__ float lrelu_w1_L1_m1[surf_L1_m1];
		__constant__ float lrelu_w1_L2_m1[surf_L2_m1];
		__constant__ float lrelu_w1_L3_m1[surf_L3_m1];

		__constant__ float lrelu_w2_L1_m1[surf_L1_m1];
		__constant__ float lrelu_w2_L2_m1[surf_L2_m1];
		__constant__ float lrelu_w2_L3_m1[surf_L3_m1];

		__constant__ float bn_w_L1_m1[surf_L1_m1];
		__constant__ float bn_w_L2_m1[surf_L2_m1];
		__constant__ float bn_w_L3_m1[surf_L3_m1];

		__constant__ float bn_b_L1_m1[surf_L1_m1];
		__constant__ float bn_b_L2_m1[surf_L2_m1];
		__constant__ float bn_b_L3_m1[surf_L3_m1];

		__constant__ float scale_L1_m1;
		__constant__ float scale_L2_m1;
		__constant__ float scale_L3_m1;
		__constant__ float scale_HL_m1;

		__constant__ float hl_w_m1[surf_hl_m1][surf_hl_connect_m1];
		__constant__ float hl_b_m1[surf_hl_m1];
		__constant__ float hl_tanh_w_m1[surf_hl_m1 / surf_hl_scale_m1];
		__constant__ float hl_bn_w_m1[surf_hl_scale_m1];
		__constant__ float hl_bn_b_m1[surf_hl_scale_m1];

		__constant__ float ol_w_m1[surf_ol_m1][surf_hl_m1];
		__constant__ float ol_b_m1[surf_ol_m1];
		__constant__ float ol_tanh_w_m1;

		SurfRef surf_ref_m1_L1;
		SurfRef surf_ref_m1_L2;
		SurfRef surf_ref_m1_L3;
		SurfRef surf_ref_m1_HL;

		Texture tex_m1_L1;
		Texture tex_m1_L2_1, tex_m1_L2_2, tex_m1_L2_3, tex_m1_L2_4;
		Texture tex_m1_L3_1, tex_m1_L3_2, tex_m1_L3_3, tex_m1_L3_4, tex_m1_L3_5, tex_m1_L3_6, tex_m1_L3_7, tex_m1_L3_8;
		Texture tex_m1_HL_1, tex_m1_HL_2, tex_m1_HL_3, tex_m1_HL_4, tex_m1_HL_5, tex_m1_HL_6, tex_m1_HL_7, tex_m1_HL_8;
		Texture	tex_m1_HL_9, tex_m1_HL_10, tex_m1_HL_11, tex_m1_HL_12, tex_m1_HL_13, tex_m1_HL_14, tex_m1_HL_15, tex_m1_HL_16;


		//model 2
		__constant__ float kernels_L1_m2[surf_L1_m2][kernel_size_L1_m2];
		__constant__ float kernels_L2_m2[surf_L2_m2][kernel_size_L2_m2];
		__constant__ float kernels_L3_m2[surf_L3_m2][kernel_size_L3_m2];

		__constant__ float conv_b_L1_m2[surf_L1_m2];
		__constant__ float conv_b_L2_m2[surf_L2_m2];
		__constant__ float conv_b_L3_m2[surf_L3_m2];

		__constant__ float lrelu_w1_L1_m2[surf_L1_m2];
		__constant__ float lrelu_w1_L2_m2[surf_L2_m2];
		__constant__ float lrelu_w1_L3_m2[surf_L3_m2];

		__constant__ float lrelu_w2_L1_m2[surf_L1_m2];
		__constant__ float lrelu_w2_L2_m2[surf_L2_m2];
		__constant__ float lrelu_w2_L3_m2[surf_L3_m2];

		__constant__ float bn_w_L1_m2[surf_L1_m2];
		__constant__ float bn_w_L2_m2[surf_L2_m2];
		__constant__ float bn_w_L3_m2[surf_L3_m2];

		__constant__ float bn_b_L1_m2[surf_L1_m2];
		__constant__ float bn_b_L2_m2[surf_L2_m2];
		__constant__ float bn_b_L3_m2[surf_L3_m2];

		__constant__ float scale_L1_m2[1];
		__constant__ float scale_L2_m2[1];
		__constant__ float scale_L3_m2[1];
		__constant__ float scale_HL_m2[1];

		__constant__ float hl_w_m2[surf_hl_m2][surf_hl_connect_m2];
		__constant__ float hl_b_m2[surf_hl_m2];
		__constant__ float hl_tanh_w_m2[surf_hl_m2 / surf_hl_scale_m2];
		__constant__ float hl_bn_w_m2[surf_hl_scale_m2];
		__constant__ float hl_bn_b_m2[surf_hl_scale_m2];

		__constant__ float ol_w_m2[surf_ol_m2][surf_hl_m2];
		__constant__ float ol_b_m2[surf_ol_m2];
		__constant__ float ol_tanh_w_m2;

		SurfRef surf_ref_m2_L1;
		SurfRef surf_ref_m2_L2;
		SurfRef surf_ref_m2_L3;
		SurfRef surf_ref_m2_HL;

		Texture tex_m2_L1;
		Texture tex_m2_L2_1, tex_m2_L2_2, tex_m2_L2_3, tex_m2_L2_4, tex_m2_L2_5, tex_m2_L2_6;
		Texture tex_m2_L3_1, tex_m2_L3_2, tex_m2_L3_3, tex_m2_L3_4, tex_m2_L3_5, tex_m2_L3_6, tex_m2_L3_7, tex_m2_L3_8, tex_m2_L3_9, tex_m2_L3_10, tex_m2_L3_11, tex_m2_L3_12;
		Texture tex_m2_HL_1, tex_m2_HL_2, tex_m2_HL_3, tex_m2_HL_4, tex_m2_HL_5, tex_m2_HL_6, tex_m2_HL_7, tex_m2_HL_8, tex_m2_HL_9, tex_m2_HL_10, tex_m2_HL_11, tex_m2_HL_12;
		Texture tex_m2_HL_13, tex_m2_HL_14, tex_m2_HL_15, tex_m2_HL_16, tex_m2_HL_17, tex_m2_HL_18, tex_m2_HL_19, tex_m2_HL_20, tex_m2_HL_21, tex_m2_HL_22, tex_m2_HL_23, tex_m2_HL_24;

		//---------------------------------------------------------------------------------------------------------------------------------------------------------------

		__device__ float tanhf(const float x)
		{
#ifdef USE_FAST_TANH
			float one = 1.f;
			const float tanh_a = 1.41645f;

			const float x_abs = fabsf(x);
			float x_p2 = __fmul_rn(x, x);
			const float x_p4 = __fmul_rn(x_p2, x_p2);
			x_p2 = __fadd_rn(x_p2, x_abs);
			x_p2 = __fadd_rn(x_p2, one);
			x_p2 = __fmaf_rn(tanh_a, x_p4, x_p2);
#ifdef USE_FAST_DIV
			x_p2 = 1.f / x_p2;
#else
			x_p2 = __fdiv_rn(one, x_p2);
#endif
			x_p2 = __fsub_rn(one, x_p2);

			if (x < 0.f) one = -1.f;
			x_p2 = __fmul_rn(one, x_p2);

			return x_p2;
#else
			float sgn = 1.f;
			if (x < 0.f) sgn = -1.f;
			return sgn * (1.f - 1.f / (1.f + fabs(x) + x * x + 1.41645f * x * x * x * x));
#endif
		}

		//---------------------------------------------------------------------------------------------------------------------------------------------------------------

		/*--------------------------- cnn model 1 ---------------------------*/
		//L1
		__global__ void conv_4x4x4_lrelu_bn_max_cu(SurfRef surf_ref)
		{
			const int i = blockIdx.x * 2 * 16;
			const int j = blockIdx.y * 2 * 16;

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const float tx = i + 2 * trdX + 0.5f;
			const float ty = j + 2 * trdY + 0.5f;

			const int offset = ((j >> 1) + trdY) * surf_ref.step + (i >> 1) + trdX;

			float temp[10];
			temp[0 * 5 + 0] = tex2D(tex_m1_L1, tx + 0.f, ty + 0.f);
			temp[0 * 5 + 1] = tex2D(tex_m1_L1, tx + 1.f, ty + 0.f);
			temp[0 * 5 + 2] = tex2D(tex_m1_L1, tx + 2.f, ty + 0.f);
			temp[0 * 5 + 3] = tex2D(tex_m1_L1, tx + 3.f, ty + 0.f);
			temp[0 * 5 + 4] = tex2D(tex_m1_L1, tx + 4.f, ty + 0.f);
			temp[1 * 5 + 0] = tex2D(tex_m1_L1, tx + 0.f, ty + 1.f);
			temp[1 * 5 + 1] = tex2D(tex_m1_L1, tx + 1.f, ty + 1.f);
			temp[1 * 5 + 2] = tex2D(tex_m1_L1, tx + 2.f, ty + 1.f);
			temp[1 * 5 + 3] = tex2D(tex_m1_L1, tx + 3.f, ty + 1.f);
			temp[1 * 5 + 4] = tex2D(tex_m1_L1, tx + 4.f, ty + 1.f);

			float c[4 * 4];
			//for (int k = 0; k < map_count; ++k)
			{
				c[4 * 0 + 0] = temp[0 * 5 + 0] * kernels_L1_m1[0][0]
					+ temp[0 * 5 + 1] * kernels_L1_m1[0][1]
					+ temp[0 * 5 + 2] * kernels_L1_m1[0][2]
					+ temp[0 * 5 + 3] * kernels_L1_m1[0][3];

				c[4 * 0 + 1] = temp[0 * 5 + 1] * kernels_L1_m1[0][0]
					+ temp[0 * 5 + 2] * kernels_L1_m1[0][1]
					+ temp[0 * 5 + 3] * kernels_L1_m1[0][2]
					+ temp[0 * 5 + 4] * kernels_L1_m1[0][3];

				c[4 * 0 + 2] = temp[1 * 5 + 0] * kernels_L1_m1[0][0]
					+ temp[1 * 5 + 1] * kernels_L1_m1[0][1]
					+ temp[1 * 5 + 2] * kernels_L1_m1[0][2]
					+ temp[1 * 5 + 3] * kernels_L1_m1[0][3];

				c[4 * 0 + 3] = temp[1 * 5 + 1] * kernels_L1_m1[0][0]
					+ temp[1 * 5 + 2] * kernels_L1_m1[0][1]
					+ temp[1 * 5 + 3] * kernels_L1_m1[0][2]
					+ temp[1 * 5 + 4] * kernels_L1_m1[0][3];
			}
			{
				c[4 * 1 + 0] = temp[0 * 5 + 0] * kernels_L1_m1[1][0]
					+ temp[0 * 5 + 1] * kernels_L1_m1[1][1]
					+ temp[0 * 5 + 2] * kernels_L1_m1[1][2]
					+ temp[0 * 5 + 3] * kernels_L1_m1[1][3];

				c[4 * 1 + 1] = temp[0 * 5 + 1] * kernels_L1_m1[1][0]
					+ temp[0 * 5 + 2] * kernels_L1_m1[1][1]
					+ temp[0 * 5 + 3] * kernels_L1_m1[1][2]
					+ temp[0 * 5 + 4] * kernels_L1_m1[1][3];

				c[4 * 1 + 2] = temp[1 * 5 + 0] * kernels_L1_m1[1][0]
					+ temp[1 * 5 + 1] * kernels_L1_m1[1][1]
					+ temp[1 * 5 + 2] * kernels_L1_m1[1][2]
					+ temp[1 * 5 + 3] * kernels_L1_m1[1][3];

				c[4 * 1 + 3] = temp[1 * 5 + 1] * kernels_L1_m1[1][0]
					+ temp[1 * 5 + 2] * kernels_L1_m1[1][1]
					+ temp[1 * 5 + 3] * kernels_L1_m1[1][2]
					+ temp[1 * 5 + 4] * kernels_L1_m1[1][3];
			}
			{
				c[4 * 2 + 0] = temp[0 * 5 + 0] * kernels_L1_m1[2][0]
					+ temp[0 * 5 + 1] * kernels_L1_m1[2][1]
					+ temp[0 * 5 + 2] * kernels_L1_m1[2][2]
					+ temp[0 * 5 + 3] * kernels_L1_m1[2][3];

				c[4 * 2 + 1] = temp[0 * 5 + 1] * kernels_L1_m1[2][0]
					+ temp[0 * 5 + 2] * kernels_L1_m1[2][1]
					+ temp[0 * 5 + 3] * kernels_L1_m1[2][2]
					+ temp[0 * 5 + 4] * kernels_L1_m1[2][3];

				c[4 * 2 + 2] = temp[1 * 5 + 0] * kernels_L1_m1[2][0]
					+ temp[1 * 5 + 1] * kernels_L1_m1[2][1]
					+ temp[1 * 5 + 2] * kernels_L1_m1[2][2]
					+ temp[1 * 5 + 3] * kernels_L1_m1[2][3];

				c[4 * 2 + 3] = temp[1 * 5 + 1] * kernels_L1_m1[2][0]
					+ temp[1 * 5 + 2] * kernels_L1_m1[2][1]
					+ temp[1 * 5 + 3] * kernels_L1_m1[2][2]
					+ temp[1 * 5 + 4] * kernels_L1_m1[2][3];
			}
			{
				c[4 * 3 + 0] = temp[0 * 5 + 0] * kernels_L1_m1[3][0]
					+ temp[0 * 5 + 1] * kernels_L1_m1[3][1]
					+ temp[0 * 5 + 2] * kernels_L1_m1[3][2]
					+ temp[0 * 5 + 3] * kernels_L1_m1[3][3];

				c[4 * 3 + 1] = temp[0 * 5 + 1] * kernels_L1_m1[3][0]
					+ temp[0 * 5 + 2] * kernels_L1_m1[3][1]
					+ temp[0 * 5 + 3] * kernels_L1_m1[3][2]
					+ temp[0 * 5 + 4] * kernels_L1_m1[3][3];

				c[4 * 3 + 2] = temp[1 * 5 + 0] * kernels_L1_m1[3][0]
					+ temp[1 * 5 + 1] * kernels_L1_m1[3][1]
					+ temp[1 * 5 + 2] * kernels_L1_m1[3][2]
					+ temp[1 * 5 + 3] * kernels_L1_m1[3][3];

				c[4 * 3 + 3] = temp[1 * 5 + 1] * kernels_L1_m1[3][0]
					+ temp[1 * 5 + 2] * kernels_L1_m1[3][1]
					+ temp[1 * 5 + 3] * kernels_L1_m1[3][2]
					+ temp[1 * 5 + 4] * kernels_L1_m1[3][3];
			}

			temp[0 * 5 + 0] = tex2D(tex_m1_L1, tx + 0.f, ty + 2.f);
			temp[0 * 5 + 1] = tex2D(tex_m1_L1, tx + 1.f, ty + 2.f);
			temp[0 * 5 + 2] = tex2D(tex_m1_L1, tx + 2.f, ty + 2.f);
			temp[0 * 5 + 3] = tex2D(tex_m1_L1, tx + 3.f, ty + 2.f);
			temp[0 * 5 + 4] = tex2D(tex_m1_L1, tx + 4.f, ty + 2.f);

			//for (int k = 0; k < map_count; ++k)
			{
				c[4 * 0 + 0] += temp[1 * 5 + 0] * kernels_L1_m1[0][4]
					+ temp[1 * 5 + 1] * kernels_L1_m1[0][5]
					+ temp[1 * 5 + 2] * kernels_L1_m1[0][6]
					+ temp[1 * 5 + 3] * kernels_L1_m1[0][7];

				c[4 * 0 + 1] += temp[1 * 5 + 1] * kernels_L1_m1[0][4]
					+ temp[1 * 5 + 2] * kernels_L1_m1[0][5]
					+ temp[1 * 5 + 3] * kernels_L1_m1[0][6]
					+ temp[1 * 5 + 4] * kernels_L1_m1[0][7];

				c[4 * 0 + 2] += temp[0 * 5 + 0] * kernels_L1_m1[0][4]
					+ temp[0 * 5 + 1] * kernels_L1_m1[0][5]
					+ temp[0 * 5 + 2] * kernels_L1_m1[0][6]
					+ temp[0 * 5 + 3] * kernels_L1_m1[0][7];

				c[4 * 0 + 3] += temp[0 * 5 + 1] * kernels_L1_m1[0][4]
					+ temp[0 * 5 + 2] * kernels_L1_m1[0][5]
					+ temp[0 * 5 + 3] * kernels_L1_m1[0][6]
					+ temp[0 * 5 + 4] * kernels_L1_m1[0][7];
			}
			{
				c[4 * 1 + 0] += temp[1 * 5 + 0] * kernels_L1_m1[1][4]
					+ temp[1 * 5 + 1] * kernels_L1_m1[1][5]
					+ temp[1 * 5 + 2] * kernels_L1_m1[1][6]
					+ temp[1 * 5 + 3] * kernels_L1_m1[1][7];

				c[4 * 1 + 1] += temp[1 * 5 + 1] * kernels_L1_m1[1][4]
					+ temp[1 * 5 + 2] * kernels_L1_m1[1][5]
					+ temp[1 * 5 + 3] * kernels_L1_m1[1][6]
					+ temp[1 * 5 + 4] * kernels_L1_m1[1][7];

				c[4 * 1 + 2] += temp[0 * 5 + 0] * kernels_L1_m1[1][4]
					+ temp[0 * 5 + 1] * kernels_L1_m1[1][5]
					+ temp[0 * 5 + 2] * kernels_L1_m1[1][6]
					+ temp[0 * 5 + 3] * kernels_L1_m1[1][7];

				c[4 * 1 + 3] += temp[0 * 5 + 1] * kernels_L1_m1[1][4]
					+ temp[0 * 5 + 2] * kernels_L1_m1[1][5]
					+ temp[0 * 5 + 3] * kernels_L1_m1[1][6]
					+ temp[0 * 5 + 4] * kernels_L1_m1[1][7];
			}
			{
				c[4 * 2 + 0] += temp[1 * 5 + 0] * kernels_L1_m1[2][4]
					+ temp[1 * 5 + 1] * kernels_L1_m1[2][5]
					+ temp[1 * 5 + 2] * kernels_L1_m1[2][6]
					+ temp[1 * 5 + 3] * kernels_L1_m1[2][7];

				c[4 * 2 + 1] += temp[1 * 5 + 1] * kernels_L1_m1[2][4]
					+ temp[1 * 5 + 2] * kernels_L1_m1[2][5]
					+ temp[1 * 5 + 3] * kernels_L1_m1[2][6]
					+ temp[1 * 5 + 4] * kernels_L1_m1[2][7];

				c[4 * 2 + 2] += temp[0 * 5 + 0] * kernels_L1_m1[2][4]
					+ temp[0 * 5 + 1] * kernels_L1_m1[2][5]
					+ temp[0 * 5 + 2] * kernels_L1_m1[2][6]
					+ temp[0 * 5 + 3] * kernels_L1_m1[2][7];

				c[4 * 2 + 3] += temp[0 * 5 + 1] * kernels_L1_m1[2][4]
					+ temp[0 * 5 + 2] * kernels_L1_m1[2][5]
					+ temp[0 * 5 + 3] * kernels_L1_m1[2][6]
					+ temp[0 * 5 + 4] * kernels_L1_m1[2][7];
			}
			{
				c[4 * 3 + 0] += temp[1 * 5 + 0] * kernels_L1_m1[3][4]
					+ temp[1 * 5 + 1] * kernels_L1_m1[3][5]
					+ temp[1 * 5 + 2] * kernels_L1_m1[3][6]
					+ temp[1 * 5 + 3] * kernels_L1_m1[3][7];

				c[4 * 3 + 1] += temp[1 * 5 + 1] * kernels_L1_m1[3][4]
					+ temp[1 * 5 + 2] * kernels_L1_m1[3][5]
					+ temp[1 * 5 + 3] * kernels_L1_m1[3][6]
					+ temp[1 * 5 + 4] * kernels_L1_m1[3][7];

				c[4 * 3 + 2] += temp[0 * 5 + 0] * kernels_L1_m1[3][4]
					+ temp[0 * 5 + 1] * kernels_L1_m1[3][5]
					+ temp[0 * 5 + 2] * kernels_L1_m1[3][6]
					+ temp[0 * 5 + 3] * kernels_L1_m1[3][7];

				c[4 * 3 + 3] += temp[0 * 5 + 1] * kernels_L1_m1[3][4]
					+ temp[0 * 5 + 2] * kernels_L1_m1[3][5]
					+ temp[0 * 5 + 3] * kernels_L1_m1[3][6]
					+ temp[0 * 5 + 4] * kernels_L1_m1[3][7];
			}

			temp[1 * 5 + 0] = tex2D(tex_m1_L1, tx + 0.f, ty + 3.f);
			temp[1 * 5 + 1] = tex2D(tex_m1_L1, tx + 1.f, ty + 3.f);
			temp[1 * 5 + 2] = tex2D(tex_m1_L1, tx + 2.f, ty + 3.f);
			temp[1 * 5 + 3] = tex2D(tex_m1_L1, tx + 3.f, ty + 3.f);
			temp[1 * 5 + 4] = tex2D(tex_m1_L1, tx + 4.f, ty + 3.f);

			//for (int k = 0; k < map_count; ++k)
			{
				c[4 * 0 + 0] += temp[0 * 5 + 0] * kernels_L1_m1[0][8]
					+ temp[0 * 5 + 1] * kernels_L1_m1[0][9]
					+ temp[0 * 5 + 2] * kernels_L1_m1[0][10]
					+ temp[0 * 5 + 3] * kernels_L1_m1[0][11];

				c[4 * 0 + 1] += temp[0 * 5 + 1] * kernels_L1_m1[0][8]
					+ temp[0 * 5 + 2] * kernels_L1_m1[0][9]
					+ temp[0 * 5 + 3] * kernels_L1_m1[0][10]
					+ temp[0 * 5 + 4] * kernels_L1_m1[0][11];

				c[4 * 0 + 2] += temp[1 * 5 + 0] * kernels_L1_m1[0][8]
					+ temp[1 * 5 + 1] * kernels_L1_m1[0][9]
					+ temp[1 * 5 + 2] * kernels_L1_m1[0][10]
					+ temp[1 * 5 + 3] * kernels_L1_m1[0][11];

				c[4 * 0 + 3] += temp[1 * 5 + 1] * kernels_L1_m1[0][8]
					+ temp[1 * 5 + 2] * kernels_L1_m1[0][9]
					+ temp[1 * 5 + 3] * kernels_L1_m1[0][10]
					+ temp[1 * 5 + 4] * kernels_L1_m1[0][11];
			}
			{
				c[4 * 1 + 0] += temp[0 * 5 + 0] * kernels_L1_m1[1][8]
					+ temp[0 * 5 + 1] * kernels_L1_m1[1][9]
					+ temp[0 * 5 + 2] * kernels_L1_m1[1][10]
					+ temp[0 * 5 + 3] * kernels_L1_m1[1][11];

				c[4 * 1 + 1] += temp[0 * 5 + 1] * kernels_L1_m1[1][8]
					+ temp[0 * 5 + 2] * kernels_L1_m1[1][9]
					+ temp[0 * 5 + 3] * kernels_L1_m1[1][10]
					+ temp[0 * 5 + 4] * kernels_L1_m1[1][11];

				c[4 * 1 + 2] += temp[1 * 5 + 0] * kernels_L1_m1[1][8]
					+ temp[1 * 5 + 1] * kernels_L1_m1[1][9]
					+ temp[1 * 5 + 2] * kernels_L1_m1[1][10]
					+ temp[1 * 5 + 3] * kernels_L1_m1[1][11];

				c[4 * 1 + 3] += temp[1 * 5 + 1] * kernels_L1_m1[1][8]
					+ temp[1 * 5 + 2] * kernels_L1_m1[1][9]
					+ temp[1 * 5 + 3] * kernels_L1_m1[1][10]
					+ temp[1 * 5 + 4] * kernels_L1_m1[1][11];
			}
			{
				c[4 * 2 + 0] += temp[0 * 5 + 0] * kernels_L1_m1[2][8]
					+ temp[0 * 5 + 1] * kernels_L1_m1[2][9]
					+ temp[0 * 5 + 2] * kernels_L1_m1[2][10]
					+ temp[0 * 5 + 3] * kernels_L1_m1[2][11];

				c[4 * 2 + 1] += temp[0 * 5 + 1] * kernels_L1_m1[2][8]
					+ temp[0 * 5 + 2] * kernels_L1_m1[2][9]
					+ temp[0 * 5 + 3] * kernels_L1_m1[2][10]
					+ temp[0 * 5 + 4] * kernels_L1_m1[2][11];

				c[4 * 2 + 2] += temp[1 * 5 + 0] * kernels_L1_m1[2][8]
					+ temp[1 * 5 + 1] * kernels_L1_m1[2][9]
					+ temp[1 * 5 + 2] * kernels_L1_m1[2][10]
					+ temp[1 * 5 + 3] * kernels_L1_m1[2][11];

				c[4 * 2 + 3] += temp[1 * 5 + 1] * kernels_L1_m1[2][8]
					+ temp[1 * 5 + 2] * kernels_L1_m1[2][9]
					+ temp[1 * 5 + 3] * kernels_L1_m1[2][10]
					+ temp[1 * 5 + 4] * kernels_L1_m1[2][11];
			}
			{
				c[4 * 3 + 0] += temp[0 * 5 + 0] * kernels_L1_m1[3][8]
					+ temp[0 * 5 + 1] * kernels_L1_m1[3][9]
					+ temp[0 * 5 + 2] * kernels_L1_m1[3][10]
					+ temp[0 * 5 + 3] * kernels_L1_m1[3][11];

				c[4 * 3 + 1] += temp[0 * 5 + 1] * kernels_L1_m1[3][8]
					+ temp[0 * 5 + 2] * kernels_L1_m1[3][9]
					+ temp[0 * 5 + 3] * kernels_L1_m1[3][10]
					+ temp[0 * 5 + 4] * kernels_L1_m1[3][11];

				c[4 * 3 + 2] += temp[1 * 5 + 0] * kernels_L1_m1[3][8]
					+ temp[1 * 5 + 1] * kernels_L1_m1[3][9]
					+ temp[1 * 5 + 2] * kernels_L1_m1[3][10]
					+ temp[1 * 5 + 3] * kernels_L1_m1[3][11];

				c[4 * 3 + 3] += temp[1 * 5 + 1] * kernels_L1_m1[3][8]
					+ temp[1 * 5 + 2] * kernels_L1_m1[3][9]
					+ temp[1 * 5 + 3] * kernels_L1_m1[3][10]
					+ temp[1 * 5 + 4] * kernels_L1_m1[3][11];
			}

			temp[0 * 5 + 0] = tex2D(tex_m1_L1, tx + 0.f, ty + 4.f);
			temp[0 * 5 + 1] = tex2D(tex_m1_L1, tx + 1.f, ty + 4.f);
			temp[0 * 5 + 2] = tex2D(tex_m1_L1, tx + 2.f, ty + 4.f);
			temp[0 * 5 + 3] = tex2D(tex_m1_L1, tx + 3.f, ty + 4.f);
			temp[0 * 5 + 4] = tex2D(tex_m1_L1, tx + 4.f, ty + 4.f);

			//for (int k = 0; k < map_count; ++k)
			{
				c[4 * 0 + 0] += temp[1 * 5 + 0] * kernels_L1_m1[0][12]
					+ temp[1 * 5 + 1] * kernels_L1_m1[0][13]
					+ temp[1 * 5 + 2] * kernels_L1_m1[0][14]
					+ temp[1 * 5 + 3] * kernels_L1_m1[0][15];

				c[4 * 0 + 1] += temp[1 * 5 + 1] * kernels_L1_m1[0][12]
					+ temp[1 * 5 + 2] * kernels_L1_m1[0][13]
					+ temp[1 * 5 + 3] * kernels_L1_m1[0][14]
					+ temp[1 * 5 + 4] * kernels_L1_m1[0][15];

				c[4 * 0 + 2] += temp[0 * 5 + 0] * kernels_L1_m1[0][12]
					+ temp[0 * 5 + 1] * kernels_L1_m1[0][13]
					+ temp[0 * 5 + 2] * kernels_L1_m1[0][14]
					+ temp[0 * 5 + 3] * kernels_L1_m1[0][15];

				c[4 * 0 + 3] += temp[0 * 5 + 1] * kernels_L1_m1[0][12]
					+ temp[0 * 5 + 2] * kernels_L1_m1[0][13]
					+ temp[0 * 5 + 3] * kernels_L1_m1[0][14]
					+ temp[0 * 5 + 4] * kernels_L1_m1[0][15];

				c[4 * 0 + 0] += conv_b_L1_m1[0];
				c[4 * 0 + 0] = lrelu_w1_L1_m1[0] * c[4 * 0 + 0] + lrelu_w2_L1_m1[0] * fmaxf(0.f, c[4 * 0 + 0]) + bn_b_L1_m1[0];

				c[4 * 0 + 1] += conv_b_L1_m1[0];
				c[4 * 0 + 1] = lrelu_w1_L1_m1[0] * c[4 * 0 + 1] + lrelu_w2_L1_m1[0] * fmaxf(0.f, c[4 * 0 + 1]) + bn_b_L1_m1[0];

				c[4 * 0 + 2] += conv_b_L1_m1[0];
				c[4 * 0 + 2] = lrelu_w1_L1_m1[0] * c[4 * 0 + 2] + lrelu_w2_L1_m1[0] * fmaxf(0.f, c[4 * 0 + 2]) + bn_b_L1_m1[0];

				c[4 * 0 + 3] += conv_b_L1_m1[0];
				c[4 * 0 + 3] = lrelu_w1_L1_m1[0] * c[4 * 0 + 3] + lrelu_w2_L1_m1[0] * fmaxf(0.f, c[4 * 0 + 3]) + bn_b_L1_m1[0];

				surf_ref.surf[0][offset] = fmaxf(fmaxf(c[4 * 0 + 0], c[4 * 0 + 1]), fmaxf(c[4 * 0 + 2], c[4 * 0 + 3]));
			}
			{
				c[4 * 1 + 0] += temp[1 * 5 + 0] * kernels_L1_m1[1][12]
					+ temp[1 * 5 + 1] * kernels_L1_m1[1][13]
					+ temp[1 * 5 + 2] * kernels_L1_m1[1][14]
					+ temp[1 * 5 + 3] * kernels_L1_m1[1][15];

				c[4 * 1 + 1] += temp[1 * 5 + 1] * kernels_L1_m1[1][12]
					+ temp[1 * 5 + 2] * kernels_L1_m1[1][13]
					+ temp[1 * 5 + 3] * kernels_L1_m1[1][14]
					+ temp[1 * 5 + 4] * kernels_L1_m1[1][15];

				c[4 * 1 + 2] += temp[0 * 5 + 0] * kernels_L1_m1[1][12]
					+ temp[0 * 5 + 1] * kernels_L1_m1[1][13]
					+ temp[0 * 5 + 2] * kernels_L1_m1[1][14]
					+ temp[0 * 5 + 3] * kernels_L1_m1[1][15];

				c[4 * 1 + 3] += temp[0 * 5 + 1] * kernels_L1_m1[1][12]
					+ temp[0 * 5 + 2] * kernels_L1_m1[1][13]
					+ temp[0 * 5 + 3] * kernels_L1_m1[1][14]
					+ temp[0 * 5 + 4] * kernels_L1_m1[1][15];

				c[4 * 1 + 0] += conv_b_L1_m1[1];
				c[4 * 1 + 0] = lrelu_w1_L1_m1[1] * c[4 * 1 + 0] + lrelu_w2_L1_m1[1] * fmaxf(0.f, c[4 * 1 + 0]) + bn_b_L1_m1[1];

				c[4 * 1 + 1] += conv_b_L1_m1[1];
				c[4 * 1 + 1] = lrelu_w1_L1_m1[1] * c[4 * 1 + 1] + lrelu_w2_L1_m1[1] * fmaxf(0.f, c[4 * 1 + 1]) + bn_b_L1_m1[1];

				c[4 * 1 + 2] += conv_b_L1_m1[1];
				c[4 * 1 + 2] = lrelu_w1_L1_m1[1] * c[4 * 1 + 2] + lrelu_w2_L1_m1[1] * fmaxf(0.f, c[4 * 1 + 2]) + bn_b_L1_m1[1];

				c[4 * 1 + 3] += conv_b_L1_m1[1];
				c[4 * 1 + 3] = lrelu_w1_L1_m1[1] * c[4 * 1 + 3] + lrelu_w2_L1_m1[1] * fmaxf(0.f, c[4 * 1 + 3]) + bn_b_L1_m1[1];

				surf_ref.surf[1][offset] = fmaxf(fmaxf(c[4 * 1 + 0], c[4 * 1 + 1]), fmaxf(c[4 * 1 + 2], c[4 * 1 + 3]));
			}
			{
				c[4 * 2 + 0] += temp[1 * 5 + 0] * kernels_L1_m1[2][12]
					+ temp[1 * 5 + 1] * kernels_L1_m1[2][13]
					+ temp[1 * 5 + 2] * kernels_L1_m1[2][14]
					+ temp[1 * 5 + 3] * kernels_L1_m1[2][15];

				c[4 * 2 + 1] += temp[1 * 5 + 1] * kernels_L1_m1[2][12]
					+ temp[1 * 5 + 2] * kernels_L1_m1[2][13]
					+ temp[1 * 5 + 3] * kernels_L1_m1[2][14]
					+ temp[1 * 5 + 4] * kernels_L1_m1[2][15];

				c[4 * 2 + 2] += temp[0 * 5 + 0] * kernels_L1_m1[2][12]
					+ temp[0 * 5 + 1] * kernels_L1_m1[2][13]
					+ temp[0 * 5 + 2] * kernels_L1_m1[2][14]
					+ temp[0 * 5 + 3] * kernels_L1_m1[2][15];

				c[4 * 2 + 3] += temp[0 * 5 + 1] * kernels_L1_m1[2][12]
					+ temp[0 * 5 + 2] * kernels_L1_m1[2][13]
					+ temp[0 * 5 + 3] * kernels_L1_m1[2][14]
					+ temp[0 * 5 + 4] * kernels_L1_m1[2][15];

				c[4 * 2 + 0] += conv_b_L1_m1[2];
				c[4 * 2 + 0] = lrelu_w1_L1_m1[2] * c[4 * 2 + 0] + lrelu_w2_L1_m1[2] * fmaxf(0.f, c[4 * 2 + 0]) + bn_b_L1_m1[2];

				c[4 * 2 + 1] += conv_b_L1_m1[2];
				c[4 * 2 + 1] = lrelu_w1_L1_m1[2] * c[4 * 2 + 1] + lrelu_w2_L1_m1[2] * fmaxf(0.f, c[4 * 2 + 1]) + bn_b_L1_m1[2];

				c[4 * 2 + 2] += conv_b_L1_m1[2];
				c[4 * 2 + 2] = lrelu_w1_L1_m1[2] * c[4 * 2 + 2] + lrelu_w2_L1_m1[2] * fmaxf(0.f, c[4 * 2 + 2]) + bn_b_L1_m1[2];

				c[4 * 2 + 3] += conv_b_L1_m1[2];
				c[4 * 2 + 3] = lrelu_w1_L1_m1[2] * c[4 * 2 + 3] + lrelu_w2_L1_m1[2] * fmaxf(0.f, c[4 * 2 + 3]) + bn_b_L1_m1[2];

				surf_ref.surf[2][offset] = fmaxf(fmaxf(c[4 * 2 + 0], c[4 * 2 + 1]), fmaxf(c[4 * 2 + 2], c[4 * 2 + 3]));
			}
			{
				c[4 * 3 + 0] += temp[1 * 5 + 0] * kernels_L1_m1[3][12]
					+ temp[1 * 5 + 1] * kernels_L1_m1[3][13]
					+ temp[1 * 5 + 2] * kernels_L1_m1[3][14]
					+ temp[1 * 5 + 3] * kernels_L1_m1[3][15];

				c[4 * 3 + 1] += temp[1 * 5 + 1] * kernels_L1_m1[3][12]
					+ temp[1 * 5 + 2] * kernels_L1_m1[3][13]
					+ temp[1 * 5 + 3] * kernels_L1_m1[3][14]
					+ temp[1 * 5 + 4] * kernels_L1_m1[3][15];

				c[4 * 3 + 2] += temp[0 * 5 + 0] * kernels_L1_m1[3][12]
					+ temp[0 * 5 + 1] * kernels_L1_m1[3][13]
					+ temp[0 * 5 + 2] * kernels_L1_m1[3][14]
					+ temp[0 * 5 + 3] * kernels_L1_m1[3][15];

				c[4 * 3 + 3] += temp[0 * 5 + 1] * kernels_L1_m1[3][12]
					+ temp[0 * 5 + 2] * kernels_L1_m1[3][13]
					+ temp[0 * 5 + 3] * kernels_L1_m1[3][14]
					+ temp[0 * 5 + 4] * kernels_L1_m1[3][15];

				c[4 * 3 + 0] += conv_b_L1_m1[3];
				c[4 * 3 + 0] = lrelu_w1_L1_m1[3] * c[4 * 3 + 0] + lrelu_w2_L1_m1[3] * fmaxf(0.f, c[4 * 3 + 0]) + bn_b_L1_m1[3];

				c[4 * 3 + 1] += conv_b_L1_m1[3];
				c[4 * 3 + 1] = lrelu_w1_L1_m1[3] * c[4 * 3 + 1] + lrelu_w2_L1_m1[3] * fmaxf(0.f, c[4 * 3 + 1]) + bn_b_L1_m1[3];

				c[4 * 3 + 2] += conv_b_L1_m1[3];
				c[4 * 3 + 2] = lrelu_w1_L1_m1[3] * c[4 * 3 + 2] + lrelu_w2_L1_m1[3] * fmaxf(0.f, c[4 * 3 + 2]) + bn_b_L1_m1[3];

				c[4 * 3 + 3] += conv_b_L1_m1[3];
				c[4 * 3 + 3] = lrelu_w1_L1_m1[3] * c[4 * 3 + 3] + lrelu_w2_L1_m1[3] * fmaxf(0.f, c[4 * 3 + 3]) + bn_b_L1_m1[3];

				surf_ref.surf[3][offset] = fmaxf(fmaxf(c[4 * 3 + 0], c[4 * 3 + 1]), fmaxf(c[4 * 3 + 2], c[4 * 3 + 3]));
			}
		}

		//L2
		__global__ void add_2_3_conv_4x3x3_lrelu_bn_max_L_cu(SurfRef surf_ref)
		{
			const int i = blockIdx.x * 2 * 16;
			const int j = blockIdx.y * 2 * 16;

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const float tx = i + 2 * trdX + 0.5f;
			const float ty = j + 2 * trdY + 0.5f;

			const int offset = ((j >> 1) + trdY) * surf_ref.step + (i >> 1) + trdX;

			float temp1[8];
			temp1[0 * 4 + 0] = tex2D(tex_m1_L2_1, tx + 0.f, ty + 0.f);
			temp1[0 * 4 + 1] = tex2D(tex_m1_L2_1, tx + 1.f, ty + 0.f);
			temp1[0 * 4 + 2] = tex2D(tex_m1_L2_1, tx + 2.f, ty + 0.f);
			temp1[0 * 4 + 3] = tex2D(tex_m1_L2_1, tx + 3.f, ty + 0.f);
			temp1[1 * 4 + 0] = tex2D(tex_m1_L2_1, tx + 0.f, ty + 1.f);
			temp1[1 * 4 + 1] = tex2D(tex_m1_L2_1, tx + 1.f, ty + 1.f);
			temp1[1 * 4 + 2] = tex2D(tex_m1_L2_1, tx + 2.f, ty + 1.f);
			temp1[1 * 4 + 3] = tex2D(tex_m1_L2_1, tx + 3.f, ty + 1.f);
			temp1[0 * 4 + 0] += tex2D(tex_m1_L2_2, tx + 0.f, ty + 0.f);
			temp1[0 * 4 + 1] += tex2D(tex_m1_L2_2, tx + 1.f, ty + 0.f);
			temp1[0 * 4 + 2] += tex2D(tex_m1_L2_2, tx + 2.f, ty + 0.f);
			temp1[0 * 4 + 3] += tex2D(tex_m1_L2_2, tx + 3.f, ty + 0.f);
			temp1[1 * 4 + 0] += tex2D(tex_m1_L2_2, tx + 0.f, ty + 1.f);
			temp1[1 * 4 + 1] += tex2D(tex_m1_L2_2, tx + 1.f, ty + 1.f);
			temp1[1 * 4 + 2] += tex2D(tex_m1_L2_2, tx + 2.f, ty + 1.f);
			temp1[1 * 4 + 3] += tex2D(tex_m1_L2_2, tx + 3.f, ty + 1.f);

			float temp2[8];
			temp2[0 * 4 + 0] = temp1[0 * 4 + 0] + tex2D(tex_m1_L2_3, tx + 0.f, ty + 0.f);
			temp2[0 * 4 + 1] = temp1[0 * 4 + 1] + tex2D(tex_m1_L2_3, tx + 1.f, ty + 0.f);
			temp2[0 * 4 + 2] = temp1[0 * 4 + 2] + tex2D(tex_m1_L2_3, tx + 2.f, ty + 0.f);
			temp2[0 * 4 + 3] = temp1[0 * 4 + 3] + tex2D(tex_m1_L2_3, tx + 3.f, ty + 0.f);
			temp2[1 * 4 + 0] = temp1[1 * 4 + 0] + tex2D(tex_m1_L2_3, tx + 0.f, ty + 1.f);
			temp2[1 * 4 + 1] = temp1[1 * 4 + 1] + tex2D(tex_m1_L2_3, tx + 1.f, ty + 1.f);
			temp2[1 * 4 + 2] = temp1[1 * 4 + 2] + tex2D(tex_m1_L2_3, tx + 2.f, ty + 1.f);
			temp2[1 * 4 + 3] = temp1[1 * 4 + 3] + tex2D(tex_m1_L2_3, tx + 3.f, ty + 1.f);

			float c[4 * 4];
			//for (int k = 0; k < map_count; ++k)
			{
				c[4 * 0 + 0] = temp1[0 * 4 + 0] * kernels_L2_m1[0][0]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[0][1]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[0][2];

				c[4 * 0 + 1] = temp1[0 * 4 + 1] * kernels_L2_m1[0][0]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[0][1]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[0][2];

				c[4 * 0 + 2] = temp1[1 * 4 + 0] * kernels_L2_m1[0][0]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[0][1]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[0][2];

				c[4 * 0 + 3] = temp1[1 * 4 + 1] * kernels_L2_m1[0][0]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[0][1]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[0][2];
			}
			{
				c[4 * 1 + 0] = temp1[0 * 4 + 0] * kernels_L2_m1[1][0]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[1][1]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[1][2];

				c[4 * 1 + 1] = temp1[0 * 4 + 1] * kernels_L2_m1[1][0]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[1][1]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[1][2];

				c[4 * 1 + 2] = temp1[1 * 4 + 0] * kernels_L2_m1[1][0]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[1][1]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[1][2];

				c[4 * 1 + 3] = temp1[1 * 4 + 1] * kernels_L2_m1[1][0]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[1][1]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[1][2];
			}
			{
				c[4 * 2 + 0] = temp2[0 * 4 + 0] * kernels_L2_m1[2][0]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[2][1]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[2][2];

				c[4 * 2 + 1] = temp2[0 * 4 + 1] * kernels_L2_m1[2][0]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[2][1]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[2][2];

				c[4 * 2 + 2] = temp2[1 * 4 + 0] * kernels_L2_m1[2][0]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[2][1]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[2][2];

				c[4 * 2 + 3] = temp2[1 * 4 + 1] * kernels_L2_m1[2][0]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[2][1]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[2][2];
			}
			{
				c[4 * 3 + 0] = temp2[0 * 4 + 0] * kernels_L2_m1[3][0]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[3][1]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[3][2];

				c[4 * 3 + 1] = temp2[0 * 4 + 1] * kernels_L2_m1[3][0]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[3][1]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[3][2];

				c[4 * 3 + 2] = temp2[1 * 4 + 0] * kernels_L2_m1[3][0]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[3][1]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[3][2];

				c[4 * 3 + 3] = temp2[1 * 4 + 1] * kernels_L2_m1[3][0]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[3][1]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[3][2];
			}

			temp1[0 * 4 + 0] = tex2D(tex_m1_L2_1, tx + 0.f, ty + 2.f);
			temp1[0 * 4 + 1] = tex2D(tex_m1_L2_1, tx + 1.f, ty + 2.f);
			temp1[0 * 4 + 2] = tex2D(tex_m1_L2_1, tx + 2.f, ty + 2.f);
			temp1[0 * 4 + 3] = tex2D(tex_m1_L2_1, tx + 3.f, ty + 2.f);
			temp1[0 * 4 + 0] += tex2D(tex_m1_L2_2, tx + 0.f, ty + 2.f);
			temp1[0 * 4 + 1] += tex2D(tex_m1_L2_2, tx + 1.f, ty + 2.f);
			temp1[0 * 4 + 2] += tex2D(tex_m1_L2_2, tx + 2.f, ty + 2.f);
			temp1[0 * 4 + 3] += tex2D(tex_m1_L2_2, tx + 3.f, ty + 2.f);

			temp2[0 * 4 + 0] = temp1[0 * 4 + 0] + tex2D(tex_m1_L2_3, tx + 0.f, ty + 2.f);
			temp2[0 * 4 + 1] = temp1[0 * 4 + 1] + tex2D(tex_m1_L2_3, tx + 1.f, ty + 2.f);
			temp2[0 * 4 + 2] = temp1[0 * 4 + 2] + tex2D(tex_m1_L2_3, tx + 2.f, ty + 2.f);
			temp2[0 * 4 + 3] = temp1[0 * 4 + 3] + tex2D(tex_m1_L2_3, tx + 3.f, ty + 2.f);

			//for (int k = 0; k < map_count; ++k)
			{
				c[4 * 0 + 0] += temp1[1 * 4 + 0] * kernels_L2_m1[0][3]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[0][4]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[0][5];

				c[4 * 0 + 1] += temp1[1 * 4 + 1] * kernels_L2_m1[0][3]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[0][4]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[0][5];

				c[4 * 0 + 2] += temp1[0 * 4 + 0] * kernels_L2_m1[0][3]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[0][4]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[0][5];

				c[4 * 0 + 3] += temp1[0 * 4 + 1] * kernels_L2_m1[0][3]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[0][4]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[0][5];
			}
			{
				c[4 * 1 + 0] += temp1[1 * 4 + 0] * kernels_L2_m1[1][3]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[1][4]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[1][5];

				c[4 * 1 + 1] += temp1[1 * 4 + 1] * kernels_L2_m1[1][3]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[1][4]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[1][5];

				c[4 * 1 + 2] += temp1[0 * 4 + 0] * kernels_L2_m1[1][3]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[1][4]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[1][5];

				c[4 * 1 + 3] += temp1[0 * 4 + 1] * kernels_L2_m1[1][3]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[1][4]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[1][5];
			}
			{
				c[4 * 2 + 0] += temp2[1 * 4 + 0] * kernels_L2_m1[2][3]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[2][4]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[2][5];

				c[4 * 2 + 1] += temp2[1 * 4 + 1] * kernels_L2_m1[2][3]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[2][4]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[2][5];

				c[4 * 2 + 2] += temp2[0 * 4 + 0] * kernels_L2_m1[2][3]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[2][4]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[2][5];

				c[4 * 2 + 3] += temp2[0 * 4 + 1] * kernels_L2_m1[2][3]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[2][4]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[2][5];
			}
			{
				c[4 * 3 + 0] += temp2[1 * 4 + 0] * kernels_L2_m1[3][3]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[3][4]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[3][5];

				c[4 * 3 + 1] += temp2[1 * 4 + 1] * kernels_L2_m1[3][3]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[3][4]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[3][5];

				c[4 * 3 + 2] += temp2[0 * 4 + 0] * kernels_L2_m1[3][3]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[3][4]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[3][5];

				c[4 * 3 + 3] += temp2[0 * 4 + 1] * kernels_L2_m1[3][3]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[3][4]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[3][5];
			}

			temp1[1 * 4 + 0] = tex2D(tex_m1_L2_1, tx + 0.f, ty + 3.f);
			temp1[1 * 4 + 1] = tex2D(tex_m1_L2_1, tx + 1.f, ty + 3.f);
			temp1[1 * 4 + 2] = tex2D(tex_m1_L2_1, tx + 2.f, ty + 3.f);
			temp1[1 * 4 + 3] = tex2D(tex_m1_L2_1, tx + 3.f, ty + 3.f);
			temp1[1 * 4 + 0] += tex2D(tex_m1_L2_2, tx + 0.f, ty + 3.f);
			temp1[1 * 4 + 1] += tex2D(tex_m1_L2_2, tx + 1.f, ty + 3.f);
			temp1[1 * 4 + 2] += tex2D(tex_m1_L2_2, tx + 2.f, ty + 3.f);
			temp1[1 * 4 + 3] += tex2D(tex_m1_L2_2, tx + 3.f, ty + 3.f);

			temp2[1 * 4 + 0] = temp1[1 * 4 + 0] + tex2D(tex_m1_L2_3, tx + 0.f, ty + 3.f);
			temp2[1 * 4 + 1] = temp1[1 * 4 + 1] + tex2D(tex_m1_L2_3, tx + 1.f, ty + 3.f);
			temp2[1 * 4 + 2] = temp1[1 * 4 + 2] + tex2D(tex_m1_L2_3, tx + 2.f, ty + 3.f);
			temp2[1 * 4 + 3] = temp1[1 * 4 + 3] + tex2D(tex_m1_L2_3, tx + 3.f, ty + 3.f);

			//for (int k = 0; k < map_count; ++k)
			{
				c[4 * 0 + 0] += temp1[0 * 4 + 0] * kernels_L2_m1[0][6]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[0][7]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[0][8];

				c[4 * 0 + 1] += temp1[0 * 4 + 1] * kernels_L2_m1[0][6]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[0][7]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[0][8];

				c[4 * 0 + 2] += temp1[1 * 4 + 0] * kernels_L2_m1[0][6]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[0][7]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[0][8];

				c[4 * 0 + 3] += temp1[1 * 4 + 1] * kernels_L2_m1[0][6]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[0][7]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[0][8];

				c[4 * 0 + 0] += conv_b_L2_m1[0];
				c[4 * 0 + 0] = lrelu_w1_L2_m1[0] * c[4 * 0 + 0] + lrelu_w2_L2_m1[0] * fmaxf(0.f, c[4 * 0 + 0]) + bn_b_L2_m1[0];

				c[4 * 0 + 1] += conv_b_L2_m1[0];
				c[4 * 0 + 1] = lrelu_w1_L2_m1[0] * c[4 * 0 + 1] + lrelu_w2_L2_m1[0] * fmaxf(0.f, c[4 * 0 + 1]) + bn_b_L2_m1[0];

				c[4 * 0 + 2] += conv_b_L2_m1[0];
				c[4 * 0 + 2] = lrelu_w1_L2_m1[0] * c[4 * 0 + 2] + lrelu_w2_L2_m1[0] * fmaxf(0.f, c[4 * 0 + 2]) + bn_b_L2_m1[0];

				c[4 * 0 + 3] += conv_b_L2_m1[0];
				c[4 * 0 + 3] = lrelu_w1_L2_m1[0] * c[4 * 0 + 3] + lrelu_w2_L2_m1[0] * fmaxf(0.f, c[4 * 0 + 3]) + bn_b_L2_m1[0];

				surf_ref.surf[0][offset] = fmaxf(fmaxf(c[4 * 0 + 0], c[4 * 0 + 1]), fmaxf(c[4 * 0 + 2], c[4 * 0 + 3]));
			}
			{
				c[4 * 1 + 0] += temp1[0 * 4 + 0] * kernels_L2_m1[1][6]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[1][7]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[1][8];

				c[4 * 1 + 1] += temp1[0 * 4 + 1] * kernels_L2_m1[1][6]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[1][7]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[1][8];

				c[4 * 1 + 2] += temp1[1 * 4 + 0] * kernels_L2_m1[1][6]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[1][7]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[1][8];

				c[4 * 1 + 3] += temp1[1 * 4 + 1] * kernels_L2_m1[1][6]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[1][7]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[1][8];

				c[4 * 1 + 0] += conv_b_L2_m1[1];
				c[4 * 1 + 0] = lrelu_w1_L2_m1[1] * c[4 * 1 + 0] + lrelu_w2_L2_m1[1] * fmaxf(0.f, c[4 * 1 + 0]) + bn_b_L2_m1[1];

				c[4 * 1 + 1] += conv_b_L2_m1[1];
				c[4 * 1 + 1] = lrelu_w1_L2_m1[1] * c[4 * 1 + 1] + lrelu_w2_L2_m1[1] * fmaxf(0.f, c[4 * 1 + 1]) + bn_b_L2_m1[1];

				c[4 * 1 + 2] += conv_b_L2_m1[1];
				c[4 * 1 + 2] = lrelu_w1_L2_m1[1] * c[4 * 1 + 2] + lrelu_w2_L2_m1[1] * fmaxf(0.f, c[4 * 1 + 2]) + bn_b_L2_m1[1];

				c[4 * 1 + 3] += conv_b_L2_m1[1];
				c[4 * 1 + 3] = lrelu_w1_L2_m1[1] * c[4 * 1 + 3] + lrelu_w2_L2_m1[1] * fmaxf(0.f, c[4 * 1 + 3]) + bn_b_L2_m1[1];

				surf_ref.surf[1][offset] = fmaxf(fmaxf(c[4 * 1 + 0], c[4 * 1 + 1]), fmaxf(c[4 * 1 + 2], c[4 * 1 + 3]));
			}
			{
				c[4 * 2 + 0] += temp2[0 * 4 + 0] * kernels_L2_m1[2][6]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[2][7]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[2][8];

				c[4 * 2 + 1] += temp2[0 * 4 + 1] * kernels_L2_m1[2][6]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[2][7]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[2][8];

				c[4 * 2 + 2] += temp2[1 * 4 + 0] * kernels_L2_m1[2][6]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[2][7]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[2][8];

				c[4 * 2 + 3] += temp2[1 * 4 + 1] * kernels_L2_m1[2][6]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[2][7]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[2][8];

				c[4 * 2 + 0] += conv_b_L2_m1[2];
				c[4 * 2 + 0] = lrelu_w1_L2_m1[2] * c[4 * 2 + 0] + lrelu_w2_L2_m1[1] * fmaxf(0.f, c[4 * 2 + 0]) + bn_b_L2_m1[2];

				c[4 * 2 + 1] += conv_b_L2_m1[2];
				c[4 * 2 + 1] = lrelu_w1_L2_m1[2] * c[4 * 2 + 1] + lrelu_w2_L2_m1[1] * fmaxf(0.f, c[4 * 2 + 1]) + bn_b_L2_m1[2];

				c[4 * 2 + 2] += conv_b_L2_m1[2];
				c[4 * 2 + 2] = lrelu_w1_L2_m1[2] * c[4 * 2 + 2] + lrelu_w2_L2_m1[1] * fmaxf(0.f, c[4 * 2 + 2]) + bn_b_L2_m1[2];

				c[4 * 2 + 3] += conv_b_L2_m1[2];
				c[4 * 2 + 3] = lrelu_w1_L2_m1[2] * c[4 * 2 + 3] + lrelu_w2_L2_m1[1] * fmaxf(0.f, c[4 * 2 + 3]) + bn_b_L2_m1[2];

				surf_ref.surf[2][offset] = fmaxf(fmaxf(c[4 * 2 + 0], c[4 * 2 + 1]), fmaxf(c[4 * 2 + 2], c[4 * 2 + 3]));
			}
			{
				c[4 * 3 + 0] += temp2[0 * 4 + 0] * kernels_L2_m1[3][6]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[3][7]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[3][8];

				c[4 * 3 + 1] += temp2[0 * 4 + 1] * kernels_L2_m1[3][6]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[3][7]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[3][8];

				c[4 * 3 + 2] += temp2[1 * 4 + 0] * kernels_L2_m1[3][6]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[3][7]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[3][8];

				c[4 * 3 + 3] += temp2[1 * 4 + 1] * kernels_L2_m1[3][6]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[3][7]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[3][8];

				c[4 * 3 + 0] += conv_b_L2_m1[3];
				c[4 * 3 + 0] = lrelu_w1_L2_m1[3] * c[4 * 3 + 0] + lrelu_w2_L2_m1[3] * fmaxf(0.f, c[4 * 3 + 0]) + bn_b_L2_m1[3];

				c[4 * 3 + 1] += conv_b_L2_m1[3];
				c[4 * 3 + 1] = lrelu_w1_L2_m1[3] * c[4 * 3 + 1] + lrelu_w2_L2_m1[3] * fmaxf(0.f, c[4 * 3 + 1]) + bn_b_L2_m1[3];

				c[4 * 3 + 2] += conv_b_L2_m1[3];
				c[4 * 3 + 2] = lrelu_w1_L2_m1[3] * c[4 * 3 + 2] + lrelu_w2_L2_m1[3] * fmaxf(0.f, c[4 * 3 + 2]) + bn_b_L2_m1[3];

				c[4 * 3 + 3] += conv_b_L2_m1[3];
				c[4 * 3 + 3] = lrelu_w1_L2_m1[3] * c[4 * 3 + 3] + lrelu_w2_L2_m1[3] * fmaxf(0.f, c[4 * 3 + 3]) + bn_b_L2_m1[3];

				surf_ref.surf[3][offset] = fmaxf(fmaxf(c[4 * 3 + 0], c[4 * 3 + 1]), fmaxf(c[4 * 3 + 2], c[4 * 3 + 3]));
			}
		}
		__global__ void add_2_3_conv_4x3x3_lrelu_bn_max_R_cu(SurfRef surf_ref)
		{
			const int i = blockIdx.x * 2 * 16;
			const int j = blockIdx.y * 2 * 16;

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const float tx = i + 2 * trdX + 0.5f;
			const float ty = j + 2 * trdY + 0.5f;

			const int offset = ((j >> 1) + trdY) * surf_ref.step + (i >> 1) + trdX;

			float temp1[8];
			temp1[0 * 4 + 0] = tex2D(tex_m1_L2_4, tx + 0.f, ty + 0.f);
			temp1[0 * 4 + 1] = tex2D(tex_m1_L2_4, tx + 1.f, ty + 0.f);
			temp1[0 * 4 + 2] = tex2D(tex_m1_L2_4, tx + 2.f, ty + 0.f);
			temp1[0 * 4 + 3] = tex2D(tex_m1_L2_4, tx + 3.f, ty + 0.f);
			temp1[1 * 4 + 0] = tex2D(tex_m1_L2_4, tx + 0.f, ty + 1.f);
			temp1[1 * 4 + 1] = tex2D(tex_m1_L2_4, tx + 1.f, ty + 1.f);
			temp1[1 * 4 + 2] = tex2D(tex_m1_L2_4, tx + 2.f, ty + 1.f);
			temp1[1 * 4 + 3] = tex2D(tex_m1_L2_4, tx + 3.f, ty + 1.f);
			temp1[0 * 4 + 0] += tex2D(tex_m1_L2_3, tx + 0.f, ty + 0.f);
			temp1[0 * 4 + 1] += tex2D(tex_m1_L2_3, tx + 1.f, ty + 0.f);
			temp1[0 * 4 + 2] += tex2D(tex_m1_L2_3, tx + 2.f, ty + 0.f);
			temp1[0 * 4 + 3] += tex2D(tex_m1_L2_3, tx + 3.f, ty + 0.f);
			temp1[1 * 4 + 0] += tex2D(tex_m1_L2_3, tx + 0.f, ty + 1.f);
			temp1[1 * 4 + 1] += tex2D(tex_m1_L2_3, tx + 1.f, ty + 1.f);
			temp1[1 * 4 + 2] += tex2D(tex_m1_L2_3, tx + 2.f, ty + 1.f);
			temp1[1 * 4 + 3] += tex2D(tex_m1_L2_3, tx + 3.f, ty + 1.f);

			float temp2[8];
			temp2[0 * 4 + 0] = temp1[0 * 4 + 0] + tex2D(tex_m1_L2_2, tx + 0.f, ty + 0.f);
			temp2[0 * 4 + 1] = temp1[0 * 4 + 1] + tex2D(tex_m1_L2_2, tx + 1.f, ty + 0.f);
			temp2[0 * 4 + 2] = temp1[0 * 4 + 2] + tex2D(tex_m1_L2_2, tx + 2.f, ty + 0.f);
			temp2[0 * 4 + 3] = temp1[0 * 4 + 3] + tex2D(tex_m1_L2_2, tx + 3.f, ty + 0.f);
			temp2[1 * 4 + 0] = temp1[1 * 4 + 0] + tex2D(tex_m1_L2_2, tx + 0.f, ty + 1.f);
			temp2[1 * 4 + 1] = temp1[1 * 4 + 1] + tex2D(tex_m1_L2_2, tx + 1.f, ty + 1.f);
			temp2[1 * 4 + 2] = temp1[1 * 4 + 2] + tex2D(tex_m1_L2_2, tx + 2.f, ty + 1.f);
			temp2[1 * 4 + 3] = temp1[1 * 4 + 3] + tex2D(tex_m1_L2_2, tx + 3.f, ty + 1.f);

			float c[4 * 4];
			//for (int k = 0; k < map_count; ++k)
			{
				c[4 * 0 + 0] = temp1[0 * 4 + 0] * kernels_L2_m1[7][0]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[7][1]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[7][2];

				c[4 * 0 + 1] = temp1[0 * 4 + 1] * kernels_L2_m1[7][0]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[7][1]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[7][2];

				c[4 * 0 + 2] = temp1[1 * 4 + 0] * kernels_L2_m1[7][0]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[7][1]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[7][2];

				c[4 * 0 + 3] = temp1[1 * 4 + 1] * kernels_L2_m1[7][0]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[7][1]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[7][2];
			}
			{
				c[4 * 1 + 0] = temp1[0 * 4 + 0] * kernels_L2_m1[6][0]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[6][1]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[6][2];

				c[4 * 1 + 1] = temp1[0 * 4 + 1] * kernels_L2_m1[6][0]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[6][1]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[6][2];

				c[4 * 1 + 2] = temp1[1 * 4 + 0] * kernels_L2_m1[6][0]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[6][1]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[6][2];

				c[4 * 1 + 3] = temp1[1 * 4 + 1] * kernels_L2_m1[6][0]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[6][1]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[6][2];
			}
			{
				c[4 * 2 + 0] = temp2[0 * 4 + 0] * kernels_L2_m1[5][0]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[5][1]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[5][2];

				c[4 * 2 + 1] = temp2[0 * 4 + 1] * kernels_L2_m1[5][0]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[5][1]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[5][2];

				c[4 * 2 + 2] = temp2[1 * 4 + 0] * kernels_L2_m1[5][0]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[5][1]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[5][2];

				c[4 * 2 + 3] = temp2[1 * 4 + 1] * kernels_L2_m1[5][0]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[5][1]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[5][2];
			}
			{
				c[4 * 3 + 0] = temp2[0 * 4 + 0] * kernels_L2_m1[4][0]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[4][1]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[4][2];

				c[4 * 3 + 1] = temp2[0 * 4 + 1] * kernels_L2_m1[4][0]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[4][1]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[4][2];

				c[4 * 3 + 2] = temp2[1 * 4 + 0] * kernels_L2_m1[4][0]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[4][1]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[4][2];

				c[4 * 3 + 3] = temp2[1 * 4 + 1] * kernels_L2_m1[4][0]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[4][1]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[4][2];
			}

			temp1[0 * 4 + 0] = tex2D(tex_m1_L2_4, tx + 0.f, ty + 2.f);
			temp1[0 * 4 + 1] = tex2D(tex_m1_L2_4, tx + 1.f, ty + 2.f);
			temp1[0 * 4 + 2] = tex2D(tex_m1_L2_4, tx + 2.f, ty + 2.f);
			temp1[0 * 4 + 3] = tex2D(tex_m1_L2_4, tx + 3.f, ty + 2.f);
			temp1[0 * 4 + 0] += tex2D(tex_m1_L2_3, tx + 0.f, ty + 2.f);
			temp1[0 * 4 + 1] += tex2D(tex_m1_L2_3, tx + 1.f, ty + 2.f);
			temp1[0 * 4 + 2] += tex2D(tex_m1_L2_3, tx + 2.f, ty + 2.f);
			temp1[0 * 4 + 3] += tex2D(tex_m1_L2_3, tx + 3.f, ty + 2.f);

			temp2[0 * 4 + 0] = temp1[0 * 4 + 0] + tex2D(tex_m1_L2_2, tx + 0.f, ty + 2.f);
			temp2[0 * 4 + 1] = temp1[0 * 4 + 1] + tex2D(tex_m1_L2_2, tx + 1.f, ty + 2.f);
			temp2[0 * 4 + 2] = temp1[0 * 4 + 2] + tex2D(tex_m1_L2_2, tx + 2.f, ty + 2.f);
			temp2[0 * 4 + 3] = temp1[0 * 4 + 3] + tex2D(tex_m1_L2_2, tx + 3.f, ty + 2.f);

			//for (int k = 0; k < map_count; ++k)
			{
				c[4 * 0 + 0] += temp1[1 * 4 + 0] * kernels_L2_m1[7][3]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[7][4]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[7][5];

				c[4 * 0 + 1] += temp1[1 * 4 + 1] * kernels_L2_m1[7][3]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[7][4]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[7][5];

				c[4 * 0 + 2] += temp1[0 * 4 + 0] * kernels_L2_m1[7][3]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[7][4]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[7][5];

				c[4 * 0 + 3] += temp1[0 * 4 + 1] * kernels_L2_m1[7][3]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[7][4]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[7][5];
			}
			{
				c[4 * 1 + 0] += temp1[1 * 4 + 0] * kernels_L2_m1[6][3]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[6][4]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[6][5];

				c[4 * 1 + 1] += temp1[1 * 4 + 1] * kernels_L2_m1[6][3]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[6][4]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[6][5];

				c[4 * 1 + 2] += temp1[0 * 4 + 0] * kernels_L2_m1[6][3]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[6][4]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[6][5];

				c[4 * 1 + 3] += temp1[0 * 4 + 1] * kernels_L2_m1[6][3]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[6][4]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[6][5];
			}
			{
				c[4 * 2 + 0] += temp2[1 * 4 + 0] * kernels_L2_m1[5][3]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[5][4]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[5][5];

				c[4 * 2 + 1] += temp2[1 * 4 + 1] * kernels_L2_m1[5][3]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[5][4]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[5][5];

				c[4 * 2 + 2] += temp2[0 * 4 + 0] * kernels_L2_m1[5][3]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[5][4]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[5][5];

				c[4 * 2 + 3] += temp2[0 * 4 + 1] * kernels_L2_m1[5][3]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[5][4]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[5][5];
			}
			{
				c[4 * 3 + 0] += temp2[1 * 4 + 0] * kernels_L2_m1[4][3]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[4][4]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[4][5];

				c[4 * 3 + 1] += temp2[1 * 4 + 1] * kernels_L2_m1[4][3]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[4][4]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[4][5];

				c[4 * 3 + 2] += temp2[0 * 4 + 0] * kernels_L2_m1[4][3]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[4][4]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[4][5];

				c[4 * 3 + 3] += temp2[0 * 4 + 1] * kernels_L2_m1[4][3]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[4][4]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[4][5];
			}

			temp1[1 * 4 + 0] = tex2D(tex_m1_L2_4, tx + 0.f, ty + 3.f);
			temp1[1 * 4 + 1] = tex2D(tex_m1_L2_4, tx + 1.f, ty + 3.f);
			temp1[1 * 4 + 2] = tex2D(tex_m1_L2_4, tx + 2.f, ty + 3.f);
			temp1[1 * 4 + 3] = tex2D(tex_m1_L2_4, tx + 3.f, ty + 3.f);
			temp1[1 * 4 + 0] += tex2D(tex_m1_L2_3, tx + 0.f, ty + 3.f);
			temp1[1 * 4 + 1] += tex2D(tex_m1_L2_3, tx + 1.f, ty + 3.f);
			temp1[1 * 4 + 2] += tex2D(tex_m1_L2_3, tx + 2.f, ty + 3.f);
			temp1[1 * 4 + 3] += tex2D(tex_m1_L2_3, tx + 3.f, ty + 3.f);

			temp2[1 * 4 + 0] = temp1[1 * 4 + 0] + tex2D(tex_m1_L2_2, tx + 0.f, ty + 3.f);
			temp2[1 * 4 + 1] = temp1[1 * 4 + 1] + tex2D(tex_m1_L2_2, tx + 1.f, ty + 3.f);
			temp2[1 * 4 + 2] = temp1[1 * 4 + 2] + tex2D(tex_m1_L2_2, tx + 2.f, ty + 3.f);
			temp2[1 * 4 + 3] = temp1[1 * 4 + 3] + tex2D(tex_m1_L2_2, tx + 3.f, ty + 3.f);

			//for (int k = 0; k < map_count; ++k)
			{
				c[4 * 0 + 0] += temp1[0 * 4 + 0] * kernels_L2_m1[7][6]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[7][7]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[7][8];

				c[4 * 0 + 1] += temp1[0 * 4 + 1] * kernels_L2_m1[7][6]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[7][7]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[7][8];

				c[4 * 0 + 2] += temp1[1 * 4 + 0] * kernels_L2_m1[7][6]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[7][7]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[7][8];

				c[4 * 0 + 3] += temp1[1 * 4 + 1] * kernels_L2_m1[7][6]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[7][7]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[7][8];

				c[4 * 0 + 0] += conv_b_L2_m1[7];
				c[4 * 0 + 0] = lrelu_w1_L2_m1[7] * c[4 * 0 + 0] + lrelu_w2_L2_m1[7] * fmaxf(0.f, c[4 * 0 + 0]) + bn_b_L2_m1[7];

				c[4 * 0 + 1] += conv_b_L2_m1[7];
				c[4 * 0 + 1] = lrelu_w1_L2_m1[7] * c[4 * 0 + 1] + lrelu_w2_L2_m1[7] * fmaxf(0.f, c[4 * 0 + 1]) + bn_b_L2_m1[7];

				c[4 * 0 + 2] += conv_b_L2_m1[7];
				c[4 * 0 + 2] = lrelu_w1_L2_m1[7] * c[4 * 0 + 2] + lrelu_w2_L2_m1[7] * fmaxf(0.f, c[4 * 0 + 2]) + bn_b_L2_m1[7];

				c[4 * 0 + 3] += conv_b_L2_m1[7];
				c[4 * 0 + 3] = lrelu_w1_L2_m1[7] * c[4 * 0 + 3] + lrelu_w2_L2_m1[7] * fmaxf(0.f, c[4 * 0 + 3]) + bn_b_L2_m1[7];

				surf_ref.surf[7][offset] = fmaxf(fmaxf(c[4 * 0 + 0], c[4 * 0 + 1]), fmaxf(c[4 * 0 + 2], c[4 * 0 + 3]));
			}
			{
				c[4 * 1 + 0] += temp1[0 * 4 + 0] * kernels_L2_m1[6][6]
					+ temp1[0 * 4 + 1] * kernels_L2_m1[6][7]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[6][8];

				c[4 * 1 + 1] += temp1[0 * 4 + 1] * kernels_L2_m1[6][6]
					+ temp1[0 * 4 + 2] * kernels_L2_m1[6][7]
					+ temp1[0 * 4 + 3] * kernels_L2_m1[6][8];

				c[4 * 1 + 2] += temp1[1 * 4 + 0] * kernels_L2_m1[6][6]
					+ temp1[1 * 4 + 1] * kernels_L2_m1[6][7]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[6][8];

				c[4 * 1 + 3] += temp1[1 * 4 + 1] * kernels_L2_m1[6][6]
					+ temp1[1 * 4 + 2] * kernels_L2_m1[6][7]
					+ temp1[1 * 4 + 3] * kernels_L2_m1[6][8];

				c[4 * 1 + 0] += conv_b_L2_m1[6];
				c[4 * 1 + 0] = lrelu_w1_L2_m1[6] * c[4 * 1 + 0] + lrelu_w2_L2_m1[6] * fmaxf(0.f, c[4 * 1 + 0]) + bn_b_L2_m1[6];

				c[4 * 1 + 1] += conv_b_L2_m1[6];
				c[4 * 1 + 1] = lrelu_w1_L2_m1[6] * c[4 * 1 + 1] + lrelu_w2_L2_m1[6] * fmaxf(0.f, c[4 * 1 + 1]) + bn_b_L2_m1[6];

				c[4 * 1 + 2] += conv_b_L2_m1[6];
				c[4 * 1 + 2] = lrelu_w1_L2_m1[6] * c[4 * 1 + 2] + lrelu_w2_L2_m1[6] * fmaxf(0.f, c[4 * 1 + 2]) + bn_b_L2_m1[6];

				c[4 * 1 + 3] += conv_b_L2_m1[6];
				c[4 * 1 + 3] = lrelu_w1_L2_m1[6] * c[4 * 1 + 3] + lrelu_w2_L2_m1[6] * fmaxf(0.f, c[4 * 1 + 3]) + bn_b_L2_m1[6];

				surf_ref.surf[6][offset] = fmaxf(fmaxf(c[4 * 1 + 0], c[4 * 1 + 1]), fmaxf(c[4 * 1 + 2], c[4 * 1 + 3]));
			}
			{
				c[4 * 2 + 0] += temp2[0 * 4 + 0] * kernels_L2_m1[5][6]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[5][7]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[5][8];

				c[4 * 2 + 1] += temp2[0 * 4 + 1] * kernels_L2_m1[5][6]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[5][7]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[5][8];

				c[4 * 2 + 2] += temp2[1 * 4 + 0] * kernels_L2_m1[5][6]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[5][7]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[5][8];

				c[4 * 2 + 3] += temp2[1 * 4 + 1] * kernels_L2_m1[5][6]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[5][7]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[5][8];

				c[4 * 2 + 0] += conv_b_L2_m1[5];
				c[4 * 2 + 0] = lrelu_w1_L2_m1[5] * c[4 * 2 + 0] + lrelu_w2_L2_m1[5] * fmaxf(0.f, c[4 * 2 + 0]) + bn_b_L2_m1[5];

				c[4 * 2 + 1] += conv_b_L2_m1[5];
				c[4 * 2 + 1] = lrelu_w1_L2_m1[5] * c[4 * 2 + 1] + lrelu_w2_L2_m1[5] * fmaxf(0.f, c[4 * 2 + 1]) + bn_b_L2_m1[5];

				c[4 * 2 + 2] += conv_b_L2_m1[5];
				c[4 * 2 + 2] = lrelu_w1_L2_m1[5] * c[4 * 2 + 2] + lrelu_w2_L2_m1[5] * fmaxf(0.f, c[4 * 2 + 2]) + bn_b_L2_m1[5];

				c[4 * 2 + 3] += conv_b_L2_m1[5];
				c[4 * 2 + 3] = lrelu_w1_L2_m1[5] * c[4 * 2 + 3] + lrelu_w2_L2_m1[5] * fmaxf(0.f, c[4 * 2 + 3]) + bn_b_L2_m1[5];

				surf_ref.surf[5][offset] = fmaxf(fmaxf(c[4 * 2 + 0], c[4 * 2 + 1]), fmaxf(c[4 * 2 + 2], c[4 * 2 + 3]));
			}
			{
				c[4 * 3 + 0] += temp2[0 * 4 + 0] * kernels_L2_m1[4][6]
					+ temp2[0 * 4 + 1] * kernels_L2_m1[4][7]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[4][8];

				c[4 * 3 + 1] += temp2[0 * 4 + 1] * kernels_L2_m1[4][6]
					+ temp2[0 * 4 + 2] * kernels_L2_m1[4][7]
					+ temp2[0 * 4 + 3] * kernels_L2_m1[4][8];

				c[4 * 3 + 2] += temp2[1 * 4 + 0] * kernels_L2_m1[4][6]
					+ temp2[1 * 4 + 1] * kernels_L2_m1[4][7]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[4][8];

				c[4 * 3 + 3] += temp2[1 * 4 + 1] * kernels_L2_m1[4][6]
					+ temp2[1 * 4 + 2] * kernels_L2_m1[4][7]
					+ temp2[1 * 4 + 3] * kernels_L2_m1[4][8];

				c[4 * 3 + 0] += conv_b_L2_m1[4];
				c[4 * 3 + 0] = lrelu_w1_L2_m1[4] * c[4 * 3 + 0] + lrelu_w2_L2_m1[4] * fmaxf(0.f, c[4 * 3 + 0]) + bn_b_L2_m1[4];

				c[4 * 3 + 1] += conv_b_L2_m1[4];
				c[4 * 3 + 1] = lrelu_w1_L2_m1[4] * c[4 * 3 + 1] + lrelu_w2_L2_m1[4] * fmaxf(0.f, c[4 * 3 + 1]) + bn_b_L2_m1[4];

				c[4 * 3 + 2] += conv_b_L2_m1[4];
				c[4 * 3 + 2] = lrelu_w1_L2_m1[4] * c[4 * 3 + 2] + lrelu_w2_L2_m1[4] * fmaxf(0.f, c[4 * 3 + 2]) + bn_b_L2_m1[4];

				c[4 * 3 + 3] += conv_b_L2_m1[4];
				c[4 * 3 + 3] = lrelu_w1_L2_m1[4] * c[4 * 3 + 3] + lrelu_w2_L2_m1[4] * fmaxf(0.f, c[4 * 3 + 3]) + bn_b_L2_m1[4];

				surf_ref.surf[4][offset] = fmaxf(fmaxf(c[4 * 3 + 0], c[4 * 3 + 1]), fmaxf(c[4 * 3 + 2], c[4 * 3 + 3]));
			}
		}

		//L3
		__global__ void add_2_3_conv_4x5x4_L_cu(SurfRef surf_ref)
		{
			__shared__ float temp1[(2 * 16) * (2 * 16)];
			__shared__ float temp2[(2 * 16) * (2 * 16)];

			const int i = blockIdx.x * (2 * 16 - 4 /*+ 1*/);
			const int j = blockIdx.y * (2 * 16 - 5 + 1);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);
			int offset = (j + 2 * trdY) * surf_ref.step + i + 2 * trdX;

			const int z1 = trdX + id_walf_w * 16;
			const int z2 = (4 * ry) * (2 * 16) + z1;
			const float tx = i + z1 + 0.5f;
			const float ty = j + 4 * ry + 0.5f;
			temp1[z2 + 0 * (2 * 16)] = tex2D(tex_m1_L3_1, tx, ty + 0.f) + tex2D(tex_m1_L3_2, tx, ty + 0.f);
			temp1[z2 + 1 * (2 * 16)] = tex2D(tex_m1_L3_1, tx, ty + 1.f) + tex2D(tex_m1_L3_2, tx, ty + 1.f);
			temp1[z2 + 2 * (2 * 16)] = tex2D(tex_m1_L3_1, tx, ty + 2.f) + tex2D(tex_m1_L3_2, tx, ty + 2.f);
			temp1[z2 + 3 * (2 * 16)] = tex2D(tex_m1_L3_1, tx, ty + 3.f) + tex2D(tex_m1_L3_2, tx, ty + 3.f);

			temp2[z2 + 0 * (2 * 16)] = temp1[z2 + 0 * (2 * 16)] + tex2D(tex_m1_L3_3, tx, ty + 0.f);
			temp2[z2 + 1 * (2 * 16)] = temp1[z2 + 1 * (2 * 16)] + tex2D(tex_m1_L3_3, tx, ty + 1.f);
			temp2[z2 + 2 * (2 * 16)] = temp1[z2 + 2 * (2 * 16)] + tex2D(tex_m1_L3_3, tx, ty + 2.f);
			temp2[z2 + 3 * (2 * 16)] = temp1[z2 + 3 * (2 * 16)] + tex2D(tex_m1_L3_3, tx, ty + 3.f);

			__syncthreads();

			if (2 * trdX > 16 * 2 - 5 || 2 * trdY > 2 * 16 - 5) return;

			const float* ptr_temp1 = temp1 + 2 * trdX + (2 * 16) * 2 * trdY;
			const float* ptr_temp2 = temp2 + 2 * trdX + (2 * 16) * 2 * trdY;

			float c0 = *(ptr_temp1 + 0 * (2 * 16) + 0) * kernels_L3_m1[0][0]
				+* (ptr_temp1 + 0 * (2 * 16) + 1) * kernels_L3_m1[0][1]
				+* (ptr_temp1 + 0 * (2 * 16) + 2) * kernels_L3_m1[0][2]
				+* (ptr_temp1 + 0 * (2 * 16) + 3) * kernels_L3_m1[0][3]
				+* (ptr_temp1 + 1 * (2 * 16) + 0) * kernels_L3_m1[0][4]
				+* (ptr_temp1 + 1 * (2 * 16) + 1) * kernels_L3_m1[0][5]
				+* (ptr_temp1 + 1 * (2 * 16) + 2) * kernels_L3_m1[0][6]
				+* (ptr_temp1 + 1 * (2 * 16) + 3) * kernels_L3_m1[0][7]
				+* (ptr_temp1 + 2 * (2 * 16) + 0) * kernels_L3_m1[0][8]
				+* (ptr_temp1 + 2 * (2 * 16) + 1) * kernels_L3_m1[0][9]
				+* (ptr_temp1 + 2 * (2 * 16) + 2) * kernels_L3_m1[0][10]
				+* (ptr_temp1 + 2 * (2 * 16) + 3) * kernels_L3_m1[0][11]
				+* (ptr_temp1 + 3 * (2 * 16) + 0) * kernels_L3_m1[0][12]
				+* (ptr_temp1 + 3 * (2 * 16) + 1) * kernels_L3_m1[0][13]
				+* (ptr_temp1 + 3 * (2 * 16) + 2) * kernels_L3_m1[0][14]
				+* (ptr_temp1 + 3 * (2 * 16) + 3) * kernels_L3_m1[0][15]
				+* (ptr_temp1 + 4 * (2 * 16) + 0) * kernels_L3_m1[0][16]
				+* (ptr_temp1 + 4 * (2 * 16) + 1) * kernels_L3_m1[0][17]
				+* (ptr_temp1 + 4 * (2 * 16) + 2) * kernels_L3_m1[0][18]
				+* (ptr_temp1 + 4 * (2 * 16) + 3) * kernels_L3_m1[0][19];

			float* ptr_dst = (float*)(surf_ref.surf[0]) + offset;
			*ptr_dst = c0;

			float c1 = *(ptr_temp1 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[0][0]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[0][1]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[0][2]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[0][3]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[0][4]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[0][5]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[0][6]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[0][7]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[0][8]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[0][9]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[0][10]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[0][11]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[0][12]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[0][13]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[0][14]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[0][15]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[0][16]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[0][17]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[0][18]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[0][19];

			*(ptr_dst + 1) = c1;

			float c2 = *(ptr_temp1 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[0][0]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[0][1]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[0][2]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[0][3]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[0][4]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[0][5]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[0][6]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[0][7]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[0][8]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[0][9]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[0][10]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[0][11]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[0][12]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[0][13]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[0][14]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[0][15]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[0][16]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[0][17]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[0][18]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[0][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			float c3 = *(ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[0][0]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[0][1]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[0][2]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[0][3]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[0][4]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[0][5]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[0][6]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[0][7]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[0][8]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[0][9]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[0][10]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[0][11]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[0][12]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[0][13]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[0][14]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[0][15]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[0][16]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[0][17]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[0][18]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[0][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp1 + 0 * (2 * 16) + 0) * kernels_L3_m1[1][0]
				+* (ptr_temp1 + 0 * (2 * 16) + 1) * kernels_L3_m1[1][1]
				+* (ptr_temp1 + 0 * (2 * 16) + 2) * kernels_L3_m1[1][2]
				+* (ptr_temp1 + 0 * (2 * 16) + 3) * kernels_L3_m1[1][3]
				+* (ptr_temp1 + 1 * (2 * 16) + 0) * kernels_L3_m1[1][4]
				+* (ptr_temp1 + 1 * (2 * 16) + 1) * kernels_L3_m1[1][5]
				+* (ptr_temp1 + 1 * (2 * 16) + 2) * kernels_L3_m1[1][6]
				+* (ptr_temp1 + 1 * (2 * 16) + 3) * kernels_L3_m1[1][7]
				+* (ptr_temp1 + 2 * (2 * 16) + 0) * kernels_L3_m1[1][8]
				+* (ptr_temp1 + 2 * (2 * 16) + 1) * kernels_L3_m1[1][9]
				+* (ptr_temp1 + 2 * (2 * 16) + 2) * kernels_L3_m1[1][10]
				+* (ptr_temp1 + 2 * (2 * 16) + 3) * kernels_L3_m1[1][11]
				+* (ptr_temp1 + 3 * (2 * 16) + 0) * kernels_L3_m1[1][12]
				+* (ptr_temp1 + 3 * (2 * 16) + 1) * kernels_L3_m1[1][13]
				+* (ptr_temp1 + 3 * (2 * 16) + 2) * kernels_L3_m1[1][14]
				+* (ptr_temp1 + 3 * (2 * 16) + 3) * kernels_L3_m1[1][15]
				+* (ptr_temp1 + 4 * (2 * 16) + 0) * kernels_L3_m1[1][16]
				+* (ptr_temp1 + 4 * (2 * 16) + 1) * kernels_L3_m1[1][17]
				+* (ptr_temp1 + 4 * (2 * 16) + 2) * kernels_L3_m1[1][18]
				+* (ptr_temp1 + 4 * (2 * 16) + 3) * kernels_L3_m1[1][19];

			ptr_dst = (float*)(surf_ref.surf[1]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp1 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[1][0]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[1][1]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[1][2]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[1][3]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[1][4]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[1][5]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[1][6]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[1][7]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[1][8]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[1][9]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[1][10]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[1][11]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[1][12]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[1][13]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[1][14]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[1][15]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[1][16]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[1][17]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[1][18]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[1][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp1 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[1][0]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[1][1]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[1][2]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[1][3]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[1][4]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[1][5]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[1][6]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[1][7]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[1][8]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[1][9]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[1][10]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[1][11]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[1][12]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[1][13]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[1][14]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[1][15]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[1][16]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[1][17]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[1][18]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[1][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[1][0]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[1][1]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[1][2]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[1][3]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[1][4]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[1][5]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[1][6]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[1][7]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[1][8]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[1][9]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[1][10]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[1][11]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[1][12]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[1][13]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[1][14]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[1][15]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[1][16]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[1][17]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[1][18]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[1][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp2 + 0 * (2 * 16) + 0) * kernels_L3_m1[2][0]
				+* (ptr_temp2 + 0 * (2 * 16) + 1) * kernels_L3_m1[2][1]
				+* (ptr_temp2 + 0 * (2 * 16) + 2) * kernels_L3_m1[2][2]
				+* (ptr_temp2 + 0 * (2 * 16) + 3) * kernels_L3_m1[2][3]
				+* (ptr_temp2 + 1 * (2 * 16) + 0) * kernels_L3_m1[2][4]
				+* (ptr_temp2 + 1 * (2 * 16) + 1) * kernels_L3_m1[2][5]
				+* (ptr_temp2 + 1 * (2 * 16) + 2) * kernels_L3_m1[2][6]
				+* (ptr_temp2 + 1 * (2 * 16) + 3) * kernels_L3_m1[2][7]
				+* (ptr_temp2 + 2 * (2 * 16) + 0) * kernels_L3_m1[2][8]
				+* (ptr_temp2 + 2 * (2 * 16) + 1) * kernels_L3_m1[2][9]
				+* (ptr_temp2 + 2 * (2 * 16) + 2) * kernels_L3_m1[2][10]
				+* (ptr_temp2 + 2 * (2 * 16) + 3) * kernels_L3_m1[2][11]
				+* (ptr_temp2 + 3 * (2 * 16) + 0) * kernels_L3_m1[2][12]
				+* (ptr_temp2 + 3 * (2 * 16) + 1) * kernels_L3_m1[2][13]
				+* (ptr_temp2 + 3 * (2 * 16) + 2) * kernels_L3_m1[2][14]
				+* (ptr_temp2 + 3 * (2 * 16) + 3) * kernels_L3_m1[2][15]
				+* (ptr_temp2 + 4 * (2 * 16) + 0) * kernels_L3_m1[2][16]
				+* (ptr_temp2 + 4 * (2 * 16) + 1) * kernels_L3_m1[2][17]
				+* (ptr_temp2 + 4 * (2 * 16) + 2) * kernels_L3_m1[2][18]
				+* (ptr_temp2 + 4 * (2 * 16) + 3) * kernels_L3_m1[2][19];

			ptr_dst = (float*)(surf_ref.surf[2]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp2 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[2][0]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[2][1]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[2][2]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[2][3]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[2][4]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[2][5]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[2][6]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[2][7]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[2][8]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[2][9]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[2][10]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[2][11]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[2][12]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[2][13]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[2][14]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[2][15]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[2][16]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[2][17]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[2][18]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[2][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp2 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[2][0]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[2][1]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[2][2]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[2][3]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[2][4]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[2][5]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[2][6]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[2][7]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[2][8]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[2][9]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[2][10]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[2][11]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[2][12]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[2][13]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[2][14]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[2][15]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[2][16]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[2][17]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[2][18]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[2][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[2][0]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[2][1]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[2][2]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[2][3]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[2][4]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[2][5]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[2][6]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[2][7]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[2][8]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[2][9]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[2][10]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[2][11]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[2][12]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[2][13]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[2][14]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[2][15]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[2][16]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[2][17]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[2][18]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[2][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp2 + 0 * (2 * 16) + 0) * kernels_L3_m1[3][0]
				+* (ptr_temp2 + 0 * (2 * 16) + 1) * kernels_L3_m1[3][1]
				+* (ptr_temp2 + 0 * (2 * 16) + 2) * kernels_L3_m1[3][2]
				+* (ptr_temp2 + 0 * (2 * 16) + 3) * kernels_L3_m1[3][3]
				+* (ptr_temp2 + 1 * (2 * 16) + 0) * kernels_L3_m1[3][4]
				+* (ptr_temp2 + 1 * (2 * 16) + 1) * kernels_L3_m1[3][5]
				+* (ptr_temp2 + 1 * (2 * 16) + 2) * kernels_L3_m1[3][6]
				+* (ptr_temp2 + 1 * (2 * 16) + 3) * kernels_L3_m1[3][7]
				+* (ptr_temp2 + 2 * (2 * 16) + 0) * kernels_L3_m1[3][8]
				+* (ptr_temp2 + 2 * (2 * 16) + 1) * kernels_L3_m1[3][9]
				+* (ptr_temp2 + 2 * (2 * 16) + 2) * kernels_L3_m1[3][10]
				+* (ptr_temp2 + 2 * (2 * 16) + 3) * kernels_L3_m1[3][11]
				+* (ptr_temp2 + 3 * (2 * 16) + 0) * kernels_L3_m1[3][12]
				+* (ptr_temp2 + 3 * (2 * 16) + 1) * kernels_L3_m1[3][13]
				+* (ptr_temp2 + 3 * (2 * 16) + 2) * kernels_L3_m1[3][14]
				+* (ptr_temp2 + 3 * (2 * 16) + 3) * kernels_L3_m1[3][15]
				+* (ptr_temp2 + 4 * (2 * 16) + 0) * kernels_L3_m1[3][16]
				+* (ptr_temp2 + 4 * (2 * 16) + 1) * kernels_L3_m1[3][17]
				+* (ptr_temp2 + 4 * (2 * 16) + 2) * kernels_L3_m1[3][18]
				+* (ptr_temp2 + 4 * (2 * 16) + 3) * kernels_L3_m1[3][19];

			ptr_dst = (float*)(surf_ref.surf[3]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp2 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[3][0]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[3][1]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[3][2]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[3][3]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[3][4]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[3][5]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[3][6]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[3][7]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[3][8]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[3][9]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[3][10]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[3][11]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[3][12]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[3][13]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[3][14]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[3][15]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[3][16]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[3][17]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[3][18]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[3][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp2 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[3][0]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[3][1]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[3][2]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[3][3]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[3][4]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[3][5]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[3][6]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[3][7]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[3][8]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[3][9]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[3][10]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[3][11]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[3][12]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[3][13]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[3][14]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[3][15]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[3][16]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[3][17]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[3][18]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[3][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[3][0]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[3][1]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[3][2]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[3][3]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[3][4]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[3][5]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[3][6]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[3][7]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[3][8]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[3][9]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[3][10]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[3][11]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[3][12]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[3][13]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[3][14]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[3][15]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[3][16]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[3][17]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[3][18]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[3][19];

			*(ptr_dst + 1) = c3;

			__syncthreads();
		}
		__global__ void add_3_conv_4x5x4_1_cu(SurfRef surf_ref)
		{
			__shared__ float temp1[(2 * 16) * (2 * 16)];
			__shared__ float temp2[(2 * 16) * (2 * 16)];

			const int i = blockIdx.x * (2 * 16 - 4 /*+ 1*/);
			const int j = blockIdx.y * (2 * 16 - 5 + 1);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);
			const int offset = (j + 2 * trdY) * surf_ref.step + i + 2 * trdX;

			const int z1 = trdX + id_walf_w * 16;
			const int z2 = (4 * ry) * (2 * 16) + z1;
			const float tx = i + z1 + 0.5f;
			const float ty = j + 4 * ry + 0.5f;
			temp1[z2 + 0 * (2 * 16)] = tex2D(tex_m1_L3_3, tx, ty + 0.f) + tex2D(tex_m1_L3_4, tx, ty + 0.f);
			temp1[z2 + 1 * (2 * 16)] = tex2D(tex_m1_L3_3, tx, ty + 1.f) + tex2D(tex_m1_L3_4, tx, ty + 1.f);
			temp1[z2 + 2 * (2 * 16)] = tex2D(tex_m1_L3_3, tx, ty + 2.f) + tex2D(tex_m1_L3_4, tx, ty + 2.f);
			temp1[z2 + 3 * (2 * 16)] = tex2D(tex_m1_L3_3, tx, ty + 3.f) + tex2D(tex_m1_L3_4, tx, ty + 3.f);

			temp2[z2 + 0 * (2 * 16)] = temp1[z2 + 0 * (2 * 16)] + tex2D(tex_m1_L3_5, tx, ty + 0.f);
			temp2[z2 + 1 * (2 * 16)] = temp1[z2 + 1 * (2 * 16)] + tex2D(tex_m1_L3_5, tx, ty + 1.f);
			temp2[z2 + 2 * (2 * 16)] = temp1[z2 + 2 * (2 * 16)] + tex2D(tex_m1_L3_5, tx, ty + 2.f);
			temp2[z2 + 3 * (2 * 16)] = temp1[z2 + 3 * (2 * 16)] + tex2D(tex_m1_L3_5, tx, ty + 3.f);

			temp1[z2 + 0 * (2 * 16)] += tex2D(tex_m1_L3_2, tx, ty + 0.f);
			temp1[z2 + 1 * (2 * 16)] += tex2D(tex_m1_L3_2, tx, ty + 1.f);
			temp1[z2 + 2 * (2 * 16)] += tex2D(tex_m1_L3_2, tx, ty + 2.f);
			temp1[z2 + 3 * (2 * 16)] += tex2D(tex_m1_L3_2, tx, ty + 3.f);

			__syncthreads();

			if (2 * trdX > 16 * 2 - 5 || 2 * trdY > 2 * 16 - 5) return;

			const float* ptr_temp1 = temp1 + 2 * trdX + (2 * 16) * 2 * trdY;
			const float* ptr_temp2 = temp2 + 2 * trdX + (2 * 16) * 2 * trdY;

			float c0 = *(ptr_temp1 + 0 * (2 * 16) + 0) * kernels_L3_m1[4][0]
				+* (ptr_temp1 + 0 * (2 * 16) + 1) * kernels_L3_m1[4][1]
				+* (ptr_temp1 + 0 * (2 * 16) + 2) * kernels_L3_m1[4][2]
				+* (ptr_temp1 + 0 * (2 * 16) + 3) * kernels_L3_m1[4][3]
				+* (ptr_temp1 + 1 * (2 * 16) + 0) * kernels_L3_m1[4][4]
				+* (ptr_temp1 + 1 * (2 * 16) + 1) * kernels_L3_m1[4][5]
				+* (ptr_temp1 + 1 * (2 * 16) + 2) * kernels_L3_m1[4][6]
				+* (ptr_temp1 + 1 * (2 * 16) + 3) * kernels_L3_m1[4][7]
				+* (ptr_temp1 + 2 * (2 * 16) + 0) * kernels_L3_m1[4][8]
				+* (ptr_temp1 + 2 * (2 * 16) + 1) * kernels_L3_m1[4][9]
				+* (ptr_temp1 + 2 * (2 * 16) + 2) * kernels_L3_m1[4][10]
				+* (ptr_temp1 + 2 * (2 * 16) + 3) * kernels_L3_m1[4][11]
				+* (ptr_temp1 + 3 * (2 * 16) + 0) * kernels_L3_m1[4][12]
				+* (ptr_temp1 + 3 * (2 * 16) + 1) * kernels_L3_m1[4][13]
				+* (ptr_temp1 + 3 * (2 * 16) + 2) * kernels_L3_m1[4][14]
				+* (ptr_temp1 + 3 * (2 * 16) + 3) * kernels_L3_m1[4][15]
				+* (ptr_temp1 + 4 * (2 * 16) + 0) * kernels_L3_m1[4][16]
				+* (ptr_temp1 + 4 * (2 * 16) + 1) * kernels_L3_m1[4][17]
				+* (ptr_temp1 + 4 * (2 * 16) + 2) * kernels_L3_m1[4][18]
				+* (ptr_temp1 + 4 * (2 * 16) + 3) * kernels_L3_m1[4][19];

			float* ptr_dst = (float*)(surf_ref.surf[4]) + offset;
			*ptr_dst = c0;

			float c1 = *(ptr_temp1 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[4][0]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[4][1]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[4][2]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[4][3]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[4][4]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[4][5]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[4][6]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[4][7]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[4][8]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[4][9]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[4][10]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[4][11]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[4][12]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[4][13]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[4][14]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[4][15]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[4][16]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[4][17]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[4][18]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[4][19];

			*(ptr_dst + 1) = c1;

			float c2 = *(ptr_temp1 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[4][0]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[4][1]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[4][2]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[4][3]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[4][4]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[4][5]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[4][6]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[4][7]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[4][8]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[4][9]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[4][10]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[4][11]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[4][12]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[4][13]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[4][14]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[4][15]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[4][16]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[4][17]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[4][18]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[4][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			float c3 = *(ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[4][0]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[4][1]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[4][2]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[4][3]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[4][4]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[4][5]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[4][6]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[4][7]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[4][8]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[4][9]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[4][10]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[4][11]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[4][12]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[4][13]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[4][14]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[4][15]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[4][16]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[4][17]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[4][18]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[4][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp1 + 0 * (2 * 16) + 0) * kernels_L3_m1[5][0]
				+* (ptr_temp1 + 0 * (2 * 16) + 1) * kernels_L3_m1[5][1]
				+* (ptr_temp1 + 0 * (2 * 16) + 2) * kernels_L3_m1[5][2]
				+* (ptr_temp1 + 0 * (2 * 16) + 3) * kernels_L3_m1[5][3]
				+* (ptr_temp1 + 1 * (2 * 16) + 0) * kernels_L3_m1[5][4]
				+* (ptr_temp1 + 1 * (2 * 16) + 1) * kernels_L3_m1[5][5]
				+* (ptr_temp1 + 1 * (2 * 16) + 2) * kernels_L3_m1[5][6]
				+* (ptr_temp1 + 1 * (2 * 16) + 3) * kernels_L3_m1[5][7]
				+* (ptr_temp1 + 2 * (2 * 16) + 0) * kernels_L3_m1[5][8]
				+* (ptr_temp1 + 2 * (2 * 16) + 1) * kernels_L3_m1[5][9]
				+* (ptr_temp1 + 2 * (2 * 16) + 2) * kernels_L3_m1[5][10]
				+* (ptr_temp1 + 2 * (2 * 16) + 3) * kernels_L3_m1[5][11]
				+* (ptr_temp1 + 3 * (2 * 16) + 0) * kernels_L3_m1[5][12]
				+* (ptr_temp1 + 3 * (2 * 16) + 1) * kernels_L3_m1[5][13]
				+* (ptr_temp1 + 3 * (2 * 16) + 2) * kernels_L3_m1[5][14]
				+* (ptr_temp1 + 3 * (2 * 16) + 3) * kernels_L3_m1[5][15]
				+* (ptr_temp1 + 4 * (2 * 16) + 0) * kernels_L3_m1[5][16]
				+* (ptr_temp1 + 4 * (2 * 16) + 1) * kernels_L3_m1[5][17]
				+* (ptr_temp1 + 4 * (2 * 16) + 2) * kernels_L3_m1[5][18]
				+* (ptr_temp1 + 4 * (2 * 16) + 3) * kernels_L3_m1[5][19];

			ptr_dst = (float*)(surf_ref.surf[5]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp1 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[5][0]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[5][1]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[5][2]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[5][3]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[5][4]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[5][5]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[5][6]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[5][7]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[5][8]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[5][9]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[5][10]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[5][11]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[5][12]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[5][13]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[5][14]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[5][15]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[5][16]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[5][17]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[5][18]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[5][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp1 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[5][0]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[5][1]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[5][2]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[5][3]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[5][4]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[5][5]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[5][6]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[5][7]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[5][8]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[5][9]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[5][10]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[5][11]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[5][12]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[5][13]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[5][14]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[5][15]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[5][16]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[5][17]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[5][18]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[5][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[5][0]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[5][1]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[5][2]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[5][3]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[5][4]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[5][5]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[5][6]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[5][7]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[5][8]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[5][9]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[5][10]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[5][11]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[5][12]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[5][13]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[5][14]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[5][15]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[5][16]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[5][17]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[5][18]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[5][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp2 + 0 * (2 * 16) + 0) * kernels_L3_m1[6][0]
				+* (ptr_temp2 + 0 * (2 * 16) + 1) * kernels_L3_m1[6][1]
				+* (ptr_temp2 + 0 * (2 * 16) + 2) * kernels_L3_m1[6][2]
				+* (ptr_temp2 + 0 * (2 * 16) + 3) * kernels_L3_m1[6][3]
				+* (ptr_temp2 + 1 * (2 * 16) + 0) * kernels_L3_m1[6][4]
				+* (ptr_temp2 + 1 * (2 * 16) + 1) * kernels_L3_m1[6][5]
				+* (ptr_temp2 + 1 * (2 * 16) + 2) * kernels_L3_m1[6][6]
				+* (ptr_temp2 + 1 * (2 * 16) + 3) * kernels_L3_m1[6][7]
				+* (ptr_temp2 + 2 * (2 * 16) + 0) * kernels_L3_m1[6][8]
				+* (ptr_temp2 + 2 * (2 * 16) + 1) * kernels_L3_m1[6][9]
				+* (ptr_temp2 + 2 * (2 * 16) + 2) * kernels_L3_m1[6][10]
				+* (ptr_temp2 + 2 * (2 * 16) + 3) * kernels_L3_m1[6][11]
				+* (ptr_temp2 + 3 * (2 * 16) + 0) * kernels_L3_m1[6][12]
				+* (ptr_temp2 + 3 * (2 * 16) + 1) * kernels_L3_m1[6][13]
				+* (ptr_temp2 + 3 * (2 * 16) + 2) * kernels_L3_m1[6][14]
				+* (ptr_temp2 + 3 * (2 * 16) + 3) * kernels_L3_m1[6][15]
				+* (ptr_temp2 + 4 * (2 * 16) + 0) * kernels_L3_m1[6][16]
				+* (ptr_temp2 + 4 * (2 * 16) + 1) * kernels_L3_m1[6][17]
				+* (ptr_temp2 + 4 * (2 * 16) + 2) * kernels_L3_m1[6][18]
				+* (ptr_temp2 + 4 * (2 * 16) + 3) * kernels_L3_m1[6][19];

			ptr_dst = (float*)(surf_ref.surf[6]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp2 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[6][0]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[6][1]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[6][2]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[6][3]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[6][4]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[6][5]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[6][6]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[6][7]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[6][8]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[6][9]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[6][10]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[6][11]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[6][12]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[6][13]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[6][14]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[6][15]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[6][16]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[6][17]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[6][18]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[6][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp2 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[6][0]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[6][1]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[6][2]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[6][3]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[6][4]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[6][5]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[6][6]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[6][7]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[6][8]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[6][9]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[6][10]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[6][11]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[6][12]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[6][13]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[6][14]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[6][15]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[6][16]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[6][17]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[6][18]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[6][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[6][0]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[6][1]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[6][2]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[6][3]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[6][4]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[6][5]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[6][6]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[6][7]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[6][8]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[6][9]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[6][10]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[6][11]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[6][12]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[6][13]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[6][14]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[6][15]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[6][16]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[6][17]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[6][18]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[6][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp2 + 0 * (2 * 16) + 0) * kernels_L3_m1[7][0]
				+* (ptr_temp2 + 0 * (2 * 16) + 1) * kernels_L3_m1[7][1]
				+* (ptr_temp2 + 0 * (2 * 16) + 2) * kernels_L3_m1[7][2]
				+* (ptr_temp2 + 0 * (2 * 16) + 3) * kernels_L3_m1[7][3]
				+* (ptr_temp2 + 1 * (2 * 16) + 0) * kernels_L3_m1[7][4]
				+* (ptr_temp2 + 1 * (2 * 16) + 1) * kernels_L3_m1[7][5]
				+* (ptr_temp2 + 1 * (2 * 16) + 2) * kernels_L3_m1[7][6]
				+* (ptr_temp2 + 1 * (2 * 16) + 3) * kernels_L3_m1[7][7]
				+* (ptr_temp2 + 2 * (2 * 16) + 0) * kernels_L3_m1[7][8]
				+* (ptr_temp2 + 2 * (2 * 16) + 1) * kernels_L3_m1[7][9]
				+* (ptr_temp2 + 2 * (2 * 16) + 2) * kernels_L3_m1[7][10]
				+* (ptr_temp2 + 2 * (2 * 16) + 3) * kernels_L3_m1[7][11]
				+* (ptr_temp2 + 3 * (2 * 16) + 0) * kernels_L3_m1[7][12]
				+* (ptr_temp2 + 3 * (2 * 16) + 1) * kernels_L3_m1[7][13]
				+* (ptr_temp2 + 3 * (2 * 16) + 2) * kernels_L3_m1[7][14]
				+* (ptr_temp2 + 3 * (2 * 16) + 3) * kernels_L3_m1[7][15]
				+* (ptr_temp2 + 4 * (2 * 16) + 0) * kernels_L3_m1[7][16]
				+* (ptr_temp2 + 4 * (2 * 16) + 1) * kernels_L3_m1[7][17]
				+* (ptr_temp2 + 4 * (2 * 16) + 2) * kernels_L3_m1[7][18]
				+* (ptr_temp2 + 4 * (2 * 16) + 3) * kernels_L3_m1[7][19];

			ptr_dst = (float*)(surf_ref.surf[7]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp2 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[7][0]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[7][1]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[7][2]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[7][3]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[7][4]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[7][5]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[7][6]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[7][7]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[7][8]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[7][9]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[7][10]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[7][11]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[7][12]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[7][13]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[7][14]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[7][15]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[7][16]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[7][17]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[7][18]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[7][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp2 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[7][0]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[7][1]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[7][2]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[7][3]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[7][4]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[7][5]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[7][6]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[7][7]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[7][8]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[7][9]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[7][10]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[7][11]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[7][12]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[7][13]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[7][14]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[7][15]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[7][16]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[7][17]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[7][18]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[7][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[7][0]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[7][1]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[7][2]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[7][3]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[7][4]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[7][5]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[7][6]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[7][7]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[7][8]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[7][9]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[7][10]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[7][11]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[7][12]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[7][13]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[7][14]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[7][15]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[7][16]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[7][17]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[7][18]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[7][19];

			*(ptr_dst + 1) = c3;

			__syncthreads();
		}
		__global__ void add_3_conv_4x5x4_2_cu(SurfRef surf_ref)
		{
			__shared__ float temp1[(2 * 16) * (2 * 16)];
			__shared__ float temp2[(2 * 16) * (2 * 16)];

			const int i = blockIdx.x * (2 * 16 - 4 /*+ 1*/);
			const int j = blockIdx.y * (2 * 16 - 5 + 1);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);
			const int offset = (j + 2 * trdY) * surf_ref.step + i + 2 * trdX;

			const int z1 = trdX + id_walf_w * 16;
			const int z2 = (4 * ry) * (2 * 16) + z1;
			const float tx = i + z1 + 0.5f;
			const float ty = j + 4 * ry + 0.5f;
			temp1[z2 + 0 * (2 * 16)] = tex2D(tex_m1_L3_6, tx, ty + 0.f) + tex2D(tex_m1_L3_5, tx, ty + 0.f);
			temp1[z2 + 1 * (2 * 16)] = tex2D(tex_m1_L3_6, tx, ty + 1.f) + tex2D(tex_m1_L3_5, tx, ty + 1.f);
			temp1[z2 + 2 * (2 * 16)] = tex2D(tex_m1_L3_6, tx, ty + 2.f) + tex2D(tex_m1_L3_5, tx, ty + 2.f);
			temp1[z2 + 3 * (2 * 16)] = tex2D(tex_m1_L3_6, tx, ty + 3.f) + tex2D(tex_m1_L3_5, tx, ty + 3.f);

			temp2[z2 + 0 * (2 * 16)] = temp1[z2 + 0 * (2 * 16)] + tex2D(tex_m1_L3_4, tx, ty + 0.f);
			temp2[z2 + 1 * (2 * 16)] = temp1[z2 + 1 * (2 * 16)] + tex2D(tex_m1_L3_4, tx, ty + 1.f);
			temp2[z2 + 2 * (2 * 16)] = temp1[z2 + 2 * (2 * 16)] + tex2D(tex_m1_L3_4, tx, ty + 2.f);
			temp2[z2 + 3 * (2 * 16)] = temp1[z2 + 3 * (2 * 16)] + tex2D(tex_m1_L3_4, tx, ty + 3.f);

			temp1[z2 + 0 * (2 * 16)] += tex2D(tex_m1_L3_7, tx, ty + 0.f);
			temp1[z2 + 1 * (2 * 16)] += tex2D(tex_m1_L3_7, tx, ty + 1.f);
			temp1[z2 + 2 * (2 * 16)] += tex2D(tex_m1_L3_7, tx, ty + 2.f);
			temp1[z2 + 3 * (2 * 16)] += tex2D(tex_m1_L3_7, tx, ty + 3.f);

			__syncthreads();

			if (2 * trdX > 16 * 2 - 5 || 2 * trdY > 2 * 16 - 5) return;

			const float* ptr_temp1 = temp1 + 2 * trdX + (2 * 16) * 2 * trdY;
			const float* ptr_temp2 = temp2 + 2 * trdX + (2 * 16) * 2 * trdY;

			float c0 = *(ptr_temp1 + 0 * (2 * 16) + 0) * kernels_L3_m1[11][0]
				+* (ptr_temp1 + 0 * (2 * 16) + 1) * kernels_L3_m1[11][1]
				+* (ptr_temp1 + 0 * (2 * 16) + 2) * kernels_L3_m1[11][2]
				+* (ptr_temp1 + 0 * (2 * 16) + 3) * kernels_L3_m1[11][3]
				+* (ptr_temp1 + 1 * (2 * 16) + 0) * kernels_L3_m1[11][4]
				+* (ptr_temp1 + 1 * (2 * 16) + 1) * kernels_L3_m1[11][5]
				+* (ptr_temp1 + 1 * (2 * 16) + 2) * kernels_L3_m1[11][6]
				+* (ptr_temp1 + 1 * (2 * 16) + 3) * kernels_L3_m1[11][7]
				+* (ptr_temp1 + 2 * (2 * 16) + 0) * kernels_L3_m1[11][8]
				+* (ptr_temp1 + 2 * (2 * 16) + 1) * kernels_L3_m1[11][9]
				+* (ptr_temp1 + 2 * (2 * 16) + 2) * kernels_L3_m1[11][10]
				+* (ptr_temp1 + 2 * (2 * 16) + 3) * kernels_L3_m1[11][11]
				+* (ptr_temp1 + 3 * (2 * 16) + 0) * kernels_L3_m1[11][12]
				+* (ptr_temp1 + 3 * (2 * 16) + 1) * kernels_L3_m1[11][13]
				+* (ptr_temp1 + 3 * (2 * 16) + 2) * kernels_L3_m1[11][14]
				+* (ptr_temp1 + 3 * (2 * 16) + 3) * kernels_L3_m1[11][15]
				+* (ptr_temp1 + 4 * (2 * 16) + 0) * kernels_L3_m1[11][16]
				+* (ptr_temp1 + 4 * (2 * 16) + 1) * kernels_L3_m1[11][17]
				+* (ptr_temp1 + 4 * (2 * 16) + 2) * kernels_L3_m1[11][18]
				+* (ptr_temp1 + 4 * (2 * 16) + 3) * kernels_L3_m1[11][19];

			float* ptr_dst = (float*)(surf_ref.surf[11]) + offset;
			*ptr_dst = c0;

			float c1 = *(ptr_temp1 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[11][0]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[11][1]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[11][2]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[11][3]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[11][4]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[11][5]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[11][6]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[11][7]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[11][8]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[11][9]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[11][10]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[11][11]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[11][12]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[11][13]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[11][14]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[11][15]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[11][16]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[11][17]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[11][18]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[11][19];

			*(ptr_dst + 1) = c1;

			float c2 = *(ptr_temp1 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[11][0]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[11][1]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[11][2]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[11][3]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[11][4]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[11][5]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[11][6]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[11][7]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[11][8]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[11][9]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[11][10]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[11][11]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[11][12]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[11][13]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[11][14]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[11][15]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[11][16]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[11][17]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[11][18]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[11][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			float c3 = *(ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[11][0]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[11][1]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[11][2]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[11][3]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[11][4]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[11][5]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[11][6]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[11][7]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[11][8]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[11][9]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[11][10]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[11][11]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[11][12]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[11][13]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[11][14]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[11][15]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[11][16]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[11][17]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[11][18]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[11][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp1 + 0 * (2 * 16) + 0) * kernels_L3_m1[10][0]
				+* (ptr_temp1 + 0 * (2 * 16) + 1) * kernels_L3_m1[10][1]
				+* (ptr_temp1 + 0 * (2 * 16) + 2) * kernels_L3_m1[10][2]
				+* (ptr_temp1 + 0 * (2 * 16) + 3) * kernels_L3_m1[10][3]
				+* (ptr_temp1 + 1 * (2 * 16) + 0) * kernels_L3_m1[10][4]
				+* (ptr_temp1 + 1 * (2 * 16) + 1) * kernels_L3_m1[10][5]
				+* (ptr_temp1 + 1 * (2 * 16) + 2) * kernels_L3_m1[10][6]
				+* (ptr_temp1 + 1 * (2 * 16) + 3) * kernels_L3_m1[10][7]
				+* (ptr_temp1 + 2 * (2 * 16) + 0) * kernels_L3_m1[10][8]
				+* (ptr_temp1 + 2 * (2 * 16) + 1) * kernels_L3_m1[10][9]
				+* (ptr_temp1 + 2 * (2 * 16) + 2) * kernels_L3_m1[10][10]
				+* (ptr_temp1 + 2 * (2 * 16) + 3) * kernels_L3_m1[10][11]
				+* (ptr_temp1 + 3 * (2 * 16) + 0) * kernels_L3_m1[10][12]
				+* (ptr_temp1 + 3 * (2 * 16) + 1) * kernels_L3_m1[10][13]
				+* (ptr_temp1 + 3 * (2 * 16) + 2) * kernels_L3_m1[10][14]
				+* (ptr_temp1 + 3 * (2 * 16) + 3) * kernels_L3_m1[10][15]
				+* (ptr_temp1 + 4 * (2 * 16) + 0) * kernels_L3_m1[10][16]
				+* (ptr_temp1 + 4 * (2 * 16) + 1) * kernels_L3_m1[10][17]
				+* (ptr_temp1 + 4 * (2 * 16) + 2) * kernels_L3_m1[10][18]
				+* (ptr_temp1 + 4 * (2 * 16) + 3) * kernels_L3_m1[10][19];

			ptr_dst = (float*)(surf_ref.surf[10]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp1 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[10][0]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[10][1]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[10][2]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[10][3]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[10][4]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[10][5]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[10][6]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[10][7]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[10][8]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[10][9]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[10][10]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[10][11]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[10][12]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[10][13]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[10][14]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[10][15]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[10][16]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[10][17]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[10][18]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[10][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp1 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[10][0]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[10][1]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[10][2]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[10][3]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[10][4]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[10][5]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[10][6]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[10][7]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[10][8]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[10][9]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[10][10]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[10][11]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[10][12]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[10][13]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[10][14]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[10][15]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[10][16]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[10][17]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[10][18]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[10][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[10][0]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[10][1]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[10][2]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[10][3]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[10][4]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[10][5]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[10][6]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[10][7]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[10][8]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[10][9]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[10][10]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[10][11]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[10][12]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[10][13]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[10][14]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[10][15]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[10][16]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[10][17]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[10][18]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[10][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp2 + 0 * (2 * 16) + 0) * kernels_L3_m1[9][0]
				+* (ptr_temp2 + 0 * (2 * 16) + 1) * kernels_L3_m1[9][1]
				+* (ptr_temp2 + 0 * (2 * 16) + 2) * kernels_L3_m1[9][2]
				+* (ptr_temp2 + 0 * (2 * 16) + 3) * kernels_L3_m1[9][3]
				+* (ptr_temp2 + 1 * (2 * 16) + 0) * kernels_L3_m1[9][4]
				+* (ptr_temp2 + 1 * (2 * 16) + 1) * kernels_L3_m1[9][5]
				+* (ptr_temp2 + 1 * (2 * 16) + 2) * kernels_L3_m1[9][6]
				+* (ptr_temp2 + 1 * (2 * 16) + 3) * kernels_L3_m1[9][7]
				+* (ptr_temp2 + 2 * (2 * 16) + 0) * kernels_L3_m1[9][8]
				+* (ptr_temp2 + 2 * (2 * 16) + 1) * kernels_L3_m1[9][9]
				+* (ptr_temp2 + 2 * (2 * 16) + 2) * kernels_L3_m1[9][10]
				+* (ptr_temp2 + 2 * (2 * 16) + 3) * kernels_L3_m1[9][11]
				+* (ptr_temp2 + 3 * (2 * 16) + 0) * kernels_L3_m1[9][12]
				+* (ptr_temp2 + 3 * (2 * 16) + 1) * kernels_L3_m1[9][13]
				+* (ptr_temp2 + 3 * (2 * 16) + 2) * kernels_L3_m1[9][14]
				+* (ptr_temp2 + 3 * (2 * 16) + 3) * kernels_L3_m1[9][15]
				+* (ptr_temp2 + 4 * (2 * 16) + 0) * kernels_L3_m1[9][16]
				+* (ptr_temp2 + 4 * (2 * 16) + 1) * kernels_L3_m1[9][17]
				+* (ptr_temp2 + 4 * (2 * 16) + 2) * kernels_L3_m1[9][18]
				+* (ptr_temp2 + 4 * (2 * 16) + 3) * kernels_L3_m1[9][19];

			ptr_dst = (float*)(surf_ref.surf[9]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp2 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[9][0]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[9][1]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[9][2]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[9][3]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[9][4]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[9][5]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[9][6]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[9][7]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[9][8]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[9][9]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[9][10]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[9][11]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[9][12]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[9][13]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[9][14]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[9][15]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[9][16]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[9][17]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[9][18]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[9][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp2 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[9][0]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[9][1]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[9][2]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[9][3]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[9][4]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[9][5]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[9][6]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[9][7]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[9][8]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[9][9]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[9][10]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[9][11]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[9][12]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[9][13]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[9][14]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[9][15]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[9][16]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[9][17]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[9][18]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[9][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[9][0]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[9][1]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[9][2]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[9][3]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[9][4]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[9][5]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[9][6]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[9][7]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[9][8]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[9][9]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[9][10]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[9][11]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[9][12]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[9][13]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[9][14]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[9][15]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[9][16]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[9][17]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[9][18]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[9][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp2 + 0 * (2 * 16) + 0) * kernels_L3_m1[8][0]
				+* (ptr_temp2 + 0 * (2 * 16) + 1) * kernels_L3_m1[8][1]
				+* (ptr_temp2 + 0 * (2 * 16) + 2) * kernels_L3_m1[8][2]
				+* (ptr_temp2 + 0 * (2 * 16) + 3) * kernels_L3_m1[8][3]
				+* (ptr_temp2 + 1 * (2 * 16) + 0) * kernels_L3_m1[8][4]
				+* (ptr_temp2 + 1 * (2 * 16) + 1) * kernels_L3_m1[8][5]
				+* (ptr_temp2 + 1 * (2 * 16) + 2) * kernels_L3_m1[8][6]
				+* (ptr_temp2 + 1 * (2 * 16) + 3) * kernels_L3_m1[8][7]
				+* (ptr_temp2 + 2 * (2 * 16) + 0) * kernels_L3_m1[8][8]
				+* (ptr_temp2 + 2 * (2 * 16) + 1) * kernels_L3_m1[8][9]
				+* (ptr_temp2 + 2 * (2 * 16) + 2) * kernels_L3_m1[8][10]
				+* (ptr_temp2 + 2 * (2 * 16) + 3) * kernels_L3_m1[8][11]
				+* (ptr_temp2 + 3 * (2 * 16) + 0) * kernels_L3_m1[8][12]
				+* (ptr_temp2 + 3 * (2 * 16) + 1) * kernels_L3_m1[8][13]
				+* (ptr_temp2 + 3 * (2 * 16) + 2) * kernels_L3_m1[8][14]
				+* (ptr_temp2 + 3 * (2 * 16) + 3) * kernels_L3_m1[8][15]
				+* (ptr_temp2 + 4 * (2 * 16) + 0) * kernels_L3_m1[8][16]
				+* (ptr_temp2 + 4 * (2 * 16) + 1) * kernels_L3_m1[8][17]
				+* (ptr_temp2 + 4 * (2 * 16) + 2) * kernels_L3_m1[8][18]
				+* (ptr_temp2 + 4 * (2 * 16) + 3) * kernels_L3_m1[8][19];

			ptr_dst = (float*)(surf_ref.surf[8]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp2 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[8][0]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[8][1]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[8][2]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[8][3]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[8][4]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[8][5]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[8][6]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[8][7]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[8][8]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[8][9]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[8][10]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[8][11]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[8][12]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[8][13]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[8][14]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[8][15]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[8][16]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[8][17]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[8][18]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[8][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp2 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[8][0]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[8][1]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[8][2]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[8][3]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[8][4]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[8][5]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[8][6]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[8][7]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[8][8]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[8][9]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[8][10]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[8][11]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[8][12]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[8][13]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[8][14]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[8][15]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[8][16]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[8][17]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[8][18]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[8][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[8][0]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[8][1]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[8][2]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[8][3]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[8][4]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[8][5]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[8][6]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[8][7]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[8][8]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[8][9]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[8][10]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[8][11]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[8][12]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[8][13]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[8][14]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[8][15]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[8][16]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[8][17]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[8][18]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[8][19];

			*(ptr_dst + 1) = c3;

			__syncthreads();
		}
		__global__ void add_2_3_conv_4x5x4_R_cu(SurfRef surf_ref)
		{
			__shared__ float temp1[(2 * 16) * (2 * 16)];
			__shared__ float temp2[(2 * 16) * (2 * 16)];

			const int i = blockIdx.x * (2 * 16 - 4 /*+ 1*/);
			const int j = blockIdx.y * (2 * 16 - 5 + 1);

			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int ry = trdY >> 1;
			const int id_walf_w = trdY - (ry << 1);
			const int offset = (j + 2 * trdY) * surf_ref.step + i + 2 * trdX;

			const int z1 = trdX + id_walf_w * 16;
			const int z2 = (4 * ry) * (2 * 16) + z1;
			const float tx = i + z1 + 0.5f;
			const float ty = j + 4 * ry + 0.5f;
			temp1[z2 + 0 * (2 * 16)] = tex2D(tex_m1_L3_8, tx, ty + 0.f) + tex2D(tex_m1_L3_7, tx, ty + 0.f);
			temp1[z2 + 1 * (2 * 16)] = tex2D(tex_m1_L3_8, tx, ty + 1.f) + tex2D(tex_m1_L3_7, tx, ty + 1.f);
			temp1[z2 + 2 * (2 * 16)] = tex2D(tex_m1_L3_8, tx, ty + 2.f) + tex2D(tex_m1_L3_7, tx, ty + 2.f);
			temp1[z2 + 3 * (2 * 16)] = tex2D(tex_m1_L3_8, tx, ty + 3.f) + tex2D(tex_m1_L3_7, tx, ty + 3.f);

			temp2[z2 + 0 * (2 * 16)] = temp1[z2 + 0 * (2 * 16)] + tex2D(tex_m1_L3_6, tx, ty + 0.f);
			temp2[z2 + 1 * (2 * 16)] = temp1[z2 + 1 * (2 * 16)] + tex2D(tex_m1_L3_6, tx, ty + 1.f);
			temp2[z2 + 2 * (2 * 16)] = temp1[z2 + 2 * (2 * 16)] + tex2D(tex_m1_L3_6, tx, ty + 2.f);
			temp2[z2 + 3 * (2 * 16)] = temp1[z2 + 3 * (2 * 16)] + tex2D(tex_m1_L3_6, tx, ty + 3.f);

			__syncthreads();

			if (2 * trdX > 16 * 2 - 5 || 2 * trdY > 2 * 16 - 5) return;

			const float* ptr_temp1 = temp1 + 2 * trdX + (2 * 16) * 2 * trdY;
			const float* ptr_temp2 = temp2 + 2 * trdX + (2 * 16) * 2 * trdY;

			float c0 = *(ptr_temp1 + 0 * (2 * 16) + 0) * kernels_L3_m1[15][0]
				+* (ptr_temp1 + 0 * (2 * 16) + 1) * kernels_L3_m1[15][1]
				+* (ptr_temp1 + 0 * (2 * 16) + 2) * kernels_L3_m1[15][2]
				+* (ptr_temp1 + 0 * (2 * 16) + 3) * kernels_L3_m1[15][3]
				+* (ptr_temp1 + 1 * (2 * 16) + 0) * kernels_L3_m1[15][4]
				+* (ptr_temp1 + 1 * (2 * 16) + 1) * kernels_L3_m1[15][5]
				+* (ptr_temp1 + 1 * (2 * 16) + 2) * kernels_L3_m1[15][6]
				+* (ptr_temp1 + 1 * (2 * 16) + 3) * kernels_L3_m1[15][7]
				+* (ptr_temp1 + 2 * (2 * 16) + 0) * kernels_L3_m1[15][8]
				+* (ptr_temp1 + 2 * (2 * 16) + 1) * kernels_L3_m1[15][9]
				+* (ptr_temp1 + 2 * (2 * 16) + 2) * kernels_L3_m1[15][10]
				+* (ptr_temp1 + 2 * (2 * 16) + 3) * kernels_L3_m1[15][11]
				+* (ptr_temp1 + 3 * (2 * 16) + 0) * kernels_L3_m1[15][12]
				+* (ptr_temp1 + 3 * (2 * 16) + 1) * kernels_L3_m1[15][13]
				+* (ptr_temp1 + 3 * (2 * 16) + 2) * kernels_L3_m1[15][14]
				+* (ptr_temp1 + 3 * (2 * 16) + 3) * kernels_L3_m1[15][15]
				+* (ptr_temp1 + 4 * (2 * 16) + 0) * kernels_L3_m1[15][16]
				+* (ptr_temp1 + 4 * (2 * 16) + 1) * kernels_L3_m1[15][17]
				+* (ptr_temp1 + 4 * (2 * 16) + 2) * kernels_L3_m1[15][18]
				+* (ptr_temp1 + 4 * (2 * 16) + 3) * kernels_L3_m1[15][19];

			float* ptr_dst = (float*)(surf_ref.surf[15]) + offset;
			*ptr_dst = c0;

			float c1 = *(ptr_temp1 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[15][0]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[15][1]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[15][2]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[15][3]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[15][4]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[15][5]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[15][6]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[15][7]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[15][8]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[15][9]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[15][10]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[15][11]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[15][12]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[15][13]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[15][14]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[15][15]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[15][16]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[15][17]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[15][18]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[15][19];

			*(ptr_dst + 1) = c1;

			float c2 = *(ptr_temp1 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[15][0]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[15][1]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[15][2]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[15][3]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[15][4]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[15][5]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[15][6]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[15][7]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[15][8]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[15][9]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[15][10]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[15][11]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[15][12]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[15][13]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[15][14]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[15][15]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[15][16]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[15][17]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[15][18]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[15][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			float c3 = *(ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[15][0]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[15][1]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[15][2]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[15][3]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[15][4]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[15][5]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[15][6]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[15][7]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[15][8]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[15][9]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[15][10]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[15][11]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[15][12]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[15][13]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[15][14]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[15][15]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[15][16]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[15][17]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[15][18]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[15][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp1 + 0 * (2 * 16) + 0) * kernels_L3_m1[14][0]
				+* (ptr_temp1 + 0 * (2 * 16) + 1) * kernels_L3_m1[14][1]
				+* (ptr_temp1 + 0 * (2 * 16) + 2) * kernels_L3_m1[14][2]
				+* (ptr_temp1 + 0 * (2 * 16) + 3) * kernels_L3_m1[14][3]
				+* (ptr_temp1 + 1 * (2 * 16) + 0) * kernels_L3_m1[14][4]
				+* (ptr_temp1 + 1 * (2 * 16) + 1) * kernels_L3_m1[14][5]
				+* (ptr_temp1 + 1 * (2 * 16) + 2) * kernels_L3_m1[14][6]
				+* (ptr_temp1 + 1 * (2 * 16) + 3) * kernels_L3_m1[14][7]
				+* (ptr_temp1 + 2 * (2 * 16) + 0) * kernels_L3_m1[14][8]
				+* (ptr_temp1 + 2 * (2 * 16) + 1) * kernels_L3_m1[14][9]
				+* (ptr_temp1 + 2 * (2 * 16) + 2) * kernels_L3_m1[14][10]
				+* (ptr_temp1 + 2 * (2 * 16) + 3) * kernels_L3_m1[14][11]
				+* (ptr_temp1 + 3 * (2 * 16) + 0) * kernels_L3_m1[14][12]
				+* (ptr_temp1 + 3 * (2 * 16) + 1) * kernels_L3_m1[14][13]
				+* (ptr_temp1 + 3 * (2 * 16) + 2) * kernels_L3_m1[14][14]
				+* (ptr_temp1 + 3 * (2 * 16) + 3) * kernels_L3_m1[14][15]
				+* (ptr_temp1 + 4 * (2 * 16) + 0) * kernels_L3_m1[14][16]
				+* (ptr_temp1 + 4 * (2 * 16) + 1) * kernels_L3_m1[14][17]
				+* (ptr_temp1 + 4 * (2 * 16) + 2) * kernels_L3_m1[14][18]
				+* (ptr_temp1 + 4 * (2 * 16) + 3) * kernels_L3_m1[14][19];

			ptr_dst = (float*)(surf_ref.surf[14]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp1 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[14][0]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[14][1]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[14][2]
				+* (ptr_temp1 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[14][3]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[14][4]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[14][5]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[14][6]
				+* (ptr_temp1 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[14][7]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[14][8]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[14][9]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[14][10]
				+* (ptr_temp1 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[14][11]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[14][12]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[14][13]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[14][14]
				+* (ptr_temp1 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[14][15]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[14][16]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[14][17]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[14][18]
				+* (ptr_temp1 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[14][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp1 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[14][0]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[14][1]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[14][2]
				+* (ptr_temp1 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[14][3]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[14][4]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[14][5]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[14][6]
				+* (ptr_temp1 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[14][7]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[14][8]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[14][9]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[14][10]
				+* (ptr_temp1 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[14][11]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[14][12]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[14][13]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[14][14]
				+* (ptr_temp1 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[14][15]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[14][16]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[14][17]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[14][18]
				+* (ptr_temp1 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[14][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[14][0]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[14][1]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[14][2]
				+* (ptr_temp1 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[14][3]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[14][4]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[14][5]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[14][6]
				+* (ptr_temp1 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[14][7]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[14][8]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[14][9]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[14][10]
				+* (ptr_temp1 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[14][11]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[14][12]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[14][13]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[14][14]
				+* (ptr_temp1 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[14][15]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[14][16]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[14][17]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[14][18]
				+* (ptr_temp1 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[14][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp2 + 0 * (2 * 16) + 0) * kernels_L3_m1[13][0]
				+* (ptr_temp2 + 0 * (2 * 16) + 1) * kernels_L3_m1[13][1]
				+* (ptr_temp2 + 0 * (2 * 16) + 2) * kernels_L3_m1[13][2]
				+* (ptr_temp2 + 0 * (2 * 16) + 3) * kernels_L3_m1[13][3]
				+* (ptr_temp2 + 1 * (2 * 16) + 0) * kernels_L3_m1[13][4]
				+* (ptr_temp2 + 1 * (2 * 16) + 1) * kernels_L3_m1[13][5]
				+* (ptr_temp2 + 1 * (2 * 16) + 2) * kernels_L3_m1[13][6]
				+* (ptr_temp2 + 1 * (2 * 16) + 3) * kernels_L3_m1[13][7]
				+* (ptr_temp2 + 2 * (2 * 16) + 0) * kernels_L3_m1[13][8]
				+* (ptr_temp2 + 2 * (2 * 16) + 1) * kernels_L3_m1[13][9]
				+* (ptr_temp2 + 2 * (2 * 16) + 2) * kernels_L3_m1[13][10]
				+* (ptr_temp2 + 2 * (2 * 16) + 3) * kernels_L3_m1[13][11]
				+* (ptr_temp2 + 3 * (2 * 16) + 0) * kernels_L3_m1[13][12]
				+* (ptr_temp2 + 3 * (2 * 16) + 1) * kernels_L3_m1[13][13]
				+* (ptr_temp2 + 3 * (2 * 16) + 2) * kernels_L3_m1[13][14]
				+* (ptr_temp2 + 3 * (2 * 16) + 3) * kernels_L3_m1[13][15]
				+* (ptr_temp2 + 4 * (2 * 16) + 0) * kernels_L3_m1[13][16]
				+* (ptr_temp2 + 4 * (2 * 16) + 1) * kernels_L3_m1[13][17]
				+* (ptr_temp2 + 4 * (2 * 16) + 2) * kernels_L3_m1[13][18]
				+* (ptr_temp2 + 4 * (2 * 16) + 3) * kernels_L3_m1[13][19];

			ptr_dst = (float*)(surf_ref.surf[13]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp2 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[13][0]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[13][1]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[13][2]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[13][3]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[13][4]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[13][5]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[13][6]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[13][7]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[13][8]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[13][9]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[13][10]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[13][11]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[13][12]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[13][13]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[13][14]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[13][15]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[13][16]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[13][17]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[13][18]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[13][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp2 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[13][0]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[13][1]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[13][2]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[13][3]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[13][4]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[13][5]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[13][6]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[13][7]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[13][8]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[13][9]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[13][10]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[13][11]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[13][12]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[13][13]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[13][14]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[13][15]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[13][16]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[13][17]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[13][18]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[13][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[13][0]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[13][1]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[13][2]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[13][3]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[13][4]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[13][5]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[13][6]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[13][7]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[13][8]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[13][9]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[13][10]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[13][11]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[13][12]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[13][13]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[13][14]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[13][15]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[13][16]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[13][17]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[13][18]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[13][19];

			*(ptr_dst + 1) = c3;

			c0 = *(ptr_temp2 + 0 * (2 * 16) + 0) * kernels_L3_m1[12][0]
				+* (ptr_temp2 + 0 * (2 * 16) + 1) * kernels_L3_m1[12][1]
				+* (ptr_temp2 + 0 * (2 * 16) + 2) * kernels_L3_m1[12][2]
				+* (ptr_temp2 + 0 * (2 * 16) + 3) * kernels_L3_m1[12][3]
				+* (ptr_temp2 + 1 * (2 * 16) + 0) * kernels_L3_m1[12][4]
				+* (ptr_temp2 + 1 * (2 * 16) + 1) * kernels_L3_m1[12][5]
				+* (ptr_temp2 + 1 * (2 * 16) + 2) * kernels_L3_m1[12][6]
				+* (ptr_temp2 + 1 * (2 * 16) + 3) * kernels_L3_m1[12][7]
				+* (ptr_temp2 + 2 * (2 * 16) + 0) * kernels_L3_m1[12][8]
				+* (ptr_temp2 + 2 * (2 * 16) + 1) * kernels_L3_m1[12][9]
				+* (ptr_temp2 + 2 * (2 * 16) + 2) * kernels_L3_m1[12][10]
				+* (ptr_temp2 + 2 * (2 * 16) + 3) * kernels_L3_m1[12][11]
				+* (ptr_temp2 + 3 * (2 * 16) + 0) * kernels_L3_m1[12][12]
				+* (ptr_temp2 + 3 * (2 * 16) + 1) * kernels_L3_m1[12][13]
				+* (ptr_temp2 + 3 * (2 * 16) + 2) * kernels_L3_m1[12][14]
				+* (ptr_temp2 + 3 * (2 * 16) + 3) * kernels_L3_m1[12][15]
				+* (ptr_temp2 + 4 * (2 * 16) + 0) * kernels_L3_m1[12][16]
				+* (ptr_temp2 + 4 * (2 * 16) + 1) * kernels_L3_m1[12][17]
				+* (ptr_temp2 + 4 * (2 * 16) + 2) * kernels_L3_m1[12][18]
				+* (ptr_temp2 + 4 * (2 * 16) + 3) * kernels_L3_m1[12][19];

			ptr_dst = (float*)(surf_ref.surf[12]) + offset;
			*ptr_dst = c0;

			c1 = *(ptr_temp2 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[12][0]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[12][1]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[12][2]
				+* (ptr_temp2 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[12][3]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[12][4]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[12][5]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[12][6]
				+* (ptr_temp2 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[12][7]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[12][8]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[12][9]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[12][10]
				+* (ptr_temp2 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[12][11]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[12][12]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[12][13]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[12][14]
				+* (ptr_temp2 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[12][15]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[12][16]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[12][17]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[12][18]
				+* (ptr_temp2 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[12][19];

			*(ptr_dst + 1) = c1;

			c2 = *(ptr_temp2 + 32 + 0 * (2 * 16) + 0) * kernels_L3_m1[12][0]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 1) * kernels_L3_m1[12][1]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 2) * kernels_L3_m1[12][2]
				+* (ptr_temp2 + 32 + 0 * (2 * 16) + 3) * kernels_L3_m1[12][3]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 0) * kernels_L3_m1[12][4]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 1) * kernels_L3_m1[12][5]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 2) * kernels_L3_m1[12][6]
				+* (ptr_temp2 + 32 + 1 * (2 * 16) + 3) * kernels_L3_m1[12][7]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 0) * kernels_L3_m1[12][8]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 1) * kernels_L3_m1[12][9]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 2) * kernels_L3_m1[12][10]
				+* (ptr_temp2 + 32 + 2 * (2 * 16) + 3) * kernels_L3_m1[12][11]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 0) * kernels_L3_m1[12][12]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 1) * kernels_L3_m1[12][13]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 2) * kernels_L3_m1[12][14]
				+* (ptr_temp2 + 32 + 3 * (2 * 16) + 3) * kernels_L3_m1[12][15]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 0) * kernels_L3_m1[12][16]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 1) * kernels_L3_m1[12][17]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 2) * kernels_L3_m1[12][18]
				+* (ptr_temp2 + 32 + 4 * (2 * 16) + 3) * kernels_L3_m1[12][19];

			ptr_dst += surf_ref.step;
			*ptr_dst = c2;

			c3 = *(ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 0) * kernels_L3_m1[12][0]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 1) * kernels_L3_m1[12][1]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 2) * kernels_L3_m1[12][2]
				+* (ptr_temp2 + 32 + 1 + 0 * (2 * 16) + 3) * kernels_L3_m1[12][3]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 0) * kernels_L3_m1[12][4]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 1) * kernels_L3_m1[12][5]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 2) * kernels_L3_m1[12][6]
				+* (ptr_temp2 + 32 + 1 + 1 * (2 * 16) + 3) * kernels_L3_m1[12][7]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 0) * kernels_L3_m1[12][8]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 1) * kernels_L3_m1[12][9]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 2) * kernels_L3_m1[12][10]
				+* (ptr_temp2 + 32 + 1 + 2 * (2 * 16) + 3) * kernels_L3_m1[12][11]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 0) * kernels_L3_m1[12][12]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 1) * kernels_L3_m1[12][13]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 2) * kernels_L3_m1[12][14]
				+* (ptr_temp2 + 32 + 1 + 3 * (2 * 16) + 3) * kernels_L3_m1[12][15]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 0) * kernels_L3_m1[12][16]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 1) * kernels_L3_m1[12][17]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 2) * kernels_L3_m1[12][18]
				+* (ptr_temp2 + 32 + 1 + 4 * (2 * 16) + 3) * kernels_L3_m1[12][19];

			*(ptr_dst + 1) = c3;

			__syncthreads();
		}

		//HL
		__global__ void lrelu_bn_add4_tanh_add52_tanh_cu(SurfRef surf_ref, int index_output)
		{
			const int trdX = threadIdx.x;
			const int trdY = threadIdx.y;

			const int i = blockIdx.x * (2 * 16) + trdX;
			const int j = blockIdx.y * (2 * 16) + trdY;

			if (i >= surf_ref.cols || j >= surf_ref.rows) return;

			const float tx = i + 0.5f;
			const float ty = j + 0.5f;

			const int offset = j * surf_ref.step + i;

			float res = ol_b_m1[index_output];

			float c0 = tex2D(tex_m1_HL_1, tx, ty);
			float c1 = tex2D(tex_m1_HL_2, tx, ty);
			float c2 = tex2D(tex_m1_HL_3, tx, ty);
			float c3 = tex2D(tex_m1_HL_4, tx, ty);
			{
				c0 += conv_b_L3_m1[0];
				c0 = lrelu_w1_L3_m1[0] * c0 + lrelu_w2_L3_m1[0] * fmaxf(0.f, c0) + bn_b_L3_m1[0];

				c1 += conv_b_L3_m1[1];
				c1 = lrelu_w1_L3_m1[1] * c1 + lrelu_w2_L3_m1[1] * fmaxf(0.f, c1) + bn_b_L3_m1[1];

				c2 += conv_b_L3_m1[2];
				c2 = lrelu_w1_L3_m1[2] * c2 + lrelu_w2_L3_m1[2] * fmaxf(0.f, c2) + bn_b_L3_m1[2];

				c3 += conv_b_L3_m1[3];
				c3 = lrelu_w1_L3_m1[3] * c3 + lrelu_w2_L3_m1[3] * fmaxf(0.f, c3) + bn_b_L3_m1[3];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c0 * hl_w_m1[0 * surf_hl_scale_m1 + i][0]
						+ c1 * hl_w_m1[0 * surf_hl_scale_m1 + i][1]
						+ c2 * hl_w_m1[0 * surf_hl_scale_m1 + i][2]
						+ c3 * hl_w_m1[0 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[0 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[0];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][0 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}
			
			c0 = tex2D(tex_m1_HL_5, tx, ty);
			{
				c0 += conv_b_L3_m1[4];
				c0 = lrelu_w1_L3_m1[4] * c0 + lrelu_w2_L3_m1[4] * fmaxf(0.f, c0) + bn_b_L3_m1[4];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c1 * hl_w_m1[1 * surf_hl_scale_m1 + i][0]
						+ c2 * hl_w_m1[1 * surf_hl_scale_m1 + i][1]
						+ c3 * hl_w_m1[1 * surf_hl_scale_m1 + i][2]
						+ c0 * hl_w_m1[1 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[1 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[1];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][1 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			c1 = tex2D(tex_m1_HL_6, tx, ty);
			{
				c1 += conv_b_L3_m1[5];
				c1 = lrelu_w1_L3_m1[5] * c1 + lrelu_w2_L3_m1[5] * fmaxf(0.f, c1) + bn_b_L3_m1[5];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c2 * hl_w_m1[2 * surf_hl_scale_m1 + i][0]
						+ c3 * hl_w_m1[2 * surf_hl_scale_m1 + i][1]
						+ c0 * hl_w_m1[2 * surf_hl_scale_m1 + i][2]
						+ c1 * hl_w_m1[2 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[2 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[2];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][2 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			c2 = tex2D(tex_m1_HL_7, tx, ty);
			{
				c2 += conv_b_L3_m1[6];
				c2 = lrelu_w1_L3_m1[6] * c2 + lrelu_w2_L3_m1[6] * fmaxf(0.f, c2) + bn_b_L3_m1[6];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c3 * hl_w_m1[3 * surf_hl_scale_m1 + i][0]
						+ c0 * hl_w_m1[3 * surf_hl_scale_m1 + i][1]
						+ c1 * hl_w_m1[3 * surf_hl_scale_m1 + i][2]
						+ c2 * hl_w_m1[3 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[3 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[3];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][3 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			c3 = tex2D(tex_m1_HL_8, tx, ty);
			{
				c3 += conv_b_L3_m1[7];
				c3 = lrelu_w1_L3_m1[7] * c3 + lrelu_w2_L3_m1[7] * fmaxf(0.f, c3) + bn_b_L3_m1[7];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c0 * hl_w_m1[4 * surf_hl_scale_m1 + i][0]
						+ c1 * hl_w_m1[4 * surf_hl_scale_m1 + i][1]
						+ c2 * hl_w_m1[4 * surf_hl_scale_m1 + i][2]
						+ c3 * hl_w_m1[4 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[4 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[4];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][4 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			c0 = tex2D(tex_m1_HL_9, tx, ty);
			{
				c0 += conv_b_L3_m1[8];
				c0 = lrelu_w1_L3_m1[8] * c0 + lrelu_w2_L3_m1[8] * fmaxf(0.f, c0) + bn_b_L3_m1[8];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c1 * hl_w_m1[5 * surf_hl_scale_m1 + i][0]
						+ c2 * hl_w_m1[5 * surf_hl_scale_m1 + i][1]
						+ c3 * hl_w_m1[5 * surf_hl_scale_m1 + i][2]
						+ c0 * hl_w_m1[5 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[5 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[5];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][5 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			c1 = tex2D(tex_m1_HL_10, tx, ty);
			{
				c1 += conv_b_L3_m1[9];
				c1 = lrelu_w1_L3_m1[9] * c1 + lrelu_w2_L3_m1[9] * fmaxf(0.f, c1) + bn_b_L3_m1[9];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c2 * hl_w_m1[6 * surf_hl_scale_m1 + i][0]
						+ c3 * hl_w_m1[6 * surf_hl_scale_m1 + i][1]
						+ c0 * hl_w_m1[6 * surf_hl_scale_m1 + i][2]
						+ c1 * hl_w_m1[6 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[6 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[6];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][6 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			c2 = tex2D(tex_m1_HL_11, tx, ty);
			{
				c2 += conv_b_L3_m1[10];
				c2 = lrelu_w1_L3_m1[10] * c2 + lrelu_w2_L3_m1[10] * fmaxf(0.f, c2) + bn_b_L3_m1[10];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c3 * hl_w_m1[7 * surf_hl_scale_m1 + i][0]
						+ c0 * hl_w_m1[7 * surf_hl_scale_m1 + i][1]
						+ c1 * hl_w_m1[7 * surf_hl_scale_m1 + i][2]
						+ c2 * hl_w_m1[7 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[7 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[7];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][7 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			c3 = tex2D(tex_m1_HL_12, tx, ty);
			{
				c3 += conv_b_L3_m1[11];
				c3 = lrelu_w1_L3_m1[11] * c3 + lrelu_w2_L3_m1[11] * fmaxf(0.f, c3) + bn_b_L3_m1[11];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c0 * hl_w_m1[8 * surf_hl_scale_m1 + i][0]
						+ c1 * hl_w_m1[8 * surf_hl_scale_m1 + i][1]
						+ c2 * hl_w_m1[8 * surf_hl_scale_m1 + i][2]
						+ c3 * hl_w_m1[8 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[8 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[8];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][8 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			c0 = tex2D(tex_m1_HL_13, tx, ty);
			{
				c0 += conv_b_L3_m1[12];
				c0 = lrelu_w1_L3_m1[12] * c0 + lrelu_w2_L3_m1[12] * fmaxf(0.f, c0) + bn_b_L3_m1[12];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c1 * hl_w_m1[9 * surf_hl_scale_m1 + i][0]
						+ c2 * hl_w_m1[9 * surf_hl_scale_m1 + i][1]
						+ c3 * hl_w_m1[9 * surf_hl_scale_m1 + i][2]
						+ c0 * hl_w_m1[9 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[9 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[9];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][9 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			c1 = tex2D(tex_m1_HL_14, tx, ty);
			{
				c1 += conv_b_L3_m1[13];
				c1 = lrelu_w1_L3_m1[13] * c1 + lrelu_w2_L3_m1[13] * fmaxf(0.f, c1) + bn_b_L3_m1[13];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c2 * hl_w_m1[10 * surf_hl_scale_m1 + i][0]
						+ c3 * hl_w_m1[10 * surf_hl_scale_m1 + i][1]
						+ c0 * hl_w_m1[10 * surf_hl_scale_m1 + i][2]
						+ c1 * hl_w_m1[10 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[10 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[10];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][10 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			c2 = tex2D(tex_m1_HL_15, tx, ty);
			{
				c2 += conv_b_L3_m1[14];
				c2 = lrelu_w1_L3_m1[14] * c2 + lrelu_w2_L3_m1[14] * fmaxf(0.f, c2) + bn_b_L3_m1[14];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c3 * hl_w_m1[11 * surf_hl_scale_m1 + i][0]
						+ c0 * hl_w_m1[11 * surf_hl_scale_m1 + i][1]
						+ c1 * hl_w_m1[11 * surf_hl_scale_m1 + i][2]
						+ c2 * hl_w_m1[11 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[11 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[11];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][11 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			c3 = tex2D(tex_m1_HL_16, tx, ty);
			{
				c3 += conv_b_L3_m1[15];
				c3 = lrelu_w1_L3_m1[15] * c3 + lrelu_w2_L3_m1[15] * fmaxf(0.f, c3) + bn_b_L3_m1[15];

				for (int i = 0; i < surf_hl_scale_m1; ++i)
				{
					float s = c0 * hl_w_m1[12 * surf_hl_scale_m1 + i][0]
						+ c1 * hl_w_m1[12 * surf_hl_scale_m1 + i][1]
						+ c2 * hl_w_m1[12 * surf_hl_scale_m1 + i][2]
						+ c3 * hl_w_m1[12 * surf_hl_scale_m1 + i][3]
						+ hl_b_m1[12 * surf_hl_scale_m1 + i];

					s *= hl_tanh_w_m1[12];
					s = tanhf(s);
					s = 0.5f * s + 0.5f;
					s = hl_bn_w_m1[i] * s + hl_bn_b_m1[i];
					res += s * ol_w_m1[index_output][12 + i * (surf_hl_m1 / surf_hl_scale_m1)];
				}
			}

			res *= ol_tanh_w_m1;
			res = tanhf(res);

			surf_ref.surf[0][offset] = scale_HL_m1 * res;
		}

		//---------------------------------------------------------------------------------------------------------------------------------------------------------------

		/*--------------------------- cnn model 2 ---------------------------*/
		//L1

		//L2

		//L3

		//HL

		//---------------------------------------------------------------------------------------------------------------------------------------------------------------

		void CNNPP::set_texture_cache_params(int model)
		{
			if (model == 1)
			{
				//L1
				tex_m1_L1.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L1.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L1.filterMode = cudaFilterModeLinear;
				tex_m1_L1.normalized = false;

				//L2
				tex_m1_L2_1.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L2_1.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L2_1.filterMode = cudaFilterModeLinear;
				tex_m1_L2_1.normalized = false;

				tex_m1_L2_2.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L2_2.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L2_2.filterMode = cudaFilterModeLinear;
				tex_m1_L2_2.normalized = false;

				tex_m1_L2_3.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L2_3.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L2_3.filterMode = cudaFilterModeLinear;
				tex_m1_L2_3.normalized = false;

				tex_m1_L2_4.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L2_4.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L2_4.filterMode = cudaFilterModeLinear;
				tex_m1_L2_4.normalized = false;

				//L3
				tex_m1_L3_1.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L3_1.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L3_1.filterMode = cudaFilterModeLinear;
				tex_m1_L3_1.normalized = false;

				tex_m1_L3_2.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L3_2.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L3_2.filterMode = cudaFilterModeLinear;
				tex_m1_L3_2.normalized = false;

				tex_m1_L3_3.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L3_3.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L3_3.filterMode = cudaFilterModeLinear;
				tex_m1_L3_3.normalized = false;

				tex_m1_L3_4.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L3_4.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L3_4.filterMode = cudaFilterModeLinear;
				tex_m1_L3_4.normalized = false;

				tex_m1_L3_5.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L3_5.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L3_5.filterMode = cudaFilterModeLinear;
				tex_m1_L3_5.normalized = false;

				tex_m1_L3_6.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L3_6.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L3_6.filterMode = cudaFilterModeLinear;
				tex_m1_L3_6.normalized = false;

				tex_m1_L3_7.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L3_7.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L3_7.filterMode = cudaFilterModeLinear;
				tex_m1_L3_7.normalized = false;

				tex_m1_L3_8.addressMode[0] = cudaAddressModeClamp;
				tex_m1_L3_8.addressMode[1] = cudaAddressModeClamp;
				tex_m1_L3_8.filterMode = cudaFilterModeLinear;
				tex_m1_L3_8.normalized = false;

				//HL
				tex_m1_HL_1.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_1.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_1.filterMode = cudaFilterModeLinear;
				tex_m1_HL_1.normalized = false;

				tex_m1_HL_2.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_2.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_2.filterMode = cudaFilterModeLinear;
				tex_m1_HL_2.normalized = false;

				tex_m1_HL_3.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_3.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_3.filterMode = cudaFilterModeLinear;
				tex_m1_HL_3.normalized = false;

				tex_m1_HL_4.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_4.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_4.filterMode = cudaFilterModeLinear;
				tex_m1_HL_4.normalized = false;

				tex_m1_HL_5.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_5.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_5.filterMode = cudaFilterModeLinear;
				tex_m1_HL_5.normalized = false;

				tex_m1_HL_6.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_6.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_6.filterMode = cudaFilterModeLinear;
				tex_m1_HL_6.normalized = false;

				tex_m1_HL_7.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_7.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_7.filterMode = cudaFilterModeLinear;
				tex_m1_HL_7.normalized = false;

				tex_m1_HL_8.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_8.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_8.filterMode = cudaFilterModeLinear;
				tex_m1_HL_8.normalized = false;

				tex_m1_HL_9.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_9.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_9.filterMode = cudaFilterModeLinear;
				tex_m1_HL_9.normalized = false;

				tex_m1_HL_10.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_10.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_10.filterMode = cudaFilterModeLinear;
				tex_m1_HL_10.normalized = false;

				tex_m1_HL_11.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_11.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_11.filterMode = cudaFilterModeLinear;
				tex_m1_HL_11.normalized = false;

				tex_m1_HL_12.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_12.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_12.filterMode = cudaFilterModeLinear;
				tex_m1_HL_12.normalized = false;

				tex_m1_HL_13.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_13.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_13.filterMode = cudaFilterModeLinear;
				tex_m1_HL_13.normalized = false;

				tex_m1_HL_14.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_14.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_14.filterMode = cudaFilterModeLinear;
				tex_m1_HL_14.normalized = false;

				tex_m1_HL_15.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_15.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_15.filterMode = cudaFilterModeLinear;
				tex_m1_HL_15.normalized = false;

				tex_m1_HL_16.addressMode[0] = cudaAddressModeClamp;
				tex_m1_HL_16.addressMode[1] = cudaAddressModeClamp;
				tex_m1_HL_16.filterMode = cudaFilterModeLinear;
				tex_m1_HL_16.normalized = false;
			}
			if (model == 2)
			{
				//L1
				tex_m2_L1.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L1.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L1.filterMode = cudaFilterModeLinear;
				tex_m2_L1.normalized = false;

				//L2
				tex_m2_L2_1.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L2_1.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L2_1.filterMode = cudaFilterModeLinear;
				tex_m2_L2_1.normalized = false;

				tex_m2_L2_2.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L2_2.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L2_2.filterMode = cudaFilterModeLinear;
				tex_m2_L2_2.normalized = false;

				tex_m2_L2_3.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L2_3.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L2_3.filterMode = cudaFilterModeLinear;
				tex_m2_L2_3.normalized = false;

				tex_m2_L2_4.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L2_4.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L2_4.filterMode = cudaFilterModeLinear;
				tex_m2_L2_4.normalized = false;

				tex_m2_L2_5.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L2_5.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L2_5.filterMode = cudaFilterModeLinear;
				tex_m2_L2_5.normalized = false;

				tex_m2_L2_6.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L2_6.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L2_6.filterMode = cudaFilterModeLinear;
				tex_m2_L2_6.normalized = false;

				//L3
				tex_m2_L3_1.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_1.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_1.filterMode = cudaFilterModeLinear;
				tex_m2_L3_1.normalized = false;

				tex_m2_L3_2.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_2.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_2.filterMode = cudaFilterModeLinear;
				tex_m2_L3_2.normalized = false;

				tex_m2_L3_3.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_3.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_3.filterMode = cudaFilterModeLinear;
				tex_m2_L3_3.normalized = false;

				tex_m2_L3_4.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_4.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_4.filterMode = cudaFilterModeLinear;
				tex_m2_L3_4.normalized = false;

				tex_m2_L3_5.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_5.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_5.filterMode = cudaFilterModeLinear;
				tex_m2_L3_5.normalized = false;

				tex_m2_L3_6.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_6.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_6.filterMode = cudaFilterModeLinear;
				tex_m2_L3_6.normalized = false;

				tex_m2_L3_7.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_7.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_7.filterMode = cudaFilterModeLinear;
				tex_m2_L3_7.normalized = false;

				tex_m2_L3_8.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_8.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_8.filterMode = cudaFilterModeLinear;
				tex_m2_L3_8.normalized = false;

				tex_m2_L3_9.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_9.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_9.filterMode = cudaFilterModeLinear;
				tex_m2_L3_9.normalized = false;

				tex_m2_L3_10.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_10.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_10.filterMode = cudaFilterModeLinear;
				tex_m2_L3_10.normalized = false;

				tex_m2_L3_11.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_11.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_11.filterMode = cudaFilterModeLinear;
				tex_m2_L3_11.normalized = false;

				tex_m2_L3_12.addressMode[0] = cudaAddressModeClamp;
				tex_m2_L3_12.addressMode[1] = cudaAddressModeClamp;
				tex_m2_L3_12.filterMode = cudaFilterModeLinear;
				tex_m2_L3_12.normalized = false;

				//HL
				tex_m2_HL_1.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_1.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_1.filterMode = cudaFilterModeLinear;
				tex_m2_HL_1.normalized = false;

				tex_m2_HL_2.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_2.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_2.filterMode = cudaFilterModeLinear;
				tex_m2_HL_2.normalized = false;

				tex_m2_HL_3.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_3.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_3.filterMode = cudaFilterModeLinear;
				tex_m2_HL_3.normalized = false;

				tex_m2_HL_4.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_4.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_4.filterMode = cudaFilterModeLinear;
				tex_m2_HL_4.normalized = false;

				tex_m2_HL_5.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_5.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_5.filterMode = cudaFilterModeLinear;
				tex_m2_HL_5.normalized = false;

				tex_m2_HL_6.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_6.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_6.filterMode = cudaFilterModeLinear;
				tex_m2_HL_6.normalized = false;

				tex_m2_HL_7.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_7.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_7.filterMode = cudaFilterModeLinear;
				tex_m2_HL_7.normalized = false;

				tex_m2_HL_8.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_8.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_8.filterMode = cudaFilterModeLinear;
				tex_m2_HL_8.normalized = false;

				tex_m2_HL_9.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_9.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_9.filterMode = cudaFilterModeLinear;
				tex_m2_HL_9.normalized = false;

				tex_m2_HL_10.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_10.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_10.filterMode = cudaFilterModeLinear;
				tex_m2_HL_10.normalized = false;

				tex_m2_HL_11.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_11.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_11.filterMode = cudaFilterModeLinear;
				tex_m2_HL_11.normalized = false;

				tex_m2_HL_12.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_12.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_12.filterMode = cudaFilterModeLinear;
				tex_m2_HL_12.normalized = false;

				tex_m2_HL_13.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_13.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_13.filterMode = cudaFilterModeLinear;
				tex_m2_HL_13.normalized = false;

				tex_m2_HL_14.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_14.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_14.filterMode = cudaFilterModeLinear;
				tex_m2_HL_14.normalized = false;

				tex_m2_HL_15.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_15.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_15.filterMode = cudaFilterModeLinear;
				tex_m2_HL_15.normalized = false;

				tex_m2_HL_16.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_16.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_16.filterMode = cudaFilterModeLinear;
				tex_m2_HL_16.normalized = false;

				tex_m2_HL_17.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_17.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_17.filterMode = cudaFilterModeLinear;
				tex_m2_HL_17.normalized = false;

				tex_m2_HL_18.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_18.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_18.filterMode = cudaFilterModeLinear;
				tex_m2_HL_18.normalized = false;

				tex_m2_HL_19.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_19.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_19.filterMode = cudaFilterModeLinear;
				tex_m2_HL_19.normalized = false;

				tex_m2_HL_20.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_20.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_20.filterMode = cudaFilterModeLinear;
				tex_m2_HL_20.normalized = false;

				tex_m2_HL_21.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_21.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_21.filterMode = cudaFilterModeLinear;
				tex_m2_HL_21.normalized = false;

				tex_m2_HL_22.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_22.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_22.filterMode = cudaFilterModeLinear;
				tex_m2_HL_22.normalized = false;

				tex_m2_HL_23.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_23.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_23.filterMode = cudaFilterModeLinear;
				tex_m2_HL_23.normalized = false;

				tex_m2_HL_24.addressMode[0] = cudaAddressModeClamp;
				tex_m2_HL_24.addressMode[1] = cudaAddressModeClamp;
				tex_m2_HL_24.filterMode = cudaFilterModeLinear;
				tex_m2_HL_24.normalized = false;
			}
		}
		void CNNPP::bind_texture(int model, CUDA::Image_32f* src, int layer, int map_count)
		{
			if (model == 1)
			{
				switch (layer)
				{
				case 1:	//L1
					if (map_count == 1) cuERR(cudaBindTexture2D(NULL, &tex_m1_L1, src[0].dataDevice, &channelDesc, (size_t)src[0].width, (size_t)src[0].height, (size_t)src[0].widthStepDevice * sizeof(float)));
					break;

				case 2:	//L2
					if (map_count > 0) cuERR(cudaBindTexture2D(NULL, &tex_m1_L2_1, src[0].dataDevice, &channelDesc, (size_t)src[0].width, (size_t)src[0].height, (size_t)src[0].widthStepDevice * 4));
					if (map_count > 1) cuERR(cudaBindTexture2D(NULL, &tex_m1_L2_2, src[1].dataDevice, &channelDesc, (size_t)src[1].width, (size_t)src[1].height, (size_t)src[1].widthStepDevice * 4));
					if (map_count > 2) cuERR(cudaBindTexture2D(NULL, &tex_m1_L2_3, src[2].dataDevice, &channelDesc, (size_t)src[2].width, (size_t)src[2].height, (size_t)src[2].widthStepDevice * 4));
					if (map_count > 3) cuERR(cudaBindTexture2D(NULL, &tex_m1_L2_4, src[3].dataDevice, &channelDesc, (size_t)src[3].width, (size_t)src[3].height, (size_t)src[3].widthStepDevice * 4));
					break;

				case 3:	//L3
					if (map_count > 0) cuERR(cudaBindTexture2D(NULL, &tex_m1_L3_1, src[0].dataDevice, &channelDesc, (size_t)src[0].width, (size_t)src[0].height, (size_t)src[0].widthStepDevice * 4));
					if (map_count > 1) cuERR(cudaBindTexture2D(NULL, &tex_m1_L3_2, src[1].dataDevice, &channelDesc, (size_t)src[1].width, (size_t)src[1].height, (size_t)src[1].widthStepDevice * 4));
					if (map_count > 2) cuERR(cudaBindTexture2D(NULL, &tex_m1_L3_3, src[2].dataDevice, &channelDesc, (size_t)src[2].width, (size_t)src[2].height, (size_t)src[2].widthStepDevice * 4));
					if (map_count > 3) cuERR(cudaBindTexture2D(NULL, &tex_m1_L3_4, src[3].dataDevice, &channelDesc, (size_t)src[3].width, (size_t)src[3].height, (size_t)src[3].widthStepDevice * 4));
					if (map_count > 4) cuERR(cudaBindTexture2D(NULL, &tex_m1_L3_5, src[4].dataDevice, &channelDesc, (size_t)src[4].width, (size_t)src[4].height, (size_t)src[4].widthStepDevice * 4));
					if (map_count > 5) cuERR(cudaBindTexture2D(NULL, &tex_m1_L3_6, src[5].dataDevice, &channelDesc, (size_t)src[5].width, (size_t)src[5].height, (size_t)src[5].widthStepDevice * 4));
					if (map_count > 6) cuERR(cudaBindTexture2D(NULL, &tex_m1_L3_7, src[6].dataDevice, &channelDesc, (size_t)src[6].width, (size_t)src[6].height, (size_t)src[6].widthStepDevice * 4));
					if (map_count > 7) cuERR(cudaBindTexture2D(NULL, &tex_m1_L3_8, src[7].dataDevice, &channelDesc, (size_t)src[7].width, (size_t)src[7].height, (size_t)src[7].widthStepDevice * 4));
					break;

				case 4:	//HL
					if (map_count > 0) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_1, src[0].dataDevice, &channelDesc, (size_t)src[0].width, (size_t)src[0].height, (size_t)src[0].widthStepDevice * 4));
					if (map_count > 1) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_2, src[1].dataDevice, &channelDesc, (size_t)src[1].width, (size_t)src[1].height, (size_t)src[1].widthStepDevice * 4));
					if (map_count > 2) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_3, src[2].dataDevice, &channelDesc, (size_t)src[2].width, (size_t)src[2].height, (size_t)src[2].widthStepDevice * 4));
					if (map_count > 3) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_4, src[3].dataDevice, &channelDesc, (size_t)src[3].width, (size_t)src[3].height, (size_t)src[3].widthStepDevice * 4));
					if (map_count > 4) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_5, src[4].dataDevice, &channelDesc, (size_t)src[4].width, (size_t)src[4].height, (size_t)src[4].widthStepDevice * 4));
					if (map_count > 5) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_6, src[5].dataDevice, &channelDesc, (size_t)src[5].width, (size_t)src[5].height, (size_t)src[5].widthStepDevice * 4));
					if (map_count > 6) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_7, src[6].dataDevice, &channelDesc, (size_t)src[6].width, (size_t)src[6].height, (size_t)src[6].widthStepDevice * 4));
					if (map_count > 7) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_8, src[7].dataDevice, &channelDesc, (size_t)src[7].width, (size_t)src[7].height, (size_t)src[7].widthStepDevice * 4));
					if (map_count > 8) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_9, src[8].dataDevice, &channelDesc, (size_t)src[8].width, (size_t)src[8].height, (size_t)src[8].widthStepDevice * 4));
					if (map_count > 9) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_10, src[9].dataDevice, &channelDesc, (size_t)src[9].width, (size_t)src[9].height, (size_t)src[9].widthStepDevice * 4));
					if (map_count > 10) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_11, src[10].dataDevice, &channelDesc, (size_t)src[10].width, (size_t)src[10].height, (size_t)src[10].widthStepDevice * 4));
					if (map_count > 11) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_12, src[11].dataDevice, &channelDesc, (size_t)src[11].width, (size_t)src[11].height, (size_t)src[11].widthStepDevice * 4));
					if (map_count > 12) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_13, src[12].dataDevice, &channelDesc, (size_t)src[12].width, (size_t)src[12].height, (size_t)src[12].widthStepDevice * 4));
					if (map_count > 13) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_14, src[13].dataDevice, &channelDesc, (size_t)src[13].width, (size_t)src[13].height, (size_t)src[13].widthStepDevice * 4));
					if (map_count > 14) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_15, src[14].dataDevice, &channelDesc, (size_t)src[14].width, (size_t)src[14].height, (size_t)src[14].widthStepDevice * 4));
					if (map_count > 15) cuERR(cudaBindTexture2D(NULL, &tex_m1_HL_16, src[15].dataDevice, &channelDesc, (size_t)src[15].width, (size_t)src[15].height, (size_t)src[15].widthStepDevice * 4));
					break;

				//default:
					//system("pause");
				}
			}
			if (model == 2)
			{
				switch (layer)
				{
				case 1:	//L1
					if (map_count == 1) cuERR(cudaBindTexture2D(NULL, &tex_m2_L1, src[0].dataDevice, &channelDesc, (size_t)src[0].width, (size_t)src[0].height, (size_t)src[0].widthStepDevice * sizeof(float)));
					break;

				case 2:	//L2
					if (map_count > 0) cuERR(cudaBindTexture2D(NULL, &tex_m2_L2_1, src[0].dataDevice, &channelDesc, (size_t)src[0].width, (size_t)src[0].height, (size_t)src[0].widthStepDevice * 4));
					if (map_count > 1) cuERR(cudaBindTexture2D(NULL, &tex_m2_L2_2, src[1].dataDevice, &channelDesc, (size_t)src[1].width, (size_t)src[1].height, (size_t)src[1].widthStepDevice * 4));
					if (map_count > 2) cuERR(cudaBindTexture2D(NULL, &tex_m2_L2_3, src[2].dataDevice, &channelDesc, (size_t)src[2].width, (size_t)src[2].height, (size_t)src[2].widthStepDevice * 4));
					if (map_count > 3) cuERR(cudaBindTexture2D(NULL, &tex_m2_L2_4, src[3].dataDevice, &channelDesc, (size_t)src[3].width, (size_t)src[3].height, (size_t)src[3].widthStepDevice * 4));
					if (map_count > 4) cuERR(cudaBindTexture2D(NULL, &tex_m2_L2_5, src[4].dataDevice, &channelDesc, (size_t)src[4].width, (size_t)src[4].height, (size_t)src[4].widthStepDevice * 4));
					if (map_count > 5) cuERR(cudaBindTexture2D(NULL, &tex_m2_L2_6, src[5].dataDevice, &channelDesc, (size_t)src[5].width, (size_t)src[5].height, (size_t)src[5].widthStepDevice * 4));
					break;

				case 3:	//L3
					if (map_count > 0) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_1, src[0].dataDevice, &channelDesc, (size_t)src[0].width, (size_t)src[0].height, (size_t)src[0].widthStepDevice * 4));
					if (map_count > 1) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_2, src[1].dataDevice, &channelDesc, (size_t)src[1].width, (size_t)src[1].height, (size_t)src[1].widthStepDevice * 4));
					if (map_count > 2) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_3, src[2].dataDevice, &channelDesc, (size_t)src[2].width, (size_t)src[2].height, (size_t)src[2].widthStepDevice * 4));
					if (map_count > 3) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_4, src[3].dataDevice, &channelDesc, (size_t)src[3].width, (size_t)src[3].height, (size_t)src[3].widthStepDevice * 4));
					if (map_count > 4) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_5, src[4].dataDevice, &channelDesc, (size_t)src[4].width, (size_t)src[4].height, (size_t)src[4].widthStepDevice * 4));
					if (map_count > 5) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_6, src[5].dataDevice, &channelDesc, (size_t)src[5].width, (size_t)src[5].height, (size_t)src[5].widthStepDevice * 4));
					if (map_count > 6) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_7, src[6].dataDevice, &channelDesc, (size_t)src[6].width, (size_t)src[6].height, (size_t)src[6].widthStepDevice * 4));
					if (map_count > 7) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_8, src[7].dataDevice, &channelDesc, (size_t)src[7].width, (size_t)src[7].height, (size_t)src[7].widthStepDevice * 4));
					if (map_count > 8) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_9, src[8].dataDevice, &channelDesc, (size_t)src[8].width, (size_t)src[8].height, (size_t)src[8].widthStepDevice * 4));
					if (map_count > 9) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_10, src[9].dataDevice, &channelDesc, (size_t)src[9].width, (size_t)src[9].height, (size_t)src[9].widthStepDevice * 4));
					if (map_count > 10) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_11, src[10].dataDevice, &channelDesc, (size_t)src[10].width, (size_t)src[10].height, (size_t)src[10].widthStepDevice * 4));
					if (map_count > 11) cuERR(cudaBindTexture2D(NULL, &tex_m2_L3_12, src[11].dataDevice, &channelDesc, (size_t)src[11].width, (size_t)src[11].height, (size_t)src[11].widthStepDevice * 4));
					break;

				case 4:	//HL
					if (map_count > 0) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_1, src[0].dataDevice, &channelDesc, (size_t)src[0].width, (size_t)src[0].height, (size_t)src[0].widthStepDevice * 4));
					if (map_count > 1) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_2, src[1].dataDevice, &channelDesc, (size_t)src[1].width, (size_t)src[1].height, (size_t)src[1].widthStepDevice * 4));
					if (map_count > 2) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_3, src[2].dataDevice, &channelDesc, (size_t)src[2].width, (size_t)src[2].height, (size_t)src[2].widthStepDevice * 4));
					if (map_count > 3) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_4, src[3].dataDevice, &channelDesc, (size_t)src[3].width, (size_t)src[3].height, (size_t)src[3].widthStepDevice * 4));
					if (map_count > 4) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_5, src[4].dataDevice, &channelDesc, (size_t)src[4].width, (size_t)src[4].height, (size_t)src[4].widthStepDevice * 4));
					if (map_count > 5) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_6, src[5].dataDevice, &channelDesc, (size_t)src[5].width, (size_t)src[5].height, (size_t)src[5].widthStepDevice * 4));
					if (map_count > 6) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_7, src[6].dataDevice, &channelDesc, (size_t)src[6].width, (size_t)src[6].height, (size_t)src[6].widthStepDevice * 4));
					if (map_count > 7) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_8, src[7].dataDevice, &channelDesc, (size_t)src[7].width, (size_t)src[7].height, (size_t)src[7].widthStepDevice * 4));
					if (map_count > 8) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_9, src[8].dataDevice, &channelDesc, (size_t)src[8].width, (size_t)src[8].height, (size_t)src[8].widthStepDevice * 4));
					if (map_count > 9) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_10, src[9].dataDevice, &channelDesc, (size_t)src[9].width, (size_t)src[9].height, (size_t)src[9].widthStepDevice * 4));
					if (map_count > 10) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_11, src[10].dataDevice, &channelDesc, (size_t)src[10].width, (size_t)src[10].height, (size_t)src[10].widthStepDevice * 4));
					if (map_count > 11) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_12, src[11].dataDevice, &channelDesc, (size_t)src[11].width, (size_t)src[11].height, (size_t)src[11].widthStepDevice * 4));
					if (map_count > 12) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_13, src[12].dataDevice, &channelDesc, (size_t)src[12].width, (size_t)src[12].height, (size_t)src[12].widthStepDevice * 4));
					if (map_count > 13) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_14, src[13].dataDevice, &channelDesc, (size_t)src[13].width, (size_t)src[13].height, (size_t)src[13].widthStepDevice * 4));
					if (map_count > 14) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_15, src[14].dataDevice, &channelDesc, (size_t)src[14].width, (size_t)src[14].height, (size_t)src[14].widthStepDevice * 4));
					if (map_count > 15) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_16, src[15].dataDevice, &channelDesc, (size_t)src[15].width, (size_t)src[15].height, (size_t)src[15].widthStepDevice * 4));
					if (map_count > 16) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_17, src[16].dataDevice, &channelDesc, (size_t)src[16].width, (size_t)src[16].height, (size_t)src[16].widthStepDevice * 4));
					if (map_count > 17) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_18, src[17].dataDevice, &channelDesc, (size_t)src[17].width, (size_t)src[17].height, (size_t)src[17].widthStepDevice * 4));
					if (map_count > 18) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_19, src[18].dataDevice, &channelDesc, (size_t)src[18].width, (size_t)src[18].height, (size_t)src[18].widthStepDevice * 4));
					if (map_count > 19) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_20, src[19].dataDevice, &channelDesc, (size_t)src[19].width, (size_t)src[19].height, (size_t)src[19].widthStepDevice * 4));
					if (map_count > 20) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_21, src[20].dataDevice, &channelDesc, (size_t)src[20].width, (size_t)src[20].height, (size_t)src[20].widthStepDevice * 4));
					if (map_count > 21) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_22, src[21].dataDevice, &channelDesc, (size_t)src[21].width, (size_t)src[21].height, (size_t)src[21].widthStepDevice * 4));
					if (map_count > 22) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_23, src[22].dataDevice, &channelDesc, (size_t)src[22].width, (size_t)src[22].height, (size_t)src[22].widthStepDevice * 4));
					if (map_count > 23) cuERR(cudaBindTexture2D(NULL, &tex_m2_HL_24, src[23].dataDevice, &channelDesc, (size_t)src[23].width, (size_t)src[23].height, (size_t)src[23].widthStepDevice * 4));
					break;

				//default:
					//system("pause");
				}
			}
		}
		void CNNPP::set_dst_surf(int model, CUDA::Image_32f* dst, int layer, int map_count)
		{
			if (model == 1)
			{
				switch (layer)
				{
				case 1:	//L1
					surf_ref_m1_L1 = SurfRef(dst, map_count);
					break;

				case 2:	//L2
					surf_ref_m1_L2 = SurfRef(dst, map_count);
					break;

				case 3:	//L3
					surf_ref_m1_L3 = SurfRef(dst, map_count);
					break;

				case 4:	//HL
					surf_ref_m1_HL = SurfRef(dst, map_count);
					break;

				//default:
					//system("pause");
				}
			}
			if (model == 2)
			{
				switch (layer)
				{
				case 1:	//L1
					surf_ref_m2_L1 = SurfRef(dst, map_count);
					break;

				case 2:	//L2
					surf_ref_m2_L2 = SurfRef(dst, map_count);
					break;

				case 3:	//L3
					surf_ref_m2_L3 = SurfRef(dst, map_count);
					break;

				case 4:	//HL
					surf_ref_m2_HL = SurfRef(dst, map_count);
					break;

				//default:
					//system("pause");
				}
			}
		}
		void CNNPP::unbind_texture(int model, int layer, int map_count)
		{
			if (model == 1)
			{
				switch (layer)
				{
				case 1:	//L1
					if (map_count == 1) cuERR(cudaUnbindTexture(&tex_m1_L1));
					break;

				case 2:	//L2
					if (map_count > 0) cuERR(cudaUnbindTexture(&tex_m1_L2_1));
					if (map_count > 1) cuERR(cudaUnbindTexture(&tex_m1_L2_2));
					if (map_count > 2) cuERR(cudaUnbindTexture(&tex_m1_L2_3));
					if (map_count > 3) cuERR(cudaUnbindTexture(&tex_m1_L2_4));
					break;

				case 3:	//L3
					if (map_count > 0) cuERR(cudaUnbindTexture(&tex_m1_L3_1));
					if (map_count > 1) cuERR(cudaUnbindTexture(&tex_m1_L3_2));
					if (map_count > 2) cuERR(cudaUnbindTexture(&tex_m1_L3_3));
					if (map_count > 3) cuERR(cudaUnbindTexture(&tex_m1_L3_4));
					if (map_count > 4) cuERR(cudaUnbindTexture(&tex_m1_L3_5));
					if (map_count > 5) cuERR(cudaUnbindTexture(&tex_m1_L3_6));
					if (map_count > 6) cuERR(cudaUnbindTexture(&tex_m1_L3_7));
					if (map_count > 7) cuERR(cudaUnbindTexture(&tex_m1_L3_8));
					break;

				case 4:	//HL
					if (map_count > 0) cuERR(cudaUnbindTexture(&tex_m1_HL_1));
					if (map_count > 1) cuERR(cudaUnbindTexture(&tex_m1_HL_2));
					if (map_count > 2) cuERR(cudaUnbindTexture(&tex_m1_HL_3));
					if (map_count > 3) cuERR(cudaUnbindTexture(&tex_m1_HL_4));
					if (map_count > 4) cuERR(cudaUnbindTexture(&tex_m1_HL_5));
					if (map_count > 5) cuERR(cudaUnbindTexture(&tex_m1_HL_6));
					if (map_count > 6) cuERR(cudaUnbindTexture(&tex_m1_HL_7));
					if (map_count > 7) cuERR(cudaUnbindTexture(&tex_m1_HL_8));
					if (map_count > 8) cuERR(cudaUnbindTexture(&tex_m1_HL_9));
					if (map_count > 9) cuERR(cudaUnbindTexture(&tex_m1_HL_10));
					if (map_count > 10) cuERR(cudaUnbindTexture(&tex_m1_HL_11));
					if (map_count > 11) cuERR(cudaUnbindTexture(&tex_m1_HL_12));
					if (map_count > 12) cuERR(cudaUnbindTexture(&tex_m1_HL_13));
					if (map_count > 13) cuERR(cudaUnbindTexture(&tex_m1_HL_14));
					if (map_count > 14) cuERR(cudaUnbindTexture(&tex_m1_HL_15));
					if (map_count > 15) cuERR(cudaUnbindTexture(&tex_m1_HL_16));
					break;

				//default:
					//system("pause");
				}
			}
			if (model == 2)
			{
				switch (layer)
				{
				case 1:	//L1
					if (map_count == 1) cuERR(cudaUnbindTexture(&tex_m2_L1));
					break;

				case 2:	//L2
					if (map_count > 0) cuERR(cudaUnbindTexture(&tex_m2_L2_1));
					if (map_count > 1) cuERR(cudaUnbindTexture(&tex_m2_L2_2));
					if (map_count > 2) cuERR(cudaUnbindTexture(&tex_m2_L2_3));
					if (map_count > 3) cuERR(cudaUnbindTexture(&tex_m2_L2_4));
					if (map_count > 4) cuERR(cudaUnbindTexture(&tex_m2_L2_5));
					if (map_count > 5) cuERR(cudaUnbindTexture(&tex_m2_L2_6));
					break;

				case 3:	//L3
					if (map_count > 0) cuERR(cudaUnbindTexture(&tex_m2_L3_1));
					if (map_count > 1) cuERR(cudaUnbindTexture(&tex_m2_L3_2));
					if (map_count > 2) cuERR(cudaUnbindTexture(&tex_m2_L3_3));
					if (map_count > 3) cuERR(cudaUnbindTexture(&tex_m2_L3_4));
					if (map_count > 4) cuERR(cudaUnbindTexture(&tex_m2_L3_5));
					if (map_count > 5) cuERR(cudaUnbindTexture(&tex_m2_L3_6));
					if (map_count > 6) cuERR(cudaUnbindTexture(&tex_m2_L3_7));
					if (map_count > 7) cuERR(cudaUnbindTexture(&tex_m2_L3_8));
					if (map_count > 8) cuERR(cudaUnbindTexture(&tex_m2_L3_9));
					if (map_count > 9) cuERR(cudaUnbindTexture(&tex_m2_L3_10));
					if (map_count > 10) cuERR(cudaUnbindTexture(&tex_m2_L3_11));
					if (map_count > 11) cuERR(cudaUnbindTexture(&tex_m2_L3_12));
					break;

				case 4:	//HL
					if (map_count > 0) cuERR(cudaUnbindTexture(&tex_m2_HL_1));
					if (map_count > 1) cuERR(cudaUnbindTexture(&tex_m2_HL_2));
					if (map_count > 2) cuERR(cudaUnbindTexture(&tex_m2_HL_3));
					if (map_count > 3) cuERR(cudaUnbindTexture(&tex_m2_HL_4));
					if (map_count > 4) cuERR(cudaUnbindTexture(&tex_m2_HL_5));
					if (map_count > 5) cuERR(cudaUnbindTexture(&tex_m2_HL_6));
					if (map_count > 6) cuERR(cudaUnbindTexture(&tex_m2_HL_7));
					if (map_count > 7) cuERR(cudaUnbindTexture(&tex_m2_HL_8));
					if (map_count > 8) cuERR(cudaUnbindTexture(&tex_m2_HL_9));
					if (map_count > 9) cuERR(cudaUnbindTexture(&tex_m2_HL_10));
					if (map_count > 10) cuERR(cudaUnbindTexture(&tex_m2_HL_11));
					if (map_count > 11) cuERR(cudaUnbindTexture(&tex_m2_HL_12));
					if (map_count > 12) cuERR(cudaUnbindTexture(&tex_m2_HL_13));
					if (map_count > 13) cuERR(cudaUnbindTexture(&tex_m2_HL_14));
					if (map_count > 14) cuERR(cudaUnbindTexture(&tex_m2_HL_15));
					if (map_count > 15) cuERR(cudaUnbindTexture(&tex_m2_HL_16));
					if (map_count > 16) cuERR(cudaUnbindTexture(&tex_m2_HL_17));
					if (map_count > 17) cuERR(cudaUnbindTexture(&tex_m2_HL_18));
					if (map_count > 18) cuERR(cudaUnbindTexture(&tex_m2_HL_19));
					if (map_count > 19) cuERR(cudaUnbindTexture(&tex_m2_HL_20));
					if (map_count > 20) cuERR(cudaUnbindTexture(&tex_m2_HL_21));
					if (map_count > 21) cuERR(cudaUnbindTexture(&tex_m2_HL_22));
					if (map_count > 22) cuERR(cudaUnbindTexture(&tex_m2_HL_23));
					if (map_count > 23) cuERR(cudaUnbindTexture(&tex_m2_HL_24));
					break;

				//default:
					//system("pause");
				}
			}
		}
		void CNNPP::set_cache_device_params(int model)
		{
			if (model == 1)
			{
				//L1
				cuERR(cudaFuncSetCacheConfig(conv_4x4x4_lrelu_bn_max_cu, cudaFuncCachePreferL1));

				//L2
				cuERR(cudaFuncSetCacheConfig(add_2_3_conv_4x3x3_lrelu_bn_max_L_cu, cudaFuncCachePreferL1));
				cuERR(cudaFuncSetCacheConfig(add_2_3_conv_4x3x3_lrelu_bn_max_R_cu, cudaFuncCachePreferL1));

				//L3
				cuERR(cudaFuncSetCacheConfig(add_2_3_conv_4x5x4_L_cu, cudaFuncCachePreferShared));
				cuERR(cudaFuncSetCacheConfig(add_3_conv_4x5x4_1_cu, cudaFuncCachePreferShared));
				cuERR(cudaFuncSetCacheConfig(add_3_conv_4x5x4_2_cu, cudaFuncCachePreferShared));
				cuERR(cudaFuncSetCacheConfig(add_2_3_conv_4x5x4_R_cu, cudaFuncCachePreferShared));

				//HL
				cuERR(cudaFuncSetCacheConfig(lrelu_bn_add4_tanh_add52_tanh_cu, cudaFuncCachePreferL1));
			}
			if (model == 2)
			{
				//L1

				//L2

				//L3

				//HL
			}
		}

		//---------------------------------------------------------------------------------------------------------------------------------------------------------------

		void CNNPP::set_kernel_on_device(int model, float* kernel, int size, int layer, int surface)
		{
			if (model == 1)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(kernels_L1_m1, (void*)kernel, size * sizeof(float), surface * kernel_size_L1_m1 * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(kernels_L2_m1, (void*)kernel, size * sizeof(float), surface * kernel_size_L2_m1 * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(kernels_L3_m1, (void*)kernel, size * sizeof(float), surface * kernel_size_L3_m1 * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
			if (model == 2)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(kernels_L1_m2, (void*)kernel, size * sizeof(float), surface * kernel_size_L1_m2 * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(kernels_L2_m2, (void*)kernel, size * sizeof(float), surface * kernel_size_L2_m2 * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(kernels_L3_m2, (void*)kernel, size * sizeof(float), surface * kernel_size_L3_m2 * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
		}
		void CNNPP::set_conv_b_on_device(int model, float* conv_b, int layer, int surface)
		{
			if (model == 1)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(conv_b_L1_m1, (void*)conv_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(conv_b_L2_m1, (void*)conv_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(conv_b_L3_m1, (void*)conv_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
			if (model == 2)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(conv_b_L1_m2, (void*)conv_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(conv_b_L2_m2, (void*)conv_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(conv_b_L3_m2, (void*)conv_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
		}
		void CNNPP::set_lrelu_w1_on_device(int model, float* lrelu_w1, int layer, int surface)
		{
			if (model == 1)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(lrelu_w1_L1_m1, (void*)lrelu_w1, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(lrelu_w1_L2_m1, (void*)lrelu_w1, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(lrelu_w1_L3_m1, (void*)lrelu_w1, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
			if (model == 2)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(lrelu_w1_L1_m2, (void*)lrelu_w1, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(lrelu_w1_L2_m2, (void*)lrelu_w1, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(lrelu_w1_L3_m2, (void*)lrelu_w1, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
		}
		void CNNPP::set_lrelu_w2_on_device(int model, float* lrelu_w2, int layer, int surface)
		{
			if (model == 1)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(lrelu_w2_L1_m1, (void*)lrelu_w2, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(lrelu_w2_L2_m1, (void*)lrelu_w2, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(lrelu_w2_L3_m1, (void*)lrelu_w2, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
			if (model == 2)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(lrelu_w2_L1_m2, (void*)lrelu_w2, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(lrelu_w2_L2_m2, (void*)lrelu_w2, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(lrelu_w2_L3_m2, (void*)lrelu_w2, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
		}
		void CNNPP::set_bn_w_on_device(int model, float* bn_w, int layer, int surface)
		{
			if (model == 1)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(bn_w_L1_m1, (void*)bn_w, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(bn_w_L2_m1, (void*)bn_w, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(bn_w_L3_m1, (void*)bn_w, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
			if (model == 2)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(bn_w_L1_m2, (void*)bn_w, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(bn_w_L2_m2, (void*)bn_w, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(bn_w_L3_m2, (void*)bn_w, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
		}
		void CNNPP::set_bn_b_on_device(int model, float* bn_b, int layer, int surface)
		{
			if (model == 1)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(bn_b_L1_m1, (void*)bn_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(bn_b_L2_m1, (void*)bn_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(bn_b_L3_m1, (void*)bn_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
			if (model == 2)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(bn_b_L1_m2, (void*)bn_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(bn_b_L2_m2, (void*)bn_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(bn_b_L3_m2, (void*)bn_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
					break;
				}
			}
		}
		void CNNPP::set_scale_on_device(int model, float* scale, int layer)
		{
			if (model == 1)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(scale_L1_m1, (void*)scale, sizeof(float), 0, cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(scale_L2_m1, (void*)scale, sizeof(float), 0, cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(scale_L3_m1, (void*)scale, sizeof(float), 0, cudaMemcpyHostToDevice);
					break;
				case 4:
					cudaMemcpyToSymbol(scale_HL_m1, (void*)scale, sizeof(float), 0, cudaMemcpyHostToDevice);
					break;
				}
			}
			if (model == 2)
			{
				switch (layer)
				{
				case 1:
					cudaMemcpyToSymbol(scale_L1_m2, (void*)scale, sizeof(float), 0, cudaMemcpyHostToDevice);
					break;
				case 2:
					cudaMemcpyToSymbol(scale_L2_m2, (void*)scale, sizeof(float), 0, cudaMemcpyHostToDevice);
					break;
				case 3:
					cudaMemcpyToSymbol(scale_L3_m2, (void*)scale, sizeof(float), 0, cudaMemcpyHostToDevice);
					break;
				case 4:
					cudaMemcpyToSymbol(scale_HL_m2, (void*)scale, sizeof(float), 0, cudaMemcpyHostToDevice);
					break;
				}
			}
		}
		void CNNPP::set_hl_w_on_device(int model, float* hl_w, int surface)
		{
			if (model == 1)
			{
				cudaMemcpyToSymbol(hl_w_m1, (void*)hl_w, surf_hl_connect_m1 * sizeof(float), surface * surf_hl_connect_m1 * sizeof(float), cudaMemcpyHostToDevice);
			}
			if (model == 2)
			{
				cudaMemcpyToSymbol(hl_w_m2, (void*)hl_w, surf_hl_connect_m2 * sizeof(float), surface * surf_hl_connect_m2 * sizeof(float), cudaMemcpyHostToDevice);
			}
		}
		void CNNPP::set_hl_b_on_device(int model, float* hl_b, int surface)
		{
			if (model == 1)
			{
				cudaMemcpyToSymbol(hl_b_m1, (void*)hl_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
			}
			if (model == 2)
			{
				cudaMemcpyToSymbol(hl_b_m2, (void*)hl_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
			}
		}
		void CNNPP::set_hl_tanh_w_on_device(int model, float* hl_tanh_w, int surface)
		{
			if (model == 1)
			{
				cudaMemcpyToSymbol(hl_tanh_w_m1, (void*)hl_tanh_w, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
			}
			if (model == 2)
			{
				cudaMemcpyToSymbol(hl_tanh_w_m2, (void*)hl_tanh_w, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
			}
		}
		void CNNPP::set_hl_bn_w_on_device(int model, float* hl_bn_w, int surface)
		{
			if (model == 1)
			{
				cudaMemcpyToSymbol(hl_bn_w_m1, (void*)hl_bn_w, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
			}
			if (model == 2)
			{
				cudaMemcpyToSymbol(hl_bn_w_m2, (void*)hl_bn_w, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
			}
		}
		void CNNPP::set_hl_bn_b_on_device(int model, float* hl_bn_b, int surface)
		{
			if (model == 1)
			{
				cudaMemcpyToSymbol(hl_bn_b_m1, (void*)hl_bn_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
			}
			if (model == 2)
			{
				cudaMemcpyToSymbol(hl_bn_b_m2, (void*)hl_bn_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
			}
		}
		void CNNPP::set_ol_w_on_device(int model, float* ol_w, int surface)
		{
			if (model == 1)
			{
				cudaMemcpyToSymbol(ol_w_m1, (void*)ol_w, surf_hl_m1 * sizeof(float), surface * surf_hl_m1 * sizeof(float), cudaMemcpyHostToDevice);
			}
			if (model == 2)
			{
				cudaMemcpyToSymbol(ol_w_m2, (void*)ol_w, surf_hl_m2 * sizeof(float), surface * surf_hl_m2 * sizeof(float), cudaMemcpyHostToDevice);
			}
		}
		void CNNPP::set_ol_b_on_device(int model, float* ol_b, int surface)
		{
			if (model == 1)
			{
				cudaMemcpyToSymbol(ol_b_m1, (void*)ol_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
			}
			if (model == 2)
			{
				cudaMemcpyToSymbol(ol_b_m2, (void*)ol_b, sizeof(float), surface * sizeof(float), cudaMemcpyHostToDevice);
			}
		}
		void CNNPP::set_ol_tanh_w_on_device(int model, float* tanh_w)
		{
			if (model == 1)
			{
				cudaMemcpyToSymbol(ol_tanh_w_m1, (void*)tanh_w, sizeof(float), 0, cudaMemcpyHostToDevice);
			}
			if (model == 2)
			{
				cudaMemcpyToSymbol(ol_tanh_w_m2, (void*)tanh_w, sizeof(float), 0, cudaMemcpyHostToDevice);
			}
		}

		//---------------------------------------------------------------------------------------------------------------------------------------------------------------

		void CNNPP::run_L1(int model, Size2d& ROI, cudaStream_t cuda_stream)
		{
			dim3 block(16, 16);
			dim3 grid(blockCount(ROI.cols, 32), blockCount(ROI.rows, 32));

			conv_4x4x4_lrelu_bn_max_cu << <grid, block, 0, cuda_stream >> >(surf_ref_m1_L1);
		}
		void CNNPP::run_L2(int model, Size2d& ROI, cudaStream_t cuda_stream)
		{
			dim3 block(16, 16);
			dim3 grid(blockCount(ROI.cols, 32), blockCount(ROI.rows, 32));

			add_2_3_conv_4x3x3_lrelu_bn_max_L_cu << <grid, block, 0, cuda_stream >> >(surf_ref_m1_L2);
			add_2_3_conv_4x3x3_lrelu_bn_max_R_cu << <grid, block, 0, cuda_stream >> >(surf_ref_m1_L2);
		}
		void CNNPP::run_L3(int model, Size2d& ROI, cudaStream_t cuda_stream)
		{
			dim3 block(16, 16);
			dim3 grid(blockCount(ROI.cols, 32 - 4 /*+ 1*/), blockCount(ROI.rows, 32 - 5 + 1));

			add_2_3_conv_4x5x4_L_cu << <grid, block, 0, cuda_stream >> >(surf_ref_m1_L3);
			add_3_conv_4x5x4_1_cu << <grid, block, 0, cuda_stream >> >(surf_ref_m1_L3);
			add_3_conv_4x5x4_2_cu << <grid, block, 0, cuda_stream >> >(surf_ref_m1_L3);
			add_2_3_conv_4x5x4_R_cu << <grid, block, 0, cuda_stream >> >(surf_ref_m1_L3);
		}
		void CNNPP::run_HL(int model, Size2d& ROI, int index_output, cudaStream_t cuda_stream)
		{
			dim3 block(32, 32);
			dim3 grid(blockCount(ROI.cols, 32), blockCount(ROI.rows, 32));

			surf_ref_m1_HL.cols = ROI.cols;
			surf_ref_m1_HL.rows = ROI.rows;

			lrelu_bn_add4_tanh_add52_tanh_cu << <grid, block, 0, cuda_stream >> >(surf_ref_m1_HL, index_output);
		}

		//---------------------------------------------------------------------------------------------------------------------------------------------------------------

	}

#endif
}