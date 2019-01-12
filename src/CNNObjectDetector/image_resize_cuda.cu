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


#include "image_resize_cuda.h"


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
			int offset;
		};

		typedef texture<float, cudaTextureType2D, cudaReadModeElementType> Texture;
		Texture tex;
		cudaChannelFormatDesc ir_tex_channelDesc;

		const uint_ QUANT_BIT = 12;	//8~12
		const uint_ QUANT_BIT2 = 2 * QUANT_BIT;
		float QUANT_BIT_f32 = 0.f;
		float QUANT_BIT2_f32 = 0.f;

		__global__ void NearestNeighborInterpolation_cu(SurfRef surf_ref, const uint_ sclx, const uint_ scly)
		{
			const uint_ i = blockIdx.x * 32;
			const uint_ j = blockIdx.y * 32;

			const uint_ ix = i + threadIdx.x;
			const uint_ iy = j + threadIdx.y;

			if (ix >= surf_ref.cols || iy >= surf_ref.rows) return;

			// make LUT for y-axis
			const uint_ x = ix * sclx;
			const uint_ px = (x >> QUANT_BIT);

			const uint_ y = iy * scly;
			const uint_ py = (y >> QUANT_BIT);

			const float tx = px + 0.5f;
			const float ty = py + 0.5f;

			surf_ref.surf[surf_ref.offset + iy * surf_ref.step + ix] = tex2D(tex, tx, ty);
		}
		__global__ void BilinearInterpolation_cu(SurfRef surf_ref, const uint_ sclx, const uint_ scly, const float QUANT_BIT_f32, const float QUANT_BIT2_f32)
		{
			const uint_ i = blockIdx.x * 32;
			const uint_ j = blockIdx.y * 32;

			const uint_ ix = i + threadIdx.x;
			const uint_ iy = j + threadIdx.y;

			if (ix >= surf_ref.cols || iy >= surf_ref.rows) return;

			// make LUT for y-axis
			const uint_ x = ix * sclx;
			const uint_ px = (x >> QUANT_BIT);
			const float fx = float(x - (px << QUANT_BIT));
			const float cx = QUANT_BIT_f32 - fx;

			const uint_ y = iy * scly;
			const uint_ py = (y >> QUANT_BIT);
			const float fy = float(y - (py << QUANT_BIT));
			const float cy = QUANT_BIT_f32 - fy;

			const float tx = px + 0.5f;
			const float ty = py + 0.5f;

			// four neighbor pixels
			const float p0 = tex2D(tex, tx, ty);
			const float p1 = tex2D(tex, tx + 1.f, ty);
			const float p2 = tex2D(tex, tx, ty + 1.f);
			const float p3 = tex2D(tex, tx + 1.f, ty + 1.f);

			// Calculate the weighted sum of pixels
			const float outv = (p0 * cx + p1 * fx) * cy + (p2 * cx + p3 * fx) * fy;
			surf_ref.surf[surf_ref.offset + iy * surf_ref.step + ix] = outv * QUANT_BIT2_f32;
		}

		void ImageResizer::Init()
		{
			tex.addressMode[0] = cudaAddressModeClamp;
			tex.addressMode[1] = cudaAddressModeClamp;
			tex.filterMode = cudaFilterModeLinear;
			tex.normalized = false;
			ir_tex_channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	
			cuERR(cudaFuncSetCacheConfig(NearestNeighborInterpolation_cu, cudaFuncCachePreferL1));
			cuERR(cudaFuncSetCacheConfig(BilinearInterpolation_cu, cudaFuncCachePreferL1));

			QUANT_BIT_f32 = static_cast<const float>(1 << QUANT_BIT);
			QUANT_BIT2_f32 = static_cast<const float>(1.0 / pow(2.0, (double)QUANT_BIT2));
		}
		void ImageResizer::FastImageResize(CUDA::Image_32f* dst, CUDA::Image_32f_pinned* src, const int type_resize, cudaStream_t cuda_stream)
		{
			if (dst->width == src->width && dst->height == src->height)
			{
				dst->copyDataDeviceToDevice((CUDA::Image_32f*)src, true, cuda_stream);
				return;
			}

			dim3 block(32, 32);
			dim3 grid(blockCount(dst->width, 32), blockCount(dst->height, 32));

			cuERR(cudaBindTexture2D(NULL, &tex, src->dataDevice, &ir_tex_channelDesc, (size_t)src->width, (size_t)src->height, (size_t)src->widthStepDevice * sizeof(float)));

			SurfRef surf_ref;
			surf_ref.surf = dst->dataDevice;
			surf_ref.cols = dst->width;
			surf_ref.rows = dst->height;
			surf_ref.step = dst->widthStepDevice;
			surf_ref.offset = dst->offsetDevice;

			const uint_ sclx = (src->width << QUANT_BIT) / dst->width + 1;
			const uint_ scly = (src->height << QUANT_BIT) / dst->height + 1;

			switch (type_resize)
			{
			default:
			case 0:
				NearestNeighborInterpolation_cu << <grid, block, 0, cuda_stream >> >(surf_ref, sclx, scly);
				break;

			case 1:
				BilinearInterpolation_cu << <grid, block, 0, cuda_stream >> >(surf_ref, sclx, scly, QUANT_BIT_f32, QUANT_BIT2_f32);
			}

			cuERR(cudaUnbindTexture(&tex));
		}
	}

#endif
}
