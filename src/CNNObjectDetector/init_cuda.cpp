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


#include "init_cuda.h"
#include <iostream>


//================================================================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_CUDA

	namespace CUDA
	{
		int InfoDevice()
		{
			int deviceCount;
			cudaDeviceProp deviceProp;

			cudaError_t cuErr = cudaGetDeviceCount(&deviceCount);
			printf("[CNNDetector] [CUDA] Device count: %d\n\n", deviceCount);

			for (int i = 0; i < deviceCount; ++i)
			{
				cuERR(cudaGetDeviceProperties(&deviceProp, i));
				printf("[CNNDetector] [CUDA] Device id: %d\n", i);
				printf("[CNNDetector] [CUDA] Device name: %s\n", deviceProp.name);
				printf("[CNNDetector] [CUDA] Total global memory: %d\n", deviceProp.totalGlobalMem);
				printf("[CNNDetector] [CUDA] Shared memory per block: %d\n", deviceProp.sharedMemPerBlock);
				printf("[CNNDetector] [CUDA] Registers per block: %d\n", deviceProp.regsPerBlock);
				printf("[CNNDetector] [CUDA] Warp size: %d\n", deviceProp.warpSize);
				printf("[CNNDetector] [CUDA] Memory pitch: %d\n", deviceProp.memPitch);
				printf("[CNNDetector] [CUDA] Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

				printf("[CNNDetector] [CUDA] Max threads dimensions: x = %d, y = %d, z = %d\n",
					deviceProp.maxThreadsDim[0],
					deviceProp.maxThreadsDim[1],
					deviceProp.maxThreadsDim[2]);

				printf("[CNNDetector] [CUDA] Max grid size: x = %d, y = %d, z = %d\n",
					deviceProp.maxGridSize[0],
					deviceProp.maxGridSize[1],
					deviceProp.maxGridSize[2]);

				printf("[CNNDetector] [CUDA] Clock rate: %d\n", deviceProp.clockRate);
				printf("[CNNDetector] [CUDA] Total constant memory: %d\n", deviceProp.totalConstMem);
				printf("[CNNDetector] [CUDA] Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
				printf("[CNNDetector] [CUDA] Texture alignment: %d\n", deviceProp.textureAlignment);
				printf("[CNNDetector] [CUDA] Device overlap: %d\n", deviceProp.deviceOverlap);
				printf("[CNNDetector] [CUDA] Multiprocessor count: %d\n", deviceProp.multiProcessorCount);

				printf("[CNNDetector] [CUDA] Kernel execution concurrent support: %s\n",
					deviceProp.concurrentKernels ? "true" : "false");

				printf("[CNNDetector] [CUDA] Kernel execution timeout enabled: %s\n\n",
					deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
			}

			return 0;
		}
		int InitDevice(int device_id, bool info)
		{
			int devices_n = 0;
			cudaError_t cuErr = cudaGetDeviceCount(&devices_n);

			if (cuErr == cudaError::cudaErrorInsufficientDriver)
			{
				printf("[CNNDetector] CUDA no support!\n");
				return -1;
			}

			if (devices_n == 0)
			{
				printf("[CNNDetector] CUDA device not found!\n");
				return -1;
			}

			if (info) InfoDevice();

			if (device_id == -1)
			{
				std::cout << "[CNNDetector] [CUDA] Enter device id: ";
				std::cin >> device_id;
			}
			else
			{
				if (info) printf("[CNNDetector] [CUDA] CUDA device id selected: %d\n", device_id);
			}

			if (device_id >= devices_n)
			{
				printf("[CNNDetector] CUDA device not found!\n");
				return -1;
			}

			//init device
			cuERR(cudaSetDeviceFlags(cudaDeviceMapHost));
			cuERR(cudaSetDevice(device_id));
			cudaDeviceProp deviceProp;
			cuERR(cudaGetDeviceProperties(&deviceProp, 0));

			if (!deviceProp.asyncEngineCount)
			{
				printf("[CNNDetector] Device no support DMA engines!\n");
				return -1;
			}
			//if (!deviceProp.canMapHostMemory)
			//{
			//	printf("[CNNDetector] Device no support memory-mapped\n");
			//	return -1;
			//}

			return 0;
		}
		void ReleaseDevice()
		{
			cuERR(cudaDeviceReset());
		}
	}

#endif
}