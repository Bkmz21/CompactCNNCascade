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


#include "config.h"

#include "image.h"
#include "cnn_simd_cntk.h"
#include "cnn_simd_v2_cntk.h"

#include "timer.h"

#include "resource.h"
#include <sstream>
#include <fstream>
#include <windows.h>

#ifdef USE_CUDA
#	include "init_cuda.h"
#	include "cnn_cuda_cntk.h"
#endif

//OpenCV
#pragma comment(lib, "opencv_core300.lib")
#pragma comment(lib, "opencv_highgui300.lib")
#pragma comment(lib, "opencv_imgcodecs300.lib")
#pragma comment(lib, "opencv_imgproc300.lib")
#pragma comment(lib, "opencv_videoio300.lib")
#pragma comment(lib, "opencv_video300.lib")
#pragma comment(lib, "opencv_objdetect300.lib")

//Intel IPP
#include "ipp.h"
#pragma comment(lib, "ippcore.lib")
#pragma comment(lib, "ippi.lib")

//NVIDIA CUDA SDK
#include <npp.h>
#include <ImageIO.h>

//NVIDIA cuDNN
#include <cudnn.h>
#pragma comment(lib, "cudnn.lib")
#pragma comment(lib, "cublas.lib")

//#ifdef USE_CUDA
#pragma comment(lib, "cudart_static.lib")
#pragma comment(lib, "FreeImage64.lib")
#pragma comment(lib, "nppi.lib")
//#endif
#ifdef USE_CL
#	pragma comment(lib, "OpenCL.lib")
#endif

#include <malloc.h>
#include <iostream>
#include <vector>
#include <time.h>

//#define ConvNeuralNetwork_v2 ConvNeuralNetwork

using namespace NeuralNetworksLib;

#define NUM_THREADS 1

int num_launch = 10000;
#define NUM_LAUNCH num_launch


//================================================================================================================================================


template<typename type>
int init_data(SIMD::TmpImage<type>& img)
{
	srand(time(NULL));

	for (int j = 0; j < img.height; ++j)
	{
		for (int i = 0; i < img.width; ++i)
		{
			for (int c = 0; c < img.nChannel; ++c)
			{
				int offset = j * img.widthStep + img.nChannel * i + c;
				img.data[offset] = 255.f * (float)rand() / (float)RAND_MAX;
			}
		}
	}
	return 0;
}

template<typename type, const int pinned_mem = 0>
int check_data(SIMD::TmpImage<type>& img, SIMD::TmpImage<type>& img_2, float eps = 1.E-3)
{
	for (int j = 0; j < img.height; ++j)
	{
		for (int i = 0; i < img.width; ++i)
		{
			for (int c = 0; c < img.nChannel; ++c)
			{
				int offset = j * img.widthStep + img.nChannel * i + c;
				int offset_2 = j * img_2.widthStep + img.nChannel * i + c;

				if (abs(img.data[offset] - img_2.data[offset_2]) > eps)
				{
					printf("\n[TEST PERFOMANCE] y = %d, x = %d", j, i);
					printf("\n[TEST PERFOMANCE] img = %f, img_2 = %f\n", float(img.data[offset]), float(img_2.data[offset_2]));
					return -1;
				}
			}
		}
	}

	return 0;
}

#ifdef USE_CUDA
template<typename type, const int pinned_mem = 0>
int init_data(CUDA::TmpImage<type, pinned_mem>& img_cu, SIMD::TmpImage<type>& img)
{
	for (int j = 0; j < img_cu.height; ++j)
	{
		for (int i = 0; i < img_cu.width; ++i)
		{
			for (int c = 0; c < img_cu.nChannel; ++c)
			{
				int offset_cu = j * img_cu.widthStepHost + img.nChannel * i + c;
				int offset = j * img.widthStep + img.nChannel * i + c;
				img_cu.dataHost[offset_cu] = img.data[offset];
			}
		}
	}
	img_cu.updateDataDevice();
	return 0;
}

template<typename type, const int pinned_mem = 0>
int check_data(SIMD::TmpImage<type>& img, CUDA::TmpImage<type, pinned_mem>& img_cu, float eps = 1.E-3)
{
	img_cu.updateDataHost();

	for (int j = 0; j < img.height; ++j)
	{
		for (int i = 0; i < img.width; ++i)
		{
			for (int c = 0; c < img.nChannel; ++c)
			{
				int offset = j * img.widthStep + img.nChannel * i + c;
				int offset_cu = j * img_cu.widthStepHost + img.nChannel * i + c;

				if (abs(img.data[offset] - img_cu.dataHost[offset_cu]) > eps)
				{
					printf("\n[TEST PERFOMANCE] y = %d, x = %d", j, i);
					printf("\n[TEST PERFOMANCE] img = %f, img_cu = %f\n", float(img.data[offset]), float(img_cu.dataHost[offset_cu]));
					return -1;
				}
			}
		}
	}

	return 0;
}
#endif

#ifdef USE_CL
template<typename type>
int init_data(CL::TmpImage<type, 0>& img_cl, SIMD::TmpImage<type>& img)
{
	for (int j = 0; j < img_cl.height; ++j)
	{
		for (int i = 0; i < img_cl.width; ++i)
		{
			for (int c = 0; c < img_cl.nChannel; ++c)
			{
				int offset_cl = j * img_cl.widthStepHost + img.nChannel * i + c;
				int offset = j * img.widthStep + img.nChannel * i + c;
				img_cl.dataHost[offset_cl] = img.data[offset];
			}
		}
	}
	img_cl.updateDataDevice();
	return 0;
}

template<typename type>
int check_data(SIMD::TmpImage<type>& img, CL::TmpImage<type, 0>& img_cl, float eps = 1.E-3)
{
	img_cl.updateDataHost();

	for (int j = 0; j < img.height; ++j)
	{
		for (int i = 0; i < img.width; ++i)
		{
			for (int c = 0; c < img.nChannel; ++c)
			{
				int offset = j * img.widthStep + img.nChannel * i + c;
				int offset_cl = j * img_cl.widthStepHost + img.nChannel * i + c;

				if (abs(img.data[offset] - img_cl.dataHost[offset_cl]) > eps)
				{
					printf("\n[TEST PERFOMANCE] y = %d, x = %d", j, i);
					printf("\n[TEST PERFOMANCE] img = %f, img_cl = %f\n", float(img.data[offset]), float(img_cl.dataHost[offset_cl]));
					return -1;
				}
			}
		}
	}

	return 0;
}
#endif


void ipp_conv(Size img, Size kernel, int k)
{
	k *= NUM_LAUNCH;

	const IppLibraryVersion* IPPver;
	IPPver = ippGetLibVersion();
	printf("IPP %s %s\n", IPPver->Name, IPPver->Version);

	ippSetNumThreads(NUM_THREADS);

	int rx = img.width;
	int ry = img.height;
	int h_s = kernel.width;

	Ipp32f* src1 = (Ipp32f*)_mm_malloc((ry*rx)*sizeof(float), 32);
	for (int i = 0; i < ry; ++i)
	{
		for (int j = 0; j < rx; ++j)
		{
			src1[i*rx + j] = 1.f;// float(rand()) / RAND_MAX;
		}
	}

	int h_size = h_s;
	float sum = 0.f;
	Ipp32f* src2 = (Ipp32f*)_mm_malloc((h_size*h_size)*sizeof(float), 32);
	for (int i = 0; i < h_size; ++i)
	{
		for (int j = 0; j < h_size; ++j)
		{
			src2[i*h_size + j] = float(rand()) / RAND_MAX;
			sum += src2[i*h_size + j];
		}
	}

	Ipp32f* dst1 = (Ipp32f*)_mm_malloc((ry*rx)*sizeof(float), 32);
	Ipp32f* dst2 = (Ipp32f*)_mm_malloc((ry*rx)*sizeof(float), 32);
	for (int i = 0; i < ry; ++i)
	{
		for (int j = 0; j < rx; ++j)
		{
			dst1[i*rx + j] = 0;
			dst2[i*rx + j] = 0;
		}
	}

	IppStatus st;
	IppiSize src1Size = { rx, ry };
	IppiSize src2Size = { h_size, h_size };
	IppiSize filterRoiSize = { rx - (h_size - 1), ry - (h_size - 1) };
	IppiPoint k_point = { h_size - 1, h_size - 1 };
	Ipp8u* buff = (Ipp8u*)_mm_malloc((ry*rx)*sizeof(float), 32);

	printf("filterRoiSize = (%d, %d)\n", filterRoiSize.width, filterRoiSize.height);

	Timer timer;
	timer.start();
	for (int i = 0; i < k; ++i)
	{
		st = ippiFilter_32f_C1R(src1, rx*sizeof(Ipp32f), dst1, rx*sizeof(Ipp32f), filterRoiSize, src2, src2Size, k_point);
	}
	printf("IPP ippiFilter test = %f ms\n", timer.get(1000) / float(NUM_LAUNCH));

	printf("sum = %f\n", sum);
	printf("dst1[5] = %f\n", dst1[(filterRoiSize.height - 1) * rx + filterRoiSize.width - 1]);
	for (int i = 0; i < filterRoiSize.height; ++i)
	{
		for (int j = 0; j < filterRoiSize.width; ++j)
		{
			if (abs(dst1[i*rx + j] - sum) > 1.E-5f)
			{
				printf("err\n");
				system("pause");
			}
		}
	}

	timer.start();
	for (int i = 0; i < k; ++i)
	{
		st = ippiConvValid_32f_C1R(src1, rx*sizeof(Ipp32f), src1Size, src2, h_size*sizeof(Ipp32f), src2Size, dst2, rx*sizeof(Ipp32f));
	}
	printf("IPP ippiConvValid test = %f ms\n", timer.get(1000) / float(NUM_LAUNCH));

	printf("sum = %f\n", sum);
	printf("dst1[5] = %f\n", dst2[(filterRoiSize.height - 1) * rx + filterRoiSize.width - 1]);
	for (int i = 0; i < filterRoiSize.height; ++i)
	{
		for (int j = 0; j < filterRoiSize.width; ++j)
		{
			if (abs(dst1[i*rx + j] - sum) > 1.E-5f)
			{
				printf("err\n");
				system("pause");
			}
		}
	}
}

#ifdef USE_CUDA
void npp_conv_32f(Size img, Size kernel, int k)
{
	k *= NUM_LAUNCH;

	npp::ImageCPU_32f_C1 oHostSrc(img.width, img.height);
	for (int i = 0; i < img.width; ++i)
	{
		for (int j = 0; j < img.height; ++j)
		{
			*oHostSrc.data(i, j) = 1.f;// float(rand()) / RAND_MAX;
		}
	}

	npp::ImageNPP_32f_C1 oDeviceSrc(oHostSrc); // malloc and memcpy to GPU 
	NppiSize kernelSize = { kernel.width, kernel.height }; // dimensions of convolution kernel (Layer_filter)
	NppiSize oSizeROI = { oHostSrc.width() - kernelSize.width + 1, oHostSrc.height() - kernelSize.height + 1 };
	npp::ImageNPP_32f_C1 oDeviceDst(oSizeROI.width, oSizeROI.height); // allocate device image of appropriately reduced size
	npp::ImageCPU_32f_C1 oHostDst(oDeviceDst.size());
	NppiPoint oAnchor = { kernelSize.width - 1, kernelSize.height - 1 }; // found that oAnchor = {2,1} or {3,1} works for kernel [-1 0 1] 
	NppStatus eStatusNPP;

	Npp32f* hostKernel = new Npp32f[kernel.width*kernel.height]; // convolving with this should do edge detection
	float sum = 0.f;
	for (int i = 0; i < kernel.width*kernel.height; ++i)
	{
		hostKernel[i] = float(rand()) / RAND_MAX;
		sum += hostKernel[i];
	}

	Npp32f* deviceKernel;
	size_t deviceKernelPitch;
	cudaMalloc((void**)&deviceKernel, kernelSize.width * kernelSize.height * sizeof(Npp32s));
	cudaMemcpy(deviceKernel, hostKernel, kernelSize.width * kernelSize.height * sizeof(Npp32s), cudaMemcpyHostToDevice);

	CUDA::Timer timer;
	timer.start();
	for (int i = 0; i < k; ++i)
	{
		eStatusNPP = nppiFilter_32f_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
			oDeviceDst.data(), oDeviceDst.pitch(),
			oSizeROI, deviceKernel, kernelSize, oAnchor);
	}
	printf("NPP nppiFilter 32f test = %f ms\n", timer.get(1000) / float(NUM_LAUNCH));

	printf("NppiFilter error status = %d\n", int(eStatusNPP)); // prints 0 (no errors)
	oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch()); // memcpy to host

	printf("sum = %f\n", sum);
	printf("dst1[5000] = %f\n",* oHostDst.data(500, 500));
	for (int i = 0; i < oHostDst.height(); ++i)
	{
		for (int j = 0; j < oHostDst.width(); ++j)
		{
			if (abs(*oHostDst.data(j, i) - sum) > 1.E-5f)
			{
				printf("err\n");
				system("pause");
			}
		}
	}
}
void npp_conv_8u(Size img, Size kernel, int k)
{
	k *= NUM_LAUNCH;

	npp::ImageCPU_8u_C1 oHostSrc(img.width, img.height);
	for (int i = 0; i < img.width; ++i)
	{
		for (int j = 0; j < img.height; ++j)
		{
			*oHostSrc.data(i, j) = int(256 * rand()) / RAND_MAX;
		}
	}

	npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc); // malloc and memcpy to GPU 
	NppiSize kernelSize = { kernel.width, kernel.height }; // dimensions of convolution kernel (Layer_filter)
	NppiSize oSizeROI = { oHostSrc.width() - kernelSize.width + 1, oHostSrc.height() - kernelSize.height + 1 };
	npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height); // allocate device image of appropriately reduced size
	npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
	NppiPoint oAnchor = { kernelSize.width - 1, kernelSize.height - 1 }; // found that oAnchor = {2,1} or {3,1} works for kernel [-1 0 1] 
	NppStatus eStatusNPP;

	Npp32f* hostKernel = new Npp32f[kernel.width*kernel.height]; // convolving with this should do edge detection
	for (int i = 0; i < kernel.width*kernel.height; ++i)
	{
		hostKernel[i] = float(rand()) / RAND_MAX;
	}

	Npp32f* deviceKernel;
	size_t deviceKernelPitch;
	cudaMalloc((void**)&deviceKernel, kernelSize.width * kernelSize.height * sizeof(Npp32s));
	cudaMemcpy(deviceKernel, hostKernel, kernelSize.width * kernelSize.height * sizeof(Npp32s), cudaMemcpyHostToDevice);

	CUDA::Timer timer;
	timer.start();
	for (int i = 0; i < k; ++i)
	{
		eStatusNPP = nppiFilter32f_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
			oDeviceDst.data(), oDeviceDst.pitch(),
			oSizeROI, deviceKernel, kernelSize, oAnchor);
	}
	printf("NPP nppiFilter 8u test = %f ms\n", timer.get(1000) / float(NUM_LAUNCH));

	printf("NppiFilter error status = %d\n", int(eStatusNPP)); // prints 0 (no errors)
	oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch()); // memcpy to host
}


void cuDNN_conv(Size img, Size kernel, int k)
{
	cudnnHandle_t hCudNN = NULL;
	cudnnTensorDescriptor_t pInputDesc = NULL;
	cudnnFilterDescriptor_t pFilterDesc = NULL;
	cudnnConvolutionDescriptor_t pConvDesc = NULL;
	cudnnTensorDescriptor_t pOutputDesc = NULL;
	cudnnStatus_t status;
	cudaError_t err;
	int n_in = 1;	// Number of images - originally 128
	int c_in = 1;	// Number of feature maps per image - originally 96
	int h_in = img.height;	// Height of each feature map - originally 221
	int w_in = img.width;	// Width of each feature map - originally 221
	int k_pFilter_in = k;	// Number of output feature maps - originally 256
	int c_pFilter_in = c_in;	// Number of input feature maps - originally 96
	int h_pFilter_in = kernel.height;	// Height of each pFilter - originally 7
	int w_pFilter_in = kernel.width;	// Width of each pFilter - originally 7
	int n_out = 0;	// Number of output images.
	int c_out = 0;	// Number of output feature maps per image.
	int h_out = 0;	// Height of each output feature map.
	int w_out = 0;	// Width of each output feature map.

	/* to change to double, chance CUDNN_DATA_FLOAT to CUDNN_DATA_DOUBLE and change each float to double below*/

	cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
	int nDataTypeSize = (((int)dataType) + 1) * sizeof(float);
	float* pImageInBatch = NULL;
	float* pFilter = NULL;
	float* pImageOutBatch = NULL;

	try
	{
		//---------------------------------------
		// Create CudNN 
		//---------------------------------------

		status = cudnnCreate(&hCudNN);
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;


		//---------------------------------------
		// Create Descriptors
		//---------------------------------------

		status = cudnnCreateTensorDescriptor(&pInputDesc);
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;

		status = cudnnCreateTensorDescriptor(&pOutputDesc);
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;

		status = cudnnCreateFilterDescriptor(&pFilterDesc);
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;

		status = cudnnCreateConvolutionDescriptor(&pConvDesc);
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;


		//---------------------------------------
		// Allocate memory for pFilter and ImageBatch 
		//---------------------------------------

		err = cudaMalloc((void**)&pImageInBatch, n_in*c_in*h_in*w_in * nDataTypeSize);
		if (err != cudaSuccess)
			throw err;

		err = cudaMalloc((void**)&pFilter, k_pFilter_in*c_pFilter_in*h_pFilter_in*w_pFilter_in * nDataTypeSize);
		if (err != cudaSuccess)
			throw err;


		//---------------------------------------
		// Fill the input image and pFilter data
		//---------------------------------------

		//TODO: Still figuring this out


		//---------------------------------------
		// Set decriptors
		//---------------------------------------

		status = cudnnSetTensor4dDescriptor(pInputDesc, CUDNN_TENSOR_NCHW, dataType, n_in, c_in, h_in, w_in);
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;

		status = cudnnSetFilter4dDescriptor(pFilterDesc, dataType, k_pFilter_in, c_pFilter_in, h_pFilter_in, w_pFilter_in);
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;

		status = cudnnSetConvolution2dDescriptor(pConvDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION);
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;


		//---------------------------------------
		// Query output layout
		//---------------------------------------

		status = cudnnGetConvolution2dForwardOutputDim(pConvDesc, pInputDesc, pFilterDesc, &n_out, &c_out, &h_out, &w_out);
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;

		printf("n = %d, c = %d, w = %d, h = %d\n", n_out, c_out, w_out, h_out);
		

		//---------------------------------------
		// Set and allocate output tensor descriptor 
		//---------------------------------------

		status = cudnnSetTensor4dDescriptor(pOutputDesc, CUDNN_TENSOR_NCHW, dataType, n_out, c_out, h_out, w_out);
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;


		err = cudaMalloc((void**)&pImageOutBatch, n_out*c_out*h_out*w_out * nDataTypeSize);
		if (err != cudaSuccess)
			throw err;


		//---------------------------------------
		// Launch convolution on GPU 
		//---------------------------------------

		cudnnConvolutionFwdAlgo_t convAlgorithm = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

		if (1)
		{
			// Choose the best according to the preference
			std::cout << "Testing cudnnGetConvolutionForwardAlgorithm ...\n";
			cudnnConvolutionFwdAlgo_t algo;
			status = cudnnGetConvolutionForwardAlgorithm(
				hCudNN,
				pInputDesc,
				pFilterDesc,
				pConvDesc,
				pOutputDesc, 
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				0,
				&algo
				);
			if (status != CUDNN_STATUS_SUCCESS)
				throw status;

			std::cout << "Fastest algorithm is Algo " << algo << "\n";
			convAlgorithm = algo;

			// New way of finding the fastest config
			// Setup for findFastest call
			std::cout << "Testing cudnnFindConvolutionForwardAlgorithm ...\n";
			int requestedAlgoCount = 5;
			int returnedAlgoCount[1];
			cudnnConvolutionFwdAlgoPerf_t* results = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t)*requestedAlgoCount);
			status = cudnnFindConvolutionForwardAlgorithm(
				hCudNN,
				pInputDesc,
				pFilterDesc,
				pConvDesc,
				pOutputDesc,
				requestedAlgoCount,
				returnedAlgoCount,
				results
				);
			if (status != CUDNN_STATUS_SUCCESS)
				throw status;

			for (int algoIndex = 0; algoIndex <* returnedAlgoCount; ++algoIndex){
				printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", cudnnGetErrorString(results[algoIndex].status), results[algoIndex].algo, results[algoIndex].time, (unsigned long long)results[algoIndex].memory);
			}
			free(results);
		}

		size_t sizeInBytes = 0;
		void* workSpace = NULL;
		status = cudnnGetConvolutionForwardWorkspaceSize(
			hCudNN,
			pInputDesc,
			pFilterDesc,
			pConvDesc,
			pOutputDesc,
			convAlgorithm,
			&sizeInBytes);
		if (sizeInBytes != 0)
		{
			cuERR(cudaMalloc(&workSpace, sizeInBytes));
		}
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;

		float alpha = 1.f;
		float beta = 0.f;

		CUDA::Timer timer;
		timer.start();
		for (int i = 0; i < NUM_LAUNCH; ++i)
		{
			status = cudnnConvolutionForward(
				hCudNN, 
				&alpha, 
				pInputDesc,
				pImageInBatch, 
				pFilterDesc, 
				pFilter, 
				pConvDesc, 
				convAlgorithm,
				workSpace, 
				sizeInBytes, 
				&beta, 
				pOutputDesc,
				pImageOutBatch);
		}
		printf("cuDNN ConvolutionForward test = %f ms\n", timer.get(1000) / float(NUM_LAUNCH));
		printf("cuDNN error status = %d\n", int(CUDNN_STATUS_SUCCESS));
		
		if (status != CUDNN_STATUS_SUCCESS)
			throw status;


		//---------------------------------------
		// Extract output data 
		//---------------------------------------

		//TBD
	}
	catch (...)
	{
	}

	//---------------------------------------
	// Clean-up
	//---------------------------------------

	if (pImageInBatch != NULL)
		cudaFree(pImageInBatch);

	if (pImageOutBatch != NULL)
		cudaFree((void*)pImageOutBatch);

	if (pFilter != NULL)
		cudaFree((void*)pFilter);

	if (pInputDesc != NULL)
		cudnnDestroyTensorDescriptor(pInputDesc);

	if (pOutputDesc != NULL)
		cudnnDestroyTensorDescriptor(pOutputDesc);

	if (pFilterDesc != NULL)
		cudnnDestroyFilterDescriptor(pFilterDesc);

	if (pConvDesc != NULL)
		cudnnDestroyConvolutionDescriptor(pConvDesc);

	if (hCudNN != NULL)
		cudnnDestroy(hCudNN);
}
#endif

int our_cnn(Size init_size)
{
	printf("\n[TEST PERFOMANCE]  test cnn\n");
	if (1)
	{
		std::string path_model;

		for (int i = 0; i < 1; ++i)
		{
			HRSRC resource = FindResource(NULL, MAKEINTRESOURCE(110 + i), RT_RCDATA);
			HGLOBAL resourceData = LoadResource(NULL, resource);
			void* pBinaryData = LockResource(resourceData);
			unsigned int resourceSize = SizeofResource(NULL, resource);

			std::stringstream ss;
			ss << "cnn4face" << i + 1 << ".rc";

			std::ofstream binFile(ss.str().c_str(), std::ios::out | std::ios::binary);
			binFile.write((char*)pBinaryData, resourceSize);
			binFile.close();

			path_model = ss.str();
		}

#ifdef USE_AVX
		SIMD::ConvNeuralNetwork_v2* cnn_simd = new SIMD::ConvNeuralNetwork_v2();
#else
		SIMD::ConvNeuralNetwork* cnn_simd = new SIMD::ConvNeuralNetwork();
#endif

		cnn_simd->Init(path_model);
		if (cnn_simd->isEmpty())
		{
			delete cnn_simd;
			cnn_simd = NULL;
			return -1;
		}
		cnn_simd->AllocateMemory(init_size);
		cnn_simd->setNumThreads(NUM_THREADS);

		SIMD::Image_32f img(init_size.width, init_size.height, ALIGN_DEF, true);
		SIMD::Image_32f resp(cnn_simd->getOutputImgSize(init_size).width, cnn_simd->getOutputImgSize(init_size).height);
		init_data<float>(img);

#ifdef USE_CUDA
		CUDA::ConvNeuralNetwork* cnn_cuda = new CUDA::ConvNeuralNetwork();
		cnn_cuda->Init(path_model);
		if (cnn_cuda->isEmpty())
		{
			delete cnn_cuda;
			cnn_cuda = NULL;
			return -1;
		}
		cnn_cuda->AllocateMemory(init_size);

		CUDA::Image_32f img_cu(init_size.width, init_size.height, init_size.width,
			roundUpMul(init_size.width, cnn_cuda->getBlockSize().width),
			roundUpMul(init_size.height, cnn_cuda->getBlockSize().height),
			ALIGN_DEF);
		CUDA::Image_32f_pinned resp_cu(cnn_cuda->getOutputImgSize(init_size).width, cnn_cuda->getOutputImgSize(init_size).height);
		init_data<float>(img_cu, img);
#endif

#ifdef USE_CL
		CL::ConvNeuralNetwork* cnn_cl = new CL::ConvNeuralNetwork();
		cnn_cl->Init(path_model, device, context, queue);
		if (cnn_cl->isEmpty())
		{
			delete cnn_cl;
			cnn_cl = NULL;
			return -1;
		}
		cnn_cl->AllocateMemory(init_size);

		CL::Image_32f img_cl(context, queue, init_size.width, init_size.height, init_size.width,
			addRoundUpMul(init_size.width, cnn_cl->getBlockSize().width),
			addRoundUpMul(init_size.height, cnn_cl->getBlockSize().height),
			ALIGN_DEF);
		CL::Image_32f resp_cl(context, queue,
			cnn_cl->getOutputImgSize(init_size).width,
			cnn_cl->getOutputImgSize(init_size).height,
			cnn_cl->getOutputImgSize(init_size).width,
			roundUpMul(cnn_cl->getOutputImgSize(init_size).width, cnn_cl->getBlockSize().width),
			roundUpMul(cnn_cl->getOutputImgSize(init_size).height, cnn_cl->getBlockSize().height),
			ALIGN_DEF);
		init_data<float>(img_cl, img);
#endif

		std::remove(path_model.c_str());

#ifdef CHECK_TEST
		Legacy::ConvNeuralNetwork* old_cnn = new Legacy::ConvNeuralNetwork(path_model_1 + ".bin");
		if (old_cnn->isEmpty())
		{
			delete old_cnn;
			old_cnn = NULL;
			return -1;
		}

		cnn_simd->setCNNRef(old_cnn);
		CUDA_CODE(cnn_cuda->setCNNRef(old_cnn);)
			CL_CODE(cnn_cl->setCNNRef(old_cnn);)
#endif

		printf("[TEST PERFOMANCE] 	init_size = (%d, %d): \n", init_size.width, init_size.height);
		for (int i = 0; i <= 0; ++i)
		{
			Size size = init_size;
			printf("[TEST PERFOMANCE] 	size = (%d, %d):\n", size.width, size.height);

			img.width = size.width;
			img.height = size.height;
			init_data(img);

			Timer timer(1, true);
			for (int i = 0; i < NUM_LAUNCH; ++i) cnn_simd->Forward(resp, img);
			printf("[TEST PERFOMANCE] 		cnn_simd %f ms\n", timer.get(1000) / float(NUM_LAUNCH));

#ifdef USE_CUDA
			img_cu.width = size.width;
			img_cu.height = size.height;
			init_data<float>(img_cu, img);

			CUDA::Timer timer_cuda(true);
			for (int i = 0; i < NUM_LAUNCH; ++i) cnn_cuda->Forward(&resp_cu, &img_cu);
			printf("[TEST PERFOMANCE] 		cnn_cuda %f ms\n", timer_cuda.get(1000) / float(NUM_LAUNCH));
			//if (check_data<float>(resp, resp_cu, 1.E-1) < 0) return -1;
#endif

#ifdef USE_CL
			img_cl.width = size.width;
			img_cl.height = size.height;
			init_data<float>(img_cl, img);

			CL::Timer timer_cl(queue, true);
			cnn_cl->Forward(&resp_cl, &img_cl);
			printf("[TEST PERFOMANCE] 		cnn_cl   %7f ms\n", timer_cl.get(1000) / float(NUM_LAUNCH));
			if (check_data<float>(resp, resp_cl, 1.E-2) < 0) return -1;
#endif

			printf("[TEST PERFOMANCE] 	success\n\n");
		}

		delete cnn_simd;

#ifdef USE_CUDA
		delete cnn_cuda;
		CUDA::ReleaseDevice();
#endif

#ifdef USE_CL	
		delete cnn_cl;
		CL::ReleaseDevice(device, context, queue);
#endif

#ifdef CHECK_TEST
		delete old_cnn;
#endif
	}
}

int main(int argc, char* argv[])
{
	system("pause");
	for (int t = 0; t < 0; ++t)
	{
		int ks = 3;
		//Size img_size(3840, 2160);
		Size img_size(55, 51);
		ipp_conv(img_size, Size(ks, ks), 8);
		//npp_conv_32f(img_size, Size(ks, ks), 4);
		//npp_conv_8u(img_size, Size(ks, ks), 4000);
		//cuDNN_conv(img_size, Size(ks, ks), 4);
		our_cnn(img_size);
		system("pause");
	}

	num_launch = 1000;
	for (int t = 0; t < 10; ++t)
	{
		int ks = 4;
		Size img_size(3840, 2160);
		//Size img_size(55, 51);
		ipp_conv(img_size, Size(ks, ks), 4);
		//npp_conv_32f(img_size, Size(ks, ks), 4);
		//npp_conv_8u(img_size, Size(ks, ks), 4000);
		//cuDNN_conv(img_size, Size(ks, ks), 4);
		our_cnn(img_size);
		//system("pause");
	}

	printf("init\n");
	Timer timer;
	SIMD::CNNPP cnnpp;

	float shift = 0.01f;
	float scale = 1.01f;

	const int num_launch = 1000;
	size_t size = roundUpMul(1E7, REG_SIZE);
	printf("size = %d\n", size);
	printf("REG_SIZE = %d\n", REG_SIZE);
	printf("num_launch = %d\n", num_launch);

	float* in = static_cast<float*>(_mm_malloc(size*sizeof(float), ALIGN_DEF));
	float* out1 = static_cast<float*>(_mm_malloc(size*sizeof(float), ALIGN_DEF));
	float* out2 = static_cast<float*>(_mm_malloc(size*sizeof(float), ALIGN_DEF));
	float* out3 = static_cast<float*>(_mm_malloc(size*sizeof(float), ALIGN_DEF));
	float* out4 = static_cast<float*>(_mm_malloc(size*sizeof(float), ALIGN_DEF));

	srand(time(NULL));
	for (size_t i = 0; i < size; ++i)
	{
		in[i] = 4.f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		out1[i] = 0.f;
		out2[i] = 0.f;
		out3[i] = 0.f;
		out4[i] = 0.f;
		//if (i % 1000 == 0) printf("%4.2d %4.2f ", i, out1[i]);
	}

	printf("\ntest\n");
	float f = 0.;
	timer.start();
	for (int i = 0; i < num_launch; ++i)
	{
		cnnpp.tanh(out1, in, size, &(shift), &(scale));
		f += out1[i];
	}
	printf("tanh_approx = %f ms\n", timer.get(1000.) / static_cast<double>(num_launch));

	timer.start();
	for (int i = 0; i < num_launch; ++i)
	{
		//cnnpp.tanh_approx_exp(out2, in, size, &(shift), &(scale));
		f += out2[i];
	}
	printf("tanh_approx_exp = %f ms\n", timer.get(1000.) / static_cast<double>(num_launch));

	timer.start();
	for (int i = 0; i < num_launch; ++i)
	{
		//cnnpp.relu(out4, in, size, &(shift), &(scale));
		f += out4[i];
	}
	printf("relu = %f ms\n", timer.get(1000.) / static_cast<double>(num_launch));

	timer.start();
	for (int i = 0; i < num_launch; ++i)
	{
		for (size_t u = 0; u < size; ++u)
		{
			out3[u] = scale * tanhf(in[u] + shift);
		}
		f += out3[i];
	}
	printf("tanhf = %f ms\n", timer.get(1000.) / static_cast<double>(num_launch));
	printf("\nf = %f ms\n", f);

	printf("\nverification\n");
	size_t idx = static_cast<size_t>(static_cast<float>(size) * static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
	printf("idx = %d\n", idx);
	printf("	tanhf = %10.8f\n", out3[idx]);
	printf("	tanh_approx = %10.8f, error = %f%%\n", out1[idx], abs(100. * (out1[idx] - out3[idx]) / out3[idx]));
	printf("	tanh_approx_exp = %10.8f, error = %f%%\n", out2[idx], abs(100. * (out2[idx] - out3[idx]) / out3[idx]));

	double sum1 = 0.;
	double sum2 = 0.;
	double sum3 = 0.;
	for (size_t i = 0; i < size; ++i)
	{
		//if (i % 1000 == 0) printf("%4.2d %4.2f ", i, out1[i]);
		sum1 += static_cast<double>(out1[i]);
		sum2 += static_cast<double>(out2[i]);
		sum3 += static_cast<double>(out3[i]);
	}
	sum1 /= static_cast<double>(size);
	sum2 /= static_cast<double>(size);
	sum3 /= static_cast<double>(size);

	printf("\naverage:\n"); 
	printf("	tanhf = %f\n", sum3);
	printf("	tanh_approx = %f, error = %f%%\n", sum1, abs(100. * (sum1 - sum3) / sum3));
	printf("	tanh_approx_exp = %f, error = %f%%\n", sum2, abs(100. * (sum2 - sum3) / sum3));

	printf("\n");
	system("pause");
}