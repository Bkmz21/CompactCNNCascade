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


#include "image_resize_cl.h"
#ifdef USE_RESOURCE
	#include <windows.h>
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_CL

	namespace CL
	{
		cl_program ImageResizeCLProgram = 0;

		cl_kernel NearestNeighborInterpolation_cl = 0;
		cl_kernel BilinearInterpolation_cl = 0;

#ifdef USE_RESOURCE
		HMODULE getCurrentModuleHandle()
		{
			HMODULE hMod = NULL;
			GetModuleHandleExW(
				GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(&getCurrentModuleHandle),
				&hMod);

			return hMod;
		}
#endif

		void ImageResizer::create_cl_program(cl_device_id _device, cl_context _context)
		{
#ifdef USE_RESOURCE
			HMODULE module = NULL;
			GetModuleHandleExW(
				GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(&ImageResizer::create_cl_program),
				&module);
			HRSRC resource = FindResource(module, MAKEINTRESOURCE(IMAGE_RESIZE_CL_KERNELS), RT_RCDATA);
			if (resource == 0)
			{
				printf("Resource IMAGE_RESIZE_CL_KERNELS not found!\n");
				return;
			}

			HGLOBAL resourceData = LoadResource(module, resource);
			void* pBinaryData = LockResource(resourceData);
			unsigned int resourceSize = SizeofResource(module, resource);

			std::string str_source_code;
			str_source_code.append((char*)pBinaryData, resourceSize);
			const char* c_source_code[1] = { str_source_code.c_str() };
#else
			std::ifstream source_file(SOURCE"imege_resize_cl_kernels.cl");
			std::string str_source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
			const char* c_source_code[1] = { str_source_code.c_str() };
#endif

			cl_int _err;
			ImageResizeCLProgram = clCreateProgramWithSource(_context, sizeof(c_source_code) / sizeof(*c_source_code), c_source_code, NULL, &_err);
			if (clBuildProgram(ImageResizeCLProgram, 1, &_device, "", NULL, NULL) != CL_SUCCESS)
			{
				char buffer[10240];
				clGetProgramBuildInfo(ImageResizeCLProgram, _device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
				fprintf(stderr, "CL Compilation failed:\n%s", buffer);
				abort();
			}
			clUnloadCompiler();

			//create kernels
			NearestNeighborInterpolation_cl = clCreateKernel(ImageResizeCLProgram, "NearestNeighborInterpolation_cl", &_err);
			clERR(_err);

			BilinearInterpolation_cl = clCreateKernel(ImageResizeCLProgram, "BilinearInterpolation_cl", &_err);
			clERR(_err);
		}
		void ImageResizer::destroy_cl_program()
		{
			if (ImageResizeCLProgram != 0)
			{
				clERR(clReleaseKernel(NearestNeighborInterpolation_cl));
				clERR(clReleaseKernel(BilinearInterpolation_cl));
				clERR(clReleaseProgram(ImageResizeCLProgram));
			}
		}

		void ImageResizer::FastImageResize(Image_32f* dst, Image_32f* src, const int type_resize, cl_command_queue _queue)
		{
			if (dst->width == src->width && dst->height == src->height && dst->widthStepDevice == src->widthStepDevice)
			{
				dst->copyDataDeviceToDevice(src);
				return;
			}

			const uint_ QUANT_BIT = 12;	//8~12
			const uint_ QUANT_BIT2 = 2 * QUANT_BIT;
			const uint_ sclx = (src->width << QUANT_BIT) / dst->width + 1;
			const uint_ scly = (src->height << QUANT_BIT) / dst->height + 1;

			//cl_event kernel_completion;
			size_t global_work_size[2] = { (size_t)dst->width, (size_t)dst->height };

			switch (type_resize)
			{
			default:
			case 0:
				clERR(clSetKernelArg(NearestNeighborInterpolation_cl, 0, sizeof(dst->dataDevice), &dst->dataDevice));
				clERR(clSetKernelArg(NearestNeighborInterpolation_cl, 1, sizeof(dst->widthStepDevice), &dst->widthStepDevice));
				clERR(clSetKernelArg(NearestNeighborInterpolation_cl, 2, sizeof(dst->offsetDevice), &dst->offsetDevice));
				clERR(clSetKernelArg(NearestNeighborInterpolation_cl, 3, sizeof(src->dataDevice), &src->dataDevice));
				clERR(clSetKernelArg(NearestNeighborInterpolation_cl, 4, sizeof(src->width), &src->width));
				clERR(clSetKernelArg(NearestNeighborInterpolation_cl, 5, sizeof(src->height), &src->height));
				clERR(clSetKernelArg(NearestNeighborInterpolation_cl, 6, sizeof(src->widthStepDevice), &src->widthStepDevice));
				clERR(clSetKernelArg(NearestNeighborInterpolation_cl, 7, sizeof(sclx), &sclx));
				clERR(clSetKernelArg(NearestNeighborInterpolation_cl, 8, sizeof(scly), &scly));
				clERR(clSetKernelArg(NearestNeighborInterpolation_cl, 9, sizeof(QUANT_BIT), &QUANT_BIT));
				clERR(clEnqueueNDRangeKernel(_queue, NearestNeighborInterpolation_cl, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
				break;

			case 1:
				const float QUANT_BIT_f32 = static_cast<const float>(1 << QUANT_BIT);
				const float QUANT_BIT2_f32 = static_cast<const float>(1.0 / pow(2.0, (double)QUANT_BIT2));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 0, sizeof(dst->dataDevice), &dst->dataDevice));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 1, sizeof(dst->widthStepDevice), &dst->widthStepDevice));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 2, sizeof(dst->offsetDevice), &dst->offsetDevice));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 3, sizeof(src->dataDevice), &src->dataDevice));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 4, sizeof(src->width), &src->width));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 5, sizeof(src->height), &src->height));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 6, sizeof(src->widthStepDevice), &src->widthStepDevice));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 7, sizeof(sclx), &sclx));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 8, sizeof(scly), &scly));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 9, sizeof(QUANT_BIT), &QUANT_BIT));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 10, sizeof(QUANT_BIT_f32), &QUANT_BIT_f32));
				clERR(clSetKernelArg(BilinearInterpolation_cl, 11, sizeof(QUANT_BIT2_f32), &QUANT_BIT2_f32));
				clERR(clEnqueueNDRangeKernel(_queue, BilinearInterpolation_cl, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
			break;
			}
		}
	}

#endif
}