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


#include "image_proc_cl.h"
#ifdef USE_RESOURCE
	#include <windows.h>
#endif


//================================================================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_CL

	namespace CL
	{
		cl_program ImageConverterCLProgram = 0;

		cl_kernel Img8uToImg32f_cl = 0;
		cl_kernel Img8uBGRToImg32fGRAY_cl = 0;
		cl_kernel Img8uBGRAToImg32fGRAY_cl = 0;
		cl_kernel ImgColBlur_cl = 0;
		cl_kernel ImgRowBlur_cl = 0;

		void ImageConverter::create_cl_program(cl_device_id _device, cl_context _context)
		{
#ifdef USE_RESOURCE
			HMODULE module = NULL;
			GetModuleHandleExW(
				GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(&ImageConverter::create_cl_program),
				&module);
			HRSRC resource = FindResource(module, MAKEINTRESOURCE(IMAGE_PROC_CL_KERNELS), RT_RCDATA);
			if (resource == 0)
			{
				printf("Resource IMAGE_PROC_CL_KERNELS not found!\n");
				return;
			}
			
			HGLOBAL resourceData = LoadResource(module, resource);
			void* pBinaryData = LockResource(resourceData);
			unsigned int resourceSize = SizeofResource(module, resource);

			std::string str_source_code;
			str_source_code.append((char*)pBinaryData, resourceSize);
			const char* c_source_code[1] = { str_source_code.c_str() };
#else
			std::ifstream source_file(SOURCE"image_proc_cl_kernels.cl");
			std::string str_source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
			const char* c_source_code[1] = { str_source_code.c_str() };
#endif

			cl_int _err;
			ImageConverterCLProgram = clCreateProgramWithSource(_context, sizeof(c_source_code) / sizeof(*c_source_code), c_source_code, NULL, &_err);
			if (clBuildProgram(ImageConverterCLProgram, 1, &_device, "", NULL, NULL) != CL_SUCCESS)
			{
				char buffer[10240];
				clGetProgramBuildInfo(ImageConverterCLProgram, _device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
				fprintf(stderr, "CL Compilation failed:\n%s", buffer);
				abort();
			}
			clUnloadCompiler();

			//create kernels
			Img8uToImg32f_cl = clCreateKernel(ImageConverterCLProgram, "Img8uToImg32f_cl", &_err);
			clERR(_err);

			Img8uBGRToImg32fGRAY_cl = clCreateKernel(ImageConverterCLProgram, "Img8uBGRToImg32fGRAY_cl", &_err);
			clERR(_err);

			Img8uBGRAToImg32fGRAY_cl = clCreateKernel(ImageConverterCLProgram, "Img8uBGRAToImg32fGRAY_cl", &_err);
			clERR(_err);

			ImgColBlur_cl = clCreateKernel(ImageConverterCLProgram, "ImgColBlur_cl", &_err);
			clERR(_err);

			ImgRowBlur_cl = clCreateKernel(ImageConverterCLProgram, "ImgRowBlur_cl", &_err);
			clERR(_err);
		}
		void ImageConverter::destroy_cl_program()
		{
			if (ImageConverterCLProgram != 0)
			{
				clERR(clReleaseKernel(Img8uToImg32f_cl));
				clERR(clReleaseKernel(Img8uBGRToImg32fGRAY_cl));
				clERR(clReleaseKernel(Img8uBGRAToImg32fGRAY_cl));
				clERR(clReleaseKernel(ImgColBlur_cl));
				clERR(clReleaseKernel(ImgRowBlur_cl));
				clERR(clReleaseProgram(ImageConverterCLProgram));
			}
		}

		int ImageConverter::Img8uToImg32fGRAY(Image_32f* dst, Image_8u* src, cl_command_queue _queue)
		{
			size_t global_work_size[2] = { (size_t)roundUp(src->width, 2), (size_t)roundUp(src->height, 2) };

			switch (src->nChannel)
			{
			case 1:
				clERR(clSetKernelArg(Img8uToImg32f_cl, 0, sizeof(dst->dataDevice), &dst->dataDevice));
				clERR(clSetKernelArg(Img8uToImg32f_cl, 1, sizeof(dst->widthStepDevice), &dst->widthStepDevice));
				clERR(clSetKernelArg(Img8uToImg32f_cl, 2, sizeof(src->dataDevice), &src->dataDevice));
				clERR(clSetKernelArg(Img8uToImg32f_cl, 3, sizeof(src->width), &src->width));
				clERR(clSetKernelArg(Img8uToImg32f_cl, 4, sizeof(src->height), &src->height));
				clERR(clSetKernelArg(Img8uToImg32f_cl, 5, sizeof(src->widthStepDevice), &src->widthStepDevice));
				clERR(clEnqueueNDRangeKernel(_queue, Img8uToImg32f_cl, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
				break;

			case 3:
				clERR(clSetKernelArg(Img8uBGRToImg32fGRAY_cl, 0, sizeof(dst->dataDevice), &dst->dataDevice));
				clERR(clSetKernelArg(Img8uBGRToImg32fGRAY_cl, 1, sizeof(dst->widthStepDevice), &dst->widthStepDevice));
				clERR(clSetKernelArg(Img8uBGRToImg32fGRAY_cl, 2, sizeof(src->dataDevice), &src->dataDevice));
				clERR(clSetKernelArg(Img8uBGRToImg32fGRAY_cl, 3, sizeof(src->width), &src->width));
				clERR(clSetKernelArg(Img8uBGRToImg32fGRAY_cl, 4, sizeof(src->height), &src->height));
				clERR(clSetKernelArg(Img8uBGRToImg32fGRAY_cl, 5, sizeof(src->widthStepDevice), &src->widthStepDevice));
				clERR(clEnqueueNDRangeKernel(_queue, Img8uBGRToImg32fGRAY_cl, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
				break;

			case 4:
				clERR(clSetKernelArg(Img8uBGRAToImg32fGRAY_cl, 0, sizeof(dst->dataDevice), &dst->dataDevice));
				clERR(clSetKernelArg(Img8uBGRAToImg32fGRAY_cl, 1, sizeof(dst->widthStepDevice), &dst->widthStepDevice));
				clERR(clSetKernelArg(Img8uBGRAToImg32fGRAY_cl, 2, sizeof(src->dataDevice), &src->dataDevice));
				clERR(clSetKernelArg(Img8uBGRAToImg32fGRAY_cl, 3, sizeof(src->width), &src->width));
				clERR(clSetKernelArg(Img8uBGRAToImg32fGRAY_cl, 4, sizeof(src->height), &src->height));
				clERR(clSetKernelArg(Img8uBGRAToImg32fGRAY_cl, 5, sizeof(src->widthStepDevice), &src->widthStepDevice));
				clERR(clEnqueueNDRangeKernel(_queue, Img8uBGRAToImg32fGRAY_cl, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
				break;

			default:
				return -1;
			}

			return 0;
		}
		int ImageConverter::Img8uToImg32fGRAY_blur(Image_32f* dst, Image_8u* src, float* kernel_col, float* kernel_row, Image_32f* temp_buff, cl_command_queue _queue)
		{
			Img8uToImg32fGRAY(dst, src, _queue);

			size_t global_work_size_row[2] = { (size_t)dst->width, (size_t)dst->height };
			clERR(clSetKernelArg(ImgRowBlur_cl, 0, sizeof(temp_buff->dataDevice), &temp_buff->dataDevice));
			clERR(clSetKernelArg(ImgRowBlur_cl, 1, sizeof(temp_buff->widthStepDevice), &temp_buff->widthStepDevice));
			clERR(clSetKernelArg(ImgRowBlur_cl, 2, sizeof(dst->dataDevice), &dst->dataDevice));
			clERR(clSetKernelArg(ImgRowBlur_cl, 3, sizeof(dst->widthStepDevice), &dst->widthStepDevice));
			clERR(clSetKernelArg(ImgRowBlur_cl, 4, sizeof(kernel_row[0]), &kernel_row[0]));
			clERR(clSetKernelArg(ImgRowBlur_cl, 5, sizeof(kernel_row[1]), &kernel_row[1]));
			clERR(clSetKernelArg(ImgRowBlur_cl, 6, sizeof(kernel_row[2]), &kernel_row[2]));
			clERR(clEnqueueNDRangeKernel(_queue, ImgRowBlur_cl, 2, NULL, global_work_size_row, NULL, 0, NULL, NULL));

			size_t global_work_size_col[2] = { (size_t)dst->width, (size_t)dst->height - 2 };
			clERR(clSetKernelArg(ImgColBlur_cl, 0, sizeof(dst->dataDevice), &dst->dataDevice));
			clERR(clSetKernelArg(ImgColBlur_cl, 1, sizeof(dst->widthStepDevice), &dst->widthStepDevice));
			clERR(clSetKernelArg(ImgColBlur_cl, 2, sizeof(temp_buff->dataDevice), &temp_buff->dataDevice));
			clERR(clSetKernelArg(ImgColBlur_cl, 3, sizeof(temp_buff->widthStepDevice), &temp_buff->widthStepDevice));
			clERR(clSetKernelArg(ImgColBlur_cl, 4, sizeof(kernel_col[0]), &kernel_col[0]));
			clERR(clSetKernelArg(ImgColBlur_cl, 5, sizeof(kernel_col[1]), &kernel_col[1]));
			clERR(clSetKernelArg(ImgColBlur_cl, 6, sizeof(kernel_col[2]), &kernel_col[2]));
			clERR(clEnqueueNDRangeKernel(_queue, ImgColBlur_cl, 2, NULL, global_work_size_col, NULL, 0, NULL, NULL));

			return 0;
		}
	}

#endif
}