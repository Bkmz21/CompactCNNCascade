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


#include "image_cl.h"
#include <iostream>


//================================================================================================================================================


namespace NeuralNetworksLib
{
#ifdef USE_CL

	namespace CL
	{
		int InfoDevice()
		{
			cl_uint platforms_n = 0;
			cl_platform_id platforms[100];
			clERR(clGetPlatformIDs(100, platforms, &platforms_n));

			if (platforms_n == 0) return -1;

			printf("[CNNDetector] OpenCL platform(s) found: %d\n\n", platforms_n);

			int devices_count = 0;
			for (int i = 0; i < (int)platforms_n; ++i)
			{
				char buffer[10240];
				printf("[CNNDetector] [OpenCL] Platform id: %d\n", i);
				clERR(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240, buffer, NULL));
				printf("[CNNDetector] [OpenCL] Version: %s\n", buffer);
				clERR(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
				printf("[CNNDetector] [OpenCL] Name: %s\n", buffer);
				clERR(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL));
				printf("[CNNDetector] [OpenCL] Vendor: %s\n", buffer);
				printf("\n");

				cl_uint devices_n = 0;
				cl_device_id devices[100];
				clERR(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 100, devices, &devices_n));
				devices_count += devices_n;

				printf("[CNNDetector] [OpenCL]	Device(s) found on platform(s): %d\n\n", devices_n);
				for (int j = 0; j < (int)devices_n; ++j)
				{
					char buffer[10240];
					cl_uint buf_uint;
					cl_ulong buf_ulong;
					cl_device_type dev_type;

					printf("[CNNDetector] [OpenCL]	Device id: %d\n", j);
					clERR(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
					printf("[CNNDetector] [OpenCL]	Device name = %s\n", buffer);
					clERR(clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL));
					if (dev_type == CL_DEVICE_TYPE_CPU) printf("[CNNDetector] [OpenCL]	Device type = CPU\n");
					if (dev_type == CL_DEVICE_TYPE_GPU) printf("[CNNDetector] [OpenCL]	Device type = GPU\n");
					if (dev_type == CL_DEVICE_TYPE_ACCELERATOR) printf("[CNNDetector] [OpenCL]	Device type = ACCELERATOR\n");
					clERR(clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
					printf("[CNNDetector] [OpenCL]	Device vendor = %s\n", buffer);
					clERR(clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
					printf("[CNNDetector] [OpenCL]	Device version = %s\n", buffer);
					clERR(clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
					printf("[CNNDetector] [OpenCL]	Driver version = %s\n", buffer);
					clERR(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL));
					printf("[CNNDetector] [OpenCL]	Device max compute units = %u\n", (unsigned int)buf_uint);
					clERR(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL));
					printf("[CNNDetector] [OpenCL]	Device max clock frequency = %u\n", (unsigned int)buf_uint);
					clERR(clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
					printf("[CNNDetector] [OpenCL]	Device global mem size = %llu\n", (unsigned long long)buf_ulong);
					clERR(clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
					printf("[CNNDetector] [OpenCL]	Device local mem size = %llu\n", (unsigned long long)buf_ulong);
					printf("\n");
				}
			}

			if (devices_count == 0) return -1;
			return 0;
		}
		int InitDevice(cl_uint platform_id, cl_uint device_id, cl_device_id& device, cl_context& context, cl_command_queue& queue, bool info)
		{
			if (info && InfoDevice() < 0)
			{
				printf("[CNNDetector] OpenCL device not found!\n");
				return -1;
			}

			cl_uint platforms_n = 0;
			cl_platform_id platforms[10];
			clERR(clGetPlatformIDs(10, platforms, &platforms_n));

			if (platforms_n == 0)
			{
				printf("[CNNDetector] OpenCL platform not found!\n");
				return -1;
			}

			if (platform_id == -1)
			{
				std::cout << "[CNNDetector] [OpenCL] Enter platform id: ";
				std::cin >> platform_id;
			}
			else
			{
				if (info) printf("[CNNDetector] [OpenCL] OpenCL platform id selected: %d\n", platform_id);
			}

			if (platform_id >= platforms_n)
			{
				printf("[CNNDetector] OpenCL platform not found!\n");
				return -1;
			}

			cl_device_id devices[10];
			cl_uint devices_n = 0;
			clERR(clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_ALL, 10, devices, &devices_n));

			if (devices_n == 0)
			{
				printf("[CNNDetector] OpenCL device not found!\n");
				return -1;
			}

			if (device_id == -1)
			{
				std::cout << "[CNNDetector] [OpenCL] Enter device id: ";
				std::cin >> device_id;
			}
			else
			{
				if (info) printf("[CNNDetector] [OpenCL] OpenCL device id selected: %d\n", device_id);
			}

			if (device_id >= devices_n)
			{
				printf("[CNNDetector] OpenCL device not found!\n");
				return -1;
			}

			device = devices[device_id];

			cl_int err;
			context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
			clERR(err);

			queue = (clCreateCommandQueue(context, device, 0, &err));
			clERR(err);

			return 0;
		}
		void ReleaseDevice(cl_device_id& device, cl_context& context, cl_command_queue& queue)
		{
			clERR(clFlush(queue));
			clERR(clFinish(queue));
			clERR(clReleaseCommandQueue(queue));
			clERR(clReleaseContext(context));
		}
	}

#endif
}