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

#pragma once

#ifdef CNNOBJECTDETECTOR_EXPORTS
#	define COMPACTCNNLIB_API __declspec(dllexport) 
#else
#	define COMPACTCNNLIB_API __declspec(dllimport) 
#endif


//========================================================================================================


namespace CompactCNNLib
{

	class COMPACTCNNLIB_API FaceDetector
	{
	public:
		enum struct Pipeline
		{
			CPU = 0,
			GPU = 1,
			GPU_CPU = 2
		};
		enum struct DetectMode
		{
			disable = 0,
			sync = 1,
			async = 2
		};
		enum struct DetectPrecision
		{
			default = 0,
			low = 1,
			normal = 2,
			high = 3,
			ultra = 4
		};

		struct Size
		{
			int width = 0;
			int height = 0;

			Size() = default;
			Size(int _width, int _height)
				: width(_width), height(_height) { }
		};
		struct Face
		{
			int x = 0;
			int y = 0;
			int width = 0;
			int height = 0;
			float score = 0.f;

			int gender = -1;
			int smile = -1;
			int glasses = -1;

			Face() = default;
			Face(int _x, int _y, int _width, int _height, float _score = 0.f)
				: x(_x), y(_y), width(_width), height(_height), score(_score) { }
		};
		struct ImageData
		{
			int cols = 0;
			int rows = 0;
			int channels = 0;
			unsigned char* data = nullptr;
			size_t step = 0;

			ImageData() = default;
			ImageData(int _cols, int _rows, int _channels, unsigned char* _data, size_t _step)
				: cols(_cols), rows(_rows), channels(_channels), data(_data), step(_step) { }
		};

		struct Param
		{
			//Detection
			Size max_image_size = Size(1920, 1080);
			//Maximum size of input image. Used for initialize memory buffers.
			//In case of exceeding this value the detector will automatically reinitialized.

			int min_face_height = 80;		//Minimum possible face size.
			int max_face_height = 0;		//Maximum possible face size.
			float scale_factor = 1.15f;		//Scale factor for building image pyramid.
			int min_neighbors = 2;			//How many neighbors each candidate rectangle should have to reject it.

			//Precision
			DetectPrecision detect_precision = DetectPrecision::high;	//Specifies condition for combining responses of stages of the cascade.
			float treshold[3];											//Threshold value for each stage of the cascade. Range of values [-1.7159, 1.7159].
			bool drop_detect = true;									//Fast reject candidates in the first stage of the cascade.
			bool equalize = true;
			bool reflection = true;

			//Processing
			Pipeline pipeline = Pipeline::GPU;
			//This detector is optimized for Intel CPU and Nvidia GPU.
			//GPU_CPU mode run processing image pyramid on both devices simultaneously.

			DetectMode detect_mode = DetectMode::async;
			//This parameter set the mode for calculating the stages of cascade:
			//	disable	- calculation of only the first stage;
			//	sync	- stages of the cascade are computed sequentially in the same thread;
			//	async	- the second and third stages of the cascade are calculated in parallel with the first stage.
			//			  In the case of using the GPU the second and third stage will be calculated on the CPU asynchronously.
			//			  You can use multiple threads on the CPU for greater efficiency.

			int num_threads = 2;	//If you use asynchronous mode on the CPU then recommended 2 threads but in fact 4 threads will be used.

			//Models
			const char* models[3];	//Path to binary files of your CNN models.
			int index_output[3];	//The output number of the CNN to be calculated	
									//The detector supports operation only with CNN for binary classification.
									//However you can specify the output number of the CNN that should be calculated.

			//Device
			bool device_info = true;	//Displays information about the available CUDA or OpenCL devices.
			int cuda_device_id = 0;		//Index of the CUDA device to be used.
			int cl_platform_id = -1;	//Index of the OpenCL platform to be used.
			int cl_device_id = -1;		//Index of the OpenCL device to be used.

			//Experimental
			bool facial_analysis = false;

			Param()
			{
				//[-1.7159, 1.7159]
				treshold[0] = -1.5f;
				treshold[1] = -1.25f;
				treshold[2] = -1.0f;

				models[0] = nullptr;
				models[1] = nullptr;
				models[2] = nullptr;

				index_output[0] = 1;
				index_output[1] = 0;
				index_output[2] = 0;
			}
		};

		FaceDetector();
		~FaceDetector();

		int Init(Param& param);
		int Detect(Face* faces, ImageData& img);
		int Clear();

		bool isEmpty() const;

		Param getParam() const;
		void setParam(Param& param);

		Size getMaxImageSize() const;
		void setMaxImageSize(Size max_image_size);

		int getMinObjectHeight() const;
		void setMinObjectHeight(int min_obj_height);

		int getMaxObjectHeight() const;
		void setMaxObjectHeight(int max_obj_height);

		int getNumThreads() const;
		void setNumThreads(int num_threads);

		int getGrayImage32F(ImageData* img, Pipeline pipeline);

		static void CNTKDump2Binary(const char* binary_file, const char* cntk_model_dump);
	};

}