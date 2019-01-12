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

#include "config.h"
#include "nn.h"
#include "image.h"

#include <algorithm>

#define MAXPOOL true


//================================================================================================================================================


namespace NeuralNetworksLib
{

	namespace Legacy
	{
		class ConvNeuralNetwork : public NeuralNetworks
		{
		public:
			enum typeInitWeight
			{
				specified = 0,
				random = 1,
				Nguyen_Vidrou = 2,
				linear_region_af = 3
			};
			enum typeConv
			{
				linear = 0,
				cyclic = 1
			};

			struct connectNeuron
			{
				int surf;
				int nx;
				int ny;

				connectNeuron()
				{
					surf = 0;
					nx = 0;
					ny = 0;
				}
			};
			struct labeledData
			{
				ArrayMatrix<float> data;
				Matrix<std::vector<float>> label;
			};

		public:
			class NetFunc
			{
			private:
				static inline void conv_init_weight(std::vector<double>& weight, Size2d& area_size);
				static inline void conv(double& result, std::vector<double>& weight, Matrix<double>& data, Size2d& area_size, const Point& offset);
				static inline double conv_diff_w(Matrix<double>& data, Size2d& area_size, Point& offset, std::vector<double>& weight, int index);
				static inline double conv_diff_x(std::vector<double>& weight, Matrix<double>& data, Size2d& area_size, Point& offset, Point& index);

				static inline void poly_init_weight(std::vector<double>& weight, Size2d& area_size);
				static inline void poly(double& result, std::vector<double>& weight, Matrix<double>& data, Size2d& area_size, const Point& offset);
				static inline double poly_diff_w(Matrix<double>& data, Size2d& area_size, Point& offset, std::vector<double>& weight, int index);
				static inline double poly_diff_x(std::vector<double>& weight, Matrix<double>& data, Size2d& area_size, Point& offset, Point& index);

			public:
				enum typeFuncNF
				{
					convolution = 0,
					polynomial  = 1,
					dispersion  = 2
				};

				static inline void net_f_init_weight(NetFunc::typeFuncNF ftype, std::vector<double>& weight, Size2d& area_size);
				static inline void net_f(double& result, NetFunc::typeFuncNF ftype, std::vector<double>& weight, Matrix<double>& data, Size2d& area_size, const Point& offset);
				static inline double net_f_diff_w(NetFunc::typeFuncNF ftype, Matrix<double>& data, Size2d& area_size, Point& offset, std::vector<double>& weight, int index);
				static inline double net_f_diff_x(NetFunc::typeFuncNF ftype, std::vector<double>& weight, Matrix<double>& data, Size2d& area_size, Point& offset, Point& index);

				static NetFunc::typeFuncNF StrToTFNF(std::string str);
				static std::string TFNFToStr(NetFunc::typeFuncNF type_nf);
			};

			struct CNN
			{
				//[layer][surf][neuron_y][neuron_x]
				Matrix<double>**      conv_neuron;
				std::vector<double>** conv_weight;
				double**              conv_bias;

				Matrix<double>**      subs_neuron;
				double**              subs_weight;
				double**              subs_bias;

				Matrix<std::vector<connectNeuron>>** conv_connect_neuron;
				Matrix<std::vector<connectNeuron>>** subs_connect_neuron;

				bool** drop_unit;
			};
			struct SNN
			{
				ArrayMatrix<double>*         hidden_weight;
				double*                      hidden_shift;
				Matrix<std::vector<double>>  hidden_neuron;

				Matrix<double>               output_weight;
				double*                      output_shift;
				Matrix<std::vector<double>>  output_neuron;
			};

			CNN cnn;
			SNN snn;

			float* output_neuron;
			int weight_count;
			int neuron_count;

			int num_threads = 0; //OpenMP only

			void Create();
			void CreateCombinMap(bool**& map, int layer_in, int layer_out, int count, int criterion_opt);

			void rotation(Matrix<double>& res, Matrix<double>& a, int p_x, int p_y);
			inline void convolution(Matrix<double>& data, int& m, int& n, int& row_offset, int& col_offset, Matrix<double>& conv, double& shift, double& result);
			inline void pooling(Matrix<double>& data, int& m, int& n, int& row_offset, int& col_offset, double& weight, double& shift, double& result);
			inline void Calculation(ArrayMatrix<double>& data);

		public:
			struct config_CNN
			{
				int* counter;

				Size min_image_size;
				Size max_image_size;

				//INPUT
				int data_surf;
				int* data_connect_map;

				//CNN
				int cnn_layer_count;
				int* surface_count;
				Size2d win_input_image;
				Size2d** win_conv;
				Size2d** win_subs;
				Point** shift_conv;
				NetFunc::typeFuncNF* cnn_net_func_type;
				typeConv* cnn_conv_type;
				ActiveFunc::typeFuncAF* cnn_conv_active_func_type;
				ActiveFunc::typeFuncAF* cnn_subs_active_func_type;
				int* type_connect;
				int* length_connect;
				bool*** conv_connect_map;

				//SNN
				int snn_hidden_count;
				int output_count;
				Size2d* win_snn_hidden_neuron;
				Point* shift_snn_hidden_neuron;
				ActiveFunc::typeFuncAF snn_layer_hidden_active_func_type;
				ActiveFunc::typeFuncAF snn_layer_output_active_func_type;
				bool snn_full_connect;
				std::vector<int>* snn_connect_map;

				//DATA
				int train_data_count;
				int test_data_count;
				labeledData* train_set;
				labeledData* test_set;

				//OTHER
				std::string log_path;
				bool draw_compute;
				bool fast_cnn;
				int thread_count;
				bool cnn_no_train;
				bool cnn_subs_layer;
				bool cnn_first_layer;
				bool dropout;

				double delta;

				config_CNN()
				{
					counter = NULL;

					data_connect_map = NULL;
					conv_connect_map = NULL;
					snn_connect_map = NULL;

					log_path = "";
					snn_full_connect = false;
					draw_compute = false;
					fast_cnn = false;
					thread_count = 0;
					cnn_no_train = false;
					cnn_subs_layer = true;
					cnn_first_layer = true;
					dropout = false;

					delta = 0.01;
				}
			};

			config_CNN cfg;

			ConvNeuralNetwork() { };
			ConvNeuralNetwork(config_CNN& _cfg);
			ConvNeuralNetwork(std::string path_file) { Init(path_file); }
			ConvNeuralNetwork(ConvNeuralNetwork* _cnn, int cols, int rows);
			~ConvNeuralNetwork() { Clear(); };

			int Init(std::string path_file);
			void AllocateMemory(Size size);

			NeuralNetworks* Copy();
			NeuralNetworks* CopyConfig();
			void CopyWeight(NeuralNetworks* copy);
			void Clear();

			bool isEmpty();

			Matrix<std::vector<double>> Calc_for_data(ArrayMatrix<double>& data);
			void Forward(SIMD::Image_32f& response_map, SIMD::Image_32f& image);
			void Calc_for_image_rgb(SIMD::Image_32f& response_map, SIMD::Image_32f& R, SIMD::Image_32f& G, SIMD::Image_32f& B);

			static ConvNeuralNetwork::typeConv StrToTConv(std::string str);
			static std::string TConvToStr(ConvNeuralNetwork::typeConv type_conv);

			Size getMinInputImgSize()	const  { return cfg.min_image_size; }
			Size getMaxInputImgSize()	const  { return cfg.max_image_size; }
			Size getInputImgSize()		const  { return Size(cfg.win_input_image.cols, cfg.win_input_image.rows); }
			Size getOutputImgSize()			   { return Size(snn.output_neuron.col_count(), snn.output_neuron.row_count()); }
			Size getOutputImgSize(const Size size);
			float getInputOutputRatio() const;

			int getNumThreads() const { return num_threads; }
			void setNumThreads(int _num_threads) { num_threads = MAX(1, _num_threads); }

			void setCNNRef(Legacy::ConvNeuralNetwork* _cnn) { }
		};
	}

}