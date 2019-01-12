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


#include "cnn.h"
#include <fstream>
#include <iostream>

#ifdef USE_OMP
#	include <omp.h>
#endif

using namespace std;


//================================================================================================================================================


namespace NeuralNetworksLib
{

	namespace Legacy
	{
		void ConvNeuralNetwork::NetFunc::conv_init_weight(vector<double>& weight, Size2d& area_size)
		{
			weight = vector<double>(area_size.cols * area_size.rows);
		}
		void ConvNeuralNetwork::NetFunc::conv(double& result, vector<double>& weight, Matrix<double>& data, Size2d& area_size, const Point& offset)
		{
			result = 0.0;
			for (int j = 0; j < area_size.rows; ++j)
			{
				for (int i = 0; i < area_size.cols; ++i)
				{
					result += data[offset.y + j][offset.x + i] * weight[j * area_size.cols + i];
				}
			}
		}
		double ConvNeuralNetwork::NetFunc::conv_diff_w(Matrix<double>& data, Size2d& area_size, Point& offset, vector<double>& /*weight*/, int index)
		{
			int t = index % area_size.cols;
			return data[offset.y + (index - t) / area_size.cols][offset.x + t];
		}
		double ConvNeuralNetwork::NetFunc::conv_diff_x(vector<double>& weight, Matrix<double>& /*data*/, Size2d& area_size, Point& /*offset*/, Point& index)
		{
			return weight[index.y * area_size.cols + index.x];
		}

		void ConvNeuralNetwork::NetFunc::poly_init_weight(vector<double>& weight, Size2d& area_size)
		{
			weight = vector<double>(area_size.cols * area_size.rows + area_size.cols * (area_size.rows - 1));
		}
		void ConvNeuralNetwork::NetFunc::poly(double& result, vector<double>& weight, Matrix<double>& data, Size2d& area_size, const Point& offset)
		{
			result = 0.0;
			for (int j = 0; j < area_size.rows; ++j)
			{
				for (int i = 0; i < area_size.cols; ++i)
				{
					result += data[offset.y + j][offset.x + i] * weight[j * area_size.cols + i];
				}
			}

			int p = area_size.cols * area_size.rows;
			for (int j = 1; j < area_size.rows; ++j)
			{
				for (int i = 0; i < area_size.cols; ++i)
				{
					result += data[offset.y][offset.x + i] * data[offset.y + j][offset.x + i] * weight[p + (j - 1) * area_size.cols + i];
				}
			}
		}
		double ConvNeuralNetwork::NetFunc::poly_diff_w(Matrix<double>& data, Size2d& area_size, Point& offset, vector<double>& /*weight*/, int index)
		{
			int p = area_size.cols * area_size.rows;
			int t = index % area_size.cols;
			if (index < p)
			{
				return data[offset.y + (index - t) / area_size.cols][offset.x + t];
			}
			else
			{
				int t2 = (index - p) % area_size.cols;
				return data[offset.y][offset.x + t2] * data[offset.y + ((index - p) - t2) / area_size.cols + 1][offset.x + t2];
			}
		}
		double ConvNeuralNetwork::NetFunc::poly_diff_x(vector<double>& weight, Matrix<double>& data, Size2d& area_size, Point& offset, Point& index)
		{
			int p = area_size.cols * offset.y;
			if (index.y > 1)
			{
				return weight[index.y * area_size.cols + index.x] + weight[p + (index.y - 1) * area_size.cols + index.x] * data[offset.y][offset.x + index.x];
			}
			else
			{
				double d = weight[index.y * area_size.cols + index.x];
				for (int i = 0; i < index.y - 1; ++i)
				{
					d += weight[p + i * area_size.cols + index.x] * data[offset.y + i + 1][offset.x + index.x];
				}
				return d;
			}
		}

		void ConvNeuralNetwork::NetFunc::net_f_init_weight(NetFunc::typeFuncNF ftype, vector<double>& weight, Size2d& area_size)
		{
			if (ftype == ConvNeuralNetwork::NetFunc::typeFuncNF::convolution)	return conv_init_weight(weight, area_size);
			if (ftype == ConvNeuralNetwork::NetFunc::typeFuncNF::polynomial)	return poly_init_weight(weight, area_size);
		}
		void ConvNeuralNetwork::NetFunc::net_f(double& result, NetFunc::typeFuncNF ftype, vector<double>& weight, Matrix<double>& data, Size2d& area_size, const Point& offset)
		{
			if (ftype == ConvNeuralNetwork::NetFunc::typeFuncNF::convolution)	return conv(result, weight, data, area_size, offset);
			if (ftype == ConvNeuralNetwork::NetFunc::typeFuncNF::polynomial)	return poly(result, weight, data, area_size, offset);
		}
		double ConvNeuralNetwork::NetFunc::net_f_diff_w(NetFunc::typeFuncNF ftype, Matrix<double>& data, Size2d& area_size, Point& offset, vector<double>& weight, int index)
		{
			if (ftype == ConvNeuralNetwork::NetFunc::typeFuncNF::convolution)	return conv_diff_w(data, area_size, offset, weight, index);
			if (ftype == ConvNeuralNetwork::NetFunc::typeFuncNF::polynomial)	return poly_diff_w(data, area_size, offset, weight, index);
			return 0.0;
		}
		double ConvNeuralNetwork::NetFunc::net_f_diff_x(NetFunc::typeFuncNF ftype, vector<double>& weight, Matrix<double>& data, Size2d& area_size, Point& offset, Point& index)
		{
			if (ftype == ConvNeuralNetwork::NetFunc::typeFuncNF::convolution)	return conv_diff_x(weight, data, area_size, offset, index);
			if (ftype == ConvNeuralNetwork::NetFunc::typeFuncNF::polynomial)	return poly_diff_x(weight, data, area_size, offset, index);
			return 0.0;
		}

		ConvNeuralNetwork::NetFunc::typeFuncNF ConvNeuralNetwork::NetFunc::StrToTFNF(string str)
		{
			if (str == "convolution") return ConvNeuralNetwork::NetFunc::typeFuncNF::convolution;
			if (str == "polynomial")  return ConvNeuralNetwork::NetFunc::typeFuncNF::polynomial;
			if (str == "dispersion")  return ConvNeuralNetwork::NetFunc::typeFuncNF::dispersion;
			return ConvNeuralNetwork::NetFunc::typeFuncNF::convolution;
		}
		string ConvNeuralNetwork::NetFunc::TFNFToStr(ConvNeuralNetwork::NetFunc::typeFuncNF type_af)
		{
			if (type_af == ConvNeuralNetwork::NetFunc::typeFuncNF::convolution) return "convolution";
			if (type_af == ConvNeuralNetwork::NetFunc::typeFuncNF::polynomial)  return "polynomial";
			if (type_af == ConvNeuralNetwork::NetFunc::typeFuncNF::dispersion)  return "dispersion";
			return "convolution";
		}


		//===================================================================================================================


		ConvNeuralNetwork::ConvNeuralNetwork(config_CNN& _cfg)
		{
			cfg = _cfg;
			*cfg.counter += 1;
			Create();
		}
		int ConvNeuralNetwork::Init(string file_name)
		{
			fstream file_bin;
			file_bin.open(file_name.c_str(), fstream::binary | fstream::in);

			if (!file_bin.is_open())
			{
				printf("[Legacy::CNN] Configuration file not found!\n");
				return -1;
			}

			cfg.counter = new int(0);

			// read input data setting
			file_bin.read((char*)&(cfg.data_surf), sizeof(cfg.data_surf));
			file_bin.read((char*)&(cfg.win_input_image.rows), sizeof(cfg.win_input_image.rows));
			file_bin.read((char*)&(cfg.win_input_image.cols), sizeof(cfg.win_input_image.cols));

			cfg.min_image_size.width = cfg.win_input_image.cols;
			cfg.min_image_size.height = cfg.win_input_image.rows;
			cfg.max_image_size.width = cfg.win_input_image.cols;
			cfg.max_image_size.height = cfg.win_input_image.rows;

			// read cnn setting
			file_bin.read((char*)&(cfg.cnn_layer_count), sizeof(cfg.cnn_layer_count));

			cfg.surface_count = new int[cfg.cnn_layer_count];
			cfg.win_conv = new Size2d*[cfg.cnn_layer_count];
			cfg.shift_conv = new Point*[cfg.cnn_layer_count];
			cfg.win_subs = new Size2d*[cfg.cnn_layer_count];
			cfg.cnn_net_func_type = new ConvNeuralNetwork::NetFunc::typeFuncNF[cfg.cnn_layer_count];
			cfg.cnn_conv_type = new ConvNeuralNetwork::typeConv[cfg.cnn_layer_count];
			cfg.cnn_conv_active_func_type = new ActiveFunc::typeFuncAF[cfg.cnn_layer_count];
			cfg.cnn_subs_active_func_type = new ActiveFunc::typeFuncAF[cfg.cnn_layer_count];
			for (int i = 0; i < cfg.cnn_layer_count; ++i)
			{
				file_bin.read((char*)&(cfg.surface_count[i]), sizeof(cfg.surface_count[i]));

				cfg.win_conv[i] = new Size2d[cfg.surface_count[i]];
				cfg.win_subs[i] = new Size2d[cfg.surface_count[i]];
				cfg.shift_conv[i] = new Point[cfg.surface_count[i]];

				file_bin.read((char*)&(cfg.win_conv[i][0].rows), sizeof(cfg.win_conv[i][0].rows));
				file_bin.read((char*)&(cfg.win_conv[i][0].cols), sizeof(cfg.win_conv[i][0].cols));
				file_bin.read((char*)&(cfg.shift_conv[i][0].y), sizeof(cfg.shift_conv[i][0].y));
				file_bin.read((char*)&(cfg.shift_conv[i][0].x), sizeof(cfg.shift_conv[i][0].x));
				file_bin.read((char*)&(cfg.win_subs[i][0].rows), sizeof(cfg.win_subs[i][0].rows));
				file_bin.read((char*)&(cfg.win_subs[i][0].cols), sizeof(cfg.win_subs[i][0].cols));

				for (int j = 1; j < cfg.surface_count[i]; ++j)
				{
					cfg.win_conv[i][j].rows = cfg.win_conv[i][0].rows;
					cfg.win_conv[i][j].cols = cfg.win_conv[i][0].cols;
					cfg.shift_conv[i][j].y = cfg.shift_conv[i][0].y;
					cfg.shift_conv[i][j].x = cfg.shift_conv[i][0].x;
					cfg.win_subs[i][j].rows = cfg.win_subs[i][0].rows;
					cfg.win_subs[i][j].cols = cfg.win_subs[i][0].cols;
				}

				file_bin.read((char*)&(cfg.cnn_net_func_type[i]), sizeof(cfg.cnn_net_func_type[i]));
				file_bin.read((char*)&(cfg.cnn_conv_type[i]), sizeof(cfg.cnn_conv_type[i]));
				file_bin.read((char*)&(cfg.cnn_conv_active_func_type[i]), sizeof(cfg.cnn_conv_active_func_type[i]));
				file_bin.read((char*)&(cfg.cnn_subs_active_func_type[i]), sizeof(cfg.cnn_subs_active_func_type[i]));
			}

			// read options connections map
			cfg.type_connect = new int[cfg.cnn_layer_count - 1];
			for (int i = 0; i < cfg.cnn_layer_count - 1; ++i)
			{
				file_bin.read((char*)&(cfg.type_connect[i]), sizeof(cfg.type_connect[i]));
			}

			cfg.length_connect = new int[cfg.cnn_layer_count - 1];
			for (int i = 0; i < cfg.cnn_layer_count - 1; ++i)
			{
				file_bin.read((char*)&(cfg.length_connect[i]), sizeof(cfg.length_connect[i]));
			}

			// read snn setting
			for (int i = 0; i < 2; ++i)
			{
				if (i == 0)
				{
					file_bin.read((char*)&(cfg.snn_hidden_count), sizeof(cfg.snn_hidden_count));

					cfg.win_snn_hidden_neuron = new Size2d[cfg.snn_hidden_count];
					cfg.shift_snn_hidden_neuron = new Point[cfg.snn_hidden_count];

					file_bin.read((char*)&(cfg.win_snn_hidden_neuron[0].rows), sizeof(cfg.win_snn_hidden_neuron[0].rows));
					file_bin.read((char*)&(cfg.win_snn_hidden_neuron[0].cols), sizeof(cfg.win_snn_hidden_neuron[0].cols));
					file_bin.read((char*)&(cfg.shift_snn_hidden_neuron[0].y), sizeof(cfg.shift_snn_hidden_neuron[0].y));
					file_bin.read((char*)&(cfg.shift_snn_hidden_neuron[0].x), sizeof(cfg.shift_snn_hidden_neuron[0].x));

					for (int j = 0; j < cfg.snn_hidden_count; ++j)
					{
						cfg.win_snn_hidden_neuron[j].rows = cfg.win_snn_hidden_neuron[0].rows;
						cfg.win_snn_hidden_neuron[j].cols = cfg.win_snn_hidden_neuron[0].cols;
						cfg.shift_snn_hidden_neuron[j].y = cfg.shift_snn_hidden_neuron[0].y;
						cfg.shift_snn_hidden_neuron[j].x = cfg.shift_snn_hidden_neuron[0].x;
					}

					file_bin.read((char*)&(cfg.snn_layer_hidden_active_func_type), sizeof(cfg.snn_layer_hidden_active_func_type));
					file_bin.read((char*)&(cfg.snn_full_connect), sizeof(cfg.snn_full_connect));
				}
				if (i == 1)
				{
					file_bin.read((char*)&(cfg.output_count), sizeof(cfg.output_count));
					file_bin.read((char*)&(cfg.snn_layer_output_active_func_type), sizeof(cfg.snn_layer_output_active_func_type));
				}
			}


			Create();


			// read weight
			int w_count = 0;
			file_bin.read((char*)&(w_count), sizeof(w_count));
			if (weight_count != w_count)
			{
				printf("[Legacy::CNN] Weight_count = %d\n", weight_count);
				printf("[Legacy::CNN] Weight_count_read = %d\n", w_count);
				cout << "[Legacy::CNN] Error non-conformity of weight";
				return -1;
			}

			for (int layer = 0; layer < cfg.cnn_layer_count; ++layer)
			{
				for (int surf = 0; surf < cfg.surface_count[layer]; ++surf)
				{
					NetFunc::net_f_init_weight(cfg.cnn_net_func_type[layer], cnn.conv_weight[layer][surf], cfg.win_conv[layer][surf]);

					for (int t = 0; t < (int)cnn.conv_weight[layer][surf].size(); ++t)
					{
						file_bin.read((char*)&(cnn.conv_weight[layer][surf][t]), sizeof(cnn.conv_weight[layer][surf][t]));
					}
					file_bin.read((char*)&(cnn.subs_weight[layer][surf]), sizeof(cnn.subs_weight[layer][surf]));
				}
			}

			for (int layer = 0; layer < cfg.cnn_layer_count; ++layer)
			{
				for (int surf = 0; surf < cfg.surface_count[layer]; ++surf)
				{
					file_bin.read((char*)&(cnn.conv_bias[layer][surf]), sizeof(cnn.conv_bias[layer][surf]));
					file_bin.read((char*)&(cnn.subs_bias[layer][surf]), sizeof(cnn.subs_bias[layer][surf]));
				}
			}

			for (int layer = 0; layer < cfg.snn_hidden_count; ++layer)
			{
				for (int k = 0; k < (int)cfg.snn_connect_map[layer].size(); ++k)
				{
					for (int y = 0; y < snn.hidden_weight[layer][k].row_count(); ++y)
						for (int x = 0; x < snn.hidden_weight[layer][k].col_count(); ++x)
						{
							file_bin.read((char*)&(snn.hidden_weight[layer][k][y][x]), sizeof(snn.hidden_weight[layer][k][y][x]));
						}
				}
				file_bin.read((char*)&(snn.hidden_shift[layer]), sizeof(snn.hidden_shift[layer]));
			}

			for (int layer = 0; layer < cfg.output_count; ++layer)
			{
				file_bin.read((char*)&(snn.output_shift[layer]), sizeof(snn.output_shift[layer]));
			}

			for (int y = 0; y < snn.output_weight.row_count(); ++y)
			{
				for (int x = 0; x < snn.output_weight.col_count(); ++x)
				{
					file_bin.read((char*)&(snn.output_weight[y][x]), sizeof(snn.output_weight[y][x]));
				}
			}

			file_bin.close();

			return 0;
		}
		ConvNeuralNetwork::ConvNeuralNetwork(ConvNeuralNetwork* _cnn, int cols, int rows)
		{
			cfg = _cnn->cfg;
			*cfg.counter += 1;

			if (cols == 0 || rows == 0) return;

			cfg.win_input_image.cols = cols;
			cfg.win_input_image.rows = rows;

			for (int surf = 0; surf < cfg.snn_hidden_count; ++surf)
			{
				cfg.shift_snn_hidden_neuron[surf].x = 1;
				cfg.shift_snn_hidden_neuron[surf].y = 1;
			}

			for (int surf = 0; surf < cfg.surface_count[cfg.cnn_layer_count - 1]; ++surf)
			{
				if (cfg.shift_conv[cfg.cnn_layer_count - 1][surf].x == 0)
					cfg.shift_conv[cfg.cnn_layer_count - 1][surf].x = 1;
				if (cfg.shift_conv[cfg.cnn_layer_count - 1][surf].y == 0)
					cfg.shift_conv[cfg.cnn_layer_count - 1][surf].y = 1;
			}

			Create();

			if (this->weight_count != _cnn->weight_count)
			{
				system("Error init");
			}

			_cnn->CopyWeight(this);
		}
		void ConvNeuralNetwork::AllocateMemory(Size size)
		{
			return;
		}

		void ConvNeuralNetwork::Create()
		{
			weight_count = 0;
			neuron_count = 0;

			//-------------------------------------------------------------------------------------------------------------------------
			// initialization of map connections
			bool **combin_map_byff;

			if (cfg.data_connect_map == NULL)
			{
				cfg.data_connect_map = new int[cfg.surface_count[0]];
				if (cfg.surface_count[0] % cfg.data_surf == 0)
				{
					int k = -1;
					for (int i = 0; i < cfg.surface_count[0]; ++i)
					{
						if (i % (cfg.surface_count[0] / cfg.data_surf) == 0) k++;
						if (k >= cfg.data_surf) k = cfg.data_surf - 1;
						cfg.data_connect_map[i] = k;
					}
				}
				else
				{
					CreateCombinMap(combin_map_byff, cfg.data_surf, cfg.surface_count[0], 1, 0);
					for (int i = 0; i < cfg.surface_count[0]; ++i)
					{
						for (int j = 0; j < cfg.data_surf; ++j)
							if (combin_map_byff[i][j] == 1)
								cfg.data_connect_map[i] = j;
					}
					for (int i = 0; i < cfg.surface_count[0]; ++i)
					{
						delete[] combin_map_byff[i];
					}
					delete[] combin_map_byff;
				}

				cfg.conv_connect_map = new bool**[cfg.cnn_layer_count - 1];
				for (int i = 0; i < cfg.cnn_layer_count - 1; ++i)
				{
					if (cfg.type_connect[i] <= 0)
					{
						cfg.conv_connect_map[i] = new bool*[cfg.surface_count[i + 1]];
						for (int j = 0; j < cfg.surface_count[i + 1]; ++j)
						{
							cfg.conv_connect_map[i][j] = new bool[cfg.surface_count[i]]();

							if (cfg.type_connect[i] == -1)
							{
								for (int p = 0; p < cfg.surface_count[i]; ++p)
									cfg.conv_connect_map[i][j][p] = 1;
							}
							else
							{
								int t = j / 2;
								int k = 0;
								for (int p = t - (cfg.length_connect[i] / 2); p <= t + (cfg.length_connect[i] / 2); ++p)
									if ((p >= 0) && (p < cfg.surface_count[i]))
									{
										k++;
										if (k > cfg.length_connect[i]) break;
										cfg.conv_connect_map[i][j][p] = 1;
									}
							}
						}
					}
					else
					{
						CreateCombinMap(cfg.conv_connect_map[i], cfg.surface_count[i], cfg.surface_count[i + 1], cfg.length_connect[i], cfg.type_connect[i]);
					}
				}

				cfg.snn_connect_map = new vector<int>[cfg.snn_hidden_count];
				if (!cfg.snn_full_connect)
				{
					if ((cfg.snn_hidden_count / cfg.surface_count[cfg.cnn_layer_count - 1]) % 2 == 0)
					{
						int k = -1;
						for (int i = 0; i < cfg.snn_hidden_count; ++i)
						{
							if (i % (cfg.snn_hidden_count / cfg.surface_count[cfg.cnn_layer_count - 1]) == 0) k++;
							if (k >= cfg.surface_count[cfg.cnn_layer_count - 1]) k = cfg.surface_count[cfg.cnn_layer_count - 1] - 1;
							cfg.snn_connect_map[i].push_back(k);
						}
					}
					else
					{
						if (cfg.snn_hidden_count == cfg.surface_count[cfg.cnn_layer_count - 1])
						{
							for (int i = 0; i < cfg.snn_hidden_count; ++i)
								cfg.snn_connect_map[i].push_back(i);
						}
						else
						{
							CreateCombinMap(combin_map_byff, cfg.surface_count[cfg.cnn_layer_count - 1], cfg.snn_hidden_count, 1, 0);
							for (int i = 0; i < cfg.snn_hidden_count; ++i)
							{
								for (int j = 0; j < cfg.surface_count[cfg.cnn_layer_count - 1]; ++j)
									if (combin_map_byff[i][j] == 1)
										cfg.snn_connect_map[i].push_back(j);
							}
							for (int i = 0; i < cfg.snn_hidden_count; ++i)
							{
								delete[] combin_map_byff[i];
							}
							delete[] combin_map_byff;
						}
					}
				}
				else
				{
					for (int i = 0; i < cfg.snn_hidden_count; ++i)
					{
						for (int k = 0; k < cfg.surface_count[cfg.cnn_layer_count - 1]; ++k)
						{
							cfg.snn_connect_map[i].push_back(k);
						}
					}
				}
			}

			//-------------------------------------------------------------------------------------------------------------------------
			// creare CNN

			// calculation of neural surfaces size
			Size2d **conv_surf_size;
			Size2d **subs_surf_size;

			conv_surf_size = new Size2d*[cfg.cnn_layer_count];
			subs_surf_size = new Size2d*[cfg.cnn_layer_count];
			for (int i = 0; i < cfg.cnn_layer_count; ++i)
			{
				conv_surf_size[i] = new Size2d[cfg.surface_count[i]];
				subs_surf_size[i] = new Size2d[cfg.surface_count[i]];
				for (int j = 0; j < cfg.surface_count[i]; ++j)
				{
					if (i == 0)
					{
						if (cfg.shift_conv[i][j].y == 0)
						{
							conv_surf_size[i][j].rows = cfg.win_input_image.rows / cfg.win_conv[i][j].rows;
						}
						else
						{
							if (cfg.cnn_conv_type[i] == typeConv::cyclic)
							{
								const float p = float(cfg.win_input_image.rows) / float(cfg.win_conv[i][j].rows);
								conv_surf_size[i][j].rows = int(std::nearbyint(p)) * cfg.win_conv[i][j].rows;
							}
							if (cfg.cnn_conv_type[i] == typeConv::linear)
							{
								conv_surf_size[i][j].rows = (cfg.win_input_image.rows - cfg.win_conv[i][j].rows) / cfg.shift_conv[i][j].y + 1;
							}
						}

						if (cfg.shift_conv[i][j].x == 0)
						{
							conv_surf_size[i][j].cols = cfg.win_input_image.cols / cfg.win_conv[i][j].cols;
						}
						else
						{
							if (cfg.cnn_conv_type[i] == typeConv::cyclic)
							{
								const float p = float(cfg.win_input_image.cols) / float(cfg.win_conv[i][j].cols);
								conv_surf_size[i][j].cols = int(std::nearbyint(p)) * cfg.win_conv[i][j].cols;
							}
							if (cfg.cnn_conv_type[i] == typeConv::linear)
							{
								conv_surf_size[i][j].cols = (cfg.win_input_image.cols - cfg.win_conv[i][j].cols) / cfg.shift_conv[i][j].x + 1;
							}
						}
					}
					else
					{
						int t = 0;
						for (int k = 0; k < cfg.surface_count[i - 1]; ++k)
						{
							if (cfg.conv_connect_map[i - 1][j][k] == 1)
							{
								t = k;
								continue;
							}
						}

						if (cfg.win_conv[i][j].rows == 0)
							cfg.win_conv[i][j].rows = subs_surf_size[i - 1][t].rows;

						if (cfg.shift_conv[i][j].y == 0)
						{
							conv_surf_size[i][j].rows = subs_surf_size[i - 1][t].rows / cfg.win_conv[i][j].rows;
						}
						else
						{
							if (cfg.cnn_conv_type[i] == typeConv::cyclic)
							{
								const float p = float(subs_surf_size[i - 1][t].rows) / float(cfg.win_conv[i][j].rows);
								conv_surf_size[i][j].rows = int(std::nearbyint(p)) * cfg.win_conv[i][j].rows;
							}
							if (cfg.cnn_conv_type[i] == typeConv::linear)
							{
								conv_surf_size[i][j].rows = (subs_surf_size[i - 1][t].rows - cfg.win_conv[i][j].rows) / cfg.shift_conv[i][j].y + 1;
							}
						}

						if (cfg.win_conv[i][j].cols == 0)
							cfg.win_conv[i][j].cols = subs_surf_size[i - 1][t].cols;

						if (cfg.shift_conv[i][j].x == 0)
						{
							conv_surf_size[i][j].cols = subs_surf_size[i - 1][t].cols / cfg.win_conv[i][j].cols;
						}
						else
						{
							if (cfg.cnn_conv_type[i] == typeConv::cyclic)
							{
								const float p = float(subs_surf_size[i - 1][t].cols) / float(cfg.win_conv[i][j].cols);
								conv_surf_size[i][j].cols = int(std::nearbyint(p)) * cfg.win_conv[i][j].cols;
							}
							if (cfg.cnn_conv_type[i] == typeConv::linear)
							{
								conv_surf_size[i][j].cols = (subs_surf_size[i - 1][t].cols - cfg.win_conv[i][j].cols) / cfg.shift_conv[i][j].x + 1;
							}
						}
					}
					subs_surf_size[i][j].rows = conv_surf_size[i][j].rows / cfg.win_subs[i][j].rows;
					subs_surf_size[i][j].cols = conv_surf_size[i][j].cols / cfg.win_subs[i][j].cols;

					if (subs_surf_size[i][j].rows * subs_surf_size[i][j].cols < 1)
					{
						printf("[Legacy::CNN] Error: invalid configuration neural surface\n");
						system("pause");
					}
				}
			}

			// initialization of neural surfaces
			cnn.conv_neuron = new Matrix<double>*[cfg.cnn_layer_count];
			cnn.subs_neuron = new Matrix<double>*[cfg.cnn_layer_count];
			cnn.drop_unit = new bool*[cfg.cnn_layer_count];
			for (int i = 0; i < cfg.cnn_layer_count; ++i)
			{
				cnn.conv_neuron[i] = new Matrix<double>[cfg.surface_count[i]]();
				cnn.subs_neuron[i] = new Matrix<double>[cfg.surface_count[i]]();
				cnn.drop_unit[i] = new bool[cfg.surface_count[i]];

				for (int j = 0; j < cfg.surface_count[i]; ++j)
				{
					cnn.conv_neuron[i][j] = Matrix<double>(conv_surf_size[i][j].rows, conv_surf_size[i][j].cols);
					cnn.subs_neuron[i][j] = Matrix<double>(subs_surf_size[i][j].rows, subs_surf_size[i][j].cols);

					neuron_count += conv_surf_size[i][j].rows * conv_surf_size[i][j].cols;
					neuron_count += subs_surf_size[i][j].rows * subs_surf_size[i][j].cols;
				}
			}

			// initialization of weight neurons
			cnn.conv_weight = new vector<double>*[cfg.cnn_layer_count];
			cnn.subs_weight = new double*[cfg.cnn_layer_count];
			for (int i = 0; i < cfg.cnn_layer_count; ++i)
			{
				cnn.conv_weight[i] = new vector<double>[cfg.surface_count[i]]();
				cnn.subs_weight[i] = new double[cfg.surface_count[i]];

				for (int j = 0; j < cfg.surface_count[i]; ++j)
				{
					//cnn.conv_weight[i][j] = Matrix<double>(cfg.win_conv[i][j].rows, cfg.win_conv[i][j].cols, cfg.win_conv[i][j].rows * cfg.win_conv[i][j].cols);
					NetFunc::net_f_init_weight(cfg.cnn_net_func_type[i], cnn.conv_weight[i][j], cfg.win_conv[i][j]);
					cnn.subs_weight[i][j] = 0.0;

					int w_c = (int)cnn.conv_weight[i][j].size();//cfg.win_conv[i][j].rows * cfg.win_conv[i][j].cols;
					weight_count += w_c + 1;
				}
			}

			// initialization of neural offset
			cnn.conv_bias = new double*[cfg.cnn_layer_count];
			cnn.subs_bias = new double*[cfg.cnn_layer_count];
			for (int i = 0; i < cfg.cnn_layer_count; ++i)
			{
				cnn.conv_bias[i] = new double[cfg.surface_count[i]];
				cnn.subs_bias[i] = new double[cfg.surface_count[i]];

				for (int j = 0; j < cfg.surface_count[i]; ++j)
				{
					cnn.conv_bias[i][j] = 0.0;
					cnn.subs_bias[i][j] = 0.0;
					weight_count += 2;
				}
			}

			//-------------------------------------------------------------------------------------------------------------------------
			// creare SNN

			// calculation of surfaces size layer hidden neurons
			Size2d lhn_size;			//layer_hidden_neuron_size
			Size2d check_lhn_size;		//check_hidden_neuron_size
			for (int neuron = 0; neuron < cfg.snn_hidden_count; ++neuron)
			{
				for (int k = 0; k < (int)cfg.snn_connect_map[neuron].size(); ++k)
				{
					if ((subs_surf_size[cfg.cnn_layer_count - 1][cfg.snn_connect_map[neuron][k]].rows < cfg.win_snn_hidden_neuron[neuron].rows) || subs_surf_size[cfg.cnn_layer_count - 1][cfg.snn_connect_map[neuron][k]].cols < cfg.win_snn_hidden_neuron[neuron].cols)
					{
						printf("[Legacy::CNN] Error: invalid configuration neural surface\n");
						system("pause");
					}

					if ((check_lhn_size.rows != lhn_size.rows) || (check_lhn_size.cols != lhn_size.cols))
					{
						printf("[Legacy::CNN] Error: invalid configuration neural surface\n");
						system("pause");
					}

					if (cfg.shift_snn_hidden_neuron[neuron].y == 0)
					{
						if (cfg.win_snn_hidden_neuron[neuron].rows != 0)
						{
							lhn_size.rows = subs_surf_size[cfg.cnn_layer_count - 1][cfg.snn_connect_map[neuron][k]].rows / cfg.win_snn_hidden_neuron[neuron].rows;
						}
						else
						{
							cfg.win_snn_hidden_neuron[neuron].rows = subs_surf_size[cfg.cnn_layer_count - 1][cfg.snn_connect_map[neuron][k]].rows;
							lhn_size.rows = 1;
						}
					}
					else
					{
						lhn_size.rows = (subs_surf_size[cfg.cnn_layer_count - 1][cfg.snn_connect_map[neuron][k]].rows - cfg.win_snn_hidden_neuron[neuron].rows) / cfg.shift_snn_hidden_neuron[neuron].y + 1;
					}

					if (cfg.shift_snn_hidden_neuron[neuron].x == 0)
					{
						if (cfg.win_snn_hidden_neuron[neuron].cols != 0)
						{
							lhn_size.cols = subs_surf_size[cfg.cnn_layer_count - 1][cfg.snn_connect_map[neuron][k]].cols / cfg.win_snn_hidden_neuron[neuron].cols;
						}
						else
						{
							cfg.win_snn_hidden_neuron[neuron].cols = subs_surf_size[cfg.cnn_layer_count - 1][cfg.snn_connect_map[neuron][k]].cols;
							lhn_size.cols = 1;
						}
					}
					else
					{
						lhn_size.cols = (subs_surf_size[cfg.cnn_layer_count - 1][cfg.snn_connect_map[neuron][k]].cols - cfg.win_snn_hidden_neuron[neuron].cols) / cfg.shift_snn_hidden_neuron[neuron].x + 1;
					}

					check_lhn_size = lhn_size;
				}
			}

			// initialization of surfaces layer hidden neurons 
			snn.hidden_neuron = Matrix<vector<double>>(lhn_size.rows, lhn_size.cols);
			for (int y = 0; y < lhn_size.rows; ++y)
			{
				for (int x = 0; x < lhn_size.cols; ++x)
				{
					snn.hidden_neuron[y][x] = vector<double>(cfg.snn_hidden_count);
				}
				neuron_count += cfg.snn_hidden_count * lhn_size.rows * lhn_size.cols;
			}

			// initialization of weight neurons and neural offset
			snn.hidden_weight = new ArrayMatrix<double>[cfg.snn_hidden_count]();
			snn.hidden_shift = new double[cfg.snn_hidden_count];
			for (int i = 0; i < cfg.snn_hidden_count; ++i)
			{
				snn.hidden_weight[i] = ArrayMatrix<double>(cfg.win_snn_hidden_neuron[i].rows, cfg.win_snn_hidden_neuron[i].cols, (int)cfg.snn_connect_map[i].size());
				snn.hidden_shift[i] = 0.0;
				weight_count += (int)cfg.snn_connect_map[i].size() * cfg.win_snn_hidden_neuron[i].rows * cfg.win_snn_hidden_neuron[i].cols + 1;
			}

			// initialization of surfaces layer output neurons 
			Size2d lon_size = lhn_size;		//layer_output_neuron_size
			snn.output_neuron = Matrix<vector<double>>(lon_size.rows, lon_size.cols);
			for (int y = 0; y < lon_size.rows; ++y)
			{
				for (int x = 0; x < lon_size.cols; ++x)
				{
					snn.output_neuron[y][x] = vector<double>(cfg.output_count);
				}
				neuron_count += cfg.output_count * lon_size.rows * lon_size.cols;
			}

			// initialization response_map (only cfg.output_count = 1!)
			output_neuron = new float[lon_size.rows * lon_size.cols];

			// initialization of weight neurons and neural offset
			snn.output_weight = Matrix<double>(cfg.output_count, cfg.snn_hidden_count);
			snn.output_shift = new double[cfg.output_count];
			weight_count += cfg.output_count * cfg.snn_hidden_count;
			for (int i = 0; i < cfg.output_count; ++i)
			{
				snn.output_shift[i] = 0.0;
				weight_count++;
			}

			// 	memory release
			for (int i = 0; i < cfg.cnn_layer_count; ++i)
			{
				delete[] conv_surf_size[i];
				delete[] subs_surf_size[i];
			}
			delete[] conv_surf_size;
			delete[] subs_surf_size;

			// set num threads
			num_threads = 1;
#ifdef USE_OMP
			num_threads = omp_get_num_procs();
#endif
		}
		void ConvNeuralNetwork::CreateCombinMap(bool**& map, int layer_in, int layer_out, int count, int criterion_opt)
		{
			enum typeOpt
			{
				disp_count = 1,
				disp_offset = 2,
				disp_count_and_offset = 3
			};
			typeOpt type_opt = (typeOpt)criterion_opt;

			map = new bool*[layer_out];
			for (int j = 0; j < layer_out; ++j)
			{
				map[j] = new bool[layer_in]();
			}

			if (count * layer_out < layer_in || count > layer_in)
			{
				count = layer_in;
			}

			int bins_offset = layer_out;
			int bins_sum = layer_in;
			int* min_val = new int[bins_offset]();
			int* max_val = new int[bins_offset]();
			int* offset = new int[bins_offset]();
			int* offset_buff = new int[bins_offset]();
			int* sum = new int[bins_sum]();
			int p = 0;
			float min_disp = FLT_MAX;
			for (int t = 1; t < bins_offset; ++t)
			{
				max_val[t] = count;
				if ((bins_offset > 16) && (count > 3))
					max_val[t] = 3;
				if ((bins_offset > 32) && (count > 2))
					max_val[t] = 2;
				if ((bins_offset > 64) && (count > 1))
					max_val[t] = 1;
			}

			int iter = 0;
			do
			{
				for (int k = 0; k <= p; ++k)
				{
					offset[k]++;
					if ((offset[k] > max_val[k]) && (offset[k + 1] > max_val[k + 1]))
						continue;
					else
						if (offset[k] <= max_val[k]) break;

					if (offset[k] > max_val[k])
					{
						p = k + 1;
						for (int t = 0; t <= k; ++t)
							offset[k] = min_val[k];
						if (p >= bins_offset) break;
						offset[p]++;
						if (offset[p] <= max_val[p]) break;
					}
				}

				for (int t = 0; t < bins_sum; ++t)
				{
					sum[t] = 0;
				}

				bool bl = false;
				for (int t = 0; t < bins_offset; ++t)
				{
					int a = 0;
					for (int l = 0; l <= t; ++l)
						a += offset[l];
					int b = a + count;
					for (int r = a; r < b; ++r)
						if (r < bins_sum)
							sum[r]++;
						else
							bl = true;
				}

				for (int r = 0; r < bins_sum; ++r)
				{
					if (sum[r] == 0) bl = true;
				}

				if (bl) continue;

				float avr = 0;
				for (int t = 0; t < bins_sum; ++t)
				{
					avr += sum[t];
				}
				avr /= bins_sum;

				float disp = 0;
				if (type_opt == disp_count || type_opt == disp_count_and_offset)
				{
					float sqr = 0;
					for (int t = 0; t < bins_sum; ++t)
					{
						sqr += (sum[t] - avr) * (sum[t] - avr);
					}
					disp = sqr / bins_sum;
				}
				if (type_opt == disp_offset || type_opt == disp_count_and_offset)
				{
					float sqr = 0;
					for (int t = 0; t < bins_offset; ++t)
					{
						sqr += (offset[t] - avr) * (offset[t] - avr);
					}
					if (type_opt == disp_count_and_offset)
						disp += sqr / bins_offset;
					else
						disp = sqr / bins_offset;
				}

				if (disp < min_disp)
				{
					for (int t = 0; t < bins_offset; ++t)
					{
						offset_buff[t] = offset[t];
					}
					min_disp = disp;
					if (disp == 0) break;
				}

				iter++;
				if (iter > 100) break;
			} while (p < bins_offset);

			for (int j = 0; j < layer_out; ++j)
			{
				int a = 0;
				for (int l = 0; l <= j; ++l)
				{
					a += offset_buff[l];
				}
				int b = a + count;
				for (int p = a; p < b; ++p)
				{
					map[j][p] = 1;
				}
			}

			delete[] min_val;
			delete[] max_val;
			delete[] offset;
			delete[] offset_buff;
			delete[] sum;
		}
		void ConvNeuralNetwork::Clear()
		{
			if (cfg.counter == NULL) return;

			// delete of neural surfaces
			for (int i = 0; i < cfg.cnn_layer_count; ++i)
			{
				for (int j = 0; j < cfg.surface_count[i]; ++j)
				{
					cnn.conv_neuron[i][j].clear();
					cnn.subs_neuron[i][j].clear();
				}
			}

			for (int i = 0; i < cfg.cnn_layer_count; ++i)
			{
				delete[] cnn.conv_neuron[i];
				delete[] cnn.subs_neuron[i];
				delete[] cnn.drop_unit[i];
			}
			delete[] cnn.conv_neuron;
			delete[] cnn.subs_neuron;
			delete[] cnn.drop_unit;
			cnn.conv_neuron = NULL;
			cnn.subs_neuron = NULL;
			cnn.drop_unit = NULL;

			// delete of weight neurons
			for (int i = 0; i < cfg.cnn_layer_count; ++i)
			{
				for (int j = 0; j < cfg.surface_count[i]; ++j)
				{
					cnn.conv_weight[i][j].clear();
				}
			}

			for (int i = 0; i < cfg.cnn_layer_count; ++i)
			{
				delete[] cnn.conv_weight[i];
				delete[] cnn.subs_weight[i];
			}
			delete[] cnn.conv_weight;
			delete[] cnn.subs_weight;
			cnn.conv_weight = NULL;
			cnn.subs_weight = NULL;

			// delete of neural offset
			for (int i = 0; i < cfg.cnn_layer_count; ++i)
			{
				delete[] cnn.conv_bias[i];
				delete[] cnn.subs_bias[i];
			}
			delete[] cnn.conv_bias;
			delete[] cnn.subs_bias;
			cnn.conv_bias = NULL;
			cnn.subs_bias = NULL;

			//-------------------------------------------------------------------------------------------------------------------------
			// delete SNN

			// delete of surfaces layer hidden neurons 
			for (int y = 0; y < snn.hidden_neuron.row_count(); ++y)
			{
				for (int x = 0; x < snn.hidden_neuron.col_count(); ++x)
				{
					snn.hidden_neuron[y][x].clear();
				}
			}
			snn.hidden_neuron.clear();

			// delete of weight neurons and neural offset
			for (int i = 0; i < cfg.snn_hidden_count; ++i)
			{
				snn.hidden_weight[i].clear();
			}

			delete[] snn.hidden_weight;
			delete[] snn.hidden_shift;
			snn.hidden_weight = NULL;
			snn.hidden_shift = NULL;

			// delete of surfaces layer output neurons 
			for (int y = 0; y < snn.output_neuron.row_count(); ++y)
			{
				for (int x = 0; x < snn.output_neuron.col_count(); ++x)
				{
					snn.output_neuron[y][x].clear();
				}
			}
			snn.output_neuron.clear();

			delete[] output_neuron;

			// delete of weight neurons and neural offset
			snn.output_weight.clear();
			delete[] snn.output_shift;
			snn.output_shift = NULL;

			//clear config
			if (*cfg.counter == 0)
			{
				delete[] cfg.data_connect_map;

				for (int i = 0; i < cfg.cnn_layer_count - 1; ++i)
				{
					for (int j = 0; j < cfg.surface_count[i + 1]; ++j)
					{
						delete[] cfg.conv_connect_map[i][j];
					}
					delete[] cfg.conv_connect_map[i];
				}
				delete[] cfg.conv_connect_map;

				delete[] cfg.snn_connect_map;

				for (int i = 0; i < cfg.cnn_layer_count; ++i)
				{
					delete[] cfg.win_conv[i];
					delete[] cfg.win_subs[i];
					delete[] cfg.shift_conv[i];
				}

				delete[] cfg.surface_count;
				delete[] cfg.win_conv;
				delete[] cfg.shift_conv;
				delete[] cfg.win_subs;
				delete[] cfg.cnn_net_func_type;
				delete[] cfg.cnn_conv_type;
				delete[] cfg.cnn_conv_active_func_type;
				delete[] cfg.cnn_subs_active_func_type;

				delete[] cfg.type_connect;
				delete[] cfg.length_connect;
				delete[] cfg.win_snn_hidden_neuron;
				delete[] cfg.shift_snn_hidden_neuron;

				delete cfg.counter;
				cfg.counter = NULL;
			}
			else
			{
				*cfg.counter -= 1;
			}
		}

		bool ConvNeuralNetwork::isEmpty()
		{
			return cfg.counter == NULL;
		}

		inline void ConvNeuralNetwork::rotation(Matrix<double>& res, Matrix<double>& a, int p_x, int p_y)
		{
			//!!!!!!!!!!!!!!
			p_x = p_x - 1;
			p_y = p_y - 1;
			//!1111111111111
			int m = a.row_count();
			int n = a.col_count();
			for (int j = 0; j < m; ++j)
				for (int i = 0; i < n; ++i)
				{
					res[j][i] = a[(m - p_y + j) % m][(n - p_x + i) % n];
				}
		}
		inline void ConvNeuralNetwork::convolution(Matrix<double>& data, int& m, int& n, int& row_offset, int& col_offset, Matrix<double>& conv, double& shift, double& result)
		{
			result = shift;
			for (int i = 0; i < m; ++i)
			{
				for (int j = 0; j < n; ++j)
				{
					double d = data[row_offset + i][col_offset + j] * conv[i][j];
					result += d;
				}
			}
		}
		inline void ConvNeuralNetwork::pooling(Matrix<double>& data, int& m, int& n, int& row_offset, int& col_offset, double& weight, double& shift, double& result)
		{
			if (!MAXPOOL || cfg.win_input_image.rows == 55 || cfg.surface_count[0] == 6)
			{
				result = 0;
				for (int i = 0; i < m; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						result += data[row_offset + i][col_offset + j];
					}
				}
				result = weight * result /*/n*m*/ + shift;
				return;
			}
			else
			{
				result = data[row_offset][col_offset];
				for (int i = 0; i < m; ++i)
				{
					for (int j = 0; j < n; ++j)
					{
						result = max<double>(result, data[row_offset + i][col_offset + j]);
					}
				}
				result = weight * result /*/avr*/ + shift;
				return;
			}
		}
		inline void ConvNeuralNetwork::Calculation(ArrayMatrix<double>& data)
		{
			for (int layer = 0; layer < cfg.cnn_layer_count; ++layer)
			{
				OMP_PRAGMA(omp parallel for num_threads(num_threads))
				for (int surf = 0; surf < cfg.surface_count[layer]; ++surf)
				{
					double _value = 0;
					int neuron_y_count = cnn.conv_neuron[layer][surf].row_count();
					int neuron_x_count = cnn.conv_neuron[layer][surf].col_count();
					for (int neuron_y = 0; neuron_y < neuron_y_count; ++neuron_y)
					{
						for (int neuron_x = 0; neuron_x < neuron_x_count; ++neuron_x)
						{
							if (layer == 0)
							{
								int m = cfg.win_conv[layer][surf].rows;
								int n = cfg.win_conv[layer][surf].cols;
								Point offset(cfg.shift_conv[layer][surf].x * neuron_x, cfg.shift_conv[layer][surf].y * neuron_y);

								if (cfg.cnn_conv_type[layer] == typeConv::cyclic)
								{
									int p_y = offset.y % m;
									int p_x = offset.x % n;
									static Matrix<double>* s_data = new Matrix<double>(m, n);
									static Matrix<double>* rot = new Matrix<double>(m, n);
									if ((s_data->row_count() != m) || (s_data->col_count() != n))
									{
										s_data->clear();
										delete s_data;
										s_data = new Matrix<double>(m, n);

										rot->clear();
										delete rot;
										rot = new Matrix<double>(m, n);
									}

									int offset_y = int(offset.y / m) * m;
									int offset_x = int(offset.x / n) * n;
									for (int y = 0; y < m; ++y)
									{
										for (int x = 0; x < n; ++x)
										{
											if ((offset_y + y < data[cfg.data_connect_map[surf]].row_count()) && (offset_x + x < data[cfg.data_connect_map[surf]].col_count()))
												(*s_data)[y][x] = data[cfg.data_connect_map[surf]][offset_y + y][offset_x + x];
											else
												(*s_data)[y][x] = 0.0;
										}
									}
									rotation(*rot, *s_data, p_x + 1, p_y + 1);

									NetFunc::net_f(_value, cfg.cnn_net_func_type[layer], cnn.conv_weight[layer][surf],* rot, cfg.win_conv[layer][surf], Point(0, 0));
								}
								if (cfg.cnn_conv_type[layer] == typeConv::linear)
								{
									NetFunc::net_f(_value, cfg.cnn_net_func_type[layer], cnn.conv_weight[layer][surf], data[cfg.data_connect_map[surf]], cfg.win_conv[layer][surf], offset);
								}

								_value += cnn.conv_bias[layer][surf];

								cnn.conv_neuron[layer][surf][neuron_y][neuron_x] = ActiveFunc::active_function(_value, cfg.cnn_conv_active_func_type[layer]);
							}
							else
							{
								cnn.conv_neuron[layer][surf][neuron_y][neuron_x] = cnn.conv_bias[layer][surf];

								for (int subs_surf = 0; subs_surf < cfg.surface_count[layer - 1]; ++subs_surf)
								{
									if (cfg.conv_connect_map[layer - 1][surf][subs_surf] == 1)
									{
										int m = cfg.win_conv[layer][surf].rows;
										int n = cfg.win_conv[layer][surf].cols;
										Point offset(cfg.shift_conv[layer][surf].x * neuron_x, cfg.shift_conv[layer][surf].y * neuron_y);

										if (cfg.cnn_conv_type[layer] == typeConv::cyclic)
										{
											int p_y = offset.y % m;
											int p_x = offset.x % n;
											static Matrix<double>* s_data = new Matrix<double>(m, n);
											static Matrix<double>* rot = new Matrix<double>(m, n);
											if ((s_data->row_count() != m) || (s_data->col_count() != n))
											{
												s_data->clear();
												delete s_data;
												s_data = new Matrix<double>(m, n);

												rot->clear();
												delete rot;
												rot = new Matrix<double>(m, n);
											}

											int offset_y = int(offset.y / m) * m;
											int offset_x = int(offset.x / n) * n;
											for (int y = 0; y < m; ++y)
											{
												for (int x = 0; x < n; ++x)
												{
													if ((offset_y + y < cnn.subs_neuron[layer - 1][subs_surf].row_count()) && (offset_x + x < cnn.subs_neuron[layer - 1][subs_surf].col_count()))
														(*s_data)[y][x] = cnn.subs_neuron[layer - 1][subs_surf][offset_y + y][offset_x + x];
													else
														(*s_data)[y][x] = 0.0;
												}
											}
											rotation(*rot, *s_data, p_x + 1, p_y + 1);

											NetFunc::net_f(_value, cfg.cnn_net_func_type[layer], cnn.conv_weight[layer][surf],* rot, cfg.win_conv[layer][surf], Point(0, 0));
										}
										if (cfg.cnn_conv_type[layer] == typeConv::linear)
										{
											NetFunc::net_f(_value, cfg.cnn_net_func_type[layer], cnn.conv_weight[layer][surf], cnn.subs_neuron[layer - 1][subs_surf], cfg.win_conv[layer][surf], offset);
										}

										cnn.conv_neuron[layer][surf][neuron_y][neuron_x] += _value;
									}
								}
								cnn.conv_neuron[layer][surf][neuron_y][neuron_x] = ActiveFunc::active_function(cnn.conv_neuron[layer][surf][neuron_y][neuron_x], cfg.cnn_conv_active_func_type[layer]);
							}
						}
					}

					neuron_y_count = cnn.subs_neuron[layer][surf].row_count();
					neuron_x_count = cnn.subs_neuron[layer][surf].col_count();
					for (int neuron_y = 0; neuron_y < neuron_y_count; ++neuron_y)
					{
						for (int neuron_x = 0; neuron_x < neuron_x_count; ++neuron_x)
						{
							int m = cfg.win_subs[layer][surf].rows;
							int n = cfg.win_subs[layer][surf].cols;
							int row_offset = cfg.win_subs[layer][surf].rows * neuron_y;
							int col_offset = cfg.win_subs[layer][surf].cols * neuron_x;

							pooling(cnn.conv_neuron[layer][surf], m, n, row_offset, col_offset, cnn.subs_weight[layer][surf], cnn.subs_bias[layer][surf], _value);

							cnn.subs_neuron[layer][surf][neuron_y][neuron_x] = ActiveFunc::active_function(_value, cfg.cnn_subs_active_func_type[layer]);
						}
					}
				}
			}

			OMP_PRAGMA(omp parallel for num_threads(num_threads))
			for (int y = 0; y < snn.hidden_neuron.row_count(); ++y)
			{
				for (int x = 0; x < snn.hidden_neuron.col_count(); ++x)
				{
					for (int neuron = 0; neuron < cfg.snn_hidden_count; ++neuron)
					{
						double _value = 0;
						int m = cfg.win_snn_hidden_neuron[neuron].rows;
						int n = cfg.win_snn_hidden_neuron[neuron].cols;
						int row_offset = cfg.shift_snn_hidden_neuron[neuron].y * y;
						int col_offset = cfg.shift_snn_hidden_neuron[neuron].x * x;

						if (!cfg.snn_full_connect)
						{
							convolution(cnn.subs_neuron[cfg.cnn_layer_count - 1][cfg.snn_connect_map[neuron][0]], m, n, row_offset, col_offset, snn.hidden_weight[neuron][0], snn.hidden_shift[neuron], _value);
						}
						else
						{
							double sum_value = snn.hidden_shift[neuron];
							for (int k = 0; k < (int)cfg.snn_connect_map[neuron].size(); ++k)
							{
								_value = 0;
								double d_shift = 0;
								convolution(cnn.subs_neuron[cfg.cnn_layer_count - 1][cfg.snn_connect_map[neuron][k]], m, n, row_offset, col_offset, snn.hidden_weight[neuron][k], d_shift/*snn.hidden_shift[neuron]*/, _value);
								sum_value += _value;
							}
							_value = sum_value;
						}

						snn.hidden_neuron[y][x][neuron] = ActiveFunc::active_function(_value, cfg.snn_layer_hidden_active_func_type);
					}

					for (int j = 0; j < snn.output_weight.row_count(); ++j)
					{
						snn.output_neuron[y][x][j] = snn.output_shift[j];
						for (int i = 0; i < snn.output_weight.col_count(); ++i)
						{
							snn.output_neuron[y][x][j] += snn.hidden_neuron[y][x][i] * snn.output_weight[j][i];
						}
					}

					for (int i = 0; i < cfg.output_count; ++i)
					{
						double _value = snn.output_neuron[y][x][i];
						snn.output_neuron[y][x][i] = ActiveFunc::active_function(_value, cfg.snn_layer_output_active_func_type); //z
					}
				}
			}
		}

		NeuralNetworks* ConvNeuralNetwork::Copy()
		{
			ConvNeuralNetwork* copy = new ConvNeuralNetwork(cfg);
			CopyWeight(copy);
			return copy;
		}
		NeuralNetworks* ConvNeuralNetwork::CopyConfig()
		{
			return new ConvNeuralNetwork(cfg);
		}
		void ConvNeuralNetwork::CopyWeight(NeuralNetworks* copy)
		{
			ConvNeuralNetwork* cnn_copy = (ConvNeuralNetwork*)copy;

			for (int layer = 0; layer < cfg.cnn_layer_count; ++layer)
			{
				for (int surf = 0; surf < cfg.surface_count[layer]; ++surf)
				{
					for (int t = 0; t < (int)cnn.conv_weight[layer][surf].size(); ++t)
					{
						cnn_copy->cnn.conv_weight[layer][surf][t] = cnn.conv_weight[layer][surf][t];
					}
					cnn_copy->cnn.subs_weight[layer][surf] = cnn.subs_weight[layer][surf];
				}
			}

			for (int layer = 0; layer < cfg.cnn_layer_count; ++layer)
			{
				for (int surf = 0; surf < cfg.surface_count[layer]; ++surf)
				{
					cnn_copy->cnn.conv_bias[layer][surf] = cnn.conv_bias[layer][surf];
					cnn_copy->cnn.subs_bias[layer][surf] = cnn.subs_bias[layer][surf];
				}
			}

			for (int neuron = 0; neuron < cfg.snn_hidden_count; ++neuron)
			{
				for (int k = 0; k < (int)cfg.snn_connect_map[neuron].size(); ++k)
				{
					for (int y = 0; y < snn.hidden_weight[neuron][k].row_count(); ++y)
					{
						for (int x = 0; x < snn.hidden_weight[neuron][k].col_count(); ++x)
						{
							cnn_copy->snn.hidden_weight[neuron][k][y][x] = snn.hidden_weight[neuron][k][y][x];
						}
					}
				}
				cnn_copy->snn.hidden_shift[neuron] = snn.hidden_shift[neuron];
			}

			for (int neuron = 0; neuron < cfg.output_count; ++neuron)
			{
				cnn_copy->snn.output_shift[neuron] = snn.output_shift[neuron];
			}

			for (int y = 0; y < snn.output_weight.row_count(); ++y)
			{
				for (int x = 0; x < snn.output_weight.col_count(); ++x)
				{
					cnn_copy->snn.output_weight[y][x] = snn.output_weight[y][x];
				}
			}
		}

		Matrix<vector<double>> ConvNeuralNetwork::Calc_for_data(ArrayMatrix<double>& data)
		{
			Calculation(data);
			return snn.output_neuron;
		}
		void ConvNeuralNetwork::Forward(SIMD::Image_32f& response_map, SIMD::Image_32f& image)
		{
			if (image.width == 0 || image.height == 0) return;

			if (cfg.win_input_image.cols != image.width || cfg.win_input_image.rows != image.height)
			{
				//printf("[Legacy::CNN] cfg.win_input_image.cols != image->width || cfg.win_input_image.rows != image->height\n");
				//system("pause");

				//printf("\n	cnn: image size = (%d, %d)\n", image.width, image.height);
				Legacy::ConvNeuralNetwork cnn_old(this, image.width, image.height);
				cnn_old.Forward(response_map, image);
				return;
			}

			ArrayMatrix<double> mdata(cfg.win_input_image.rows, cfg.win_input_image.cols, 1);
			if (cfg.data_surf == 3 || cfg.win_conv[2][0].cols < 1 * 5)
			{
				for (int j = 0; j < cfg.win_input_image.rows; ++j)
				{
					for (int i = 0; i < cfg.win_input_image.cols; ++i)
					{
						mdata[0][j][i] = image.data[j * image.widthStep + i];// 0.0235294117647059f * image.data[j * image.widthStep + i] - 3.0f;
					}
				}
			}
			else
			{
				for (int j = 0; j < cfg.win_input_image.rows; ++j)
				{
					for (int i = 0; i < cfg.win_input_image.cols; ++i)
					{
						mdata[0][j][i] = 0.0235294117647059f * image.data[j * image.widthStep + i] - 3.0f;
					}
				}
			}

			Calculation(mdata);

			for (int j = 0; j < snn.output_neuron.row_count(); ++j)
			{
				for (int i = 0; i < snn.output_neuron.col_count(); ++i)
				{
					output_neuron[j * snn.output_neuron.col_count() + i] = (float)snn.output_neuron[j][i][0];
				}
			}

			Size out_size = getOutputImgSize();
			if (response_map.isEmpty())
			{
				response_map.width = out_size.width;
				response_map.height = out_size.height;
				response_map.widthStep = out_size.width;
				response_map.data = output_neuron;
				response_map.sharingData = true;
			}
			else
			{
				response_map.width = out_size.width;
				response_map.height = out_size.height;
				response_map.copyData(out_size.width, out_size.height, output_neuron, snn.output_neuron.col_count());
			}

			mdata.clear();
		}
		void ConvNeuralNetwork::Calc_for_image_rgb(SIMD::Image_32f& response_map, SIMD::Image_32f& R, SIMD::Image_32f& G, SIMD::Image_32f& B)
		{
			if (cfg.win_input_image.cols != R.width || cfg.win_input_image.rows != R.height)
			{
				printf("[Legacy::CNN] cfg.win_input_image.cols != image->width || cfg.win_input_image.rows != image->height\n");
				system("pause");
			}

			ArrayMatrix<double> mdata(cfg.win_input_image.rows, cfg.win_input_image.cols, 3);
			for (int j = 0; j < cfg.win_input_image.rows; ++j)
			{
				for (int i = 0; i < cfg.win_input_image.cols; ++i)
				{
					mdata[0][j][i] = R.data[j * R.widthStep + i];// 0.0235294117647059f * image.data[j * image.widthStep + i] - 3.0f;
					mdata[1][j][i] = G.data[j * G.widthStep + i];
					mdata[2][j][i] = B.data[j * B.widthStep + i];
				}
			}

			Calculation(mdata);

			for (int j = 0; j < snn.output_neuron.row_count(); ++j)
			{
				for (int i = 0; i < snn.output_neuron.col_count(); ++i)
				{
					output_neuron[j * snn.output_neuron.col_count() + i] = (float)snn.output_neuron[j][i][0];
				}
			}

			Size out_size = getOutputImgSize();
			response_map.width = out_size.width;
			response_map.height = out_size.height;
			response_map.widthStep = out_size.width;
			response_map.data = output_neuron;
			response_map.sharingData = true;

			mdata.clear();
		}

		Size ConvNeuralNetwork::getOutputImgSize(const Size size)
		{ 
			if (cfg.win_input_image.cols != size.width || cfg.win_input_image.rows != size.height)
			{
				Legacy::ConvNeuralNetwork cnn_old(this, size.width, size.height);
				return cnn_old.getOutputImgSize();
			}

			return Size(snn.output_neuron.col_count(), snn.output_neuron.row_count()); 
		}

		float ConvNeuralNetwork::getInputOutputRatio() const
		{
			float ratio = 1.0f;
			for (int layer = 0; layer < cfg.cnn_layer_count; ++layer)
			{
				ratio *= cfg.win_subs[layer][0].rows;
			}
			return ratio;
		}

		ConvNeuralNetwork::typeConv ConvNeuralNetwork::StrToTConv(string str)
		{
			if (str == "linear") return linear;
			if (str == "cyclic") return cyclic;
			return linear;
		}
		string ConvNeuralNetwork::TConvToStr(ConvNeuralNetwork::typeConv type_conv)
		{
			if (type_conv == linear) return "linear";
			if (type_conv == cyclic) return "cyclic";
			return "linear";
		}
	}

}