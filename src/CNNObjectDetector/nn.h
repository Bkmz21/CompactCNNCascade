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

#include "af.h"
#include "matrix.h"


//========================================================================================================


namespace NeuralNetworksLib
{
	namespace Legacy
	{
		class NeuralNetworks
		{
		public:
			enum typeDataSet
			{
				train = 0,
				test = 1
			};

			struct dataNeuron
			{
				double net;
				double out;
				double error;
				void* options;

				dataNeuron()
				{
					net = 0;
					out = 0;
					error = 0;
					options = NULL;
				}
			};
			struct dataWeight
			{
				double* w;
				double* dydw;
				bool fixed;
				dataNeuron* neuron_in;
				dataNeuron* neuron_out;
				void* options;

				dataWeight()
				{
					w = NULL;
					dydw = new double;
					*dydw = 0.0;
					fixed = false;
					neuron_in = NULL;
					neuron_out = NULL;
					options = NULL;
				}
			};
			struct st_error
			{
				double avr;
				double avr_sqr;
				long int data;
				double avr_w;

				st_error()
				{
					avr = 0.0;
					avr_sqr = 0.0;
					data = 0;
					avr_w = 0.0;
				}
			};

			std::vector<dataNeuron*> neurons_ref;
			std::vector<dataWeight*> weights_ref;

			virtual NeuralNetworks* Copy() { return NULL; }
			virtual NeuralNetworks* CopyConfig() { return NULL; }
			//virtual void CopyWeight(NeuralNetworks* copy) { }

			//virtual int DataCount(typeDataSet type_data_set) { return 0; }

			//virtual void SetTrainParam(string str) { }
			//virtual string GetTrainParam() { return ""; }

			//virtual int  GetOutputCount() { return -1; }
			//virtual void GetdEdOut(Matrix<vector<double>>& vector) { }

			//virtual st_error Calc_ApproxError(typeDataSet type_data_set, double part = 1.0) { return st_error(); }
			//virtual bool Calc_dEdw(int data_index, typeDataSet type_data_set) { return 0; }
			//virtual bool Calc_dYdw(int data_index, typeDataSet type_data_set, bool dEdw = false, bool skip_data = false) { return 0; }
			//virtual void Calc_divergence(Matrix<vector<double>>& res, int index, typeDataSet type_data_set) { }

			//virtual void DataTrainCreate() { }
			//virtual void DataTrainClear() { }

			//virtual void SetDropUnit() { }

			//virtual void SaveToFile(string file_name = "") { }

			virtual void Clear() { }
		};
	}

}