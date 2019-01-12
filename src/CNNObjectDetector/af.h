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

#include <string>


//========================================================================================================


namespace NeuralNetworksLib
{

	namespace Legacy
	{
		class ActiveFunc
		{
		private:
			static inline double _linear(double& input);
			static inline double diff_linear(double& input);

			static inline double _sigmoid(double& input);
			static inline double diff_sigmoid(double& input);

			static inline double _tgh(double& input);
			static inline double diff_tgh(double& input);

			static inline double _tgh_2(double& input);
			static inline double diff_tgh_2(double& input);

			static inline double _approx_tgh(double& input);
			static inline double diff_approx_tgh(double& input);

			static inline double _approx_tgh_2(double& input);
			static inline double diff_approx_tgh_2(double& input);

			static inline double _softplus(double& input);
			static inline double diff_softplus(double& input);

			static inline double _relu(double& input);
			static inline double diff_relu(double& input);

		public:
			enum typeFuncAF
			{
				linear			= 0,
				sigmoid			= 1,
				tgh				= 2,
				tgh_2			= 3,
				approx_tgh		= 4,
				approx_tgh_2	= 5,
				softplus		= 7,
				relu			= 8
			};

			static double active_function(double& input, ActiveFunc::typeFuncAF& ftype);

			static ActiveFunc::typeFuncAF StrToTFAF(std::string str);
			static std::string TFAFToStr(ActiveFunc::typeFuncAF type_af);
		};
	}
}