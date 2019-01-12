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


#include "af.h"

using namespace std;


//================================================================================================================================================


namespace NeuralNetworksLib
{

	namespace Legacy
	{
		double ActiveFunc::_linear(double& input)
		{
			return input;
		}
		double ActiveFunc::diff_linear(double& input)
		{
			return 1.0;
		}

		double ActiveFunc::_sigmoid(double& input)
		{
			const double k = 1.0;
			return 1.0 / (1.0 + exp(-k * input));
		}
		double ActiveFunc::diff_sigmoid(double& input)
		{
			const double k = 1.0;
			double exp_val = exp(-k * input);
			return k * exp_val / ((1 + exp_val) * (1 + exp_val));
		}

		double ActiveFunc::_tgh(double& input)
		{
			const double a = 1.7159;
			const double b = 2.0 / 3.0;
			return a * tanh(b * input);
		}
		double ActiveFunc::diff_tgh(double& input)
		{
			const double a = 1.7159;
			const double b = 2.0 / 3.0;
			return a * b * (1.0 - tanh(b * input) * tanh(b * input));
		}

		double ActiveFunc::_tgh_2(double& input)
		{
			return tanh(input);
		}
		double ActiveFunc::diff_tgh_2(double& input)
		{
			return 1.0 - tanh(input) * tanh(input);
		}

		double ActiveFunc::_approx_tgh(double& input)
		{
			const double a = 1.7159;
			const double b = 2.0 / 3.0;
			const double c = 1.41645;
			double u = b * input;
			double sgn = 1.0;
			if (input < 0) sgn = -1.0;
			return a * sgn * (1.0 - 1.0 / (1.0 + abs(u) + u*u + c*u*u*u*u));
		}
		double ActiveFunc::diff_approx_tgh(double& input)
		{
			const double a = 1.7159;
			const double b = 2.0 / 3.0;
			const double c = 1.41645;
			double u = b * input;
			double sgn = 1.0;
			if (input < 0) sgn = -1.0;
			double p = u*u;
			double g = c*p*u;
			double d1 = sgn + 2.0*u + 4.0*g;
			double d2 = 1.0 + abs(u) + p + g*u;
			return a * sgn * b * d1 / (d2 * d2);
		}

		double ActiveFunc::_approx_tgh_2(double& input)
		{
			const double a = 1.0;
			const double b = 1.0;
			const double c = 1.41645;
			double u = b * input;
			double sgn = 1.0;
			if (input < 0) sgn = -1.0;
			return a * sgn * (1.0 - 1.0 / (1.0 + abs(u) + u*u + c*u*u*u*u));
		}
		double ActiveFunc::diff_approx_tgh_2(double& input)
		{
			const double a = 1.0;
			const double b = 1.0;
			const double c = 1.41645;
			double u = b * input;
			double sgn = 1.0;
			if (input < 0) sgn = -1.0;
			double p = u*u;
			double g = c*p*u;
			double d1 = sgn + 2.0*u + 4.0*g;
			double d2 = 1.0 + abs(u) + p + g*u;
			return a * sgn * b * d1 / (d2 * d2);
		}

		double ActiveFunc::_softplus(double& input)
		{
			return log(1.0 + exp(input));
		}
		double ActiveFunc::diff_softplus(double& input)
		{
			return 1.0 / (1.0 + exp(-input));
		}

		double ActiveFunc::_relu(double& input)
		{
			if (input > 0) return input;
			return 0.0;
		}
		double ActiveFunc::diff_relu(double& input)
		{
			if (input > 0) return 1.0;
			return 0.0;
		}

		double ActiveFunc::active_function(double& input, ActiveFunc::typeFuncAF& ftype)
		{
			if (ftype == linear)		return _linear(input);
			if (ftype == sigmoid)		return _sigmoid(input);
			if (ftype == tgh)			return _tgh(input);
			if (ftype == tgh_2)			return _tgh_2(input);
			if (ftype == approx_tgh)	return _approx_tgh(input);
			if (ftype == approx_tgh_2)	return _approx_tgh_2(input);
			if (ftype == softplus)		return _softplus(input);
			if (ftype == relu)			return _relu(input);
			return 0;
		}

		ActiveFunc::typeFuncAF ActiveFunc::StrToTFAF(string str)
		{
			if (str == "linear")		return linear;
			if (str == "sigmoid")		return sigmoid;
			if (str == "tgh")			return tgh;
			if (str == "tgh_2")			return tgh_2;
			if (str == "approx_tgh")	return approx_tgh;
			if (str == "approx_tgh_2")  return approx_tgh_2;
			if (str == "softplus")		return softplus;
			if (str == "relu")			return relu;
			return approx_tgh;
		}
		string ActiveFunc::TFAFToStr(ActiveFunc::typeFuncAF type_af)
		{
			if (type_af == linear)			return "linear";
			if (type_af == sigmoid)			return "sigmoid";
			if (type_af == tgh)				return "tgh";
			if (type_af == tgh_2)			return "tgh_2";
			if (type_af == approx_tgh)		return "approx_tgh";
			if (type_af == approx_tgh_2)	return "approx_tgh_2";
			if (type_af == softplus)		return "softplus";
			if (type_af == relu)			return "relu";
			return "approx_tgh";
		}
	}

}
