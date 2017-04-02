## Compact Convolutional Neural Network Cascade ##

This is a binary library for very fast detection of simple objects in images using CPU or GPU.<br>
Implemented of the algorithm described in the following paper:

	I.A. Kalinovskii, V.G. Spitsyn,
	Compact Convolutional Neural Network Cascade for Face Detection,
	http://arxiv.org/abs/1508.01292

If you use the provided binaries for your work, please cite this paper. 

examples/main.cpp shows how to use the library.<br>
You need a processor with AVX or AVX2 (2x speed up due to used INT16) instruction set support.<br>
Supported Nvidia GPUs with compute capability 3.0 and higher (library builded with CUDA 8.0).<br>

![Examples](/test_images/7.jpg_result.jpg "Detection example")
This image has a resolution of 4800x2400 was processed for 400 ms on GT640M GPU at searches minimum face of size 20x20 pixels. This detector capable of processing 4K video stream in real time.<br>

You can trainig own cascade using [Microsoft Cognitive Toolkit](https://github.com/Microsoft/CNTK) (recommended version [1.7.2](https://github.com/Microsoft/CNTK/releases/tag/v1.7.2)).<br>
You should not change model prototype (see cntk folder). Other CNN architectures are currently not supported.<br>

If you need the source code, a commercial license is needed.

## Contact

For any additional information contact me at <kua_21@mail.ru>.

Copyright (c) 2017, Ilya Kalinovskii.<br>
All rights reserved.