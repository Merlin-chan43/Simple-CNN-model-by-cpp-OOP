//
// Created by ´÷ìÇÃú on 2025/5/20
//

#ifndef CNN_H
#define CNN_H

#include "Tensor.h"
#include "Relu.h"
#include "maxPooling.h"
#include "softMax.h"
#include "fc_layer.h"
#include "flatten.h"
#include "layer.h"
#include "Conv.h"
//------------------------
#include <vector>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

using namespace std;

class CNN
{
private:
	vector<shared_ptr<layer>> layers;
public:
	CNN() = default;
	Tensor predict(Tensor& input);
	void add_layer(shared_ptr<layer> Layer);
	Tensor load_image_as_tensor(const char* path);
	~CNN() = default;
};

#endif