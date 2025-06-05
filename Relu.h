//
// Created by 陈笑凡 on 2025/5/15.
//

#ifndef RELU_H
#define RELU_H

#include "layer.h"
#include "Tensor.h"

using namespace std;

class reluLayer : public layer
{
public:
    reluLayer() = default;
    void forward(const Tensor& input, Tensor& output) override;
    std::vector<int> get_output_shape(const std::vector<int>& input_shape)const override;
    virtual ~reluLayer() = default;
};

#endif //RELU_H
