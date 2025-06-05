//
// Created by 陈笑凡 on 2025/5/15.
//

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "layer.h"
#include "Tensor.h"

class softMax : public layer
{
public:
    softMax() = default;
    void forward(const Tensor& input, Tensor& output) override;
    std::vector<int> get_output_shape(const std::vector<int>& input_shape)const override;
    ~softMax() = default;
};

#endif //SOFTMAX_H
