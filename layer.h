//
// Created by 陈笑凡 on 2025/5/14.
//

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Tensor.h"

class layer
{
public:
    virtual void forward(const Tensor& input, Tensor& output) = 0;

    virtual std::vector<int> get_output_shape(const std::vector<int>& input_shape)const = 0;

    virtual ~layer()  = default;
};

#endif //LAYER_H
