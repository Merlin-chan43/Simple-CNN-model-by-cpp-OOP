//
// Created by 陈笑凡 on 2025/5/15.
//

#ifndef FLATTEN_H
#define FLATTEN_H

#include "layer.h"
#include "Tensor.h"

using namespace std;

class flattenLayer : public layer
{
public:
    flattenLayer() = default;
    vector<int> get_output_shape(const vector<int>& input_shape) const override;
    void forward(const Tensor& input, Tensor& output) override;
    ~flattenLayer() = default;
};

#endif //FLATTEN_H
