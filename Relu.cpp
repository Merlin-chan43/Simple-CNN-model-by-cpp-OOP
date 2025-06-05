//
// Created by 陈笑凡 on 2025/5/15.
//

#include "Relu.h"
#include <algorithm>

using namespace std;

vector<int> reluLayer::get_output_shape(const vector<int>& input_shape) const
{
    return input_shape;
}

void reluLayer::forward(const Tensor& input, Tensor& output)
{
    output.shape = input.shape;
    output.data.resize(input.size());
    for (int i = 0; i < input.size(); i++)
    {
        output.data[i] = max(0.0f, input.data[i]);
    }
}