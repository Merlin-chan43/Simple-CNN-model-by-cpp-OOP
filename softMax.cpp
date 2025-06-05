//
// Created by 陈笑凡 on 2025/5/15.
//

#include "softMax.h"
#include <cmath>

using namespace std;

vector<int> softMax::get_output_shape(const vector<int>& input_shape)const
{
    return input_shape;
}

void softMax::forward(const Tensor& input, Tensor& output)
{
    float max_val = std::max(input.data[0], input.data[1]);
    output.shape = {input.size()};
    output.data.resize(input.size());

    float total = 0.0f;
    for (int i = 0; i < input.data.size(); i++)
    {
         total += exp(input.data[i] - max_val);
    }

    for (int i = 0; i < input.data.size(); i++)
    {
        output.data[i] = exp(input.data[i] - max_val) / total;
    }
}