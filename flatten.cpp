//
// Created by 陈笑凡 on 2025/5/15.
//

#include "flatten.h"

using namespace std;

vector<int> flattenLayer::get_output_shape(const vector<int>& input_shape)const
{
    int total_size = 1;
    for (int dim : input_shape)
    {
        total_size *= dim;
    }
    return { total_size };
}

void flattenLayer::forward(const Tensor& input, Tensor& output)
{
    output.shape = {input.size()};
    output.data = input.data;
}