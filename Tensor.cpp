//
// Created by 陈笑凡 on 2025/5/14.
//
#include "Tensor.h"
using namespace std;



int Tensor::size() const 
{
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

int Tensor::caculate_linear_index(const vector<int>& indices) const
{
    if (indices.size() != size())
    {
        throw invalid_argument("Tensor::caculate_linear_index: Indices dimension mismatch with tensor shape (" + to_string(indices.size()) + "!=" + to_string(shape.size()) + ")");
    }

    int linear_index = 0;
    int stride = 1;

    for (int i = shape.size() - 1; i >= 0; i--)
    {
        if (indices[i] < 0 || indices[i] >= shape[i])
        {
            throw out_of_range("Tensor::caculate_linear_index: indices out of dimesion" + to_string(i) + ".Index: " + to_string(indices[i]) + ", Dimension size: " + to_string(shape[i]));
        }
        linear_index += stride * indices[i];
        stride *= shape[i];
    }
    return linear_index;
}

float& Tensor::operator()(const vector<int>& indices)
{
    int idx = caculate_linear_index(indices);
    return data[idx];
}

const float& Tensor::operator()(const vector<int>& indices) const
{
    int idx = caculate_linear_index(indices);
    return data[idx];
}
