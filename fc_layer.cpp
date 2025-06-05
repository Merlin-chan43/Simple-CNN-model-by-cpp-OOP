//
// Created by 陈笑凡 on 2025/5/18.
//

#include "fc_layer.h"

fc_layer::fc_layer(const float* weights_data, int in_features, int out_features, const float* biases_data, int bias_size)
{
    weights.shape = {out_features, in_features};
    int weights_total_size = weights.size();

    if (weights_total_size <= 0)
    {
        throw std::invalid_argument("fc_layer: weights size must be greater than zero");
    }

    weights.data.resize(weights_total_size);

    if (weights_data)
    {
        std::copy(weights_data, weights_data + weights_total_size, weights.data.begin());
    }
    else
    {
        throw std::invalid_argument("fc_layer: weights data is empty");
    }

    biases.shape = {bias_size};
    int bias_total_size = biases.size();

    if (bias_total_size <= 0)
    {
        throw std::invalid_argument("fc_layer: biases must be greater than zero");
    }

    biases.data.resize(bias_total_size);

    if (biases_data)
    {
        std::copy(biases_data, biases_data + bias_total_size, biases.data.begin());
    }
    else
    {
        throw std::invalid_argument("fc_layer: biases data is empty");
    }
}

std::vector<int> fc_layer::get_output_shape(const std::vector<int>& input_shape) const
{
    if (input_shape.size() != 1)
    {
        throw std::invalid_argument("fc_layer: input shape must be 1-dimensional");
    }

    int in_features = this->weights.shape[1];
    if (input_shape[0] != in_features)
    {
        throw std::invalid_argument("fc_layer: input shape must have the same number of elements");
    }

    int out_features = this->weights.shape[0];

    return {out_features};
}

void fc_layer::forward(const Tensor &input, Tensor &output)
{
    if (input.shape.size() != 1)
    {
        throw std::invalid_argument("fc_layer: input shape must be 1-dimensional");
    }

    int out_features = this->weights.shape[0];
    int in_features = this->weights.shape[1];
    if (input.shape[0] != in_features)
    {
        throw std::invalid_argument("fc_layer: input shape must have the same number of elements");
    }

    output.shape = {out_features};
    output.data.resize(output.size());

    for (int o = 0; o < out_features; o++)
    {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++)
        {
            sum += input.data[i] * this->weights({o, i});
        }
        float final_output_value = sum + biases({o});
        output({o}) = final_output_value;
    }
}

