//
// Created by 陈笑凡 on 2025/5/17.
//

#include "maxPooling.h"
#include <iostream>

using namespace std;

std::vector<int> maxPooling::get_output_shape(const std::vector<int>& input_shape) const
{
    if (input_shape.size() < 3)
    {
        throw runtime_error("Invalid shape for maxPooling");
    }

    int out_c = input_shape[0];
    int out_h = floor((input_shape[1] - pool_h) / stride_h) + 1;
    int out_w = floor((input_shape[2] - pool_w) / stride_w) + 1;

    return vector<int>({out_c, out_h, out_w});
}

void maxPooling::forward(const Tensor &input, Tensor &output)
{
    vector<int> output_shape = get_output_shape(input.shape);
    int out_c = output_shape[0];
    int out_h = output_shape[1];
    int out_w = output_shape[2];

    output.shape = output_shape;
    output.data.resize(output.size());

    for (int oc = 0; oc < out_c; oc++)
    {
        for (int oh = 0; oh < out_h; oh++)
        {
            for (int ow = 0; ow < out_w; ow++)
            {
                int ih_start = oh * stride_h;
                int iw_start = ow * stride_w;

                float max_val = numeric_limits<float>::lowest();

                for (int ph = 0; ph < pool_h; ph++)
                {
                    for (int pw = 0; pw < pool_w; pw++)
                    {
                        int ih = ih_start + ph;
                        int iw = iw_start + pw;

                        float current_input_value = input({oc, ih, iw});

                        max_val = max(max_val, current_input_value);
                    }
                }

                output({oc, oh, ow}) = max_val;
            }
        }
    }
}
