//
// Created by 陈笑凡 on 2025/5/18.
//

#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "layer.h"
#include "Tensor.h"

class fc_layer : public layer
{
private:
    Tensor weights;
    Tensor biases;
public:
    fc_layer(const float* weights_data,  int in_features, int out_features, const float* biases_data, int bias_size);
    void forward(const Tensor &input, Tensor &output) override;
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
    ~fc_layer() = default;
};

#endif //FC_LAYER_H
