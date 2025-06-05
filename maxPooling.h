//
// Created by 陈笑凡 on 2025/5/17.
//

#ifndef MAXPOOLING_H
#define MAXPOOLING_H

#include "layer.h"
#include "Tensor.h"

class maxPooling : public layer
{
private:
    int pool_h, pool_w;
    int stride_h, stride_w;

public:
    maxPooling() = default;
    maxPooling(int h, int w, int stride_h, int stride_w) : pool_h(h), pool_w(w), stride_h(stride_h), stride_w(stride_w) {}
    void forward(const Tensor &input, Tensor &output) override;
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;
    ~maxPooling() = default;
};



#endif //MAXPOOLING_H
