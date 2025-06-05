//
// Created by 陈笑凡 on 2025/5/12.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <string>
#include <initializer_list>

using namespace std;

struct Tensor
{
    vector<float> data;
    vector<int> shape;// 存储每一维度的尺寸，例如 {通道, 高度, 宽度}

    // 构造函数声明
    Tensor(vector<int, std::allocator<int>> m_shape) : shape(m_shape)
    {
        int total = 1;
        if (shape.empty()) total = 0;
        else
        {
            for (auto shapes : m_shape) total *= shapes;
        }
        data.resize(total);
    }
    // 默认构造函数声明
    Tensor() = default;

    // 计算张量总元素数量的方法声明
    int size() const
    {
        int total = 1;
        if (shape.empty()) return 0;
        for (int entry : shape) total *= entry;
        return total;
    }

    // 根据多维索引计算在一维 data 数组中的线性偏移量
    int caculate_linear_index(const vector<int>& indices) const
    {

        int linear_index = 0;
        int stride = 1;

        for (int i = shape.size() - 1; i >= 0; i--)
        {
            
            linear_index += stride * indices[i];
            stride *= shape[i];
        }
        return linear_index;
    }

    // 重载 () 操作符的声明，允许使用 (d0, d1, ...) 方式访问元素
    float& operator()(const vector<int>& indices)
    {
        int idx = caculate_linear_index(indices);
        return data[idx];
    }
    const float& operator()(const vector<int>& indices) const
    {
        int idx = caculate_linear_index(indices);
        return data[idx];
    }
};


#endif //TENSOR_H
