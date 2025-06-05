//
// Created by 戴烨铭 on 2025/5/21
//
#ifndef CONV_LAYER_H // 头文件保护宏
#define CONV_LAYER_H

#include "layer.h"  // 包含 Layer 基类的定义
#include "Tensor.h" // 包含 Tensor 结构的定义
#include <vector>   // 包含 std::vector

// --- 卷积层类声明 ---
// 继承自 Layer，实现卷积层功能 (融合了 BN 参数)
class Conv : public layer { // 命名与讨论保持一致
private:
    Tensor weights_;    // 权重 Tensor，形状 {out_channels, in_channels, kernel_size, kernel_size}
    Tensor biases_;     // 偏置 Tensor，形状 {out_channels}
    int pad_;           // 填充大小 (这里假设高度和宽度方向填充相同)
    int stride_;        // 步长 (这里假设高度和宽度方向步长相同)
    int kernel_size_;   // 卷积核边长 (这里假设是方形核)
    int in_channels_;   // 输入通道数 (显式存储，也可用 weights_.shape[1] 得到)
    int out_channels_;  // 输出通道数 (显式存储，也可用 weights_.shape[0] 得到)

public:
    // 构造函数：接收原始权重和偏置数据指针及所有必要参数
    Conv(int pad, int stride,int kernel_size, int out_channels, int in_channels,   const float* weights_data,
         const float* biases_data, int bias_size);

    // 实现基类中的 forward 方法
    // 对输入 Tensor (3D 特征图) 执行卷积计算，结果存入输出 Tensor
    void forward(const Tensor& input, Tensor& output) override;

    // 实现基类中的 get_output_shape 方法
    // 根据输入形状、卷积核尺寸、步长和填充计算输出形状
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;

    // 析构函数
    ~Conv() override = default;
};

#endif // CONV_LAYER_H