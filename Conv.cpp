//
// Created by 戴烨铭 on 2025/5/21
//
#include "Conv.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>    // 用于抛出异常

// --- 卷积层类实现 ---

// 构造函数实现
Conv::Conv(int pad, int stride, int kernel_size, int in_channels, int out_channels,  const float* weights_data,
    const float* biases_data, int bias_size)
    : pad_(pad), stride_(stride), kernel_size_(kernel_size), in_channels_(in_channels), out_channels_(out_channels)
{
    // --- 1. 初始化 weights_ 成员 Tensor ---
    // 卷积核形状是 {out_channels, in_channels, kernel_size, kernel_size} (4D)
    weights_.shape = { out_channels_, in_channels_, kernel_size_, kernel_size_ };
    int weights_total_size = weights_.size();

    if (weights_total_size <= 0) {
        throw std::invalid_argument("SimpleConvBNLayer: weights total size must be greater than zero.");
    }
    // 检查传入的 in_channels 和 out_channels 是否与权重 Tensor 的形状匹配
    // (这在实际情况中更严谨，这里假设传入的 out_channels, in_channels, kernel_size 正确定义了权重形状)
    if (weights_.shape[0] != out_channels_ || weights_.shape[1] != in_channels_ ||
        weights_.shape[2] != kernel_size_ || weights_.shape[3] != kernel_size_) {
        throw std::invalid_argument("SimpleConvBNLayer: Inconsistent dimensions in weights data provided.");
    }

    weights_.data.resize(weights_total_size); // 调整数据向量大小

    if (weights_data) {
        std::copy(weights_data, weights_data + weights_total_size, weights_.data.begin());
    }
    else {
        throw std::invalid_argument("SimpleConvBNLayer: weights_data pointer is null.");
    }


    // --- 2. 初始化 biases_ 成员 Tensor ---
    // 偏置是 1D Tensor，形状 {out_channels}
    biases_.shape = { out_channels_ };
    int bias_total_size = biases_.size();

    if (bias_total_size != out_channels_) { // 偏置的数量必须等于输出通道数
        throw std::invalid_argument("SimpleConvBNLayer: biases size must be equal to out_channels.");
    }

    biases_.data.resize(bias_total_size); // 调整数据向量大小

    if (biases_data) {
        std::copy(biases_data, biases_data + bias_total_size, biases_.data.begin());
    }
    else {
        throw std::invalid_argument("SimpleConvBNLayer: biases_data pointer is null.");
    }

    // std::cout << "SimpleConvBNLayer constructed with kernel_size=" << kernel_size_
    //           << ", stride=" << stride_ << ", pad=" << pad_
    //           << ", in_channels=" << in_channels_ << ", out_channels=" << out_channels_ << std::endl; // 调试信息
}

// get_output_shape 方法实现
// 根据输入形状、卷积核尺寸、步长和填充计算输出形状
std::vector<int> Conv::get_output_shape(const std::vector<int>& input_shape) const {
    // 输入 Tensor 必须是 3D (通道, 高度, 宽度)
    if (input_shape.size() != 3) {
        throw std::invalid_argument("SimpleConvBNLayer expects 3D input shape [C, H, W].");
    }

    // 检查输入通道数是否与该层期望的输入通道数匹配
    if (input_shape[0] != in_channels_) {
        throw std::invalid_argument("SimpleConvBNLayer: Input channels mismatch (" +
            std::to_string(input_shape[0]) + " != " + std::to_string(in_channels_) + ").");
    }

    int in_h = input_shape[1];
    int in_w = input_shape[2];

    // 计算输出高度和宽度
    // 公式: floor((输入尺寸 + 2 * 填充 - 核尺寸) / 步长) + 1
    // 注意：需要进行浮点数转换以避免整数除法截断
    int H_out = static_cast<int>(std::floor(static_cast<float>(in_h + 2 * pad_ - kernel_size_) / stride_)) + 1;
    int W_out = static_cast<int>(std::floor(static_cast<float>(in_w + 2 * pad_ - kernel_size_) / stride_)) + 1;

    // 检查计算出的输出尺寸是否有效
    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("SimpleConvBNLayer: Output spatial dimensions are <= 0. Input shape might be too small for kernel/stride/padding.");
    }

    int out_c = out_channels_; // 输出通道数就是该层使用的滤波器数量
    //cout << out_c << H_out << W_out << endl;

    return { out_c, H_out, W_out }; // 返回计算出的输出形状
}


// forward 方法实现
// 对输入 Tensor 执行卷积计算
void Conv::forward(const Tensor& input, Tensor& output) {
    // 1. 输入 Tensor 形状检查 (通常在 get_output_shape 内部已包含，这里再确认一次)
    if (input.shape.size() != 3) {
        throw std::invalid_argument("SimpleConvBNLayer forward: Input tensor must be 3D [C, H, W].");
    }
    int in_c = input.shape[0];
    int in_h = input.shape[1];
    int in_w = input.shape[2];

    // 检查输入通道数是否匹配
    if (in_c != in_channels_) {
        throw std::invalid_argument("SimpleConvBNLayer forward: Input channels mismatch for forward pass.");
    }


    // 2. 确定输出 Tensor 形状和大小
    // 调用 get_output_shape 计算输出形状
    std::vector<int> output_shape = get_output_shape(input.shape);
    int out_c = output_shape[0];
    int out_h = output_shape[1];
    int out_w = output_shape[2];

    output.shape = output_shape; // 设置输出 Tensor 的形状
    output.data.resize(output.size()); // 根据形状调整 output.data 的大小，分配内存


    // 3. 核心计算：卷积
    // 遍历输出 Tensor 的每一个位置 [oc, oh, ow]
    for (int oc = 0; oc < out_c; ++oc) { // 遍历输出通道 (对应滤波器)
        for (int oh = 0; oh < out_h; ++oh) { // 遍历输出高度
            for (int ow = 0; ow < out_w; ++ow) { // 遍历输出宽度

                float sum = 0.0f; // 初始化当前输出位置的累加值

                // 计算当前输出位置 [oc, oh, ow] 对应的输入 Tensor 中的卷积窗口区域的起始坐标 (左上角)
                // 考虑步长和填充
                int ih_start = oh * stride_ - pad_;
                int iw_start = ow * stride_ - pad_;

                // --- 中层循环：遍历输入通道 ---
                for (int ic = 0; ic < in_channels_; ++ic) { // 遍历输入通道 (滤波器的深度)

                    // --- 内层循环：遍历卷积核的像素 ---
                    for (int kh = 0; kh < kernel_size_; ++kh) { // 遍历卷积核高度
                        for (int kw = 0; kw < kernel_size_; ++kw) { // 遍历卷积核宽度

                            // 计算卷积核像素在输入 Tensor 中的实际坐标
                            int ih = ih_start + kh;
                            int iw = iw_start + kw;

                            // 检查当前输入坐标是否在填充后的输入边界内
                            // 只有在有效边界内的像素才参与计算，否则视为0 (填充区域)
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                // 访问输入 Tensor 元素: input({ic, ih, iw})
                                // 访问权重 Tensor 元素: weights_({oc, ic, kh, kw})
                                sum += input({ ic, ih, iw }) * weights_({ oc, ic, kh, kw });
                            }
                            // 如果 ih 或 iw 超出了输入 Tensor 的实际边界 (由于 padding 或窗口部分在外面)，
                            // 那么根据卷积的定义，它们被认为是 0，所以不需要在这里显式加 0。
                        }
                    }
                }
                // 加上偏置项 (bias 是每个输出通道一个值)
                sum += biases_.data[oc]; // biases_ 是 1D Tensor，直接用索引 oc 访问

                // 将结果存入输出 Tensor 的对应位置
                output({ oc, oh, ow }) = sum;
            }
        }
    }
}