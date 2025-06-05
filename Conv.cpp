//
// Created by ������ on 2025/5/21
//
#include "Conv.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>    // �����׳��쳣

// --- �������ʵ�� ---

// ���캯��ʵ��
Conv::Conv(int pad, int stride, int kernel_size, int in_channels, int out_channels,  const float* weights_data,
    const float* biases_data, int bias_size)
    : pad_(pad), stride_(stride), kernel_size_(kernel_size), in_channels_(in_channels), out_channels_(out_channels)
{
    // --- 1. ��ʼ�� weights_ ��Ա Tensor ---
    // �������״�� {out_channels, in_channels, kernel_size, kernel_size} (4D)
    weights_.shape = { out_channels_, in_channels_, kernel_size_, kernel_size_ };
    int weights_total_size = weights_.size();

    if (weights_total_size <= 0) {
        throw std::invalid_argument("SimpleConvBNLayer: weights total size must be greater than zero.");
    }
    // ��鴫��� in_channels �� out_channels �Ƿ���Ȩ�� Tensor ����״ƥ��
    // (����ʵ������и��Ͻ���������贫��� out_channels, in_channels, kernel_size ��ȷ������Ȩ����״)
    if (weights_.shape[0] != out_channels_ || weights_.shape[1] != in_channels_ ||
        weights_.shape[2] != kernel_size_ || weights_.shape[3] != kernel_size_) {
        throw std::invalid_argument("SimpleConvBNLayer: Inconsistent dimensions in weights data provided.");
    }

    weights_.data.resize(weights_total_size); // ��������������С

    if (weights_data) {
        std::copy(weights_data, weights_data + weights_total_size, weights_.data.begin());
    }
    else {
        throw std::invalid_argument("SimpleConvBNLayer: weights_data pointer is null.");
    }


    // --- 2. ��ʼ�� biases_ ��Ա Tensor ---
    // ƫ���� 1D Tensor����״ {out_channels}
    biases_.shape = { out_channels_ };
    int bias_total_size = biases_.size();

    if (bias_total_size != out_channels_) { // ƫ�õ���������������ͨ����
        throw std::invalid_argument("SimpleConvBNLayer: biases size must be equal to out_channels.");
    }

    biases_.data.resize(bias_total_size); // ��������������С

    if (biases_data) {
        std::copy(biases_data, biases_data + bias_total_size, biases_.data.begin());
    }
    else {
        throw std::invalid_argument("SimpleConvBNLayer: biases_data pointer is null.");
    }

    // std::cout << "SimpleConvBNLayer constructed with kernel_size=" << kernel_size_
    //           << ", stride=" << stride_ << ", pad=" << pad_
    //           << ", in_channels=" << in_channels_ << ", out_channels=" << out_channels_ << std::endl; // ������Ϣ
}

// get_output_shape ����ʵ��
// ����������״������˳ߴ硢�����������������״
std::vector<int> Conv::get_output_shape(const std::vector<int>& input_shape) const {
    // ���� Tensor ������ 3D (ͨ��, �߶�, ���)
    if (input_shape.size() != 3) {
        throw std::invalid_argument("SimpleConvBNLayer expects 3D input shape [C, H, W].");
    }

    // �������ͨ�����Ƿ���ò�����������ͨ����ƥ��
    if (input_shape[0] != in_channels_) {
        throw std::invalid_argument("SimpleConvBNLayer: Input channels mismatch (" +
            std::to_string(input_shape[0]) + " != " + std::to_string(in_channels_) + ").");
    }

    int in_h = input_shape[1];
    int in_w = input_shape[2];

    // ��������߶ȺͿ��
    // ��ʽ: floor((����ߴ� + 2 * ��� - �˳ߴ�) / ����) + 1
    // ע�⣺��Ҫ���и�����ת���Ա������������ض�
    int H_out = static_cast<int>(std::floor(static_cast<float>(in_h + 2 * pad_ - kernel_size_) / stride_)) + 1;
    int W_out = static_cast<int>(std::floor(static_cast<float>(in_w + 2 * pad_ - kernel_size_) / stride_)) + 1;

    // �������������ߴ��Ƿ���Ч
    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("SimpleConvBNLayer: Output spatial dimensions are <= 0. Input shape might be too small for kernel/stride/padding.");
    }

    int out_c = out_channels_; // ���ͨ�������Ǹò�ʹ�õ��˲�������
    //cout << out_c << H_out << W_out << endl;

    return { out_c, H_out, W_out }; // ���ؼ�����������״
}


// forward ����ʵ��
// ������ Tensor ִ�о������
void Conv::forward(const Tensor& input, Tensor& output) {
    // 1. ���� Tensor ��״��� (ͨ���� get_output_shape �ڲ��Ѱ�����������ȷ��һ��)
    if (input.shape.size() != 3) {
        throw std::invalid_argument("SimpleConvBNLayer forward: Input tensor must be 3D [C, H, W].");
    }
    int in_c = input.shape[0];
    int in_h = input.shape[1];
    int in_w = input.shape[2];

    // �������ͨ�����Ƿ�ƥ��
    if (in_c != in_channels_) {
        throw std::invalid_argument("SimpleConvBNLayer forward: Input channels mismatch for forward pass.");
    }


    // 2. ȷ����� Tensor ��״�ʹ�С
    // ���� get_output_shape ���������״
    std::vector<int> output_shape = get_output_shape(input.shape);
    int out_c = output_shape[0];
    int out_h = output_shape[1];
    int out_w = output_shape[2];

    output.shape = output_shape; // ������� Tensor ����״
    output.data.resize(output.size()); // ������״���� output.data �Ĵ�С�������ڴ�


    // 3. ���ļ��㣺���
    // ������� Tensor ��ÿһ��λ�� [oc, oh, ow]
    for (int oc = 0; oc < out_c; ++oc) { // �������ͨ�� (��Ӧ�˲���)
        for (int oh = 0; oh < out_h; ++oh) { // ��������߶�
            for (int ow = 0; ow < out_w; ++ow) { // ����������

                float sum = 0.0f; // ��ʼ����ǰ���λ�õ��ۼ�ֵ

                // ���㵱ǰ���λ�� [oc, oh, ow] ��Ӧ������ Tensor �еľ�������������ʼ���� (���Ͻ�)
                // ���ǲ��������
                int ih_start = oh * stride_ - pad_;
                int iw_start = ow * stride_ - pad_;

                // --- �в�ѭ������������ͨ�� ---
                for (int ic = 0; ic < in_channels_; ++ic) { // ��������ͨ�� (�˲��������)

                    // --- �ڲ�ѭ������������˵����� ---
                    for (int kh = 0; kh < kernel_size_; ++kh) { // ��������˸߶�
                        for (int kw = 0; kw < kernel_size_; ++kw) { // ��������˿��

                            // ������������������ Tensor �е�ʵ������
                            int ih = ih_start + kh;
                            int iw = iw_start + kw;

                            // ��鵱ǰ���������Ƿ������������߽���
                            // ֻ������Ч�߽��ڵ����زŲ�����㣬������Ϊ0 (�������)
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                // �������� Tensor Ԫ��: input({ic, ih, iw})
                                // ����Ȩ�� Tensor Ԫ��: weights_({oc, ic, kh, kw})
                                sum += input({ ic, ih, iw }) * weights_({ oc, ic, kh, kw });
                            }
                            // ��� ih �� iw ���������� Tensor ��ʵ�ʱ߽� (���� padding �򴰿ڲ���������)��
                            // ��ô���ݾ���Ķ��壬���Ǳ���Ϊ�� 0�����Բ���Ҫ��������ʽ�� 0��
                        }
                    }
                }
                // ����ƫ���� (bias ��ÿ�����ͨ��һ��ֵ)
                sum += biases_.data[oc]; // biases_ �� 1D Tensor��ֱ�������� oc ����

                // ������������ Tensor �Ķ�Ӧλ��
                output({ oc, oh, ow }) = sum;
            }
        }
    }
}