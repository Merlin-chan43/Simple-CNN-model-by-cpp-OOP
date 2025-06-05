//
// Created by ������ on 2025/5/21
//
#ifndef CONV_LAYER_H // ͷ�ļ�������
#define CONV_LAYER_H

#include "layer.h"  // ���� Layer ����Ķ���
#include "Tensor.h" // ���� Tensor �ṹ�Ķ���
#include <vector>   // ���� std::vector

// --- ����������� ---
// �̳��� Layer��ʵ�־���㹦�� (�ں��� BN ����)
class Conv : public layer { // ���������۱���һ��
private:
    Tensor weights_;    // Ȩ�� Tensor����״ {out_channels, in_channels, kernel_size, kernel_size}
    Tensor biases_;     // ƫ�� Tensor����״ {out_channels}
    int pad_;           // ����С (�������߶ȺͿ�ȷ��������ͬ)
    int stride_;        // ���� (�������߶ȺͿ�ȷ��򲽳���ͬ)
    int kernel_size_;   // ����˱߳� (��������Ƿ��κ�)
    int in_channels_;   // ����ͨ���� (��ʽ�洢��Ҳ���� weights_.shape[1] �õ�)
    int out_channels_;  // ���ͨ���� (��ʽ�洢��Ҳ���� weights_.shape[0] �õ�)

public:
    // ���캯��������ԭʼȨ�غ�ƫ������ָ�뼰���б�Ҫ����
    Conv(int pad, int stride,int kernel_size, int out_channels, int in_channels,   const float* weights_data,
         const float* biases_data, int bias_size);

    // ʵ�ֻ����е� forward ����
    // ������ Tensor (3D ����ͼ) ִ�о�����㣬���������� Tensor
    void forward(const Tensor& input, Tensor& output) override;

    // ʵ�ֻ����е� get_output_shape ����
    // ����������״������˳ߴ硢�����������������״
    std::vector<int> get_output_shape(const std::vector<int>& input_shape) const override;

    // ��������
    ~Conv() override = default;
};

#endif // CONV_LAYER_H