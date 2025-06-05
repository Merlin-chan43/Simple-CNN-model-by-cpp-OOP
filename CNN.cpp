//
// Created by ������ on 2025/5/20
//

#include "CNN.h"
#include "opencv2/imgproc/types_c.h"

using namespace std;

Tensor CNN::load_image_as_tensor(const char* path)
{
    cv::Mat image = cv::imread(path);  // ʹ�ô����·������
    if (image.empty()) {
        std::cerr << "�޷�����ͼ��" << path << std::endl;
        throw std::runtime_error("Image load failed");  // ������Ĵ�����
    }

    // ת��Ϊ�������Ͳ���һ���� [0, 1]
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F, 1.0/255.0);

    // ȷ����������
    if (!floatImage.isContinuous()) {
        floatImage = floatImage.clone();
    }

    // ֱ��ʹ�� OpenCV ���ݳ�ʼ�� vector
    const int totalElements = floatImage.rows * floatImage.cols * floatImage.channels();
    std::vector<float> floatArray(
        reinterpret_cast<float*>(floatImage.data),
        reinterpret_cast<float*>(floatImage.data) + totalElements
    );
    int m_size = floatArray.size();
    vector <float> floatFinal;
    for (int i = 0; i <= 2; i++)
    {
        for (int j = i;j <= m_size - 1;j += 3)
        {
            floatFinal.push_back(floatArray[j]);
        }
    }
    Tensor temp;
    temp.shape = { floatImage.channels(), floatImage.rows, floatImage.cols };
    temp.data = std::move(floatFinal);
    // ���� Tensor�����蹹�캯������ vector ��ά�Ȳ�����
    return temp;
}

void CNN::add_layer(shared_ptr<layer> Layer)
{
    layers.push_back(Layer);
}

Tensor CNN::predict(Tensor& input)
{
    Tensor current_tensor_input = input;
    for (auto m_layer : layers)
    {
        Tensor current_tensor_output;

        m_layer->forward(current_tensor_input, current_tensor_output);//ÿһ���forward�������������Թ��ڴ˲��ٽ��С�
        //cout << current_tensor_output.data[1] << endl;
        current_tensor_input = current_tensor_output;
    }
    return current_tensor_input;
}
