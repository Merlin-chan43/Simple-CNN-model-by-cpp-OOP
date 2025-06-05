//
// Created by 戴烨铭 on 2025/5/20
//

#include "CNN.h"
#include "opencv2/imgproc/types_c.h"

using namespace std;

Tensor CNN::load_image_as_tensor(const char* path)
{
    cv::Mat image = cv::imread(path);  // 使用传入的路径参数
    if (image.empty()) {
        std::cerr << "无法加载图像：" << path << std::endl;
        throw std::runtime_error("Image load failed");  // 更合理的错误处理
    }

    // 转换为浮点类型并归一化到 [0, 1]
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F, 1.0/255.0);

    // 确保数据连续
    if (!floatImage.isContinuous()) {
        floatImage = floatImage.clone();
    }

    // 直接使用 OpenCV 数据初始化 vector
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
    // 返回 Tensor（假设构造函数接受 vector 和维度参数）
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

        m_layer->forward(current_tensor_input, current_tensor_output);//每一层的forward方法里已做断言故在此不再进行。
        //cout << current_tensor_output.data[1] << endl;
        current_tensor_input = current_tensor_output;
    }
    return current_tensor_input;
}
