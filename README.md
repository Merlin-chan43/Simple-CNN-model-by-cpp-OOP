# Simple CNN model
## 1. Project Implementation Overview

This document details the implementation of a Convolutional Neural Network (CNN) project, designed for a binary classification task. The project is structured using Object-Oriented Programming (OOP) principles, where each component of the neural network is represented by a dedicated C++ class. This modular approach facilitates understanding, maintainability, and extensibility of the network architecture.

The core components and their functionalities are outlined as follows:

### 1.1 Data Representation: `Tensor`

  The fundamental data structure used throughout the network is the `Tensor`. Represented solely by `Tensor.h`, this custom class is designed to efficiently handle multi-dimensional numerical data. Its implementation, including all method definitions, is entirely contained within `Tensor.h`, providing a self-contained data handling unit. Internally, it stores all elements in a contiguous `std::vector<float>`, optimizing memory access. Its primary features include:

  - **Multi-dimensional Storage:** Capable of representing scalars (0D), vectors (1D), matrices (2D), and higher-dimensional data (e.g., 3D for image feature maps [channels, height, width]).
  - **Shape Management:** A `std::vector<int>` stores the dimensions of the tensor, allowing flexible shape manipulation.
  - **Linear Indexing:** Provides methods to convert multi-dimensional coordinates (e.g., `{c, h, w}`) into a single linear index for efficient access to the underlying `std::vector<float>` data. This is crucial for correctly mapping conceptual multi-dimensional operations to linear memory.

### 1.2 Abstract Base Layer: `Layer`

Defined in `layer.h`, the `Layer` class serves as an abstract base class for all operational layers within the CNN. It establishes a common interface that all concrete layers must adhere to, enabling polymorphic behavior. Key elements include:

- **`forward` Method:** A pure virtual function (`virtual void forward(const Tensor& input, Tensor& output) = 0;`) that dictates every concrete layer must implement its specific forward propagation logic. This method takes an input `Tensor` and computes its output, storing the result in an `output Tensor`.
- **`get_output_shape` Method:** A pure virtual function (`virtual std::vector<int> get_output_shape(const std::vector<int>& input_shape) const = 0;`) designed to calculate and return the expected output shape of a layer given its input shape. This is vital for network validation and memory pre-allocation.
- **Virtual Destructor:** Ensures proper memory deallocation for derived class objects when managed through base class pointers.

### 1.3 Concrete Layer Implementations

Building upon the `Layer` abstract base class, specific operational layers of the CNN are implemented. Each class encapsulates the unique mathematical transformations and parameter handling for its respective layer type. These implementations bridge the gap between abstract definitions and practical computations.

- **`Relu` (Relu.h, Relu.cpp):** Implements the Rectified Linear Unit activation function (f(x)=max(0,x)). It performs an element-wise non-linear transformation without altering the input tensor's shape.
- **`Flatten` (flatten.h, flatten.cpp):** Converts a multi-dimensional input tensor (e.g., a 3D feature map) into a one-dimensional vector. This layer reshapes the data to be compatible with subsequent fully connected layers without changing the actual data values or their linear order.
- **`SoftMax` (softMax.h, softMax.cpp):** Transforms a vector of raw scores (logits) into a probability distribution. The output values are in the range (0, 1) and sum to 1, making it ideal for the final classification layer.
- **`MaxPooling` (maxPooling.h, maxPooling.cpp):** Performs down-sampling by selecting the maximum value within a sliding window over the input feature map. It reduces the spatial dimensions (height and width) of the input while retaining the number of channels, providing translation invariance.
- **`fc_layer` (fc_layer.h, fc_layer.cpp):** Implements the fully connected layer, performing a linear transformation (Y=W⋅X+B). It involves matrix multiplication of the input vector with a learnable weight matrix and the addition of a bias vector. This layer has trainable parameters (weights and biases) that are loaded from pre-trained data.
- **`Conv` (Conv.h, Conv.cpp):** Implements the convolutional layer, the core feature extraction component of a CNN. It applies learnable filters (kernels) that slide across the input, performing dot products to produce feature maps. This implementation also handles padding and stride, and implicitly incorporates Batch Normalization parameters that are fused with the convolution weights.

### 1.4 Network Orchestration: `CNN`

The `CNN` class (CNN.h, CNN.cpp) acts as the central orchestrator of the entire neural network. It encapsulates the sequence of concrete `Layer` objects and manages the overall forward pass.

- **Layer Management:** Stores dynamically allocated `Layer` objects in a `std::vector<Layer*>`, preserving the architectural sequence of the network.
- **`add_layer` Method:** Provides an interface for adding individual `Layer` instances to the network's processing pipeline.
- **`predict` Method:** Orchestrates the sequential execution of forward propagation through all added layers. It takes the initial network input `Tensor` (e.g., pre-processed image data) and passes it through each layer, using the output of one layer as the input for the next, ultimately returning the final prediction `Tensor`.
- **`load_image_as_tensor` Method:** Facilitates the initial data preparation by loading an image file, resizing it, normalizing pixel values, and transforming its dimensions (`HWC` to `CHW`) into a suitable `Tensor` format for the network's input.
- **Memory Management:** The destructor ensures proper deallocation of all dynamically created `Layer` objects added to the network, preventing memory leaks.

### 1.5 Entry Point and Model Initialization: `main.cpp`

The `main.cpp` file serves as the application's entry point, handling the overall program flow. It orchestrates the initialization of the CNN model, the loading of pre-trained parameters, and the execution of the prediction process.

- **Parameter Definition and Loading:** The pre-trained model weights and biases are directly defined as global arrays within `main.cpp`. This consolidates the model's numerical parameters alongside the main application logic, making them immediately accessible for network assembly.
- **Network Assembly:** Instantiates the `CNN` class and dynamically creates instances of each concrete layer (`Conv`, `Relu`, `MaxPooling`, `Flatten`, `fc_layer`, `SoftMax`), passing the loaded weights and biases to their respective constructors where applicable. These layers are then added to the `CNN` object in the correct architectural sequence.
- **Image Processing and Prediction:** Utilizes the `CNN::load_image_as_tensor` method to load and prepare input images (`man.jpg`, `plane.jpg`). It then invokes the `CNN::predict` method to perform the forward pass, obtaining the classification probabilities.
- **Result Interpretation:** Interprets the final output `Tensor` (the Softmax probabilities) to determine and display the prediction (face or background).
- **Resource Management:** Ensures proper cleanup and deallocation of all dynamically created resources before program termination.
## 2. Development Challenges and Solutions

During the development of this CNN project, our team encountered several significant challenges, primarily related to data handling and inter-module communication. Addressing these issues was crucial for achieving a correctly functioning model.

### 2.1 Parameter Mismatch in Layer Initialization

**Challenge:** One developer implemented the `fc_layer` class, designed to receive pre-trained weights and biases through its constructor. Concurrently, another developer was responsible for the `main.cpp` file, which orchestrates the network assembly and passes these parameters to the layer constructors. A critical mismatch occurred in the interpretation and passing of parameter dimensions. The `fc_layer` was designed to expect a specific `[out_features, in_features]` shape for its weight matrix, but the `main.cpp` was incorrectly providing or interpreting the dimensions, leading to runtime errors during the layer's initialization or during the forward pass due to incorrect tensor shapes and memory access. This was often manifested as `std::invalid_argument` exceptions related to tensor size or shape mismatches.

**Solution:** The resolution involved a detailed collaborative debugging session to meticulously cross-reference the `fc_layer`'s constructor signature and its internal `Tensor` shape expectations with how the parameters were being retrieved and passed from `main.cpp`. This included:

- **Verifying `fc_layer.h` and `fc_layer.cpp`:** Ensuring the `fc_layer` constructor correctly initialized `weights_` and `biases_` with their intended `[out_features, in_features]` and `[out_features]` shapes, respectively, by properly setting the `Tensor::shape` member and resizing `Tensor::data`.
- **Tracing `main.cpp`'s Parameter Retrieval:** Confirming that the `in_features` and `out_features` values extracted from `conv_params` and `fc_params` (defined in `main.cpp`) precisely matched the dimensions required by the `fc_layer` constructor.
- **Ensuring Correct Indexing:** Double-checking that when passing the `fc0_weight` and `fc0_bias` raw arrays, the `out_features` and `in_features` values were correctly derived and supplied in the constructor's argument list, aligning with the `fc_layer`'s internal `weights_.shape[0]` and `weights_.shape[1]` logic. This direct comparison and correction of parameter values and their interpretation across both modules resolved the initialization failures.

### 2.2 Incorrect Data Ordering in OpenCV to Tensor Conversion

**Challenge:** A significant problem arose during the initial data preparation phase, specifically when converting image data loaded via OpenCV into the custom `Tensor` format. The intended order for the `Tensor` was `[Channel, Height, Width]` (CHW), where all pixels of the first channel are stored contiguously, followed by all pixels of the second channel, and so on. However, the implemented conversion code mistakenly adopted a different internal iteration order, effectively converting the data into a `[Height, Width, Channel]` (HWC) or `[Height, Channel, Width]` (HCW) like representation but flattened into a CHW structure. This meant that after processing the first pixel of the first channel, the code immediately processed the first pixel of the second channel (at the same spatial `(h, w)` location), rather than moving to the next `(h, w)` location within the *same* channel. This subtle ordering error propagated incorrect data through the network, leading to nonsensical prediction results.

**Solution:** The issue was identified through careful debugging of the `CNN::load_image_as_tensor` method and visual inspection of intermediate `Tensor` data. The fix involved adjusting the nested loops responsible for copying pixel data from the OpenCV `cv::Mat` into the `Tensor`'s `data` vector.

- **Understanding OpenCV's Layout:** Acknowledging that `cv::Mat` typically stores pixel data in a `[Height, Width, Channel]` (HWC) layout.

- Correcting Loop Order and Linear Index Calculation:

   The nested loops were re-ordered to explicitly iterate through 

  `Channel`, then`Height`, then`Width`, and the linear index calculation within`Tensor::data`was adjusted to match this CHW target order:

  C++

  ```
  // Corrected logic within CNN::load_image_as_tensor
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
  ```

This corrected indexing ensured that the image data was correctly arranged in the CHW format within the `Tensor`'s linear `data` array, allowing subsequent convolutional layers to interpret and process the features accurately.

## 3. Project Summary and Reflections

This project has been a deeply insightful journey into the practical implementation of Convolutional Neural Networks (CNNs) from fundamental principles. By building a forward propagation pipeline in C++, we gained a comprehensive understanding of the intricate mechanisms that power modern deep learning models.

One of the most significant takeaways has been the profound importance of **data representation and consistency**. The `Tensor` class, serving as the universal data carrier, underscored how crucial a well-designed data structure is for efficient computation and seamless data flow between diverse layers. The challenges encountered during OpenCV image conversion vividly highlighted that the *order* in which multi-dimensional data is flattened into a one-dimensional array is not merely an implementation detail but fundamentally affects the network's ability to interpret features correctly. Misalignments, such as the `[Height, Width, Channel]` (HWC) versus `[Channel, Height, Width]` (CHW) order, led to immediate and significant prediction inaccuracies, emphasizing the need for meticulous attention to data layout from the input pipeline onward.

Furthermore, the experience elucidated the power and necessity of **Object-Oriented Programming (OOP) in complex system design**. The `Layer` abstract base class provided a robust blueprint, enforcing a common interface (`forward` and `get_output_shape`) that enabled polymorphic behavior. This design pattern allowed the `CNN` class to orchestrate the entire network's forward pass by simply iterating through a collection of `Layer*` pointers, calling the generic `forward` method, without needing to know the specific type of each concrete layer. This modularity not only made the `CNN` class cleaner and easier to manage but also significantly improved the extensibility of the network, allowing for easy addition or modification of layer types without altering the core network execution logic.

The collaborative aspect of the project, though presenting initial hurdles like parameter mismatches between `fc_layer` and `main.cpp`, proved invaluable. These challenges underscored the critical importance of **clear communication and strict adherence to defined interfaces** within a team. Differences in interpretation of parameter meanings (e.g., `in_features` vs. `out_features` or specific array indexing conventions) can halt progress until meticulously resolved through joint debugging and verification of design choices. This collaborative debugging process also reinforced the importance of unit testing individual layer implementations to isolate and identify errors more quickly.

In conclusion, this project served as an excellent practical exercise, solidifying theoretical knowledge of CNNs with hands-on C++ implementation. It not only deepened our understanding of core deep learning concepts like convolution, pooling, and activation functions but also provided invaluable lessons in robust software engineering principles, data integrity, and effective team collaboration. The successful implementation of this forward propagation pipeline lays a strong foundation for future exploration into more complex CNN architectures, backpropagation, and model training.
