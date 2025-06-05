# Simple CNN model
## 1. Project Implementation Overview

This document details the implementation of a Convolutional Neural Network (CNN) project, designed for a binary classification task. The project is structured using Object-Oriented Programming (OOP) principles, where each component of the neural network is represented by a dedicated C++ class. This modular approach facilitates understanding, maintainability, and extensibility of the network architecture.

The core components and their functionalities are outlined as follows:

### 1.1 Data Representation: `Tensor`

The fundamental data structure used throughout the network is the `Tensor`. Represented by `Tensor.h` and `Tensor.cpp`, this custom class is designed to efficiently handle multi-dimensional numerical data. Internally, it stores all elements in a contiguous `std::vector<float>`, optimizing memory access. Its primary features include:

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
- **`SoftMax` (SoftMax.h, SoftMax.cpp):** Transforms a vector of raw scores (logits) into a probability distribution. The output values are in the range (0, 1) and sum to 1, making it ideal for the final classification layer.
- **`MaxPooling` (MaxPooling.h, MaxPooling.cpp):** Performs down-sampling by selecting the maximum value within a sliding window over the input feature map. It reduces the spatial dimensions (height and width) of the input while retaining the number of channels, providing translation invariance.
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

- **Parameter Loading:** Accesses the pre-trained model weights and biases, which are defined as global arrays in `face_binary_cls.cpp` and declared via `model_weights.h`.
- **Network Assembly:** Instantiates the `CNN` class and dynamically creates instances of each concrete layer (`Conv`, `Relu`, `MaxPooling`, `Flatten`, `fc_layer`, `SoftMax`), passing the loaded weights and biases to their respective constructors where applicable. These layers are then added to the `CNN` object in the correct architectural sequence.
- **Image Processing and Prediction:** Utilizes the `CNN::load_image_as_tensor` method to load and prepare input images (`man.jpg`, `plane.jpg`). It then invokes the `CNN::predict` method to perform the forward pass, obtaining the classification probabilities.
- **Result Interpretation:** Interprets the final output `Tensor` (the Softmax probabilities) to determine and display the prediction (face or background).
- **Resource Management:** Ensures proper cleanup and deallocation of all dynamically created resources before program termination.
