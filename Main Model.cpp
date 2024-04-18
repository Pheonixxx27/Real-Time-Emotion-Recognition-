#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>

// Define constants
#define MODEL_PATH "emotiondetector.pb"
#define LABELS_FILE "emotion_labels.txt"

// Function to load TensorFlow model from a file
TF_Buffer* ReadBinaryFile(const char* file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    auto buffer = TF_NewBufferFromString(nullptr, file_size);
    if (!buffer) {
        std::cerr << "Error allocating buffer for file: " << file_path << std::endl;
        return nullptr;
    }

    file.read((char*)TF_TensorData(buffer), file_size);
    file.close();

    TF_Buffer* result = TF_NewBuffer();
    result->data = TF_TensorData(buffer);
    result->length = TF_TensorByteSize(buffer);
    result->data_deallocator = [](void* data, size_t length, void* arg) {
        TF_DeleteBuffer(static_cast<TF_Buffer*>(arg));
    };
    result->arg = buffer;

    return result;
}

// Function to load emotion labels
std::vector<std::string> LoadLabels(const char* labels_file) {
    std::ifstream file(labels_file);
    std::vector<std::string> labels;
    std::string label;
    while (std::getline(file, label)) {
        labels.push_back(label);
    }
    file.close();
    return labels;
}

// Function to preprocess image
cv::Mat PreprocessImage(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error loading image: " << image_path << std::endl;
        return cv::Mat();
    }

 //Other preprocessing model also to be performed here

    // Return the preprocessed image
    return image;
}

// Function to convert image data to TensorFlow tensor
TF_Tensor* ImageToTensor(const cv::Mat& image) {
    int64_t dims[4] = {1, image.rows, image.cols, 1};
    std::vector<float> input_data(image.rows * image.cols);

    // Normalize and copy image data to input tensor
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            input_data[i * image.cols + j] = image.at<uchar>(i, j) / 255.0f;
        }
    }

    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dims, 4, input_data.data(), sizeof(float) * input_data.size(), nullptr, nullptr);
    return input_tensor;
}

// Function to perform inference using TensorFlow session
std::string PerformInference(TF_Session* session, TF_Tensor* input_tensor, std::vector<std::string>& labels) {
    // Get input and output tensors from the session
    TF_Output inputs = {TF_GraphOperationByName(TF_GraphOperationByName(TF_SessionGraph(session), "input"), "input_image"), 0};
    TF_Output outputs = {TF_GraphOperationByName(TF_SessionGraph(session), "output"), 0};

    TF_Tensor* output_tensor = nullptr;

    // Run the session to perform inference
    TF_SessionRun(session, nullptr, &inputs, &input_tensor, 1, &outputs, &output_tensor, 1, nullptr, 0, nullptr, TF_SessionRunOptions(), TF_NewStatus());

    if (!output_tensor) {
        std::cerr << "Error getting output tensor during inference." << std::endl;
        return "Unknown";
    }

    // Interpret output probabilities and determine predicted emotion
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    int output_size = TF_TensorByteSize(output_tensor) / sizeof(float);
    int max_index = std::distance(output_data, std::max_element(output_data, output_data + output_size));
    std::string predicted_emotion = labels[max_index];

    TF_DeleteTensor(output_tensor);
    return predicted_emotion;
}

// Function to perform emotion detection on an image
std::string EmotionDetection(const std::string& image_path, TF_Session* session, std::vector<std::string>& labels) {
    cv::Mat image = PreprocessImage(image_path);
    if (image.empty()) {
        return "Unknown";
    }

    TF_Tensor* input_tensor = ImageToTensor(image);
    if (!input_tensor) {
        std::cerr << "Error converting image to TensorFlow tensor." << std::endl;
        return "Unknown";
    }

    std::string predicted_emotion = PerformInference(session, input_tensor, labels);
    TF_DeleteTensor(input_tensor);

    return predicted_emotion;
}

int main() {
    // Initialize TensorFlow session
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* session_options = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, session_options, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error creating TensorFlow session." << std::endl;
        return 1;
    }

    // Load TensorFlow model
    TF_Buffer* model_buffer = ReadBinaryFile(MODEL_PATH);
    if (!model_buffer) {
        std::cerr << "Error loading TensorFlow model." << std::endl;
        return 1;
    }

    TF_ImportGraphDefOptions* import_options = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, model_buffer, import_options, status);
    TF_DeleteImportGraphDefOptions(import_options);
    TF_DeleteBuffer(model_buffer);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error importing TensorFlow model." << std::endl;
        return 1;
    }

    // Load emotion labels
    std::vector<std::string> labels = LoadLabels(LABELS_FILE);

    // Perform emotion detection on an image
    std::string image_path = "path_to_your_image.jpg"; // Replace with your image path
    std::string predicted_emotion = EmotionDetection(image_path, session, labels);

    std::cout << "Predicted emotion: " << predicted_emotion << std::endl;

    // Cleanup TensorFlow resources
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    return 0;
}
