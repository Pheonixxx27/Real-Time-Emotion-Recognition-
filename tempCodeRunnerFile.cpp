#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void loadModel(cv::dnn::Net& net) {
    String modelArchitecture = "facialemotionmodel.json";
    String modelWeights = "facialemotionmodel.h5";
    net = cv::dnn::readNetFromTensorflow(modelArchitecture, modelWeights);
}

cv::Mat extractFeatures(cv::Mat image) {
    cv::Mat feature;
    cv::resize(image, feature, cv::Size(48, 48));
    feature.convertTo(feature, CV_32F);
    feature /= 255.0;
    return feature;
}

int main() {
    cv::VideoCapture webcam(0);
    if (!webcam.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    cv::dnn::Net model;
    loadModel(model);

    cv::CascadeClassifier faceCascade;
    std::string haarFile = cv::samples::findFile("haarcascade_frontalface_default.xml");
    if (!faceCascade.load(haarFile)) {
        std::cerr << "Error: Could not load Haar cascade." << std::endl;
        return -1;
    }

    std::map<int, std::string> labels = {{0, "angry"}, {1, "disgust"}, {2, "UNSAFE"}, {3, "SAFE"}, {4, "neutral"}, {5, "UNSAFE"}, {6, "surprise"}};

    while (true) {
        cv::Mat frame;
        webcam.read(frame);
        if (frame.empty()) {
            std::cerr << "Error: Empty frame." << std::endl;
            break;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.3, 5);

        for (const auto& face : faces) {
            cv::Mat faceROI = gray(face);
            cv::resize(faceROI, faceROI, cv::Size(48, 48));

            cv::Mat blob = cv::dnn::blobFromImage(faceROI, 1.0, cv::Size(48, 48), cv::Scalar(0, 0, 0), false, false);
            model.setInput(blob);
            cv::Mat pred = model.forward();

            cv::Point textOrg(face.x - 10, face.y - 10);
            int labelId = cv::minMaxLoc(pred).maxLoc.x;
            std::string predictionLabel = labels[labelId];

            cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);
            cv::putText(frame, predictionLabel, textOrg, cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(0, 0, 255));
        }

        cv::imshow("Output", frame);

        if (cv::waitKey(27) == 27) {
            break;
        }
    }

    return 0;
}
