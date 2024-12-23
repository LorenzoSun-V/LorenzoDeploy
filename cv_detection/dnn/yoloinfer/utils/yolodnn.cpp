#include "yolodnn.h"

YOLODNNModelManager::YOLODNNModelManager()
    : m_model_param{3, 640, 640, 1, 0.25f, 0.45f, 0.0f, 0.0f, false} {
    }

YOLODNNModelManager::~YOLODNNModelManager() {
}

bool YOLODNNModelManager::loadModel(const std::string model_name){
    struct stat buffer;
    if (!stat(model_name.c_str(), &buffer) == 0) {
        std::cerr << "Error: File " << model_name << " does not exist!" << std::endl;
        return false;
    }
    net = cv::dnn::readNetFromONNX(model_name);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Running on CPU with OpenCV backend" << std::endl;
    
    // 获取输入维度
    std::vector<cv::dnn::MatShape> inLayerShapes;
    std::vector<cv::dnn::MatShape> outLayerShapes;
    cv::dnn::MatShape emptyShape;
    net.getLayerShapes(emptyShape, 0, inLayerShapes, outLayerShapes); 
    m_model_param.batch_size = inLayerShapes[0][0];
    m_model_param.input_channel = inLayerShapes[0][1];
    m_model_param.input_height = inLayerShapes[0][2];
    m_model_param.input_width = inLayerShapes[0][3];
    std::cout << "batch_size: " << m_model_param.batch_size << " input_channel: " << m_model_param.input_channel << " input_height: " << m_model_param.input_height << " input_width: " << m_model_param.input_width << std::endl;
    return true;
}

cv::Mat YOLODNNModelManager::preprocess(cv::Mat img) {
    int col = img.cols;
    int row = img.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    img.copyTo(result(cv::Rect(0, 0, col, row)));
    m_model_param.x_factor = result.cols / static_cast<float>(m_model_param.input_width);
    m_model_param.y_factor = result.rows / static_cast<float>(m_model_param.input_height);
    return result;
}

void YOLODNNModelManager::doInference(cv::Mat input_data){
    cv::Mat blob;
    cv::dnn::blobFromImage(input_data, blob, 1.0/255.0, {m_model_param.input_width, m_model_param.input_height}, cv::Scalar(), true, false);
    net.setInput(blob);
}

void YOLODNNModelManager::postprocess(std::vector<DetBox>& detBoxs){
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    if (dimensions > rows){ // Check if the shape[2] is more than shape[1] (yolov8)
        m_model_param.yolov8 = true;
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];

        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }

    float *data = reinterpret_cast<float*>(outputs[0].data);
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i){
        if (m_model_param.yolov8){
            float *classes_scores = data + 4;
            int classes_size = dimensions - 4;
            cv::Mat scores(1, classes_size, CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > m_model_param.conf_thresh){
                confidences.emplace_back(maxClassScore);
                class_ids.emplace_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * m_model_param.x_factor);
                int top = int((y - 0.5 * h) * m_model_param.y_factor);
                int width = int(w * m_model_param.x_factor);
                int height = int(h * m_model_param.y_factor);

                boxes.emplace_back(cv::Rect(left, top, width, height));
            }
        }
        else {  // yolov5
        
            float confidence = data[4];
            int classes_size = dimensions - 5;
            if (confidence >= m_model_param.conf_thresh){
                float *classes_scores = data + 5;
                cv::Mat scores(1, classes_size, CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > m_model_param.conf_thresh){
                    confidences.emplace_back(confidence);
                    class_ids.emplace_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * m_model_param.x_factor);
                    int top = int((y - 0.5 * h) * m_model_param.y_factor);
                    int width = int(w * m_model_param.x_factor);
                    int height = int(h * m_model_param.y_factor);

                    boxes.emplace_back(cv::Rect(left, top, width, height));
                }
            }
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, m_model_param.conf_thresh, m_model_param.iou_thresh, nms_result);
    for (unsigned long i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];
        DetBox detBox;
        detBox.classID = class_ids[idx];
        detBox.confidence = confidences[idx];
        detBox.x = boxes[idx].x;
        detBox.y = boxes[idx].y;
        detBox.w = boxes[idx].width;
        detBox.h = boxes[idx].height;
        detBoxs.emplace_back(detBox);
    }
}

bool YOLODNNModelManager::inference(cv::Mat& frame, std::vector<DetBox>& detBoxs){
    if (frame.empty()) {
        std::cerr << "Input frame is empty." << std::endl;
        return false;
    }
    // 前处理
    cv::Mat input_data = frame;
    input_data = preprocess(frame);
    // 推理
    doInference(input_data);
    // 后处理
    postprocess(detBoxs);
    return true;
}