/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-21 14:19:07
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-08-22 14:13:54
 * @Description: YOLOv10模型前处理、推理、后处理代码
 */
#include "yolov10.h"

YOLOV10ModelManager::YOLOV10ModelManager()
    : conf(0.25) {
}

YOLOV10ModelManager::~YOLOV10ModelManager() {
}

// Load the model from the serialized engine file
bool YOLOV10ModelManager::loadModel(const std::string model_path) {
    auto model = core.read_model(model_path);
    compiled_model = core.compile_model(model, "CPU");
    infer_request = compiled_model.create_infer_request();
    auto inputDims = infer_request.get_input_tensor().get_shape();
    m_kBatchSize = inputDims[0];
    m_channel = inputDims[1];
    m_kInputH = inputDims[2];
    m_kInputW = inputDims[3];
    std::cout << "inputDims: " << inputDims[0] << " " << inputDims[1] << " " << inputDims[2] << " " << inputDims[3] << std::endl;
    return true;
}

void YOLOV10ModelManager::preprocess(cv::Mat* img, std::vector<float>& data) {
    data.clear();
    // 保持长宽比的resize
    cv::Mat mat;
    int rh = img->rows;
    int rw = img->cols;
    int rc = img->channels();
    cv::cvtColor(*img, mat, cv::COLOR_BGR2RGB);
    int max_image_length = rw > rh ? rw : rh;
    int length = m_kInputW > m_kInputH ? m_kInputW : m_kInputH;
    cv::Mat max_image = cv::Mat::zeros(max_image_length, max_image_length, CV_8UC3);
    cv::Rect roi(0, 0, rw, rh);
    mat.copyTo(cv::Mat(max_image, roi));
    cv::Mat resize_img;
    cv::resize(max_image, resize_img, cv::Size(m_kInputW, m_kInputH), 0.0f, 0.0f, cv::INTER_LINEAR);
    factor = (float)((float)max_image_length / (float)length);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.0);
    for (int i = 0; i < m_channel; ++i) {
        for (int j = 0; j < m_kInputH; ++j) {
            for (int k = 0; k < m_kInputW; ++k) {
                data.emplace_back(resize_img.at<cv::Vec3f>(j, k)[i]);
            }
        }
    }
}

bool YOLOV10ModelManager::postProcess(float* output_data, std::vector<DetBox>& detBoxs) {
    if(NULL == output_data) {
        std::cerr << "result data is NULL." << std::endl;
        return false;
    }
    // Preprocessing output results
    for (int i = 0; i < kOutputSize; i++){
        float confidence = output_data[i * 6 + 4];
        if (confidence > conf) {
            float xmin = output_data[i * 6 + 0] * factor;
            float ymin = output_data[i * 6 + 1] * factor;
            float xmax = output_data[i * 6 + 2] * factor;
            float ymax = output_data[i * 6 + 3] * factor; 
            xmin = std::max(0.0f, xmin);
            ymin = std::max(0.0f, ymin);
            int label_id = static_cast<int>(output_data[i * 6 + 5]);

            DetBox box;
            box.x = xmin;
            box.y = ymin;
            box.w = xmax - xmin;
            box.h = ymax - ymin;
            box.confidence = confidence;
            box.classID = label_id;
            detBoxs.emplace_back(box);
        }
    }
    return true;
}

bool YOLOV10ModelManager::doInference(cv::Mat img, std::vector<DetBox>& detBoxs){
    preprocess(&img, input_data);

    input_port = compiled_model.input();
    input_tensor = ov::Tensor(input_port.get_element_type(), input_port.get_shape(), input_data.data());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    output_data = infer_request.get_output_tensor(0).data<float>();
    if (!postProcess(output_data, detBoxs)){
        return false;
    }

    return true;
}

// Perform inference on the input frame
bool YOLOV10ModelManager::inference(cv::Mat frame, std::vector<DetBox>& detBoxs) {
    if (!doInference(frame, detBoxs)) {
        return false;
    }
    return true;
}
