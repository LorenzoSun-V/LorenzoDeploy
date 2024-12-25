/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-21 14:19:07
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-24 12:00:00
 * @Description: YOLOv8OBB模型前处理、推理、CUDA加速后处理代码
 */

#include "yolov8obb.h"
#include "postprocess.h"
#include <cuda.h>
#include <cuda_runtime.h>

// Constructor
YOLOV8OBBModel::YOLOV8OBBModel()
    : m_trt{ Logger(),  nullptr, nullptr, nullptr, nullptr},
      m_kOutputSize(0),  
      m_kInputSize(0),   
      kMaxInputImageSize(9000 * 9000),     
      inputSrcDevice(nullptr), outputSrcDevice(nullptr) {
}

// Destructor
YOLOV8OBBModel::~YOLOV8OBBModel() {
    if (inputSrcDevice) {
        cudaFree(inputSrcDevice);
        inputSrcDevice = nullptr;
    }
    if (outputSrcDevice) {
        cudaFree(outputSrcDevice);
        outputSrcDevice = nullptr;
    }
    if (m_trt.context) {
        m_trt.context->destroy();
        m_trt.context = nullptr;
    }
    if (m_trt.engine) {
        m_trt.engine->destroy();
        m_trt.engine = nullptr;
    }
    if (m_trt.runtime) {
        m_trt.runtime->destroy();
        m_trt.runtime = nullptr;
    }
    if (m_trt.stream) {
        cudaStreamDestroy(m_trt.stream);
        m_trt.stream = nullptr;
    }
    cuda_preprocess_destroy();
}

// Load the model from the serialized engine file
bool YOLOV8OBBModel::loadModel(const std::string engine_name) {
    struct stat buffer;
    if (stat(engine_name.c_str(), &buffer) != 0) {
        std::cerr << "Error: File " << engine_name << " does not exist!" << std::endl;
        return false;
    }

    if (!deserializeEngine(engine_name)) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        return false;
    }

    if(m_trt.engine->getNbBindings() != 2){
        std::cerr << "Please input correct network model." << std::endl;
        return false;
    }
    
    auto inputDims = m_trt.engine->getBindingDimensions(0);

    m_model.batch_size = inputDims.d[0];
    m_model.input_channel = inputDims.d[1];
    m_model.input_height = inputDims.d[2];
    m_model.input_width = inputDims.d[3];

    std::cout << "m_kBatchSize: " << m_model.batch_size
             << " m_channel: " << m_model.input_channel 
             << " m_kInputH: " << m_model.input_height 
             << " m_kInputW: " <<  m_model.input_width
             << std::endl;
             
    auto out_dims = m_trt.engine->getBindingDimensions(1);
    
    m_model.num_classes = out_dims.d[2] - 5;
    m_model.num_bboxes = out_dims.d[1];
    m_model.bbox_element = out_dims.d[2];

    std::cout << "Output size parameters: "
            << "batch_size=" << m_model.batch_size
            << ", num_bboxes=" << m_model.num_bboxes
            << ", bbox_element=" << m_model.bbox_element
            << std::endl;

    if (cudaStreamCreate(&m_trt.stream) != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream." << std::endl;
        return false;
    }

    cuda_preprocess_init(kMaxInputImageSize);
    m_kInputSize = m_model.batch_size * m_model.input_channel * m_model.input_height * m_model.input_width;
    if (cudaMalloc((void**)&inputSrcDevice, m_kInputSize * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input." << std::endl;
        return false;
    }
    
    m_kOutputSize = m_model.batch_size * m_model.num_bboxes *  m_model.bbox_element;
    if (cudaMalloc((void**)&outputSrcDevice,  m_kOutputSize * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for output." << std::endl;
        return false;
    }

    inputData.resize(m_kInputSize);
    output_data.resize(m_kOutputSize);

    return true;
}

// Deserialize the engine from file
bool YOLOV8OBBModel::deserializeEngine(const std::string engine_name) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return false;
    }
    
    cudaSetDevice(0);
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    file.read(serialized_engine, size);
    file.close();

    m_trt.runtime = createInferRuntime(m_trt.gLogger);
    if (m_trt.runtime == nullptr) {
        std::cerr << "Failed to create Infer Runtime." << std::endl;
        goto FAILED;
    }

    m_trt.engine = m_trt.runtime->deserializeCudaEngine(serialized_engine, size);
    if (m_trt.engine == nullptr) {
        std::cerr << "Failed to deserialize CUDA engine." << std::endl;
        goto FAILED;
    }  

    m_trt.context = m_trt.engine->createExecutionContext();
    if (m_trt.context == nullptr) {
        std::cerr << "Failed to create execution context." << std::endl;
        goto FAILED;
    }

    delete[] serialized_engine;
    return true;

FAILED: 
    delete[] serialized_engine;
    return false;
}

// Perform inference with CUDA-based post-processing
bool YOLOV8OBBModel::doInference(std::vector<cv::Mat> batch_images, std::vector<std::vector<DetBox>>& batch_result) 
{
    if(batch_images.size() != m_model.batch_size) {
        std::cerr << "Batch size mismatch." << std::endl;
        return false;
    }

    // Preprocess and copy input to device
    cuda_batch_preprocess(batch_images, inputSrcDevice, m_model.input_width, m_model.input_height, m_trt.stream);
    void* bindings[] = { inputSrcDevice, outputSrcDevice };
    if (!m_trt.context->enqueueV2(bindings, m_trt.stream, nullptr)) {
        std::cerr << "Failed to enqueue inference." << std::endl;
        return false;
    }

    // Perform CUDA-based post-processing
    // Define maximum number of objects per image
    const int max_objects = m_model.max_objects;

    // Allocate device memory for decoded boxes and NMS output
    // Each image will have max_objects decoded boxes
    float* d_parray;
    size_t parray_size = m_model.batch_size * max_objects * sizeof(DecodedBBox);
    if (cudaMalloc(&d_parray, parray_size) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for decoded boxes." << std::endl;
        return false;
    }

    // Launch decode and NMS for each image in the batch
    for (int i = 0; i < m_model.batch_size; ++i) {
        float* predict = outputSrcDevice + i * m_model.num_bboxes * m_model.bbox_element;
        float* parray = d_parray + i * max_objects * sizeof(DecodedBBox); // Adjust pointer for each image

        cuda_decode_and_nms_obb(
            predict,
            m_model.num_bboxes,
            m_model.num_classes,
            m_model.conf_thresh,
            m_model.iou_thresh,
            parray,
            max_objects,
            m_trt.stream
        );
    }

    // Allocate host memory to retrieve results
    std::vector<DecodedBBox> h_decoded_boxes(m_model.batch_size * max_objects);
    if (cudaMemcpyAsync(
            h_decoded_boxes.data(),
            d_parray,
            parray_size,
            cudaMemcpyDeviceToHost,
            m_trt.stream) != cudaSuccess) {
        std::cerr << "Failed to copy decoded boxes from device to host." << std::endl;
        cudaFree(d_parray);
        return false;
    }

    // Synchronize the stream to ensure all operations are complete
    if (cudaStreamSynchronize(m_trt.stream) != cudaSuccess) {
        std::cerr << "Failed to synchronize CUDA stream." << std::endl;
        cudaFree(d_parray);
        return false;
    }

    // Post-process to convert DecodedBBox to DetBox and organize into batch_result
    batch_result.resize(m_model.batch_size);
    for (int i = 0; i < m_model.batch_size; ++i) {
        for (int j = 0; j < max_objects; ++j) {
            int idx = i * max_objects + j;
            const DecodedBBox& box = h_decoded_boxes[idx];
            if (box.confidence > 0.0f) { // Box is kept after NMS
                // Transform coordinates back to original image space
                DetBox det;
                float r_w = static_cast<float>(m_model.input_width) / static_cast<float>(batch_images[i].cols);
                float r_h = static_cast<float>(m_model.input_height) / static_cast<float>(batch_images[i].rows);
                float image_y_pad = (static_cast<float>(m_model.input_height) - r_w * static_cast<float>(batch_images[i].rows)) / 2.0f;
                float image_x_pad = (static_cast<float>(m_model.input_width) - r_h * static_cast<float>(batch_images[i].cols)) / 2.0f;

                float origin_x1 = box.center_x - box.w / 2.0f;
                float origin_y1 = box.center_y - box.h / 2.0f;
                float origin_x2 = box.center_x + box.w / 2.0f;
                float origin_y2 = box.center_y + box.h / 2.0f;

                float x1, y1, x2, y2;
                if (r_h > r_w) {
                    x1 = origin_x1 / r_w;
                    x2 = origin_x2 / r_w;
                    y1 = (origin_y1 - image_y_pad) / r_w;
                    y2 = (origin_y2 - image_y_pad) / r_w;
                } else {
                    x1 = (origin_x1 - image_x_pad) / r_h;
                    x2 = (origin_x2 - image_x_pad) / r_h;
                    y1 = origin_y1 / r_h;
                    y2 = origin_y2 / r_h;
                }

                det.x = x1;
                det.y = y1;
                det.w = x2 - x1;
                det.h = y2 - y1;
                det.confidence = box.confidence;
                det.classID = box.class_id;
                det.radian = box.radian;

                batch_result[i].emplace_back(det);
            }
        }
    }

    // Free device memory
    cudaFree(d_parray);

    return true;
}

// Single image inference using CUDA-based post-processing
bool YOLOV8OBBModel::inference(cv::Mat frame, std::vector<DetBox>& result) 
{
    if(frame.empty() ){
        std::cerr << "Input image data is empty." << std::endl;
        return false;
    }
    std::vector<cv::Mat> batch_images;
    batch_images.push_back(frame);

    std::vector<std::vector<DetBox>> batch_result;
    if (!doInference(batch_images, batch_result)) {
        return false;
    }

    if(batch_result.empty() || batch_result[0].empty()) {
        return false;
    }

    result = batch_result[0];
    return true;
}

// Batch inference using CUDA-based post-processing
bool YOLOV8OBBModel::batch_inference(std::vector<cv::Mat> batch_images, std::vector<std::vector<DetBox>>& batch_result)
{
    if(batch_images.empty() ) {
        std::cerr << "Input images data is empty." << std::endl;
        return false;
    }

    // Ensure the batch size does not exceed the model's batch size
    for (size_t i = 0; i < batch_images.size(); i += m_model.batch_size) {
        size_t current_batch_size = std::min(static_cast<size_t>(m_model.batch_size), batch_images.size() - i);
        std::vector<cv::Mat> img_batch;
        for (size_t j = i; j < i + current_batch_size && j < batch_images.size(); j++) {
            img_batch.emplace_back(batch_images[j]);
        }

        if (!doInference(img_batch, batch_result)) {
            return false;
        }
    }

    return true;
}
