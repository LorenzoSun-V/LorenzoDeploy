/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-21 14:19:07
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-13 11:19:47
 * @Description: YOLOv8OBB模型前处理、推理、后处理代码
 */
#include "yolov8obb.h"

YOLOV8OBBModel::YOLOV8OBBModel()
    : m_trt{ Logger(),  nullptr, nullptr, nullptr, nullptr},
    m_kOutputSize(0),  
    m_kInputSize(0),
    m_kDecodeSize(0),   
    kMaxInputImageSize(9000 * 9000),     
    inputSrcDevice(nullptr), outputSrcDevice(nullptr) {
}

YOLOV8OBBModel::~YOLOV8OBBModel() {
    if (inputSrcDevice) {
        cudaFreeHost(inputSrcDevice);
        inputSrcDevice = nullptr;
    }
    if (outputSrcDevice) {
        cudaFreeHost(outputSrcDevice);
        outputSrcDevice = nullptr;
    }
    if (m_trt.context) {
        cudaFreeHost(m_trt.context);
        m_trt.context = nullptr;
    }
    if (m_trt.engine) {
        cudaFreeHost(m_trt.engine);
        m_trt.engine = nullptr;
    }
    if (m_trt.runtime) {
        cudaFreeHost(m_trt.runtime);
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
    
    //? m_model.bbox_element + 1，多出的一位是用于存放keepflag，keepflag为0表示不保留，1表示保留
    m_kDecodeSize = m_model.batch_size * m_model.num_bboxes *  (m_model.bbox_element+1);
    decode_ptr_host = new float[m_kDecodeSize];
    if (cudaMalloc((void**)&decode_ptr_device, (m_kDecodeSize) * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for decode." << std::endl;
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
    if (NULL == m_trt.runtime) {
        std::cerr << "Failed to create Infer Runtime." << std::endl;
        goto FAILED;
    }

    m_trt.engine = m_trt.runtime->deserializeCudaEngine(serialized_engine, size);
    if (NULL == m_trt.engine) {
        std::cerr << "Failed to deserialize CUDA engine." << std::endl;
        goto FAILED;
    }  

    m_trt.context = m_trt.engine->createExecutionContext();
    if (NULL == m_trt.context) {
        std::cerr << "Failed to create execution context." << std::endl;
        goto FAILED;
    }

    delete[] serialized_engine;
    return true;

FAILED: 
        delete[] serialized_engine;
        return false;
}

bool YOLOV8OBBModel::doInference(std::vector<cv::Mat> batch_images) 
{
    cuda_batch_preprocess(batch_images, inputSrcDevice, m_model.input_width, m_model.input_height, m_trt.stream);
    void* bindings[] = { inputSrcDevice, outputSrcDevice };
    if (!m_trt.context->enqueueV2(bindings, m_trt.stream, nullptr)) {
        std::cerr << "Failed to enqueue inference." << std::endl;
        return false;
    }

    if (cudaMemcpyAsync(inputSrcDevice, inputData.data(), m_kInputSize * sizeof(float), 
            cudaMemcpyHostToDevice, m_trt.stream) != cudaSuccess) {
        std::cerr << "Failed to copy input data to device." << std::endl;
        return false;
    }
    
    if (cudaMemcpyAsync(output_data.data(), outputSrcDevice, m_kOutputSize * sizeof(float), 
            cudaMemcpyDeviceToHost, m_trt.stream) != cudaSuccess) {
        std::cerr << "Failed to copy output data to host." << std::endl;
        return false;
    }
    cudaStreamSynchronize(m_trt.stream);
    return true;
}

bool YOLOV8OBBModel::inference(cv::Mat frame, std::vector<DetBox>& result) 
{
    if(frame.empty() ){
        std::cerr << "Inut images data is empty." << std::endl;
        return false;
    }
    std::vector<cv::Mat> batch_images;
    batch_images.push_back(frame);

    if (!doInference(batch_images)) {
        return false;
    }

    // std::vector<BBox> bboxes;
    // nms_obb(bboxes, output_data.data(), m_model);
    // postprocess(bboxes, frame, m_model.input_width, m_model.input_height, detBoxs);
    std::vector<std::vector<BBox>> batch_bboxes;
    std::vector<std::vector<DetBox>> batch_result;
    //! CPU NMS
    // nms_obb_batch(batch_bboxes, output_data.data(), m_model);
    // for (int i = 0; i < batch_images.size(); i++) {
    //     std::vector<BBox> bboxes;
    //     for (int j = 0; j < m_model.num_bboxes; j++) {
    //         BBox box;
    //         box.center_x = output_data[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 0];
    //         box.center_y = output_data[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 1];
    //         box.w = output_data[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 2];
    //         box.h = output_data[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 3];
    //         box.radian = output_data[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 4 + m_model.num_classes];
    //         box.score = output_data[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 4 + m_model.num_classes + 1];
    //         box.class_id = output_data[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 4 + m_model.num_classes + 2];
    //         bboxes.push_back(box);
    //     }
    //     batch_bboxes.push_back(bboxes);
    // }
    
    //! GPU NMS
    cuda_decode_obb(output_data.data(), m_model.num_bboxes, m_model.num_classes, m_model.iou_thresh, decode_ptr_device, m_model.max_objects, m_trt.stream);
    // cuda_nms_obb(decode_ptr_device, m_model.iou_thresh, m_model.max_objects, m_trt.stream);
    if (cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, m_kDecodeSize * sizeof(float), 
            cudaMemcpyDeviceToHost, m_trt.stream) != cudaSuccess) {
        std::cerr << "Failed to copy decode data to host." << std::endl;
        return false;
    }
    for (int i = 0; i < batch_images.size(); i++) {
        std::vector<BBox> bboxes;
        for (int j = 0; j < m_model.num_bboxes; j++) {
            int keepflag = decode_ptr_host[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 4 + m_model.num_classes + 3];
            if (keepflag == 0) {
                continue;
            }
            BBox box;
            box.center_x = decode_ptr_host[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 0];
            box.center_y = decode_ptr_host[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 1];
            box.w = decode_ptr_host[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 2];
            box.h = decode_ptr_host[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 3];
            box.radian = decode_ptr_host[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 4 + m_model.num_classes];
            box.score = decode_ptr_host[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 4 + m_model.num_classes + 1];
            box.class_id = decode_ptr_host[i * m_model.num_bboxes * m_model.bbox_element + j * m_model.bbox_element + 4 + m_model.num_classes + 2];
            bboxes.push_back(box);
        }
        batch_bboxes.push_back(bboxes);
    }

    
    bool bres = postprocess_batch(batch_bboxes, batch_images, m_model.input_width, m_model.input_height, batch_result);
    if( !bres ) {
        return false;
    }
    result = batch_result[0];
    return true;
}

bool YOLOV8OBBModel::batch_inference(std::vector<cv::Mat> batch_images, std::vector<std::vector<DetBox>>& batch_result)
{
    if(batch_images.empty() ) {
        std::cerr << "Input images data is empty." << std::endl;
        return false;
    }


    for (size_t i = 0; i < batch_images.size(); i += m_model.batch_size) {
        std::vector<cv::Mat> img_batch;
        for (size_t j = i; j < i + m_model.batch_size && j < batch_images.size(); j++) {
            img_batch.emplace_back(batch_images[j]);
        }

        if (!doInference(img_batch)) {
            return false;
        }

        std::vector<std::vector<BBox>> batch_size_bboxes;
        std::vector<std::vector<DetBox>> batch_size_result;
        nms_obb_batch(batch_size_bboxes, output_data.data(), m_model);

        bool bres = postprocess_batch(batch_size_bboxes, img_batch, m_model.input_width, m_model.input_height, batch_size_result);
        if( !bres ) {
            return false;
        }

        for(int j=0; j<batch_size_result.size();j++)
        {
            batch_result.push_back( batch_size_result[j] );
        }
    }

    return true;
}
   