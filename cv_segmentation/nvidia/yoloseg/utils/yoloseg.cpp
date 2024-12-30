/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-26 08:51:12
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-30 16:35:41
 * @Description: 
 */
#include "yoloseg.h"

YOLOSegModel::YOLOSegModel()
    : m_trt{ Logger(),  nullptr, nullptr, nullptr, nullptr}, 
    yolov8(true),
    m_kDetOutputSize(0),  
    m_kSegOutputSize(0),  
    kMaxInputImageSize(9000 * 9000),
    gpu_input_data(nullptr),
    gpu_det_output_data(nullptr),
    gpu_seg_output_data(nullptr) {
}

YOLOSegModel::~YOLOSegModel()
{
    if (gpu_input_data)        { cudaFree(gpu_input_data);         gpu_input_data = nullptr; }
    if (gpu_det_output_data)   { cudaFree(gpu_det_output_data);    gpu_det_output_data = nullptr; }
    if (gpu_seg_output_data)   { cudaFree(gpu_seg_output_data);    gpu_seg_output_data = nullptr; }
    if (m_trt.context)         { m_trt.context->destroy();         m_trt.context = nullptr; }
    if (m_trt.engine)          { m_trt.engine->destroy();          m_trt.engine = nullptr; }
    if (m_trt.runtime)         { m_trt.runtime->destroy();         m_trt.runtime = nullptr; }
    if (m_trt.stream)          { cudaStreamDestroy(m_trt.stream);  m_trt.stream = nullptr; }

    cuda_preprocess_destroy();
}

// set if use YOLOv8 
void YOLOSegModel::setYOLOv8Flag(bool enable){
    yolov8 = enable;
}

// deserialize the engine from file
bool YOLOSegModel::deserializeEngine(const std::string engine_name)
{
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return false;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    char* serialized_engine = new char[size];
    file.read(serialized_engine, size);
    file.close();

    cudaSetDevice(0);
    m_trt.runtime = createInferRuntime(m_trt.gLogger);
    if (!m_trt.runtime) {
        std::cerr << "Failed to create Infer Runtime." << std::endl;
        delete[] serialized_engine;
        return false;
    }

    m_trt.engine = m_trt.runtime->deserializeCudaEngine(serialized_engine, size);
    if (!m_trt.engine) {
        std::cerr << "Failed to deserialize CUDA engine." << std::endl;
        delete[] serialized_engine;
        return false;
    }

    m_trt.context = m_trt.engine->createExecutionContext();
    if (!m_trt.context) {
        std::cerr << "Failed to create execution context." << std::endl;
        delete[] serialized_engine;
        return false;
    }

    delete[] serialized_engine;
    return true;
}

// Load the model from the serialized engine file
bool YOLOSegModel::loadModel(const std::string engine_name) {
    struct stat buffer;
    if (stat(engine_name.c_str(), &buffer) != 0) {
        std::cerr << "Error: File " << engine_name << " does not exist!" << std::endl;
        return false;
    }
    if (!deserializeEngine(engine_name)) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        return false;
    }
    // instance segmentation model has 1 input and 2 outputs
    if(m_trt.engine->getNbBindings() != 3){
        std::cerr << "Please input correct network model." << std::endl;
        return false;
    }
    //input     
    auto inputDims = m_trt.engine->getBindingDimensions(0);
    m_model.batch_size = inputDims.d[0];
    m_model.input_channel = inputDims.d[1];
    m_model.input_height = inputDims.d[2];
    m_model.input_width = inputDims.d[3];

    std::cout << "[Model] batch_size=" << m_model.batch_size
              << ", input_channel="    << m_model.input_channel
              << ", input_height="     << m_model.input_height
              << ", input_width="      << m_model.input_width
              << std::endl;
    // out_dims_seg: [1, 32, 160, 160]
    auto out_dims_seg = m_trt.engine->getBindingDimensions(1);
    m_model.seg_output = out_dims_seg.d[1];
    m_model.seg_output_height = out_dims_seg.d[2];
    m_model.seg_output_width = out_dims_seg.d[3];
    std::cout << "[SegOutput] seg_output="       << m_model.seg_output
              << ", seg_output_height="         << m_model.seg_output_height
              << ", seg_output_width="          << m_model.seg_output_width
              << std::endl;

    // yolov5: out_dims_det: [1, 25200, 117], 第三维度是117（85+32），其中前面85字段包括四个坐标属性（cx、cy、w、h）、一个confidence和80个类别分数，后面32个字段是每个检测框的mask系数。
    // yolov8: out_dims_det: [1, 8400, 116], 第三维度是116（84+32），其中前面84字段包括四个坐标属性（cx、cy、w、h）和80个类别分数，后面32个字段是每个检测框的mask系数。
    auto out_dims_det = m_trt.engine->getBindingDimensions(2);
    if (yolov8) {
        m_model.num_classes = out_dims_det.d[2] - 32 - 4;
    } else {
        m_model.num_classes = out_dims_det.d[2] - 32 - 5;
    }
    m_model.num_bboxes = out_dims_det.d[1];
    m_model.bbox_element = out_dims_det.d[2];
    
    std::cout << "[DetOutput] num_classes="   << m_model.num_classes
              << ", num_bboxes="            << m_model.num_bboxes
              << ", bbox_element="          << m_model.bbox_element
              << std::endl;

    if (cudaStreamCreate(&m_trt.stream) != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream." << std::endl;
        return false;
    }
    // prepare cpu and gpu buffers
    cuda_preprocess_init(kMaxInputImageSize);
    // allocate memory for input data
    int kInputSize = m_model.batch_size * m_model.input_channel * m_model.input_height * m_model.input_width;
    cudaError_t err = cudaMalloc(&gpu_input_data, kInputSize * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for input data: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // allocate memory for det output
    m_kDetOutputSize = m_model.batch_size * m_model.num_bboxes * m_model.bbox_element;
    if (cudaMalloc(&gpu_det_output_data, m_kDetOutputSize * sizeof(float)) != cudaSuccess) {
        std::cerr << "CUDA malloc failed for det data." << std::endl;
        return false;
    }

    // allocate memory for seg output
    m_kSegOutputSize = m_model.batch_size * m_model.seg_output * m_model.seg_output_height * m_model.seg_output_width;
    if (cudaMalloc(&gpu_seg_output_data, m_kSegOutputSize * sizeof(float)) != cudaSuccess) {
        std::cerr << "CUDA malloc failed for seg data." << std::endl;
        return false;
    }
    // check the output size
    std::cout << "Det Output Size: " << m_kDetOutputSize << std::endl;
    std::cout << "Seg Output Size: " << m_kSegOutputSize << std::endl;

    cpu_det_output_data.resize(m_kDetOutputSize);
    cpu_seg_output_data.resize(m_kSegOutputSize);

    std::cout << "[Alloc] DetOutputSize=" << m_kDetOutputSize
              << ", SegOutputSize="       << m_kSegOutputSize << std::endl;

    return true;
}

bool YOLOSegModel::doInference(std::vector<cv::Mat> img_batch) 
{
    // 1) preprocess and copy input to GPU
    cuda_batch_preprocess(img_batch, gpu_input_data, m_model.input_width, m_model.input_height, m_trt.stream);
    // 2) bind input and output buffers
    void* bindings[] = { gpu_input_data, gpu_seg_output_data, gpu_det_output_data};
    if (!m_trt.context->enqueueV2(bindings, m_trt.stream, nullptr)) {
        std::cerr << "Failed to enqueue inference." << std::endl;
        return false;
    }
    // 3) copy output data to host asynchronously
    if (cudaMemcpyAsync(cpu_det_output_data.data(), gpu_det_output_data, 
        m_kDetOutputSize * sizeof(float), cudaMemcpyDeviceToHost, m_trt.stream) != cudaSuccess){
        std::cerr << "Failed to copy det output data to host." << std::endl;
        return false;
    }
    if (cudaMemcpyAsync(cpu_seg_output_data.data(), gpu_seg_output_data, 
        m_kSegOutputSize * sizeof(float), cudaMemcpyDeviceToHost, m_trt.stream) != cudaSuccess){
        std::cerr << "Failed to copy seg output data to host." << std::endl;
        return false;
    }

    // 4) wait for the stream to finish
    if (cudaStreamSynchronize(m_trt.stream) != cudaSuccess) {
        std::cerr << "CUDA stream synchronization failed." << std::endl;
        return false;
    }

    return true;
}

bool YOLOSegModel::inference(cv::Mat frame, std::vector<SegBox>& result, std::vector<cv::Mat>& masks) 
{
    if(frame.empty() ){
        std::cerr << "Inut images data is empty." << std::endl;
        return false;
    }

    if (frame.cols <= 0 ||frame.cols <= 0) {
        std::cerr << "Invalid input image size: " << frame.cols << "x" << frame.cols << std::endl;
        return false;
    }

    std::vector<cv::Mat> batch_images{frame};

    if (!doInference(batch_images)) {
        return false;
    }

    std::vector<std::vector<InstanceSegResult>> batch_size_bboxes;
    std::vector<std::vector<cv::Mat>> batch_size_masks;

    std::vector<std::vector<SegBox>> batch_det_result;
    std::vector<std::vector<cv::Mat>> batch_seg_result;
    
    batch_nms(batch_size_bboxes, cpu_det_output_data.data(), m_model, yolov8);
    for (size_t i = 0; i < m_model.batch_size; i++) {
        std::vector<InstanceSegResult> dets = batch_size_bboxes[i];
        std::vector<cv::Mat> batch_masks;
        process_mask(cpu_seg_output_data.data(), m_kSegOutputSize, dets, batch_masks, m_model);
        batch_size_masks.push_back(batch_masks);
    }
    
    bool bres = postprocess_batch(batch_size_bboxes, batch_size_masks, batch_images, m_model.input_width, m_model.input_height, batch_det_result, batch_seg_result);
    if( !bres ) {
        return false;
    }
    result = batch_det_result[0];
    masks  = batch_seg_result[0];
    return true;
}

bool YOLOSegModel::batch_inferenece(std::vector<cv::Mat> batch_images, std::vector<std::vector<SegBox>>& batch_result, std::vector<std::vector<cv::Mat>>& batch_masks) 
{
    if(batch_images.empty() ) {
        std::cerr << "Input images data is empty." << std::endl;
        return false;
    }

    for (size_t i = 0; i < batch_images.size(); i += m_model.batch_size){
        std::vector<cv::Mat> img_batch;
        for (size_t j = i; j < i + m_model.batch_size && j < batch_images.size(); j++) {
            img_batch.emplace_back(batch_images[j]);
        }

        if (!doInference(img_batch)) {
            return false;
        }

        std::vector<std::vector<InstanceSegResult>> batch_size_bboxes;
        std::vector<std::vector<cv::Mat>> batch_size_masks;

        batch_nms(batch_size_bboxes, cpu_det_output_data.data(), m_model, yolov8);
        for (size_t j = 0; j < m_model.batch_size; j++) {
            std::vector<InstanceSegResult> dets = batch_size_bboxes[j];
            std::vector<cv::Mat> batch_mask;
            size_t offset = j * m_model.seg_output * m_model.seg_output_height * m_model.seg_output_width;
            process_mask(cpu_seg_output_data.data() + offset, m_kSegOutputSize / m_model.batch_size, dets, batch_mask, m_model);
            batch_size_masks.push_back(batch_mask);
        }

        bool bres = postprocess_batch(batch_size_bboxes, batch_size_masks, img_batch, m_model.input_width, m_model.input_height, batch_result, batch_masks);
        if( !bres ) {
            return false;
        }
    
    }

    return true;
}