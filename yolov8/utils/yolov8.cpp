#include "yolov8.h"

YOLOV8ModelManager::YOLOV8ModelManager()
    : runtime(nullptr), engine(nullptr), context(nullptr), output_buffer_host(nullptr),
     decode_ptr_host(nullptr), decode_ptr_device(nullptr) {
    kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
}

YOLOV8ModelManager::~YOLOV8ModelManager() {
    if (context) cudaFreeHost(context);
    if (engine) cudaFreeHost(engine);
    if (runtime) cudaFreeHost(runtime);
    cudaStreamDestroy(stream);

    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    CUDA_CHECK(cudaFree(decode_ptr_device));

    if(decode_ptr_host) delete[] decode_ptr_host;
    if(output_buffer_host) delete[] output_buffer_host;
}

bool YOLOV8ModelManager::loadModel(const std::string& engine_name) {
    if (!deserializeEngine(engine_name)) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        return false;
    }

    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);

    auto out_dims = engine->getBindingDimensions(1);
    model_bboxes = out_dims.d[0];

    if (!prepareBuffer()) {
        std::cerr << "Failed to prepareBuffer engine." << std::endl;
        return false;
    }

    return true;
}

bool YOLOV8ModelManager::deserializeEngine(const std::string& engine_name) {
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

    runtime = createInferRuntime(gLogger);
    if (NULL == runtime) {
        std::cerr << "Failed to create Infer Runtime." << std::endl;
        goto FAILED;
    }
    
    engine = runtime->deserializeCudaEngine(serialized_engine, size);
    if (NULL == engine) {
        std::cerr << "Failed to deserialize CUDA engine." << std::endl;
        goto FAILED;
    }   
    
    context = engine->createExecutionContext();
    if (NULL == context) {
        std::cerr << "Failed to create execution context." << std::endl;
        goto FAILED;
    }
    delete[] serialized_engine;
    return true;

FAILED: 
        delete[] serialized_engine;
        return false;
}

bool YOLOV8ModelManager::prepareBuffer() {
    int nbBindings = engine->getNbIOTensors();
    if (nbBindings != 2) {
        std::cerr << "Error: Number of bindings is not 2!" << std::endl;
        return false; 
    }

    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    if(inputIndex != 0){
        std::cerr << "Error: Input tensor name is not " << kInputTensorName << "!" << std::endl;
        return false;
    } 
    if(outputIndex != 1){
        std::cerr << "Error: Output tensor name is not " << kOutputTensorName << "!" << std::endl;
        return false;
    }

    m_kBatchSize = engine->getMaxBatchSize();
    auto inputDims = engine->getBindingDimensions(inputIndex);
    m_channel = inputDims.d[0];
    m_kInputH = inputDims.d[1];
    m_kInputW = inputDims.d[2];

    CUDA_CHECK(cudaMalloc((void**)&device_buffers[0], m_kBatchSize * m_channel * m_kInputH * m_kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&device_buffers[1], m_kBatchSize * kOutputSize * sizeof(float)));

    if (m_kBatchSize > 1) {//模型batch大于1进行cpu前处理
        output_buffer_host = new float[m_kBatchSize * kOutputSize];
     } else {
        decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void**)&decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }

    return true;
}

void YOLOV8ModelManager::infer(std::vector<cv::Mat> img_batch, std::vector<std::vector<Detection>>& res_batch)
{
    if(NULL == device_buffers){
        std::cout <<"device_buffers is empty"<< std::endl;
        return;
    }
    // infer on the batch asynchronously, and DMA output back to host
    //auto start = std::chrono::system_clock::now();
    context->enqueue(m_kBatchSize, (void **)device_buffers, stream, nullptr);

    if (m_kBatchSize > 1) {
        CUDA_CHECK(cudaMemcpyAsync(output_buffer_host, device_buffers[1], m_kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    } else  {
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float *)device_buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);//cuda nms
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost, stream));       
    }
    //auto end = std::chrono::system_clock::now();
    //std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (m_kBatchSize > 1) {
        batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
    } else {
        //Process gpu decode and nms results
        batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
    }
}

bool YOLOV8ModelManager::inference(cv::Mat& frame, std::vector<DetBox>& detBoxs) {
    if (frame.empty()) {
        std::cerr << "Input frame is empty." << std::endl;
        return false;
    }

    // Preprocess
    std::vector<cv::Mat> img_batch = {frame};
    cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);

    std::vector<std::vector<Detection>> res_batch;
    infer(img_batch, res_batch);

    for (auto& obj : res_batch[0]) {
        DetBox detres;
        detres.classID = obj.class_id;
        detres.confidence = obj.conf;

        cv::Rect r = get_rect(frame, obj.bbox);

        detres.x = std::max(0, r.x);
        detres.y = std::max(0, r.y);
        detres.w = r.width;
        detres.h = r.height;
        detBoxs.push_back(detres);
    }
    
    return true;
}

bool YOLOV8ModelManager::batchInference(std::vector<cv::Mat>& img_frames, std::vector<std::vector<DetBox>>& batchDetBoxs) {
    if (img_frames.empty()) {
        std::cerr << "Input batch is empty." << std::endl;
        return false;
    }

    // batch predict
    for(size_t i = 0; i < img_frames.size(); i += m_kBatchSize)
    {
        std::vector<cv::Mat> img_batch;
        for (size_t j = i; j < i + m_kBatchSize && j < img_frames.size(); j++) {
         
            img_batch.push_back(img_frames[j]);
        }

        cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
        std::vector<std::vector<Detection>> res_batch;
        infer(img_batch, res_batch);

        int index = 0;
        for (auto& objects : res_batch) {
            std::vector<DetBox> detresult;
            for (auto& obj : objects) {
                DetBox detbox;
                detbox.classID = obj.class_id;
                detbox.confidence = obj.conf;

                cv::Rect r = get_rect(img_batch[index], obj.bbox);
                detbox.x = std::max(0, r.x);
                detbox.y = std::max(0, r.y);
                detbox.w = r.width;
                detbox.h = r.height;
                detresult.push_back(detbox);
            }
            index++;//遍历四张图中下一张
            batchDetBoxs.push_back(detresult);
        }
    }
        
    return true;
}
