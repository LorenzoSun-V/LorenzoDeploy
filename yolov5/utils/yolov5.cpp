#include "yolov5.h"

YOLOV5Model::YOLOV5Model()
    : runtime(nullptr), engine(nullptr), context(nullptr), cpu_output_buffer(nullptr),m_kBatchSize(1) {
    kOutputSize =  kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
}

YOLOV5Model::~YOLOV5Model() {
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));

    if(NULL != cpu_output_buffer){
        delete[] cpu_output_buffer;
        cpu_output_buffer = NULL;
    }

    cuda_preprocess_destroy();
    // Destroy the engine
    if (context) cudaFreeHost(context);
    if (engine) cudaFreeHost(engine);
    if (runtime) cudaFreeHost(runtime);
}

bool YOLOV5Model::prepareBuffer() 
{
    if (engine->getNbBindings() != 2) {
        std::cerr << "Error: Number of bindings is not 2!" << std::endl;
        return false; 
    }

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    // const int inputIndex = engine->getBindingIndex(kInputTensorName);
    // const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    // if(inputIndex != 0){
    //     std::cerr << "Error: Input tensor name is not " << kInputTensorName << "!" << std::endl;
    //     return false;
    // } 
    // if(outputIndex != 1){
    //     std::cerr << "Error: Output tensor name is not " << kOutputTensorName << "!" << std::endl;
    //     return false;
    // } 

    m_kBatchSize = engine->getMaxBatchSize();
    auto inputDims = engine->getBindingDimensions(0);
    m_channel = inputDims.d[0];
    m_kInputH = inputDims.d[1];
    m_kInputW = inputDims.d[2];
    std::cout << "Input tensor: " << m_kBatchSize << "x" << m_channel << "x" << m_kInputH << "x" << m_kInputW << std::endl;

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&device_buffers[0], m_kBatchSize * m_channel * m_kInputH * m_kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&device_buffers[1], m_kBatchSize * kOutputSize * sizeof(float)));

    cpu_output_buffer = new float[m_kBatchSize * kOutputSize];
    return true;
}

void YOLOV5Model::doInference(std::vector<cv::Mat> img_batch, std::vector<std::vector<Detection>>& res_batch) 
{
      // Preprocess
    cuda_batch_preprocess(img_batch, device_buffers[0], m_kInputW, m_kInputH, stream);

    // Run inference
    context->enqueue(m_kBatchSize, (void **)device_buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, device_buffers[1], m_kBatchSize* kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    // NMS
    batch_nms(res_batch, cpu_output_buffer, m_kBatchSize, kOutputSize, kConfThresh, kNmsThresh);
}       

bool YOLOV5Model::deserializeEngine(const std::string& engine_name) {
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
    if (!serialized_engine) {
        std::cerr << "Failed to allocate memory for serialized engine." << std::endl;
        return false;
    }
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

bool YOLOV5Model::loadModel(const std::string& engine_name) {
    if (!deserializeEngine(engine_name)) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        return false;
    }

    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    if (!prepareBuffer()) {
        std::cerr << "Failed to prepareBuffer engine." << std::endl;
        return false;
    }
    return true;
}

bool YOLOV5Model::inference(cv::Mat& frame, std::vector<DetBox>& detBoxs) {
    if (frame.empty()) {
        std::cerr << "Input frame is empty." << std::endl;
        return false;
    }

    // Preprocess
    std::vector<cv::Mat> img_batch;
    img_batch.clear();
    img_batch.push_back(frame); 

    std::vector<std::vector<Detection>> res_batch;
    doInference(img_batch, res_batch);

    for (auto& obj : res_batch[0]) {
        DetBox detbox;
        detbox.classID = obj.class_id;
        detbox.confidence = obj.conf;

        cv::Rect r = get_rect(img_batch[0], m_kInputW, m_kInputH, obj.bbox);
        detbox.x = std::max(0, r.x);
        detbox.y = std::max(0, r.y);

        detbox.w = r.width;
        detbox.h = r.height;
        detBoxs.push_back(detbox);
    }
    
    return true;
}


bool YOLOV5Model::batchInference(std::vector<cv::Mat>& batchframes, std::vector<std::vector<DetBox>>& batchDetBoxs) {
    if (batchframes.empty()) {
        std::cerr << "Input batch is empty." << std::endl;
        return false;
    }

    // batch predict
    for(size_t i = 0; i < batchframes.size(); i += m_kBatchSize)
    {
        std::vector<cv::Mat> img_batch;
        for (size_t j = i; j < i + m_kBatchSize && j < batchframes.size(); j++) {
            if (batchframes[j].empty()) {
                std::cout << "batchInference failed to decode frame!"<< std::endl;
                continue;
            }
            img_batch.push_back(batchframes[j]);
        }

        std::vector<std::vector<Detection>> res_batch;
        doInference(img_batch, res_batch);
        int index = 0;
        for (auto& objects : res_batch) {
            std::vector<DetBox> detresult;
            for (auto& obj : objects) {
                DetBox detbox;
                detbox.classID = obj.class_id;
                detbox.confidence = obj.conf;

                cv::Rect r = get_rect(img_batch[index], m_kInputW, m_kInputH, obj.bbox);
                detbox.x = std::max(0, r.x);
                detbox.y = std::max(0, r.y);

                detbox.w = r.width;
                detbox.h = r.height;
                detresult.push_back(detbox);
            }
            index++; //遍历四张图中下一张
            batchDetBoxs.push_back(detresult);
        }
    }
              
    return true;
}

