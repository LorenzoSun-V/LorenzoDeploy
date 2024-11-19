#include "yoloe2e.h"


YOLOE2EModelManager::YOLOE2EModelManager()
    : runtime(nullptr), engine(nullptr), context(nullptr), stream(0), num_dets(nullptr),
      det_boxes(nullptr), det_scores(nullptr), det_classes(nullptr), blob(nullptr), in_size(0),
      num_size1(0), boxes_size2(0), scores_size3(0), classes_size4(0) {
    std::fill(std::begin(buffs), std::end(buffs), nullptr);
}

YOLOE2EModelManager::~YOLOE2EModelManager() {
    if (context) {
        context->destroy();
        context = nullptr;
    }
    if (engine) {
        engine->destroy();
        engine = nullptr;
    }
    if (runtime) {
        runtime->destroy();
        runtime = nullptr;
    }
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
    if (num_dets) {
        delete[] num_dets;
        num_dets = nullptr;
    }
    if (det_boxes) {
        delete[] det_boxes;
        det_boxes = nullptr;
    }
    if (det_scores) {
        delete[] det_scores;
        det_scores = nullptr;
    }
    if (det_classes) {
        delete[] det_classes;
        det_classes = nullptr;
    }
    for (auto &buff : buffs) {
        if (buff) {
            cudaFree(buff);
            buff = nullptr;
        }
    }
    if (blob) {
        delete[] blob;
        blob = nullptr;
    }
}

// Load the model from the serialized engine file
bool YOLOE2EModelManager::loadModel(const std::string engine_name) {
    if (!deserializeEngine(engine_name)) {
        std::cerr << "Failed to deserialize engine." << std::endl;
        return false;
    }

    const int inputIndex = engine->getBindingIndex("images");
    auto inputDims = engine->getBindingDimensions(inputIndex);

    m_kBatchSize = inputDims.d[0];
    m_channel = inputDims.d[1];
    m_kInputH = inputDims.d[2];
    m_kInputW = inputDims.d[3];
    blob = new float[m_kInputH * m_kInputW * 3];

    std::cout << "m_kBatchSize: " << m_kBatchSize << " m_channel: " << m_channel << " m_kInputH: " << m_kInputH << " m_kInputW: " << m_kInputW << std::endl;
    in_size = 1;
    for (int j = 0; j < inputDims.nbDims; j++) {
        in_size *= inputDims.d[j];
    }

    auto out_dims1 = engine->getBindingDimensions(engine->getBindingIndex("num"));
    num_size1 = 1;
    for (int j = 0; j < out_dims1.nbDims; j++) {
        num_size1 *= out_dims1.d[j];
    }
    auto out_dims2 = engine->getBindingDimensions(engine->getBindingIndex("boxes"));
    boxes_size2 = 1;
    for (int j = 0; j < out_dims2.nbDims; j++) {
        boxes_size2 *= out_dims2.d[j];
    }
    auto out_dims3 = engine->getBindingDimensions(engine->getBindingIndex("scores"));
    scores_size3 = 1;
    for (int j = 0; j < out_dims3.nbDims; j++) {
        scores_size3 *= out_dims3.d[j];
    }
    auto out_dims4 = engine->getBindingDimensions(engine->getBindingIndex("classes"));
    classes_size4 = 1;
    for (int j = 0; j < out_dims4.nbDims; j++) {
        classes_size4 *= out_dims4.d[j];
    }

    std::cout << "num_size1: " << num_size1 << " boxes_size2: " << boxes_size2 << " scores_size3: " << scores_size3 << " classes_size4: " << classes_size4 << std::endl;
    cudaError_t state;
    state = cudaMalloc(&buffs[0], in_size * sizeof(float));
    if (state != cudaSuccess) {
        std::cerr << "allocate memory for input failed, state: " << state << std::endl;
        return false;
    }

    state = cudaMalloc(&buffs[1], num_size1 * sizeof(int));
    if (state != cudaSuccess) {
        std::cerr << "allocate memory for output num failed, state: " << state << std::endl;
        return false;
    }

    state = cudaMalloc(&buffs[2], boxes_size2 * sizeof(float));
    if (state != cudaSuccess) {
        std::cerr << "allocate memory for output boxes failed, state: " << state << std::endl;
        return false;
    }

    state = cudaMalloc(&buffs[3], scores_size3 * sizeof(float));
    if (state != cudaSuccess) {
        std::cerr << "allocate memory for output scores failed, state: " << state << std::endl;
        return false;
    }

    state = cudaMalloc(&buffs[4], classes_size4 * sizeof(int));
    if (state != cudaSuccess) {
        std::cerr << "allocate memory for output classes failed, state: " << state << std::endl;
        return false;
    }

    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream." << std::endl;
        return false;
    }

    num_dets = new int[num_size1];
    det_boxes = new float[boxes_size2];
    det_scores = new float[scores_size3];
    det_classes = new int[classes_size4];

    return true;
}

// Deserialize the engine from file
bool YOLOE2EModelManager::deserializeEngine(const std::string engine_name) {
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

    initLibNvInferPlugins(&gLogger, "");
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

float YOLOE2EModelManager::letterbox(
    const cv::Mat& image,
    cv::Mat& out_image,
    const cv::Size& new_shape = cv::Size(640, 640),
    int stride=32,
    const cv::Scalar& color=cv::Scalar(114, 114, 114),
    bool fixed_shape = false,
    bool scale_up = true) {
    cv::Size shape = image.size();
    float r = std::min(
        (float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);
    if (!scale_up) {
        r = std::min(r, 1.0f);
    }

    int newUnpad[2]{
        (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

    cv::Mat tmp;
    if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
        cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
    } else {
        tmp = image.clone();
    }

    float dw = new_shape.width - newUnpad[0];
    float dh = new_shape.height - newUnpad[1];

    if (!fixed_shape) {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }

    dw /= 2.0f;
    dh /= 2.0f;

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return 1.0f / r;
}

float* YOLOE2EModelManager::blobFromImage(cv::Mat& img) {
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < img_h; h++) {
            for (size_t w = 0; w < img_w; w++) {
                blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c] / 255.0;
            }
        }
    }
    return blob;
}

void YOLOE2EModelManager::postProcess(cv::Mat img,  float scale, std::vector<DetBox>& detBoxs) {
    int img_w = img.cols;
    int img_h = img.rows;
    int x_offset = (m_kInputW * scale - img_w) / 2;
    int y_offset = (m_kInputH * scale - img_h) / 2;

    for (size_t i = 0; i < num_dets[0]; i++) {
       
        if(det_scores[i] < 0.25) continue;
        DetBox detbox;
        float x0 = (det_boxes[i * 4]) * scale - x_offset;
        float y0 = (det_boxes[i * 4 + 1]) * scale - y_offset;
        float x1 = (det_boxes[i * 4 + 2]) * scale - x_offset;
        float y1 = (det_boxes[i * 4 + 3]) * scale - y_offset;
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
        detbox.x = x0;
        detbox.y = y0;
        detbox.w = x1 - x0;
        detbox.h = y1 - y0;

        detbox.classID = det_classes[i];
        detbox.confidence = det_scores[i];
        std::cout << "x: " <<detbox.x <<" y: "<<detbox.y<<" w: "<<detbox.w<<" h: "<< detbox.h<<" classID: "<< detbox.classID<<" confidence: "<<detbox.confidence<< std::endl;
        detBoxs.push_back(detbox);
    }
    std::cout << "size = " <<detBoxs.size() << std::endl;
}

bool YOLOE2EModelManager::doInference(cv::Mat img, std::vector<DetBox>& detBoxs) {
    cv::Mat pr_img;
    float scale = letterbox(img, pr_img, {m_kInputW, m_kInputH}, 32, {114, 114, 114}, true);
    cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB);
    blob = blobFromImage(pr_img);

    cudaError_t state = cudaMemcpyAsync(buffs[0], blob, in_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (state != cudaSuccess) {
        std::cerr << "blob transmit to cuda failed, state: " << state << std::endl;
        return false;
    }

    if (!context->enqueueV2(buffs, stream, nullptr)) {
        std::cerr << "Failed to enqueueV2 inference." << std::endl;
        return false;
    }

    if (cudaMemcpyAsync(num_dets, buffs[1], num_size1 * sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cerr << "num_dets transmit to host failed." << std::endl;
        return false;
    }
    if (cudaMemcpyAsync(det_boxes, buffs[2], boxes_size2 * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cerr << "det_boxes transmit to host failed." << std::endl;
        return false;
    }
    if (cudaMemcpyAsync(det_scores, buffs[3], scores_size3 * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cerr << "det_scores transmit to host failed." << std::endl;
        return false;
    }
    if (cudaMemcpyAsync(det_classes, buffs[4], classes_size4 * sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cerr << "det_classes transmit to host failed." << std::endl;
        return false;
    }

    cudaStreamSynchronize(stream);
    postProcess(img, scale, detBoxs);
    return true;
}

// Perform inference on the input frame
bool YOLOE2EModelManager::inference(cv::Mat frame, std::vector<DetBox>& detBoxs) {
    if (!doInference(frame, detBoxs)) {
        return false;
    }

    if (detBoxs.empty()) {
        return false;
    }

    return true;
}
