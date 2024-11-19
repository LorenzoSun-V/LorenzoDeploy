/**
* @file    fastinfer.cpp.
*
* @brief
*
* 深度学习模型接口文件
*
* @copyright
*
* All Rights Reserved.
*/
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <utility>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include "fastdeploy/vision.h"
#include "fastinfer.h"

struct ModelParam {
    bool bParamIsOk;
    MODEL_TYPE_E eModelVersion;
    //onnx
    std::shared_ptr<fastdeploy::vision::detection::YOLOv7End2EndTRT> model0;
    std::shared_ptr<fastdeploy::vision::detection::YOLOv7> model1;
    std::shared_ptr<fastdeploy::vision::detection::YOLOv5> model2;
    std::shared_ptr<fastdeploy::vision::detection::YOLOv8> model9;  
    
    /// custom config
    int startid;

    ModelParam() 
    {    
        startid = 0;
        bParamIsOk = false;
    }
};

struct ModelDetectDnnInstance {
    std::shared_ptr<ModelParam> _param;
};


double GetCurrentTimeStampMS()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

ENUM_ERROR_CODE ReadFrameFromPath(const char* pImagePath, cv::Mat& frame)
{
    if( pImagePath == NULL )
    {
        std::cout << "please check input image path!" << std::endl;
	    return ERR_INVALID_PARAM;
    }
    
	cv::Mat imgsrc = cv::imread(pImagePath);
	if( imgsrc.empty() )
	{
		std::cout<<"image is empty!"<<std::endl;
		return ERR_GET_IMAGE_EMPTY;
	}
    frame = imgsrc;
	return ENUM_OK;
}

void* LoadDeepModelModules(const char* pmodelPath, const char* pSerializefile , int iStartnumber, MODEL_TYPE_E eModelversion, RUN_PLATFROM_E ePlatfrom)
{   
    if ( pmodelPath == NULL)
    {
        std::cout << "input model path is empty!" << std::endl;
	    return NULL;
    }

    ModelDetectDnnInstance *_instance = new ModelDetectDnnInstance();
    if (NULL == _instance) {
        std::cout << "malloc memory error!" << std::endl;
        return NULL;
    }

    _instance->_param = std::make_shared<ModelParam>();
    _instance->_param->startid = iStartnumber;

    auto option = fastdeploy::RuntimeOption();
    switch (ePlatfrom)
    {
        case RUN_CPU:
            option.UseCpu();
            break;
        case RUN_GPU:
            option.UseGpu();
            break;
        case RUN_TRT:
            option.UseGpu();
            option.UseTrtBackend();
            option.trt_option.enable_fp16 = true;
            if(NULL == pSerializefile)  {
                option.trt_option.serialize_file = "./model_fp16.trt";
            } 
            else {
                option.trt_option.serialize_file = pSerializefile; 
            }
            break;      
        default:
            std::cout << "ePlatfrom is error!" << std::endl;
            return NULL;
    }

 std::cerr << "eModelversion: "<<eModelversion << std::endl;
    switch(eModelversion)
    {
        case MODEL_CODE_0:
            _instance->_param->model0 = std::move(std::make_shared<fastdeploy::vision::detection::YOLOv7End2EndTRT>(pmodelPath));
            if (!_instance->_param->model0->Initialized()) {
                std::cerr << "Failed to initialize." << std::endl;
                return NULL;
            }
            break;
        case MODEL_CODE_1:
            _instance->_param->model1 = std::move(std::make_shared<fastdeploy::vision::detection::YOLOv7>(pmodelPath," ", option));
            if (!_instance->_param->model1->Initialized()) {
                std::cerr << "Failed to initialize." << std::endl;
                return NULL;
            }
            break;
        case MODEL_CODE_2:
            _instance->_param->model2 = std::move(std::make_shared<fastdeploy::vision::detection::YOLOv5>(pmodelPath," ", option));
            if (!_instance->_param->model2->Initialized()) {
                std::cerr << "Failed to initialize." << std::endl;
                return NULL;
            }
            break;
        case MODEL_CODE_9:
            _instance->_param->model9 = std::move(std::make_shared<fastdeploy::vision::detection::YOLOv8>(pmodelPath," ", option));
            if (!_instance->_param->model9->Initialized()) {
                std::cerr << "Failed to initialize." << std::endl;
                return NULL;
            }   
            break;
       default:
             std::cout << "input model version error!" << std::endl;
             return NULL;
    }      
    
    _instance->_param->eModelVersion = eModelversion;
    _instance->_param->bParamIsOk = true; 
    return (void*)_instance; 
}

ENUM_ERROR_CODE InferenceGetDetectResult(void* pDeepInstance, cv::Mat frame, DetBox* pDetBoxs, int* pndetcount)
{
    struct ModelDetectDnnInstance* _instance = 
           reinterpret_cast<struct ModelDetectDnnInstance*>(pDeepInstance);
           
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cerr << "pInstance is NULL" << std::endl;
	    return ERR_INVALID_PARAM;
    }
    
    if (frame.empty())
    {
        std::cerr <<  "Failed to read frame." << std::endl;
        return ERR_GET_IMAGE_EMPTY;
    }

    fastdeploy::vision::DetectionResult result;
    bool bres = false;
    switch(_instance->_param->eModelVersion)
    {
        case MODEL_CODE_0: 
            bres = _instance->_param->model0->Predict(&frame, &result);
            break;
        case MODEL_CODE_1: 
            bres = _instance->_param->model1->Predict(&frame, &result);
            break;
        case MODEL_CODE_2:
            bres = _instance->_param->model2->Predict(&frame, &result);
            break;
        case MODEL_CODE_9: 
            bres = _instance->_param->model9->Predict(frame, &result);
            break;
       default:
            std::cerr << "Unimplemented function!" << std::endl;
            return ERR_INVALID_PARAM;
    }      
       
    if(!bres) {
        std::cerr << "Failed to predict." << std::endl;
        return ERR_DETECT_PREDICT_FAIL;
    }

    *pndetcount = result.boxes.size();
    for (int i = 0; i < result.boxes.size(); i++)
    {
        float score = result.scores[i]*100; 
        float x1 = result.boxes[i][0];
        float y1 = result.boxes[i][1];
        float x2 = result.boxes[i][2];
        float y2 = result.boxes[i][3];
        float box_h = y2 - y1;
        float box_w = x2 - x1;
       
        pDetBoxs[i].classID = result.label_ids[i] + _instance->_param->startid ;       
        pDetBoxs[i].confidence = score;
        pDetBoxs[i].x = x1;
        pDetBoxs[i].y = y1;                       
        pDetBoxs[i].w = box_w;   
        pDetBoxs[i].h = box_h;   
    }

    return ENUM_OK;
}  


ENUM_ERROR_CODE BatchInferenceGetDetectResult(void* pDeepInstance, BatchFrames* pBatchframes, int numFrames, BatchDetBox* pBatchDetBoxs)
{
    struct ModelDetectDnnInstance* _instance = reinterpret_cast<struct ModelDetectDnnInstance*>(pDeepInstance);

    if (!_instance || !_instance->_param->bParamIsOk) {
        std::cout << "pInstance is NULL" << std::endl;
        return ERR_INPUT_INSTANCE_INVALID;
    }

    if (pBatchframes == nullptr || numFrames <= 0)
    {
        std::cout << "Invalid input frames." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }

    std::vector<cv::Mat> batch_frames;
    for (int i = 0; i < numFrames; i++)
    {
        cv::Mat frame = pBatchframes[i].frame;

        if (frame.empty())
        {
            std::cout << "Failed to decode frame at index " << i << std::endl;
            return ERR_INVALID_PARAM;
        }
        batch_frames.push_back(frame);
    }
    
    std::vector<fastdeploy::vision::DetectionResult> batch_results;
    memset(&batch_results, 0x00, sizeof(fastdeploy::vision::DetectionResult));  

    bool bres = false;
    switch(_instance->_param->eModelVersion)
    {
        case MODEL_CODE_1: 
            bres = _instance->_param->model1->BatchPredict(batch_frames, &batch_results);
            break;
        case MODEL_CODE_2:
            bres = _instance->_param->model2->BatchPredict(batch_frames, &batch_results);
            break;
        case MODEL_CODE_9: 
            bres = _instance->_param->model9->BatchPredict(batch_frames, &batch_results);
            break;
       default:
            std::cerr << "Unimplemented function!" << std::endl;
            return ERR_INVALID_PARAM;
    }      

    if(!bres) {
        std::cerr << "Failed to predict." << std::endl;
        return ERR_DETECT_PREDICT_FAIL;
    }
    
    for(int i = 0; i < batch_results.size(); i++)
    {
        pBatchDetBoxs[i].ndetcount = batch_results[i].boxes.size();
        
        for (int j = 0; j < pBatchDetBoxs[i].ndetcount; j++)
        {
            float score = batch_results[i].scores[j] * 100;
            float x1 = batch_results[i].boxes[j][0];
            float y1 = batch_results[i].boxes[j][1];
            float x2 = batch_results[i].boxes[j][2];
            float y2 = batch_results[i].boxes[j][3];
            float box_h = y2 - y1;
            float box_w = x2 - x1;

            memset(pBatchDetBoxs[i].pdetbox, 0x00, sizeof(DetBox));   
            pBatchDetBoxs[i].pdetbox[j].classID = batch_results[i].label_ids[j] + _instance->_param->startid;
            pBatchDetBoxs[i].pdetbox[j].confidence = score;
            pBatchDetBoxs[i].pdetbox[j].x = x1;
            pBatchDetBoxs[i].pdetbox[j].y = y1;
            pBatchDetBoxs[i].pdetbox[j].w = box_w;
            pBatchDetBoxs[i].pdetbox[j].h = box_h;      
        }
   
    }

    return ENUM_OK;
}
 
ENUM_ERROR_CODE RectDetectResultForImage(cv::Mat &frame, DetBox* pDetBoxs, int detCount)
{
    if (frame.empty())
    {
        std::cerr <<  "frame is empty." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    
    if(detCount == 0)
    {
    	std::cerr << "detCount is NULL!" << std::endl;
	    return ERR_INVALID_PARAM;
    }
     
	//生成随机颜色
	std::vector<cv::Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(cv::Scalar(b, g, r));
	}
    
    
    for (size_t i = 0; i < detCount; i++)
    {
        const DetBox& obj = pDetBoxs[i];
        int labelID = obj.classID;
        float conf = obj.confidence;
 
        cv::rectangle(frame, cv::Rect(obj.x, obj.y, obj.w, obj.h),color[labelID],2,8);
            
        std::string strid = std::to_string(labelID).c_str();
        std::string strscore = std::to_string(conf).c_str();
        cv::putText(frame, strid+':'+strscore, cv::Point(obj.x, obj.y-5), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 1, 8, 0);                    
    }

    return ENUM_OK;
}  
  
ENUM_ERROR_CODE DestoryDeepmodeInstance( void **pDLInstance)
{

    struct ModelDetectDnnInstance* _instance =
         reinterpret_cast<struct ModelDetectDnnInstance*>(*pDLInstance);		 
	if(!_instance || !_instance->_param->bParamIsOk)
    {
    	std::cerr << "DestoryDeepmodeInstance pInstance is NULL" << std::endl;
	    return ERR_INVALID_PARAM;       
    }
    
	if (_instance)
    {
        delete _instance;
        *pDLInstance = NULL;
    }
         

    return ENUM_OK;
}


