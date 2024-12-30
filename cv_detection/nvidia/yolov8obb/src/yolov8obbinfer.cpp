/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-20 13:37:13
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-23 15:55:16
 * @Description: YOLOv8OBB模型GPU推理接口
 */

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "yolov8obb.h"
#include "yolov8obbinfer.h"

class YOLOV8OBBModelParam {
public:
    bool bParamIsOk;

    YOLOV8OBBModel yolov8obbmodel;

    YOLOV8OBBModelParam() 
    {
	    bParamIsOk = false;	
    }
};

class YOLOV8OBBModelInstance {
public:
   std::shared_ptr<YOLOV8OBBModelParam> _param;

   YOLOV8OBBModelInstance() {
        _param = std::make_shared<YOLOV8OBBModelParam>();
    }
};

ENUM_ERROR_CODE LoadDeepModelModules(const char* pWeightsfile, void** pDeepInstance)
{   
    if ( pWeightsfile == NULL) {
        std::cerr << "LoadDeepModelModules input weights file is NULL!" << std::endl;
	    return ERR_INVALID_PARAM;
    }

    //load dnn model     
    YOLOV8OBBModelInstance* _instance =  new YOLOV8OBBModelInstance(); 
    if (NULL == _instance) {
         std::cerr <<  "Init YOLOV8OBBModelInstance failed." << std::endl;
        return ERR_NO_FREE_MEMORY;
    }
    
    if (!_instance->_param->yolov8obbmodel.loadModel(pWeightsfile)){
        delete _instance;
        pDeepInstance = NULL;
        return ERR_MODEL_DESERIALIZE_FAIL;
    }

    _instance->_param->bParamIsOk = true; 
    *pDeepInstance = (void*)_instance;  

    return ENUM_OK;
}

ENUM_ERROR_CODE InferenceGetDetectResult(void* pDeepInstance, cv::Mat frame, std::vector<DetBox> &detBoxs)
{
    YOLOV8OBBModelInstance* _instance = static_cast<YOLOV8OBBModelInstance*>(pDeepInstance); 
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cerr << "InferenceGetDetectResult pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if (frame.empty())
    {
        std::cerr <<  "Failed to read frame." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }

    if (!_instance->_param->yolov8obbmodel.inference(frame, detBoxs)){
        return ERR_DETECT_OBJECT_EMPTY;
    }
    return ENUM_OK;
}

ENUM_ERROR_CODE BatchInferenceGetDetectResult(void* pDeepInstance, std::vector<cv::Mat> batch_images, std::vector<std::vector<DetBox>> &batch_result)
{
    YOLOV8OBBModelInstance* _instance = static_cast<YOLOV8OBBModelInstance*>(pDeepInstance); 
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cerr << "BatchInferenceGetDetectResult pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if ( batch_images.empty() )
    {
        std::cerr << "Invalid input frames." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }

    if (_instance->_param->yolov8obbmodel.batch_inference(batch_images, batch_result)){
        return ERR_DETECT_OBJECT_EMPTY;
    }
    return ENUM_OK;
}
  
  
ENUM_ERROR_CODE DestoryDeepmodeInstance(void **pDeepInstance)
{
    YOLOV8OBBModelInstance* _instance = static_cast<YOLOV8OBBModelInstance*>(*pDeepInstance); 	 
	if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cerr << "DestoryDeepmodeInstance pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
	if (_instance)
    {    
        delete _instance;
        *pDeepInstance = NULL;
    }
        
    return ENUM_OK;
}

