/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-06-20 13:37:13
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-07-03 09:02:07
 * @Description: yolov10模型推理接口
 */


#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "yolov10.h"
#include "yolov10infer.h"

class YOLOV10ModelParam {
public:
    bool bParamIsOk;

    YOLOV10ModelManager yolov10model;

    YOLOV10ModelParam() 
    {
	    bParamIsOk = false;	
    }

};

class YOLOV10ModelInstance {
public:
   std::shared_ptr<YOLOV10ModelParam> _param;

   YOLOV10ModelInstance() {
        _param = std::make_shared<YOLOV10ModelParam>();
    }
};

bool checkFileExists(const char* pWeightsfile) {
    struct stat buffer;
    return (stat(pWeightsfile, &buffer) == 0);
}

ENUM_ERROR_CODE LoadDeepModelModules(const char* pWeightsfile, void** pDeepInstance)
{   
    if ( pWeightsfile == NULL)
    {
        std::cout << "LoadDeepModelModules input weights file is NULL!" << std::endl;
	    return ERR_INVALID_PARAM;
    }

    if (!checkFileExists(pWeightsfile)) {
        std::cerr << "Error: File " << pWeightsfile << " does not exist!" << std::endl;
        return ERR_MODEL_INPUTPATH_NOT_EXIST;
    }

    //load dnn model     
    YOLOV10ModelInstance* _instance =  new YOLOV10ModelInstance(); 
    if (NULL == _instance) {
         std::cout <<  "Init YOLOV10ModelInstance failed." << std::endl;
        return ERR_NO_FREE_MEMORY;
    }
    if (!_instance->_param->yolov10model.loadModel(pWeightsfile)){
        delete _instance;
        _instance = NULL;
        return ERR_MODEL_DESERIALIZE_FAIL;
    }

    _instance->_param->bParamIsOk = true; 
    *pDeepInstance = (void*)_instance;  

    return ENUM_OK;
}

ENUM_ERROR_CODE InferenceGetDetectResult(void* pDeepInstance, cv::Mat frame, std::vector<DetBox> &detBoxs)
{
    YOLOV10ModelInstance* _instance = static_cast<YOLOV10ModelInstance*>(pDeepInstance); 
    if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "InferenceGetDetectResult pDeepInstance is NULL" << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if (frame.empty())
    {
        std::cout <<  "Failed to read frame." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    if (!_instance->_param->yolov10model.inference(frame, detBoxs)){
        return ERR_DETECT_OBJECT_EMPTY;
    }
    return ENUM_OK;
}

ENUM_ERROR_CODE BatchInferenceGetDetectResult(void* pDeepInstance, std::vector<cv::Mat> img_batch, std::vector<std::vector<DetBox>> &batchDetBoxs )
{
    YOLOV10ModelInstance* _instance = static_cast<YOLOV10ModelInstance*>(pDeepInstance); 
    if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "BatchInferenceGetDetectResult pDeepInstance is NULL" << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if ( img_batch.empty() )
    {
        std::cout << "Invalid input frames." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    _instance->_param->yolov10model.batchInference(img_batch, batchDetBoxs);

    return ENUM_OK;
}
  
ENUM_ERROR_CODE DestoryDeepmodeInstance(void *pDeepInstance)
{
    YOLOV10ModelInstance* _instance = static_cast<YOLOV10ModelInstance*>(pDeepInstance); 	 
	if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "DestoryDeepmodeInstance pDeepInstance is NULL" << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
	if (_instance)
    {    
        delete _instance;
        _instance = NULL;
    }
         

    return ENUM_OK;
}

