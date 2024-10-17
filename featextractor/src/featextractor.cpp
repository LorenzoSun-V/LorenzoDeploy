/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-07-03 10:01:14
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-09-03 17:38:29
 * @Description: 
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "featextract.h"
#include "featextractor.h"

class FeatExtractorParam{
public:
    bool bParamIsOk;

    FeatExtractModel featextractmodel;

    FeatExtractorParam() 
    {
	    bParamIsOk = false;	
    }
};

class FeatExtractorInstance {
public:
    std::shared_ptr<FeatExtractorParam> _param;
    
    FeatExtractorInstance() {
          _param = std::make_shared<FeatExtractorParam>();
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
    FeatExtractorInstance* _instance =  new FeatExtractorInstance(); 
    if (NULL == _instance) {
        std::cout <<  "Init FeatExtractorInstance failed." << std::endl;
        return ERR_NO_FREE_MEMORY;
    }
    if (!_instance->_param->featextractmodel.loadModel(pWeightsfile)){
        delete _instance;
        pDeepInstance = NULL;
        return ERR_MODEL_DESERIALIZE_FAIL;
    }

    _instance->_param->bParamIsOk = true; 
    *pDeepInstance = (void*)_instance;  

    return ENUM_OK;
}

ENUM_ERROR_CODE InferenceGetFeature(void* pDeepInstance, cv::Mat frame, std::vector<float> &objects)
{  
    FeatExtractorInstance* _instance = static_cast<FeatExtractorInstance*>(pDeepInstance);
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "InferenceGetFeature pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if (frame.empty()){
        std::cout <<  "Failed to read frame." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }

    if (!_instance->_param->featextractmodel.inference(frame, objects)){
        return ERR_DETECT_OBJECT_EMPTY;
    }
    return ENUM_OK;
}  

ENUM_ERROR_CODE BatchInferenceGetFeature(void* pDeepInstance, std::vector<cv::Mat> frames, std::vector<std::vector<float>>& batch_objects)
{
    FeatExtractorInstance* _instance = static_cast<FeatExtractorInstance*>(pDeepInstance); 
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "BatchInferenceGetDetectResult pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if (frames.empty())
    {
        std::cout << "Invalid input frames." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    if (!_instance->_param->featextractmodel.batchInference(frames, batch_objects)){
        return ERR_DETECT_OBJECT_EMPTY;
    }

    return ENUM_OK;
}

ENUM_ERROR_CODE DestoryDeepmodeInstance(void **pDeepInstance)
{
    FeatExtractorInstance* _instance = static_cast<FeatExtractorInstance*>(*pDeepInstance);
	if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "DestoryDeepmodeInstance pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
	if (_instance)
    {    
        delete _instance;
        *pDeepInstance = NULL;
    }
         
    return ENUM_OK;
}