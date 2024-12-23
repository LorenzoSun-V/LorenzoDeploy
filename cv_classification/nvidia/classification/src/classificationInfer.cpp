/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-07-03 10:01:14
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-12-20 14:55:22
 * @Description: 
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "classification.h"
#include "classificationInfer.h"

class ClasssificationParam{
public:
    bool bParamIsOk;

    ClasssificationModel classsificationmodel;

    ClasssificationParam() 
    {
	    bParamIsOk = false;	
    }
};

class ClasssificationInstance {
public:
    std::shared_ptr<ClasssificationParam> _param;
    
    ClasssificationInstance() {
          _param = std::make_shared<ClasssificationParam>();
     }
};

ENUM_ERROR_CODE LoadDeepModelModules(const char* pWeightsfile, void** pDeepInstance)
{   
    if ( pWeightsfile == NULL)
    {
        std::cout << "LoadDeepModelModules input weights file is NULL!" << std::endl;
	    return ERR_INVALID_PARAM;
    }

    //load dnn model     
    ClasssificationInstance* _instance =  new ClasssificationInstance(); 
    if (NULL == _instance) {
        std::cout <<  "Init ClasssificationInstance failed." << std::endl;
        return ERR_NO_FREE_MEMORY;
    }

    if (!_instance->_param->classsificationmodel.loadModel(pWeightsfile)){
        delete _instance;
        pDeepInstance = NULL;
        return ERR_MODEL_DESERIALIZE_FAIL;
    }

    _instance->_param->bParamIsOk = true; 
    *pDeepInstance = (void*)_instance;  

    return ENUM_OK;
}

ENUM_ERROR_CODE InferenceGetFeature(void* pDeepInstance, cv::Mat frame, std::vector<ClsResult> &objects, int topK)
{  
    ClasssificationInstance* _instance = static_cast<ClasssificationInstance*>(pDeepInstance);
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "InferenceGetFeature pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if (frame.empty()){
        std::cout <<  "Failed to read frame." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }

    if (!_instance->_param->classsificationmodel.inference(frame, objects, topK)) {
        return ERR_DETECT_OBJECT_EMPTY;
    }
    return ENUM_OK;
}  


ENUM_ERROR_CODE DestoryDeepmodeInstance(void **pDeepInstance)
{
    ClasssificationInstance* _instance = static_cast<ClasssificationInstance*>(*pDeepInstance);
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