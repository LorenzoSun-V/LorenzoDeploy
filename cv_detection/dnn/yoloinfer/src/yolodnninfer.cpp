/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-12-23 08:59:24
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-23 15:47:45
 * @Description: 
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "yolodnn.h"
#include "yolodnninfer.h"

class YOLOModelParam {
public:
    bool bParamIsOk;
    YOLODNNModelManager yolodnnmodel;
    YOLOModelParam() 
    {
	    bParamIsOk = false;	
    }
};

class YOLOModelInstance {
public:
    std::shared_ptr<YOLOModelParam> _param;

    YOLOModelInstance() {
        _param = std::make_shared<YOLOModelParam>();
    }
};

ENUM_ERROR_CODE LoadDNNModelModules(const char* pWeightsfile, void** pDNNInstance)
{   
    if (pWeightsfile == NULL){
        std::cerr << "input pWeightsfile is NULL!" << std::endl;
	    return ERR_INVALID_PARAM;
    }

    //load dnn model     
    YOLOModelInstance* _instance =  new YOLOModelInstance(); 
    if (NULL == _instance) {
         std::cout <<  "Init YOLOModelInstance failed." << std::endl;
        return ERR_NO_FREE_MEMORY;
    }
    bool bOk = _instance->_param->yolodnnmodel.loadModel(pWeightsfile);
    if( !bOk ) {
        delete _instance;
        pDNNInstance = NULL;
        return ERR_MODEL_DESERIALIZE_FAIL;
    }

    _instance->_param->bParamIsOk = true; 
    *pDNNInstance = (void*)_instance;  

    return ENUM_OK;
}

ENUM_ERROR_CODE InferenceDNNGetDetectResult(void* pDNNInstance, cv::Mat frame, std::vector<DetBox> &detBoxs)
{
    YOLOModelInstance* _instance = static_cast<YOLOModelInstance*>(pDNNInstance); 
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "InferenceGetDetectResult pDNNInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if (frame.empty()) {
        std::cout <<  "Failed to read frame." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    bool bOK = _instance->_param->yolodnnmodel.inference(frame, detBoxs);
    if(!bOK) {
        return ERR_DETECT_OBJECT_EMPTY;
    }
    return ENUM_OK;
}  

ENUM_ERROR_CODE DestoryDNNModeInstance( void **pDNNInstance)
{
    YOLOModelInstance* _instance = static_cast<YOLOModelInstance*>(*pDNNInstance); 	 
	if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "DestoryDNNmodeInstance pDNNInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
	if (_instance)
    {    
        delete _instance;
        *pDNNInstance = NULL;
    }   
    return ENUM_OK;
}

