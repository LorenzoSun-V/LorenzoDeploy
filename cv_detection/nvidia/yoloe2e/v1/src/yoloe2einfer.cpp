/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-11-19 15:16:36
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-12-23 15:53:39
 * @Description: 
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "yoloe2e.h"
#include "yoloe2einfer.h"

class YOLOE2EModelParam {
public:
    bool bParamIsOk;

    YOLOE2EModelManager yoloe2emodel;

    YOLOE2EModelParam() 
    {
	    bParamIsOk = false;	
    }
};

class YOLOE2EModelInstance {
public:
   std::shared_ptr<YOLOE2EModelParam> _param;

   YOLOE2EModelInstance() {
        _param = std::make_shared<YOLOE2EModelParam>();
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
    YOLOE2EModelInstance* _instance =  new YOLOE2EModelInstance(); 
    if (NULL == _instance) {
         std::cout <<  "Init YOLOE2EModelInstance failed." << std::endl;
        return ERR_NO_FREE_MEMORY;
    }
    
    if (!_instance->_param->yoloe2emodel.loadModel(pWeightsfile)){
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
    YOLOE2EModelInstance* _instance = static_cast<YOLOE2EModelInstance*>(pDeepInstance); 
    if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "InferenceGetDetectResult pDeepInstance is NULL" << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if (frame.empty())
    {
        std::cout <<  "Failed to read frame." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    if (!_instance->_param->yoloe2emodel.inference(frame, detBoxs)){
        return ERR_DETECT_OBJECT_EMPTY;
    }
    return ENUM_OK;
}
  
ENUM_ERROR_CODE DestoryDeepmodeInstance(void **pDeepInstance)
{
    YOLOE2EModelInstance* _instance = static_cast<YOLOE2EModelInstance*>(*pDeepInstance); 	 
	if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "DestoryDeepmodeInstance pDeepInstance is NULL" << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
	if (_instance)
    {    
        delete _instance;
        *pDeepInstance = NULL;   
    }
        
    return ENUM_OK;
}

