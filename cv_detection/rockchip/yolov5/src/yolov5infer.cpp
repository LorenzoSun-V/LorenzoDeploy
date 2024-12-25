/*
 * @FilePath: /jack/bt_alg_api/cv_detection/rockchip/yolov5/src/yolov5infer.cpp
 * @Description: 瑞星微模型推理接口
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-11-12 18:14:02
 */

#include "yolov5.h"
#include "rkinfer.h"

//初始化推理参数
class RkDetctModelParam {
public:
    bool bParamIsOk;

    YOLOV5Model yolov5model;
    RkDetctModelParam() 
    {
	    bParamIsOk = false;	
    }
};

class RkDetectModelInstance {
public:
   std::shared_ptr<RkDetctModelParam> _param;

   RkDetectModelInstance() {
        _param = std::make_shared<RkDetctModelParam>();
    }
};

bool checkFileExists(const char* pWeightfile) {
    struct stat buffer;
    return (stat(pWeightfile, &buffer) == 0);
}

ENUM_ERROR_CODE InitRKInferenceInstance(const char* pWeightfile, int classnum, void** pDeepInstance)
{
    if (pWeightfile == NULL)
    {
        std::cerr << "input pWeightfile is NULL!" << std::endl;
	    return ERR_INVALID_PARAM;
    }

    if (!checkFileExists(pWeightfile)) {
        std::cerr << "Error: File " << pWeightfile << " does not exist!" << std::endl;
        return ERR_MODEL_INPUTPATH_NOT_EXIST;
    }

    //返回句柄指针
    RkDetectModelInstance* _instance =  new RkDetectModelInstance(); 
    if (NULL == _instance) {
         std::cout <<  "Init VideoStreamInstance failed." << std::endl;
        return ERR_NO_FREE_MEMORY;
    }
    
    ENUM_ERROR_CODE eOK = _instance->_param->yolov5model.loadModel(pWeightfile, classnum);
    if( eOK != ENUM_OK ) {
        delete _instance;
        pDeepInstance = NULL;
        return eOK;
    }

    _instance->_param->bParamIsOk = true; 
    *pDeepInstance = (void*)_instance;  
    return ENUM_OK;
}

ENUM_ERROR_CODE InferenceGetDetectResult(void* pDeepInstance, cv::Mat frame, std::vector<DetBox> &detBoxs)
{
    RkDetectModelInstance* _instance = static_cast<RkDetectModelInstance*>(pDeepInstance);
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "InferenceGetDetectResult pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if( frame.empty() )
    {
	    printf("frame is empty!");
	    return ERR_GET_IMAGE_EMPTY;
	}
    
    ENUM_ERROR_CODE eOK = _instance->_param->yolov5model.inference(frame, detBoxs);
    if( eOK != ENUM_OK ) {
        return eOK;
    }
    
    return ENUM_OK;
}


ENUM_ERROR_CODE DestoryInferenceInstance( void **pDeepInstance)
{
    RkDetectModelInstance* _instance = static_cast<RkDetectModelInstance*>(*pDeepInstance);
    if (!_instance || !_instance->_param->bParamIsOk) {
	    std::cout << "DestoryInferenceInstance pDeepInstance is NULL" << std::endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }	 
	
	if (_instance)
    {
        delete _instance;
        *pDeepInstance = NULL;
    }

    return ENUM_OK;
}
