/*
 * @FilePath: /bt_zs_4x_api/yolov5/src/yolov5infer.cpp
 * @Description: yolov5模型推理接口
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-07-04 11:47:12
 */
#include "yolov5infer.h"
#include "yolov5.h"

class YOLOV5DetctModel {
public:
    bool bParamIsOk;

    YOLOV5Model yolov5model;
    YOLOV5DetctModel() {
    }

    ~YOLOV5DetctModel() {
        bParamIsOk =false;
    }
};

class YOLOV5DetectModelInstance {
public:
   std::shared_ptr<YOLOV5DetctModel> _param;

   YOLOV5DetectModelInstance() {
        _param = std::make_shared<YOLOV5DetctModel>();
    }
};

bool checkFileExists(const char* pWeightsfile) {
    struct stat buffer;
    return (stat(pWeightsfile, &buffer) == 0);
}

ENUM_ERROR_CODE LoadDeepModelModules(const char* pWeightsfile, void** pDeepInstance)
{   
    if (pWeightsfile == NULL)
    {
        std::cerr << "input pWeightsfile is NULL!" << std::endl;
	    return ERR_INVALID_PARAM;
    }
    if (!checkFileExists(pWeightsfile)) {
        std::cerr << "Error: File " << pWeightsfile << " does not exist!" << std::endl;
        return ERR_MODEL_INPUTPATH_NOT_EXIST;
    }

    YOLOV5DetectModelInstance* _instance =  new YOLOV5DetectModelInstance(); 
    if (NULL == _instance) {
         std::cout <<  "Init VideoStreamInstance failed." << std::endl;
        return ERR_NO_FREE_MEMORY;
    }

    bool bOK = _instance->_param->yolov5model.loadModel(pWeightsfile);
    if( !bOK ) {
        delete _instance;
        pDeepInstance = NULL;
        return ERR_MODEL_DESERIALIZE_FAIL;
    }
    _instance->_param->bParamIsOk = true; 
    *pDeepInstance = (void*)_instance;  
    return ENUM_OK;
}

//GPU推理获得检测结果
ENUM_ERROR_CODE InferenceGetDetectResult(void* pDeepInstance, cv::Mat frame, std::vector<DetBox> &detBoxs)
{
    YOLOV5DetectModelInstance* _instance = static_cast<YOLOV5DetectModelInstance*>(pDeepInstance);          
    if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "InferenceGetDetectResult pDeepInstance is NULL" << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }

    if (frame.empty())
    {
        std::cout <<  "Failed to read frame." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }

    _instance->_param->yolov5model.inference(frame, detBoxs);

    return ENUM_OK;
}  


//多batch推理获得检测结果
ENUM_ERROR_CODE BatchInferenceGetDetectResult(void* pDeepInstance, std::vector<cv::Mat> img_batch, std::vector<std::vector<DetBox>> &batchDetBoxs )
{
    YOLOV5DetectModelInstance* _instance = static_cast<YOLOV5DetectModelInstance*>(pDeepInstance);  
    if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "BatchInferenceGetDetectResult pDeepInstance is NULL" << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }
    
    if ( img_batch.empty() )
    {
        std::cout << "Invalid input frames." << std::endl;
        return ERR_INPUT_IMAGE_EMPTY;
    }
    bool bOK = _instance->_param->yolov5model.batchInference(img_batch, batchDetBoxs);
    if(!bOK)
    {
        return ERR_DETECT_OBJECT_EMPTY; 
    }
    return ENUM_OK;
}  

  
ENUM_ERROR_CODE DestoryDeepmodeInstance( void *pDeepInstance)
{
    YOLOV5DetectModelInstance* _instance = static_cast<YOLOV5DetectModelInstance*>(pDeepInstance);  	 
	    if (!_instance || !_instance->_param->bParamIsOk) {
	    cout << "DestoryDeepmodeInstance pDeepInstance is NULL" << endl;
	    return ERR_INPUT_INSTANCE_INVALID;
    }

  	if (_instance)
    {
        delete _instance;
        pDeepInstance = NULL;
    }
         

    return ENUM_OK;
}

