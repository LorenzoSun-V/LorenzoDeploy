#include "yolov5.h"

//获得模型数据类型 
inline const char *get_type_string(rknn_tensor_type type)
{
    switch (type)
    {
    case RKNN_TENSOR_FLOAT32:
        return "FP32";
    case RKNN_TENSOR_FLOAT16:
        return "FP16";
    case RKNN_TENSOR_INT8:
        return "INT8";
    case RKNN_TENSOR_UINT8:
        return "UINT8";
    case RKNN_TENSOR_INT16:
        return "INT16";
    default:
        return "UNKNOW";
    }
}

inline const char *get_qnt_type_string(rknn_tensor_qnt_type type)
{
    switch (type)
    {
    case RKNN_TENSOR_QNT_NONE:
        return "NONE";
    case RKNN_TENSOR_QNT_DFP:
        return "DFP";
    case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:
        return "AFFINE";
    default:
        return "UNKNOW";
    }
}

//获得输入通道格式
inline const char *get_format_string(rknn_tensor_format fmt)
{
    switch (fmt)
    {
    case RKNN_TENSOR_NCHW:
        return "NCHW";
    case RKNN_TENSOR_NHWC:
        return "NHWC";
    default:
        return "UNKNOW";
    }
}

//打印模型信息
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

//遍历数据
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

//模型加载
static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}
YOLOV5Model::YOLOV5Model(){

}

YOLOV5Model::~YOLOV5Model() {

}

ENUM_ERROR_CODE YOLOV5Model::loadModel(const char* pWeightfile, int class_num)
{
    classnum = class_num;
     //加载模型
    int model_data_size = 0;
    rknn_context ctx;
    unsigned char *model_data = load_model(pWeightfile, &model_data_size);
    if(NULL == model_data){
        std::cout << "load_model failed!" << std::endl;
	    return ERR_ROCKCHIP_LOAD_MODEL_FAILED;
    }
    //模型初始化，配置获取实时结果，不进行预处理
    int ret = rknn_init(&ctx, model_data, model_data_size, 0);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return ERR_ROCKCHIP_RKNN_INIT_FAILED;
    }
     //获得sdk版本   
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return ERR_ROCKCHIP_QUERY_VERSION_FAILED;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version,
           version.drv_version);

    //获得模型输入输出通道信息
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return ERR_ROCKCHIP_QUERY_IN_OUT_HEAD_FAILED;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);
    
    //获得输入输出属性
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return ERR_ROCKCHIP_QUERY_INPUT_ATTR_FAILED;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
        if(output_attrs[i].qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC || output_attrs[i].type != RKNN_TENSOR_UINT8)
        {
            fprintf(stderr,"The model required for a Affine asymmetric u8 quantized rknn model, but output quant type is %s, output data type is %s\n", 
                    get_qnt_type_string(output_attrs[i].qnt_type),get_type_string(output_attrs[i].type));
            return ERR_ROCKCHIP_NOT_UINT8_TYPE;
        }
    }
    
    std::vector<float> out_scales;
    std::vector<uint32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    //检查模型通道前置还是后置，与量化相关
    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        width = input_attrs[0].dims[0];
        height = input_attrs[0].dims[1];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        width = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
    }

    printf("model input height=%d, width=%d, channel=%d\n", height, width,
           channel);
    return ENUM_OK;
}


ENUM_ERROR_CODE YOLOV5Model::inference(cv::Mat frame, std::vector<DetBox>& detBoxs)
{
 //配置nms参数
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    int img_height= frame.rows;
    int img_width = frame.cols;

    //将float32数据转为uint8进行推理
    cv::Mat resize_img;
    cv::resize(frame, resize_img, cv::Size(width, height));
    cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);

    cv::Mat img_uchar;
    resize_img.convertTo(img_uchar, CV_8U); 
    unsigned char *resize_buf = (uchar*)img_uchar.data;
    //配置推理参数
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = resize_buf;

    rknn_inputs_set(ctx, io_num.n_input, inputs);
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }
    //运行推理
    int ret = rknn_run(ctx, NULL);
    if(ret < 0 ){
        std::cout << "ret is failed" << std::endl;
	    return ERR_ROCKCHIP_RUN_FAILED;
    }
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if(ret < 0 ){
        std::cout << "ret is empty" << std::endl;
	    return ERR_ROCKCHIP_OUTPUT_GET_FAILED;
    }
    //配置原始图像缩放因子
    float scale_w = (float)width / img_width;
    float scale_h = (float)height / img_height;
    //进行后处理获得结果
    detect_result_group_t detect_result_group;
    post_process(classnum, (uint8_t *)outputs[0].buf, (uint8_t *)outputs[1].buf, (uint8_t *)outputs[2].buf, height, width,
                box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
    //将结果输出
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        DetBox detres;   
        detres.classID =  det_result->classId;   
        detres.confidence =  det_result->prop;

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        detres.x = x1;
        detres.y = y1;                       
        detres.w = x2-x1;  
        detres.h = y2-y1;  
        detBoxs.push_back(detres);  
        std::cout<< detres.x<<" "<< detres.y <<" "<<detres.w<<" "<<detres.h<<" "<< detres.confidence<<" "<<detres.classID<<std::endl; 
    }
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    return ENUM_OK;
}