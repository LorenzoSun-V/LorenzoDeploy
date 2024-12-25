#include "feat.h"

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

FeatExtractModel::FeatExtractModel():m_iwidth(640), m_iheight(640), m_channel(3){

}

FeatExtractModel::~FeatExtractModel() {

}

int FeatExtractModel::loadModel(const char* pWeightfile)
{
    //加载模型获得模型长度和模型内存地址指针
    int model_data_size = 0;
    rknn_context ctx;
    unsigned char *model_data = load_model(pWeightfile, &model_data_size);
    if(NULL == model_data){
        std::cout << "load_model failed!" << std::endl;
	    return -1;
    }

    //模型初始化，配置获取实时结果，不进行预处理
    int ret = rknn_init(&ctx, model_data, model_data_size, 0);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -2;
    }
    
    //获得sdk版本   
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -3;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    //获得模型输入输出通道信息
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -4;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    
    //获得输入信息
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -5;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    //获得输出信息
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -6;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    std::vector<float> out_scales;
    std::vector<uint32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    //检查模型通道前置还是后置，与量化相关
    // int channel = 3;
    // int width = 0;
    // int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        m_iwidth = input_attrs[0].dims[0];
        m_iheight = input_attrs[0].dims[1];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        m_iwidth = input_attrs[0].dims[1];
        m_iheight = input_attrs[0].dims[2];
    }

    printf("model input height=%d, width=%d\n", m_iheight, m_iwidth);
    return 0;
}


int FeatExtractModel::inference(cv::Mat frame, std::vector<float>& features)
{
    //将float32数据转为uint8进行推理
    cv::Mat resize_img;
    cv::resize(frame, resize_img, cv::Size(m_iwidth, m_iheight));
    cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
    cv::Mat img_uchar;
    resize_img.convertTo(img_uchar, CV_8U); 
    unsigned char *resize_buf = (uchar*)img_uchar.data;

    //配置推理参数
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = m_iwidth * m_iheight * m_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = resize_buf;

    rknn_inputs_set(ctx, io_num.n_input, inputs);
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));

    std::cout << "io_num.n_output: "<<io_num.n_output << std::endl;
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }
    
    //运行推理
    int ret = rknn_run(ctx, NULL);
    if(ret < 0 ){
        std::cout << "ret is failed" << std::endl;
	    return -1;
    }
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if(ret < 0 ){
        std::cout << "ret is empty" << std::endl;
	    return -2;
    }

    float* output_buf = static_cast<float*>(outputs[0].buf);
   
    for(int i = 0; i < io_num.n_output; i++) {
        features.push_back(output_buf[i]);
    }

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    return 0;
}