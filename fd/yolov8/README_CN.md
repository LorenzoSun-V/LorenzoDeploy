[English](README.md) | 简体中文
# YOLOv8 C++部署示例

本目录下提供`infer.cc`快速完成YOLOv8在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.3以上(x.x.x>=1.0.3)

```bash
mkdir build
cd build
# 下载 FastDeploy 预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 1. 下载官方转换好的 YOLOv8 ONNX 模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov8s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU推理
./infer_demo yolov8s.onnx 000000014439.jpg 0
# GPU推理
./infer_demo yolov8s.onnx 000000014439.jpg 1
# GPU上TensorRT推理
./infer_demo yolov8s.onnx 000000014439.jpg 2
```
运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184309358-d803347a-8981-44b6-b589-4608021ad0f4.jpg">

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

如果用户使用华为昇腾NPU部署, 请参考以下方式在部署前初始化部署环境:
- [如何使用华为昇腾NPU部署](../../../../../docs/cn/faq/use_sdk_on_ascend.md)

## YOLOv8 C++接口

### YOLOv8类

```c++
fastdeploy::vision::detection::YOLOv8(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

YOLOv8模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX时，此参数传入空字符串即可
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX格式

#### Predict函数

> ```c++
> YOLOv8::Predict(cv::Mat* im, DetectionResult* result)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度, DetectionResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)

### 类成员变量
#### 预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **size**(vector&lt;int&gt;): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[640, 640]
> > * **padding_value**(vector&lt;float&gt;): 通过此参数可以修改图片在resize时候做填充(padding)的值, 包含三个浮点型元素, 分别表示三个通道的值, 默认值为[114, 114, 114]
> > * **is_no_pad**(bool): 通过此参数让图片是否通过填充的方式进行resize, `is_no_pad=ture` 表示不使用填充的方式，默认值为`is_no_pad=false`
> > * **is_mini_pad**(bool): 通过此参数可以将resize之后图像的宽高设置为最接近`size`成员变量的值, 并且满足填充的像素大小是可以被`stride`成员变量整除的。默认值为`is_mini_pad=false`
> > * **stride**(int): 配合`stris_mini_pad`成员变量使用, 默认值为`stride=32`

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
