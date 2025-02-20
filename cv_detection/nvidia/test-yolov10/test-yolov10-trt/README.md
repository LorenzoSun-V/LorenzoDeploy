<!--
 * @FilePath: /jack/bt_alg_api/cv_detection/nvidia/test-yolov10/test-yolov10-trt/readme.md
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2025-02-20 13:30:32
-->

## 导出ONNX模型

```
yolo export model=yolov10-s.pt format=onnx
```

## 设置环境变量

如果提示`error while loading shared libraries: libopencv_imgproc.so.408: cannot open shared object file: No such file or directory`，则需要：

```
export LD_LIBRARY_PATH=${TensorRT_Path}/lib
```

## 测试单batch推理

```
./test-infer ${engine_path} ${image_folder}
```

- engine_path：engine文件路径

- image_folder：图片文件夹路径

## 测试多batch推理

```
./test-batchinfer ${engine_path} ${image_folder}
```

- engine_path：engine文件路径

- image_folder：图片文件夹路径

## 验证模型精度

```
./test-precious ${engine_path} ${image_folder}
```

- engine_path：engine文件路径

- image_folder：验证集/测试集图片文件夹路径