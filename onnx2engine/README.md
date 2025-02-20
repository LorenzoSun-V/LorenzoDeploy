<!--
 * @FilePath: /jack/github/bt_alg_api/cv_detection/nvidia/onnx2engine/readme.md
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2025-02-20 11:51:41
-->
## 设置环境变量

如果提示`error while loading shared libraries: libnvinfer.so.8: cannot open shared object file: No such file or directory`，则需要：
```
export LD_LIBRARY_PATH=${TensorRT_Path}/lib
```

## 模型转换
```
./onnx2engine ${ONNX_Path}
```
