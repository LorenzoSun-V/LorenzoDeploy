<!--
 * @FilePath: /jack/github/bt_alg_api/cv_segmentation/nvidia/test-yolov5seg/readme.md
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-26 15:34:09
-->
## 设置环境变量
```
export LD_LIBRARY_PATH=/home/sysadmin/jack/nvlibs
```
## 测试单batch推理
```
./build/test-infer/test-infer   /home/sysadmin/jack/images/seg/ /home/sysadmin/jack/models/yolov5m-seg_b1.bin 
```