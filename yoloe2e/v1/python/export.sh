#! /bin/bash
###
 # @FilePath: /bt_alg_api/yoloe2e/model/export.sh
 # @Description: 接口代码
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-07-26 15:24:44
### 

#python3 export.py -o /home/mic-710aix/model/model8_b16m_20240627_cls2_kjg_v0.2.1.onnx  -e /home/mic-710aix/model/model8_b16m_20240627_cls2_kjg_v0.2.1_e2e.bin   --end2end --v8 -p fp16

python3 export.py -o /home/mic-710aix/model/model10_b16m_20240701_cls2_kjg_v0.2.1.onnx  -e /home/mic-710aix/model/model10_b16m_20240701_cls2_kjg_v0.2.1_e2e.bin   --end2end --v10 -p fp16
