#! /bin/bash

input_model=/home/jack/data1/project/lorenzo/yolov5-v7.0/runs/kjg/model5_b16m_20240701_cls2_kjg_v0.2.1/weights/best.onnx
outputmodel=/home/jack/data1/project/lorenzo/yolov5-v7.0/runs/kjg/model5_b16m_20240701_cls2_kjg_v0.2.1/weights/best_1.onnx
numcls=2

python add_postprocess.py --inputmodel ${input_model} --outputmodel ${outputmodel} --numcls ${numcls}
python add_nms.py --model ${outputmodel} --numcls ${numcls}