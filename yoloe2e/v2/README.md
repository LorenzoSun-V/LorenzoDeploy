# YOLO Series End-to-End TensorRT C++ 

## Support
[YOLOv9](https://github.com/WongKinYiu/yolov9)、[YOLOv8](https://v8docs.ultralytics.com/)、[YOLOv7](https://github.com/WongKinYiu/yolov7)、[YOLOv6](https://github.com/meituan/YOLOv6)、 [YOLOV5](https://github.com/ultralytics/yolov5)

- [x] YOLOv9
- [x] YOLOv8
- [x] YOLOv7
- [x] YOLOv6
- [x] YOLOv5

##  Prepare Model

Prepare python environment.

```
pip install onnx_graphsurgeon
pip install onnx
pip install numpy
pip install ultralytics
```

1. Prepare onnx model

* YOLOv5/v7

    Use `export.py` in [YOLOV5](https://github.com/ultralytics/yolov5)&[YOLOv7](https://github.com/WongKinYiu/yolov7) to get onnx model.

    ```
    python export.py --weights ${model_path} --imgsz {img_size} --include onnx --opset 12
    ```

    Use `run_yolo_nms.sh` in `python/yolo` to get onnx_nms model. The final onnx_nms model will be saved in the same dir of onnx_postprocess model.

    ```
    sh run_yolo_nms.sh
    ```

    * input_model: input onnx model
    * outputmodel: output onnx_postprocess model
    * numcls: number of classes 

* YOLOv8/YOLOv9

    Use `yolo` to get onnx model.
    ```
    yolo export model=${torch_model} format=onnx imgsz=${image_size} opset=12
    ```

    * torch_model: pt model path
    * image_size: image size

    Use `run_yolo_nms.sh` in `python/yolo` to get onnx_nms model. The final onnx_nms model will be saved in the same dir of onnx_postprocess model.

    ```
    sh run_yolo_nms.sh
    ```

    * input_model: input onnx model
    * outputmodel: output onnx_postprocess model
    * numcls: number of classes 

2. Prepare tensorrt engine

    Use binary `trtexec` or [onnx2engine](../../onnx2engine/README.md) in `bt_alg_api` to get tensorrt engine.

    ```
    ./trtexec --onnx=${onnx_nms_path} --saveEngine=${output_engine_path} --workspace=9000 --verbose   
    ```

    * onnx_nms_path: exported onnx_nms model by `run_yolo_nms.sh`
    * output_engine_path: output engine model


## C++ Inference 

1. Build `yoloe2ev2`.

    ```
    cd yoloe2ev2
    sh build.sh
    ```

    * DTRT_PATH: Tensorrt path.

2. Build `test-yoloe2ev2`.

    ```
    cd test-yoloe2ev2
    sh build.sh
    ```

3. Use `test-infer`.

    ```
    cd test-yoloe2ev2/build/test-infer
    ./test-infer ${image_folder} ${engine_path}
    ```

    * image_folder: image folder for inference
    * engine_path: tensorrt engine path generated by `trtexec`

4. Use `test-precious` to record label files of images.

    ```
    cd test-yoloe2ev2/build/test-precious
    ./test-precious ${image_folder} ${engine_path}
    
    ```