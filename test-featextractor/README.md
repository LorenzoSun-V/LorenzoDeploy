<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-10-28 10:58:00
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-10-28 10:59:53
 * @Description: 
-->
## 1. 构建并安装 featextractor

```
    cd featextractor
    sh build.sh
```

## 2. 构建并安装 test-featextractor

```
    cd test-featextractor
    sh build.sh
```

* 测试单batch 

    ```
        cd build/test-infer
        ./testinfer ${image_folder} ${model_path}
    ```

    * image_folder: 待推理的图片文件夹路径
    * model_path: tensorrt模型路径

* 测试多batch

    ```
        cd build/test-batchinfer
        ./testb
        atch ${image_folder} ${model_path}
    ```
    
    * image_folder: 待推理的图片文件夹路径
    * model_path: tensorrt模型路径