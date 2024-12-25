<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-11-07 10:43:03
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-11-07 15:09:28
 * @Description: 
-->
## 英伟达图像分类

### 1. featextractor和test-featextractor

基于英伟达硬件的特征提取代码，目前支持单图和多图特征提取。

## 瑞星微图像分类

### 1. featextractor与test-featextractor
基于瑞星微硬件的特征提取代码，目前支持单图和多图特征提取，在RK3399pro设备上测试通过，去除rag入口参数改为opencv读图进行模型推理。

## 相似度计算代码
### 1. calsimilarity
输入两组特征向量，通过欧氏距离和余弦相似度计算两组特征之间的相似度，测试代码可以从系列数据集中找出最相似的ID。