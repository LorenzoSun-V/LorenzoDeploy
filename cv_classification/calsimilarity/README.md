<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-10-28 11:09:44
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-10-28 11:35:52
 * @Description: 
-->

## 利用 Eigen 库进行相似度计算

1. searchTopkEuclideanDistance

    该函数利用Eigen计算向量之间的欧氏距离，并找到距离最小向量的Topk索引。


2. searchTopkCosineDistance

    该函数利用Eigen计算向量之间的余弦距离，并找到距离最大向量的Topk索引。


## 利用计算公式进行相似度计算

1. searchTopk2EuclideanDistance

    该函数利用欧氏距离公式（calculEuclideanDistance函数）计算向量之间的欧氏距离，并找到距离最小向量的Topk索引。

2. searchTopk2CosineDistance

    该函数利用余弦距离公式（calculCosineSimilar函数）计算向量之间的余弦距离，并找到距离最大向量的Topk索引。


## 从文件/文件夹中直接读取特征向量

1. readFeatureFromFile

    该函数直接从文件中读取固定维度的向量值。

2. readSearchLibFromDirectory

    该函数直接从文件夹中的文件读取固定维度的向量值，并保存在vector\<map\>中。**文件夹中的文件名的前4个字符是ID。用于作为map的索引**