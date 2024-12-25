/*
 * @FilePath: /jack/github/bt_alg_api/cv_classification/nvidia/classification/utils/postprocess.cpp
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-20 15:45:13
 */
#include "postprocess.h"


// 元素交换
static void swap(element_t* a, element_t* b) {
    element_t temp = *a;
    *a = *b;
    *b = temp;
}

// 快速排序的分区函数
static int partition(element_t arr[], int low, int high) {
    float pivot = arr[high].value;
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j].value >= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

// 快速排序
static void quick_sort(element_t arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

// 获取Top-K及其索引
static void get_topk_with_indices(float arr[], int size, int k, ClsResult* result) {
    element_t* elements = (element_t*)malloc(size * sizeof(element_t));
    for (int i = 0; i < size; i++) {
        elements[i].value = arr[i];
        elements[i].index = i;
    }

    quick_sort(elements, 0, size - 1);

    for (int i = 0; i < k; i++) {
        result[i].score = elements[i].value;
        result[i].class_id = elements[i].index;
    }

    free(elements);
}

// Softmax函数
void softmax(std::vector<std::vector<float>> output, int classnum, int batchsize, int top_k, std::vector<ClsResult>& cls_rets) 
{
    std::cout << "top_k: " << top_k << std::endl;

    for (int batch_idx = 0; batch_idx < batchsize; ++batch_idx) {
        const std::vector<float>& array = output[batch_idx];

        if (array.size() != static_cast<size_t>(classnum)) {
            std::cerr << "Error: Input size does not match the expected class number!" << std::endl;
            continue;
        }

        // Softmax计算
        std::vector<float> softmax_array = array; // 复制一份进行计算

        float max_val = *std::max_element(softmax_array.begin(), softmax_array.end());

        for (float& val : softmax_array) {
            val -= max_val; // 防止溢出，做数值稳定化处理
        }

        float sum = 0.0f;
        for (float& val : softmax_array) {
            val = std::exp(val);
            sum += val;
        }

        for (float& val : softmax_array) {
            val /= sum; // 归一化
        }

 
        // 调用 get_topk_with_indices
        ClsResult* result = new ClsResult[top_k];
        get_topk_with_indices(softmax_array.data(), classnum, top_k, result);

        // 将每个Top-K结果加入cls_rets
        for (int i = 0; i < top_k; ++i) {
            cls_rets.push_back(result[i]);
        }

        delete[] result;
    }
}
