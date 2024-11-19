/*
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-07-04 14:10:35
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-10-17 13:16:41
 * @Description: 
 */
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <map>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

struct input_search_param {
    int feature_id;
    std::vector<float> feature_value;
};

float calculEuclideanDistance(std::vector<float> feature1, std::vector<float> feature2, int feature_dim) {
    float diff = 0.0, sum = 0.0;
    for (int i = 0; i < feature_dim; i++) {
        diff = feature1[i] - feature2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

float calculCosineSimilar(const std::vector<float>& feature1, const std::vector<float>& feature2, int feature_dim) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int i = 0; i < feature_dim; ++i) {
        dot_product += feature1[i] * feature2[i];
        norm_a += feature1[i] * feature1[i];
        norm_b += feature2[i] * feature2[i];
    }
    return dot_product / (sqrt(norm_a) * sqrt(norm_b));
}

std::vector<int> searchTopkEuclideanDistance(const std::vector<float>& input_feat, const std::vector<input_search_param>& search_lib, int topk = 1) {
    // 确保输入特征向量和搜索库中的特征向量非空且维度一致
    if (input_feat.empty() || search_lib.empty() || search_lib[0].feature_value.size() != input_feat.size()) {
        throw std::invalid_argument("Invalid input dimensions.");
    }

    int n = input_feat.size();
    int m = search_lib.size();

    Eigen::VectorXf input_vec = Eigen::Map<const Eigen::VectorXf>(input_feat.data(), n);

    // 创建一个字典来存储特征的ID和对应的欧氏距离
    std::vector<std::pair<int, float>> distances;

    // 遍历搜索库中的特征向量，计算欧氏距离
    for (const auto& item : search_lib) {
        Eigen::VectorXf lib_vec = Eigen::Map<const Eigen::VectorXf>(item.feature_value.data(), n);
        float dist = (input_vec - lib_vec).squaredNorm();
        distances.push_back(std::make_pair(item.feature_id, dist));
    }

    // 对字典中的距离进行排序，按从小到大排序
    std::sort(distances.begin(), distances.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second < b.second;
    });

    // 取前topk个最小的距离对应的特征ID
    std::vector<int> result;
    for (int i = 0; i < topk; ++i) {
        result.push_back(distances[i].first);
    }

    return result;
}

std::vector<int> searchTopkCosineDistance(const std::vector<float>& input_feat, const std::vector<input_search_param>& search_lib, int topk = 1) {
    // 确保输入特征向量和搜索库中的特征向量非空且维度一致
    if (input_feat.empty() || search_lib.empty() || search_lib[0].feature_value.size() != input_feat.size()) {
        throw std::invalid_argument("Invalid input dimensions.");
    }

    int n = input_feat.size();
    int m = search_lib.size();

    Eigen::VectorXf input_vec = Eigen::Map<const Eigen::VectorXf>(input_feat.data(), n);

    // 创建一个字典来存储特征的ID和对应的余弦相似度
    std::vector<std::pair<int, float>> similarities;

    // 计算输入特征向量的范数
    float input_norm = input_vec.norm();

    // 遍历搜索库中的特征向量，计算欧氏距离
    for (const auto& item : search_lib) {
        Eigen::VectorXf lib_vec = Eigen::Map<const Eigen::VectorXf>(item.feature_value.data(), n);
        // 计算搜索库中特征向量的范数
        float lib_norm = lib_vec.norm();

        // 计算余弦相似度
        float cosine_similarity = input_vec.dot(lib_vec) / (input_norm * lib_norm);

        // 将特征ID和对应的相似度存入字典
        similarities.push_back(std::make_pair(item.feature_id, cosine_similarity));
    }

    // 对字典中的相似度进行排序，按从大到小排序
    std::sort(similarities.begin(), similarities.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
    });

    // 取前topk个最大的相似度对应的特征ID
    std::vector<int> result;
    for (int i = 0; i < topk; ++i) {
        result.push_back(similarities[i].first);
    }

    return result;
}

std::vector<int> searchTopk2EuclideanDistance(const std::vector<float>& input_feat, const std::vector<input_search_param>& search_lib, int topk = 1) {
    // 确保输入特征向量和搜索库中的特征向量非空且维度一致
    if (input_feat.empty() || search_lib.empty() || search_lib[0].feature_value.size() != input_feat.size()) {
        throw std::invalid_argument("Invalid input dimensions.");
    }

    int n = input_feat.size();
    int m = search_lib.size();

    // 创建一个字典来存储特征的ID和对应的欧氏距离
    std::vector<std::pair<int, float>> distances;

    // 遍历搜索库中的特征向量，计算欧氏距离
    for (const auto& item : search_lib) {
        float dist = calculEuclideanDistance(input_feat, item.feature_value, n);
        distances.push_back(std::make_pair(item.feature_id, dist));
    }

    // 对字典中的距离进行排序，按从小到大排序
    std::sort(distances.begin(), distances.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second < b.second;
    });

    // 取前topk个最小的距离对应的特征ID
    std::vector<int> result;
    for (int i = 0; i < topk; ++i) {
        result.push_back(distances[i].first);
    }

    return result;
}

std::vector<int> searchTopk2CosineDistance(const std::vector<float>& input_feat, const std::vector<input_search_param>& search_lib, int topk = 1) {
    // 确保输入特征向量和搜索库中的特征向量非空且维度一致
    if (input_feat.empty() || search_lib.empty() || search_lib[0].feature_value.size() != input_feat.size()) {
        throw std::invalid_argument("Invalid input dimensions.");
    }

    int n = input_feat.size();
    int m = search_lib.size();

    // 创建一个字典来存储特征的ID和对应的余弦相似度
    std::vector<std::pair<int, float>> similarities;

    // 遍历搜索库中的特征向量，计算欧氏距离
    for (const auto& item : search_lib) {
        float cosine_similarity = calculCosineSimilar(input_feat, item.feature_value, n);
        similarities.push_back(std::make_pair(item.feature_id, cosine_similarity));
    }

    // 对字典中的相似度进行排序，按从大到小排序
    std::sort(similarities.begin(), similarities.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
    });

    // 取前topk个最大的相似度对应的特征ID
    std::vector<int> result;
    for (int i = 0; i < topk; ++i) {
        result.push_back(similarities[i].first);
    }

    return result;
}

// 从文件中读取特征向量
std::vector<float> readFeatureFromFile(const std::string& file_path) {
    std::vector<float> feature;
    std::ifstream infile(file_path);
    if (!infile) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return feature;
    }

    float value;
    while (infile >> value) {
        feature.push_back(value);
    }

    return feature;
}

// 从目录中读取特征库
std::vector<input_search_param> readSearchLibFromDirectory(const std::string& dir_path) {
    std::vector<input_search_param> search_lib;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_name = entry.path().filename().string();
            std::string file_path = entry.path().string();

            // 提取 feature_id（假设文件名的前4个字符是ID）
            int feature_id = std::stoi(file_name.substr(0, 4));

            // 读取特征向量
            std::vector<float> feature_value = readFeatureFromFile(file_path);

            // 存储到 search_lib
            search_lib.push_back({feature_id, feature_value});
        }
    }
    return search_lib;
}


int main() {
    // test1：随机生成向量进行测试
    std::srand(static_cast<unsigned int>(std::time(0)));
    // 生成随机的512维特征向量
    std::vector<float> input_feat(512);
    for (int i = 0; i < 512; ++i) {
        input_feat[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    // 生成随机的100x512特征向量矩阵
    std::vector<input_search_param> search_lib(100);
    for (int i = 0; i < 100; ++i) {
        search_lib[i].feature_id = i;
        search_lib[i].feature_value.resize(512);
        for (int j = 0; j < 512; ++j) {
            search_lib[i].feature_value[j] = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    // test2：固定值进行测试
    // std::vector<float> input_feat = {1.0, 2.0, 3.0, 4.0};
    // std::vector<input_search_param> search_lib = {
    //     {1, {1.1, 2.1, 3.1, 4.1}},
    //     {2, {2.0, 3.0, 4.0, 5.0}},
    //     {3, {0.9, 1.9, 2.9, 3.9}}
    // };

    // test3：提取的特征进行测试
    // std::string input_feat_path = "/lorenzo/bt_repo/bt_zs_4x_api/test-featextractor/build/test-batchinfer/1006-14_crop_cls0_obj0.txt";
    // std::string search_lib_dir = "/lorenzo/bt_repo/bt_zs_4x_api/test-featextractor/build/test-batchinfer/search_lib/";

    // // 从文件中读取输入特征
    // std::vector<float> input_feat = readFeatureFromFile(input_feat_path);
    // if (input_feat.empty()) {
    //     std::cerr << "Failed to read input feature." << std::endl;
    //     return -1;
    // }

    // // 从目录中读取特征库
    // std::vector<input_search_param> search_lib = readSearchLibFromDirectory(search_lib_dir);
    // if (search_lib.empty()) {
    //     std::cerr << "Failed to read search library." << std::endl;
    //     return -1;
    // }

    // 查找 top k 最相似的特征
    int topk = 1;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> topk_ids = searchTopkEuclideanDistance(input_feat, search_lib, topk);
    auto end = std::chrono::high_resolution_clock::now();
    std::vector<int> topk_ids2 = searchTopk2EuclideanDistance(input_feat, search_lib, topk);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::vector<int> topk_ids3 = searchTopkCosineDistance(input_feat, search_lib, topk);
    auto end3 = std::chrono::high_resolution_clock::now();
    std::vector<int> topk_ids4 = searchTopk2CosineDistance(input_feat, search_lib, topk);
    auto end4 = std::chrono::high_resolution_clock::now();

    std::cout << "searchTopk time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    std::cout << "searchTopk2 time: " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - end).count() << " us" << std::endl;
    std::cout << "searchTopk3 time: " << std::chrono::duration_cast<std::chrono::microseconds>(end3 - end2).count() << " us" << std::endl;
    std::cout << "searchTopk4 time: " << std::chrono::duration_cast<std::chrono::microseconds>(end4 - end3).count() << " us" << std::endl;

    // 输出结果
    std::cout << "Top " << topk << " most similar feature IDs: ";
    for (int id : topk_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    std::cout << "Top " << topk << " most similar feature IDs: ";
    for (int id : topk_ids2) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    std::cout << "Top " << topk << " most similar feature IDs: ";
    for (int id : topk_ids3) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    std::cout << "Top " << topk << " most similar feature IDs: ";
    for (int id : topk_ids4) {
        std::cout << id << " ";
    }

    return 0;
}
