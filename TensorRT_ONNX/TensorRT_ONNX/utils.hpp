#ifndef CUSTOM_UTILS_H
#define CUSTOM_UTILS_H

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <io.h> // access
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem> // for create floder 

#include "NvInferRuntime.h"
#include "NvInferPlugin.h"

// CUDA RUNTIME API for cuda error check
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


// Load file name list from given directory path 1
int SearchFile(const std::string& folder_path, std::vector<std::string> &file_names, bool recursive = false);

// Make binary file (serialize) 
template<class T>
void tofile(std::vector<T> &Buffer, std::string fname = "../Validation_py/C_Tensor") {
    std::ofstream fs(fname, std::ios::binary);
    if (fs.is_open())
        fs.write((const char*)Buffer.data(), Buffer.size() * sizeof(T));
    fs.close();
    std::cout << "Done! file production to " << fname << std::endl;
}

// Load binary file (unserialize) 
// example) 
// fromfile(input, "../Unet_py/input_data");
template<class T>
void fromfile(std::vector<T>& Buffer, std::string fname = "../Validation_py/C_Tensor") {
    std::ifstream ifs(fname, std::ios::binary);
    if (ifs.is_open())
        ifs.read((char*)Buffer.data(), Buffer.size() * sizeof(T));
    ifs.close();
    std::cout << "Done! file load from " << fname << std::endl;
}
// Do argmax
// example) 
// std::cout << "index : "<< argMax(output) << " , label name : " << class_names[argMax(output) ] << " , prob : " << output[argMax(output) ] << std::endl;
int argMax(std::vector<float> &output);

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

// Print TRT Tensor dimensions information
void show_dims(nvinfer1::ITensor* tensor);

// Do data pre-processing
void Preprocess(std::vector<float> &output, std::vector<uint8_t>& input, int BatchSize, int channels, int height, int width);

// Load file name list from given directory path 2
int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);


void mkdir(const std::string &path);

#endif