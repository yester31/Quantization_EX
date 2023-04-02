#include <iostream>
#include <iterator>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "cuda_runtime_api.h"
#include "calibrator.hpp"
#include "utils.hpp"		


Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchsize, int input_c, int input_w, int input_h, const char* img_dir, const char* calib_table_name, const char* input_blob_name, bool read_cache)
    : batchsize_(batchsize)
    , input_c_(input_c)
    , input_w_(input_w)
    , input_h_(input_h)
    , img_idx_(0)
    , img_dir_(img_dir)
    , calib_table_name_(calib_table_name)
    , input_blob_name_(input_blob_name)
    , read_cache_(read_cache)
{
    input_count_ = input_c * input_w * input_h * batchsize;
    input_size_ = input_c * input_w * input_h;
    CHECK(cudaMalloc(&device_input_, input_count_ * sizeof(float)));
    read_files_in_dir(img_dir, img_files_);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    CHECK(cudaFree(device_input_));
}

int Int8EntropyCalibrator2::getBatchSize() const noexcept
{
    return batchsize_;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    if (img_idx_ + batchsize_ > (int)img_files_.size()) {
        return false;
    }

    std::vector<uint8_t> input_imgs_(input_count_, 0);
    std::vector<float> input(input_size_);
    cv::Mat img(input_h_, input_w_, CV_8UC3);
    for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
        std::cout << img_files_[i] << "  " << i << std::endl;
        cv::Mat temp = cv::imread(img_dir_ + img_files_[i]);
        if (temp.empty()) {
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        cv::resize(temp, img, img.size(), cv::INTER_LINEAR);
        memcpy(input_imgs_.data() + (i - img_idx_) * input_size_, img.data, input_size_);
        Preprocess(input, input_imgs_, batchsize_, input_c_, input_h_, input_w_);
    }
    img_idx_ += batchsize_;
    CHECK(cudaMemcpy(device_input_, input.data(), input_count_ * sizeof(float), cudaMemcpyHostToDevice));

    assert(!strcmp(names[0], input_blob_name_));
    bindings[0] = device_input_;
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) noexcept
{
    std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good()){
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}