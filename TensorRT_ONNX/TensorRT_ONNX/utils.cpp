#include "utils.hpp"

int SearchFile(const std::string& folder_path, std::vector<std::string> &file_names, bool recursive)
{
    _finddata_t file_info;
    std::string any_file_pattern = folder_path + "\\*";
    intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);

    if (handle == -1)
    {
        std::cerr << "folder path not exist: " << folder_path << std::endl;
        return -1;
    }

    do
    {
        std::string file_name = file_info.name;
        if (recursive) {
            if (file_info.attrib & _A_SUBDIR)//check whtether it is a sub direcotry or a file
            {
                if (file_name != "." && file_name != "..")
                {
                    std::string sub_folder_path = folder_path + "//" + file_name;
                    SearchFile(sub_folder_path, file_names);
                    std::cout << "a sub_folder path: " << sub_folder_path << std::endl;
                }
            }
            else
            {
                std::string file_path = folder_path + "/" + file_name;
                file_names.push_back(file_path);
            }
        }
        else {
            if (!(file_info.attrib & _A_SUBDIR))//check whtether it is a sub direcotry or a file
            {
                std::string file_path = folder_path + "/" + file_name;
                file_names.push_back(file_path);
            }
        }
    } while (_findnext(handle, &file_info) == 0);
    _findclose(handle);
    return 0;
}

int argMax(std::vector<float> &output) 
{
    return max_element(output.begin(), output.end()) - output.begin();
}

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        nvinfer1::Weights wt{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

void show_dims(nvinfer1::ITensor* tensor)
{
    std::cout << "=== show dims ===" << std::endl;
    int dims = tensor->getDimensions().nbDims;
    std::cout << "size :: " << dims << std::endl;
    for (int i = 0; i < dims; i++) {
        std::cout << tensor->getDimensions().d[i] << std::endl;
    }
}

void Preprocess(std::vector<float> &output, std::vector<uint8_t>& input, int BatchSize, int channels, int height, int width)
{
    /*
        INPUT  = BGR[NHWC](0, 255)
        OUTPUT = RGB[NCHW](0.f,1.f)
        This equation include 3 steps
        1. Scale Image to range [0.f, 1.0f], /255
        2. Shuffle form HWC to CHW
        3. BGR -> RGB
    */
    int offset = channels * height * width;
    int b_off, c_off, h_off, h_off_o, w_off_o;
    for (int b = 0; b < BatchSize; b++) {
        b_off = b * offset;
        for (int c = 0; c < channels; c++) {
            c_off = c * height * width + b_off;
            for (int h = 0; h < height; h++) {
                h_off = h * width + c_off;
                h_off_o = h * width * channels + b_off;
                for (int w = 0; w < width; w++) {
                    int dstIdx = h_off + w;
                    int srcIdx = h_off_o + w * channels + 2 - c;
                    output[dstIdx] = (static_cast<const float>(input[srcIdx]) / 255.f);
                }
            }
        }
    }
};

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    _finddata_t file_info;
    const std::string folder_path = p_dir_name;
    std::string any_file_pattern = folder_path + "\\*";
    intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);

    if (handle == -1)
    {
        std::cerr << "folder path not exist: " << folder_path << std::endl;
        return -1;
    }
    do
    {
        std::string file_name = file_info.name;
        if (!(file_info.attrib & _A_SUBDIR))//check whtether it is a sub direcotry or a file
        {
            //std::string file_path = folder_path + "/" + file_name;
            file_names.push_back(file_name);
        }

    } while (_findnext(handle, &file_info) == 0);
    _findclose(handle);
    return 0;
}


void mkdir(const std::string &path) {
    std::experimental::filesystem::path p(path);

    if (std::experimental::filesystem::is_directory(p)) {
        std::cout << "The folder already exists. : "<< path << std::endl;
    }
    else {
        std::experimental::filesystem::create_directories(p);
        std::cout << "The folder was created. : " << path << std::endl;
    }
}