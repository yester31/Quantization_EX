#include "utils.hpp"            // custom util function
#include "logging.hpp"          // Nvidia logger
#include "calibrator.hpp"       // for ptq
#include "parserOnnxConfig.h"   // for onnx-parsing


using namespace nvinfer1;
sample::Logger gLogger;

static const int maxBatchSize = 256;
static const int INPUT_H = 32;
static const int INPUT_W = 32;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 10;
static const int precision_mode = 8;        // fp32 : 32, fp16 : 16, int8(ptq) : 8
const char* INPUT_BLOB_NAME = "input";      // use same input name with onnx model
const char* OUTPUT_BLOB_NAME = "output";    // use same output name with onnx model
const char* engine_dir_path = "../Engine";  // Engine directory path
const char* engineFileName = "resnet18";    // model name
const char* onnx_file = "../../python/model/resnet_cifar10_e100_mse_qat.onnx"; // onnx model file path
bool serialize = false;                     // force serialize flag (IF true, recreate the engine file unconditionally)
uint64_t iter_count = 40;                // the number of test iterations

// Creat the engine using onnx.
void createEngineFromOnnx(int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engine_file_path);


int main()
{
    char engine_file_path[256];
    sprintf(engine_file_path, "%s/%s_%d.engine", engine_dir_path, engineFileName, precision_mode);
    mkdir(engine_dir_path);
    /*
    /! 1) Create engine file 
    /! If force serialize flag is true, recreate unconditionally
    /! If force serialize flag is false, engine file is not created if engine file exist.
    /!                                   create the engine file if engine file doesn't exist.
    */
    bool exist_engine = false;
    if ((access(engine_file_path, 0) != -1)) {
        exist_engine = true;
    }

    if (!((serialize == false)/*Force Serialize flag*/ && (exist_engine == true) /*Whether the resnet18.engine file exists*/)) {
        std::cout << "===== Create Engine file =====" << std::endl << std::endl;

        IBuilder* builder = createInferBuilder(gLogger);
        if (!builder){
            std::string msg("failed to make builder");
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }

        IBuilderConfig* config = builder->createBuilderConfig();
        if (!config) {
            std::string msg("failed to make config");
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }
        
        // ***  create tensorrt model from ONNX Model ***
        createEngineFromOnnx(maxBatchSize, builder, config, DataType::kFLOAT, engine_file_path);

        builder->destroy();
        config->destroy();
        std::cout << "===== Create Engine file =====" << std::endl << std::endl; 
    }

    // 2) load engine file
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::cout << "===== Engine file load =====" << std::endl << std::endl;
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
    }
    else {
        std::cout << "[ERROR] Engine file load error" << std::endl;
    }

    // 3) deserialize TensorRT Engine from file
    std::cout << "===== Engine file deserialize =====" << std::endl << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext* context = engine->createExecutionContext();
    delete[] trtModelStream;

    void* buffers[2];
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    // Allocate GPU memory space for input and output
    CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float)));

    // 4) prepare input data (test dataset first intput)
    std::string img_dir = "../../python/data/";
    std::vector<std::string> file_names;
    if (SearchFile(img_dir.c_str(), file_names) < 0) { // load input data
        std::cerr << "[ERROR] Data search error" << std::endl;
    }
    else {
        std::cout << "Total number of images : " << file_names.size() << std::endl << std::endl;
    }

    std::vector<float> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
    std::vector<float> outputs(maxBatchSize * OUTPUT_SIZE);
    for (int idx = 0; idx < file_names.size(); idx++) {
        fromfile(input, file_names[idx]);
    }
    std::cout << "===== input load done =====" << std::endl << std::endl;

    uint64_t dur_time = 0;


    // CUDA stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Warm-up
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // 5) Inference
    for (int i = 0; i < iter_count; i++) {
        auto start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context->enqueueV2(buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
        dur_time += dur;
        //std::cout << dur << " milliseconds" << std::endl;
    }
    tofile(outputs, "../../python/data/trt_output.bin");

    dur_time /= 1000.f; //microseconds -> milliseconds

    // 6) Print Results
    std::cout << "==================================================" << std::endl;
    std::cout << "Model : " << engineFileName << ", Precision : " << precision_mode << std::endl;
    std::cout << "BatchSize : " << maxBatchSize << std::endl;
    std::cout << iter_count << " th Iteration time : " << dur_time << " [milliseconds]" << std::endl;
    std::cout << "Average fps : " << 1000.f * (float)iter_count  * maxBatchSize / (dur_time ) << " [frame/sec]" << std::endl;
    std::cout << "Avg inference time (w data transfer) : " << dur_time / ((float)iter_count * maxBatchSize) << " [milliseconds]" << std::endl;
    //int max_index = max_element(outputs.begin(), outputs.end()) - outputs.begin();
    //std::cout << "Index : " << max_index << ", Probability : " << outputs[max_index] << std::endl;
    //std::cout << "Class Name : " << class_names[max_index] << std::endl;
    std::cout << "==================================================" << std::endl;

    // Release stream and buffers ...
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

//==================================================
//Model : resnet18, Precision : 8
//BatchSize : 256
//40 th Iteration time : 79[milliseconds]
//Average fps : 129620[frame / sec]
//Avg inference time(w data transfer) : 0.00771484[milliseconds]
//==================================================

// Creat the engine using onnx.
void createEngineFromOnnx(int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engine_file_path)
{
    std::cout << "==== model build start ====" << std::endl << std::endl;

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    if (!network) {
        std::string msg("failed to make network");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(onnx_file, (int)nvinfer1::ILogger::Severity::kINFO)) {
        std::string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    // Build engine
    //builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1ULL << 30); // 30:1GB, 29:512MB
    if (precision_mode == 16) {
        std::cout << "==== precision f16 ====" << std::endl << std::endl;
        config->setFlag(BuilderFlag::kFP16);
    }
    else if (precision_mode == 8) {
        std::cout << "==== precision int8 ====" << std::endl << std::endl;
        std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        //Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(maxBatchSize, INPUT_C, INPUT_W, INPUT_H, "../../calib_data/", "../Engine/resnet18_i8_calib.table", INPUT_BLOB_NAME);
        //config->setInt8Calibrator(calibrator);
    }
    else {
        std::cout << "==== precision f32 ====" << std::endl << std::endl;
    }

    bool dlaflag = false;
    int32_t dlaCore = builder->getNbDLACores();
    bool allowGPUFallback = true;
    std::cout << "the number of DLA engines available to this builder :: " << dlaCore << std::endl << std::endl;
    if (dlaCore >= 0 && dlaflag) {
        if (builder->getNbDLACores() == 0) {
            std::cerr << "Trying to use DLA core on a platform that doesn't have any DLA cores"
                << std::endl;
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback) {
            config->setFlag(BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(BuilderFlag::kINT8)) {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(dlaCore);
    }

    std::cout << "Building engine, please wait for a while..." << std::endl;

    IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);
    if (!engine) {
        std::string msg("failed to make engine");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    std::cout << "==== model build done ====" << std::endl << std::endl;

    std::cout << "==== model selialize start ====" << std::endl << std::endl;
    std::ofstream p(engine_file_path, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl << std::endl;
    }
    p.write(reinterpret_cast<const char*>(engine->data()), engine->size());
    std::cout << "==== model selialize done ====" << std::endl << std::endl;

    engine->destroy();
    network->destroy();
    p.close();
}
