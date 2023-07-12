#  by yhpark 2023-07-10
import tensorrt as trt
import common
from utils import *
from PIL import Image
import json

TRT_LOGGER = trt.Logger()
genDir('./trt_model')

def get_engine(onnx_file_path, engine_file_path="", precision ='fp32'):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) \
                as builder, builder.create_network(common.EXPLICIT_BATCH) \
                as network, builder.create_builder_config() \
                as config, trt.OnnxParser(network, TRT_LOGGER) \
                as parser, trt.Runtime(TRT_LOGGER) \
                as runtime:

            config.max_workspace_size = 1 << 30  # 29 : 512MiB, 30 : 1024MiB

            if precision == "fp16":
                if not builder.platform_has_fast_fp16:
                    print("FP16 is not supported natively on this platform/device")
                else:
                    config.set_flag(trt.BuilderFlag.FP16)
                    print("Using FP16 mode.")
            elif precision == "int8":
                if not builder.platform_has_fast_int8:
                    print("INT8 is not supported natively on this platform/device")
                else:
                    config.set_flag(trt.BuilderFlag.INT8)
                    print("Using INT8 mode.")

            elif precision == "fp32":
                print("Using FP32 mode.")
            else:
                raise NotImplementedError(f"Currently hasn't been implemented: {precision}.")

            if not os.path.exists(engine_file_path):
                # Parse model file
                if not os.path.exists(onnx_file_path):
                    print(f"ONNX file {onnx_file_path} not found, please run 3_resnet18_onnx.py first to generate it.")
                    exit(0)
                print("Loading ONNX file from path {}...".format(onnx_file_path))
                with open(onnx_file_path, "rb") as model:
                    print("Beginning ONNX file parsing")
                    if not parser.parse(model.read()):
                        print("ERROR: Failed to parse the ONNX file.")
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        return None

                network.get_input(0).shape = [1, 3, 224, 224]
                print("Completed parsing of ONNX file")
                print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
                plan = builder.build_serialized_network(network, config)
                engine = runtime.deserialize_cuda_engine(plan)
                print("Completed creating Engine")

            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    engine_file_path = engine_file_path.replace('.trt', f'_{precision}.trt')
    print(engine_file_path)

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():
    set_random_seeds()
    device = device_check()
    dur_time = 0
    iteration = 10000

    # 2. input
    transform_ = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    val_dataset = datasets.ImageFolder('H:/dataset/imagenet100/val', transform=transform_)

    test_path = 'H:/dataset/imagenet100/val/n02077923/ILSVRC2012_val_00023081.JPEG'
    img = Image.open(test_path)
    img = transform_(img).unsqueeze(dim=0)
    num_img = np.array(img)
    x = np.array(num_img, dtype=np.float32, order="C")

    classes = val_dataset.classes
    class_to_idx = val_dataset.class_to_idx
    class_count = len(classes)

    json_file = open('H:/dataset/imagenet100/Labels.json')
    class_name = json.load(json_file)

    # 3. tensorrt model
    method = ["percentile", "mse", "entropy"]
    model_name = f'resnet18_{method[0]}'
    onnx_model_path = f"onnx_model/{model_name}.onnx"
    engine_file_path = f"trt_model/{model_name}.trt"

    precision = "int8" # int8 OR fp16 OR fp32

    # Output shapes expected by the post-processor
    output_shapes = [(1, class_count)]

    # Do inference with TensorRT
    t_outputs = []
    with get_engine(onnx_model_path, engine_file_path, precision) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = x
        for _ in range(100):
            t_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        for i in range(iteration):
            begin = time.time()
            t_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            torch.cuda.synchronize()
            dur = time.time() - begin
            dur_time += dur
        print('[TensorRT] {} iteration time : {} [sec]'.format(iteration, dur_time))

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    t_outputs = [output.reshape(shape) for output, shape in zip(t_outputs, output_shapes)]

    # 4. results
    print(f'{iteration}th iteration time : {dur_time} [sec]')
    print(f'Average fps : {1/(dur_time/iteration)} [fps]')
    print(f'Average inference time : {(dur_time/iteration)*1000} [msec]')
    max_tensor = torch.from_numpy(t_outputs[0]).max(dim=1)
    max_value = max_tensor[0].cpu().data.numpy()[0]
    max_index = max_tensor[1].cpu().data.numpy()[0]
    print(f'Resnet18 max index : {max_index} , value : {max_value}, class name : {classes[max_index]} {class_name.get(classes[max_index])}')

if __name__ == '__main__':
    main()

# [TensorRT] 10000 iteration time : 14.867701053619385 [sec]
# 10000th iteration time : 14.867701053619385 [sec]
# Average fps : 672.5989420916966 [fps]
# Average inference time : 1.4867701053619384 [msec]
# Resnet18 max index : 99 , value : 14.780750274658203, class name : n02077923 sea lion

# [TensorRT] 10000 iteration time : 11.687422037124634 [sec]
# 10000th iteration time : 11.687422037124634 [sec]
# Average fps : 855.6206807827591 [fps]
# Average inference time : 1.1687422037124633 [msec]
# Resnet18 max index : 99 , value : 11.432705879211426, class name : n02077923 sea lion

# [TensorRT] 10000 iteration time : 10.248055219650269 [sec]
# 10000th iteration time : 10.248055219650269 [sec]
# Average fps : 975.7948982188706 [fps]
# Average inference time : 1.024805521965027 [msec]
# Resnet18 max index : 99 , value : 9.233514785766602, class name : n02077923 sea lion


# [TensorRT] 10000 iteration time : 5.419851779937744 [sec]
# 10000th iteration time : 5.419851779937744 [sec]
# Average fps : 1845.0689070531864 [fps]
# Average inference time : 0.5419851779937743 [msec]
# Resnet18 max index : 99 , value : 14.799619674682617, class name : n02077923 sea lion

# [TensorRT] 10000 iteration time : 5.184077262878418 [sec]
# 10000th iteration time : 5.184077262878418 [sec]
# Average fps : 1928.9835959056636 [fps]
# Average inference time : 0.5184077262878417 [msec]
# Resnet18 max index : 99 , value : 11.449399948120117, class name : n02077923 sea lion

# [TensorRT] 10000 iteration time : 5.005523204803467 [sec]
# 10000th iteration time : 5.005523204803467 [sec]
# Average fps : 1997.7931558490561 [fps]
# Average inference time : 0.5005523204803467 [msec]
# Resnet18 max index : 99 , value : 9.26755428314209, class name : n02077923 sea lion