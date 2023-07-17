#  by yhpark 2023-07-17
import tensorrt as trt
import common
from utils import *
from PIL import Image
import json
from calibrator import EngineCalibrator

# TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE
genDir("./trt_model")


def get_engine(
    onnx_file_path,
    engine_file_path="",
    precision="fp32",
    TORCH_QUANTIZATION=False,
    gen_force=False,
):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    f"ONNX file {onnx_file_path} not found, please run 3_resnet18_onnx.py first to generate it."
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            outputs = [network.get_output(i) for i in range(network.num_outputs)]

            print("Network Description")
            for input in inputs:
                batch_size = input.shape[0]
                print(
                    "Input '{}' with shape {} and dtype {}".format(
                        input.name, input.shape, input.dtype
                    )
                )
            for output in outputs:
                print(
                    "Output '{}' with shape {} and dtype {}".format(
                        output.name, output.shape, output.dtype
                    )
                )
            assert batch_size > 0

            config.max_workspace_size = 1 << 31  # 29 : 512MiB, 30 : 1024MiB
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
                    config.set_flag(trt.BuilderFlag.FP16)
                    config.set_flag(trt.BuilderFlag.INT8)
                    print("Using INT8 mode.")
                    if TORCH_QUANTIZATION:
                        print("Using Pytorch Quantization mode.")
                    else:
                        print("Using TensorRT PTQ mode.")
                        inputs = [
                            network.get_input(i) for i in range(network.num_inputs)
                        ]

                        calib_cache = "./trt_model/cache_table.table"
                        config.int8_calibrator = EngineCalibrator(
                            TRT_LOGGER, calib_cache
                        )
                        if not os.path.exists(calib_cache):
                            calib_shape = [batch_size] + list(inputs[0].shape[1:])
                            calib_dtype = trt.nptype(inputs[0].dtype)
                            config.int8_calibrator.set_calibrator(
                                batch_size, calib_shape, calib_dtype, "./calib_data"
                            )

            elif precision == "fp32":
                print("Using FP32 mode.")
            else:
                raise NotImplementedError(
                    f"Currently hasn't been implemented: {precision}."
                )

            # network.get_input(0).shape = [1, 3, 224, 224]
            print("Completed parsing of ONNX file")
            print(
                "Building an engine from file {}; this may take a while...".format(
                    onnx_file_path
                )
            )
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")

            with open(engine_file_path, "wb") as f:
                f.write(plan)

            return engine

    engine_file_path = engine_file_path.replace(".trt", f"_{precision}.trt")
    print(engine_file_path)

    if os.path.exists(engine_file_path):
        if gen_force:
            return build_engine()
        else:
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    set_random_seeds()
    dur_time = 0
    iteration = 10000

    # 2. input
    transform_ = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_dataset = datasets.ImageFolder(
        "H:/dataset/imagenet100/val", transform=transform_
    )

    test_path = "H:/dataset/imagenet100/val/n02077923/ILSVRC2012_val_00023081.JPEG"
    img = Image.open(test_path)
    img = transform_(img).unsqueeze(dim=0)
    num_img = np.array(img)
    x = np.array(num_img, dtype=np.float32, order="C")

    classes = val_dataset.classes
    class_to_idx = val_dataset.class_to_idx
    class_count = len(classes)

    json_file = open("H:/dataset/imagenet100/Labels.json")
    class_name = json.load(json_file)

    # 3. tensorrt model
    gen_force = True
    precision = "int8"  # fp32, fp16, int8
    TORCH_QUANTIZATION = True
    QUANT_MODE = "QAT"
    if TORCH_QUANTIZATION:
        method = ["percentile", "mse", "entropy"]
        model_name = f"resnet18_{method[1]}"
        if QUANT_MODE == "QAT":
            model_name = model_name.replace("_", "_qat_")
        elif QUANT_MODE == "PTQ":
            model_name = model_name.replace("_", "_ptq_")
        model_name += "_2"
        precision = "int8"
    else:
        model_name = "resnet18"

    onnx_model_path = f"onnx_model/{model_name}.onnx"
    engine_file_path = f"trt_model/{model_name}.trt"

    # Output shapes expected by the post-processor
    output_shapes = [(1, class_count)]

    # Do inference with TensorRT
    t_outputs = []
    with get_engine(
        onnx_model_path, engine_file_path, precision, TORCH_QUANTIZATION, gen_force
    ) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = x

        # warm-up
        for _ in range(100):
            t_outputs = common.do_inference_v2(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
        torch.cuda.synchronize()

        for i in range(iteration):
            begin = time.time()
            t_outputs = common.do_inference_v2(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
            torch.cuda.synchronize()
            dur = time.time() - begin
            dur_time += dur

        print("[TensorRT] {} iteration time : {} [sec]".format(iteration, dur_time))

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    t_outputs = [
        output.reshape(shape) for output, shape in zip(t_outputs, output_shapes)
    ]

    # 4. results
    if TORCH_QUANTIZATION:
        print(f"Using Pytorch Quantization [{QUANT_MODE}] mode.")
    else:
        if precision == "int8":
            print(f"Using TensorRT PTQ mode.")

    print(engine_file_path)
    print(f"Using precision {precision} mode.")
    print(f"{iteration}th iteration time : {dur_time} [sec]")
    print(f"Average fps : {1/(dur_time/iteration)} [fps]")
    print(f"Average inference time : {(dur_time/iteration) * 1000} [msec]")
    max_tensor = torch.from_numpy(t_outputs[0]).max(dim=1)
    max_value = max_tensor[0].cpu().data.numpy()[0]
    max_index = max_tensor[1].cpu().data.numpy()[0]
    print(
        f"Resnet18 max index : {max_index} , value : {max_value}, class name : {classes[max_index]} {class_name.get(classes[max_index])}"
    )


if __name__ == "__main__":
    main()


# Using precision fp32 mode.
# 10000th iteration time : 12.061801433563232 [sec]
# Average fps : 829.0635569720082 [fps]
# Average inference time : 1.2061801433563233 [msec]
# Resnet18 max index : 99 , value : 21.666141510009766, class name : n02077923 sea lion

# Using precision fp16 mode.
# 10000th iteration time : 5.546254873275757 [sec]
# Average fps : 1803.0184743554257 [fps]
# Average inference time : 0.5546254873275757 [msec]
# Resnet18 max index : 99 , value : 21.66958236694336, class name : n02077923 sea lion

# Using TensorRT PTQ mode.
# trt_model/resnet18.trt
# Using precision int8 mode.
# 10000th iteration time : 4.262109994888306 [sec]
# Average fps : 2346.25573061074 [fps]
# Average inference time : 0.42621099948883057 [msec]
# Resnet18 max index : 99 , value : 19.75248146057129, class name : n02077923 sea lion


# Using Pytorch Quantization [QAT] mode.
# Using precision int8 mode.
# 10000th iteration time : 5.898566961288452 [sec]
# Average fps : 1695.3270286882785 [fps]
# Average inference time : 0.5898566961288453 [msec]
# Resnet18 max index : 99 , value : 21.456743240356445, class name : n02077923 sea lion

# Using Pytorch Quantization [QAT2] mode.
# Using precision int8 mode.
# 10000th iteration time : 5.852148056030273 [sec]
# Average fps : 1708.774266176609 [fps]
# Average inference time : 0.5852148056030273 [msec]
# Resnet18 max index : 99 , value : 23.088726043701172, class name : n02077923 sea lion

# Using Pytorch Quantization [PTQ] mode.
# Using precision int8 mode.
# 10000th iteration time : 5.982958793640137 [sec]
# Average fps : 1671.4138179641088 [fps]
# Average inference time : 0.5982958793640136 [msec]
# Resnet18 max index : 99 , value : 22.055187225341797, class name : n02077923 sea lion

# Using Pytorch Quantization [PTQ2] mode.
# Using precision int8 mode.
# 10000th iteration time : 5.745854139328003 [sec]
# Average fps : 1740.3852860715906 [fps]
# Average inference time : 0.5745854139328003 [msec]
# Resnet18 max index : 99 , value : 21.805843353271484, class name : n02077923 sea lion
