import tensorrt as trt
import sys, os
import torch
import numpy as np
import common
print(trt.__version__)

TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):
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
            config.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            if not os.path.exists(engine_file_path):
                # Parse model file
                if not os.path.exists(onnx_file_path):
                    print(
                        "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
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

                network.get_input(0).shape = [1, 3, 224, 224]
                print("Completed parsing of ONNX file")
                print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
                plan = builder.build_serialized_network(network, config)
                engine = runtime.deserialize_cuda_engine(plan)
                print("Completed creating Engine")

            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():

    onnx_file_path = "../model/resnet18_imagenet100_best.onnx"
    engine_file_path = "../model/resnet18_imagenet100_best.trt"

    x = torch.ones((1, 3, 224, 224))
    x = np.array(x, dtype=np.float32, order="C")
    # Output shapes expected by the post-processor
    output_shapes = [(1, 100)]
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = x
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

    tt = 1


if __name__ == "__main__":
    main()