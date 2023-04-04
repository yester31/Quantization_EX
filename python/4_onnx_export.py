# by yhpark 2023-04-02
# resnet18 onnx export example
# https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/docs/source/userguide.rst

from utils import *

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
quant_nn.TensorQuantizer.use_fb_fake_quant = True
quant_modules.initialize()

device = device_check()  # device check & define
set_random_seeds(random_seed=777)

# 1. model generation
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)

# data loader
train_loader, test_loader = prepare_dataloader(num_workers=0, train_batch_size=256, eval_batch_size=256)

# weight load
model_filename = 'resnet18_cifar10_e100_mse_qat.pth'
model = load_model(model, model_filename, device)
model = model.to(device)  # to gpu
model.eval()

from pytorch_quantization import nn as quant_nn
quant_nn.TensorQuantizer.use_fb_fake_quant = True

dummy_input = torch.randn(256, 3, 32, 32, device=device)

# enable_onnx_checker needs to be disabled. See notes below.
export_model_path = "model/resnet18_cifar10_e100_mse_qat.onnx"

torch.onnx.export(model,  # pytorch model
                  dummy_input,  # model dummy input
                  export_model_path,  # onnx model path
                  opset_version=17,  # the version of the opset
                  input_names=['input'],  # input name
                  output_names=['output'])  # output name

onnx_model = onnx.load(export_model_path)
onnx.checker.check_model(onnx_model)


