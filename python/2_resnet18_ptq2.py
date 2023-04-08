# by yhpark 2023-04-01
# resnet18 post training quantization example
# https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/docs/source/userguide.rst
# https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/examples/calibrate_quant_resnet50.ipynb

from utils import *

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

from pytorch_quantization import quant_modules
quant_modules.initialize()

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""
    print('Feed data to the network and collect statistic')
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            #print(F"{name:40}: {module}")
    model.cuda()

device = device_check()  # device check & define
set_random_seeds(random_seed=777)

# 1. model generation
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 100)
# print(f"model: {model}")              # print model structure
# summary(model, (3, 32, 32))         # print output shape & total parameter sizes for given input size

# data loader
batch_size = 256
num_workers = 8
dataset_path = '/home/yhpark/Desktop/work/git/Pytorch_Quantization_EX/imagenet100'
train_dataset = datasets.ImageFolder(f'{dataset_path}/train', transform=transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),]))
val_dataset = datasets.ImageFolder(f'{dataset_path}/val', transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# weight load
model_filename = 'resnet18_imagenet100_best.pth'
model = load_model(model, model_filename, device)
model = model.to(device)  # to gpu
model.eval()

criterion = nn.CrossEntropyLoss()
print("=================================================")
print("fp32 model")
eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion, time_check=True)
print("Eval Loss: {:.3f} Eval Acc: {:.3f}".format(eval_loss, eval_accuracy))
print("=================================================")
# It is a bit slow since we collect histograms on CPU
with torch.no_grad():
    collect_stats(model, train_loader, num_batches=16)
    method = "percentile"
    compute_amax(model, method=method, percentile=99.99)
    print(f"{method} calibration")
    eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion,
                                              time_check=True)
    print("Eval Loss: {:.3f} Eval Acc: {:.3f}".format(eval_loss, eval_accuracy))
    print("=================================================")
    # Save the model
    torch.save(model.state_dict(), f"model/resnet18_imagenet100_best_{method}.pth")

    for method in ["mse", "entropy"]:
        compute_amax(model, method=method)
        print(f"{method} calibration")
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device,
                                                  criterion=criterion, time_check=True)
        print("Eval Loss: {:.3f} Eval Acc: {:.3f}".format(eval_loss, eval_accuracy))
        print("=================================================")
        # Save the model
        torch.save(model.state_dict(), f"model/resnet18_imagenet100_best_{method}.pth")