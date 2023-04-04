# by yhpark 2023-04-02
# resnet18 quantization aware training model example
# https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/examples/finetune_quant_resnet50.ipynb

from utils import *

from pytorch_quantization import quant_modules
quant_modules.initialize()

device = device_check()  # device check & define
set_random_seeds(random_seed=777)

# 1. model generation
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)

# data loader
train_loader, test_loader = prepare_dataloader(num_workers=0, train_batch_size=256, eval_batch_size=256)

# weight load
model_filename = 'resnet18_cifar10_e100_mse.pth'
model = load_model(model, model_filename, device)
model = model.to(device)  # to gpu

new_model_filename = 'resnet18_cifar10_e100_mse_qat.pth'
num_epochs = 10

if 0:
    model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, device=device, learning_rate=1e-3,
                    num_epochs=num_epochs)

    save_model(model, new_model_filename)

model = load_model(model, new_model_filename, device)

model.eval()
criterion = nn.CrossEntropyLoss()
eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion, time_check=True)
print("Epoch: {:03d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(0, eval_loss, eval_accuracy))