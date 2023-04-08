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
model.fc = nn.Linear(model.fc.in_features, 100)

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
model_filename = 'resnet18_imagenet100_best_entropy.pth'
model = load_model(model, model_filename, device)
model = model.to(device)  # to gpu

new_model_filename = 'resnet18_imagenet100_best_entropy_qat'
num_epochs = 4


model = train_model(model_filename=new_model_filename, model=model, train_loader=train_loader, test_loader=test_loader, device=device, learning_rate=1e-3,
                num_epochs=num_epochs)

save_model(model, f'{new_model_filename}.pth')

model = load_model(model, f'{new_model_filename}.pth', device)

model.eval()
criterion = nn.CrossEntropyLoss()
eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion, time_check=True)
print("Epoch: {:03d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(0, eval_loss, eval_accuracy))