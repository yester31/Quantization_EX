# by yhpark 2023-04-04
# resnet18 imagenet100 dataset train example
# https://github.com/leimao/PyTorch-Quantization-Aware-Training/blob/main/cifar.py
# https://www.kaggle.com/datasets/ambityga/imagenet100?select=Labels.json
from utils import *

if not os.path.exists('model'):
    os.makedirs('model')
    print('make directory {} is done'.format('./model'))

if not os.path.exists('data'):
    os.makedirs('data')
    print('make directory {} is done'.format('./data'))

set_random_seeds(random_seed=777)
 # device check & define
def main():

    # 1. model generation
    device = device_check()
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)              # to gpu
    # print(f"model: {model}")              # print model structure
    # summary(model, (3, 224, 224))         # print output shape & total parameter sizes for given input size

    # data loader
    batch_size = 64
    num_workers = 8
    train_dataset = datasets.ImageFolder('H:/dataset/imagenet100/train', transform=transforms.Compose([
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),]))
    val_dataset = datasets.ImageFolder('H:/dataset/imagenet100/val', transform=transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Train model.
    print("Training Model...")
    num_epochs = 200
    model_filename = 'resnet18_imagenet100'
    print(f'model file name : {model_filename}')

    model = train_model(model_filename=model_filename, model=model, train_loader=train_loader, test_loader=test_loader, device=device, learning_rate=1e-1, num_epochs=num_epochs)

    save_model(model, f'{model_filename}_e{str(num_epochs)}.pth')
    model = load_model(model, model_filename, device)

    eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=None)
    print("Eval Loss: {:.3f} Eval Acc: {:.3f}".format(eval_loss, eval_accuracy))

def model_onnx_export():

    # 1. model generation
    device = device_check()
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)              # to gpu

    model_filename = 'resnet_imagenet100_e100'
    model = load_model(model, f'{model_filename}.pth', device)

    # enable_onnx_checker needs to be disabled. See notes below.
    export_model_path = f'model/{model_filename}.onnx'

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(model,  # pytorch model
                      dummy_input,  # model dummy input
                      export_model_path,  # onnx model path
                      opset_version=17,  # the version of the opset
                      input_names=['input'],  # input name
                      output_names=['output'])  # output name

    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)

if __name__ == '__main__':
    main()
    model_onnx_export()
