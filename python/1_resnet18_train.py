# by yhpark 2023-04-01
# resnet18 cifar10 dataset train example
# https://github.com/leimao/PyTorch-Quantization-Aware-Training/blob/main/cifar.py
from utils import *

if not os.path.exists('model'):
    os.makedirs('model')
    print('make directory {} is done'.format('./model'))

if not os.path.exists('data'):
    os.makedirs('data')
    print('make directory {} is done'.format('./data'))

set_random_seeds(random_seed=777)
device = device_check()  # device check & define
# device = torch.device("cpu:0")
def main():

    # 1. model generation
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)              # to gpu
    # print(f"model: {model}")              # print model structure
    # summary(model, (3, 224, 224))         # print output shape & total parameter sizes for given input size

    # data loader
    train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=256, eval_batch_size=256)

    # Train model.
    print("Training Model...")
    num_epochs = 100
    model_filename = f'resnet_cifar10_e{str(num_epochs)}.pth'
    print(f'model file name : {model_filename}')

    model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, device=device, learning_rate=1e-1, num_epochs=num_epochs)

    save_model(model, model_filename)
    model = load_model(model, model_filename, device)

    eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=None)
    print("Eval Loss: {:.3f} Eval Acc: {:.3f}".format(eval_loss, eval_accuracy))

def model_onnx_export():

    # 1. model generation
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)              # to gpu

    model_filename = 'resnet_cifar10_e100.pth'
    model = load_model(model, model_filename, device)

    # enable_onnx_checker needs to be disabled. See notes below.
    export_model_path = "model/resnet_cifar10_e100.onnx"

    dummy_input = torch.randn(256, 3, 32, 32, device=device)

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
