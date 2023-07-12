#  by yhpark 2023-07-12
from utils import *

def main():
    set_random_seeds()
    device = device_check()

    # 0. dataset
    data_dir = 'H:/dataset/imagenet100'  # dataset path
    batch_size = 256
    workers = 8

    print(f"=> Custom {data_dir} is used!")
    print(f"=> Batch_size : {batch_size}")
    valdir = os.path.join(data_dir, 'val')

    val_dataset = datasets.ImageFolder(valdir, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=workers, pin_memory=True, sampler=None)

    classes = val_dataset.classes
    class_to_idx = val_dataset.class_to_idx


    # 1. model
    class_count = len(classes)
    model_name = 'resnet18'
    model = models.__dict__[model_name]().to(device)
    # 학습 데이터셋의 클래스 수에 맞게 출력값이 생성 되도록 마지막 레이어 수정
    model.fc = nn.Linear(model.fc.in_features, class_count)
    model = model.to(device)

    if False:
        print(f"model: {model}")  # print model structure
        summary(model, (3, 224, 224))  # print output shape & total parameter sizes for given input size


    # 2. evaluate model
    print("=> Model inference test has started!")
    check_path = './checkpoints/model_best_resnet18_base.pth.tar'
    load_checkpoint(check_path, model, device)
    test_acc1 = test(val_loader, model, device, class_to_idx, classes, class_acc=False, print_freq=10)
    print(f"acc after model train : {test_acc1}")



if __name__ == '__main__':
    main()
