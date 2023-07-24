#  by yhpark 2023-07-12
# tensorboard --logdir ./logs
from quant_utils import *
from utils import *

genDir("./ptq_model")


def main():
    set_random_seeds()
    device = device_check()

    # 0. dataset
    batch_size = 256
    workers = 8
    data_dir = "H:/dataset/imagenet100"  # dataset path

    print(f"=> Custom {data_dir} is used!")
    traindir = os.path.join(data_dir, "train")
    valdir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        sampler=None,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=None,
    )

    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    class_count = len(classes)

    # 1. model
    model_name = "resnet18"
    model = models.__dict__[model_name]().to(device)
    # 학습 데이터셋의 클래스 수에 맞게 출력값이 생성 되도록 마지막 레이어 수정
    if class_count != model.fc.out_features:
        model.fc = nn.Linear(model.fc.in_features, class_count)
    model = model.to(device)

    check_path = "./checkpoints/resnet18.pth.tar"
    model.load_state_dict(torch.load(check_path, map_location=device))

    bn_folding = True
    if bn_folding:
        model = fuse_bn_recursively(model)

    model.eval()

    # evaluate model status
    if False:
        print(f"model: {model}")  # print model structure
        summary(
            model, (3, 224, 224)
        )  # print output shape & total parameter sizes for given input size

    test_acc1 = test(
        val_loader, model, device, class_to_idx, classes, class_acc=False, print_freq=10
    )
    print(f"acc before ptq : {test_acc1}")

    # 2. ptq with calibration
    print("=================================================")
    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, train_loader, num_batches=4)
        for method in ["percentile", "mse", "entropy"]:
            if method == "percentile":
                compute_amax(model, method=method, percentile=99.99)
            else:
                compute_amax(model, method=method)
            print(f"{method} calibration")
            test_acc1 = test(
                val_loader,
                model,
                device,
                class_to_idx,
                classes,
                class_acc=False,
                print_freq=10,
            )
            print(" Eval Acc: {:.3f}".format(test_acc1))
            print("=================================================")
            # Save the model
            torch.save(model.state_dict(), f"ptq_model/{model_name}_{method}_4.pth.tar")


if __name__ == "__main__":
    main()
