#  by yhpark 2023-07-13
# tensorboard --logdir ./logs
from quant_utils import *
from utils import *
import onnx
from onnxsim import simplify
genDir("./onnx_model")


def main():
    set_random_seeds()
    device = device_check()

    # 0. dataset
    data_dir = "H:/dataset/imagenet100"  # dataset path
    print(f"=> Custom {data_dir} is used!")

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
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

    # 1. model
    class_count = len(val_dataset.classes)
    model_name = "resnet18"
    model = models.__dict__[model_name]().to(device)
    # 학습 데이터셋의 클래스 수에 맞게 출력값이 생성 되도록 마지막 레이어 수정
    model.fc = nn.Linear(model.fc.in_features, class_count)
    model = model.to(device)

    #MODE = "PTQ"
    MODE = "nn"
    if MODE in ["PTQ", "QAT"]:
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        method = ["percentile", "mse", "entropy"]
        model_name = f"resnet18_{method[2]}"
        if MODE == "QAT":
            check_path = f"./qat_model/{model_name}.pth.tar"
            model_name = model_name.replace("_", "_qat_")
        elif MODE == "PTQ":
            check_path = f"./ptq_model/{model_name}.pth.tar"
            model_name = model_name.replace("_", "_ptq_")
        model_name += "_2"
    else:
        check_path = "./checkpoints/resnet18.pth.tar"

    model.load_state_dict(torch.load(check_path, map_location=device))
    model.eval()
    # evaluate model status
    if False:
        print(f"model: {model}")  # print model structure
        summary(
            model, (3, 224, 224)
        )  # print output shape & total parameter sizes for given input size

    # export onnx model
    export_model_path = f"./onnx_model/{model_name}.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)

    with torch.no_grad():
        torch.onnx.export(
            model,  # pytorch model
            dummy_input,  # model dummy input
            export_model_path,  # onnx model path
            opset_version=17,  # the version of the opset
            input_names=["input"],  # input name
            output_names=["output"],
        )  # output name

        print("ONNX Model exported at ", export_model_path)

    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX Model check done!")

    if 0:
        sim_model_path = f"./onnx_model/{model_name}_sim.onnx"
        model_simp, check = simplify(onnx_model)  # convert(simplify)
        onnx.save(model_simp, sim_model_path)
        model_simp = onnx.load(sim_model_path)
        onnx.checker.check_model(model_simp)
        print("sim ONNX Model check done!")


if __name__ == "__main__":
    main()
