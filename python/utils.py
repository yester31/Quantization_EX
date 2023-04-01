# by yhpark 2023-04-01
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms

import cv2, os, struct, time
import numpy as np
import random
from torchsummary import summary

import onnx

from absl import logging
logging.set_verbosity(logging.FATAL)  # Disable logging as they are too noisy in notebook
from tqdm import tqdm

# check device & get device info
def device_check():
    print('[Device Check]')
    if torch.cuda.is_available():
        print(f'gpu device count : {torch.cuda.device_count()}')
        print(f'device_name : {torch.cuda.get_device_name(0)}')
        print(f'torch gpu available : {torch.cuda.is_available()}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
    print(f"device : {device} is available")
    return device

# load resnet18 model
def load_resnet18(num_classes=1000):
    if not os.path.exists('model'):  # 저장할 폴더가 없다면
        os.makedirs('model')  # 폴더 생성
        print('make directory {} is done'.format('./model'))

    if os.path.isfile('python/model/resnet18.pth'):  # resnet18.pth 파일이 있다면
        net = torch.load('python/model/resnet18.pth')  # resnet18.pth 파일 로드
        num_features = net.fc.in_features
        if num_classes != num_features:
            net.fc = nn.Linear(num_features, num_classes)
    else:  # resnet18.pth 파일이 없다면
        net = torchvision.models.resnet18(num_classes=num_classes, pretrained=True)  # torchvision에서 resnet18 pretrained weight 다운로드 수행
        num_features = net.fc.in_features
        if num_classes != num_features:
            net.fc = nn.Linear(num_features, num_classes)
        torch.save(net, 'python/model/resnet18.pth')  # resnet18.pth 파일 저장
    return net

def save_model(model, model_filename):
    if not os.path.exists('model'):
        os.makedirs('model')
    model_filepath = os.path.join('model', model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filename, device):
    model_filepath = os.path.join('model', model_filename)
    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model

# preprocess function
def preprocess(img, half, device):
    with torch.no_grad():
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # bgr -> rgb
        img3 = img2.transpose(2, 0, 1)              # hwc -> chw
        img4 = img3.astype(np.float32)              # uint -> float32
        img4 /= 255                                 # 1/255
        img5 = torch.from_numpy(img4)               # numpy -> tensor
        if half:                                    # f32 -> f16
            img5 = img5.half()
        img6 = img5.unsqueeze(0)                    # [c,h,w] -> [1,c,h,w]
        img6 = img6.to(device)

    return img6

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(num_workers=8,
                       train_batch_size=256,
                       eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.CIFAR10(root="data",
                                             train=True,
                                             download=True,
                                             transform=train_transform)
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(root="data",
                                            train=False,
                                            download=True,
                                            transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=train_batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=eval_batch_size,
                                              sampler=test_sampler,
                                              num_workers=num_workers)

    return train_loader, test_loader



def train_model(model, train_loader, test_loader, device, learning_rate=1e-1, num_epochs=200):
    # The training configurations were not carefully selected.
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs*0.8), int(num_epochs*0.9)], gamma=0.1, last_epoch=-1)

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy = evaluate_model(model=model,test_loader=test_loader, device=device, criterion=criterion)
    print("Epoch: {:03d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(0, eval_loss, eval_accuracy))

    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion)

        # Set learning rate scheduler
        scheduler.step()

        print("Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}"
            .format(epoch + 1, train_loss, train_accuracy, eval_loss, eval_accuracy))

    return model


def evaluate_model(model, test_loader, device, criterion=None, time_check=False):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0
    dur_time = 0
    iteration = 0
    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        begin = time.time()
        outputs = model(inputs)
        dur = time.time() - begin
        dur_time += dur
        iteration += 1
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    if time_check :
        print(f'batch_size : {test_loader.batch_size}')
        print(f'{iteration}th iteration time : {dur_time} [sec]')
        print(f'Average fps : {test_loader.batch_size/(dur_time/iteration)} [fps]')
        print(f'Average inference time : {((dur_time/iteration)/test_loader.batch_size)*1000} [msec]')
    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy