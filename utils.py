import shutil
import random
import os
import time
import numpy as np
from enum import Enum
from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data
from torch.optim.lr_scheduler import StepLR, MultiStepLR



def genDir(dir_path):
    if not os.path.exists(dir_path):  # 저장할 폴더가 없다면
        os.makedirs(dir_path)  # 폴더 생성
        print(f'make directory {dir_path} is done')

def load_checkpoint(path, model, device):
    # path = ./xxxx.pth.tar
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        if torch.cuda.is_available():
            checkpoint = torch.load(path, map_location=device)
        else:
            checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded checkpoint {path}")
    else:
        print(f"=> no checkpoint found at {path}")

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def device_check():
    print("pytorch:", torch.__version__)
    print('[Device Check]')
    if torch.cuda.is_available():
        print(f'Torch gpu available : {torch.cuda.is_available()}')
        print(f'The number of gpu device : {torch.cuda.device_count()}')
        for g_idx in range(torch.cuda.device_count()):
            print(f'{g_idx} device name : {torch.cuda.get_device_name(g_idx)}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
    print(f"device : {device} is available")
    return device


def save_checkpoint(state, is_best, model_name):
    filename = f'./checkpoints/checkpoint_{model_name}.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'./checkpoints/model_best_{model_name}.pth.tar')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        # dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


color_map = [(244, 67, 54), (233, 30, 99), (156, 39, 176), (103, 58, 183), (63, 81, 181), (33, 150, 243), (3, 169, 244),
             (0, 188, 212), (0, 150, 136), (76, 175, 80)]

def train(train_loader, model, criterion, optimizer, epoch, device, scaler, use_amp, writer, regularizer=None, print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))
    print_freq = len(train_loader) // print_freq

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure dogs_vs_cats_dataset loading time
        data_time.update(time.time() - end)

        # move dogs_vs_cats_dataset to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        #with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        with torch.autocast(device_type='cuda', enabled=use_amp):
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            if regularizer:
                scaler.unscale_(optimizer)
                regularizer(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:  # print_freq(10) 미니 배치 마다 출력
            progress.display(i + 1)
            writer.add_scalar("Loss/train", losses.val, epoch * len(train_loader) + i)
            writer.add_scalar("Acc(top1)/train", int(top1.val), epoch * len(train_loader) + i)
            writer.add_scalar("Acc(top5)/train", top5.val, epoch * len(train_loader) + i)
            writer.close()


def validate(val_loader, model, criterion, epoch, device, class_to_idx, classes, writer, class_acc=True, print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Validate: ')
    print_freq = len(val_loader) // print_freq

    # switch to evaluate mode
    model.eval()
    out_list = []
    target_list = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            out_list.append(output)
            target_list.append(target)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i + 1)

        if class_acc:
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}

            outputs = torch.cat(out_list, 0)
            targets = torch.cat(target_list, 0)

            _, pred = outputs.topk(1, 1, True, True)
            pred = pred.t()

            for label, prediction in zip(targets.cpu(), pred.view(-1).cpu()):
                if label.item() == prediction.item():
                    correct_pred[classes[label.item()]] += 1
                total_pred[classes[label.item()]] += 1

            # 각 분류별 정확도(accuracy)를 출력
            total_correct = 0
            total_count = 0
            for classname, correct_count in correct_pred.items():
                total_correct += correct_count
                total_count += total_pred[classname]
                if correct_count == 0:
                    acc = 0
                else:
                    acc = 100 * float(correct_count) / total_pred[classname]
                print(
                    f'[{class_to_idx[classname]}]Accuracy for {classname:5s} : {acc:.1f} %, ({correct_count}/{total_pred[classname]})')
            acc = 100 * float(total_correct) / total_count
            print(f'Total Accuracy : {acc:.1f} %, ({total_correct}/{total_count})')

    progress.display_summary()
    writer.add_scalar("Loss/valid", losses.avg, epoch)
    writer.add_scalar("Acc(top1)/valid", top1.avg, epoch)
    writer.add_scalar("Acc(top5)/valid", top5.avg, epoch)

    return top1.avg


def test(val_loader, model, device, class_to_idx, classes, class_acc =True, print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')
    print_freq = len(val_loader) // print_freq
    # switch to evaluate mode
    model.eval()
    out_list = []
    target_list = []
    with torch.no_grad():
        end = time.time()
        dur_time = 0
        count = 0
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

            # compute output
            begin = time.time()
            output = model(images)
            if i > 10:
                torch.cuda.synchronize()
                count += 1
                dur_time += time.time() - begin

            out_list.append(output)
            target_list.append(target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i + 1)

        print(f'batch_size : {val_loader.batch_size} ')
        print(f'Average fps : {1 / (dur_time / (count * val_loader.batch_size))} [fps]')
        print(f'Average inference time : {(dur_time / (count * val_loader.batch_size)) * 1000} [msec]')

        if class_acc:

            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}

            outputs = torch.cat(out_list, 0)
            targets = torch.cat(target_list, 0)

            _, pred = outputs.topk(1, 1, True, True)
            pred = pred.t()

            for label, prediction in zip(targets.cpu(), pred.view(-1).cpu()):
                if label.item() == prediction.item():
                    correct_pred[classes[label.item()]] += 1
                total_pred[classes[label.item()]] += 1

            # 각 분류별 정확도(accuracy)를 출력
            total_correct = 0
            total_count = 0
            for classname, correct_count in correct_pred.items():
                total_correct += correct_count
                total_count += total_pred[classname]
                if correct_count == 0:
                    acc = 0
                else:
                    acc = 100 * float(correct_count) / total_pred[classname]
                print(
                    f'[{class_to_idx[classname]}]Accuracy for {classname:5s} : {acc:.1f} %, ({correct_count}/{total_pred[classname]})')
            acc = 100 * float(total_correct) / total_count
            print(f'Total Accuracy : {acc:.1f} %, ({total_correct}/{total_count})')

    progress.display_summary()

    return top1.avg
